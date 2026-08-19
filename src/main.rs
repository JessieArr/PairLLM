mod llm;

use chrono::{DateTime, Local};
use eframe::egui;
use llm::{ChatProgressEvent, ChatTurn, LlmConfig, OllamaMetrics};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

const ASSISTANT_NAME: &str = "Assistant";
const USER_NAME: &str = "You";

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([720.0, 820.0])
            .with_min_inner_size([480.0, 400.0])
            .with_title("PairLLM Chat"),
        ..Default::default()
    };

    eframe::run_native(
        "PairLLM Chat",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(ChatApp::new()))
        }),
    )
}

#[derive(Clone)]
struct Message {
    author: String,
    content: String,
    created_at: DateTime<Local>,
    is_thinking: bool,
    metrics: Option<OllamaMetrics>,
}

struct CommandPrompt {
    command: String,
    response_tx: Sender<Result<String, String>>,
}

enum LlmEvent {
    Models { models: Vec<String> },
    Thinking { content: String },
    CommandApprovalNeeded {
        command: String,
        response_tx: Sender<Result<String, String>>,
    },
    Reply {
        content: String,
        metrics: OllamaMetrics,
    },
    Failed { message: String },
}

struct ChatApp {
    messages: Vec<Message>,
    draft: String,
    status: Option<String>,
    scroll_to_bottom: bool,
    llm: LlmConfig,
    show_settings: bool,
    llm_status: String,
    llm_busy: bool,
    thinking_message_index: Option<usize>,
    command_prompt: Option<CommandPrompt>,
    llm_tx: Sender<LlmEvent>,
    llm_rx: Receiver<LlmEvent>,
}

impl ChatApp {
    fn new() -> Self {
        let (llm_tx, llm_rx) = mpsc::channel();
        let mut app = Self {
            messages: Vec::new(),
            draft: String::new(),
            status: None,
            scroll_to_bottom: false,
            llm: LlmConfig::default(),
            show_settings: false,
            llm_status: "Checking for Ollama…".into(),
            llm_busy: false,
            thinking_message_index: None,
            command_prompt: None,
            llm_tx,
            llm_rx,
        };
        app.check_llm();
        app
    }

    fn check_llm(&mut self) {
        self.llm_status = "Checking for Ollama…".into();
        let tx = self.llm_tx.clone();
        let base_url = self.llm.base_url.clone();

        thread::spawn(move || {
            match llm::list_models(&base_url) {
                Ok(models) => {
                    let _ = tx.send(LlmEvent::Models { models });
                }
                Err(err) => {
                    let _ = tx.send(LlmEvent::Failed { message: err });
                }
            }
        });
    }

    fn send_message(&mut self, ctx: &egui::Context) {
        let content = self.draft.trim();

        if content.is_empty() {
            self.status = Some("Enter a message before sending.".into());
            return;
        }

        if self.llm_busy {
            return;
        }

        self.messages.push(Message {
            author: USER_NAME.into(),
            content: content.to_string(),
            created_at: Local::now(),
            is_thinking: false,
            metrics: None,
        });

        self.draft.clear();
        self.status = None;
        self.scroll_to_bottom = true;

        if self.llm.enabled && self.llm_status.starts_with("Connected") {
            self.request_llm_reply(ctx);
        }
    }

    fn request_llm_reply(&mut self, ctx: &egui::Context) {
        self.llm_busy = true;
        self.llm_status = format!("Connected · {} (thinking…)", self.llm.model);

        self.messages.push(Message {
            author: ASSISTANT_NAME.into(),
            content: "Thinking…".into(),
            created_at: Local::now(),
            is_thinking: true,
            metrics: None,
        });
        self.thinking_message_index = Some(self.messages.len() - 1);
        self.scroll_to_bottom = true;
        ctx.request_repaint();

        let tx = self.llm_tx.clone();
        let config = self.llm.clone();
        let turns = self
            .messages
            .iter()
            .filter(|message| !message.is_thinking)
            .map(|message| ChatTurn {
                role: if message.author == ASSISTANT_NAME {
                    "assistant".into()
                } else {
                    "user".into()
                },
                content: message.content.clone(),
            })
            .collect::<Vec<_>>();

        thread::spawn(move || {
            let (progress_tx, progress_rx) = mpsc::channel();
            let event_tx = tx.clone();
            let progress_handle = thread::spawn(move || {
                while let Ok(event) = progress_rx.recv() {
                    match event {
                        ChatProgressEvent::Thinking(content) => {
                            let _ = event_tx.send(LlmEvent::Thinking { content });
                        }
                        ChatProgressEvent::CommandApprovalNeeded {
                            command,
                            response_tx,
                        } => {
                            let _ = event_tx.send(LlmEvent::CommandApprovalNeeded {
                                command,
                                response_tx,
                            });
                        }
                    }
                }
            });

            let result = llm::chat(&config, &turns, &progress_tx);
            drop(progress_tx);
            progress_handle.join().ok();

            match result {
                Ok(reply) => {
                    let _ = tx.send(LlmEvent::Reply {
                        content: reply.content,
                        metrics: reply.metrics,
                    });
                }
                Err(err) => {
                    let _ = tx.send(LlmEvent::Failed { message: err });
                }
            }
        });
    }

    fn approve_command(&mut self) {
        let Some(prompt) = self.command_prompt.take() else {
            return;
        };

        thread::spawn(move || {
            let result = llm::execute_shell_command(&prompt.command);
            let _ = prompt.response_tx.send(result);
        });
    }

    fn reject_command(&mut self) {
        if let Some(prompt) = self.command_prompt.take() {
            let _ = prompt
                .response_tx
                .send(Ok("User rejected running this command.".into()));
        }
    }

    fn remove_thinking_message(&mut self) {
        if let Some(index) = self.thinking_message_index.take() {
            if index < self.messages.len() {
                self.messages.remove(index);
            }
        }
    }

    fn poll_llm_events(&mut self, ctx: &egui::Context) {
        while let Ok(event) = self.llm_rx.try_recv() {
            match event {
                LlmEvent::Models { models } => {
                    if models.is_empty() {
                        self.llm_status =
                            "Connected, but no models found. Run `ollama pull llama3.2`.".into();
                    } else if !models.iter().any(|name| name == &self.llm.model) {
                        self.llm.model = models[0].clone();
                        self.llm_status = format!("Connected · using {}", self.llm.model);
                    } else {
                        self.llm_status = format!("Connected · using {}", self.llm.model);
                    }
                }
                LlmEvent::Thinking { content } => {
                    if let Some(index) = self.thinking_message_index {
                        if let Some(message) = self.messages.get_mut(index) {
                            message.content = content;
                            self.scroll_to_bottom = true;
                        }
                    }
                }
                LlmEvent::CommandApprovalNeeded {
                    command,
                    response_tx,
                } => {
                    self.command_prompt = Some(CommandPrompt {
                        command,
                        response_tx,
                    });
                }
                LlmEvent::Reply { content, metrics } => {
                    if let Some(index) = self.thinking_message_index.take() {
                        if let Some(message) = self.messages.get_mut(index) {
                            message.content = content;
                            message.is_thinking = false;
                            message.created_at = Local::now();
                            message.metrics = Some(metrics);
                        }
                    } else {
                        self.messages.push(Message {
                            author: ASSISTANT_NAME.into(),
                            content,
                            created_at: Local::now(),
                            is_thinking: false,
                            metrics: Some(metrics),
                        });
                    }
                    self.llm_busy = false;
                    self.llm_status = format!("Connected · using {}", self.llm.model);
                    self.scroll_to_bottom = true;
                }
                LlmEvent::Failed { message } => {
                    if self.llm_busy {
                        self.remove_thinking_message();
                        self.llm_busy = false;
                    }
                    self.command_prompt = None;
                    self.llm_status = message.clone();
                    self.status = Some(message);
                }
            }
            ctx.request_repaint();
        }
    }
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_llm_events(ctx);
        self.show_command_prompt(ctx);

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.heading("PairLLM Chat");
                ui.label(
                    egui::RichText::new("Chat locally with Ollama when available")
                        .weak()
                        .italics(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Settings").clicked() {
                        self.show_settings = !self.show_settings;
                    }
                });
            });

            ui.horizontal(|ui| {
                let color = if self.llm_status.starts_with("Connected") {
                    egui::Color32::from_rgb(120, 214, 143)
                } else if self.llm_busy {
                    egui::Color32::from_rgb(255, 196, 96)
                } else {
                    egui::Color32::from_rgb(255, 143, 143)
                };
                ui.colored_label(color, &self.llm_status);
                if ui.small_button("Refresh").clicked() {
                    self.check_llm();
                }
            });

            if self.show_settings {
                ui.separator();
                ui.checkbox(&mut self.llm.enabled, "Reply with local LLM");
                ui.horizontal(|ui| {
                    ui.label("Ollama URL");
                    ui.text_edit_singleline(&mut self.llm.base_url);
                });
                ui.horizontal(|ui| {
                    ui.label("Model");
                    ui.text_edit_singleline(&mut self.llm.model);
                });
                ui.horizontal(|ui| {
                    ui.label("Context");
                    ui.add(
                        egui::DragValue::new(&mut self.llm.num_ctx)
                            .range(512..=262_144)
                            .speed(256),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Tavily API key");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.llm.tavily_api_key)
                            .password(true)
                            .hint_text("optional — uses keyless mode if empty"),
                    );
                });
                ui.label(
                    egui::RichText::new(
                        "Recommended: install Ollama, run `ollama pull llama3.2`, then Refresh. \
                         CLI commands require your approval before running.",
                    )
                    .weak()
                    .small(),
                );
            }

            ui.add_space(8.0);
        });

        egui::TopBottomPanel::bottom("composer").show(ctx, |ui| {
            ui.add_space(8.0);

            let composer_enabled = !self.llm_busy && self.command_prompt.is_none();

            ui.horizontal(|ui| {
                let response = ui.add_enabled(
                    composer_enabled,
                    egui::TextEdit::multiline(&mut self.draft)
                        .desired_width(f32::INFINITY)
                        .desired_rows(3)
                        .hint_text("Write a message… (Enter to send, Shift+Enter for newline)"),
                );

                let send_clicked = ui
                    .add_enabled(composer_enabled, egui::Button::new("Send"))
                    .clicked();

                let enter_pressed = response.has_focus()
                    && ui.input(|input| {
                        input.key_pressed(egui::Key::Enter) && !input.modifiers.shift
                    });

                if send_clicked || enter_pressed {
                    self.send_message(ctx);
                    response.request_focus();
                }
            });

            if let Some(status) = &self.status {
                ui.colored_label(egui::Color32::from_rgb(255, 143, 143), status);
            } else {
                ui.add_space(4.0);
            }

            ui.add_space(8.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.messages.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() * 0.35);
                    ui.label(
                        egui::RichText::new("No messages yet. Say hello.")
                            .weak()
                            .size(16.0),
                    );
                });
                return;
            }

            let scroll_to_bottom = self.scroll_to_bottom;
            self.scroll_to_bottom = false;

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(scroll_to_bottom)
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 10.0;

                    for message in &self.messages {
                        let is_user = message.author == USER_NAME;
                        let is_assistant = message.author == ASSISTANT_NAME;

                        ui.horizontal(|ui| {
                            if is_user {
                                ui.add_space(ui.available_width() * 0.15);
                            }

                            ui.vertical(|ui| {
                                ui.set_max_width(ui.available_width());

                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(if message.is_thinking {
                                            "Assistant (thinking)"
                                        } else {
                                            &message.author
                                        })
                                        .strong()
                                        .color(if is_assistant {
                                            egui::Color32::from_rgb(120, 214, 143)
                                        } else {
                                            egui::Color32::from_rgb(108, 140, 255)
                                        }),
                                    );
                                    ui.label(
                                        egui::RichText::new(
                                            message.created_at.format("%b %d, %H:%M").to_string(),
                                        )
                                        .weak()
                                        .small(),
                                    );
                                });

                                let frame = egui::Frame::new()
                                    .fill(if is_user {
                                        egui::Color32::from_rgb(36, 48, 74)
                                    } else if is_assistant {
                                        egui::Color32::from_rgb(28, 44, 38)
                                    } else {
                                        egui::Color32::from_rgb(31, 36, 48)
                                    })
                                    .stroke(egui::Stroke::new(
                                        1.0,
                                        egui::Color32::from_rgb(42, 49, 64),
                                    ))
                                    .inner_margin(10.0)
                                    .corner_radius(8.0);

                                frame.show(ui, |ui| {
                                    let text = if message.is_thinking {
                                        egui::RichText::new(&message.content)
                                            .weak()
                                            .italics()
                                    } else {
                                        egui::RichText::new(&message.content)
                                    };
                                    ui.label(text);
                                });

                                if is_assistant && !message.is_thinking {
                                    if let Some(metrics) = &message.metrics {
                                        let summary = metrics.summary_line();
                                        let tooltip = metrics.tooltip_text();
                                        ui.add_space(4.0);
                                        ui.label(egui::RichText::new(summary).weak().small())
                                            .on_hover_text(tooltip);
                                    }
                                }
                            });

                            if !is_user {
                                ui.add_space(ui.available_width() * 0.15);
                            }
                        });
                    }
                });
        });
    }
}

impl ChatApp {
    fn show_command_prompt(&mut self, ctx: &egui::Context) {
        let Some(prompt) = self.command_prompt.as_ref() else {
            return;
        };

        let command = prompt.command.clone();

        egui::Window::new("Approve command")
            .collapsible(false)
            .resizable(true)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .default_width(520.0)
            .show(ctx, |ui| {
                ui.label("The assistant wants to run this shell command:");
                ui.add_space(8.0);

                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(15, 18, 26))
                    .stroke(egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgb(255, 196, 96),
                    ))
                    .inner_margin(12.0)
                    .corner_radius(8.0)
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new(format!("$ {command}")).monospace());
                    });

                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new("Only approve commands you trust.")
                        .weak()
                        .small(),
                );

                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if ui.button("Reject").clicked() {
                        self.reject_command();
                    }

                    if ui.button("Run command").clicked() {
                        self.approve_command();
                    }
                });
            });
    }
}
