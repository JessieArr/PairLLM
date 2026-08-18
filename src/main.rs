mod llm;

use chrono::{DateTime, Local};
use eframe::egui;
use llm::{ChatTurn, LlmConfig};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

const ASSISTANT_NAME: &str = "Assistant";

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
}

enum LlmEvent {
    Models { models: Vec<String> },
    Reply { content: String },
    Failed { message: String },
}

struct ChatApp {
    messages: Vec<Message>,
    author: String,
    draft: String,
    status: Option<String>,
    scroll_to_bottom: bool,
    llm: LlmConfig,
    show_settings: bool,
    llm_status: String,
    llm_busy: bool,
    llm_tx: Sender<LlmEvent>,
    llm_rx: Receiver<LlmEvent>,
}

impl ChatApp {
    fn new() -> Self {
        let (llm_tx, llm_rx) = mpsc::channel();
        let mut app = Self {
            messages: Vec::new(),
            author: String::new(),
            draft: String::new(),
            status: None,
            scroll_to_bottom: false,
            llm: LlmConfig::default(),
            show_settings: false,
            llm_status: "Checking for Ollama…".into(),
            llm_busy: false,
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
        let author = self.author.trim();
        let content = self.draft.trim();

        if author.is_empty() || content.is_empty() {
            self.status = Some("Enter your name and a message before sending.".into());
            return;
        }

        if self.llm_busy {
            return;
        }

        self.messages.push(Message {
            author: author.to_string(),
            content: content.to_string(),
            created_at: Local::now(),
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
        ctx.request_repaint();

        let tx = self.llm_tx.clone();
        let base_url = self.llm.base_url.clone();
        let model = self.llm.model.clone();
        let turns = self
            .messages
            .iter()
            .map(|message| ChatTurn {
                role: if message.author == ASSISTANT_NAME {
                    "assistant".into()
                } else {
                    "user".into()
                },
                content: if message.author == ASSISTANT_NAME {
                    message.content.clone()
                } else {
                    format!("{}: {}", message.author, message.content)
                },
            })
            .collect::<Vec<_>>();

        thread::spawn(move || {
            let result = llm::chat(&base_url, &model, &turns);
            match result {
                Ok(content) => {
                    let _ = tx.send(LlmEvent::Reply { content });
                }
                Err(err) => {
                    let _ = tx.send(LlmEvent::Failed { message: err });
                }
            }
        });
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
                LlmEvent::Reply { content } => {
                    self.messages.push(Message {
                        author: ASSISTANT_NAME.into(),
                        content,
                        created_at: Local::now(),
                    });
                    self.llm_busy = false;
                    self.llm_status = format!("Connected · using {}", self.llm.model);
                    self.scroll_to_bottom = true;
                }
                LlmEvent::Failed { message } => {
                    self.llm_busy = false;
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
                ui.label(
                    egui::RichText::new(
                        "Recommended: install Ollama, run `ollama pull llama3.2`, then Refresh.",
                    )
                    .weak()
                    .small(),
                );
            }

            ui.add_space(8.0);
        });

        egui::TopBottomPanel::bottom("composer").show(ctx, |ui| {
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                ui.label("Name");
                ui.add_enabled(
                    !self.llm_busy,
                    egui::TextEdit::singleline(&mut self.author)
                        .desired_width(140.0)
                        .hint_text("Your name"),
                );
            });

            ui.add_space(6.0);

            ui.horizontal(|ui| {
                let response = ui.add_enabled(
                    !self.llm_busy,
                    egui::TextEdit::multiline(&mut self.draft)
                        .desired_width(f32::INFINITY)
                        .desired_rows(3)
                        .hint_text("Write a message… (Enter to send, Shift+Enter for newline)"),
                );

                let send_clicked = ui
                    .add_enabled(!self.llm_busy, egui::Button::new("Send"))
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

            if self.llm_busy {
                ui.label(
                    egui::RichText::new("Waiting for the assistant…")
                        .weak()
                        .italics(),
                );
            }

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

            let author = self.author.trim().to_string();
            let scroll_to_bottom = self.scroll_to_bottom;
            self.scroll_to_bottom = false;

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(scroll_to_bottom)
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 10.0;

                    for message in &self.messages {
                        let is_self = message.author == author;
                        let is_assistant = message.author == ASSISTANT_NAME;

                        ui.horizontal(|ui| {
                            if is_self {
                                ui.add_space(ui.available_width() * 0.15);
                            }

                            ui.vertical(|ui| {
                                ui.set_max_width(ui.available_width());

                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(&message.author)
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
                                    .fill(if is_self {
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
                                    ui.label(&message.content);
                                });
                            });

                            if !is_self {
                                ui.add_space(ui.available_width() * 0.15);
                            }
                        });
                    }
                });
        });
    }
}
