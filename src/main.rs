mod llm;
mod permissions;
mod settings;

use chrono::{DateTime, Local};
use eframe::egui;
use llm::{ChatProgressEvent, ChatTrace, ChatTurn, LlmConfig, OllamaMetrics, ToolActionUpdate, QWEN_MODEL_OPTIONS, qwen_model_index};
use permissions::{
    FileAccess, FilePermissionChoice, PathPermission, PathPermissionRule, PathPermissionState,
    SharedPathPermissions,
};
use std::sync::{Arc, Mutex};
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
enum PermissionPrompt {
    Pending {
        directory: String,
        tool_name: String,
        arguments: String,
        access: FileAccess,
    },
    Resolved {
        directory: String,
        choice: FilePermissionChoice,
        saved_persistent: bool,
    },
}

#[derive(Clone)]
struct Message {
    author: String,
    content: String,
    created_at: DateTime<Local>,
    is_thinking: bool,
    is_tool: bool,
    tool_success: Option<bool>,
    metrics: Option<OllamaMetrics>,
    trace: Option<ChatTrace>,
    permission: Option<PermissionPrompt>,
}

struct CommandPrompt {
    command: String,
    response_tx: Sender<Result<String, String>>,
}

struct FilePermissionPrompt {
    response_tx: Sender<FilePermissionChoice>,
    message_index: usize,
}

enum LlmEvent {
    Models { models: Vec<String> },
    Thinking { content: String },
    ToolAction { update: ToolActionUpdate },
    CommandApprovalNeeded {
        command: String,
        response_tx: Sender<Result<String, String>>,
    },
    FilePermissionNeeded {
        tool_name: String,
        arguments: String,
        directory: String,
        access: FileAccess,
        response_tx: Sender<FilePermissionChoice>,
    },
    Reply {
        content: String,
        metrics: OllamaMetrics,
        trace: ChatTrace,
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
    pending_tool_message_index: Option<usize>,
    command_prompt: Option<CommandPrompt>,
    file_permission_prompt: Option<FilePermissionPrompt>,
    path_permissions: SharedPathPermissions,
    new_persistent_rule: String,
    new_persistent_permission: PathPermission,
    llm_tx: Sender<LlmEvent>,
    llm_rx: Receiver<LlmEvent>,
    details_trace: Option<ChatTrace>,
}

impl ChatApp {
    fn new() -> Self {
        let (llm_tx, llm_rx) = mpsc::channel();
        let llm = settings::load().unwrap_or_else(LlmConfig::default);
        let path_permissions = Arc::new(Mutex::new(PathPermissionState {
            persistent: llm.path_permissions.clone(),
            session: Vec::new(),
        }));
        let mut app = Self {
            messages: Vec::new(),
            draft: String::new(),
            status: None,
            scroll_to_bottom: false,
            llm: llm.clone(),
            show_settings: false,
            llm_status: "Checking for Ollama…".into(),
            llm_busy: false,
            thinking_message_index: None,
            pending_tool_message_index: None,
            command_prompt: None,
            file_permission_prompt: None,
            path_permissions,
            new_persistent_rule: String::new(),
            new_persistent_permission: PathPermission::AllowDirectory,
            llm_tx,
            llm_rx,
            details_trace: None,
        };
        app.check_llm();
        app
    }

    fn save_settings(&mut self, refresh_ollama: bool) {
        if let Ok(mut state) = self.path_permissions.lock() {
            state.sync_persistent(&self.llm.path_permissions);
        }

        if let Err(err) = settings::save(&self.llm) {
            self.status = Some(err);
        }

        if refresh_ollama && !self.llm_busy {
            self.check_llm();
        }
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
            is_tool: false,
            tool_success: None,
            metrics: None,
            trace: None,
            permission: None,
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
            is_tool: false,
            tool_success: None,
            metrics: None,
            trace: None,
            permission: None,
        });
        self.thinking_message_index = Some(self.messages.len() - 1);
        self.scroll_to_bottom = true;
        ctx.request_repaint();

        let tx = self.llm_tx.clone();
        let config = self.llm.clone();
        let permissions = Arc::clone(&self.path_permissions);
        let turns = self
            .messages
            .iter()
            .filter(|message| !message.is_thinking && !message.is_tool)
            .map(|message| ChatTurn {
                role: if message.author == ASSISTANT_NAME {
                    "assistant".into()
                } else {
                    "user".into()
                },
                content: message.content.clone(),
            })
            .collect::<Vec<_>>();

        let ctx = ctx.clone();
        thread::spawn(move || {
            let (progress_tx, progress_rx) = mpsc::channel();
            let event_tx = tx.clone();
            let progress_ctx = ctx.clone();
            let progress_handle = thread::spawn(move || {
                while let Ok(event) = progress_rx.recv() {
                    match event {
                        ChatProgressEvent::Thinking(content) => {
                            let _ = event_tx.send(LlmEvent::Thinking { content });
                        }
                        ChatProgressEvent::ToolAction(update) => {
                            let _ = event_tx.send(LlmEvent::ToolAction { update });
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
                        ChatProgressEvent::FilePermissionNeeded {
                            tool_name,
                            arguments,
                            directory,
                            access,
                            response_tx,
                        } => {
                            let _ = event_tx.send(LlmEvent::FilePermissionNeeded {
                                tool_name,
                                arguments,
                                directory,
                                access,
                                response_tx,
                            });
                        }
                    }
                    progress_ctx.request_repaint();
                }
            });

            let result = llm::chat(&config, &turns, &progress_tx, &permissions);
            drop(progress_tx);
            progress_handle.join().ok();

            match result {
                Ok(reply) => {
                    let _ = tx.send(LlmEvent::Reply {
                        content: reply.content,
                        metrics: reply.metrics,
                        trace: reply.trace,
                    });
                }
                Err(err) => {
                    let _ = tx.send(LlmEvent::Failed { message: err });
                }
            }
            ctx.request_repaint();
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

    fn resolve_file_permission(&mut self, choice: FilePermissionChoice) {
        let Some(prompt) = self.file_permission_prompt.take() else {
            return;
        };

        if let Some(message) = self.messages.get_mut(prompt.message_index) {
            if let Some(PermissionPrompt::Pending { directory, .. }) = message.permission.clone() {
                message.permission = Some(PermissionPrompt::Resolved {
                    directory,
                    choice,
                    saved_persistent: false,
                });
                message.created_at = Local::now();
            }
        }

        let _ = prompt.response_tx.send(choice);
    }

    fn save_permission_for_all_sessions(&mut self, message_index: usize) {
        let Some(PermissionPrompt::Resolved {
            directory,
            choice,
            saved_persistent: false,
            ..
        }) = self.messages.get(message_index).and_then(|m| m.permission.clone())
        else {
            return;
        };

        let rule = choice.to_rule(std::path::Path::new(&directory));
        self.llm.path_permissions.push(rule.clone());
        if let Ok(mut state) = self.path_permissions.lock() {
            state.add_persistent_rule(rule);
        }
        self.save_settings(false);

        if let Some(message) = self.messages.get_mut(message_index) {
            if let Some(PermissionPrompt::Resolved {
                saved_persistent, ..
            }) = message.permission.as_mut()
            {
                *saved_persistent = true;
            }
        }
    }

    fn add_persistent_permission_rule(&mut self) {
        let path = self.new_persistent_rule.trim();
        if path.is_empty() {
            self.status = Some("Enter a directory path for the permission rule.".into());
            return;
        }

        let rule = PathPermissionRule {
            path: path.to_string(),
            permission: self.new_persistent_permission.clone(),
        };
        self.llm.path_permissions.push(rule.clone());
        if let Ok(mut state) = self.path_permissions.lock() {
            state.add_persistent_rule(rule);
        }
        self.new_persistent_rule.clear();
        self.save_settings(false);
    }

    fn remove_persistent_permission_rule(&mut self, index: usize) {
        if index >= self.llm.path_permissions.len() {
            return;
        }

        self.llm.path_permissions.remove(index);
        if let Ok(mut state) = self.path_permissions.lock() {
            state.sync_persistent(&self.llm.path_permissions);
        }
        self.save_settings(false);
    }

    fn push_file_permission_prompt(
        &mut self,
        tool_name: String,
        arguments: String,
        directory: String,
        access: FileAccess,
        response_tx: Sender<FilePermissionChoice>,
    ) {
        let insert_at = self.thinking_message_index.unwrap_or(self.messages.len());
        self.messages.insert(
            insert_at,
            Message {
                author: "File access".into(),
                content: String::new(),
                created_at: Local::now(),
                is_thinking: false,
                is_tool: false,
                tool_success: None,
                metrics: None,
                trace: None,
                permission: Some(PermissionPrompt::Pending {
                    directory,
                    tool_name,
                    arguments,
                    access,
                }),
            },
        );
        self.file_permission_prompt = Some(FilePermissionPrompt {
            response_tx,
            message_index: insert_at,
        });

        if let Some(thinking_index) = self.thinking_message_index {
            self.thinking_message_index = Some(thinking_index + 1);
        }

        self.scroll_to_bottom = true;
    }

    fn push_tool_action(&mut self, update: ToolActionUpdate) {
        if update.completed {
            if let Some(index) = self.pending_tool_message_index.take() {
                if let Some(message) = self.messages.get_mut(index) {
                    message.content = format_tool_message_content(&update);
                    message.created_at = Local::now();
                    message.tool_success = Some(update.success);
                }
            }
            return;
        }

        let insert_at = self.thinking_message_index.unwrap_or(self.messages.len());
        self.messages.insert(
            insert_at,
            Message {
                author: format!("Tool · {}", update.name),
                content: format_tool_message_content(&update),
                created_at: Local::now(),
                is_thinking: false,
                is_tool: true,
                tool_success: None,
                metrics: None,
                trace: None,
                permission: None,
            },
        );
        self.pending_tool_message_index = Some(insert_at);

        if let Some(thinking_index) = self.thinking_message_index {
            self.thinking_message_index = Some(thinking_index + 1);
        }

        self.scroll_to_bottom = true;
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
                            "Connected, but no models found. Run `ollama pull qwen3:4b`.".into();
                    } else if models.iter().any(|name| name == &self.llm.model) {
                        self.llm_status = format!("Connected · using {}", self.llm.model);
                    } else {
                        self.llm_status = format!(
                            "Connected · `{}` not installed — run `ollama pull {}`",
                            self.llm.model, self.llm.model
                        );
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
                LlmEvent::ToolAction { update } => {
                    self.push_tool_action(update);
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
                LlmEvent::FilePermissionNeeded {
                    tool_name,
                    arguments,
                    directory,
                    access,
                    response_tx,
                } => {
                    self.push_file_permission_prompt(
                        tool_name,
                        arguments,
                        directory,
                        access,
                        response_tx,
                    );
                }
                LlmEvent::Reply {
                    content,
                    metrics,
                    trace,
                } => {
                    if let Some(index) = self.thinking_message_index.take() {
                        if let Some(message) = self.messages.get_mut(index) {
                            message.content = content;
                            message.is_thinking = false;
                            message.is_tool = false;
                            message.tool_success = None;
                            message.created_at = Local::now();
                            message.metrics = Some(metrics);
                            message.trace = Some(trace);
                        }
                    } else {
                        self.messages.push(Message {
                            author: ASSISTANT_NAME.into(),
                            content,
                            created_at: Local::now(),
                            is_thinking: false,
                            is_tool: false,
                            tool_success: None,
                            metrics: Some(metrics),
                            trace: Some(trace),
                            permission: None,
                        });
                    }
                    self.pending_tool_message_index = None;
                    self.llm_busy = false;
                    self.llm_status = format!("Connected · using {}", self.llm.model);
                    self.scroll_to_bottom = true;
                }
                LlmEvent::Failed { message } => {
                    if self.llm_busy {
                        self.remove_thinking_message();
                        self.llm_busy = false;
                    }
                    self.pending_tool_message_index = None;
                    self.command_prompt = None;
                    self.file_permission_prompt = None;
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
        self.show_details_modal(ctx);

        if self.llm_busy {
            ctx.request_repaint();
        }

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
                let mut settings_changed = false;
                let mut ollama_changed = false;

                if ui
                    .checkbox(&mut self.llm.enabled, "Reply with local LLM")
                    .changed()
                {
                    settings_changed = true;
                }
                ui.horizontal(|ui| {
                    ui.label("Ollama URL");
                    if ui.text_edit_singleline(&mut self.llm.base_url).changed() {
                        settings_changed = true;
                        ollama_changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Qwen model");
                    let mut selected = qwen_model_index(&self.llm.model);
                    let selected_text = selected
                        .map(|index| QWEN_MODEL_OPTIONS[index].label)
                        .unwrap_or("Custom");

                    let mut model_changed = false;
                    egui::ComboBox::from_id_salt("qwen_model")
                        .selected_text(selected_text)
                        .show_ui(ui, |ui| {
                            for (index, option) in QWEN_MODEL_OPTIONS.iter().enumerate() {
                                if ui
                                    .selectable_label(selected == Some(index), option.label)
                                    .clicked()
                                {
                                    selected = Some(index);
                                    self.llm.model = option.tag.to_string();
                                    model_changed = true;
                                }
                            }
                        });

                    if model_changed {
                        settings_changed = true;
                        ollama_changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Model tag");
                    if ui.text_edit_singleline(&mut self.llm.model).changed() {
                        settings_changed = true;
                        ollama_changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Context");
                    if ui
                        .add(
                            egui::DragValue::new(&mut self.llm.num_ctx)
                                .range(512..=262_144)
                                .speed(256),
                        )
                        .changed()
                    {
                        settings_changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Tavily API key");
                    if ui
                        .add(
                            egui::TextEdit::singleline(&mut self.llm.tavily_api_key)
                                .password(true)
                                .hint_text("optional — uses keyless mode if empty"),
                        )
                        .changed()
                    {
                        settings_changed = true;
                    }
                });

                ui.add_space(8.0);
                ui.label(egui::RichText::new("File access permissions").strong());
                ui.label(
                    egui::RichText::new(
                        "Rules apply to ls, cat, and sed. The most specific matching path wins.",
                    )
                    .weak()
                    .small(),
                );

                let mut remove_index: Option<usize> = None;
                for (index, rule) in self.llm.path_permissions.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.monospace(&rule.path);
                        ui.label(
                            egui::RichText::new(rule.permission.label())
                                .weak()
                                .small(),
                        );
                        if ui.small_button("Remove").clicked() {
                            remove_index = Some(index);
                        }
                    });
                }
                if let Some(index) = remove_index {
                    self.remove_persistent_permission_rule(index);
                }

                ui.horizontal(|ui| {
                    ui.label("Add rule");
                    if ui
                        .add(
                            egui::TextEdit::singleline(&mut self.new_persistent_rule)
                                .hint_text("/path/to/directory"),
                        )
                        .changed()
                    {
                        settings_changed = true;
                    }
                    egui::ComboBox::from_id_salt("new_path_permission")
                        .selected_text(self.new_persistent_permission.label())
                        .show_ui(ui, |ui| {
                            for option in [
                                PathPermission::AllowDirectory,
                                PathPermission::AllowRecursive,
                                PathPermission::Deny,
                            ] {
                                if ui
                                    .selectable_label(
                                        self.new_persistent_permission == option,
                                        option.label(),
                                    )
                                    .clicked()
                                {
                                    self.new_persistent_permission = option;
                                    settings_changed = true;
                                }
                            }
                        });
                    if ui.button("Add").clicked() {
                        self.add_persistent_permission_rule();
                        settings_changed = true;
                    }
                });

                if settings_changed {
                    self.save_settings(ollama_changed);
                }

                ui.label(
                    egui::RichText::new(
                        "Pull models with Ollama (e.g. `ollama pull qwen3:4b`), pick a size above, \
                         then Refresh. Shell commands and file tool access require your approval.",
                    )
                    .weak()
                    .small(),
                );
            }

            ui.add_space(8.0);
        });

        egui::TopBottomPanel::bottom("composer").show(ctx, |ui| {
            ui.add_space(8.0);

            let composer_enabled = !self.llm_busy
                && self.command_prompt.is_none()
                && self.file_permission_prompt.is_none();

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
                    let mut open_details: Option<ChatTrace> = None;
                    let mut permission_choice: Option<FilePermissionChoice> = None;
                    let mut save_permission_index: Option<usize> = None;

                    for (index, message) in self.messages.iter().enumerate() {
                        let is_user = message.author == USER_NAME;
                        let is_assistant = message.author == ASSISTANT_NAME;
                        let is_tool = message.is_tool;
                        let is_permission = message.permission.is_some();

                        ui.horizontal(|ui| {
                            if is_user {
                                ui.add_space(ui.available_width() * 0.15);
                            }

                            ui.vertical(|ui| {
                                ui.set_max_width(ui.available_width());

                                ui.horizontal(|ui| {
                                    let author_label = if message.is_thinking {
                                        "Assistant (thinking)"
                                    } else {
                                        &message.author
                                    };
                                    ui.label(
                                        egui::RichText::new(author_label)
                                        .strong()
                                        .color(if is_permission {
                                            egui::Color32::from_rgb(196, 160, 255)
                                        } else if is_tool {
                                            egui::Color32::from_rgb(255, 196, 96)
                                        } else if is_assistant {
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
                                    } else if is_permission {
                                        egui::Color32::from_rgb(40, 32, 52)
                                    } else if is_tool {
                                        egui::Color32::from_rgb(44, 36, 24)
                                    } else if is_assistant {
                                        egui::Color32::from_rgb(28, 44, 38)
                                    } else {
                                        egui::Color32::from_rgb(31, 36, 48)
                                    })
                                    .stroke(egui::Stroke::new(
                                        1.0,
                                        if is_permission {
                                            egui::Color32::from_rgb(196, 160, 255)
                                        } else if is_tool {
                                            egui::Color32::from_rgb(255, 196, 96)
                                        } else {
                                            egui::Color32::from_rgb(42, 49, 64)
                                        },
                                    ))
                                    .inner_margin(10.0)
                                    .corner_radius(8.0);

                                frame.show(ui, |ui| {
                                    if let Some(permission) = &message.permission {
                                        match permission {
                                            PermissionPrompt::Pending {
                                                directory,
                                                tool_name,
                                                arguments,
                                                access,
                                            } => {
                                                ui.label(format!(
                                                    "The assistant wants to {} files via `{tool_name}` in:",
                                                    access.label()
                                                ));
                                                ui.label(
                                                    egui::RichText::new(directory).monospace(),
                                                );
                                                ui.add_space(6.0);
                                                ui.label(
                                                    egui::RichText::new(arguments)
                                                        .monospace()
                                                        .weak(),
                                                );
                                                ui.add_space(8.0);

                                                let is_active = self
                                                    .file_permission_prompt
                                                    .as_ref()
                                                    .is_some_and(|prompt| {
                                                        prompt.message_index == index
                                                    });

                                                ui.horizontal(|ui| {
                                                    ui.add_enabled_ui(is_active, |ui| {
                                                        if ui
                                                            .button("Allow for this directory")
                                                            .clicked()
                                                        {
                                                            permission_choice =
                                                                Some(FilePermissionChoice::AllowDirectory);
                                                        }
                                                        if ui.button("Allow recursively").clicked()
                                                        {
                                                            permission_choice = Some(
                                                                FilePermissionChoice::AllowRecursive,
                                                            );
                                                        }
                                                        if ui.button("Reject").clicked() {
                                                            permission_choice =
                                                                Some(FilePermissionChoice::Reject);
                                                        }
                                                    });
                                                });
                                            }
                                            PermissionPrompt::Resolved {
                                                directory,
                                                choice,
                                                saved_persistent,
                                            } => {
                                                ui.label(format!(
                                                    "The assistant requested access to `{directory}`"
                                                ));
                                                ui.add_space(6.0);
                                                let status_color = match choice {
                                                    FilePermissionChoice::Reject => {
                                                        egui::Color32::from_rgb(255, 143, 143)
                                                    }
                                                    _ => egui::Color32::from_rgb(196, 220, 196),
                                                };
                                                ui.label(
                                                    egui::RichText::new(choice.session_status())
                                                        .color(status_color)
                                                        .strong(),
                                                );
                                                if !saved_persistent {
                                                    ui.add_space(8.0);
                                                    if ui.button("Save for all sessions").clicked()
                                                    {
                                                        save_permission_index = Some(index);
                                                    }
                                                } else {
                                                    ui.add_space(6.0);
                                                    ui.label(
                                                        egui::RichText::new(
                                                            "Saved for all sessions",
                                                        )
                                                        .weak()
                                                        .small(),
                                                    );
                                                }
                                            }
                                        }
                                    } else if is_tool {
                                        if let Some((arguments, summary)) =
                                            message.content.split_once("\n\n")
                                        {
                                            ui.label(
                                                egui::RichText::new(arguments).monospace().weak(),
                                            );
                                            ui.add_space(6.0);
                                            let summary_color = match message.tool_success {
                                                Some(false) => {
                                                    egui::Color32::from_rgb(255, 143, 143)
                                                }
                                                Some(true) => {
                                                    egui::Color32::from_rgb(196, 220, 196)
                                                }
                                                None => egui::Color32::from_rgb(180, 180, 180),
                                            };
                                            ui.label(
                                                egui::RichText::new(summary)
                                                    .color(summary_color)
                                                    .italics(),
                                            );
                                        } else {
                                            ui.label(
                                                egui::RichText::new(&message.content).monospace(),
                                            );
                                        }
                                    } else {
                                        let text = if message.is_thinking {
                                            egui::RichText::new(&message.content)
                                                .weak()
                                                .italics()
                                        } else {
                                            egui::RichText::new(&message.content)
                                        };
                                        ui.label(text);
                                    }
                                });

                                if is_assistant && !message.is_thinking {
                                    ui.horizontal(|ui| {
                                        if let Some(metrics) = &message.metrics {
                                            let summary = metrics.summary_line();
                                            let tooltip = metrics.tooltip_text();
                                            ui.label(egui::RichText::new(summary).weak().small())
                                                .on_hover_text(tooltip);
                                        }

                                        if message.trace.is_some() {
                                            if ui.small_button("Details").clicked() {
                                                open_details = message.trace.clone();
                                            }
                                        }
                                    });
                                }
                            });

                            if !is_user {
                                ui.add_space(ui.available_width() * 0.15);
                            }
                        });
                    }

                    if let Some(trace) = open_details {
                        self.details_trace = Some(trace);
                    }
                    if let Some(choice) = permission_choice {
                        self.resolve_file_permission(choice);
                    }
                    if let Some(index) = save_permission_index {
                        self.save_permission_for_all_sessions(index);
                    }
                });
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let _ = settings::save(&self.llm);
    }
}

impl ChatApp {
    fn show_details_modal(&mut self, ctx: &egui::Context) {
        let Some(trace) = self.details_trace.clone() else {
            return;
        };

        let mut open = true;
        egui::Window::new("LLM exchange details")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .default_width(720.0)
            .default_height(640.0)
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new(
                        "Everything sent to and received from Ollama for this reply, including \
                         the system prompt and any tool-call rounds.",
                    )
                    .weak()
                    .small(),
                );
                ui.add_space(8.0);

                egui::ScrollArea::vertical()
                    .id_salt("details_trace")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for (index, round) in trace.rounds.iter().enumerate() {
                            let default_open = index + 1 == trace.rounds.len();
                            ui.push_id(index, |ui| {
                                egui::CollapsingHeader::new(format!("Round {}", index + 1))
                                    .default_open(default_open)
                                    .show(ui, |ui| {
                                        ui.label(egui::RichText::new("Sent to LLM").strong());
                                        ui.add_space(4.0);
                                        show_json_block(ui, &round.request, "request");

                                        ui.add_space(12.0);
                                        ui.label(egui::RichText::new("Received from LLM").strong());
                                        ui.add_space(4.0);
                                        show_json_block(ui, &round.response, "response");
                                    });
                            });

                            ui.add_space(8.0);
                        }
                    });
            });

        if !open {
            self.details_trace = None;
        }
    }

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

fn format_tool_message_content(update: &ToolActionUpdate) -> String {
    format!("{}\n\n{}", update.arguments, update.summary)
}

fn show_json_block(ui: &mut egui::Ui, value: &serde_json::Value, block_id: &str) {
    let text = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());

    ui.push_id(block_id, |ui| {
        egui::Frame::new()
            .fill(egui::Color32::from_rgb(15, 18, 26))
            .stroke(egui::Stroke::new(
                1.0,
                egui::Color32::from_rgb(42, 49, 64),
            ))
            .inner_margin(10.0)
            .corner_radius(8.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("json")
                    .max_height(320.0)
                    .show(ui, |ui| {
                        ui.add(
                            egui::Label::new(egui::RichText::new(text).monospace())
                                .selectable(true),
                        );
                    });
            });
    });
}
