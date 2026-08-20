use chrono::Local;
use crate::permissions::{
    self, FileAccess, FilePermissionChoice, PathPermissionRule, PermissionCheck,
    SharedPathPermissions, check_path_permission, file_access_for_tool,
    is_file_permission_tool, permission_directory_for_target,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{mpsc::{self, Sender}, OnceLock};

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:11434";
const DEFAULT_MODEL: &str = "qwen3:4b";
const DEFAULT_NUM_CTX: u32 = 16384;
const MAX_TOOL_ROUNDS: usize = 8;
const TAVILY_SEARCH_URL: &str = "https://api.tavily.com/search";
const MAX_COMMAND_OUTPUT: usize = 8000;
const MAX_TOOL_RESULT_CHARS: usize = 4000;
const MAX_CONTEXT_MESSAGES: usize = 20;
const MAX_CONTEXT_MESSAGE_CHARS: usize = 3000;
const KEEP_ALIVE: &str = "30m";

#[derive(Clone, Copy, PartialEq, Eq)]
enum ToolPlatform {
    All,
    Unix,
}

impl ToolPlatform {
    fn is_supported(self) -> bool {
        match self {
            ToolPlatform::All => true,
            ToolPlatform::Unix => cfg!(unix),
        }
    }
}

struct ToolSpec {
    name: &'static str,
    platform: ToolPlatform,
    prompt_line: &'static str,
    json_example: &'static str,
}

const TOOL_SPECS: &[ToolSpec] = &[
    ToolSpec {
        name: "web_search",
        platform: ToolPlatform::All,
        prompt_line: "- web_search: searches the web for up-to-date information.",
        json_example: r#"{"tool":"web_search","query":"your search query"}"#,
    },
    ToolSpec {
        name: "ls",
        platform: ToolPlatform::Unix,
        prompt_line: "- ls: lists files in a directory, like `ls /path/to/dir`. You choose the \
directory path and optional flags (a, l, R).",
        json_example: r#"{"tool":"ls","path":"/path/to/dir","flags":"la"}"#,
    },
    ToolSpec {
        name: "cat",
        platform: ToolPlatform::Unix,
        prompt_line: "- cat: prints a file's contents, like `cat /path/to/file`. Optional flags: n (line numbers). \
When looking for something specific in a file, pass a grep-style pattern and use flags \"n\" \
so you know which lines matched.",
        json_example: r#"{"tool":"cat","path":"/path/to/file","pattern":"fn main","flags":"n"}"#,
    },
    ToolSpec {
        name: "sed",
        platform: ToolPlatform::Unix,
        prompt_line: "- sed: search-and-replace in a file, like `sed -i 's/pattern/replacement/' file.txt`. \
You choose the file path and the full sed substitution expression. In the replacement, escape \
special characters: \\ for backslash, \\& for &, \\/ for / (or use another delimiter), and \\1 etc. \
for backreferences.",
        json_example: r#"{"tool":"sed","path":"/path/to/file.txt","expression":"s/old/new/"}"#,
    },
    ToolSpec {
        name: "run_command",
        platform: ToolPlatform::All,
        prompt_line: "- run_command: runs a shell command on the user's machine. The user must approve \
before it runs.",
        json_example: r#"{"tool":"run_command","command":"the shell command"}"#,
    },
];

fn available_tool_specs() -> impl Iterator<Item = &'static ToolSpec> {
    TOOL_SPECS
        .iter()
        .filter(|spec| spec.platform.is_supported())
}

fn tool_system_prompt_body() -> String {
    let specs: Vec<_> = available_tool_specs().collect();
    let count = specs.len();
    let tool_word = if count == 1 { "tool" } else { "tools" };

    let mut lines = vec![
        "You are a helpful assistant in a chat app.".into(),
        String::new(),
        format!("You have {count} {tool_word}:"),
    ];

    for spec in &specs {
        lines.push(spec.prompt_line.to_string());
    }

    lines.push(String::new());
    lines.push("If you need a tool, respond with ONLY a JSON object and nothing else:".into());
    for spec in &specs {
        lines.push(spec.json_example.to_string());
    }
    lines.push(String::new());
    lines.push("After you receive a tool result, answer the user in plain language.".into());
    lines.join("\n")
}

fn operating_system_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "macOS"
    } else if cfg!(target_os = "linux") {
        "Linux"
    } else if cfg!(target_os = "windows") {
        "Windows"
    } else if cfg!(target_os = "freebsd") {
        "FreeBSD"
    } else {
        "Unknown"
    }
}

fn environment_context() -> String {
    let mut lines = vec![format!("Operating system: {}", operating_system_name())];

    match home_dir() {
        Some(path) => {
            lines.push(format!("Home directory: {}", path.display()));
            lines.push("In file tool paths, `~` expands to this home directory.".into());
        }
        None => lines.push("Home directory: (unknown)".into()),
    }

    lines.join("\n")
}

fn system_prompt() -> String {
    let now = Local::now()
        .format("%A, %B %d, %Y at %I:%M %p %Z")
        .to_string();
    format!(
        "Current time: {now}\n{}\n\n{}",
        environment_context(),
        tool_system_prompt_body()
    )
}

pub struct QwenModelOption {
    pub label: &'static str,
    pub tag: &'static str,
}

pub const QWEN_MODEL_OPTIONS: &[QwenModelOption] = &[
    QwenModelOption {
        label: "Qwen3 0.6B",
        tag: "qwen3:0.6b",
    },
    QwenModelOption {
        label: "Qwen3 1.7B",
        tag: "qwen3:1.7b",
    },
    QwenModelOption {
        label: "Qwen3 4B",
        tag: "qwen3:4b",
    },
];

pub fn qwen_model_index(model: &str) -> Option<usize> {
    QWEN_MODEL_OPTIONS
        .iter()
        .position(|option| option.tag == model)
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub enabled: bool,
    pub base_url: String,
    pub model: String,
    pub num_ctx: u32,
    pub tavily_api_key: String,
    #[serde(default)]
    pub path_permissions: Vec<PathPermissionRule>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: DEFAULT_BASE_URL.into(),
            model: DEFAULT_MODEL.into(),
            num_ctx: DEFAULT_NUM_CTX,
            tavily_api_key: std::env::var("TAVILY_API_KEY").unwrap_or_default(),
            path_permissions: Vec::new(),
        }
    }
}

#[derive(Clone, Default)]
pub struct OllamaMetrics {
    pub total_duration_ns: u64,
    pub load_duration_ns: u64,
    pub prompt_eval_count: u64,
    pub prompt_eval_duration_ns: u64,
    pub eval_count: u64,
    pub eval_duration_ns: u64,
    pub request_count: u32,
    pub http_headers: Vec<(String, String)>,
}

impl OllamaMetrics {
    pub fn merge(&mut self, other: &Self) {
        self.total_duration_ns += other.total_duration_ns;
        self.load_duration_ns += other.load_duration_ns;
        self.prompt_eval_count += other.prompt_eval_count;
        self.prompt_eval_duration_ns += other.prompt_eval_duration_ns;
        self.eval_count += other.eval_count;
        self.eval_duration_ns += other.eval_duration_ns;
        self.request_count += other.request_count;

        for (name, value) in &other.http_headers {
            if let Some(existing) = self
                .http_headers
                .iter_mut()
                .find(|(existing_name, _)| existing_name == name)
            {
                existing.1 = value.clone();
            } else {
                self.http_headers.push((name.clone(), value.clone()));
            }
        }
    }

    pub fn summary_line(&self) -> String {
        format!(
            "{:.2}s · {:.1} tok/s",
            self.total_duration_ns as f64 / 1e9,
            self.generation_tokens_per_second()
        )
    }

    pub fn tooltip_text(&self) -> String {
        let mut lines = vec![
            format!(
                "Ollama usage ({} API request{})",
                self.request_count,
                if self.request_count == 1 { "" } else { "s" }
            ),
            format!(
                "total_duration: {:.2} ms",
                self.total_duration_ns as f64 / 1e6
            ),
            format!(
                "load_duration: {:.2} ms",
                self.load_duration_ns as f64 / 1e6
            ),
            format!("prompt_eval_count: {}", self.prompt_eval_count),
            format!(
                "prompt_eval_duration: {:.2} ms ({:.1} tok/s prefill)",
                self.prompt_eval_duration_ns as f64 / 1e6,
                self.prefill_tokens_per_second()
            ),
            format!("eval_count: {}", self.eval_count),
            format!(
                "eval_duration: {:.2} ms ({:.1} tok/s gen)",
                self.eval_duration_ns as f64 / 1e6,
                self.generation_tokens_per_second()
            ),
        ];

        if !self.http_headers.is_empty() {
            lines.push(String::new());
            lines.push("Response headers:".into());
            for (name, value) in &self.http_headers {
                lines.push(format!("{name}: {value}"));
            }
        }

        lines.join("\n")
    }

    fn prefill_tokens_per_second(&self) -> f64 {
        tokens_per_second(self.prompt_eval_count, self.prompt_eval_duration_ns)
    }

    fn generation_tokens_per_second(&self) -> f64 {
        tokens_per_second(self.eval_count, self.eval_duration_ns)
    }
}

pub struct ChatResult {
    pub content: String,
    pub metrics: OllamaMetrics,
    pub trace: ChatTrace,
}

#[derive(Clone, Default)]
pub struct ChatTrace {
    pub rounds: Vec<ChatRoundTrace>,
}

#[derive(Clone)]
pub struct ChatRoundTrace {
    pub request: Value,
    pub response: Value,
}

fn tokens_per_second(token_count: u64, duration_ns: u64) -> f64 {
    if duration_ns == 0 {
        return 0.0;
    }

    token_count as f64 / (duration_ns as f64 / 1e9)
}

#[derive(Clone, Serialize)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

pub enum ChatProgressEvent {
    Thinking(String),
    ToolAction(ToolActionUpdate),
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
}

#[derive(Clone)]
pub struct ToolActionUpdate {
    pub name: String,
    pub arguments: String,
    pub summary: String,
    pub success: bool,
    pub completed: bool,
}

#[derive(Clone)]
struct ToolInvocation {
    name: String,
    arguments: Value,
}

#[derive(Serialize)]
struct ModelOptions {
    num_ctx: u32,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Value>,
    stream: bool,
    tools: Vec<Value>,
    keep_alive: String,
    truncate: bool,
    options: ModelOptions,
}

#[derive(Deserialize, Default)]
struct StreamChunk {
    message: ChatMessage,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    total_duration: u64,
    #[serde(default)]
    load_duration: u64,
    #[serde(default)]
    prompt_eval_count: u64,
    #[serde(default)]
    prompt_eval_duration: u64,
    #[serde(default)]
    eval_count: u64,
    #[serde(default)]
    eval_duration: u64,
}

struct StreamChatResult {
    message: ChatMessage,
    metrics: OllamaMetrics,
}

#[derive(Deserialize, Default)]
struct ChatMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
}

#[derive(Serialize)]
struct TavilySearchRequest<'a> {
    query: &'a str,
    max_results: u8,
    include_answer: bool,
}

#[derive(Clone, Deserialize, Serialize)]
struct ToolCall {
    function: ToolFunction,
}

#[derive(Clone, Deserialize, Serialize)]
struct ToolFunction {
    name: String,
    #[serde(default)]
    arguments: Value,
}

#[derive(Deserialize)]
struct TagsResponse {
    models: Vec<TagModel>,
}

#[derive(Deserialize)]
struct TagModel {
    name: String,
}

#[derive(Deserialize)]
struct TavilySearchResponse {
    #[serde(default)]
    answer: Option<String>,
    #[serde(default)]
    results: Vec<TavilyResult>,
}

#[derive(Deserialize)]
struct TavilyResult {
    title: String,
    url: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    score: f32,
}

pub fn list_models(base_url: &str) -> Result<Vec<String>, String> {
    let url = format!("{}/api/tags", normalize_base_url(base_url));
    let response = reqwest::blocking::get(&url).map_err(connection_error)?;
    if !response.status().is_success() {
        return Err(format!("Ollama returned HTTP {}", response.status()));
    }

    let tags: TagsResponse = response.json().map_err(|err| err.to_string())?;
    Ok(tags.models.into_iter().map(|m| m.name).collect())
}

pub fn chat(
    config: &LlmConfig,
    messages: &[ChatTurn],
    progress_tx: &Sender<ChatProgressEvent>,
    permissions: &SharedPathPermissions,
) -> Result<ChatResult, String> {
    let messages = trim_context_messages(messages);

    let mut api_messages = vec![json!({
        "role": "system",
        "content": system_prompt(),
    })];
    api_messages.extend(messages.iter().map(|turn| {
        json!({
            "role": turn.role,
            "content": turn.content,
        })
    }));

    let mut thinking = String::from("Thinking…");
    let mut metrics = OllamaMetrics::default();
    let mut trace = ChatTrace::default();

    for _ in 0..MAX_TOOL_ROUNDS {
        send_thinking(progress_tx, &thinking);

        let request = build_chat_request(config, &api_messages);
        let request_value =
            serde_json::to_value(&request).unwrap_or_else(|_| Value::String("<request>".into()));

        let response = stream_chat_request(config, &api_messages, &thinking, progress_tx)?;
        metrics.merge(&response.metrics);
        let assistant = response.message;

        trace.rounds.push(ChatRoundTrace {
            request: request_value,
            response: trace_response_value(&assistant, &response.metrics),
        });

        if !assistant.content.trim().is_empty() {
            thinking.push_str("\n\n");
            thinking.push_str(assistant.content.trim());
            send_thinking(progress_tx, &thinking);
        }

        if let Some(invocation) = parse_tool_request(&assistant.content, &assistant.tool_calls) {
            thinking.push_str("\n\n→ Calling ");
            thinking.push_str(&invocation.name);
            append_tool_preview(&mut thinking, &invocation);
            thinking.push('…');
            send_thinking(progress_tx, &thinking);
            send_tool_action(progress_tx, &invocation, None);

            let result = if invocation.name == "run_command" {
                request_command_approval(&invocation, progress_tx, &mut thinking)
            } else if is_file_permission_tool(&invocation.name) {
                execute_file_tool_with_permission(
                    &invocation,
                    config,
                    permissions,
                    progress_tx,
                    &mut thinking,
                )
            } else {
                execute_tool(&invocation, config)
            };

            send_tool_action(progress_tx, &invocation, Some(&result));

            let result = match result {
                Ok(output) => output,
                Err(message) => message,
            };

            thinking.push_str("\n\n← ");
            thinking.push_str(&summarize_tool_result(&invocation.name, &result));
            send_thinking(progress_tx, &thinking);

            if assistant.tool_calls.is_empty() {
                api_messages.push(json!({
                    "role": "assistant",
                    "content": assistant.content,
                }));
            } else {
                api_messages.push(json!({
                    "role": "assistant",
                    "content": assistant.content,
                    "tool_calls": assistant.tool_calls,
                }));
            }

            api_messages.push(json!({
                "role": "tool",
                "tool_name": invocation.name,
                "content": truncate(&result, MAX_TOOL_RESULT_CHARS),
            }));
            continue;
        }

        if assistant.content.trim().is_empty() {
            return Err("The model returned an empty response.".into());
        }

        return Ok(ChatResult {
            content: assistant.content,
            metrics,
            trace,
        });
    }

    Err("The model kept requesting tools without producing a final answer.".into())
}

pub fn execute_shell_command(command: &str) -> Result<String, String> {
    let command = command.trim();
    if command.is_empty() {
        return Err("Command is empty.".into());
    }

    let output = run_shell(command)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let mut result = format!("Exit code: {}\n", output.status.code().unwrap_or(-1));
    if !stdout.trim().is_empty() {
        result.push_str("\nstdout:\n");
        result.push_str(stdout.trim_end());
        result.push('\n');
    }
    if !stderr.trim().is_empty() {
        result.push_str("\nstderr:\n");
        result.push_str(stderr.trim_end());
        result.push('\n');
    }
    if stdout.trim().is_empty() && stderr.trim().is_empty() {
        result.push_str("\n(no output)\n");
    }

    Ok(truncate(&result, MAX_COMMAND_OUTPUT))
}

fn request_command_approval(
    invocation: &ToolInvocation,
    progress_tx: &Sender<ChatProgressEvent>,
    thinking: &mut String,
) -> Result<String, String> {
    let command = command_from_arguments(&invocation.arguments)?;

    thinking.push_str("\n\n→ Requesting approval to run command:\n");
    thinking.push_str(&command);
    thinking.push_str("\n\n⏸ Waiting for your approval…");
    send_thinking(progress_tx, thinking);

    let (response_tx, response_rx) = mpsc::channel();
    progress_tx
        .send(ChatProgressEvent::CommandApprovalNeeded {
            command,
            response_tx,
        })
        .map_err(|_| "Could not request command approval.".to_string())?;

    response_rx
        .recv()
        .map_err(|_| "Command approval channel closed.".to_string())?
}

fn execute_file_tool_with_permission(
    invocation: &ToolInvocation,
    config: &LlmConfig,
    permissions: &SharedPathPermissions,
    progress_tx: &Sender<ChatProgressEvent>,
    thinking: &mut String,
) -> Result<String, String> {
    let target = tool_target_path(invocation)?;
    let access = file_access_for_tool(&invocation.name)
        .ok_or_else(|| format!("Unknown file tool: {}", invocation.name))?;

    loop {
        let check = permissions
            .lock()
            .map_err(|_| "Could not read file permission state.".to_string())
            .map(|state| check_path_permission(&target, &state))?;

        match check {
            PermissionCheck::Allowed => return execute_tool(invocation, config),
            PermissionCheck::Denied => {
                let directory = permission_directory_for_target(&target);
                return Ok(format!(
                    "Access denied: file access was rejected for {}.",
                    directory.display()
                ));
            }
            PermissionCheck::NeedsPrompt => {
                let directory = permission_directory_for_target(&target);
                let choice = request_file_permission(
                    invocation,
                    &directory,
                    access,
                    progress_tx,
                    thinking,
                )?;

                let rule = choice.to_rule(&directory);
                permissions
                    .lock()
                    .map_err(|_| "Could not update file permission state.".to_string())?
                    .add_session_rule(rule);
            }
        }
    }
}

fn request_file_permission(
    invocation: &ToolInvocation,
    directory: &std::path::Path,
    access: FileAccess,
    progress_tx: &Sender<ChatProgressEvent>,
    thinking: &mut String,
) -> Result<FilePermissionChoice, String> {
    let arguments = format_tool_arguments(invocation);
    let directory_display = directory.display().to_string();

    thinking.push_str("\n\n→ Requesting permission to ");
    thinking.push_str(access.label());
    thinking.push_str(" files in ");
    thinking.push_str(&directory_display);
    thinking.push_str("\n\n⏸ Waiting for your decision…");
    send_thinking(progress_tx, thinking);

    let (response_tx, response_rx) = mpsc::channel();
    progress_tx
        .send(ChatProgressEvent::FilePermissionNeeded {
            tool_name: invocation.name.clone(),
            arguments,
            directory: directory_display,
            access,
            response_tx,
        })
        .map_err(|_| "Could not request file permission.".to_string())?;

    response_rx
        .recv()
        .map_err(|_| "File permission channel closed.".to_string())
}

fn tool_target_path(invocation: &ToolInvocation) -> Result<PathBuf, String> {
    let path = path_from_arguments(&invocation.arguments)?;
    Ok(permissions::normalize_path(&path))
}

fn command_from_arguments(arguments: &Value) -> Result<String, String> {
    let parsed = normalize_arguments(arguments);
    parsed
        .get("command")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|command| !command.is_empty())
        .map(str::to_string)
        .ok_or("run_command requires a non-empty \"command\" argument.".into())
}

#[cfg(unix)]
fn run_shell(command: &str) -> Result<std::process::Output, String> {
    Command::new("sh")
        .arg("-c")
        .arg(command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("Failed to run command: {err}"))
}

#[cfg(windows)]
fn run_shell(command: &str) -> Result<std::process::Output, String> {
    Command::new("cmd")
        .args(["/C", command])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("Failed to run command: {err}"))
}

fn send_thinking(progress_tx: &Sender<ChatProgressEvent>, content: &str) {
    let _ = progress_tx.send(ChatProgressEvent::Thinking(content.to_string()));
}

fn send_tool_action(
    progress_tx: &Sender<ChatProgressEvent>,
    invocation: &ToolInvocation,
    result: Option<&Result<String, String>>,
) {
    let arguments = format_tool_arguments(invocation);
    let update = match result {
        None => ToolActionUpdate {
            name: invocation.name.clone(),
            arguments,
            summary: "Running…".into(),
            success: true,
            completed: false,
        },
        Some(Ok(output)) => ToolActionUpdate {
            name: invocation.name.clone(),
            arguments,
            summary: tool_result_summary(&invocation.name, output),
            success: true,
            completed: true,
        },
        Some(Err(message)) => ToolActionUpdate {
            name: invocation.name.clone(),
            arguments,
            summary: message.clone(),
            success: false,
            completed: true,
        },
    };

    let _ = progress_tx.send(ChatProgressEvent::ToolAction(update));
}

fn format_tool_arguments(invocation: &ToolInvocation) -> String {
    let parsed = normalize_arguments(&invocation.arguments);

    match invocation.name.as_str() {
        "web_search" => parsed
            .get("query")
            .and_then(Value::as_str)
            .map(|query| format!("query: {query}"))
            .unwrap_or_else(|| "query: (missing)".into()),
        "ls" => format_ls_arguments(&parsed),
        "cat" => format_cat_arguments(&parsed),
        "sed" => format_sed_arguments(&parsed),
        "run_command" => command_from_arguments(&parsed)
            .map(|command| format!("command: {command}"))
            .unwrap_or_else(|err| err),
        _ => serde_json::to_string_pretty(&parsed).unwrap_or_else(|_| parsed.to_string()),
    }
}

fn tool_result_summary(tool_name: &str, result: &str) -> String {
    match tool_name {
        "ls" => {
            let lines = result.lines().filter(|line| !line.trim().is_empty()).count();
            if result.contains("… truncated") {
                format!("{lines} lines of output (truncated)")
            } else {
                format!("{lines} lines of output")
            }
        }
        "cat" => {
            let content = result
                .split_once("\n\n")
                .map(|(_, body)| body)
                .unwrap_or(result);
            let lines = content.lines().count();
            if result.contains("… truncated") {
                format!("{lines} lines read (truncated)")
            } else {
                let chars = content.chars().count();
                format!("{lines} lines, {chars} characters read")
            }
        }
        "sed" => result.to_string(),
        "web_search" => {
            if result.contains("No results found.") {
                return "No results found".into();
            }

            let count = result
                .lines()
                .filter(|line| {
                    line.chars()
                        .next()
                        .is_some_and(|ch| ch.is_ascii_digit())
                })
                .count();
            if count == 0 {
                "Search completed".into()
            } else {
                format!("{count} sources")
            }
        }
        "run_command" => result
            .lines()
            .next()
            .map(|line| format!("Completed ({line})"))
            .unwrap_or_else(|| "Completed".into()),
        _ => truncate(result, 120),
    }
}

fn append_tool_preview(thinking: &mut String, invocation: &ToolInvocation) {
    match invocation.name.as_str() {
        "web_search" => {
            if let Some(query) = invocation.arguments.get("query").and_then(Value::as_str) {
                thinking.push_str(&format!(" ({query})"));
            }
        }
        "ls" => {
            if let Ok(path) = path_from_arguments(&invocation.arguments) {
                let flags = invocation
                    .arguments
                    .get("flags")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                thinking.push_str(&format!(" (ls {flags} {})", path.display()));
            }
        }
        "cat" => {
            if let Ok(path) = path_from_arguments(&invocation.arguments) {
                let flags = invocation
                    .arguments
                    .get("flags")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if let Some(pattern) = grep_pattern_from_arguments(&invocation.arguments) {
                    if flags.contains('n') {
                        thinking.push_str(&format!(" (grep -n '{pattern}' {})", path.display()));
                    } else {
                        thinking.push_str(&format!(" (grep '{pattern}' {})", path.display()));
                    }
                } else if flags.contains('n') {
                    thinking.push_str(&format!(" (cat -n {})", path.display()));
                } else {
                    thinking.push_str(&format!(" (cat {})", path.display()));
                }
            }
        }
        "sed" => {
            if let (Ok(path), Ok(expression)) = (
                path_from_arguments(&invocation.arguments),
                sed_expression_from_arguments(&invocation.arguments),
            ) {
                thinking.push_str(&format!(
                    " (sed -i '{expression}' {})",
                    path.display()
                ));
            }
        }
        "run_command" => {
            if let Ok(command) = command_from_arguments(&invocation.arguments) {
                thinking.push_str(&format!("\n  `$ {command}`"));
            }
        }
        _ => {}
    }
}

fn summarize_tool_result(tool_name: &str, result: &str) -> String {
    match tool_name {
        "cat" | "run_command" | "web_search" => truncate(result, 600),
        "ls" | "sed" => result.to_string(),
        _ => truncate(result, 600),
    }
}

fn truncate(text: &str, max_len: usize) -> String {
    if text.chars().count() <= max_len {
        return text.to_string();
    }

    let end = text
        .char_indices()
        .nth(max_len)
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    format!("{}…", &text[..end])
}

fn trim_context_messages(messages: &[ChatTurn]) -> Vec<ChatTurn> {
    let start = messages.len().saturating_sub(MAX_CONTEXT_MESSAGES);
    messages[start..]
        .iter()
        .map(|turn| ChatTurn {
            role: turn.role.clone(),
            content: truncate(&turn.content, MAX_CONTEXT_MESSAGE_CHARS),
        })
        .collect()
}

fn build_chat_request(config: &LlmConfig, messages: &[Value]) -> ChatRequest {
    ChatRequest {
        model: config.model.clone(),
        messages: messages.to_vec(),
        stream: true,
        tools: tool_definitions(),
        keep_alive: KEEP_ALIVE.into(),
        truncate: true,
        options: ModelOptions {
            num_ctx: config.num_ctx,
        },
    }
}

fn trace_response_value(message: &ChatMessage, metrics: &OllamaMetrics) -> Value {
    json!({
        "message": {
            "role": "assistant",
            "content": message.content,
            "tool_calls": message.tool_calls,
        },
        "usage": {
            "total_duration_ns": metrics.total_duration_ns,
            "load_duration_ns": metrics.load_duration_ns,
            "prompt_eval_count": metrics.prompt_eval_count,
            "prompt_eval_duration_ns": metrics.prompt_eval_duration_ns,
            "eval_count": metrics.eval_count,
            "eval_duration_ns": metrics.eval_duration_ns,
        },
        "http_headers": metrics.http_headers,
    })
}

fn stream_chat_request(
    config: &LlmConfig,
    messages: &[Value],
    thinking_prefix: &str,
    progress_tx: &Sender<ChatProgressEvent>,
) -> Result<StreamChatResult, String> {
    let url = format!("{}/api/chat", normalize_base_url(&config.base_url));
    let request = build_chat_request(config, messages);

    let response = http_client()
        .post(url)
        .json(&request)
        .send()
        .map_err(connection_error)?;

    if !response.status().is_success() {
        return Err(format!("Ollama returned HTTP {}", response.status()));
    }

    let http_headers = response
        .headers()
        .iter()
        .map(|(name, value)| {
            (
                name.to_string(),
                value.to_str().unwrap_or("<binary>").to_string(),
            )
        })
        .collect();

    let mut content = String::new();
    let mut tool_calls = Vec::new();
    let mut metrics = OllamaMetrics {
        http_headers,
        ..OllamaMetrics::default()
    };
    let reader = BufReader::new(response);

    for line in reader.lines() {
        let line = line.map_err(|err| format!("Could not read Ollama stream: {err}"))?;
        if line.trim().is_empty() {
            continue;
        }

        let chunk: StreamChunk =
            serde_json::from_str(&line).map_err(|err| format!("Invalid Ollama stream chunk: {err}"))?;

        if !chunk.message.content.is_empty() {
            content.push_str(&chunk.message.content);
            send_thinking(progress_tx, &format_streaming_thinking(thinking_prefix, &content));
        }

        if !chunk.message.tool_calls.is_empty() {
            tool_calls = chunk.message.tool_calls;
        }

        if chunk.done {
            if content.is_empty() && !chunk.message.content.is_empty() {
                content = chunk.message.content;
            }

            metrics.total_duration_ns += chunk.total_duration;
            metrics.load_duration_ns += chunk.load_duration;
            metrics.prompt_eval_count += chunk.prompt_eval_count;
            metrics.prompt_eval_duration_ns += chunk.prompt_eval_duration;
            metrics.eval_count += chunk.eval_count;
            metrics.eval_duration_ns += chunk.eval_duration;
            metrics.request_count += 1;
        }
    }

    Ok(StreamChatResult {
        message: ChatMessage { content, tool_calls },
        metrics,
    })
}

fn format_streaming_thinking(prefix: &str, streamed: &str) -> String {
    if prefix.trim().is_empty() {
        streamed.to_string()
    } else {
        format!("{prefix}\n\n{streamed}")
    }
}

fn http_client() -> &'static reqwest::blocking::Client {
    static CLIENT: OnceLock<reqwest::blocking::Client> = OnceLock::new();
    CLIENT.get_or_init(reqwest::blocking::Client::new)
}

fn tool_definition(spec: &ToolSpec) -> Value {
    match spec.name {
        "web_search" => json!({
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information using Tavily",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }),
        "ls" => json!({
            "type": "function",
            "function": {
                "name": "ls",
                "description": "List files in a directory using ls. Optional flags: a (all), l (long), R (recursive).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list"
                        },
                        "flags": {
                            "type": "string",
                            "description": "Optional ls flags as a string using a, l, and/or R (e.g. \"la\" or \"-laR\")"
                        }
                    },
                    "required": ["path"]
                }
            }
        }),
        "cat" => json!({
            "type": "function",
            "function": {
                "name": "cat",
                "description": "Print a file's contents using cat, like cat /path/to/file. Optional flag n adds line numbers (cat -n). When looking for something specific in a file, pass a grep-style pattern and use flags \"n\" so you know which lines matched.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to print"
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Optional grep-style regex; when set, only matching lines are returned"
                        },
                        "flags": {
                            "type": "string",
                            "description": "Optional cat flags as a string (e.g. \"n\" or \"-n\" for line numbers). Use with pattern when searching for something specific."
                        }
                    },
                    "required": ["path"]
                }
            }
        }),
        "sed" => json!({
            "type": "function",
            "function": {
                "name": "sed",
                "description": "Search-and-replace in a file using sed. In the replacement portion of the expression, escape special characters: backslash, &, the delimiter, and backreferences (e.g. \\\\&, \\/, \\\\1).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to modify"
                        },
                        "expression": {
                            "type": "string",
                            "description": "Full sed substitution expression (e.g. s/old/new/ or s/old/new/g). Escape \\, &, /, and backreferences in the replacement."
                        }
                    },
                    "required": ["path", "expression"]
                }
            }
        }),
        "run_command" => json!({
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command on the user's machine. Requires user approval before execution.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to run"
                        }
                    },
                    "required": ["command"]
                }
            }
        }),
        _ => json!({}),
    }
}

fn tool_definitions() -> Vec<Value> {
    available_tool_specs()
        .map(tool_definition)
        .collect()
}

fn execute_tool(invocation: &ToolInvocation, config: &LlmConfig) -> Result<String, String> {
    match invocation.name.as_str() {
        "web_search" => web_search(config, &invocation.arguments),
        "ls" => ls(&invocation.arguments),
        "cat" => cat(&invocation.arguments),
        "sed" => sed(&invocation.arguments),
        other => Err(format!("Unknown tool: {other}")),
    }
}

fn path_from_arguments(arguments: &Value) -> Result<PathBuf, String> {
    let parsed = normalize_arguments(arguments);
    let path = parsed
        .get("path")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .ok_or("A non-empty \"path\" argument is required.")?;
    expand_path(path)
}

fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| std::env::var("USERPROFILE").ok().map(PathBuf::from))
}

fn expand_path(path: &str) -> Result<PathBuf, String> {
    if path == "~" {
        return home_dir().ok_or_else(|| "Could not resolve home directory.".to_string());
    }

    if let Some(rest) = path.strip_prefix("~/") {
        let home = home_dir().ok_or_else(|| "Could not resolve home directory.".to_string())?;
        return Ok(home.join(rest));
    }

    Ok(PathBuf::from(path))
}

fn missing_path_error(kind: &str, path: &Path) -> String {
    format!(
        "{kind} does not exist: {}. Verify the path and try again — use ls on the parent directory if you need to find the correct name.",
        path.display()
    )
}

fn wrong_path_type_error(expected: &str, path: &Path) -> String {
    format!(
        "Path is not a {expected}: {}. Check the path and try again.",
        path.display()
    )
}

fn require_file_exists(path: &PathBuf) -> Result<(), String> {
    if !path.exists() {
        return Err(missing_path_error("File", path));
    }
    if !path.is_file() {
        return Err(wrong_path_type_error("file", path));
    }
    Ok(())
}

fn require_directory_exists(path: &PathBuf) -> Result<(), String> {
    if !path.exists() {
        return Err(missing_path_error("Directory", path));
    }
    if !path.is_dir() {
        return Err(wrong_path_type_error("directory", path));
    }
    Ok(())
}

fn sed_expression_from_arguments(arguments: &Value) -> Result<String, String> {
    let parsed = normalize_arguments(arguments);
    parsed
        .get("expression")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|expression| !expression.is_empty())
        .ok_or("sed requires \"expression\" (e.g. s/old/new/).".into())
        .map(str::to_string)
}

fn validate_sed_expression(expression: &str) -> Result<(), String> {
    if expression.contains('\n') {
        return Err("sed expression must be a single line.".into());
    }
    if expression.starts_with('-') {
        return Err("sed expression must not start with '-'.".into());
    }
    Ok(())
}

fn format_sed_command(path: &PathBuf, expression: &str) -> String {
    format!("sed -i '{expression}' {}", path.display())
}

fn format_sed_arguments(arguments: &Value) -> String {
    match (
        path_from_arguments(arguments),
        sed_expression_from_arguments(arguments),
    ) {
        (Ok(path), Ok(expression)) => {
            format!(
                "command: {}\nexpression: {expression}\npath: {}",
                format_sed_command(&path, &expression),
                path.display()
            )
        }
        (Err(err), _) | (_, Err(err)) => err,
    }
}

fn parse_ls_flags(flags: Option<&str>) -> Result<String, String> {
    let Some(raw) = flags.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(String::new());
    };

    let raw = raw.trim_start_matches('-');
    let mut has_a = false;
    let mut has_l = false;
    let mut has_r = false;

    for ch in raw.chars() {
        match ch {
            'a' => has_a = true,
            'l' => has_l = true,
            'R' => has_r = true,
            _ => {
                return Err(format!(
                    "Unsupported ls flag: {ch}. Only a, l, and R are supported."
                ));
            }
        }
    }

    let mut flag_chars = String::new();
    if has_a {
        flag_chars.push('a');
    }
    if has_l {
        flag_chars.push('l');
    }
    if has_r {
        flag_chars.push('R');
    }

    Ok(flag_chars)
}

fn format_ls_command(path: &PathBuf, flags: &str) -> String {
    if flags.is_empty() {
        format!("ls {}", path.display())
    } else {
        format!("ls -{flags} {}", path.display())
    }
}

fn format_ls_arguments(arguments: &Value) -> String {
    let parsed = normalize_arguments(arguments);
    match (
        path_from_arguments(&parsed),
        parse_ls_flags(parsed.get("flags").and_then(Value::as_str)),
    ) {
        (Ok(path), Ok(flags)) => {
            let mut lines = vec![format!(
                "command: {}",
                format_ls_command(&path, &flags)
            )];
            lines.push(format!("path: {}", path.display()));
            if flags.is_empty() {
                lines.push("flags: (none)".into());
            } else {
                lines.push(format!("flags: {flags}"));
            }
            lines.join("\n")
        }
        (Err(err), _) | (_, Err(err)) => err,
    }
}

fn ls(arguments: &Value) -> Result<String, String> {
    #[cfg(not(unix))]
    {
        let _ = arguments;
        return Err("The ls tool is only available on Linux and macOS.".into());
    }

    #[cfg(unix)]
    {
        let parsed = normalize_arguments(arguments);
        let path = path_from_arguments(&parsed)?;
        let flags = parse_ls_flags(parsed.get("flags").and_then(Value::as_str))?;
        require_directory_exists(&path)?;

        let command_label = format_ls_command(&path, &flags);
        let mut command = Command::new("ls");
        if !flags.is_empty() {
            command.arg(format!("-{flags}"));
        }
        command.arg(&path);

        let output = command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|err| format!("Could not run ls: {err}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            if stderr.is_empty() {
                return Err(format!("ls failed for {command_label}"));
            }
            return Err(format!("ls failed for {command_label}: {stderr}"));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.trim().is_empty() {
            return Ok(format!("{command_label}\n\n(empty directory)"));
        }

        Ok(truncate(
            &format!("{command_label}\n\n{}", stdout.trim_end()),
            MAX_COMMAND_OUTPUT,
        ))
    }
}

fn sed(arguments: &Value) -> Result<String, String> {
    #[cfg(not(unix))]
    {
        let _ = arguments;
        return Err("The sed tool is only available on Linux and macOS.".into());
    }

    #[cfg(unix)]
    {
        let parsed = normalize_arguments(arguments);
        let path = path_from_arguments(&parsed)?;
        let expression = sed_expression_from_arguments(&parsed)?;
        validate_sed_expression(&expression)?;
        require_file_exists(&path)?;

        let command_label = format_sed_command(&path, &expression);
        let path_arg = path.to_string_lossy().into_owned();
        let output = if cfg!(target_os = "macos") {
            Command::new("sed")
                .args(["-i", "", &expression, &path_arg])
                .output()
        } else {
            Command::new("sed")
                .args(["-i", &expression, &path_arg])
                .output()
        }
        .map_err(|err| format!("Could not run sed: {err}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            if stderr.is_empty() {
                return Err(format!("sed failed for {command_label}"));
            }
            return Err(format!("sed failed for {command_label}: {stderr}"));
        }

        Ok(format!("Applied {command_label} successfully."))
    }
}

fn parse_cat_flags(flags: Option<&str>) -> Result<String, String> {
    let Some(raw) = flags.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(String::new());
    };

    let raw = raw.trim_start_matches('-');
    let mut has_n = false;

    for ch in raw.chars() {
        match ch {
            'n' => has_n = true,
            _ => {
                return Err(format!(
                    "Unsupported cat flag: {ch}. Only n is supported."
                ));
            }
        }
    }

    if has_n {
        Ok("n".into())
    } else {
        Ok(String::new())
    }
}

fn grep_pattern_from_arguments(arguments: &Value) -> Option<String> {
    let parsed = normalize_arguments(arguments);
    parsed
        .get("pattern")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|pattern| !pattern.is_empty())
        .map(str::to_string)
}

fn format_cat_command(path: &PathBuf, pattern: Option<&str>, flags: &str) -> String {
    let line_numbers = flags.contains('n');
    match pattern {
        Some(pattern) if line_numbers => format!("grep -n '{}' {}", pattern, path.display()),
        Some(pattern) => format!("grep '{}' {}", pattern, path.display()),
        None if line_numbers => format!("cat -n {}", path.display()),
        None => format!("cat {}", path.display()),
    }
}

fn format_cat_arguments(arguments: &Value) -> String {
    let parsed = normalize_arguments(arguments);
    match (
        path_from_arguments(&parsed),
        parse_cat_flags(parsed.get("flags").and_then(Value::as_str)),
    ) {
        (Ok(path), Ok(flags)) => {
            let pattern = grep_pattern_from_arguments(&parsed);
            let mut lines = vec![format!(
                "command: {}",
                format_cat_command(&path, pattern.as_deref(), &flags)
            )];
            lines.push(format!("path: {}", path.display()));
            if let Some(pattern) = pattern {
                lines.push(format!("pattern: {pattern}"));
            }
            if flags.is_empty() {
                lines.push("flags: (none)".into());
            } else {
                lines.push(format!("flags: {flags}"));
            }
            lines.join("\n")
        }
        (Err(err), _) | (_, Err(err)) => err,
    }
}

fn cat(arguments: &Value) -> Result<String, String> {
    #[cfg(not(unix))]
    {
        let _ = arguments;
        return Err("The cat tool is only available on Linux and macOS.".into());
    }

    #[cfg(unix)]
    {
        let parsed = normalize_arguments(arguments);
        let path = path_from_arguments(&parsed)?;
        require_file_exists(&path)?;

        let pattern = grep_pattern_from_arguments(&parsed);
        let flags = parse_cat_flags(parsed.get("flags").and_then(Value::as_str))?;
        let line_numbers = flags.contains('n');
        let command_label = format_cat_command(&path, pattern.as_deref(), &flags);
        let output = if let Some(pattern) = &pattern {
            let mut command = Command::new("grep");
            if line_numbers {
                command.arg("-n");
            }
            command
                .arg(pattern)
                .arg(&path)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .map_err(|err| format!("Could not run grep: {err}"))?
        } else {
            let mut command = Command::new("cat");
            if line_numbers {
                command.arg("-n");
            }
            command
                .arg(&path)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .map_err(|err| format!("Could not run cat: {err}"))?
        };

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            if pattern.is_some() && output.status.code() == Some(1) && stderr.is_empty() {
                return Ok(format!("{command_label}\n\n(no matching lines)"));
            }
            if stderr.is_empty() {
                return Err(format!("{command_label} failed"));
            }
            return Err(format!("{command_label} failed: {stderr}"));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.is_empty() {
            let empty_message = if pattern.is_some() {
                "(no matching lines)"
            } else {
                "(empty file)"
            };
            return Ok(format!("{command_label}\n\n{empty_message}"));
        }

        Ok(truncate(
            &format!("{command_label}\n\n{}", stdout.trim_end()),
            MAX_COMMAND_OUTPUT,
        ))
    }
}

fn web_search(config: &LlmConfig, arguments: &Value) -> Result<String, String> {
    let parsed = normalize_arguments(arguments);
    let query = parsed
        .get("query")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|query| !query.is_empty())
        .ok_or("web_search requires a non-empty \"query\" argument.")?;

    let request = TavilySearchRequest {
        query,
        max_results: 5,
        include_answer: true,
    };

    let api_key = config.tavily_api_key.trim();
    let mut http = reqwest::blocking::Client::new().post(TAVILY_SEARCH_URL);

    if api_key.is_empty() {
        http = http.header("X-Tavily-Access-Mode", "keyless");
    } else {
        http = http.header("Authorization", format!("Bearer {api_key}"));
    }

    let response = http
        .json(&request)
        .send()
        .map_err(|err| format!("Tavily request failed: {err}"))?;

    if !response.status().is_success() {
        return Err(format!("Tavily returned HTTP {}", response.status()));
    }

    let body: TavilySearchResponse = response
        .json()
        .map_err(|err| format!("Could not parse Tavily response: {err}"))?;

    Ok(format_search_results(query, &body))
}

fn format_search_results(query: &str, response: &TavilySearchResponse) -> String {
    let mut output = format!("Search results for \"{query}\":\n");

    if let Some(answer) = &response.answer {
        if !answer.trim().is_empty() {
            output.push_str("\nSummary:\n");
            output.push_str(answer.trim());
            output.push('\n');
        }
    }

    if response.results.is_empty() {
        output.push_str("\nNo results found.");
        return output;
    }

    output.push_str("\nSources:\n");
    for (index, result) in response.results.iter().enumerate() {
        output.push_str(&format!(
            "{}. {} ({:.2})\n   {}\n   {}\n",
            index + 1,
            result.title,
            result.score,
            result.url,
            result.content.trim()
        ));
    }

    output
}

fn parse_tool_request(content: &str, tool_calls: &[ToolCall]) -> Option<ToolInvocation> {
    for call in tool_calls {
        if is_supported_tool(&call.function.name) {
            return Some(ToolInvocation {
                name: call.function.name.clone(),
                arguments: normalize_arguments(&call.function.arguments),
            });
        }
    }

    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }

    for candidate in json_candidates(trimmed) {
        if let Ok(value) = serde_json::from_str::<Value>(&candidate) {
            if let Some(invocation) = tool_from_json(&value) {
                return Some(invocation);
            }
        }
    }

    None
}

fn tool_from_json(value: &Value) -> Option<ToolInvocation> {
    let name = value
        .get("tool")
        .or_else(|| value.get("name"))
        .and_then(Value::as_str)?;

    if !is_supported_tool(name) {
        return None;
    }

    let arguments = match name {
        "web_search" => {
            let query = value
                .get("query")
                .or_else(|| value.pointer("/arguments/query"))
                .cloned()
                .unwrap_or(Value::Null);
            json!({ "query": query })
        }
        "run_command" => {
            let command = value
                .get("command")
                .or_else(|| value.pointer("/arguments/command"))
                .cloned()
                .unwrap_or(Value::Null);
            json!({ "command": command })
        }
        "cat" => {
            json!({
                "path": value.get("path").or_else(|| value.pointer("/arguments/path")).cloned().unwrap_or(Value::Null),
                "pattern": value.get("pattern").or_else(|| value.pointer("/arguments/pattern")).cloned().unwrap_or(Value::Null),
                "flags": value.get("flags").or_else(|| value.pointer("/arguments/flags")).cloned().unwrap_or(Value::Null),
            })
        }
        "ls" => {
            json!({
                "path": value.get("path").or_else(|| value.pointer("/arguments/path")).cloned().unwrap_or(Value::Null),
                "flags": value.get("flags").or_else(|| value.pointer("/arguments/flags")).cloned().unwrap_or(Value::Null),
            })
        }
        "sed" => {
            json!({
                "path": value.get("path").or_else(|| value.pointer("/arguments/path")).cloned().unwrap_or(Value::Null),
                "expression": value.get("expression").or_else(|| value.pointer("/arguments/expression")).cloned().unwrap_or(Value::Null),
            })
        }
        _ => json!({}),
    };

    Some(ToolInvocation {
        name: name.to_string(),
        arguments,
    })
}

fn normalize_arguments(arguments: &Value) -> Value {
    if let Some(raw) = arguments.as_str() {
        serde_json::from_str(raw).unwrap_or_else(|_| arguments.clone())
    } else {
        arguments.clone()
    }
}

fn is_supported_tool(name: &str) -> bool {
    TOOL_SPECS
        .iter()
        .any(|spec| spec.name == name && spec.platform.is_supported())
}

fn json_candidates(text: &str) -> Vec<String> {
    let mut candidates = vec![text.to_string()];

    if let (Some(start), Some(end)) = (text.find('{'), text.rfind('}')) {
        if start < end {
            candidates.push(text[start..=end].to_string());
        }
    }

    if text.starts_with("```") {
        let stripped = text
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
        candidates.push(stripped.to_string());
    }

    candidates
}

fn normalize_base_url(base_url: &str) -> String {
    base_url.trim().trim_end_matches('/').to_string()
}

fn connection_error(err: reqwest::Error) -> String {
    if err.is_connect() {
        "Could not connect to Ollama. Install it from https://ollama.com and run `ollama serve`.".into()
    } else {
        err.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn ls_availability_matches_platform() {
        assert_eq!(is_supported_tool("ls"), cfg!(unix));
    }

    #[test]
    fn cat_availability_matches_platform() {
        assert_eq!(is_supported_tool("cat"), cfg!(unix));
    }

    #[test]
    fn parses_ls_flags() {
        assert_eq!(parse_ls_flags(Some("la")).expect("flags"), "al");
        assert_eq!(parse_ls_flags(Some("-aR")).expect("flags"), "aR");
        assert_eq!(parse_ls_flags(None).expect("flags"), "");
        assert!(parse_ls_flags(Some("x")).is_err());
    }

    #[test]
    fn system_prompt_includes_environment_context() {
        let prompt = system_prompt();
        assert!(prompt.contains("Operating system:"));
        assert!(prompt.contains(operating_system_name()));
        if let Some(home) = home_dir() {
            assert!(prompt.contains(&home.display().to_string()));
        }
        assert!(prompt.contains("Home directory:"));
    }

    #[test]
    fn system_prompt_lists_only_available_tools() {
        let prompt = tool_system_prompt_body();
        if cfg!(unix) {
            assert!(prompt.contains("\"tool\":\"ls\""));
            assert!(prompt.contains("\"tool\":\"cat\""));
            assert!(prompt.contains("sed"));
        } else {
            assert!(!prompt.contains("\"tool\":\"ls\""));
            assert!(!prompt.contains("\"tool\":\"cat\""));
            assert!(!prompt.contains("\"tool\":\"sed\""));
        }
    }

    #[test]
    fn sed_availability_matches_platform() {
        assert_eq!(is_supported_tool("sed"), cfg!(unix));
    }

    #[test]
    fn summarizes_ls_result() {
        let output = "ls -la /tmp\n\nfile1\nfile2\n";
        assert_eq!(tool_result_summary("ls", output), "3 lines of output");
    }

    #[test]
    fn formats_metrics_summary() {
        let metrics = OllamaMetrics {
            total_duration_ns: 2_500_000_000,
            eval_count: 100,
            eval_duration_ns: 2_000_000_000,
            request_count: 2,
            ..OllamaMetrics::default()
        };

        assert_eq!(metrics.summary_line(), "2.50s · 50.0 tok/s");
        assert!(metrics.tooltip_text().contains("prompt_eval_count"));
    }

    #[test]
    fn trims_old_context_messages() {
        let messages: Vec<ChatTurn> = (0..30)
            .map(|index| ChatTurn {
                role: if index % 2 == 0 {
                    "user".into()
                } else {
                    "assistant".into()
                },
                content: format!("message {index}"),
            })
            .collect();

        let trimmed = trim_context_messages(&messages);
        assert_eq!(trimmed.len(), MAX_CONTEXT_MESSAGES);
        assert_eq!(trimmed[0].content, "message 10");
        assert_eq!(trimmed.last().unwrap().content, "message 29");
    }

    #[test]
    fn parses_json_ls_request() {
        let request = parse_tool_request(r#"{"tool":"ls","path":".","flags":"la"}"#, &[]);
        let invocation = request.expect("tool request");
        assert_eq!(invocation.name, "ls");
        assert_eq!(
            invocation.arguments.get("path").and_then(Value::as_str),
            Some(".")
        );
        assert_eq!(
            invocation.arguments.get("flags").and_then(Value::as_str),
            Some("la")
        );
    }

    #[test]
    #[cfg(unix)]
    fn ls_lists_directory() {
        let dir = std::env::temp_dir().join(format!("pairllm-ls-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir");
        fs::write(dir.join("alpha.txt"), "a").expect("write file");

        let result = ls(&json!({ "path": dir })).expect("ls");
        assert!(result.contains("alpha.txt"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn summarizes_cat_result() {
        assert_eq!(
            tool_result_summary("cat", "cat /tmp/file\n\none\ntwo\nthree"),
            "3 lines, 13 characters read"
        );
    }

    #[test]
    fn formats_tool_arguments_for_cat() {
        let invocation = ToolInvocation {
            name: "cat".into(),
            arguments: json!({ "path": "~/notes.txt" }),
        };
        let args = format_tool_arguments(&invocation);
        assert!(args.starts_with("command: cat "));
        assert!(args.contains("notes.txt"));
    }

    #[test]
    fn parses_json_cat_request() {
        let request = parse_tool_request(r#"{"tool":"cat","path":"README.md"}"#, &[]);
        let invocation = request.expect("tool request");
        assert_eq!(invocation.name, "cat");
        assert_eq!(
            invocation.arguments.get("path").and_then(Value::as_str),
            Some("README.md")
        );
    }

    #[test]
    fn parses_json_sed_request() {
        let request = parse_tool_request(
            r#"{"tool":"sed","path":"src/main.rs","expression":"s/old/new/"}"#,
            &[],
        );
        let invocation = request.expect("tool request");
        assert_eq!(invocation.name, "sed");
        assert_eq!(
            invocation.arguments.get("path").and_then(Value::as_str),
            Some("src/main.rs")
        );
        assert_eq!(
            invocation.arguments.get("expression").and_then(Value::as_str),
            Some("s/old/new/")
        );
    }

    #[test]
    fn cat_errors_when_file_missing() {
        let result = cat(&json!({ "path": "/tmp/pairllm-definitely-missing-file" }));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("File does not exist"));
        assert!(err.contains("try again"));
    }

    #[test]
    #[cfg(unix)]
    fn cat_reads_file() {
        let dir = std::env::temp_dir().join(format!("pairllm-cat-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("sample.txt");
        fs::write(&path, "hello world\n").expect("write sample");

        let result = cat(&json!({ "path": path })).expect("cat");
        assert!(result.starts_with("cat "));
        assert!(result.contains("hello world"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(unix)]
    fn cat_filters_lines_with_pattern() {
        let dir = std::env::temp_dir().join(format!("pairllm-cat-grep-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("sample.txt");
        fs::write(
            &path,
            "alpha\nbeta line\nalpha again\ngamma\n",
        )
        .expect("write sample");

        let result = cat(&json!({ "path": path, "pattern": "alpha" })).expect("cat with pattern");
        assert!(result.starts_with("grep 'alpha' "));
        assert!(result.contains("alpha"));
        assert!(result.contains("alpha again"));
        assert!(!result.contains("beta line"));
        assert!(!result.contains("gamma"));

        let no_match = cat(&json!({ "path": path, "pattern": "missing" })).expect("cat with pattern");
        assert!(no_match.contains("(no matching lines)"));

        let numbered = cat(&json!({ "path": path, "pattern": "alpha", "flags": "n" }))
            .expect("cat with pattern and line numbers");
        assert!(numbered.starts_with("grep -n 'alpha' "));
        assert!(numbered.contains("1:alpha"));
        assert!(numbered.contains("3:alpha again"));

        let numbered_full = cat(&json!({ "path": path, "flags": "n" })).expect("cat -n");
        assert!(numbered_full.starts_with("cat -n "));
        assert!(numbered_full.contains("1\talpha"));
        assert!(numbered_full.contains("2\tbeta line"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parses_cat_flags() {
        assert_eq!(parse_cat_flags(Some("n")).expect("flags"), "n");
        assert_eq!(parse_cat_flags(Some("-n")).expect("flags"), "n");
        assert_eq!(parse_cat_flags(None).expect("flags"), "");
        assert!(parse_cat_flags(Some("x")).is_err());
    }

    #[test]
    fn parses_json_cat_request_with_pattern() {
        let request = parse_tool_request(
            r#"{"tool":"cat","path":"README.md","pattern":"fn main","flags":"n"}"#,
            &[],
        );
        let invocation = request.expect("tool request");
        assert_eq!(invocation.name, "cat");
        assert_eq!(
            invocation.arguments.get("path").and_then(Value::as_str),
            Some("README.md")
        );
        assert_eq!(
            invocation.arguments.get("pattern").and_then(Value::as_str),
            Some("fn main")
        );
        assert_eq!(
            invocation.arguments.get("flags").and_then(Value::as_str),
            Some("n")
        );
    }

    #[test]
    #[cfg(unix)]
    fn sed_replaces_text_in_file() {
        let dir = std::env::temp_dir().join(format!("pairllm-sed-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("sample.txt");
        fs::write(&path, "hello old world\n").expect("write sample");

        let result = sed(&json!({
            "path": path,
            "expression": "s/old/new/"
        }))
        .expect("sed");

        assert!(result.contains("sed -i 's/old/new/'"));
        assert_eq!(fs::read_to_string(&path).expect("read"), "hello new world\n");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(unix)]
    fn sed_errors_when_file_missing() {
        let result = sed(&json!({
            "path": "/tmp/pairllm-definitely-missing-file",
            "expression": "s/old/new/"
        }));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("File does not exist"));
        assert!(err.contains("try again"));
    }

    #[test]
    fn expands_tilde_in_path() {
        let home = home_dir().expect("home directory");
        assert_eq!(
            expand_path("~/target.txt").expect("expand"),
            home.join("target.txt")
        );
        assert_eq!(expand_path("~").expect("expand"), home);
        assert_eq!(
            expand_path("/etc/hosts").expect("expand"),
            PathBuf::from("/etc/hosts")
        );
    }

    #[test]
    fn parses_json_web_search_request() {
        let request = parse_tool_request(r#"{"tool":"web_search","query":"rust egui"}"#, &[]);
        let invocation = request.expect("tool request");
        assert_eq!(invocation.name, "web_search");
        assert_eq!(
            invocation.arguments.get("query").and_then(Value::as_str),
            Some("rust egui")
        );
    }

    #[test]
    fn parses_json_run_command_request() {
        let request = parse_tool_request(r#"{"tool":"run_command","command":"ls -la"}"#, &[]);
        let invocation = request.expect("tool request");
        assert_eq!(invocation.name, "run_command");
        assert_eq!(
            invocation.arguments.get("command").and_then(Value::as_str),
            Some("ls -la")
        );
    }

    #[test]
    fn parses_native_web_search_call() {
        let calls = vec![ToolCall {
            function: ToolFunction {
                name: "web_search".into(),
                arguments: json!({"query": "latest ai news"}),
            },
        }];
        let request = parse_tool_request("", &calls).expect("tool request");
        assert_eq!(request.name, "web_search");
        assert_eq!(
            request.arguments.get("query").and_then(Value::as_str),
            Some("latest ai news")
        );
    }
}
