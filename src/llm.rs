use chrono::Local;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::{mpsc::{self, Sender}, OnceLock};

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:11434";
const DEFAULT_MODEL: &str = "llama3.2";
const DEFAULT_NUM_CTX: u32 = 16384;
const MAX_TOOL_ROUNDS: usize = 8;
const TAVILY_SEARCH_URL: &str = "https://api.tavily.com/search";
const MAX_COMMAND_OUTPUT: usize = 8000;
const MAX_TOOL_RESULT_CHARS: usize = 4000;
const MAX_CONTEXT_MESSAGES: usize = 20;
const MAX_CONTEXT_MESSAGE_CHARS: usize = 3000;
const KEEP_ALIVE: &str = "30m";

const SYSTEM_PROMPT: &str = "\
You are a helpful assistant in a chat app.

You have three tools:
- get_time: returns the current local date and time.
- web_search: searches the web for up-to-date information.
- run_command: runs a shell command on the user's machine. The user must approve \
before it runs.

If you need a tool, respond with ONLY a JSON object and nothing else:
{\"tool\":\"get_time\"}
{\"tool\":\"web_search\",\"query\":\"your search query\"}
{\"tool\":\"run_command\",\"command\":\"the shell command\"}

After you receive a tool result, answer the user in plain language.";

#[derive(Clone)]
pub struct LlmConfig {
    pub enabled: bool,
    pub base_url: String,
    pub model: String,
    pub num_ctx: u32,
    pub tavily_api_key: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: DEFAULT_BASE_URL.into(),
            model: DEFAULT_MODEL.into(),
            num_ctx: DEFAULT_NUM_CTX,
            tavily_api_key: std::env::var("TAVILY_API_KEY").unwrap_or_default(),
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
    CommandApprovalNeeded {
        command: String,
        response_tx: Sender<Result<String, String>>,
    },
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
) -> Result<ChatResult, String> {
    let messages = trim_context_messages(messages);

    let mut api_messages = vec![json!({
        "role": "system",
        "content": SYSTEM_PROMPT,
    })];
    api_messages.extend(messages.iter().map(|turn| {
        json!({
            "role": turn.role,
            "content": turn.content,
        })
    }));

    let mut thinking = String::from("Thinking…");
    let mut metrics = OllamaMetrics::default();

    for _ in 0..MAX_TOOL_ROUNDS {
        send_thinking(progress_tx, &thinking);

        let response = stream_chat_request(config, &api_messages, &thinking, progress_tx)?;
        metrics.merge(&response.metrics);
        let assistant = response.message;

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

            let result = if invocation.name == "run_command" {
                request_command_approval(&invocation, progress_tx, &mut thinking)?
            } else {
                execute_tool(&invocation, config)?
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

fn append_tool_preview(thinking: &mut String, invocation: &ToolInvocation) {
    match invocation.name.as_str() {
        "web_search" => {
            if let Some(query) = invocation.arguments.get("query").and_then(Value::as_str) {
                thinking.push_str(&format!(" ({query})"));
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
        "get_time" => format!("Current time: {result}"),
        "run_command" => truncate(result, 600),
        "web_search" => truncate(result, 600),
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

fn stream_chat_request(
    config: &LlmConfig,
    messages: &[Value],
    thinking_prefix: &str,
    progress_tx: &Sender<ChatProgressEvent>,
) -> Result<StreamChatResult, String> {
    let url = format!("{}/api/chat", normalize_base_url(&config.base_url));
    let request = ChatRequest {
        model: config.model.clone(),
        messages: messages.to_vec(),
        stream: true,
        tools: tool_definitions(),
        keep_alive: KEEP_ALIVE.into(),
        truncate: true,
        options: ModelOptions {
            num_ctx: config.num_ctx,
        },
    };

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

fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the current local date and time",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }),
        json!({
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
        json!({
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
    ]
}

fn execute_tool(invocation: &ToolInvocation, config: &LlmConfig) -> Result<String, String> {
    match invocation.name.as_str() {
        "get_time" => Ok(get_time()),
        "web_search" => web_search(config, &invocation.arguments),
        other => Err(format!("Unknown tool: {other}")),
    }
}

fn get_time() -> String {
    Local::now()
        .format("%A, %B %d, %Y at %I:%M %p %Z")
        .to_string()
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
    matches!(name, "get_time" | "web_search" | "run_command")
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
    fn parses_json_get_time_request() {
        let request = parse_tool_request(r#"{"tool":"get_time"}"#, &[]);
        assert_eq!(request.as_ref().map(|r| r.name.as_str()), Some("get_time"));
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
