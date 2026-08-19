use chrono::Local;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::mpsc::Sender;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:11434";
const DEFAULT_MODEL: &str = "llama3.2";
const MAX_TOOL_ROUNDS: usize = 6;
const TAVILY_SEARCH_URL: &str = "https://api.tavily.com/search";

const SYSTEM_PROMPT: &str = "\
You are a helpful assistant in a chat app.

You have two tools:
- get_time: returns the current local date and time.
- web_search: searches the web for up-to-date information.

If you need a tool, respond with ONLY a JSON object and nothing else:
{\"tool\":\"get_time\"}
{\"tool\":\"web_search\",\"query\":\"your search query\"}

After you receive a tool result, answer the user in plain language.";

#[derive(Clone)]
pub struct LlmConfig {
    pub enabled: bool,
    pub base_url: String,
    pub model: String,
    pub tavily_api_key: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: DEFAULT_BASE_URL.into(),
            model: DEFAULT_MODEL.into(),
            tavily_api_key: std::env::var("TAVILY_API_KEY").unwrap_or_default(),
        }
    }
}

#[derive(Clone, Serialize)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

#[derive(Clone)]
struct ToolInvocation {
    name: String,
    arguments: Value,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Value>,
    stream: bool,
    tools: Vec<Value>,
}

#[derive(Serialize)]
struct TavilySearchRequest<'a> {
    query: &'a str,
    max_results: u8,
    include_answer: bool,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
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
    thinking_tx: &Sender<String>,
) -> Result<String, String> {
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

    for _ in 0..MAX_TOOL_ROUNDS {
        send_thinking(thinking_tx, &thinking);

        let response = send_chat_request(config, &api_messages)?;
        let assistant = response.message;

        if !assistant.content.trim().is_empty() {
            thinking.push_str("\n\n");
            thinking.push_str(assistant.content.trim());
            send_thinking(thinking_tx, &thinking);
        }

        if let Some(invocation) = parse_tool_request(&assistant.content, &assistant.tool_calls) {
            thinking.push_str("\n\n→ Calling ");
            thinking.push_str(&invocation.name);
            if invocation.name == "web_search" {
                if let Some(query) = invocation.arguments.get("query").and_then(Value::as_str) {
                    thinking.push_str(&format!(" ({query})"));
                }
            }
            thinking.push('…');
            send_thinking(thinking_tx, &thinking);

            let result = execute_tool(&invocation, config)?;

            thinking.push_str("\n\n← ");
            thinking.push_str(&summarize_tool_result(&invocation.name, &result));
            send_thinking(thinking_tx, &thinking);

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
                "content": result,
            }));
            continue;
        }

        if assistant.content.trim().is_empty() {
            return Err("The model returned an empty response.".into());
        }

        return Ok(assistant.content);
    }

    Err("The model kept requesting tools without producing a final answer.".into())
}

fn send_thinking(thinking_tx: &Sender<String>, content: &str) {
    let _ = thinking_tx.send(content.to_string());
}

fn summarize_tool_result(tool_name: &str, result: &str) -> String {
    match tool_name {
        "get_time" => format!("Current time: {result}"),
        "web_search" => truncate(result, 600),
        _ => truncate(result, 600),
    }
}

fn truncate(text: &str, max_len: usize) -> String {
    if text.chars().count() <= max_len {
        return text.to_string();
    }

    let end = text.char_indices().nth(max_len).map(|(i, _)| i).unwrap_or(text.len());
    format!("{}…", &text[..end])
}

fn send_chat_request(config: &LlmConfig, messages: &[Value]) -> Result<ChatResponse, String> {
    let url = format!("{}/api/chat", normalize_base_url(&config.base_url));
    let request = ChatRequest {
        model: config.model.clone(),
        messages: messages.to_vec(),
        stream: false,
        tools: tool_definitions(),
    };

    let response = reqwest::blocking::Client::new()
        .post(url)
        .json(&request)
        .send()
        .map_err(connection_error)?;

    if !response.status().is_success() {
        return Err(format!("Ollama returned HTTP {}", response.status()));
    }

    response.json().map_err(|err| err.to_string())
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

    let mut arguments = json!({});
    if name == "web_search" {
        let query = value
            .get("query")
            .or_else(|| value.pointer("/arguments/query"))
            .cloned()
            .unwrap_or(Value::Null);
        arguments = json!({ "query": query });
    }

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
    matches!(name, "get_time" | "web_search")
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
