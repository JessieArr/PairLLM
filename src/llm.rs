use chrono::Local;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:11434";
const DEFAULT_MODEL: &str = "llama3.2";
const MAX_TOOL_ROUNDS: usize = 4;

const SYSTEM_PROMPT: &str = "\
You are a helpful assistant in a chat app.

You have one tool:
- get_time: returns the current local date and time.

If you need the current time to answer the user, request the tool by responding with \
ONLY this JSON object and nothing else:
{\"tool\":\"get_time\"}

After you receive the tool result, answer the user in plain language.";

#[derive(Clone)]
pub struct LlmConfig {
    pub enabled: bool,
    pub base_url: String,
    pub model: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: DEFAULT_BASE_URL.into(),
            model: DEFAULT_MODEL.into(),
        }
    }
}

#[derive(Clone, Serialize)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Value>,
    stream: bool,
    tools: Vec<Value>,
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

#[derive(Debug, Deserialize)]
struct ToolRequest {
    tool: String,
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

pub fn chat(base_url: &str, model: &str, messages: &[ChatTurn]) -> Result<String, String> {
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

    for _ in 0..MAX_TOOL_ROUNDS {
        let response = send_chat_request(base_url, model, &api_messages)?;
        let assistant = response.message;

        if let Some(tool_name) = parse_tool_request(&assistant.content, &assistant.tool_calls) {
            let result = execute_tool(&tool_name)?;

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
                "tool_name": tool_name,
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

fn send_chat_request(
    base_url: &str,
    model: &str,
    messages: &[Value],
) -> Result<ChatResponse, String> {
    let url = format!("{}/api/chat", normalize_base_url(base_url));
    let request = ChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        stream: false,
        tools: vec![get_time_tool_definition()],
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

fn get_time_tool_definition() -> Value {
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
    })
}

fn execute_tool(name: &str) -> Result<String, String> {
    match name {
        "get_time" => Ok(get_time()),
        other => Err(format!("Unknown tool: {other}")),
    }
}

fn get_time() -> String {
    Local::now()
        .format("%A, %B %d, %Y at %I:%M %p %Z")
        .to_string()
}

fn parse_tool_request(content: &str, tool_calls: &[ToolCall]) -> Option<String> {
    for call in tool_calls {
        if call.function.name == "get_time" {
            return Some("get_time".into());
        }
    }

    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }

    for candidate in json_candidates(trimmed) {
        if let Ok(request) = serde_json::from_str::<ToolRequest>(&candidate) {
            if request.tool == "get_time" {
                return Some("get_time".into());
            }
        }

        if let Ok(value) = serde_json::from_str::<Value>(&candidate) {
            if value.get("tool").and_then(Value::as_str) == Some("get_time") {
                return Some("get_time".into());
            }
            if value.get("name").and_then(Value::as_str) == Some("get_time") {
                return Some("get_time".into());
            }
        }
    }

    None
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
    fn parses_json_tool_request() {
        let request = parse_tool_request(r#"{"tool":"get_time"}"#, &[]);
        assert_eq!(request.as_deref(), Some("get_time"));
    }

    #[test]
    fn parses_native_tool_call() {
        let calls = vec![ToolCall {
            function: ToolFunction {
                name: "get_time".into(),
                arguments: json!({}),
            },
        }];
        let request = parse_tool_request("", &calls);
        assert_eq!(request.as_deref(), Some("get_time"));
    }
}
