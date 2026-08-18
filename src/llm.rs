use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:11434";
const DEFAULT_MODEL: &str = "llama3.2";

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
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatTurn],
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct ChatMessage {
    content: String,
}

#[derive(Deserialize)]
struct TagsResponse {
    models: Vec<TagModel>,
}

#[derive(Deserialize)]
struct TagModel {
    name: String,
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
    let url = format!("{}/api/chat", normalize_base_url(base_url));
    let request = ChatRequest {
        model,
        messages,
        stream: false,
    };

    let response = reqwest::blocking::Client::new()
        .post(url)
        .json(&request)
        .send()
        .map_err(connection_error)?;

    if !response.status().is_success() {
        return Err(format!("Ollama returned HTTP {}", response.status()));
    }

    let body: ChatResponse = response.json().map_err(|err| err.to_string())?;
    Ok(body.message.content)
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
