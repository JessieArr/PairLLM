use crate::llm::LlmConfig;
use std::fs;
use std::path::PathBuf;

pub fn load() -> Option<LlmConfig> {
    let path = settings_path()?;
    let contents = fs::read_to_string(path).ok()?;
    serde_json::from_str(&contents).ok()
}

pub fn save(config: &LlmConfig) -> Result<(), String> {
    let path = settings_path().ok_or("Could not resolve settings path.")?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("Could not create settings dir: {err}"))?;
    }

    let json =
        serde_json::to_string_pretty(config).map_err(|err| format!("Could not encode settings: {err}"))?;
    fs::write(&path, json).map_err(|err| format!("Could not write settings: {err}"))
}

pub fn settings_path() -> Option<PathBuf> {
    if let Ok(home) = std::env::var("HOME") {
        return Some(PathBuf::from(home).join(".config/pairllm/settings.json"));
    }

    std::env::var("USERPROFILE")
        .ok()
        .map(|profile| PathBuf::from(profile).join(".config/pairllm/settings.json"))
}
