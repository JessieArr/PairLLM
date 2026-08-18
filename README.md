# PairLLM

A simple Rust desktop chat app built with [egui](https://github.com/emilk/egui). It can chat on its own or reply through a local LLM when one is available.

## Run

```bash
cargo run
```

- **Enter** sends a message
- **Shift+Enter** inserts a newline
- Open **Settings** to configure the local LLM

## Local LLM setup (recommended: Ollama)

**Ollama** is the easiest option for app integration:

1. Install from [ollama.com](https://ollama.com)
2. Pull a model:

```bash
ollama pull llama3.2
```

3. Make sure Ollama is running (`ollama serve` — often started automatically)
4. Click **Refresh** in the app header

The app talks to Ollama at `http://127.0.0.1:11434` using its native `/api/chat` endpoint. If a model is available, your messages get an **Assistant** reply automatically.

### Other local options

| Tool | Ease | Notes |
|------|------|-------|
| **Ollama** | Easiest | Simple HTTP API, one-command model install — **supported by this app** |
| **LM Studio** | Easy | GUI for models; exposes an OpenAI-compatible API (not wired in yet) |
| **llama.cpp** | Moderate | Run `llama-server`; OpenAI-compatible API, more manual setup |
| **LocalAI** | Moderate | OpenAI-compatible wrapper for many backends |

For llama.cpp specifically, you'd typically run something like:

```bash
llama-server -m your-model.gguf --port 8080
```

That server speaks OpenAI's chat format, not Ollama's. Ollama is still the path of least resistance for this app today.

## Requirements

Linux builds need system libraries for the windowing backend (typically `libxkbcommon`, `libwayland`, and/or X11 development packages depending on your distro).
