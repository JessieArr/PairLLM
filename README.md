# PairLLM

A simple Rust desktop chat app built with [egui](https://github.com/emilk/egui). It can chat on its own or reply through a local LLM when one is available.

## Run

Development build:

```bash
cargo run
```

Optimized release build:

```bash
cargo build --release
./target/release/pairllm
```

Release builds enable thin LTO, single codegen unit, and strip debug symbols for a smaller, faster binary.

- **Enter** sends a message
- **Shift+Enter** inserts a newline
- Open **Settings** to configure the local LLM

Settings (model, Ollama URL, context size, Tavily key, etc.) are saved to `~/.config/pairllm/settings.json` and restored on the next launch.

## Local LLM setup (recommended: Ollama)

**Ollama** is the easiest option for app integration:

1. Install from [ollama.com](https://ollama.com)
2. Pull a model:

```bash
ollama pull qwen3:4b
```

Other sizes: `qwen3:0.6b`, `qwen3:1.7b`, `qwen3:4b` — selectable in **Settings**.

3. Make sure Ollama is running (`ollama serve` — often started automatically)
4. Click **Refresh** in the app header

The app talks to Ollama at `http://127.0.0.1:11434` using its native `/api/chat` endpoint. If a model is available, your messages get an **Assistant** reply automatically.

### Tools

The assistant can call tools when it needs extra information. The current local time is included in every request's system prompt.

| Tool | Purpose | JSON request |
|------|---------|--------------|
| `web_search` | Web search via [Tavily](https://tavily.com) | `{"tool":"web_search","query":"your search"}` |
| `list_files` | List files in a directory (like `ls`) | `{"tool":"list_files","path":"/path/to/dir"}` |
| `read_file` | Read a text file | `{"tool":"read_file","path":"/path/to/file"}` |
| `replace_in_file` | Replace a line range in a file (1-based, inclusive) | `{"tool":"replace_in_file","path":"/path/to/file","start_line":1,"end_line":3,"text":"new content"}` |
| `run_command` | Run a shell command (requires approval) | `{"tool":"run_command","command":"ls -la"}` |

The app runs tools locally (or calls Tavily), sends the result back to the model, and then the model answers the user. **CLI commands are never run automatically** — you must approve them in a confirmation dialog first.

#### Tavily setup

Web search works out of the box with Tavily's **keyless mode** (no account or API key required, rate-limited). For higher limits, add a free API key from [tavily.com](https://tavily.com):

```bash
export TAVILY_API_KEY=tvly-your-key-here
cargo run
```

Or paste the key in **Settings** in the app. When a key is provided, it takes precedence over keyless mode.

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
