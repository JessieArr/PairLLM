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
| `ls` | List files in a directory (Linux/macOS only; optional flags `a`, `l`, `R`) | `{"tool":"ls","path":"/path/to/dir","flags":"la"}` |
| `cat` | Print a file's contents (Linux/macOS only) | `{"tool":"cat","path":"/path/to/file"}` |
| `sed` | Search-and-replace in a file (like `sed -i 's/pattern/replacement/' file.txt`; Linux/macOS only). Escape `\`, `&`, `/`, and backreferences in the replacement. | `{"tool":"sed","path":"/path/to/file.txt","expression":"s/old/new/"}` |

Some tools are platform-specific. On Windows, `ls`, `cat`, and `sed` are omitted from the tool list exposed to the model.

| `run_command` | Run a shell command (requires approval) | `{"tool":"run_command","command":"ls -la"}` |

The app runs tools locally (or calls Tavily), sends the result back to the model, and then the model answers the user. **Shell commands and file tool access require your approval** before they run.

#### File access permissions

`ls`, `cat`, and `sed` prompt inline in the chat the first time the assistant accesses a directory. Choose:

- **Allow for this directory** — allow files in that directory only (not subdirectories)
- **Allow recursively** — allow that directory and everything beneath it
- **Reject** — block access for this session

The prompt stays visible with your decision for the session. Use **Save for all sessions** to persist the rule to `settings.json`.

Manage saved rules under **Settings → File access permissions**. When multiple rules match a path, the **most specific** path wins (e.g. allow `/home/you` but deny `/home/you/.ssh` blocks `.ssh`).

Persistent rules are stored in `settings.json`:

```json
"path_permissions": [
  { "path": "/home/you/projects", "permission": "allow_recursive" },
  { "path": "/home/you/.ssh", "permission": "deny" }
]
```

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
