# 🤖 SuperCoder

[![Version](https://img.shields.io/badge/version-0.3.7-blue.svg)](https://github.com/Mage212/supercoder)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**AI Coding Assistant for the Terminal** — A powerful, extensible, and terminal-native coding agent designed to help you build, search, and fix code with natural language.

---

## 🆕 What's New in v0.3.7

- **Host-Enforced Modes**: SuperCoder now has `ask`, `plan`, `code`, and `accept-edits` modes enforced by the host before tools run.
- **Cache-Friendly Mode Switching**: Mode changes no longer rebuild the system prompt or tool schema list; SuperCoder announces mode policy in-band only when needed.
- **Plan Files**: `plan` mode can save dated plans under `.supercoder/plans/` while blocking project file edits and shell commands.
- **Mode UX**: Use `/plan`, `/accept-edits`, `/accept`, `/edit`, or `Shift+Tab` to switch modes with a live toolbar indicator.

See [CHANGELOG.md](CHANGELOG.md) for the full release history.

---

## ✨ Core Features

### 🔍 Code Search
Performs code searches across your project using `ripgrep` (`rg`) when available, with a Python fallback. Results are capped and include compact metadata so local models get precise context without being flooded by search output.

### 🧭 File Discovery
Use the `glob` tool to find matching files by pattern (for example `**/*.py`) without reading their contents. This is useful before targeted `file-read` calls and keeps context small.

### 🛡️ Host-Side Permissions
SuperCoder enforces command and path safety in application code instead of relying only on model instructions. Shell commands are checked against `allow` / `ask` / `deny` rules, while sensitive files such as `.env`, private keys, credentials, and SSH/AWS secrets are blocked before they can be read, edited, searched, listed, or attached with \@path. `.env.example` remains allowed as a safe template.

### 🧷 Read-Before-Edit Freshness
Before editing an existing file, SuperCoder verifies that the model has fresh file context from `file-read` or an explicit \@file attachment. If the file was never read, or changed externally after it was read, `code-edit` is blocked and the model is asked to read the file again.

### 🧭 Host-Enforced Modes
Switch between `ask`, `plan`, `code`, and `accept-edits` with slash commands or `Shift+Tab`. `code` reads and searches freely, then asks before shell commands or file edits; `accept-edits` enables file edits without per-edit prompts. `plan` can persist dated plans only under `.supercoder/plans/`. Mode changes are enforced by SuperCoder itself and announced without rebuilding the system prompt, preserving prompt-cache locality for local models.

### 📎 Explicit Context References
Mention files or directories directly in a prompt with \@path, for example `Review @supercoder/repl.py`. SuperCoder attaches bounded file content or a directory file listing before the model call, with autocomplete suggestions while typing \@ma.

### 📁 Project Structure Exploration
Provides an organized, tree-based view of your project's folders and files, intelligently ignoring build artifacts and junk files (`.git`, `node_modules`, etc.).

### ✏️ Intelligent Code Editing
Modifies your codebase seamlessly using diff-based operations. Every edit is **atomic** and protected by a **checkpoint system**:
- **Atomic Writes**: Changes are written to temporary files first, then moved to the original path.
- **Auto-Backups**: Original file state is saved before any modification.
- **Host Approval**: In `code` mode, every file edit asks for manual approval before execution. Use `accept-edits` when you want edits applied without per-edit prompts.
- **Smart Undo**: Revert any number of changes with the `/undo` command.
- **Operations**: `search_replace`, `insert_after`, `replace_lines`, and `create`.

### 📜 Supercoder Rules (Custom rules)
Leverage project-specific rules to guide the agent. Place `.md` files in `.supercoder/rules/` and they will be automatically loaded into the agent's context. In lean mode, rules are compacted but still included.

### 🗺️ RepoMap Support
Uses `tree-sitter` and `networkx` to generate a high-level map of your repository, helping the LLM understand relationships between files and symbols. Runtime artifacts, virtual environments, cache folders, and `.supercoder` internals are ignored to avoid prompt pollution.

### 🧠 Context Management
- **Token Counter**: Real-time monitoring of context usage.
- **Cache-Aware Compaction**: Use `/compact` to summarize conversation history without switching to a separate summarization prompt, which keeps local-model prompt cache useful.
- **Auto-Compact**: Long sessions compact automatically around 75% of usable context, with emergency trimming left as a fallback.
- **Protected Recent Steps**: After compacting, SuperCoder keeps the summary plus the last 6 exact conversation steps.
- **Tool Output Compaction**: Large tool outputs are summarized for the model and stored in full under `.supercoder/tool-outputs/`.

### 🧪 Debug Diagnostics
Run with `--debug` to write JSONL logs to `~/.supercoder/logs/`. Logs include native tool-call metadata, tool result masking events, permission decisions, edit confirmations, freshness checks, offload paths, API request messages, reasoning, responses, and errors.

### 💾 Session Persistence
- **Auto-Save**: Your conversation is automatically saved after each message exchange.
- **Interactive Resume**: Use `/continue` to browse sessions with arrow keys and select one to restore.
- **Visual History**: Restored sessions render full conversation with original styling — tool calls, reasoning, markdown responses.
- **Session Storage**: Up to 10 sessions stored in `.supercoder/sessions/`.
- **Compact Integration**: When you `/compact`, the session file is also updated with the summary.

---

## 🚀 Getting Started

### Installation

**From GitHub (recommended):**
```bash
pip install git+https://github.com/Mage212/supercoder.git
```

**For development (editable mode):**
```bash
git clone https://github.com/Mage212/supercoder.git
cd supercoder
uv sync --dev
uv run supercoder
```
### First Run — Interactive Setup

On first launch, if no config file or API key is found, SuperCoder automatically starts an interactive setup wizard:

```
╭──────────────────────────────────────────────────────╮
│ 🚀 SuperCoder Setup                                  │
│ No API key configured. Let's set up your provider!   │
╰──────────────────────────────────────────────────────╯

Choose a provider:
  1. OpenAI           (https://api.openai.com/v1)
  2. OpenRouter       (https://openrouter.ai/api/v1)
  3. Anthropic via OR (https://openrouter.ai/api/v1)
  4. Ollama (local)   (http://localhost:11434/v1)
  5. Custom endpoint

Provider (1): 2
...
✓ Configuration saved to: ~/.supercoder/config.yaml
```

After saving, the REPL starts immediately — no restart needed.

SuperCoder supports multiple models and endpoints. Configure them via environment variables or a config file.

**Configuration files (in order of priority):**
1. Environment variables (highest priority)
2. `.supercoder.yaml` in your project directory
3. `~/.supercoder/config.yaml` (global config)

**Environment Variables:**
```bash
export SUPERCODER_API_KEY="sk-..."
export SUPERCODER_MODEL="gpt-4o"
export SUPERCODER_BASE_URL="https://api.openai.com/v1"  # Optional
```

**Custom Endpoints (OpenRouter, Ollama, LM Studio, etc.):**
```bash
export SUPERCODER_BASE_URL="https://openrouter.ai/api/v1"
export SUPERCODER_API_KEY="sk-or-..."
export SUPERCODER_MODEL="openai/gpt-4o"
```

**Example `~/.supercoder/config.yaml`:**
```yaml
# Default model profile to use on startup
default_model: "default"

# Model profiles - define multiple LLM configurations
models:
  default:
    api_key: "sk-..."
    endpoint: "https://api.openai.com/v1"
    model: "gpt-4o-mini"
  
  # OpenRouter with Qwen-style model
  openrouter-qwen:
    api_key: "sk-or-v1-..."
    endpoint: "https://openrouter.ai/api/v1"
    model: "openai/gpt-oss-20b:free"
    tool_calling_type: "qwen_like"  # See Tool Calling Types below
  
  # Local Ollama
  ollama:
    api_key: "ollama"
    endpoint: "http://localhost:11434/v1"
    model: "qwen2.5-coder:7b"
    tool_calling_type: "supercoder"
    lean: true

# Shared settings (applied to all models)
temperature: 0.2
max_context_tokens: 32000
reserved_for_response: 4096
auto_compact: true
auto_compact_threshold: 0.75
protected_recent_steps: 6
compression_threshold: 0.95
request_timeout: 300.0
debug: false
streaming: false

permissions:
  command-exec:
    allow:
      - "uv run pytest*"
      - "uv run ruff*"
      - "uv run pyright*"
      - "git status*"
      - "git diff*"
    ask:
      - "git commit*"
      - "git push*"
    deny:
      - "sudo *"
      - "rm -rf *"
      - "curl * | sh"
      - "curl * | bash"
      - "wget * | sh"
      - "wget * | bash"
  paths:
    deny:
      - ".env"
      - ".env.*"
      - "**/.env"
      - "**/.env.*"
      - "**/*.pem"
      - "**/*.key"
      - "**/credentials.json"
      - "**/.aws/credentials"
      - "**/.ssh/id_*"
    allow:
      - ".env.example"
      - "**/.env.example"
```

### Tool Calling Types

Native API tool calling is the default path and passes schemas through the OpenAI-compatible `tools` parameter. `tool_calling_type` is mainly used by the deprecated streaming mode for models that emit tool calls as text:

| Type | Format | Best for |
|------|--------|----------|
| `supercoder` (default) | `<@TOOL>{"name": "...", "arguments": {...}}</@TOOL>` | Most instruction-following models |
| `qwen_like` | `to=tool:name {"arg": "value"}` | Qwen, GPT-OSS, DeepResearch models |
| `json_block` | ` ```json {"tool": "...", "arguments": {...}} ``` ` | Models trained on markdown |
| `xml_function` | `<function_call name="...">...</function_call>` | XML-style models |
| `glm_tool_call` | `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` | GLM-4 models |

---

## ⌨️ Usage

Launch the interactive REPL:

```bash
supercoder
```

### CLI Options

```bash
supercoder --help
supercoder --model gpt-4o              # Use specific model or profile name
supercoder --endpoint http://...       # Override LLM API endpoint
supercoder --temperature 0.5           # Override temperature
supercoder --debug                     # Enable debug mode
supercoder --no-repo-map               # Disable RepoMap
supercoder --max-context 16000         # Override context token limit
supercoder --stream                    # Enable deprecated text-streaming mode
```

### Slash Commands

| Command | Description |
|---------|-------------|
| `/ask` | Switch to Ask mode (Q&A without edits) |
| `/ask <question>` | Ask one question without editing, then return to previous mode |
| `/plan` | Switch to Plan mode (read/search, dated plans only) |
| `/code` | Switch to Code mode (edits require approval) |
| `/code <request>` | Execute one request in code mode, then return to previous mode |
| `/accept-edits` | Switch to editing mode (file edits apply without per-edit prompts) |
| `/undo` | Revert changes to a specific checkpoint |
| `/help` | Show available commands |
| `/continue` | Resume a previous session (interactive picker) |
| `/compact` | Cache-aware summary compaction that keeps recent steps |
| `/stats` | View current token usage and context status |
| `/clear` | Clear conversation history |
| `/config` | Show current active configuration |
| `/models` | List available model profiles |
| `/model <name>` | Switch to a specific model profile |
| `/debug` | Toggle verbose debug logging |
| `/exit` | Exit the application |

---

## 🛡️ Safety & Integrity

### Atomic File Writes
SuperCoder uses an `AtomicFileWriter` to ensure that files are never left in a corrupted state if a write operation is interrupted. This uses the `tempfile` + `os.replace` pattern, preserves existing file permissions on rewrite, and is standard for safe filesystem operations.

### Checkpoint System
Every user message that leads to a file modification creates a new **Checkpoint**. 
- **Backups**: Stored in project-local `.supercoder/checkpoints/`.
- **Created Files**: Tracked and automatically deleted on rollback.
- **Rotation**: Automatically keeps only the last 10 checkpoints to save space.
- **Self-Healing**: Incomplete or orphaned checkpoint directories are automatically cleaned on startup.

### Command Execution Confirmation
Before running any shell command, SuperCoder pauses and asks for explicit approval:
```
⚡ Run Command?
Command:
  <the command to execute>
  [y] Once   [s] Session   [a] Always   [d] Always deny   [n] No
```
Single keypress response:
- `[y]` approves once.
- `[s]` allows the exact command for the current process.
- `[a]` saves a project-local allow rule in `.supercoder/permissions.yaml`.
- `[d]` saves a project-local deny rule in `.supercoder/permissions.yaml`.

Use `/permissions` to inspect, remove, or clear project-local command approval rules.

### Interruption (ESC-ESC)
Press **ESC twice** to abort at any time — during generation, tool calls, or streaming.
1. The keyboard listener detects the interrupt.
2. The current LLM stream is aborted immediately.
3. Any partial file changes from the current turn are **rolled back** automatically.

---

## 📁 Project Structure

```text
supercoder/
├── agent/            # CoderAgent logic and prompts
├── context/          # Token counting, context window, and session management
├── llm/              # LLM providers (OpenAI-compatible endpoints)
├── repomap/          # Repository mapping logic (tree-sitter)
├── tools/            # Core tools (Search, Edit, Structure, Exec)
├── rules_loader.py   # Supercoder Rules loading logic
├── config.py         # Configuration management
├── logging.py        # Conversation logging (JSONL → ~/.supercoder/logs/)
├── setup_wizard.py   # Interactive first-run setup wizard
├── repl.py           # Interactive REPL interface
└── main.py           # CLI entry point
```

---

## 📦 Dependencies

**Core:**
- `openai` — LLM API client
- `click` — CLI framework
- `rich` — Beautiful terminal output
- `prompt-toolkit` — Interactive input
- `networkx` — Graph-based RepoMap
- `tree-sitter-languages` — Code parsing for RepoMap
- `tiktoken` — Token counting
- `pyyaml` — Configuration files
- `questionary` — Interactive terminal prompts
- `json-repair` — Recovery for malformed JSON tool arguments

---

## ⚖️ License

MIT
