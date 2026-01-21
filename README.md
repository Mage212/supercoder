# 🤖 SuperCoder

[![Version](https://img.shields.io/badge/version-0.2.8-blue.svg)](https://github.com/Mage212/supercoder)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**AI Coding Assistant for the Terminal** — A powerful, extensible, and terminal-native coding agent designed to help you build, search, and fix code with natural language.

---

## 🆕 What's New in v0.2.8

- **Reasoning Block Display**: Added dedicated support for displaying model "thinking" or "reasoning" steps. Reasoning is displayed in a distinct `💭 Reasoning` block before the main response or tool calls.
- **Incremental Multi-Stage Output**: Improved the REPL to display reasoning and tool calls incrementally. Long multi-turn interactions are now much easier to follow as each stage is rendered as it happens.
- **Improved GLM-4 Integration**: Enhanced tool call parsing and filtering specifically for GLM-4 models, ensuring that raw tool tags are hidden from the final output even if they appear in the reasoning stream.
- **Advanced Session Logging**: Added detailed logging of reasoning steps and streaming events to `.supercoder/logs/`, making it easier to analyze model behavior and debug complex interactions.

### v0.2.7

- **Model-Specific Context Limits**: Each model profile can now have its own `max_context_tokens` limit. This allows automatic switching between models with different context window sizes (e.g., 8k for local models vs 128k for cloud models) without manual reconfiguration.
- **Improved Context Management**: Default context limits specified in `config.yaml` are now correctly respected unless overridden by the CLI flag.

### v0.2.6

- **GLM-4 Support**: Added a dedicated `glm_tool_call` format support specifically optimized for GLM-4.7-Flash and similar models.
- **Multi-Tool Support for GLM**: The agent can now parse multiple tool calls in a single GLM model response.
- **Improved Display Filtering**: Enhanced response filtering to hide raw GLM tool call tags from the assistant's output panel.

### v0.2.5

- **Multiple Tool Call Support**: Updated the parser framework to support models that send multiple tool calls in one turn across different formats.

### v0.2.4

- **Atomic File Writes**: Enhanced reliability by using temporary files for all write operations, preventing data loss on crashes.
- **Checkpoint & Rollback**: Automatic backup before every file modification. Use `/undo` to revert changes instantly.
- **Graceful Interruption**: Press **Double-ESC** during agent work to stop it safely without losing session state or leaving messy file edits.
- **Improved Undo Integration**: The agent is now aware when you perform an undo and will re-evaluate file contents accordingly.

---

## ✨ Core Features

### 🔍 Code Search
Performs complex code searches across your project to quickly locate specific patterns using `git grep` with context-aware output and fallback to standard `grep`.

### 📁 Project Structure Exploration
Provides an organized, tree-based view of your project's folders and files, intelligently ignoring build artifacts and junk files (`.git`, `node_modules`, etc.).

### ✏️ Intelligent Code Editing
Modifies your codebase seamlessly using diff-based operations. Every edit is **atomic** and protected by a **checkpoint system**:
- **Atomic Writes**: Changes are written to temporary files first, then moved to the original path.
- **Auto-Backups**: Original file state is saved before any modification.
- **Smart Undo**: Revert any number of changes with the `/undo` command.
- **Operations**: `search_replace`, `insert_after`, `replace_lines`, and `create`.

### 📜 Supercoder Rules (Custom rules)
Leverage project-specific rules to guide the agent. Place `.md` files in `.supercoder/rules/` and they will be automatically loaded into the agent's context.

### 🗺️ RepoMap Support
Uses `tree-sitter` and `networkx` to generate a high-level map of your repository, helping the LLM understand relationships between files and symbols.

### 🧠 Context Management
- **Token Counter**: Real-time monitoring of context usage.
- **Smart Compaction**: Use `/compact` to summarize conversation history and free up token space without losing key context.

### 💾 Session Persistence
- **Auto-Save**: Your conversation is automatically saved after each message exchange.
- **Resume Sessions**: Use `/continue` to pick up where you left off after closing SuperCoder.
- **Session History**: Up to 10 sessions are stored in `.supercoder/sessions/`.
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
pip install -e .
```

### Configuration

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

# Shared settings (applied to all models)
temperature: 0.2
max_context_tokens: 32000
request_timeout: 60.0
debug: false
```

### Tool Calling Types

Different models expect tools to be called in different formats. Use `tool_calling_type` to specify the format:

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
supercoder --model gpt-4o          # Use specific model
supercoder --debug                 # Enable debug mode
supercoder --no-repo-map           # Disable RepoMap
```

### Slash Commands

| Command | Description |
|---------|-------------|
| `/ask` | Switch to Ask mode (Q&A without edits) |
| `/ask <question>` | Ask one question without editing, then return |
| `/code` | Switch to Code mode (full editing) |
| `/undo` | Revert changes to a specific checkpoint |
| `/help` | Show available commands |
| `/continue` | Resume a previous session |
| `/sessions` | List saved sessions |
| `/tools` | List active tools and their descriptions |
| `/compact` | Summarize history to save context tokens |
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
SuperCoder uses an `AtomicFileWriter` to ensure that files are never left in a corrupted state if a write operation is interrupted. This uses the `tempfile` + `os.replace` pattern, which is standard for safe filesystem operations.

### Checkpoint System
Every user message that leads to a file modification creates a new **Checkpoint**. 
- **Backups**: Stored in project-local `.supercoder/checkpoints/`.
- **Created Files**: Tracked and automatically deleted on rollback.
- **Rotation**: Automatically keeps only the last 10 checkpoints to save space.
- **Self-Healing**: Incomplete or orphaned checkpoint directories are automatically cleaned on startup.

### Interruption (ESC-ESC)
If the agent is stuck or generating unwanted code, you can press **ESC twice** quickly.
1. The background keyboard listener detects the interrupt.
2. The current LLM stream is aborted immediately.
3. Any partial file changes from the current turn are **rolled back** automatically to maintain project integrity.

---

## 📁 Project Structure

```text
supercoder/
├── agent/          # CoderAgent logic and prompts
├── context/        # Token counting, context window, and session management
├── llm/            # LLM providers (OpenAI-compatible endpoints)
├── repomap/        # Repository mapping logic (tree-sitter)
├── tools/          # Core tools (Search, Edit, Structure, Exec)
├── rules_loader.py # Supercoder Rules loading logic
├── config.py       # Configuration management
├── logging.py      # Conversation logging
├── repl.py         # Interactive REPL interface
└── main.py         # CLI entry point
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

---

## ⚖️ License

MIT
