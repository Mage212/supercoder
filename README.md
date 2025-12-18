# 🤖 SuperCoder

**AI Coding Assistant for the Terminal** — A powerful, extensible, and terminal-native coding agent designed to help you build, search, and fix code with natural language.

---

## ✨ Core Features

### 🔍 Code Search
Performs complex code searches across your project to quickly locate specific patterns using `git grep` with context-aware output and fallback to standard `grep`.

### 📁 Project Structure Exploration
Provides an organized, tree-based view of your project's folders and files, intelligently ignoring build artifacts and junk files (`.git`, `node_modules`, etc.).

### ✏️ Intelligent Code Editing
Modifies your codebase seamlessly using diff-based operations. Supported operations include:
- `search_replace`: Precise text replacement.
- `insert_after`/`insert_before`: Contextual code insertion.
- `replace_lines`: Range-based line modification.
- `create`: New file generation.

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

**Example `.supercoder.yaml`:**
```yaml
api_key: "sk-..."
model: "gpt-4o"
base_url: "https://api.openai.com/v1"
max_context_tokens: 32000
use_repo_map: true

# Multiple model profiles
model_profiles:
  gpt4:
    model: "gpt-4o"
    base_url: "https://api.openai.com/v1"
  local:
    model: "qwen2.5-coder:7b"
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"
```

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
