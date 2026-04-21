# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Commits

All commits must include `Co-Authored-By: GLM-5.1` (no email) on the last line of the commit message body.

## Commands

All commands must be run from the `supercoder/` subdirectory (where `pyproject.toml` lives).

```bash
# Install for development
uv sync --dev

# Run the CLI
uv run supercoder

# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_tool_parser.py

# Run a single test
uv run pytest tests/test_tool_parser.py::test_name

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run pyright
```

## Architecture

SuperCoder is a terminal REPL that wraps an agentic LLM loop with tool-calling to read and edit code.

### Bootstrap

`supercoder/main.py` — Click CLI entry point. Creates `Config`, `OpenAIClient`, `CoderAgent`, `SuperCoderREPL`, then calls `repl.run()`. On first launch with no API key, launches an interactive setup wizard (`setup_wizard.py`) instead of crashing.

### Agent loop

`supercoder/agent/coder_agent.py` — The core agentic loop. Two modes:

- **Native mode (default, `chat_turn`)**: Non-streaming. Sends tool schemas via the OpenAI `tools` parameter. The API returns structured `tool_calls` — no text parsing needed. This is the primary path.
- **Streaming mode (deprecated, `chat_stream`)**: Text-based tool parsing via `ToolCallParser`. Retained for backward compatibility but should not be used for new work.

Both modes iterate up to 50 tool-call rounds per user message. Each iteration: call LLM → execute tools → append results → loop.

Events are yielded as dicts (`{"type": "response", ...}`, `{"type": "tool_call", ...}`, etc.) consumed by the REPL for display.

### Tool call parsing

`supercoder/agent/tool_parser.py` — **Only used in legacy streaming mode.** A waterfall parser that tries 6 parsers in order. Supports: `supercoder` tags, `qwen_like`, `json_block`, `xml_function`, `glm_tool_call`. Includes JSON repair (`_repair_json`) for malformed output from small/local models.

### Tools

`supercoder/tools/` — All inherit `BaseTool` (which provides `ToolDefinition` → OpenAI schema via `to_openai_schema()`).

- `FileReadTool` — Path-validated file reading
- `CodeSearchTool` — git grep with fallback to regular grep
- `CodeEditTool` — Diff-based edits. Operations: `search_replace`, `insert_after`, `replace_lines`, `create`. Injected with `CheckpointManager` and `allowed_root` at init.
- `ProjectStructureTool` — tree output
- `CommandExecutionTool` — subprocess with timeout, streaming output, and interactive process handling

Tools are registered in `supercoder/tools/__init__.py` as `ALL_TOOLS`. The agent mode system (`agent_modes.py`) filters which tools are available: CODE mode gets all, ASK mode gets only read-only tools.

### LLM client

`supercoder/llm/openai_client.py` — Single client class `OpenAIClient` (extends `BaseLLM`). Works with any OpenAI-compatible endpoint (OpenAI, OpenRouter, Ollama, LM Studio). Key method: `chat_with_tools()` returns a `CompletionResult` with content, parsed `NativeToolCall` list, and reasoning content (for GLM/DeepSeek models).

### Safe editing

- `CheckpointManager` (`checkpoint.py`) backs up files before every edit. On success, checkpoint is committed; on error/abort, files are rolled back automatically.
- `AbortController` (`abort_controller.py`) handles double-ESC interruption. Raises `AgentAbortedError` which triggers checkpoint rollback.
- All file writes go through `AtomicWriter` (`utils/atomic_writer.py`) — write to temp, then `os.replace`.

### Context management

`supercoder/context/window_manager.py` — Tracks token counts per message using tiktoken. When history exceeds 70% of the configured limit, it signals compression. Always reserves 4096 tokens for the model response. `/compact` summarization uses a separate LLM call to produce a structured summary.

### Configuration

`supercoder/config.py` — Multi-model support via named `ModelProfile`s. Load priority: env vars > `.supercoder.yaml` (local) > `~/.supercoder/config.yaml` (global). Each profile has its own `tool_calling_type`, `max_context_tokens`, `endpoint`, and `api_key`. Runtime switching via `/model` command reinitializes the LLM client and rebuilds the system prompt.

### System prompts

`supercoder/agent/prompts.py` — Builds the system prompt dynamically. When `native_tools=True` (default), minimal tool instructions are included (tools are passed via API). When `native_tools=False` (streaming mode), verbose format-specific instructions are injected via `tool_calling_prompts.py`.

### RepoMap

`supercoder/repomap/` — Uses tree-sitter to extract symbols from code files, builds a networkx graph of references, generates a token-limited summary injected into the system prompt. Persisted to `.supercoder/repomap/repo_map.txt`.

### REPL

`supercoder/repl.py` — Interactive loop using prompt_toolkit. Handles multiline input (`{ ... }` blocks), autocomplete, slash commands, and event dispatch from the agent. Displays reasoning, tool calls, diffs, and status footer with token usage bar.

### Session persistence

`supercoder/context/session_manager.py` — Auto-saves after each turn to `.supercoder/sessions/` (max 10). `/continue` resumes; `/compact` updates the session with the summary.

## Key design constraints

- All file writes must go through `AtomicWriter` — write to temp, then rename. Never write directly.
- Tool results are appended as `role="tool"` messages with `tool_call_id` before the next LLM call (native mode).
- Shell commands require explicit user confirmation before execution.
- The REPL uses prompt_toolkit for input; history stored in `.supercoder/history`.
- Logs (JSONL) go to `~/.supercoder/logs/` (outside the project tree). Each session creates `session_YYYYMMDD_HHMMSS.jsonl`. Log writes are non-blocking.
- Streaming mode (`chat_stream`) is deprecated — all new work should target native tool calling (`chat_turn`).


## AXME Code

### Session Start (MANDATORY)
Call axme_context tool with this project's path at the start of every session.
This loads: oracle, decisions, safety rules, memories, test plan, active plans.
Do NOT skip - without context you will miss critical project rules.

### Pending Audits Check (MANDATORY at session start)
When you call axme_context at session start, its output may contain a section
titled "## ⚠️ Pending audits (knowledge base may be incomplete)". This means
a previous session's LLM audit is still running in the background, and the
knowledge base you just loaded does not yet include its extracted memories,
decisions, or handoff.

When you see this section, you MUST:
1. Tell the user there is a pending audit, quote how many sessions and how
   long they have been running.
2. Offer the user two options:
   a) Wait a few minutes, then you will re-run axme_context before starting
      work so the knowledge base is fresh.
   b) Add a TODO to check back in N minutes, continue with other work in
      parallel, and re-run axme_context periodically until the pending
      audits section disappears.
3. Keep the TODO open until all pending audits are gone. Do NOT silently
   remove it — only mark it done after the pending section is empty.

This prevents you from missing freshly-extracted rules from the previous
session that might contradict what you are about to do.

### Storage paths (critical)
For any direct inspection of .axme-code/ files via Bash (ls, cat, grep, find),
ALWAYS use the absolute path from axme_context output's "# AXME Storage Root"
header. Do NOT use relative paths from your cwd. In a multi-repo workspace the
workspace root and each child repo both have their own separate .axme-code/
storage, and reading the wrong one silently gives you stale or missing data.

Every session's meta.json contains an "origin" field with the absolute path of
the directory where the MCP server was running when the session was created.
Whenever you pick up a session file directly (not via axme_context) — for
example to audit a previous run, verify an audit log, or cross-reference past
work — read meta.origin FIRST to confirm which .axme-code/ storage that session
belongs to. This is the authoritative per-session source of truth.

### Reloading axme-code after code changes
Running 'npm run build' in axme-code does NOT reload the MCP server attached to
the current VS Code window — Node caches modules in memory for the server's
lifetime. After any code change to axme-code, close and reopen the VS Code
window (Developer: Reload Window) before testing new behavior. The detached
audit worker reads fresh code from disk on each invocation, so audit-logic
iterations take effect immediately; only changes to the MCP server itself
(tool definitions, cleanupAndExit, startup) require a window reload.

### During Work
- Error pattern or successful approach discovered -> call axme_save_memory immediately
- Architectural decision made or discovered -> call axme_save_decision immediately
- New safety constraint found -> call axme_update_safety immediately
Do not defer - save when discovered.

### Available AXME Tools
axme_context, axme_oracle, axme_decisions, axme_memories, axme_save_memory, axme_save_decision,
axme_update_safety, axme_safety, axme_status, axme_worklog, axme_workspace
