# Changelog

## v0.3.5

- **Host-Side Permission Policy**: Added deterministic `allow` / `ask` / `deny` command decisions before shell commands are confirmed or executed.
- **Sensitive Path Protection**: Built-in deny rules now protect `.env`, private keys, credentials, AWS credentials, and SSH private-key patterns across file-read, code-edit, code-search, glob, project-structure, and \@path attachments.
- **Configurable Safety Rules**: Added a `permissions` config section for command and path policies, with safe defaults in the generated config template.
- **Audit-Friendly Debug Logs**: Denied and cancelled `command-exec` requests now emit `permission_decision`, `tool_call`, and `tool_result` events for easier session auditing.

## v0.3.4

- **Explicit Context References**: Mention files or directories with \@path to attach bounded context before the model call.
- **Path Autocomplete**: Typing \@ma now suggests matching files and folders, while ignoring runtime/cache directories.
- **Compact-Safe Attachments**: Attached context is stored as `context_attachment` and kept with the related user prompt during compaction.
- **Debug Visibility**: Debug JSONL logs include `context_attachment` metadata without dumping full attached file contents into the event.

## v0.3.3

- **Context-Efficient Tool Results**: Large tool outputs are compacted before they enter model context, with full output offloaded to `.supercoder/tool-outputs/` for inspection.
- **Improved `file-read`**: File reads now include byte metadata, binary-file protection, `maxBytes` caps, and nearby path suggestions for typos.
- **Faster Search + `glob`**: `code-search` now prefers `ripgrep` (`rg`) with a Python fallback, and the new `glob` tool lists matching paths without reading file contents.
- **Structured Debug Diagnostics**: Debug JSONL logs now preserve native tool-call metadata and include `tool_output_masked` events with size/offload details.
- **Cleaner RepoMap Context**: RepoMap ignores runtime and environment directories such as `.supercoder`, `.venv`, cache folders, and dependency trees.

## v0.3.2

- **Cache-Aware Compact for Local Models**: `/compact` now runs as an in-band chat turn instead of a separate summarization prompt, allowing llama.cpp/Ollama/LM Studio-style backends to reuse prompt cache.
- **Automatic Context Compaction**: SuperCoder can auto-compact around 75% of usable context before the emergency trimming fallback is needed.
- **Protected Recent Steps**: Compact keeps the summary plus the last 6 exact conversation steps, preserving the current working state without relying only on abstraction.

## v0.3.1

- **Streaming Tool Call Parser**: Native tool calls that arrive as streamed fragments are now assembled by a dedicated bracket-depth parser instead of a raw string buffer.
- **Truncated Tool Call Detection**: SuperCoder now detects incomplete streamed tool-call JSON and warns when a response was cut off before execution.
- **3-Level JSON Recovery**: Tool argument parsing now tries exact JSON, targeted repair, then `json-repair` for malformed local-model output.
- **Lean Mode Keeps Project Rules**: Lean prompts still stay compact, but project rules are now preserved in a shortened mandatory section instead of being dropped.
- **Debug-Only Conversation Logging**: JSONL logging and full tracebacks are enabled through debug mode, so normal runs avoid unnecessary log files.
- **Safer Atomic Writes**: Atomic rewrites now preserve existing file permissions.
- **Safer Command Guidance**: CODE mode now distinguishes useful validation commands from destructive, sudo/admin, network-install, and broad filesystem operations that require extra care.

## v0.3.0

- **Native API Tool Calling**: Migrated from text-based streaming parsing to native OpenAI-compatible `tools` parameter. More reliable, no format-dependent parsing. Streaming mode still available via `--stream`.
- **Interactive Session Picker**: `/continue` now shows an arrow-key navigable session list with relative timestamps ("5m ago") and message counts. No more typing numbers.
- **Visual Session History**: Restored sessions render full conversation history with the same styling as live output: tool calls interleaved with results, reasoning blocks, markdown responses.
- **Message Display Types**: Each message now stores its role (`user_input`, `thinking`, `response`, `tool_call`, `tool_result`, `error`) for accurate session restoration. Backward compatible with old sessions.
- **Fuzzy Matching for Edits**: Three-tier edit matching: exact -> whitespace-normalized -> fuzzy (`SequenceMatcher`), so local/weak models with formatting inconsistencies still apply edits correctly.
- **Lean Prompt Mode**: Optional 75% shorter system prompts for weak/local models via `lean: true` in model profile config.
- **Parser Hardening**: Fixed 4 Qwen3.5-4B failure modes: missing closing tags, single-quoted JSON, extra characters, and hallucinated tool names.
- **Live Generation Progress**: Spinner shows token count, phase label ("response" / "tool call"), and elapsed time during generation. Works even when providers buffer output.
- **Streaming Abort**: Double-ESC now works during active API streaming, with checkpoint rollback on abort.
- **Inline Auto-suggest**: Gray-text suggestions for slash commands while typing. Enter to accept.
- **Command Approval Menu**: Instant key-press approval (`[y]`/`[a]`/`[n]`) for shell commands instead of typing.
- **Interactive Setup Wizard**: On first launch, a guided TUI wizard configures provider, model, context size, and API key.
- **Full Traceback Logging**: Debug logging can capture complete Python tracebacks to `~/.supercoder/logs/` (JSONL format).

## v0.2.9

- **Enhanced Autocomplete**: Added intelligent autocompletion for slash commands and file paths using `prompt_toolkit`.
- **Multiline Input Support**:
  - Use `{ ... }` blocks for pasting large snippets of code.
  - Use `Alt+Enter` (or `Esc+Enter`) to insert newlines without submitting the message.
- **Improved Streaming Output**: Integrated a custom markdown streaming renderer for smoother, live-updating responses that work better with terminal scrollback.

## v0.2.8

- **Reasoning Block Display**: Added dedicated support for displaying model "thinking" or "reasoning" steps. Reasoning is displayed in a distinct reasoning block before the main response or tool calls.
- **Incremental Multi-Stage Output**: Improved the REPL to display reasoning and tool calls incrementally. Long multi-turn interactions are now much easier to follow as each stage is rendered as it happens.
- **Improved GLM-4 Integration**: Enhanced tool call parsing and filtering specifically for GLM-4 models, ensuring that raw tool tags are hidden from the final output even if they appear in the reasoning stream.
- **Advanced Session Logging**: Added detailed logging of reasoning steps and streaming events to `.supercoder/logs/`, making it easier to analyze model behavior and debug complex interactions.

## v0.2.7

- **Model-Specific Context Limits**: Each model profile can now have its own `max_context_tokens` limit. This allows automatic switching between models with different context window sizes (e.g., 8k for local models vs 128k for cloud models) without manual reconfiguration.
- **Improved Context Management**: Default context limits specified in `config.yaml` are now correctly respected unless overridden by the CLI flag.

## v0.2.6

- **GLM-4 Support**: Added a dedicated `glm_tool_call` format support specifically optimized for GLM-4.7-Flash and similar models.
- **Multi-Tool Support for GLM**: The agent can now parse multiple tool calls in a single GLM model response.
- **Improved Display Filtering**: Enhanced response filtering to hide raw GLM tool call tags from the assistant's output panel.

## v0.2.5

- **Multiple Tool Call Support**: Updated the parser framework to support models that send multiple tool calls in one turn across different formats.

## v0.2.4

- **Atomic File Writes**: Enhanced reliability by using temporary files for all write operations, preventing data loss on crashes.
- **Checkpoint & Rollback**: Automatic backup before every file modification. Use `/undo` to revert changes instantly.
- **Graceful Interruption**: Press **Double-ESC** during agent work to stop it safely without losing session state or leaving messy partial file edits.
- **Improved Undo Integration**: The agent is now aware when you perform an undo and will re-evaluate file contents accordingly.
