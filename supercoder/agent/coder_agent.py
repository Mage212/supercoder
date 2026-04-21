"""Main coding agent with context management."""

import json
from pathlib import Path

from rich.console import Console

from ..abort_controller import AbortController, AgentAbortedError
from ..checkpoint import CheckpointManager
from ..context.session_manager import ChatSession, SessionManager
from ..context.window_manager import ContextConfig, ContextStats, ContextWindowManager
from ..llm.base import BaseLLM, Message
from ..logging import get_logger
from ..repomap import RepoMap
from ..rules_loader import SupercoderRulesLoader
from ..tools.base import BaseTool
from ..tools.code_edit import CodeEditTool
from ..tools.file_read import FileReadTool
from ..tools.project_structure import ProjectStructureTool
from .agent_modes import MODE_CONFIGS, AgentMode
from .prompts import CONTEXT_SUMMARY_PROMPT, build_system_prompt
from .tool_parser import ToolCallParser

console = Console()


class CoderAgent:
    """Main coding agent that orchestrates LLM and tools."""

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[BaseTool] | None = None,
        context_config: ContextConfig | None = None,
        use_repo_map: bool = False,
        repo_root: str = ".",
        tool_calling_type: str = "supercoder",
        streaming: bool = False,
        lean: bool = False,
    ):
        self.llm = llm
        self.repo_root = Path(repo_root).resolve()
        self.streaming = streaming  # False = native API (default), True = deprecated streaming
        self.lean = lean  # Shorter prompts for weak/local models

        # Abort controller for graceful interruption
        self.abort_controller = AbortController()

        # Checkpoint manager for safe file editing with rollback
        self.checkpoint_manager = CheckpointManager(self.repo_root)

        # Initialize tools and inject checkpoint_manager where needed
        self.tools = {}
        for t in tools or []:
            # Inject checkpoint_manager and repo_root into code-edit tool
            if isinstance(t, CodeEditTool):
                t.checkpoint = self.checkpoint_manager
                t.allowed_root = self.repo_root
            # Inject allowed_root into file-read and project-structure tools
            elif isinstance(t, (FileReadTool, ProjectStructureTool)):
                t.allowed_root = self.repo_root
            self.tools[t.definition.name] = t

        # Agent mode (code or ask)
        self._mode = AgentMode.CODE

        # RepoMap setup
        self.repo_map = RepoMap(self.repo_root) if use_repo_map else None

        # Supercoder Rules setup
        self.rules_loader = SupercoderRulesLoader(repo_root)
        self.rules_loader.ensure_rules_dir()  # Create .supercoder/rules/ if missing
        project_rules = self.rules_loader.get_rules_for_prompt()

        # Store tool calling type for prompt generation
        self.tool_calling_type = tool_calling_type
        self._tools_list = tools or []  # Keep reference for prompt rebuilding
        self._project_rules = project_rules

        # Build OpenAI-compatible tool schemas for native mode
        self._tools_schema = [t.definition.to_openai_schema() for t in self._tools_list]

        # Build system prompt template with tools and project rules
        mode_config = MODE_CONFIGS[self._mode]
        suffix = (
            mode_config.lean_prompt_suffix
            if self.lean and mode_config.lean_prompt_suffix
            else mode_config.prompt_suffix
        )
        self.base_system_prompt = build_system_prompt(
            self._get_tools_for_mode(),
            rules=project_rules,
            tool_calling_type=self.tool_calling_type,
            mode_suffix=suffix,
            native_tools=not self.streaming,
            lean=self.lean,
        )

        # Setup context management
        config = context_config or ContextConfig()
        self.context = ContextWindowManager(config)
        self._update_system_prompt()

        # Multi-format tool call parser (used only in deprecated streaming mode)
        self.tool_parser = ToolCallParser(debug=False)

        # Session management
        self.session_manager = SessionManager(self.repo_root)
        self.current_session: ChatSession | None = None

        self.debug = False
        self._chunk_callback = None  # Set by REPL for live token counting

    def set_chunk_callback(self, callback):
        """Set a callback invoked with approx token count during generation."""
        self._chunk_callback = callback

    def _update_system_prompt(self):
        """Update system prompt with latest RepoMap if enabled."""
        prompt = self.base_system_prompt

        if self.repo_map:
            try:
                map_content = self.repo_map.get_repo_map(max_tokens=2000)
                if map_content:
                    prompt += f"\n\n# Repository Structure\n{map_content}"
            except Exception as e:
                if self.debug:
                    console.print(f"[red]Error generating RepoMap: {e}[/]")
                get_logger().log_error(e)

        self.context.set_system_prompt(prompt)
        # Log the updated system prompt
        get_logger().log_system_prompt(prompt)

    def _get_tools_for_mode(self) -> list:
        """Return tools available in current mode."""
        mode_config = MODE_CONFIGS[self._mode]

        if mode_config.allowed_tools is None:
            # All tools allowed
            return self._tools_list

        # Filter to only allowed tools
        return [t for t in self._tools_list if t.definition.name in mode_config.allowed_tools]

    @property
    def mode(self) -> AgentMode:
        """Get current agent mode."""
        return self._mode

    def set_mode(self, mode: AgentMode) -> None:
        """Switch agent mode and update available tools and prompt."""
        if mode == self._mode:
            return

        self._mode = mode

        # Rebuild system prompt with new mode's tools and suffix
        mode_config = MODE_CONFIGS[self._mode]
        suffix = (
            mode_config.lean_prompt_suffix
            if self.lean and mode_config.lean_prompt_suffix
            else mode_config.prompt_suffix
        )
        self.base_system_prompt = build_system_prompt(
            self._get_tools_for_mode(),
            rules=self._project_rules,
            tool_calling_type=self.tool_calling_type,
            mode_suffix=suffix,
            native_tools=not self.streaming,
            lean=self.lean,
        )
        self._update_system_prompt()

    def set_tool_calling_type(self, tool_calling_type: str) -> None:
        """Update tool calling type and rebuild system prompt.

        Call this when switching to a model with a different tool_calling_type.
        """
        if tool_calling_type != self.tool_calling_type:
            self.tool_calling_type = tool_calling_type
            mode_config = MODE_CONFIGS[self._mode]
            suffix = (
                mode_config.lean_prompt_suffix
                if self.lean and mode_config.lean_prompt_suffix
                else mode_config.prompt_suffix
            )
            self.base_system_prompt = build_system_prompt(
                self._get_tools_for_mode(),
                rules=self._project_rules,
                tool_calling_type=self.tool_calling_type,
                mode_suffix=suffix,
                native_tools=not self.streaming,
                lean=self.lean,
            )
            self._update_system_prompt()

    # ------------------------------------------------------------------
    # Primary path: native tool calling (non-streaming)
    # ------------------------------------------------------------------

    def chat_turn(self, user_message: str):
        """Process user message using native API tool calls (non-streaming).

        Yields event dicts consumed by the REPL for display:
          - ``{"type": "thinking", "content": "..."}``      → reasoning
          - ``{"type": "response", "content": "..."}``      → full text response
          - ``{"type": "tool_call", "content": {...}}``      → tool invocation
          - ``{"type": "tool_result", "content": {...}}``    → tool output
          - ``{"type": "command_confirm", ...}``             → confirm shell cmd
          - ``{"type": "command_waiting", ...}``             → interactive cmd
          - ``{"type": "error", "content": "..."}``
          - ``{"type": "rollback", "content": {...}}``
          - ``{"type": "done", "content": ""}``
        """
        MAX_TOOL_ITERATIONS = 50

        # Reset abort controller
        self.abort_controller.reset()

        # Create checkpoint for this interaction
        checkpoint_active = False
        if user_message:
            self.checkpoint_manager.create(description=user_message[:100])
            checkpoint_active = True
            self.context.add_message(Message("user", user_message, display_type="user_input"))
            get_logger().log_user_input(user_message)

        # Update RepoMap if enabled
        if self.repo_map:
            self._update_system_prompt()

        tool_iterations = 0

        while True:
            if tool_iterations >= MAX_TOOL_ITERATIONS:
                if checkpoint_active:
                    self.checkpoint_manager.rollback()
                yield {
                    "type": "error",
                    "content": f"Tool call limit ({MAX_TOOL_ITERATIONS}) reached. Stopping to prevent infinite loop.",
                }
                return

            messages = self.context.get_messages_for_api()
            get_logger().log_messages(messages)

            # --- Call LLM with interruptible streaming ---
            try:
                result = self.llm.chat_with_tools_interruptible(
                    messages,
                    self._tools_schema,
                    self.abort_controller,
                    on_chunk=self._chunk_callback,
                )
            except Exception as e:
                if checkpoint_active:
                    restored = self.checkpoint_manager.rollback()
                    if restored:
                        yield {"type": "rollback", "content": {"files": restored, "reason": str(e)}}
                yield {"type": "error", "content": str(e)}
                return

            # Update context with actual token usage from API
            if result.usage and result.usage.total_tokens:
                self.context.update_actual_usage(result.usage.total_tokens)

            # 1. Reasoning
            if result.reasoning:
                self.context.add_message(
                    Message(role="assistant", content=result.reasoning, display_type="thinking")
                )
                yield {"type": "thinking", "content": result.reasoning}
                get_logger().log_reasoning(result.reasoning, stage="pre_response")

            # 2. Text response
            if result.content:
                # Add assistant message (with tool_calls metadata for API replay)
                self.context.add_message(
                    Message(
                        role="assistant",
                        content=result.content,
                        tool_calls=result.raw_tool_calls,
                        display_type="response",
                    )
                )
                get_logger().log_model_response(result.content, self.llm.model)
                yield {"type": "response", "content": result.content}
            elif result.tool_calls:
                # Assistant message with no text, only tool calls
                self.context.add_message(
                    Message(
                        role="assistant",
                        content="",
                        tool_calls=result.raw_tool_calls,
                        display_type="tool_call",
                    )
                )

            self._save_current_session()

            # 3. Tool calls
            if not result.tool_calls:
                # No tool calls — conversation turn is done
                if checkpoint_active:
                    self.checkpoint_manager.rollback()
                yield {"type": "done", "content": ""}
                return

            tool_iterations += 1
            has_file_edits = False

            for tc in result.tool_calls:
                yield {"type": "tool_call", "content": {"name": tc.name, "arguments": tc.arguments}}

                name = tc.name
                if name == "code-edit":
                    has_file_edits = True

                if name not in self.tools:
                    error_msg = (
                        f"Unknown tool: '{name}'. Available tools: {', '.join(self.tools.keys())}"
                    )
                    yield {"type": "error", "content": error_msg}
                    # Add tool error result for context
                    self.context.add_message(
                        Message(
                            role="tool",
                            content=f"ERROR - {error_msg}",
                            tool_call_id=tc.id,
                            name=name,
                            display_type="error",
                        )
                    )
                    continue

                try:
                    tool = self.tools[name]
                    args_str = json.dumps(tc.arguments)

                    # Confirm shell commands
                    if name == "command-exec":
                        _cmd_str = tc.arguments.get("command", args_str)
                        confirm_result: dict = {}
                        yield {
                            "type": "command_confirm",
                            "content": {"command": _cmd_str},
                            "result": confirm_result,
                        }
                        if not confirm_result.get("approved", False):
                            tool_result = "Command execution cancelled by user."
                            yield {
                                "type": "tool_result",
                                "content": {"name": name, "result": tool_result},
                            }
                            self.context.add_message(
                                Message(
                                    role="tool",
                                    content=tool_result,
                                    tool_call_id=tc.id,
                                    name=name,
                                    display_type="tool_result",
                                )
                            )
                            continue

                    # Execute tool (streaming for command-exec)
                    if name == "command-exec" and hasattr(tool, "execute_streaming"):
                        tool_result = ""
                        for event in tool.execute_streaming(args_str):
                            if event["type"] == "waiting_input":
                                process_to_kill = event.get("process")
                                yield {
                                    "type": "command_waiting",
                                    "content": event["content"],
                                    "process": process_to_kill,
                                    "tool_name": name,
                                }
                                if process_to_kill and process_to_kill.poll() is not None:
                                    partial = "".join(event.get("stdout", []))
                                    tool_result = (
                                        f"⚠️ INTERACTIVE PROCESS KILLED BY USER\n"
                                        f"DO NOT attempt to run this command again.\n"
                                        f"Partial output:\n{partial}"
                                    )
                                    break
                            elif event["type"] in ("done", "error"):
                                tool_result = event["content"]
                    else:
                        tool_result = tool.execute(args_str)

                    yield {"type": "tool_result", "content": {"name": name, "result": tool_result}}
                    get_logger().log_tool_call(name, args_str)
                    get_logger().log_tool_result(name, tool_result)

                    # Add tool result as role="tool" with tool_call_id
                    self.context.add_message(
                        Message(
                            role="tool",
                            content=tool_result,
                            tool_call_id=tc.id,
                            name=name,
                            display_type="tool_result",
                        )
                    )

                except Exception as e:
                    get_logger().log_error(e)
                    error_result = f"Error executing tool: {e}"
                    yield {"type": "error", "content": error_result}
                    self.context.add_message(
                        Message(
                            role="tool",
                            content=error_result,
                            tool_call_id=tc.id,
                            name=name,
                            display_type="error",
                        )
                    )
                    if checkpoint_active:
                        restored = self.checkpoint_manager.rollback()
                        if restored:
                            yield {
                                "type": "rollback",
                                "content": {"files": restored, "reason": str(e)},
                            }
                        checkpoint_active = False

            # Commit checkpoint after successful file edits
            if checkpoint_active and has_file_edits:
                self.checkpoint_manager.commit()
                checkpoint_active = False

            # Loop back — LLM will see the tool results and continue
            continue

    # ------------------------------------------------------------------
    # Deprecated: streaming with text-based tool parsing
    # ------------------------------------------------------------------

    def chat_stream(self, user_message: str):
        """Process user message and yield response chunks.

        .. deprecated::
            Use ``chat_turn()`` instead. This method uses text-based tool
            parsing which is fragile and requires per-model format templates.

        Yields:
            dict: Event dictionary with 'type' and 'content' keys.
                  Types: 'token', 'tool_call', 'tool_result', 'error', 'done', 'rollback', 'aborted'
        """
        MAX_TOOL_ITERATIONS = 50

        # Reset abort controller for new interaction
        self.abort_controller.reset()

        # Update RepoMap occasionally
        if self.repo_map:
            self._update_system_prompt()

        # Create checkpoint for this interaction (will be committed after successful tool execution)
        checkpoint_active = False
        if user_message:
            self.checkpoint_manager.create(description=user_message[:100])
            checkpoint_active = True

        # Add user message to context (only once, before the loop)
        if user_message:
            self.context.add_message(Message("user", user_message, display_type="user_input"))
            get_logger().log_user_input(user_message)

        tool_iterations = 0

        while True:
            # Guard against infinite tool-call loops
            if tool_iterations >= MAX_TOOL_ITERATIONS:
                if checkpoint_active:
                    self.checkpoint_manager.rollback()
                yield {
                    "type": "error",
                    "content": f"Tool call limit ({MAX_TOOL_ITERATIONS}) reached. Stopping to prevent infinite loop.",
                }
                return

            # Get messages for API
            messages = self.context.get_messages_for_api()
            get_logger().log_messages(messages)

            # Stream response
            response_text = ""
            reasoning_text = ""

            try:
                for chunk in self.llm.chat_stream(messages):
                    # Check for abort between chunks
                    if self.abort_controller.is_aborted:
                        raise AgentAbortedError("Agent execution aborted by user")

                    if not chunk.is_done:
                        if chunk.reasoning:
                            reasoning_text += chunk.reasoning
                            yield {"type": "reasoning", "content": chunk.reasoning}
                            get_logger().log_stream_event("reasoning", chunk.reasoning)
                        if chunk.content:
                            yield {"type": "token", "content": chunk.content}
                            response_text += chunk.content
                            if len(response_text) <= 100:
                                get_logger().log_stream_event("token", chunk.content)

                if reasoning_text:
                    get_logger().log_reasoning(reasoning_text, stage="pre_response")

            except AgentAbortedError:
                if checkpoint_active:
                    restored = self.checkpoint_manager.rollback()
                    if restored:
                        yield {
                            "type": "rollback",
                            "content": {"files": restored, "reason": "Aborted by user"},
                        }
                yield {"type": "aborted", "content": "Agent interrupted by user (ESC)"}
                return

            except Exception as e:
                get_logger().log_error(e)
                if checkpoint_active:
                    restored = self.checkpoint_manager.rollback()
                    if restored:
                        yield {"type": "rollback", "content": {"files": restored, "reason": str(e)}}
                yield {"type": "error", "content": str(e)}
                return

            # Add assistant response to context
            if response_text:
                self.context.add_message(Message("assistant", response_text))
                get_logger().log_model_response(response_text, self.llm.model)
                self._save_current_session()

            # Check for tool calls (may be multiple)
            tool_calls = self._extract_all_tool_calls(response_text)
            if tool_calls:
                tool_iterations += 1
                all_results = []
                has_file_edits = False

                for tool_call_data in tool_calls:
                    yield {"type": "tool_call", "content": tool_call_data}

                    name = tool_call_data.get("name", "")
                    args = tool_call_data.get("arguments", "")

                    # Normalize invented tool names that small models hallucinate
                    TOOL_ALIASES: dict[str, str] = {
                        "file-create": "code-edit",  # qwen3.5 invents this
                        "file-write": "code-edit",
                        "create-file": "code-edit",
                        "write-file": "code-edit",
                        "file_read": "file-read",  # underscore variants
                        "file_edit": "code-edit",
                        "code_edit": "code-edit",
                        "code_search": "code-search",
                        "run-command": "command-exec",
                        "run_command": "command-exec",
                        "execute": "command-exec",
                    }
                    name = TOOL_ALIASES.get(name, name)

                    if name == "code-edit":
                        has_file_edits = True

                    if name not in self.tools:
                        error_msg = f"Unknown tool: '{name}'. Available tools: {', '.join(self.tools.keys())}"
                        yield {"type": "error", "content": error_msg}
                        all_results.append(f"[{name}]: ERROR - {error_msg}")
                        continue

                    try:
                        tool = self.tools[name]

                        # Ask user to confirm before running any shell command
                        if name == "command-exec":
                            try:
                                import json as _json

                                _cmd_args = _json.loads(args) if isinstance(args, str) else args
                                _cmd_str = _cmd_args.get("command", str(args))
                            except Exception:
                                _cmd_str = str(args)
                            confirm_result: dict = {}
                            yield {
                                "type": "command_confirm",
                                "content": {"command": _cmd_str},
                                "result": confirm_result,
                            }
                            if not confirm_result.get("approved", False):
                                result = "Command execution cancelled by user."
                                yield {
                                    "type": "tool_result",
                                    "content": {"name": name, "result": result},
                                }
                                all_results.append(f"[{name}]: {result}")
                                continue

                        # Use streaming for command-exec to handle interactive commands
                        if name == "command-exec" and hasattr(tool, "execute_streaming"):
                            result = ""
                            process_to_kill = None

                            for event in tool.execute_streaming(args):
                                if event["type"] == "waiting_input":
                                    process_to_kill = event.get("process")
                                    yield {
                                        "type": "command_waiting",
                                        "content": event["content"],
                                        "process": process_to_kill,
                                        "tool_name": name,
                                    }
                                    if process_to_kill and process_to_kill.poll() is not None:
                                        partial_output = "".join(event.get("stdout", []))
                                        result = (
                                            f"⚠️ INTERACTIVE PROCESS KILLED BY USER\n"
                                            f"The command requires user input which cannot be provided in this environment.\n"
                                            f"DO NOT attempt to run this command again.\n\n"
                                            f"Partial output before kill:\n{partial_output}"
                                        )
                                        break
                                elif event["type"] == "output":
                                    pass
                                elif event["type"] == "done":
                                    result = event["content"]
                                elif event["type"] == "stalled":
                                    result = f"{event['content']}\n\n(waiting for command to complete...)"
                                elif event["type"] == "error":
                                    result = event["content"]
                        else:
                            result = tool.execute(args)

                        yield {"type": "tool_result", "content": {"name": name, "result": result}}
                        get_logger().log_tool_call(name or "", args)
                        get_logger().log_tool_result(name or "", result)
                        all_results.append(f"[{name}]: {result}")
                    except Exception as e:
                        get_logger().log_error(e)
                        result = f"Error executing tool: {e}"
                        yield {"type": "error", "content": result}
                        all_results.append(f"[{name}]: {result}")
                        if checkpoint_active:
                            restored = self.checkpoint_manager.rollback()
                            if restored:
                                yield {
                                    "type": "rollback",
                                    "content": {"files": restored, "reason": str(e)},
                                }
                            checkpoint_active = False

                # Commit checkpoint after successful file edits
                if checkpoint_active and has_file_edits:
                    self.checkpoint_manager.commit()
                    checkpoint_active = False

                # Add combined tool results to context and loop back for next LLM turn
                combined_results = "\n\n".join(all_results)
                self.context.add_message(
                    Message("user", f"<@TOOL_RESULT>{combined_results}</@TOOL_RESULT>")
                )
                continue  # next iteration of the while loop (replaces recursive call)

            else:
                # No tool calls parsed — but maybe the response TRIED to call a tool
                # and the JSON was malformed? Detect and retry.
                _tag_markers = ["<@TOOL>", "to=tool:", "<function_call", "<tool_call>", "```json"]
                has_tool_attempt = any(marker in response_text for marker in _tag_markers)

                if has_tool_attempt and tool_iterations < MAX_TOOL_ITERATIONS:
                    # The model tried to make a tool call but the JSON was broken.
                    # Tell the model about it and let it retry.
                    tool_iterations += 1
                    error_msg = (
                        "ERROR: Your tool call could not be parsed — the JSON was malformed. "
                        "Common issues: unescaped quotes inside strings, raw newlines instead of \\n. "
                        "Please retry the SAME tool call with properly escaped JSON."
                    )
                    yield {"type": "error", "content": error_msg}
                    self.context.add_message(
                        Message("user", f"<@TOOL_RESULT>{error_msg}</@TOOL_RESULT>")
                    )
                    continue  # retry

                # Truly no tool calls — signal completion
                if checkpoint_active:
                    self.checkpoint_manager.rollback()  # cleanup only, no files to restore

                yield {"type": "done", "content": ""}
                return

    def _extract_tool_call(self, text: str) -> dict | None:
        """Extract tool call from response text using multi-format parser."""
        result = self.tool_parser.parse(text)
        if result:
            if self.debug:
                console.print(f"[dim]Parsed tool call via {result.format_name}: {result.name}[/]")
            return result.to_dict()
        return None

    def _extract_all_tool_calls(self, text: str) -> list[dict]:
        """Extract ALL tool calls from response text using multi-format parser."""
        results = self.tool_parser.parse_all(text)
        if results:
            if self.debug:
                console.print(f"[dim]Parsed {len(results)} tool calls[/]")
            return [r.to_dict() for r in results]
        return []

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.context.clear()

    def get_context_stats(self) -> str:
        """Get current context statistics."""
        return str(self.context.get_stats())

    def set_debug(self, enabled: bool) -> None:
        """Enable or disable debug mode."""
        self.debug = enabled
        self.tool_parser.debug = enabled

    # Session management methods
    def start_new_session(self) -> None:
        """Create and activate a new session."""
        self.current_session = self.session_manager.create_new_session()

    def load_session(self, session_id: str) -> bool:
        """Load an existing session and restore context.

        Returns True if session was loaded successfully.
        """
        session = self.session_manager.load_session(session_id)
        if session:
            self.current_session = session
            # Clear existing context and restore from session
            self.context.clear()
            for msg in session.messages:
                self.context.add_message(msg)
            return True
        return False

    def handle_undo(self, restored_files: list[str]) -> None:
        """Handle undo event by updating context."""
        if not restored_files:
            return

        file_list = ", ".join(f"`{Path(f).name}`" for f in restored_files)
        message = f"[SYSTEM] Undo operation performed by user. The following files were reverted to their previous state: {file_list}. The content of these files in your context is now invalid. You must re-read them if needed."

        # Add as user message (more reliably attended to than system role mid-chat)
        self.context.add_message(Message(role="user", content=message))
        get_logger().log_system_prompt(f"Undo event: {message}")

    def _save_current_session(self) -> None:
        """Save current session state."""
        if self.current_session:
            self.current_session.messages = self.context.get_messages()
            self.session_manager.save_session(self.current_session)

    def compact_context(self) -> tuple[str, ContextStats, ContextStats]:
        """Compact the current context by summarizing it.

        This method:
        1. Gets all messages from history
        2. Asks the LLM to create a summary (emphasizing recent messages)
        3. Clears the history
        4. Injects the summary as the new starting context

        Returns:
            tuple: (summary_text, stats_before, stats_after)
        """

        # Get stats before compaction
        stats_before = self.context.get_stats()

        # Get conversation history
        messages = self.context.get_messages()
        if not messages:
            return ("No context to compact.", stats_before, stats_before)

        # Format conversation for summarization
        conversation_text = "\n\n".join(
            [f"[{msg.role.upper()}]: {msg.content}" for msg in messages]
        )

        # Build summarization prompt
        summary_prompt = CONTEXT_SUMMARY_PROMPT.format(conversation_history=conversation_text)

        # Call LLM synchronously to get summary
        summary_messages = [Message("user", summary_prompt)]

        try:
            summary = self.llm.chat(summary_messages)
        except Exception as e:
            get_logger().log_error(e)
            return (f"Error generating summary: {e}", stats_before, stats_before)

        # Clear history and set summary as initial context
        self.context.set_initial_summary(summary)

        # Update session with compacted state
        if self.current_session:
            self.session_manager.update_session_after_compact(self.current_session, summary)

        # Get stats after compaction
        stats_after = self.context.get_stats()

        return (summary, stats_before, stats_after)
