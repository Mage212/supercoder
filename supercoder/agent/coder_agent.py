"""Main coding agent with context management."""

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from rich.console import Console

from ..abort_controller import AbortController, AgentAbortedError
from ..checkpoint import CheckpointManager
from ..context.freshness import FileFreshnessTracker
from ..context.references import ContextAttachment, expand_context_references
from ..context.session_manager import ChatSession, SessionManager
from ..context.window_manager import ContextConfig, ContextStats, ContextWindowManager
from ..llm.base import BaseLLM, CompletionResult, Message, NativeToolCall
from ..logging import get_logger
from ..permissions import PermissionAction, PermissionDecision, PermissionPolicy
from ..repomap import RepoMap
from ..rules_loader import SupercoderRulesLoader
from ..tools.base import BaseTool
from ..tools.code_edit import CodeEditTool
from ..tools.code_search import CodeSearchTool
from ..tools.file_read import FileReadTool
from ..tools.glob_tool import GlobTool
from ..tools.project_structure import ProjectStructureTool
from .agent_modes import MODE_CONFIGS, READ_ONLY_TOOLS, AgentMode
from .loop_guard import AgentLoopGuard, LoopGuardDecision
from .prompts import CACHE_AWARE_COMPACT_REQUEST, build_system_prompt
from .tool_output import ToolOutputMasker
from .tool_parser import ToolCallParser

console = Console()


@dataclass
class ModeToolDecision:
    """Host-side decision for a tool call under the active agent mode."""

    allowed: bool
    reason: str
    arguments: dict | None = None


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
        permissions: dict | None = None,
        loop_detection: dict | bool | None = None,
    ):
        self.llm = llm
        self.repo_root = Path(repo_root).resolve()
        self.streaming = streaming  # False = native API (default), True = deprecated streaming
        self.lean = lean  # Shorter prompts for weak/local models
        self.output_masker = ToolOutputMasker(self.repo_root)
        self.permission_policy = PermissionPolicy(self.repo_root, permissions)
        self.loop_detection = loop_detection
        self.freshness_tracker = FileFreshnessTracker(self.repo_root)

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
                t.permission_policy = self.permission_policy
                t.freshness_tracker = self.freshness_tracker
            # Inject allowed_root into read-only path tools
            elif isinstance(t, (FileReadTool, CodeSearchTool, GlobTool, ProjectStructureTool)):
                t.allowed_root = self.repo_root
                t.permission_policy = self.permission_policy
                if isinstance(t, FileReadTool):
                    t.freshness_tracker = self.freshness_tracker
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

        # Build a stable system prompt template with all tools and project rules.
        # Mode changes are enforced host-side and announced in-band so local
        # backends can keep reusing the prompt/KV cache prefix.
        self.base_system_prompt = build_system_prompt(
            self._tools_list,
            rules=project_rules,
            tool_calling_type=self.tool_calling_type,
            native_tools=not self.streaming,
            lean=self.lean,
        )
        self._mode_policy_needs_announcement = True

        # Setup context management
        config = context_config or ContextConfig()
        self.context = ContextWindowManager(config)
        self.context.set_tools_schema(self._tools_schema)
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

    def _record_response_usage(
        self, result: CompletionResult, request_messages: list[Message]
    ) -> None:
        """Store latest context usage from an API response or fallback estimate."""
        usage = result.usage
        has_reported_usage = bool(
            usage and (usage.total_tokens or usage.prompt_tokens or usage.completion_tokens)
        )
        fallback_total = None
        if not has_reported_usage:
            fallback_total = self._estimate_response_total_tokens(request_messages, result)
        self.context.update_actual_usage(usage, fallback_total_tokens=fallback_total)

    def _estimate_response_total_tokens(
        self, request_messages: list[Message], result: CompletionResult
    ) -> int:
        """Estimate total tokens for the latest request/response pair."""
        prompt_tokens = self.context.counter.count_api_payload(
            request_messages,
            self._tools_schema,
        )
        response_payload: dict = {
            "role": "assistant",
            "content": result.content,
        }
        if result.reasoning:
            response_payload["reasoning"] = result.reasoning
        if result.raw_tool_calls is not None:
            response_payload["tool_calls"] = result.raw_tool_calls
        elif result.tool_calls:
            response_payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(
                            tc.arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                }
                for tc in result.tool_calls
            ]
        return prompt_tokens + self.context.counter.count_serialized(response_payload)

    def _expand_context_references(self, user_message: str) -> ContextAttachment | None:
        """Expand @path references into a bounded attachment message."""
        max_total_tokens = min(12_000, max(1_000, int(self.context.usable_tokens() * 0.30)))
        return expand_context_references(
            user_message,
            self.repo_root,
            permission_policy=self.permission_policy,
            freshness_tracker=self.freshness_tracker,
            max_total_tokens=max_total_tokens,
        )

    def _log_permission_decision(
        self,
        tool_name: str,
        subject: str,
        decision: PermissionDecision,
    ) -> None:
        """Log a host-side permission decision."""
        get_logger().log_permission_decision(
            tool_name=tool_name,
            subject=subject,
            action=decision.action.value,
            reason=decision.reason,
            source=decision.source,
            matched_rule=decision.matched_rule,
        )

    def _check_command_permission(self, command: str) -> PermissionDecision:
        """Evaluate and log command-exec permission."""
        decision = self.permission_policy.check_command(command)
        self._log_permission_decision("command-exec", command, decision)
        return decision

    def _log_edit_confirmation(
        self,
        arguments: dict,
        approved: bool,
        decision: str | None = None,
    ) -> None:
        """Log a host-side edit confirmation without file contents."""
        filepath = str(arguments.get("filepath") or arguments.get("fileName") or "")
        operation = str(arguments.get("operation") or "")
        get_logger().log_edit_confirmation(
            mode=self._mode.value,
            filepath=filepath,
            operation=operation,
            approved=approved,
            decision=decision,
        )

    def _preview_code_edit_for_confirmation(
        self, arguments: dict
    ) -> tuple[object | None, str | None]:
        """Prepare a code-edit preview before asking the user for approval."""
        tool = self.tools.get("code-edit")
        preview_edit = getattr(tool, "preview_edit", None)
        if not callable(preview_edit):
            return None, "Error: code-edit preview is unavailable."

        try:
            preview = preview_edit(arguments)
        except Exception as exc:
            get_logger().log_error(exc)
            return None, f"Error preparing edit preview: {exc}"

        if not getattr(preview, "ok", False):
            error = str(getattr(preview, "error", "") or "Error preparing edit preview")
            return None, error

        if not str(getattr(preview, "diff", "")):
            return None, "Error: Edit preview produced no changes."

        return preview, None

    def _add_command_permission_rule(
        self,
        command: str,
        action: PermissionAction,
        scope: str,
        *,
        source: str,
    ) -> tuple[bool, str | None]:
        """Persist a command permission rule and log the change."""
        try:
            rule = self.permission_policy.add_command_rule(action, command, scope=scope)
        except Exception as exc:
            get_logger().log_error(exc)
            return False, f"Error saving permission rule: {exc}"

        get_logger().log_permission_rule_change(
            action="add",
            scope=rule.scope,
            rule_action=rule.action.value,
            rule=rule.pattern,
            source=source,
        )
        return True, None

    def _apply_command_confirmation(
        self,
        command: str,
        confirm_result: dict,
    ) -> tuple[bool, str]:
        """Apply a command confirmation result from the REPL."""
        decision = confirm_result.get("decision")
        if decision is None:
            return (
                bool(confirm_result.get("approved", False)),
                "Command execution cancelled by user.",
            )

        if decision == "approve_once":
            return True, ""
        if decision == "allow_session":
            ok, error = self._add_command_permission_rule(
                command,
                PermissionAction.ALLOW,
                "session",
                source="command_confirm",
            )
            return ok, error or ""
        if decision == "allow_persistent":
            ok, error = self._add_command_permission_rule(
                command,
                PermissionAction.ALLOW,
                "persistent",
                source="command_confirm",
            )
            return ok, error or ""
        if decision == "deny_persistent":
            ok, error = self._add_command_permission_rule(
                command,
                PermissionAction.DENY,
                "persistent",
                source="command_confirm",
            )
            if not ok:
                return False, error or "Error saving permission rule."
            return False, "Command execution denied by user and saved as a persistent deny rule."

        return False, "Command execution cancelled by user."

    def _log_mode_tool_decision(
        self,
        tool_name: str,
        decision: ModeToolDecision,
        arguments: dict | None = None,
    ) -> None:
        """Log a host-side mode policy decision."""
        get_logger().log_mode_policy(
            mode=self._mode.value,
            tool_name=tool_name,
            action="allow" if decision.allowed else "deny",
            reason=decision.reason,
            subject=self._mode_subject(tool_name, arguments),
        )

    def _mode_subject(self, tool_name: str, arguments: dict | None = None) -> str:
        if tool_name == "code-edit" and arguments:
            value = arguments.get("filepath") or arguments.get("fileName") or ""
            return str(value)
        if tool_name == "command-exec" and arguments:
            return str(arguments.get("command", ""))
        return tool_name

    def _log_early_tool_outcome(self, tool_name: str, arguments: str, result: str) -> None:
        """Log a tool request that resolved before normal tool execution."""
        logger = get_logger()
        logger.log_tool_call(tool_name, arguments)
        logger.log_tool_result(tool_name, result)

    def _looks_like_text_tool_attempt(self, text: str) -> bool:
        """Return True when text appears to contain a non-native tool call."""
        if any(marker in text for marker in ("<@TOOL", "<tool_call", "<function_call")):
            return True

        if re.search(r"\bto=(?:tool[:\s.]|TOOL\s+)[a-zA-Z0-9_-]+\s*\{", text, re.IGNORECASE):
            return True

        for match in re.finditer(r"```json\s*\n(.*?)\n?```", text, re.DOTALL | re.IGNORECASE):
            block = match.group(1)
            has_tool_name = re.search(r"""["']?(?:tool|name)["']?\s*:""", block)
            has_arguments = re.search(r"""["']?(?:arguments|args)["']?\s*:""", block)
            if has_tool_name and has_arguments:
                return True

        return False

    def _fallback_tool_call_arguments(self, arguments: object) -> dict:
        """Normalize fallback parser arguments into a native tool-call dict."""
        if isinstance(arguments, dict):
            return dict(arguments)

        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {"_raw": arguments}
            return parsed if isinstance(parsed, dict) else {"_raw": arguments}

        return {"_raw": str(arguments)}

    def _strip_fallback_tool_markup(self, text: str, raw_matches: list[str]) -> str:
        """Remove parsed textual tool-call markup from display content."""
        cleaned = text
        for raw_match in raw_matches:
            if raw_match:
                cleaned = cleaned.replace(raw_match, "")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _parse_text_tool_calls(
        self, text: str, iteration: int
    ) -> tuple[list[NativeToolCall], list[dict], str, list[str]]:
        """Parse textual tool calls into native-compatible tool call objects."""
        parsed_calls = self.tool_parser.parse_all(text)
        native_calls: list[NativeToolCall] = []
        raw_tool_calls: list[dict] = []
        raw_matches: list[str] = []
        formats: list[str] = []

        for index, parsed in enumerate(parsed_calls):
            name = str(parsed.name or "")
            if not name:
                continue

            arguments = self._fallback_tool_call_arguments(parsed.arguments)
            call_id = f"fallback_call_{iteration}_{index}"
            args_json = json.dumps(arguments)

            native_calls.append(NativeToolCall(id=call_id, name=name, arguments=arguments))
            raw_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args_json},
                }
            )
            raw_matches.append(parsed.raw_match)
            formats.append(parsed.format_name)

        cleaned_content = self._strip_fallback_tool_markup(text, raw_matches)
        return native_calls, raw_tool_calls, cleaned_content, formats

    def _tool_call_retry_instruction(self, attempt: int, max_attempts: int) -> str:
        """Short cache-friendly correction appended only to the retry API call."""
        return (
            "Your previous response looked like a tool call, but SuperCoder could not parse it. "
            f"Retry the same step now using the native tool call interface only "
            f"(retry {attempt}/{max_attempts}). Do not write XML, JSON, or markdown tool-call "
            "syntax in normal text."
        )

    def _add_context_attachment(self, attachment: ContextAttachment) -> dict:
        """Add expanded @path context to history and return an event payload."""
        self.context.add_message(
            Message("user", attachment.content, display_type="context_attachment")
        )
        payload = attachment.to_log_dict()
        get_logger().log_context_attachment(payload)
        return payload

    def _announce_mode_policy_if_needed(self) -> None:
        """Add the current mode policy to history once after a mode change."""
        if not self._mode_policy_needs_announcement:
            return
        mode_config = MODE_CONFIGS[self._mode]
        content = f"[Mode Policy]\n{mode_config.instruction}"
        self.context.add_message(Message("user", content, display_type="mode_policy"))
        get_logger().log_mode_policy(
            mode=self._mode.value,
            tool_name="mode",
            action="announce",
            reason=mode_config.instruction,
            subject=self._mode.value,
        )
        self._mode_policy_needs_announcement = False

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
        """Switch agent mode without changing the stable system prompt."""
        if mode == self._mode:
            return

        self._mode = mode
        self._mode_policy_needs_announcement = True

    def set_tool_calling_type(self, tool_calling_type: str) -> None:
        """Update tool calling type and rebuild system prompt.

        Call this when switching to a model with a different tool_calling_type.
        """
        if tool_calling_type != self.tool_calling_type:
            self.tool_calling_type = tool_calling_type
            self.base_system_prompt = build_system_prompt(
                self._tools_list,
                rules=self._project_rules,
                tool_calling_type=self.tool_calling_type,
                native_tools=not self.streaming,
                lean=self.lean,
            )
            self._update_system_prompt()

    def _check_mode_tool(self, tool_name: str, arguments: dict | None) -> ModeToolDecision:
        """Return the host-side mode decision for a requested tool call."""
        args = arguments or {}

        if self._mode == AgentMode.ASK:
            if tool_name in READ_ONLY_TOOLS:
                decision = ModeToolDecision(True, "Tool allowed in ASK mode", args)
            else:
                decision = ModeToolDecision(
                    False,
                    f"{tool_name} is not allowed in ASK mode. ASK mode is read-only.",
                    args,
                )
            self._log_mode_tool_decision(tool_name, decision, args)
            return decision

        if self._mode == AgentMode.PLAN:
            if tool_name in READ_ONLY_TOOLS:
                decision = ModeToolDecision(True, "Tool allowed in PLAN mode", args)
            elif tool_name == "code-edit":
                decision = self._prepare_plan_edit(args)
            else:
                decision = ModeToolDecision(
                    False,
                    f"{tool_name} is not allowed in PLAN mode. "
                    "PLAN mode blocks shell commands and project file edits.",
                    args,
                )
            self._log_mode_tool_decision(tool_name, decision, decision.arguments or args)
            return decision

        decision = ModeToolDecision(True, f"Tool allowed in {self._mode.value} mode", args)
        self._log_mode_tool_decision(tool_name, decision, args)
        return decision

    def _prepare_plan_edit(self, arguments: dict) -> ModeToolDecision:
        """Normalize and validate the narrow PLAN-mode code-edit exception."""
        updated = dict(arguments)
        raw_path = str(updated.get("filepath") or updated.get("fileName") or "").strip()
        operation = str(updated.get("operation") or "search_replace")

        plan_dir = self.repo_root / ".supercoder" / "plans"
        plan_dir_resolved = plan_dir.resolve()
        try:
            plan_dir_resolved.relative_to(self.repo_root)
        except ValueError:
            return ModeToolDecision(
                False,
                "PLAN mode cannot write plans because .supercoder/plans resolves outside the project.",
                updated,
            )

        if raw_path:
            path = Path(raw_path)
            if path.is_absolute():
                candidate = path.resolve()
                try:
                    candidate.relative_to(plan_dir_resolved)
                except ValueError:
                    return ModeToolDecision(
                        False,
                        "PLAN mode can only edit plan files under .supercoder/plans/.",
                        updated,
                    )
            else:
                normalized_parts = path.parts
                if normalized_parts[:2] == (".supercoder", "plans"):
                    candidate = (self.repo_root / path).resolve()
                    try:
                        candidate.relative_to(plan_dir_resolved)
                    except ValueError:
                        return ModeToolDecision(
                            False,
                            "PLAN mode blocked a plan path that escapes .supercoder/plans/.",
                            updated,
                        )
                elif len(normalized_parts) == 1:
                    candidate = (plan_dir_resolved / normalized_parts[0]).resolve()
                else:
                    return ModeToolDecision(
                        False,
                        "PLAN mode cannot edit project files. Save plans under .supercoder/plans/.",
                        updated,
                    )
        else:
            candidate = plan_dir_resolved / "plan.md"

        candidate = self._date_prefixed_plan_path(candidate, plan_dir_resolved, operation)
        updated["filepath"] = candidate.relative_to(self.repo_root).as_posix()
        updated.pop("fileName", None)

        return ModeToolDecision(
            True,
            "PLAN mode allows code-edit only for dated plan files under .supercoder/plans/.",
            updated,
        )

    def _date_prefixed_plan_path(
        self,
        candidate: Path,
        plan_dir: Path,
        operation: str,
    ) -> Path:
        """Return a date-prefixed plan path, adding create suffixes on collisions."""
        name = candidate.name.strip() or "plan.md"
        if name in {".", ".."}:
            name = "plan.md"
        if not Path(name).suffix:
            name = f"{name}.md"

        today = date.today().isoformat()
        if not re.match(r"^\d{4}-\d{2}-\d{2}-", name):
            name = f"{today}-{name}"

        normalized = candidate.with_name(name).resolve()
        try:
            normalized.relative_to(plan_dir)
        except ValueError:
            normalized = (plan_dir / name).resolve()

        if operation != "create" or not normalized.exists():
            return normalized

        stem = normalized.stem
        suffix = normalized.suffix
        parent = normalized.parent
        idx = 2
        while True:
            deduped = parent / f"{stem}-{idx}{suffix}"
            if not deduped.exists():
                return deduped.resolve()
            idx += 1

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
          - ``{"type": "edit_confirm", ...}``                → confirm file edit
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
        has_file_edits = False
        if user_message:
            self.checkpoint_manager.create(description=user_message[:100])
            checkpoint_active = True
            self._announce_mode_policy_if_needed()
            attachment = self._expand_context_references(user_message)
            if attachment:
                yield {
                    "type": "context_attachment",
                    "content": self._add_context_attachment(attachment),
                }
            self.context.add_message(Message("user", user_message, display_type="user_input"))
            get_logger().log_user_input(user_message)

        # Update RepoMap if enabled
        if self.repo_map:
            self._update_system_prompt()

        tool_iterations = 0
        malformed_tool_retries = 0
        max_malformed_tool_retries = 2
        retry_messages: list[Message] = []
        loop_guard = AgentLoopGuard.from_config(self.loop_detection)

        while True:
            if tool_iterations >= MAX_TOOL_ITERATIONS:
                if checkpoint_active:
                    self.checkpoint_manager.rollback()
                yield {
                    "type": "error",
                    "content": f"Tool call limit ({MAX_TOOL_ITERATIONS}) reached. Stopping to prevent infinite loop.",
                }
                return

            self._announce_mode_policy_if_needed()
            messages = self.context.get_messages_for_api()
            if retry_messages:
                messages = [*messages, *retry_messages]
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
                    rollback_result = self.checkpoint_manager.rollback()
                    if rollback_result:
                        yield {
                            "type": "rollback",
                            "content": {
                                "restored": rollback_result.restored,
                                "failed": rollback_result.failed,
                                "reason": str(e),
                            },
                        }
                yield {"type": "error", "content": str(e)}
                return

            # Update context from the latest model response. Auto-compact decisions
            # are made after this response, not before the next request is built.
            self._record_response_usage(result, messages)

            # Warn about truncated responses
            if result.truncated:
                yield {
                    "type": "warning",
                    "content": "Response was truncated — tool call arguments may be incomplete. "
                    "Tool calls from this response will not be executed until retried.",
                }

            # 1. Reasoning
            if result.reasoning:
                self.context.add_message(
                    Message(role="assistant", content=result.reasoning, display_type="thinking")
                )
                yield {"type": "thinking", "content": result.reasoning}
                get_logger().log_reasoning(result.reasoning, stage="pre_response")

            effective_tool_calls = result.tool_calls
            effective_raw_tool_calls = result.raw_tool_calls
            display_content = result.content
            assistant_loop_decision: LoopGuardDecision | None = None

            if not effective_tool_calls and result.content:
                (
                    fallback_tool_calls,
                    fallback_raw_tool_calls,
                    fallback_display_content,
                    fallback_formats,
                ) = self._parse_text_tool_calls(result.content, tool_iterations)
                if fallback_tool_calls:
                    effective_tool_calls = fallback_tool_calls
                    effective_raw_tool_calls = fallback_raw_tool_calls
                    display_content = fallback_display_content
                    malformed_tool_retries = 0
                    retry_messages = []
                    get_logger().log_tool_call_fallback_parse(
                        success=True,
                        count=len(fallback_tool_calls),
                        formats=fallback_formats,
                        reason="parsed_text_tool_calls",
                    )
                elif self._looks_like_text_tool_attempt(result.content):
                    get_logger().log_tool_call_fallback_parse(
                        success=False,
                        count=0,
                        reason="malformed_text_tool_call",
                    )
                    if malformed_tool_retries < max_malformed_tool_retries:
                        malformed_tool_retries += 1
                        retry_reason = "Malformed text tool-call output"
                        get_logger().log_tool_call_retry(
                            attempt=malformed_tool_retries,
                            max_attempts=max_malformed_tool_retries,
                            reason=retry_reason,
                        )
                        yield {
                            "type": "tool_retry",
                            "content": {
                                "attempt": malformed_tool_retries,
                                "max_attempts": max_malformed_tool_retries,
                                "reason": retry_reason,
                            },
                        }
                        retry_messages = [
                            Message(
                                "user",
                                self._tool_call_retry_instruction(
                                    malformed_tool_retries, max_malformed_tool_retries
                                ),
                                display_type="tool_retry",
                            )
                        ]
                        continue

                    if checkpoint_active:
                        rollback_result = self.checkpoint_manager.rollback()
                        if rollback_result:
                            yield {
                                "type": "rollback",
                                "content": {
                                    "restored": rollback_result.restored,
                                    "failed": rollback_result.failed,
                                    "reason": "Malformed tool call after retries",
                                },
                            }
                    yield {
                        "type": "error",
                        "content": "Tool call format was invalid after 2 retries. Stopping.",
                    }
                    return

            if effective_tool_calls:
                malformed_tool_retries = 0
            retry_messages = []
            if effective_tool_calls and not result.truncated:
                assistant_loop_decision = loop_guard.observe_assistant(
                    display_content,
                    effective_tool_calls,
                )

            # 2. Text response
            if display_content:
                # Add assistant message (with tool_calls metadata for API replay)
                self.context.add_message(
                    Message(
                        role="assistant",
                        content=display_content,
                        tool_calls=effective_raw_tool_calls,
                        display_type="response",
                    )
                )
                get_logger().log_model_response(display_content, self.llm.model)
                yield {"type": "response", "content": display_content}
            elif effective_tool_calls:
                # Assistant message with no text, only tool calls
                self.context.add_message(
                    Message(
                        role="assistant",
                        content="",
                        tool_calls=effective_raw_tool_calls,
                        display_type="tool_call",
                    )
                )

            self._save_current_session()

            if result.truncated and effective_tool_calls:
                retry_messages = [
                    Message(
                        "user",
                        "The previous response was truncated while producing tool calls. "
                        "SuperCoder did not execute those tool calls because their arguments "
                        "may be incomplete. Retry the same step with complete native tool calls.",
                        display_type="tool_retry",
                    )
                ]
                for tc in effective_tool_calls:
                    name = tc.name
                    arguments = dict(tc.arguments)
                    tool_result = (
                        "Tool call was not executed because the model response was truncated. "
                        "Retry with a complete native tool call."
                    )
                    yield {"type": "tool_call", "content": {"name": name, "arguments": arguments}}
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
                self._save_current_session()
                auto_compact_event = self._auto_compact_if_needed()
                if auto_compact_event:
                    yield auto_compact_event
                continue

            if assistant_loop_decision:
                for tc in effective_tool_calls:
                    name = tc.name
                    arguments = dict(tc.arguments)
                    tool_result = assistant_loop_decision.message
                    yield {"type": "tool_call", "content": {"name": name, "arguments": arguments}}
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
                self._save_current_session()
                if assistant_loop_decision.stop:
                    if checkpoint_active:
                        rollback_result = self.checkpoint_manager.rollback()
                        if rollback_result:
                            yield {
                                "type": "rollback",
                                "content": {
                                    "restored": rollback_result.restored,
                                    "failed": rollback_result.failed,
                                    "reason": assistant_loop_decision.reason,
                                },
                            }
                    yield {"type": "error", "content": assistant_loop_decision.message}
                    return
                auto_compact_event = self._auto_compact_if_needed()
                if auto_compact_event:
                    yield auto_compact_event
                continue

            # 3. Tool calls
            if not effective_tool_calls:
                # No tool calls — conversation turn is done
                if checkpoint_active:
                    if has_file_edits:
                        # Preserve successful edits instead of rolling them back.
                        self.checkpoint_manager.commit()
                    else:
                        # Cleanup empty checkpoint (no files to keep).
                        self.checkpoint_manager.rollback()
                    checkpoint_active = False
                auto_compact_event = self._auto_compact_if_needed()
                if auto_compact_event:
                    yield auto_compact_event
                yield {"type": "done", "content": ""}
                return

            tool_iterations += 1
            has_file_edits = False
            pending_loop_decision: LoopGuardDecision | None = None

            def remember_loop_result(tool_name: str, tool_arguments: dict, tool_text: str) -> None:
                nonlocal pending_loop_decision
                decision = loop_guard.observe_tool_result(
                    tool_name,
                    tool_arguments,
                    tool_text,
                )
                if decision and pending_loop_decision is None:
                    pending_loop_decision = decision

            for call_index, tc in enumerate(effective_tool_calls):
                name = tc.name
                arguments = dict(tc.arguments)
                loop_decision = loop_guard.observe_tool_call(name, arguments)
                if loop_decision:
                    tool_result = loop_decision.message
                    yield {"type": "tool_call", "content": {"name": name, "arguments": arguments}}
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
                    if loop_decision.stop:
                        for remaining_tc in effective_tool_calls[call_index + 1 :]:
                            remaining_name = remaining_tc.name
                            remaining_arguments = dict(remaining_tc.arguments)
                            remaining_result = (
                                "Skipped because loop detection stopped this turn before "
                                "executing remaining tool calls."
                            )
                            yield {
                                "type": "tool_call",
                                "content": {
                                    "name": remaining_name,
                                    "arguments": remaining_arguments,
                                },
                            }
                            yield {
                                "type": "tool_result",
                                "content": {
                                    "name": remaining_name,
                                    "result": remaining_result,
                                },
                            }
                            self.context.add_message(
                                Message(
                                    role="tool",
                                    content=remaining_result,
                                    tool_call_id=remaining_tc.id,
                                    name=remaining_name,
                                    display_type="tool_result",
                                )
                            )
                        self._save_current_session()
                        if checkpoint_active:
                            rollback_result = self.checkpoint_manager.rollback()
                            if rollback_result:
                                yield {
                                    "type": "rollback",
                                    "content": {
                                        "restored": rollback_result.restored,
                                        "failed": rollback_result.failed,
                                        "reason": loop_decision.reason,
                                    },
                                }
                        yield {"type": "error", "content": loop_decision.message}
                        return
                    continue

                if name not in self.tools:
                    yield {
                        "type": "tool_call",
                        "content": {"name": name, "arguments": arguments},
                    }
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
                    remember_loop_result(name, arguments, error_msg)
                    continue

                mode_decision = self._check_mode_tool(name, arguments)
                arguments = mode_decision.arguments or arguments
                yield {"type": "tool_call", "content": {"name": name, "arguments": arguments}}

                if not mode_decision.allowed:
                    tool_result = f"Error: {mode_decision.reason}"
                    args_str = json.dumps(arguments)
                    self._log_early_tool_outcome(name, args_str, tool_result)
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
                    remember_loop_result(name, arguments, tool_result)
                    continue

                if name == "code-edit" and self._mode == AgentMode.CODE:
                    preview, preview_error = self._preview_code_edit_for_confirmation(arguments)
                    if preview_error:
                        tool_result = preview_error
                        args_str = json.dumps(arguments)
                        self._log_early_tool_outcome(name, args_str, tool_result)
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
                        remember_loop_result(name, arguments, tool_result)
                        continue

                    confirm_result: dict = {}
                    yield {
                        "type": "edit_confirm",
                        "content": {"arguments": arguments, "preview": preview},
                        "result": confirm_result,
                    }
                    approved = bool(confirm_result.get("approved", False))
                    confirm_decision = str(confirm_result.get("decision") or "")
                    self._log_edit_confirmation(arguments, approved, confirm_decision or None)
                    if approved and confirm_decision == "apply_and_accept_edits":
                        self.set_mode(AgentMode.ACCEPT_EDITS)
                    if not approved:
                        tool_result = "File edit cancelled by user."
                        args_str = json.dumps(arguments)
                        self._log_early_tool_outcome(name, args_str, tool_result)
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
                        remember_loop_result(name, arguments, tool_result)
                        continue

                if name == "code-edit":
                    has_file_edits = True

                try:
                    tool = self.tools[name]
                    args_str = json.dumps(arguments)

                    # Confirm shell commands
                    if name == "command-exec":
                        _cmd_str = arguments.get("command", args_str)
                        decision = self._check_command_permission(_cmd_str)
                        if decision.action == PermissionAction.DENY:
                            tool_result = self.permission_policy.format_denial(
                                f"command '{_cmd_str}'", decision
                            )
                            self._log_early_tool_outcome(name, args_str, tool_result)
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
                            remember_loop_result(name, arguments, tool_result)
                            continue
                        if decision.action == PermissionAction.ASK:
                            confirm_result: dict = {}
                            yield {
                                "type": "command_confirm",
                                "content": {"command": _cmd_str},
                                "result": confirm_result,
                            }
                            approved, denial_reason = self._apply_command_confirmation(
                                _cmd_str, confirm_result
                            )
                            if not approved:
                                tool_result = (
                                    denial_reason or "Command execution cancelled by user."
                                )
                                self._log_early_tool_outcome(name, args_str, tool_result)
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
                                remember_loop_result(name, arguments, tool_result)
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

                    masked_result = self.output_masker.mask(name, tc.id, tool_result)
                    self._log_tool_output_masking(name, tc.id, masked_result)
                    offload_path = None
                    if masked_result.offload_path:
                        try:
                            offload_path = masked_result.offload_path.relative_to(
                                self.repo_root
                            ).as_posix()
                        except ValueError:
                            offload_path = masked_result.offload_path.as_posix()
                    yield {
                        "type": "tool_result",
                        "content": {
                            "name": name,
                            "result": masked_result.model_text,
                            "display_result": masked_result.display_text,
                            "display_summary": self._tool_display_summary(
                                name,
                                arguments,
                                masked_result.model_text,
                                "success",
                                masked=masked_result.masked,
                                original_size=masked_result.original_size,
                                offload_path=offload_path,
                            ),
                            "display_policy": self._tool_display_policy(
                                name, masked_result.model_text, masked_result.masked, "success"
                            ),
                            "masked": masked_result.masked,
                            "offload_path": offload_path,
                            "original_size": masked_result.original_size,
                            "omitted_chars": masked_result.omitted_chars,
                        },
                    }
                    get_logger().log_tool_call(name, args_str)
                    get_logger().log_tool_result(name, tool_result)

                    # Add tool result as role="tool" with tool_call_id
                    self.context.add_message(
                        Message(
                            role="tool",
                            content=masked_result.model_text,
                            tool_call_id=tc.id,
                            name=name,
                            display_type="tool_result",
                            display_summary=self._tool_display_summary(
                                name,
                                arguments,
                                masked_result.model_text,
                                "success",
                                masked=masked_result.masked,
                                original_size=masked_result.original_size,
                                offload_path=offload_path,
                            ),
                            display_result=masked_result.display_text,
                            display_policy=self._tool_display_policy(
                                name, masked_result.model_text, masked_result.masked, "success"
                            ),
                            display_meta={
                                "tool_name": name,
                                "tool_call_id": tc.id,
                                "status": "success",
                                "masked": masked_result.masked,
                                "offload_path": offload_path,
                                "original_size": masked_result.original_size,
                                "omitted_chars": masked_result.omitted_chars,
                                "truncation_kind": "offloaded"
                                if masked_result.masked
                                else "inline",
                            },
                        )
                    )
                    remember_loop_result(name, arguments, masked_result.model_text)

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
                            display_summary=f"{name} failed",
                            display_result=error_result,
                            display_policy="error",
                            display_meta={
                                "tool_name": name,
                                "tool_call_id": tc.id,
                                "status": "error",
                                "truncation_kind": "inline",
                            },
                        )
                    )
                    remember_loop_result(name, arguments, error_result)
                    if checkpoint_active and has_file_edits:
                        rollback_result = self.checkpoint_manager.rollback()
                        if rollback_result:
                            yield {
                                "type": "rollback",
                                "content": {
                                    "restored": rollback_result.restored,
                                    "failed": rollback_result.failed,
                                    "reason": str(e),
                                },
                            }
                        checkpoint_active = False

            if pending_loop_decision:
                if pending_loop_decision.stop:
                    if checkpoint_active:
                        rollback_result = self.checkpoint_manager.rollback()
                        if rollback_result:
                            yield {
                                "type": "rollback",
                                "content": {
                                    "restored": rollback_result.restored,
                                    "failed": rollback_result.failed,
                                    "reason": pending_loop_decision.reason,
                                },
                            }
                    yield {"type": "error", "content": pending_loop_decision.message}
                    return
                retry_messages = [
                    Message(
                        "user",
                        pending_loop_decision.message,
                        display_type="tool_retry",
                    )
                ]
                yield {"type": "warning", "content": pending_loop_decision.message}

            # Commit checkpoint after successful file edits, then start a fresh
            # checkpoint so subsequent edits in the same turn stay protected.
            if checkpoint_active and has_file_edits:
                self.checkpoint_manager.commit()
                self.checkpoint_manager.create(description="continued edits")

            auto_compact_event = self._auto_compact_if_needed()
            if auto_compact_event:
                yield auto_compact_event

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
            self._announce_mode_policy_if_needed()
            attachment = self._expand_context_references(user_message)
            if attachment:
                yield {
                    "type": "context_attachment",
                    "content": self._add_context_attachment(attachment),
                }
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
            self._announce_mode_policy_if_needed()
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
                    rollback_result = self.checkpoint_manager.rollback()
                    if rollback_result:
                        yield {
                            "type": "rollback",
                            "content": {
                                "restored": rollback_result.restored,
                                "failed": rollback_result.failed,
                                "reason": "Aborted by user",
                            },
                        }
                yield {"type": "aborted", "content": "Agent interrupted by user (ESC)"}
                return

            except Exception as e:
                get_logger().log_error(e)
                if checkpoint_active:
                    rollback_result = self.checkpoint_manager.rollback()
                    if rollback_result:
                        yield {
                            "type": "rollback",
                            "content": {
                                "restored": rollback_result.restored,
                                "failed": rollback_result.failed,
                                "reason": str(e),
                            },
                        }
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

                    name = str(tool_call_data.get("name") or "")
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

                    if name not in self.tools:
                        error_msg = f"Unknown tool: '{name}'. Available tools: {', '.join(self.tools.keys())}"
                        yield {"type": "error", "content": error_msg}
                        all_results.append(f"[{name}]: ERROR - {error_msg}")
                        continue

                    try:
                        tool = self.tools[name]
                        try:
                            parsed_args = json.loads(args) if isinstance(args, str) else dict(args)
                        except Exception:
                            parsed_args = {}

                        mode_decision = self._check_mode_tool(name, parsed_args)
                        if mode_decision.arguments is not None:
                            parsed_args = mode_decision.arguments
                            args = json.dumps(parsed_args)
                        if not mode_decision.allowed:
                            result = f"Error: {mode_decision.reason}"
                            self._log_early_tool_outcome(name, args, result)
                            yield {
                                "type": "tool_result",
                                "content": {"name": name, "result": result},
                            }
                            all_results.append(f"[{name}]: {result}")
                            continue

                        if name == "code-edit" and self._mode == AgentMode.CODE:
                            preview, preview_error = self._preview_code_edit_for_confirmation(
                                parsed_args
                            )
                            if preview_error:
                                result = preview_error
                                self._log_early_tool_outcome(name, args, result)
                                yield {
                                    "type": "tool_result",
                                    "content": {"name": name, "result": result},
                                }
                                all_results.append(f"[{name}]: {result}")
                                continue

                            confirm_result: dict = {}
                            yield {
                                "type": "edit_confirm",
                                "content": {"arguments": parsed_args, "preview": preview},
                                "result": confirm_result,
                            }
                            approved = bool(confirm_result.get("approved", False))
                            confirm_decision = str(confirm_result.get("decision") or "")
                            self._log_edit_confirmation(
                                parsed_args, approved, confirm_decision or None
                            )
                            if approved and confirm_decision == "apply_and_accept_edits":
                                self.set_mode(AgentMode.ACCEPT_EDITS)
                            if not approved:
                                result = "File edit cancelled by user."
                                self._log_early_tool_outcome(name, args, result)
                                yield {
                                    "type": "tool_result",
                                    "content": {"name": name, "result": result},
                                }
                                all_results.append(f"[{name}]: {result}")
                                continue

                        if name == "code-edit":
                            has_file_edits = True

                        # Ask user to confirm before running any shell command
                        if name == "command-exec":
                            try:
                                _cmd_str = parsed_args.get("command", str(args))
                            except Exception:
                                _cmd_str = str(args)
                            decision = self._check_command_permission(_cmd_str)
                            if decision.action == PermissionAction.DENY:
                                result = self.permission_policy.format_denial(
                                    f"command '{_cmd_str}'", decision
                                )
                                self._log_early_tool_outcome(name, args, result)
                                yield {
                                    "type": "tool_result",
                                    "content": {"name": name, "result": result},
                                }
                                all_results.append(f"[{name}]: {result}")
                                continue
                            if decision.action == PermissionAction.ASK:
                                confirm_result: dict = {}
                                yield {
                                    "type": "command_confirm",
                                    "content": {"command": _cmd_str},
                                    "result": confirm_result,
                                }
                                approved, denial_reason = self._apply_command_confirmation(
                                    _cmd_str, confirm_result
                                )
                                if not approved:
                                    result = denial_reason or "Command execution cancelled by user."
                                    self._log_early_tool_outcome(name, args, result)
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

                        masked_result = self.output_masker.mask(name or "", None, result)
                        self._log_tool_output_masking(name or "", None, masked_result)
                        yield {
                            "type": "tool_result",
                            "content": {"name": name, "result": masked_result.model_text},
                        }
                        get_logger().log_tool_call(name or "", args)
                        get_logger().log_tool_result(name or "", result)
                        all_results.append(f"[{name}]: {masked_result.model_text}")
                    except Exception as e:
                        get_logger().log_error(e)
                        result = f"Error executing tool: {e}"
                        yield {"type": "error", "content": result}
                        all_results.append(f"[{name}]: {result}")
                        if checkpoint_active:
                            rollback_result = self.checkpoint_manager.rollback()
                            if rollback_result:
                                yield {
                                    "type": "rollback",
                                    "content": {
                                        "restored": rollback_result.restored,
                                        "failed": rollback_result.failed,
                                        "reason": str(e),
                                    },
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

    def _log_tool_output_masking(self, tool_name: str, tool_call_id: str | None, masked_result):
        """Log structured metadata for tool output context preparation."""
        offload_path = masked_result.offload_path.as_posix() if masked_result.offload_path else None
        get_logger().log_tool_output_masked(
            tool_name,
            tool_call_id,
            masked_result.masked,
            len(masked_result.full_text),
            len(masked_result.model_text),
            offload_path,
        )

    def _tool_display_summary(
        self,
        tool_name: str,
        arguments: dict,
        result: str,
        status: str,
        masked: bool = False,
        original_size: int = 0,
        offload_path: str | None = None,
    ) -> str:
        """Build a compact, deterministic UI summary for a tool result."""
        if tool_name == "file-read":
            path = self._tool_argument_path(arguments)
            line_count = len(result.splitlines()) if result else 0
            target = f" {path}" if path else ""
            summary = f"{tool_name}{target} · {line_count} lines"
        elif tool_name == "code-search":
            query = arguments.get("query") or arguments.get("pattern")
            match_count = len([line for line in result.splitlines() if line.strip()])
            target = f" {query}" if query else ""
            summary = f"{tool_name}{target} · {match_count} matches"
        elif tool_name == "code-edit":
            path = self._tool_argument_path(arguments)
            target = f" {path}" if path else ""
            summary = f"{tool_name}{target} · changes prepared"
        elif tool_name == "command-exec":
            command = arguments.get("command") or arguments.get("cmd")
            first_line = str(command).splitlines()[0][:80] if command else ""
            summary = f"{tool_name} {first_line}".strip()
        else:
            summary = f"{tool_name} {status}"

        if not masked:
            return summary

        parts = [summary]
        if original_size > 0:
            parts.append(f"{original_size:,} chars")
        if offload_path:
            parts.append(f"saved to {offload_path}")
        return " · ".join(parts)

    def _tool_argument_path(self, arguments: dict) -> str | None:
        """Return a path-like argument from known tool schema aliases."""
        for key in ("fileName", "filename", "filepath", "file_path", "path", "file"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _tool_display_policy(self, tool_name: str, result: str, masked: bool, status: str) -> str:
        """Choose the default UI visibility policy for a tool result."""
        if status == "error":
            return "error"
        if tool_name == "code-edit" and "--- " in result and "+++ " in result:
            return "expanded"
        return "compact"

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
            session.messages = self._repair_incomplete_tool_exchanges(session.messages)
            # Clear existing context and restore from session
            self.context.clear()
            for msg in session.messages:
                self.context.add_message(msg)
            return True
        return False

    def _repair_incomplete_tool_exchanges(self, messages: list[Message]) -> list[Message]:
        """Remove API-invalid assistant/tool pairs left by interrupted sessions."""
        tool_result_ids = {m.tool_call_id for m in messages if m.role == "tool" and m.tool_call_id}
        incomplete_call_ids: set[str] = set()
        repaired: list[Message] = []
        inserted_warning = False

        for msg in messages:
            if msg.role == "assistant" and msg.tool_calls:
                call_ids: list[str] = []
                for tc in msg.tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    call_id = tc.get("id")
                    if isinstance(call_id, str):
                        call_ids.append(call_id)
                if call_ids and any(call_id not in tool_result_ids for call_id in call_ids):
                    incomplete_call_ids.update(call_ids)
                    repaired.append(
                        Message(
                            role=msg.role,
                            content=msg.content,
                            display_type=msg.display_type,
                            display_summary=msg.display_summary,
                            display_result=msg.display_result,
                            display_policy=msg.display_policy,
                            display_meta=msg.display_meta,
                        )
                    )
                    if not inserted_warning:
                        repaired.append(
                            Message(
                                role="user",
                                content=(
                                    "[SYSTEM] Previous session ended before all tool results were "
                                    "recorded. SuperCoder removed that incomplete tool exchange "
                                    "from API replay; re-run any missing checks if needed."
                                ),
                                display_type="error",
                                display_summary="Interrupted tool exchange removed from API replay",
                                display_policy="error",
                            )
                        )
                        inserted_warning = True
                    continue
            repaired.append(msg)

        if not incomplete_call_ids:
            return messages

        return [
            msg
            for msg in repaired
            if not (msg.role == "tool" and msg.tool_call_id in incomplete_call_ids)
        ]

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

    def _auto_compact_if_needed(self) -> dict | None:
        """Run cache-aware compact at safe boundaries when the context is large."""
        if not self.context.should_auto_compact():
            return None

        summary, stats_before, stats_after = self.compact_context()
        if summary.startswith("Error generating summary:"):
            if self.context.should_emergency_compress():
                self.context.force_compress()
                self._save_current_session()
            return {"type": "warning", "content": summary}

        return {
            "type": "auto_compact",
            "content": {
                "summary": summary,
                "stats_before": stats_before,
                "stats_after": stats_after,
            },
        }

    def compact_context(self) -> tuple[str, ContextStats, ContextStats]:
        """Compact the current context with a cache-aware in-band summary request.

        This method:
        1. Appends a temporary maintenance request to the existing API message prefix
        2. Asks the LLM to create a summary without calling tools
        3. Replaces old history with the summary and protected recent messages

        Returns:
            tuple: (summary_text, stats_before, stats_after)
        """

        # Get stats before compaction
        stats_before = self.context.get_stats()

        # Get conversation history
        messages = self.context.get_messages()
        if not messages:
            return ("No context to compact.", stats_before, stats_before)

        recent_messages = self.context.get_protected_recent_messages()
        summary_messages = self.context.get_messages_for_api()
        summary_messages.append(Message("user", CACHE_AWARE_COMPACT_REQUEST))

        try:
            result = self.llm.chat_with_tools_interruptible(
                summary_messages,
                self._tools_schema,
                self.abort_controller,
                on_chunk=self._chunk_callback,
                max_completion_tokens=2048,
                tool_choice="none",
            )
        except AgentAbortedError:
            raise
        except Exception as e:
            get_logger().log_error(e)
            try:
                result = self.llm.chat_with_tools_interruptible(
                    summary_messages,
                    None,
                    self.abort_controller,
                    on_chunk=self._chunk_callback,
                    max_completion_tokens=2048,
                )
            except AgentAbortedError:
                raise
            except Exception as fallback_error:
                get_logger().log_error(fallback_error)
                return (
                    f"Error generating summary: {fallback_error}",
                    stats_before,
                    stats_before,
                )

        summary = result.content.strip()
        if result.truncated:
            return (
                "Error generating summary: compact summary was truncated",
                stats_before,
                stats_before,
            )
        if result.tool_calls or not summary:
            return (
                "Error generating summary: compact request did not return plain summary text",
                stats_before,
                stats_before,
            )

        # Clear old history and keep an exact protected tail after the summary.
        self.context.set_initial_summary(summary, recent_messages)
        self._mode_policy_needs_announcement = True

        # Update session with compacted state
        if self.current_session:
            self.session_manager.update_session_after_compact(
                self.current_session, summary, recent_messages
            )

        # Get stats after compaction
        stats_after = self.context.get_stats()

        return (summary, stats_before, stats_after)
