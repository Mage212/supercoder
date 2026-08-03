"""Interactive REPL for SuperCoder."""

import re
import shlex
import sys
import textwrap
import threading
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style as PromptStyle
from pygments.lexers.markup import MarkdownLexer
from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import __version__
from .abort_controller import AgentAbortedError, InterruptHandler, KeyboardListener
from .agent.agent_modes import MODE_CONFIGS, MODE_CYCLE, AgentMode
from .context.references import summarize_attachment_content, summarize_context_attachment
from .logging import get_logger
from .permissions import PermissionAction
from .ui import render, theme
from .utils import format_relative_time


class SuperCoderREPL:
    """Interactive Read-Eval-Print Loop for SuperCoder."""

    def __init__(self, agent):
        self.agent = agent
        self.console = Console()

        # Initialize commands BEFORE session setup (session uses commands for autocomplete)
        self.commands = {
            "/ask": self.cmd_ask,
            "/plan": self.cmd_plan,
            "/code": self.cmd_code,
            "/accept-edits": self.cmd_accept_edits,
            "/accept": self.cmd_accept_edits,
            "/edit": self.cmd_accept_edits,
            "/clear": self.cmd_clear,
            "/compact": self.cmd_compact,
            "/continue": self.cmd_continue,
            "/undo": self.cmd_undo,
            "/help": self.cmd_help,
            "/config": self.cmd_config,
            "/stats": self.cmd_stats,
            "/debug": self.cmd_debug,
            "/models": self.cmd_models,
            "/model": self.cmd_model,
            "/permissions": self.cmd_permissions,
            "exit": self.cmd_exit,
            "/exit": self.cmd_exit,
            "quit": self.cmd_quit,
            "/quit": self.cmd_quit,
        }

        # Now setup session (uses self.commands)
        self.session = self._setup_session()

        # Setup interrupt handler for double-ESC
        self.interrupt_handler = InterruptHandler(
            on_interrupt=self._on_interrupt, on_first_press=self._on_first_esc, timeout=0.5
        )

        # Setup keyboard listener for background ESC detection
        self.keyboard_listener = KeyboardListener(self.interrupt_handler)

    def _on_interrupt(self):
        """Called when double-ESC triggers interrupt."""
        self.agent.abort_controller.abort()
        self.console.print("\n[bold red]⚠ Interrupting...[/]")

    def _on_first_esc(self):
        """Called after first ESC press."""
        # Use plain text: manual ANSI escapes can leak as "[33m" while
        # prompt_toolkit/Rich are both managing terminal state.
        print("\rPress ESC again to interrupt", end="", flush=True)

    def _setup_session(self):
        """Configure prompt_toolkit session."""
        from prompt_toolkit.completion import ThreadedCompleter
        from prompt_toolkit.key_binding import KeyBindings

        from .autocomplete import AutoCompleter, SlashCommandAutoSuggest

        style = PromptStyle.from_dict(
            {
                "prompt": "#00aa00 bold",
                "bottom-toolbar": "#666666",
            }
        )

        # Enhanced autocomplete with file and command support
        auto_completer = AutoCompleter(
            repo_root=self.agent.repo_root,
            commands=list(self.commands.keys()),
        )
        completer = ThreadedCompleter(auto_completer)

        # Inline auto-suggest for slash commands (gray text)
        auto_suggest = SlashCommandAutoSuggest(commands=list(self.commands.keys()))

        # Key bindings for multiline support
        kb = KeyBindings()

        @kb.add("escape", "enter")  # Alt+Enter or Escape then Enter
        def _(event):
            """Insert newline without submitting."""
            event.current_buffer.insert_text("\n")

        @kb.add("enter")
        def _(event):
            """Accept auto-suggestion if present, otherwise submit."""
            buff = event.current_buffer
            if buff.suggestion and buff.suggestion.text:
                buff.insert_text(buff.suggestion.text)
            else:
                buff.validate_and_handle()

        @kb.add("s-tab")
        def _(event):
            """Cycle agent mode without submitting the prompt."""
            self._cycle_mode()
            event.app.invalidate()

        # History file in project-specific directory
        history_path = self.agent.repo_root / ".supercoder" / "history"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        return PromptSession(
            history=FileHistory(str(history_path)),
            lexer=PygmentsLexer(MarkdownLexer),
            style=style,
            completer=completer,
            auto_suggest=auto_suggest,
            key_bindings=kb,
            bottom_toolbar=self._get_bottom_toolbar,
            complete_while_typing=True,
            multiline=False,  # We handle multiline via { } or Alt+Enter
        )

    def run(self):
        """Start the REPL loop."""
        # Beautiful startup header
        header = Text()
        header.append("🚀 SuperCoder CLI", style="bold green")
        header.append(f" v{__version__}\n", style="dim")
        header.append("Model: ", style="dim")
        header.append(f"{self.agent.llm.model}", style="cyan bold")
        header.append(f" • Context: {self.agent.context.config.max_tokens:,}", style="dim")
        header.append(f" • Tools: {len(self.agent.tools)}\n", style="dim")
        header.append("/help", style="cyan")
        header.append(" for commands • ", style="dim")
        header.append("ESC×2", style="yellow")  # noqa: RUF001
        header.append(" to interrupt • ", style="dim")
        header.append("{ }", style="cyan")
        header.append(" for multiline", style="dim")
        self.console.print(Panel(header, border_style="green", box=box.ROUNDED))

        # Start a new session on fresh start
        self.agent.start_new_session()

        # Multiline state
        multiline_mode = False
        multiline_buffer = []

        while True:
            try:
                # Show different prompt in multiline mode
                prompt = "...> " if multiline_mode else self._get_prompt()

                user_input = self.session.prompt(prompt).strip()

                # Handle multiline mode
                if multiline_mode:
                    if user_input == "}":
                        # End multiline - join and process
                        multiline_mode = False
                        user_input = "\n".join(multiline_buffer)
                        multiline_buffer = []
                    else:
                        # Continue collecting lines
                        multiline_buffer.append(user_input)
                        continue
                elif user_input == "{":
                    # Start multiline mode
                    multiline_mode = True
                    multiline_buffer = []
                    self.console.print(
                        "[dim]Multiline mode: enter lines, end with } on its own line[/]"
                    )
                    continue

                if not user_input:
                    continue

                # Check for slash commands
                cmd_parts = user_input.split()
                cmd = cmd_parts[0].lower()

                if cmd in self.commands:
                    if self.commands[cmd](user_input):
                        break
                    continue

                # Process chat - replace input line(s) with styled version
                # Calculate actual terminal lines (including wrapped text)
                import shutil

                terminal_width = shutil.get_terminal_size().columns
                prompt_prefix_len = len(self._get_prompt())

                # Calculate total visual lines by accounting for terminal wrapping
                visual_lines = 0
                for line in user_input.split("\n"):
                    # First line includes prompt, subsequent lines don't (in prompt_toolkit)
                    line_len = len(line) + prompt_prefix_len if visual_lines == 0 else len(line)

                    # Count how many terminal lines this logical line takes
                    if line_len == 0:
                        visual_lines += 1
                    else:
                        visual_lines += (line_len + terminal_width - 1) // terminal_width

                # Move up and clear each visual line
                for _ in range(visual_lines):
                    sys.stdout.write("\033[A\033[2K")  # Move up + clear line
                sys.stdout.flush()
                self.console.print(render.render_user_message(user_input, live=True))
                self._handle_chat(user_input)

            except KeyboardInterrupt:
                if multiline_mode:
                    multiline_mode = False
                    multiline_buffer = []
                    self.console.print("\n[dim]Multiline cancelled[/]")
                else:
                    self.console.print("\n[dim]Use 'exit' to quit[/]")
                continue
            except EOFError:
                break

        self.console.print("[green]Goodbye![/]")

    def _handle_chat(self, message):
        """Handle chat interaction — dispatches to native or streaming handler."""
        if self.agent.streaming:
            return self._handle_chat_streaming(message)
        return self._handle_chat_native(message)

    def _handle_chat_native(self, message):
        """Handle chat using native API tool calls (non-streaming).

        Clean and simple: receives complete responses from chat_turn(),
        no streaming buffer, no tag filtering, no paragraph boundary detection.
        """
        errors = []
        rollback_info = None
        touched_files = set()

        spinner = self.console.status("[bold blue]SuperCoder is thinking...[/]", spinner="dots")
        spinner.start()

        # Live token counter + elapsed timer for generation progress
        # The timer thread ensures the spinner always shows activity,
        # even when the provider (e.g. LMStudio) buffers tool call arguments
        # and doesn't stream chunks incrementally.
        _gen_tokens = [0]
        _gen_start = time.monotonic()
        _gen_stop = threading.Event()
        _gen_phase = ["response"]  # "response" or "tool_call"

        def _tick():
            while not _gen_stop.wait(0.7):
                elapsed = int(time.monotonic() - _gen_start)
                n = _gen_tokens[0]
                label = _gen_phase[0]
                spinner.update(f"[bold blue]Generating {label}... {n:,} tokens ({elapsed}s)[/]")

        _tick_thread = threading.Thread(target=_tick, daemon=True)
        _tick_thread.start()

        def _on_chunk(n):
            _gen_tokens[0] = n

        self.agent.set_chunk_callback(_on_chunk)

        # Setup keyboard listener for ESC (between-iteration abort only)
        if hasattr(self, "keyboard_listener"):
            self.keyboard_listener.start()

        try:
            for event in self.agent.chat_turn(message):
                event_type = event.get("type")
                content = event.get("content")

                if event_type == "thinking":
                    spinner.stop()
                    self._print_output_spacer()
                    self.console.print(render.render_reasoning(content.strip()))
                    spinner.update("[bold blue]SuperCoder is thinking...[/]")
                    spinner.start()

                elif event_type == "response":
                    spinner.stop()
                    self._print_output_spacer()
                    # Full response — render in a branded panel with a model header.
                    self.console.print(
                        render.render_assistant_message(content, model=self.agent.llm.config.model)
                    )
                    spinner.update("[bold blue]SuperCoder is thinking...[/]")
                    spinner.start()

                elif event_type == "tool_call":
                    spinner.stop()
                    self._print_output_spacer()
                    self._display_tool_call(content)
                    self._track_files(content, touched_files)
                    name = content.get("name", "tool")
                    spinner.update(f"[bold blue]Executing {name}...[/]")
                    spinner.start()

                elif event_type == "tool_result":
                    spinner.stop()
                    self._print_output_spacer()
                    self._display_tool_result(content)
                    _gen_tokens[0] = 0
                    _gen_start = time.monotonic()
                    _gen_phase[0] = "tool call"
                    spinner.update("[bold blue]SuperCoder is thinking...[/]")
                    spinner.start()

                elif event_type == "context_attachment":
                    spinner.stop()
                    self._print_output_spacer()
                    self._display_context_attachment(content)
                    spinner.update("[bold blue]SuperCoder is thinking...[/]")
                    spinner.start()

                elif event_type == "error":
                    errors.append(content)

                elif event_type == "warning":
                    spinner.stop()
                    self._print_output_spacer()
                    self._print_block(f"[yellow]{content}[/]", "Warning", "yellow", "!")
                    spinner.update("[bold blue]SuperCoder is thinking...[/]")
                    spinner.start()

                elif event_type == "tool_retry":
                    spinner.stop()
                    attempt = content.get("attempt", 0)
                    max_attempts = content.get("max_attempts", 0)
                    self._print_output_spacer()
                    self.console.print(
                        "[yellow]Tool call format was invalid. "
                        f"Retrying model response ({attempt}/{max_attempts})...[/]"
                    )
                    _gen_tokens[0] = 0
                    _gen_start = time.monotonic()
                    spinner.update("[bold blue]Retrying tool call format...[/]")
                    spinner.start()

                elif event_type == "auto_compact":
                    spinner.stop()
                    before = content.get("stats_before")
                    after = content.get("stats_after")
                    self.console.print(
                        "[dim]Context auto-compacted: "
                        f"{before.used_tokens:,} -> {after.used_tokens:,} tokens[/]"
                    )
                    spinner.update("[bold blue]SuperCoder is thinking...[/]")
                    spinner.start()

                elif event_type == "rollback":
                    rollback_info = content

                elif event_type == "command_confirm":
                    spinner.stop()
                    if hasattr(self, "keyboard_listener"):
                        self.keyboard_listener.stop()
                    result = self._handle_command_confirm(content.get("command", ""))
                    event["result"].update(result)
                    if hasattr(self, "keyboard_listener"):
                        self.keyboard_listener.start()
                    spinner.update("[bold blue]Running command...[/]")
                    spinner.start()

                elif event_type == "edit_confirm":
                    spinner.stop()
                    if hasattr(self, "keyboard_listener"):
                        self.keyboard_listener.stop()
                    result = self._handle_edit_confirm(
                        content.get("arguments", {}), content.get("preview")
                    )
                    event["result"].update(result)
                    if hasattr(self, "keyboard_listener"):
                        self.keyboard_listener.start()
                    spinner.update("[bold blue]Applying edit...[/]")
                    spinner.start()

                elif event_type == "command_waiting":
                    spinner.stop()
                    if hasattr(self, "keyboard_listener"):
                        self.keyboard_listener.stop()
                    self._handle_command_waiting(event)
                    if hasattr(self, "keyboard_listener"):
                        self.keyboard_listener.start()
                    spinner.start()

                elif event_type == "done":
                    spinner.stop()

        except Exception:
            raise
        finally:
            _gen_stop.set()
            spinner.stop()
            if hasattr(self, "keyboard_listener"):
                self.keyboard_listener.stop()
            self.agent.abort_controller.reset()

        # === Post-processing ===
        if rollback_info:
            restored = rollback_info.get("restored", [])
            failed = rollback_info.get("failed", [])
            reason = rollback_info.get("reason", "Unknown")
            rollback_lines = [f"[dim]Reason: {reason}[/]"]
            rollback_lines.extend(f"  ✓ Restored: {f}" for f in restored)
            rollback_lines.extend(f"  [red]✗ Failed: {f}[/]" for f in failed)
            rollback_content = "\n".join(rollback_lines)
            if failed:
                self._print_block(rollback_content, "PARTIAL ROLLBACK", "yellow", "⚠")
            else:
                self._print_block(rollback_content, "Files Rolled Back", "cyan", "↩")

        for error in errors:
            self._print_block(f"[red]{error}[/]", "Error", "red", "❌")

        self._display_status_footer(touched_files)
        self.console.print(Rule(style="dim grey50"))

    def _handle_chat_streaming(self, message):
        """Handle chat interaction with streaming output.

        .. deprecated::
            Use ``_handle_chat_native()`` instead. Streaming mode is deprecated.

        Uses a state machine to transition between:
        - SPINNER: waiting for LLM response (console.status)
        - STREAMING: printing completed paragraphs as Markdown
        """
        from .streaming_buffer import StreamingDisplayBuffer

        # Buffers
        reasoning_text = ""
        errors = []
        was_aborted = False
        rollback_info = None
        touched_files = set()

        # --- Streaming state ---
        is_streaming = False
        display_buffer = None
        accumulated_display = ""  # All safe text received so far (tags already stripped by buffer)
        _printed_up_to = 0  # Character offset into accumulated_display up to which we've printed

        # --- Spinner (manual start/stop) ---
        spinner = self.console.status("[bold blue]SuperCoder is thinking...[/]", spinner="dots")
        spinner.start()

        def flush_reasoning():
            """Output accumulated reasoning as a block."""
            nonlocal reasoning_text
            clean = self._filter_special_tokens(reasoning_text)
            if clean.strip():
                self._print_block(clean.strip(), "Reasoning", "magenta", "💭")
            reasoning_text = ""

        def start_streaming():
            """Switch from spinner to paragraph streaming."""
            nonlocal is_streaming, display_buffer, accumulated_display, _printed_up_to
            # Stop spinner FIRST — printing while Rich's Live/Status is active
            # corrupts cursor tracking and produces rendering artifacts.
            spinner.stop()
            flush_reasoning()
            display_buffer = StreamingDisplayBuffer(self.agent.tool_calling_type)
            accumulated_display = ""
            _printed_up_to = 0
            is_streaming = True

        def print_new_paragraphs():
            """Print newly completed paragraphs as Markdown using offset tracking.

            Uses character offset (_printed_up_to) into accumulated_display so that
            leading/trailing empty strings from split() do not distort the count.
            """
            nonlocal _printed_up_to
            unprinted = accumulated_display[_printed_up_to:]
            if not unprinted:
                return
            # Find the last paragraph boundary in unprinted text
            boundary = unprinted.rfind("\n\n")
            if boundary > 0:
                to_print = unprinted[:boundary].strip()
                if to_print:
                    self.console.print(Markdown(to_print))
                _printed_up_to += boundary + 2  # advance past the \n\n
            elif len(unprinted) >= 300:
                # Very long paragraph — force-print at last line break
                last_nl = unprinted.rfind("\n")
                if last_nl > 0:
                    to_print = unprinted[:last_nl].strip()
                    if to_print:
                        self.console.print(Markdown(to_print))
                    _printed_up_to += last_nl + 1

        def stop_streaming():
            """Finalize streaming, print any remaining text."""
            nonlocal is_streaming, display_buffer, accumulated_display, _printed_up_to
            if not is_streaming or display_buffer is None:
                spinner.stop()
                flush_reasoning()
                return

            # Flush any text still held in the buffer
            remaining = display_buffer.flush()
            accumulated_display += remaining

            # Print everything after the last printed offset.
            # StreamingDisplayBuffer already stripped tool-call tags, so we do NOT
            # call _filter_special_tokens here — that would destroy \n\n boundaries.
            unprinted = accumulated_display[_printed_up_to:].strip()
            if unprinted:
                self.console.print(Markdown(unprinted))

            display_buffer = None
            accumulated_display = ""
            _printed_up_to = 0
            is_streaming = False

        # --- Event processing ---
        if hasattr(self, "keyboard_listener"):
            self.keyboard_listener.start()

        try:
            for event in self.agent.chat_stream(message):
                event_type = event.get("type")
                content = event.get("content")

                if event_type == "reasoning":
                    reasoning_text += content

                elif event_type == "token":
                    if not is_streaming:
                        start_streaming()

                    assert display_buffer is not None  # set by start_streaming()
                    chunk = display_buffer.add(content)
                    if chunk:
                        accumulated_display += chunk
                        print_new_paragraphs()

                elif event_type == "tool_call":
                    stop_streaming()
                    self._display_tool_call(content)
                    self._track_files(content, touched_files)
                    # Dynamic spinner text
                    name = content.get("name", "tool")
                    spinner.update(f"[bold blue]Executing {name}...[/]")
                    spinner.start()

                elif event_type == "tool_result":
                    spinner.stop()
                    self._display_tool_result(content)
                    # Back to waiting spinner for next LLM turn
                    spinner.update("[bold blue]SuperCoder is thinking...[/]")
                    spinner.start()

                elif event_type == "context_attachment":
                    stop_streaming()
                    self._display_context_attachment(content)
                    spinner.update("[bold blue]SuperCoder is thinking...[/]")
                    spinner.start()

                elif event_type == "error":
                    errors.append(content)

                elif event_type == "aborted":
                    was_aborted = True
                    if is_streaming:
                        display_buffer = None
                        accumulated_display = ""
                        _printed_up_to = 0
                        is_streaming = False
                    spinner.stop()
                    reasoning_text = ""

                elif event_type == "rollback":
                    rollback_info = content

                elif event_type == "command_confirm":
                    stop_streaming()
                    # Stop the keyboard listener: it holds the terminal in raw mode
                    # which breaks sys.stdin.readline() used in the confirm prompt.
                    if hasattr(self, "keyboard_listener"):
                        self.keyboard_listener.stop()
                    result = self._handle_command_confirm(content.get("command", ""))
                    event["result"].update(result)
                    # Restart listener for the upcoming LLM turn
                    if hasattr(self, "keyboard_listener"):
                        self.keyboard_listener.start()
                    spinner.update("[bold blue]Running command...[/]")
                    spinner.start()

                elif event_type == "edit_confirm":
                    stop_streaming()
                    if hasattr(self, "keyboard_listener"):
                        self.keyboard_listener.stop()
                    result = self._handle_edit_confirm(
                        content.get("arguments", {}), content.get("preview")
                    )
                    event["result"].update(result)
                    if hasattr(self, "keyboard_listener"):
                        self.keyboard_listener.start()
                    spinner.update("[bold blue]Applying edit...[/]")
                    spinner.start()

                elif event_type == "command_waiting":
                    spinner.stop()
                    flush_reasoning()
                    # Stop raw-mode listener before interactive stdin read
                    if hasattr(self, "keyboard_listener"):
                        self.keyboard_listener.stop()
                    self._handle_command_waiting(event)
                    if hasattr(self, "keyboard_listener"):
                        self.keyboard_listener.start()
                    spinner.start()

                elif event_type == "done":
                    stop_streaming()

        except Exception:
            raise

        finally:
            spinner.stop()
            if hasattr(self, "keyboard_listener"):
                self.keyboard_listener.stop()
            self.agent.abort_controller.reset()

        # === Post-processing ===

        # Display Abort notification
        if was_aborted:
            self._print_block(
                "[bold yellow]Agent execution was interrupted by user (ESC)[/]",
                "Interrupted",
                "yellow",
                "⚠",
            )

        # Display Rollback info
        if rollback_info:
            restored = rollback_info.get("restored", [])
            failed = rollback_info.get("failed", [])
            reason = rollback_info.get("reason", "Unknown")
            rollback_lines = [f"[dim]Reason: {reason}[/]"]
            rollback_lines.extend(f"  ✓ Restored: {f}" for f in restored)
            rollback_lines.extend(f"  [red]✗ Failed: {f}[/]" for f in failed)
            rollback_content = "\n".join(rollback_lines)
            if failed:
                self._print_block(rollback_content, "PARTIAL ROLLBACK", "yellow", "⚠")
            else:
                self._print_block(rollback_content, "Files Rolled Back", "cyan", "↩")

        # Display Errors
        for error in errors:
            self._print_block(f"[red]{error}[/]", "Error", "red", "❌")

        # Display Status Footer
        self._display_status_footer(touched_files)

        self.console.print(Rule(style="dim grey50"))

    def _track_files(self, tool_call, touched_files):
        """Extract file paths from tool arguments to track active files."""
        tool_call.get("name")
        args = tool_call.get("arguments", {})

        # Handle string args (sometimes args is a JSON string)
        if isinstance(args, str):
            try:
                import json

                args = json.loads(args)
            except Exception:
                return

        if not isinstance(args, dict):
            return

        # Look for common file arguments
        for key in [
            "file",
            "filepath",
            "fileName",
            "path",
            "filename",
            "target_file",
            "source_file",
        ]:
            if key in args and isinstance(args[key], str):
                from pathlib import Path

                try:
                    p = Path(args[key])
                    # Store relative path if possible
                    try:
                        rel_path = p.relative_to(self.agent.repo_root)
                        touched_files.add(str(rel_path))
                    except ValueError:
                        touched_files.add(p.name)
                except Exception:
                    pass

    def _display_status_footer(self, touched_files):
        """Display a status footer with the unified progress bar and active files."""
        stats = self.agent.context.get_stats()
        bar = render.render_context_bar(
            stats.used_tokens, stats.total_tokens, width=theme.BAR_WIDTH_FOOTER
        )
        # render_context_bar already includes the "used/total tokens" label, so we
        # append the active-files segment separately with a separator.
        if touched_files:
            files_str = ", ".join(sorted(touched_files))
            self.console.print(
                Group(bar, Text(f"Active: {files_str}", style="dim")), justify="right"
            )
        else:
            self.console.print(bar, justify="right")

    def _render_session_history(self, messages: list) -> None:
        """Render session messages visually after restore.

        Reuses the same rendering helpers as live output (_print_block,
        _display_tool_call, _display_tool_result) to maintain visual fidelity.
        Tool calls are interleaved with their matching tool results by tool_call_id.
        """
        import json

        MAX_TURNS = 6

        # Filter out system messages
        showable = [m for m in messages if m.role != "system"]
        to_show = self._tail_turn_groups(showable, MAX_TURNS)

        if len(showable) > len(to_show):
            skipped = len(showable) - len(to_show)
            self.console.print(f"[dim]... {skipped} earlier messages not shown[/]\n")

        # Build index: tool_call_id → position for fast lookup
        result_index: dict[str, int] = {}
        for i, m in enumerate(to_show):
            if m.tool_call_id:
                result_index[m.tool_call_id] = i

        consumed: set[int] = set()
        i = 0
        while i < len(to_show):
            if i in consumed:
                i += 1
                continue

            msg = to_show[i]
            dt = msg.display_type

            if dt == "user_input":
                self._print_output_spacer()
                self.console.print(render.render_user_message(msg.content, live=False))

            elif dt == "thinking":
                text = msg.content[:500] + ("..." if len(msg.content) > 500 else "")
                self._print_output_spacer()
                self.console.print(render.render_reasoning(text))

            elif dt in ("response", "tool_call"):
                # Render text content
                if msg.content and msg.content.strip():
                    self._print_output_spacer()
                    self.console.print(
                        render.render_assistant_message(
                            msg.content, model=self.agent.llm.config.model
                        )
                    )

                # Interleave: tool_call → matching tool_result
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "?")
                        args_str = fn.get("arguments", "{}")
                        try:
                            args_obj = (
                                json.loads(args_str) if isinstance(args_str, str) else args_str
                            )
                        except Exception:
                            args_obj = {"_raw": args_str}
                        self._print_output_spacer()
                        self._display_tool_call({"name": name, "arguments": args_obj})

                        # Find and render matching tool result
                        tc_id = tc.get("id", "")
                        j = result_index.get(tc_id)
                        if j is not None and j not in consumed:
                            result_msg = to_show[j]
                            self._print_output_spacer()
                            self._display_tool_result(
                                self._tool_result_payload_from_message(result_msg, name)
                            )
                            consumed.add(j)

            elif dt == "tool_result":
                # Only render if not already consumed by interleaving above
                self._print_output_spacer()
                self._display_tool_result(self._tool_result_payload_from_message(msg, "tool"))

            elif dt == "error":
                self._print_output_spacer()
                if msg.role == "tool":
                    self._display_tool_result(self._tool_result_payload_from_message(msg, "tool"))
                else:
                    self._print_block(msg.content, "Error", "red", "❌")

            elif dt == "compact_summary":
                text = msg.content[:200]
                self._print_output_spacer()
                self._print_block(f"[dim]{text}...[/]", "Context Summary", "dim", "📋")

            elif dt == "context_attachment":
                self._print_output_spacer()
                self._print_block(
                    f"[dim]{summarize_attachment_content(msg.content)}[/]",
                    "Attached Context",
                    "cyan",
                    "@",
                )

            elif dt == "mode_policy":
                self._print_output_spacer()
                self.console.print(f"[dim]{msg.content}[/]")

            else:
                # Fallback for old sessions without display_type
                if msg.role == "user" and msg.content:
                    self._print_output_spacer()
                    self.console.print(render.render_user_message(msg.content, live=False))
                elif msg.role == "assistant" and msg.content:
                    self._print_output_spacer()
                    self.console.print(
                        render.render_assistant_message(
                            msg.content, model=self.agent.llm.config.model
                        )
                    )
                elif msg.role == "tool":
                    self._print_output_spacer()
                    self._display_tool_result({"name": msg.name or "tool", "result": msg.content})

            i += 1

    def _tail_turn_groups(self, messages: list, max_turns: int) -> list:
        """Return the last complete user-turn groups for session restore."""
        groups: list[list] = []
        current: list = []

        for msg in messages:
            if msg.display_type == "user_input" and current:
                groups.append(current)
                current = [msg]
            else:
                current.append(msg)
        if current:
            groups.append(current)

        selected: list = []
        turns = 0
        for group in reversed(groups):
            selected = [*group, *selected]
            if any(msg.display_type == "user_input" for msg in group):
                turns += 1
            if turns >= max_turns:
                break
        return selected

    def _tool_result_payload_from_message(self, msg, fallback_name: str) -> dict:
        """Build a render payload from persisted UI-only message metadata."""
        payload = {
            "name": msg.name or fallback_name,
            "result": msg.content,
            "display_summary": getattr(msg, "display_summary", None),
            "display_result": getattr(msg, "display_result", None),
            "display_policy": getattr(msg, "display_policy", None),
        }
        payload.update(getattr(msg, "display_meta", None) or {})
        if getattr(msg, "display_type", None) == "error":
            if not payload.get("display_policy"):
                payload["display_policy"] = "error"
            if payload.get("display_result") is None:
                payload["display_result"] = msg.content
        return payload

    def _print_block(self, content, title: str, color: str, icon: str = ""):
        """Print content in a rounded panel (unified with the ui.render style).

        Args:
            content: Rich renderable (Text, Markdown, Syntax, str)
            title: Block title (e.g. "Reasoning", "Tool Call")
            color: Color for the border (e.g. "magenta", "yellow")
            icon: Optional emoji icon
        """
        full_title = f"[bold {color}]{icon} {title}[/]" if icon else f"[bold {color}]{title}[/]"
        self.console.print(Panel(content, title=full_title, border_style=color, box=box.ROUNDED))

    def _print_output_spacer(self) -> None:
        """Separate consecutive agent outputs in terminal scrollback."""
        self.console.print()

    def _display_context_attachment(self, summary: dict):
        """Display a compact summary of host-attached @path context."""
        self.console.print(f"[dim]{summarize_context_attachment(summary)}[/]")

    def _select_confirmation_action(
        self,
        message: str,
        options: list[tuple[str, str]],
        cancel_value: str,
    ) -> str:
        """Show an arrow-key confirmation menu and return the selected value."""
        from questionary import Choice, Style, select

        style = Style(
            [
                ("qmark", "fg:#00aa00 bold"),
                ("pointer", "fg:#00aa00 bold"),
                ("highlighted", "bold"),
            ]
        )
        question = select(
            message,
            choices=[Choice(title=title, value=value) for title, value in options],
            style=style,
            qmark="▸",
            pointer=">",
            instruction="(↑↓ select, Enter confirm, Esc cancel)",
            use_shortcuts=False,
            use_arrow_keys=True,
            use_jk_keys=False,
            use_emacs_keys=False,
        )

        key_bindings = getattr(question.application, "key_bindings", None)
        if key_bindings is not None:

            @key_bindings.add("escape", eager=True)
            def _(event):
                event.app.exit(result=cancel_value)

        try:
            selected = question.unsafe_ask()
        except (KeyboardInterrupt, EOFError):
            selected = cancel_value

        if isinstance(selected, str) and selected:
            return selected
        return cancel_value

    def _handle_command_confirm(self, command: str) -> dict[str, object]:
        """Ask user to approve or deny a shell command before it runs.

        Returns a structured decision consumed by the agent.
        """
        while True:
            preview, is_long = self._format_command_confirmation_preview(command)
            self._print_block(preview, "Run Command?", "yellow", "⚡")

            choices = [
                ("Run once", "approve_once"),
            ]
            if is_long:
                choices.append(("Show full command", "show_full"))
            choices.extend(
                [
                    ("Allow for this session", "allow_session"),
                    ("Always allow for this project", "allow_persistent"),
                    ("Always deny for this project", "deny_persistent"),
                    ("Cancel", "deny_once"),
                ]
            )

            choice = self._select_confirmation_action(
                "Choose command action",
                choices,
                "deny_once",
            )

            if choice == "show_full":
                self._print_block(
                    Syntax(
                        command,
                        "bash",
                        theme="monokai",
                        word_wrap=True,
                        background_color="default",
                    ),
                    "Full Command",
                    "yellow",
                    "⚡",
                )
                continue
            break

        if choice == "approve_once":
            self.console.print("[green]✓ Approved[/]")
            return {"approved": True, "decision": "approve_once"}
        if choice == "allow_session":
            self.console.print("[green]✓ Approved (allowed this session)[/]")
            return {"approved": True, "decision": "allow_session"}
        if choice == "allow_persistent":
            self.console.print("[green]✓ Approved (saved for this project)[/]")
            return {"approved": True, "decision": "allow_persistent"}
        if choice == "deny_persistent":
            self.console.print("[yellow]✓ Denied (saved for this project)[/]")
            return {"approved": False, "decision": "deny_persistent"}
        self.console.print("[red]✗ Cancelled[/]")
        return {"approved": False, "decision": "deny_once"}

    def _handle_edit_confirm(
        self, arguments: dict, preview: object | None = None
    ) -> dict[str, object]:
        """Ask user to approve or deny a file edit before it runs."""
        preview_content = self._format_edit_diff_preview(
            arguments, preview
        ) or self._format_edit_preview(arguments)
        transient_lines = self._print_transient_block(
            preview_content, "Apply File Edit?", "cyan", "📝"
        )

        choice = self._select_confirmation_action(
            "Choose edit action",
            [
                ("Apply edit", "apply"),
                ("Apply and switch to accept-edits", "apply_accept_edits"),
                ("Cancel", "cancel"),
            ],
            "cancel",
        )
        self._clear_transient_lines(transient_lines + 1)

        if choice == "apply":
            self.console.print("[green]✓ Edit approved[/]")
            return {"approved": True}
        if choice == "apply_accept_edits":
            self.console.print("[green]✓ Edit approved; switched to accept-edits[/]")
            return {"approved": True, "decision": "apply_and_accept_edits"}
        self.console.print("[red]✗ Edit cancelled[/]")
        return {"approved": False}

    def _print_transient_block(self, content, title: str, color: str, icon: str) -> int:
        """Render a block and return the number of terminal lines it occupied."""
        with self.console.capture() as capture:
            self._print_block(content, title, color, icon)
        rendered = capture.get()
        self.console.file.write(rendered)
        self.console.file.flush()
        return self._terminal_line_count(rendered)

    def _clear_transient_lines(self, line_count: int) -> None:
        """Clear recently rendered confirmation UI from the terminal."""
        if line_count <= 0:
            return
        self.console.file.write(f"\x1b[{line_count}A\x1b[J")
        self.console.file.flush()

    def _terminal_line_count(self, rendered: str) -> int:
        """Count rendered terminal lines in captured Rich output."""
        plain = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", rendered)
        return max(1, len(plain.splitlines()))

    def _format_command_confirmation_preview(self, command: str):
        """Build a bounded command confirmation preview."""
        width = max(40, min(160, self.console.size.width - 12))
        visual_lines = self._split_visual_lines(command, width)
        logical_lines = command.count("\n") + 1 if command else 0
        is_long = len(command) > 1200 or logical_lines > 12 or len(visual_lines) > 12

        root = self._root_command(command)
        summary = Text()
        summary.append("Root command: ", style="bold")
        summary.append(root or "(unknown)", style="yellow")
        summary.append("\nSize: ", style="bold")
        summary.append(f"{len(command):,} chars", style="cyan")
        summary.append(" · ")
        summary.append(f"{logical_lines:,} logical lines", style="cyan")
        summary.append(" · ")
        summary.append(f"{len(visual_lines):,} visual lines", style="cyan")
        if is_long:
            summary.append("\nPreview: ", style="bold")
            summary.append(
                "showing head/tail only; choose Show full command to inspect all.", style="dim"
            )
        else:
            summary.append("\nCommand:", style="bold")

        preview_text = self._command_preview_text(visual_lines, is_long)
        syntax = Syntax(
            preview_text,
            "bash",
            theme="monokai",
            line_numbers=False,
            word_wrap=False,
            background_color="default",
        )
        return Group(summary, syntax), is_long

    def _root_command(self, command: str) -> str:
        """Extract the executable/root command for confirmation summary."""
        stripped = command.strip()
        if not stripped:
            return ""
        try:
            parts = shlex.split(stripped, posix=True)
            if parts:
                return parts[0]
        except ValueError:
            pass
        return stripped.split(maxsplit=1)[0]

    def _split_visual_lines(self, text: str, width: int) -> list[str]:
        """Split text into terminal-width visual lines without losing newlines."""
        visual_lines: list[str] = []
        for logical_line in text.split("\n"):
            if logical_line == "":
                visual_lines.append("")
                continue
            wrapped = textwrap.wrap(
                logical_line,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            )
            visual_lines.extend(wrapped or [""])
        return visual_lines

    def _command_preview_text(self, visual_lines: list[str], is_long: bool) -> str:
        """Return a head/tail preview for long commands."""
        if not is_long:
            return "\n".join(visual_lines)
        head_count = 6
        tail_count = 3
        hidden = max(0, len(visual_lines) - head_count - tail_count)
        head = visual_lines[:head_count]
        tail = visual_lines[-tail_count:] if hidden else []
        marker = [f"... {hidden:,} visual lines hidden ..."] if hidden else []
        return "\n".join([*head, *marker, *tail])

    def _format_edit_diff_preview(self, arguments: dict, preview: object | None = None):
        """Build a syntax-highlighted diff preview for a pending code-edit call."""
        if preview is None:
            preview = self._get_code_edit_preview(arguments)
        if preview is None or not getattr(preview, "ok", False):
            return None

        diff = getattr(preview, "diff", "")
        if not diff:
            return None

        header = Text()
        header.append("File: ", style="bold")
        header.append(
            str(arguments.get("filepath") or arguments.get("fileName") or ""), style="yellow"
        )
        header.append("\nOperation: ", style="bold")
        header.append(str(arguments.get("operation") or "search_replace"), style="cyan")
        message = str(getattr(preview, "message", ""))
        if message:
            header.append("\n")
            header.append(message, style="green")

        syntax = Syntax(
            diff, "diff", theme="monokai", line_numbers=False, background_color="default"
        )
        return Group(header, syntax)

    def _get_code_edit_preview(self, arguments: dict):
        """Return CodeEditTool preview data when available."""
        tools = getattr(self.agent, "tools", None)
        if not isinstance(tools, dict):
            return None
        tool = tools.get("code-edit")
        preview_edit = getattr(tool, "preview_edit", None)
        if not callable(preview_edit):
            return None
        try:
            return preview_edit(arguments)
        except Exception as exc:
            get_logger().log_error(exc)
            return None

    def _format_edit_preview(self, arguments: dict) -> Text:
        """Build a compact preview for a pending code-edit call."""
        filepath = str(arguments.get("filepath") or arguments.get("fileName") or "")
        operation = str(arguments.get("operation") or "search_replace")
        preview = Text()
        preview.append("File: ", style="bold")
        preview.append(filepath or "(missing)", style="yellow")
        preview.append("\nOperation: ", style="bold")
        preview.append(operation, style="cyan")

        if operation in {"create", "append"}:
            content = self._short_preview(str(arguments.get("content", "")))
            preview.append("\n\nContent preview:\n", style="bold")
            preview.append(content or "(empty)")
        elif operation == "search_replace":
            search = self._short_preview(str(arguments.get("search", "")), limit=500)
            replace = self._short_preview(str(arguments.get("replace", "")), limit=500)
            preview.append("\n\nSearch:\n", style="bold")
            preview.append(search or "(empty)")
            preview.append("\n\nReplace:\n", style="bold")
            preview.append(replace or "(empty)")
        elif operation == "replace_lines":
            preview.append("\n\nLines: ", style="bold")
            preview.append(f"{arguments.get('startLine', '?')} - {arguments.get('endLine', '?')}")
            content = self._short_preview(str(arguments.get("content", "")))
            preview.append("\n\nReplacement preview:\n", style="bold")
            preview.append(content or "(empty)")
        else:
            preview.append("\n\nArguments preview:\n", style="bold")
            safe_args = {
                key: self._short_preview(str(value), limit=300)
                for key, value in arguments.items()
                if key not in {"content", "search", "replace"}
            }
            preview.append(str(safe_args))

        return preview

    def _short_preview(self, value: str, limit: int = 1200) -> str:
        """Return a bounded preview string for terminal confirmation prompts."""
        if len(value) <= limit:
            return value
        return value[:limit] + "\n... truncated ..."

    def _handle_command_waiting(self, event):
        """Handle a command that appears to be waiting for input."""
        content = event.get("content", "")
        process = event.get("process")
        event.get("tool_name", "command-exec")

        # Display warning
        self._print_block(f"[yellow]{content}[/]", "Process Stalled", "yellow", "⚠️")

        # Simple stdin-based menu (most reliable across terminals)
        self.console.print("\n[bold]Options:[/]")
        self.console.print("  [cyan]k[/] - Kill the process")
        self.console.print("  [cyan]w[/] - Wait longer (continue until timeout)")

        try:
            import sys

            self.console.print("\n[bold cyan]Action [k/w]>[/] ", end="")
            choice = sys.stdin.readline().strip().lower()

            if choice.startswith("k"):
                if process and hasattr(process, "kill"):
                    try:
                        process.kill()
                        process.wait(timeout=5)
                        self.console.print("[green]✓ Process killed[/]")
                        return "killed"
                    except Exception as e:
                        self.console.print(f"[red]Failed to kill process: {e}[/]")
                        return f"error: {e}"
                return "killed"
            else:
                self.console.print("[dim]Continuing to wait for process...[/]")
                return "wait"

        except (KeyboardInterrupt, EOFError):
            # User pressed Ctrl+C - kill the process
            if process and hasattr(process, "kill"):
                try:
                    process.kill()
                    process.wait(timeout=5)
                    self.console.print("\n[green]✓ Process killed (interrupted)[/]")
                except Exception:
                    pass
            return "killed"

    @staticmethod
    def _strip_nested_json(prefix_pattern: str, text: str, flags: int = 0) -> str:
        """Remove occurrences of prefix_pattern followed by a balanced {...} block.

        Unlike `{[^}]*}` regex, this handles nested braces and string literals
        correctly, so tool calls with code snippets in their arguments are fully
        stripped rather than leaving orphaned fragment text.
        """
        import re

        result = []
        last = 0
        for m in re.finditer(prefix_pattern, text, flags=flags):
            brace_start = m.end()
            # Skip whitespace between prefix and opening brace
            while brace_start < len(text) and text[brace_start] in " \t\n":
                brace_start += 1
            if brace_start >= len(text) or text[brace_start] != "{":
                # No JSON object follows — keep as-is
                result.append(text[last : m.end()])
                last = m.end()
                continue
            # Walk the string to find the matching closing brace
            depth = 0
            in_str = False
            esc = False
            end = brace_start
            for j, ch in enumerate(text[brace_start:], brace_start):
                if esc:
                    esc = False
                    continue
                if ch == "\\" and in_str:
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
            result.append(text[last : m.start()])
            last = end
        result.append(text[last:])
        return "".join(result)

    def _filter_special_tokens(self, text: str) -> str:
        """Remove special tokens from display text while preserving normal content."""
        import re

        # Remove tool_code blocks: ```tool_code ... ```
        text = re.sub(r"```tool_code\s*\n?.*?\n?```", "", text, flags=re.DOTALL)
        # Remove our native tool call format: <@TOOL>...</@TOOL>
        text = re.sub(r"<@TOOL>.*?</@TOOL>", "", text, flags=re.DOTALL)
        # Remove GLM-style tool calls: <tool_call>...</tool_call>
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
        # Remove model-generated TOOL_RESULT blocks (model shouldn't generate these!)
        text = re.sub(r"<@TOOL_RESULT>.*?</@TOOL_RESULT>", "", text, flags=re.DOTALL)
        # Remove complete Qwen-style blocks: <|start|>...<|call|>
        text = re.sub(r"<\|start\|>.*?<\|call\|>", "", text, flags=re.DOTALL)
        # Remove gpt-oss format: <|channel|>...to=...<|message|>{...} (nested-brace-aware)
        text = self._strip_nested_json(r"<\|channel\|>.*?<\|message\|>", text, flags=re.DOTALL)
        # Remove simple tool call format: to=tool.name {...} (nested-brace-aware)
        text = self._strip_nested_json(
            r"to=(?:tool[:\.]|TOOL\s+)[\w-]+\s*", text, flags=re.IGNORECASE
        )
        # Remove any remaining special markers
        text = re.sub(r"<\|[^|]+\|>", "", text)
        # Collapse 3+ consecutive newlines to 2 (preserve paragraph breaks for Markdown).
        # Do NOT collapse \n\n → \n — that destroys Markdown paragraph structure.
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _display_tool_call(self, tool_call):
        """Display tool call in a panel."""
        name = tool_call.get("name")
        args = tool_call.get("arguments")

        # Parse args if string
        if isinstance(args, str):
            try:
                import json

                args_obj = json.loads(args)
                # Pretty print JSON args
                args_str = json.dumps(args_obj, indent=2, ensure_ascii=False)
            except Exception:
                args_obj = {"_raw": args}
                args_str = args
        else:
            import json

            args_obj = args if isinstance(args, dict) else {}
            args_str = json.dumps(args, indent=2, ensure_ascii=False)

        if not getattr(self, "_show_agent_details", False):
            self.console.print(f"[yellow]-> {self._tool_call_summary(name, args_obj)}[/]")
            return

        self._print_block(
            Syntax(args_str, "json", theme="monokai", word_wrap=True, background_color="default"),
            f"Tool Call: {name}",
            "yellow",
            "🔧",
        )

    def _tool_call_summary(self, name: str, args: dict) -> str:
        """Return one-line summary for compact tool-call rendering."""
        if name == "file-read":
            path = self._tool_argument_path(args)
            return f"{name} {path}".strip()
        if name == "code-search":
            query = args.get("query") or args.get("pattern")
            return f"{name} {query}".strip()
        if name == "code-edit":
            path = self._tool_argument_path(args)
            operation = args.get("operation") or args.get("op")
            suffix = " ".join(str(part) for part in (operation, path) if part)
            return f"{name} {suffix}".strip()
        if name == "command-exec":
            command = args.get("command") or args.get("cmd")
            first_line = str(command).splitlines()[0][:100] if command else ""
            return f"{name} {first_line}".strip()
        for value in args.values():
            if isinstance(value, str) and value.strip():
                return f"{name} {value.strip()[:100]}".strip()
        return name or "tool"

    def _tool_argument_path(self, args: dict) -> str | None:
        """Return a path-like argument from known tool schema aliases."""
        for key in ("fileName", "filename", "filepath", "file_path", "path", "file"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _display_tool_result(self, result_data):
        """Display tool result in a panel with format-aware rendering."""
        name = result_data.get("name")
        model_result = result_data.get("result", "")
        display_result = result_data.get("display_result")
        result = display_result if display_result is not None else model_result
        policy = result_data.get("display_policy")

        if policy == "error":
            self._print_block(result, f"Error: {name}", "red", "❌")
            return

        if policy == "hidden" and not getattr(self, "_show_agent_details", False):
            summary = result_data.get("display_summary") or f"{name} result hidden"
            self.console.print(f"[dim]· {summary}[/]")
            return

        if policy == "compact" and not getattr(self, "_show_agent_details", False):
            self.console.print(f"[green]✔ {self._tool_result_summary(name, result_data)}[/]")
            return

        if (
            result_data.get("masked") or self._is_compacted_tool_output(model_result)
        ) and not getattr(self, "_show_agent_details", False):
            self.console.print(f"[green]✔ {self._tool_result_summary(name, result_data)}[/]")
            return

        if result_data.get("masked") or self._is_compacted_tool_output(model_result):
            self._display_large_tool_output_result(name, result_data, result, model_result)
            return

        # Diff results (code-edit) — syntax-highlighted diff
        if self._is_diff_result(result):
            self._display_diff_result(name, result)
            return

        # File read — show with line numbers
        if name == "file-read" and result:
            display = result[:800] + "\n..." if len(result) > 800 else result
            syntax = Syntax(
                display,
                "text",
                theme="monokai",
                line_numbers=True,
                background_color="default",
            )
            self._print_block(syntax, f"Result: {name}", "green", "✔")
            return

        # Command exec — show with shell highlighting
        if name == "command-exec" and result:
            display = result[:800] + "\n..." if len(result) > 800 else result
            syntax = Syntax(
                display,
                "bash",
                theme="monokai",
                line_numbers=False,
                background_color="default",
            )
            self._print_block(syntax, f"Result: {name}", "green", "✔")
            return

        # Default: truncated dim text
        display_result = result[:500] + "..." if len(result) > 500 else result
        self._print_block(f"[dim]{display_result}[/]", f"Result: {name}", "green", "✔")

    def _tool_result_summary(self, name: str, result_data: dict) -> str:
        """Return a compact one-line summary for tool result rendering."""
        summary = result_data.get("display_summary")
        if summary:
            return summary

        parts = [str(name or "tool")]
        original_size = result_data.get("original_size")
        offload_path = result_data.get("offload_path")
        if isinstance(original_size, int) and original_size > 0:
            parts.append(f"{original_size:,} chars")
        if offload_path:
            parts.append(f"saved to {offload_path}")
        return " · ".join(parts)

    def _is_diff_result(self, result: str) -> bool:
        """Check if result contains unified diff format."""
        return self._split_unified_diff_result(result) is not None

    def _display_diff_result(self, name: str, result: str):
        """Display a result containing diff with syntax highlighting."""
        split_result = self._split_unified_diff_result(result)
        if split_result is None:
            self._print_block(f"[dim]{result}[/]", f"Result: {name}", "green", "✔")
            return
        message, diff_text = split_result

        # Display message part (success message)
        if message:
            self.console.print(f"[bold green]✔ {name}[/]: {message}")

        # Display diff with syntax highlighting
        if diff_text:
            diff_text = self._bounded_diff_text(diff_text)
            syntax = Syntax(
                diff_text, "diff", theme="monokai", line_numbers=False, background_color="default"
            )
            self._print_block(syntax, "Changes", "cyan", "📝")

    def _split_unified_diff_result(self, result: str) -> tuple[str, str] | None:
        """Split a tool result into message and strict unified diff parts."""
        if not result:
            return None
        lines = result.split("\n")
        start = self._find_unified_diff_start(lines)
        if start is None:
            return None
        message = "\n".join(lines[:start]).strip()
        diff_text = "\n".join(lines[start:]).strip("\n")
        return message, diff_text

    def _find_unified_diff_start(self, lines: list[str]) -> int | None:
        """Find a real unified diff header, not compact head/tail markers."""
        for idx in range(len(lines) - 2):
            if not lines[idx].startswith("--- "):
                continue
            if lines[idx].startswith("--- head") or lines[idx].startswith("--- tail"):
                continue
            if not lines[idx + 1].startswith("+++ "):
                continue
            if any(line.startswith("@@") for line in lines[idx + 2 :]):
                return idx
        return None

    def _bounded_diff_text(self, diff_text: str, max_lines: int = 260) -> str:
        """Keep very large diffs readable in terminal scrollback."""
        lines = diff_text.splitlines()
        if len(lines) <= max_lines:
            return diff_text
        head_count = 180
        tail_count = 60
        hidden = len(lines) - head_count - tail_count
        return "\n".join(
            [
                *lines[:head_count],
                f"... {hidden:,} diff lines hidden ...",
                *lines[-tail_count:],
            ]
        )

    def _is_compacted_tool_output(self, result: str) -> bool:
        """Return True for model-facing compacted tool output payloads."""
        return result.startswith("[Tool output compacted]")

    def _display_large_tool_output_result(
        self, name: str, result_data: dict, display_result: str, model_result: str
    ) -> None:
        """Display a user-friendly large-output preview."""
        text = display_result
        if self._is_compacted_tool_output(model_result) and display_result == model_result:
            text = self._friendly_compacted_tool_output(model_result)

        original_size = result_data.get("original_size")
        offload_path = result_data.get("offload_path")
        header = Text()
        header.append("Large tool output", style="bold")
        if isinstance(original_size, int) and original_size > 0:
            header.append(f" · {original_size:,} chars", style="cyan")
        if offload_path:
            header.append("\nFull output saved to: ", style="bold")
            header.append(str(offload_path), style="yellow")

        syntax = Syntax(
            text,
            "text",
            theme="monokai",
            line_numbers=False,
            word_wrap=True,
            background_color="default",
        )
        self._print_block(Group(header, syntax), f"Result: {name}", "green", "✔")

    def _friendly_compacted_tool_output(self, result: str) -> str:
        """Convert legacy model-facing compact output into user-facing text."""
        original_size = self._extract_compacted_field(result, "Original size")
        saved_to = self._extract_compacted_field(result, "Full output saved to")
        omitted = self._extract_compacted_field(result, "Omitted middle")
        head = ""
        tail = ""
        if "\n--- head ---\n" in result:
            _prefix, rest = result.split("\n--- head ---\n", 1)
            if "\n--- tail ---\n" in rest:
                head, tail = rest.split("\n--- tail ---\n", 1)
            else:
                head = rest

        parts = []
        if original_size:
            parts.append(f"Original size: {original_size}")
        if saved_to:
            parts.append(f"Full output saved to: {saved_to}")
        if omitted:
            parts.append(f"Hidden middle: {omitted}")
        if head:
            parts.extend(["", "Preview head:", head.rstrip()])
        if tail:
            parts.extend(["", "Preview tail:", tail.lstrip()])
        return "\n".join(parts)

    def _extract_compacted_field(self, result: str, field: str) -> str:
        match = re.search(rf"^{re.escape(field)}:\s*(.*)$", result, flags=re.MULTILINE)
        return match.group(1).strip() if match else ""

    # Commands

    def _get_prompt(self) -> str:
        """Get prompt string with model tag and current mode icon.

        The prompt color is applied via the prompt_toolkit ``PromptStyle``
        (see _setup_session / _refresh_prompt_style), NOT via rich markup —
        prompt_toolkit does not interpret rich markup strings.
        """
        model_tag = self.agent.llm.config.model.split("/")[-1][:15]
        config = MODE_CONFIGS[self.agent.mode]
        return f"[{model_tag}] {config.icon} "

    def _get_bottom_toolbar(self):
        """Return the prompt bottom toolbar with the active mode.

        Returns a list of (style, text) tuples so prompt_toolkit can render
        the mode token in the mode color while keeping the rest dim.
        """
        config = MODE_CONFIGS[self.agent.mode]
        return [
            ("class:bottom-toolbar", " "),
            (f"fg:{config.color} bold", f"{config.icon} {config.name.upper()}"),
            ("class:bottom-toolbar", f"  ·  {config.toolbar}  ·  Shift+Tab cycles"),
        ]

    def _refresh_prompt_style(self) -> None:
        """Rebuild the prompt_toolkit style so the prompt color tracks the mode.

        Called after every mode change (_cycle_mode, cmd_* setters) because
        PromptStyle is built once at session setup and does not observe mode
        changes on its own.
        """
        if not getattr(self, "session", None):
            return
        config = MODE_CONFIGS[self.agent.mode]
        from prompt_toolkit.styles import Style as PtStyle

        self.session.style = PtStyle.from_dict(
            {
                "prompt": f"fg:{config.color} bold",
                "bottom-toolbar": "#666666",
            }
        )

    def _cycle_mode(self) -> AgentMode:
        """Switch to the next mode in the keyboard cycle."""
        current_idx = MODE_CYCLE.index(self.agent.mode)
        next_mode = MODE_CYCLE[(current_idx + 1) % len(MODE_CYCLE)]
        self._set_mode(next_mode)
        return next_mode

    def _set_mode(self, mode: AgentMode) -> None:
        """Switch the agent mode and refresh prompt_toolkit styling to match.

        Centralizes the mode switch so every entry point (Shift+Tab cycle,
        /ask, /plan, /code, /accept-edits) keeps the prompt color in sync.
        """
        self.agent.set_mode(mode)
        self._refresh_prompt_style()

    def cmd_ask(self, user_input: str):
        """Switch to ask mode or ask a question without editing.

        /ask          - Switch to ask mode (sticky)
        /ask <text>   - Ask one question in ask mode, then return to previous mode
        """
        # Extract text after /ask command
        parts = user_input.split(maxsplit=1)
        question = parts[1].strip() if len(parts) > 1 else ""

        if question:
            # One-shot ask: execute in ask mode, then return
            original_mode = self.agent.mode
            self._set_mode(AgentMode.ASK)
            try:
                self._handle_chat(question)
            finally:
                self._set_mode(original_mode)
        else:
            # Sticky switch to ask mode
            self._set_mode(AgentMode.ASK)
            self.console.print("[cyan]Switched to ask mode[/] - questions only, no edits")
            self.console.print("[dim]Use /plan, /code, or /accept-edits to switch modes[/]")
        return False

    def cmd_plan(self, user_input: str):
        """Switch to plan mode or handle one request in plan mode."""
        parts = user_input.split(maxsplit=1)
        request = parts[1].strip() if len(parts) > 1 else ""

        if request:
            original_mode = self.agent.mode
            self._set_mode(AgentMode.PLAN)
            try:
                self._handle_chat(request)
            finally:
                self._set_mode(original_mode)
        else:
            self._set_mode(AgentMode.PLAN)
            self.console.print("[cyan]Switched to plan mode[/] - read/search plus dated plans only")
        return False

    def cmd_code(self, user_input: str):
        """Switch to code mode (file edits require approval).

        /code         - Switch to code mode (sticky)
        /code <text>  - Execute one request in code mode
        """
        parts = user_input.split(maxsplit=1)
        request = parts[1].strip() if len(parts) > 1 else ""

        if request:
            # One-shot code request
            original_mode = self.agent.mode
            self._set_mode(AgentMode.CODE)
            try:
                self._handle_chat(request)
            finally:
                self._set_mode(original_mode)
        else:
            # Sticky switch to code mode
            self._set_mode(AgentMode.CODE)
            self.console.print("[cyan]Switched to code mode[/] - edits ask for approval")
            self.console.print("[dim]Use /accept-edits to apply edits without per-edit prompts[/]")
        return False

    def cmd_accept_edits(self, user_input: str):
        """Switch to accept-edits mode or handle one request with edits enabled."""
        parts = user_input.split(maxsplit=1)
        request = parts[1].strip() if len(parts) > 1 else ""

        if request:
            original_mode = self.agent.mode
            self._set_mode(AgentMode.ACCEPT_EDITS)
            try:
                self._handle_chat(request)
            finally:
                self._set_mode(original_mode)
        else:
            self._set_mode(AgentMode.ACCEPT_EDITS)
            self.console.print("[cyan]Switched to accept-edits mode[/] - file edits are enabled")
        return False

    def cmd_clear(self, _):
        self.agent.clear_history()
        self.console.print("[dim]History cleared[/]")
        return False

    def cmd_compact(self, _):
        """Compact context by summarizing it."""
        # Check if there's anything to compact
        stats = self.agent.context.get_stats()
        if stats.message_count == 0:
            self.console.print("[yellow]No context to compact[/]")
            return False

        self.console.print(
            f"[dim]Current context: {stats.used_tokens:,} tokens, {stats.message_count} messages[/]"
        )

        self.agent.abort_controller.reset()
        try:
            # Show spinner while compacting
            with self.console.status("[bold blue]Compacting context...[/]", spinner="dots"):
                summary, stats_before, stats_after = self.agent.compact_context()
        except AgentAbortedError:
            self._print_block(
                "[bold yellow]Context compact was interrupted by user (ESC)[/]",
                "Compact Interrupted",
                "yellow",
                "⚠",
            )
            return False
        finally:
            self.agent.abort_controller.reset()

        if summary.startswith("Error generating summary:"):
            self._print_block(f"[red]{summary}[/]", "Compact Failed", "red", "❌")
            return False

        # Display results
        tokens_saved = stats_before.used_tokens - stats_after.used_tokens
        reduction_pct = (
            (tokens_saved / stats_before.used_tokens * 100) if stats_before.used_tokens > 0 else 0
        )

        self.console.print("\n[green]✓ Context compacted![/]")
        self.console.print(
            f"  [dim]Before:[/] {stats_before.used_tokens:,} tokens ({stats_before.message_count} messages)"
        )
        self.console.print(
            f"  [dim]After:[/]  {stats_after.used_tokens:,} tokens ({stats_after.message_count} messages)"
        )
        self.console.print(
            f"  [dim]Saved:[/]  {tokens_saved:,} tokens ({reduction_pct:.1f}% reduction)"
        )

        # Show summary preview
        self.console.print("\n[bold]Summary preview:[/]")
        preview = summary[:500] + "..." if len(summary) > 500 else summary
        self.console.print(Panel(Markdown(preview), border_style="dim"))

        return False

    def _pick_session(self) -> dict | None:
        """Show interactive session picker. Returns session dict or None."""
        sessions = self.agent.session_manager.list_sessions()

        if not sessions:
            self.console.print("[yellow]No previous sessions found[/]")
            return None

        from questionary import Choice, Style, select

        choices = []
        for s in sessions:
            rel = format_relative_time(s["last_modified"])
            compacted = " (compacted)" if s.get("is_compacted") else ""
            title = s.get("title", "Untitled")
            msg_count = s.get("message_count", 0)
            display = f"{title}{compacted}  {rel} · {msg_count} msgs"
            choices.append(Choice(title=display, value=s))

        style = Style(
            [
                ("qmark", "fg:#00aa00 bold"),
                ("pointer", "fg:#00aa00 bold"),
                ("highlighted", "bold"),
            ]
        )

        return select(
            "Resume which session?",
            choices=choices,
            style=style,
            qmark="▸",
            instruction="(↑↓ navigate, Enter select, Ctrl+C cancel)",
        ).unsafe_ask()

    def cmd_continue(self, _):
        """Continue a previous session."""
        session_meta = self._pick_session()

        if session_meta is None:
            self.console.print("[dim]Cancelled[/]")
            return False

        session_id = session_meta["id"]
        if self.agent.load_session(session_id):
            session = self.agent.current_session
            title = session_meta.get("title", "Untitled")
            rel = format_relative_time(session_meta["last_modified"])
            self.console.print(Rule(f"[bold blue]Restored: {title} — {rel}[/]", style="blue"))
            self._render_session_history(session.messages)
            self.console.print(Rule(style="dim grey50"))
            self.console.print("[green]✓ Session restored — continue the conversation[/]")
        else:
            self.console.print("[red]Failed to load session[/]")

        return False

    def cmd_undo(self, _):
        """Undo changes to a selected checkpoint."""
        checkpoints = self.agent.checkpoint_manager.list_checkpoints()

        if not checkpoints:
            self.console.print("[yellow]No checkpoints available[/]")
            return False

        self.console.print("\n[bold]Available Checkpoints:[/]")
        for i, cp in enumerate(checkpoints, 1):
            ts = cp.timestamp[:16].replace("T", " ")
            files_count = len(cp.files)
            self.console.print(f"  [cyan]{i}[/]. {cp.description}")
            self.console.print(f"      [dim]{ts} • {files_count} file(s)[/]")

        self.console.print("\n[dim]Enter checkpoint number (or 'cancel'):[/]")

        try:
            choice = self.session.prompt("Undo> ").strip()

            if choice.lower() == "cancel" or not choice:
                self.console.print("[dim]Cancelled[/]")
                return False

            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                cp = checkpoints[idx]
                undo_result = self.agent.checkpoint_manager.undo_by_id(cp.id)
                if undo_result:
                    self.agent.handle_undo(undo_result.restored)
                    self.console.print(f"[green]✓ Restored to: {cp.description}[/]")
                    for f in undo_result.restored:
                        self.console.print(f"  [dim]Restored: {f}[/]")
                    for f in undo_result.failed:
                        self.console.print(f"  [red]Failed to restore: {f}[/]")
                else:
                    self.console.print("[yellow]No files were restored[/]")
            else:
                self.console.print("[red]Invalid selection[/]")
        except ValueError:
            self.console.print("[red]Invalid selection[/]")
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[dim]Cancelled[/]")

        return False

    def cmd_help(self, _):
        table = Table(
            title="SuperCoder Commands",
            box=box.SIMPLE_HEAVY,
            title_style="bold",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Command", style="green", min_width=14)
        table.add_column("Description")

        # Mode
        table.add_section()
        table.add_row("[bold dim]Mode[/]", "")
        table.add_row("/ask", "Q&A mode (read-only, no edits)")
        table.add_row("/plan", "Planning mode (read/search, save dated plans only)")
        table.add_row("/code", "Code mode (file edits require approval)")
        table.add_row("/accept-edits", "Editing mode (file edits enabled)")
        table.add_row("Shift+Tab", "Cycle ask -> plan -> code -> accept-edits")

        # Context
        table.add_section()
        table.add_row("[bold dim]Context[/]", "")
        table.add_row("/compact", "Summarize and compress context")
        table.add_row("/stats", "Show context window stats")
        table.add_row("/clear", "Clear conversation history")
        table.add_row("@path", "Attach a file or directory listing to the next prompt")

        # Session
        table.add_section()
        table.add_row("[bold dim]Session[/]", "")
        table.add_row("/continue", "Resume a previous session")
        table.add_row("/undo", "Undo changes to a checkpoint")

        # Config
        table.add_section()
        table.add_row("[bold dim]Config[/]", "")
        table.add_row("/models", "List available model profiles")
        table.add_row("/model <name>", "Switch to a model profile")
        table.add_row("/config", "Show current configuration")
        table.add_row("/permissions", "Show or manage saved command approvals")
        table.add_row("/debug", "Toggle debug mode")

        table.add_section()
        table.add_row("/exit", "Quit SuperCoder")

        self.console.print(table)
        return False

    def cmd_config(self, _):
        """Show current configuration in a table."""
        config = self.agent.llm.config
        masked_key = (
            f"{config.api_key[:4]}...{config.api_key[-4:]}" if config.api_key else "Not Set"
        )

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Key", style="cyan", min_width=14)
        table.add_column("Value")

        table.add_row("Model", config.model)
        table.add_row("Base URL", config.base_url)
        table.add_row("Temperature", str(config.temperature))
        table.add_row("Context Size", f"{config.max_context_tokens:,}")
        table.add_row("Debug Mode", str(config.debug))
        table.add_row("API Key", masked_key)

        self._print_block(table, "Configuration", "cyan", "⚙")
        return False

    def cmd_permissions(self, user_input: str):
        """Show or manage project-local command permission rules."""
        parts = user_input.split(maxsplit=2)
        policy = self.agent.permission_policy

        if len(parts) == 1:
            rules = policy.list_command_rules()
            table = Table(
                title="Command Permissions",
                box=box.SIMPLE,
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("ID", style="cyan", min_width=4)
            table.add_column("Scope", min_width=10)
            table.add_column("Action", min_width=6)
            table.add_column("Rule")

            if rules:
                for rule in rules:
                    action_style = "green" if rule.action == PermissionAction.ALLOW else "red"
                    table.add_row(
                        rule.id,
                        rule.scope,
                        f"[{action_style}]{rule.action.value}[/]",
                        rule.pattern,
                    )
            else:
                table.add_row("-", "-", "-", "No session or persistent command rules")

            self.console.print(table)
            self.console.print(
                f"[dim]Persistent rules file: {policy.persistent_path}[/]\n"
                "[dim]Use /permissions remove <id> or /permissions clear[/]"
            )
            return False

        action = parts[1].lower()
        if action == "remove":
            if len(parts) < 3:
                self.console.print("[yellow]Usage: /permissions remove <id>[/]")
                return False
            removed = policy.remove_persistent_command_rule(parts[2])
            if not removed:
                self.console.print(f"[red]No persistent permission rule found: {parts[2]}[/]")
                return False
            get_logger().log_permission_rule_change(
                action="remove",
                scope=removed.scope,
                rule_action=removed.action.value,
                rule=removed.pattern,
                source="/permissions",
            )
            self.console.print(f"[green]✓ Removed {removed.id}: {removed.pattern}[/]")
            return False

        if action == "clear":
            count = policy.clear_persistent_command_rules()
            get_logger().log_permission_rule_change(
                action="clear",
                scope="persistent",
                rule_action="all",
                rule="*",
                source="/permissions",
            )
            self.console.print(f"[green]✓ Cleared {count} persistent permission rule(s)[/]")
            return False

        self.console.print("[yellow]Usage: /permissions [remove <id>|clear][/]")
        return False

    def cmd_stats(self, _):
        """Show context stats with a visual progress bar."""
        stats = self.agent.context.get_stats()
        config = self.agent.llm.config

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Label", style="cyan", min_width=10)
        table.add_column("Value")

        # Progress bar (unified renderer; stats uses the wider width).
        bar = render.render_context_bar(
            stats.used_tokens, stats.total_tokens, width=theme.BAR_WIDTH_STATS
        )

        table.add_row(
            "Context",
            f"{bar}  {stats.utilization_percent:.1f}%",
        )
        table.add_row("Messages", str(stats.message_count))
        table.add_row("Available", f"{stats.available_tokens:,} tokens")
        table.add_row("Model", config.model)
        table.add_row("Mode", self.agent.mode.value.upper())

        self._print_block(table, "Context Stats", "cyan", "📊")
        return False

    def cmd_debug(self, _):
        self.agent.set_debug(not self.agent.debug)
        self.console.print(f"[dim]Debug mode: {self.agent.debug}[/]")
        return False

    def cmd_exit(self, _):
        return True

    def cmd_quit(self, _):
        return True

    def cmd_models(self, _):
        """List available model profiles in a table."""
        config = self.agent.llm.config
        current = config.current_profile_name
        models = config.get_available_models()

        if not models:
            self.console.print("[yellow]No model profiles defined in config[/]")
            return False

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Profile", style="cyan")
        table.add_column("Model")
        table.add_column("Status")

        for name in models:
            profile = config.get_model_profile(name)
            status = "[green]● active[/]" if name == current else "[dim]○[/]"
            table.add_row(name, profile.model, status)

        self._print_block(table, "Model Profiles", "cyan", "🤖")
        self.console.print("[dim]Use /model <name> to switch[/]")
        return False

    def cmd_model(self, user_input: str):
        """Switch to a different model profile."""
        parts = user_input.split()

        if len(parts) < 2:
            self.console.print("[yellow]Usage: /model <profile-name>[/]")
            self.console.print("[dim]Use /models to see available profiles[/]")
            return False

        profile_name = parts[1]
        config = self.agent.llm.config
        profile = config.get_model_profile(profile_name)

        if not profile:
            available = ", ".join(config.get_available_models())
            self.console.print(f"[red]Unknown profile: {profile_name}[/]")
            self.console.print(f"[dim]Available: {available}[/]")
            return False

        # Switch in config
        config.switch_to_model(profile_name)

        # Switch in LLM client
        self.agent.llm.switch_model(profile)

        # Update lean mode before rebuilding prompt
        self.agent.lean = profile.lean

        # Update tool calling type in agent (rebuilds system prompt if needed)
        self.agent.set_tool_calling_type(profile.tool_calling_type)

        # Update context window limit if model has specific setting
        context_info = ""
        if profile.max_context_tokens:
            self.agent.context.set_max_tokens(profile.max_context_tokens)
            context_info = f"{profile.max_context_tokens:,}"

        # Reset prompt_toolkit buffer to prevent double input issue
        # This clears any stale state that might cause the next input to be processed twice
        try:
            if hasattr(self.session, "app") and self.session.app is not None:
                self.session.app.current_buffer.reset()
        except Exception:
            pass  # Ignore if not in active input session

        self.console.print(f"[green]✓ Switched to {profile_name}[/]")
        self.console.print(f"[dim]Model: {profile.model}[/]")
        self.console.print(f"[dim]Endpoint: {profile.endpoint}[/]")
        self.console.print(f"[dim]Tool calling: {profile.tool_calling_type}[/]")
        if context_info:
            self.console.print(f"[dim]Context: {context_info} tokens[/]")
        return False
