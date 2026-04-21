"""Interactive REPL for SuperCoder."""

import sys
import threading
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style as PromptStyle
from pygments.lexers.markup import MarkdownLexer
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import __version__
from .abort_controller import InterruptHandler, KeyboardListener
from .agent.agent_modes import AgentMode
from .utils import format_relative_time


class SuperCoderREPL:
    """Interactive Read-Eval-Print Loop for SuperCoder."""

    def __init__(self, agent):
        self.agent = agent
        self.console = Console()

        # Initialize commands BEFORE session setup (session uses commands for autocomplete)
        self.commands = {
            "/ask": self.cmd_ask,
            "/code": self.cmd_code,
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
        # Use direct print to avoid conflict with Rich spinner in main thread
        # \x1b[33m is Yellow, \x1b[0m is Reset
        print("\r\x1b[33mPress ESC again to interrupt\x1b[0m", end="", flush=True)

    def _setup_session(self):
        """Configure prompt_toolkit session."""
        from prompt_toolkit.completion import ThreadedCompleter
        from prompt_toolkit.key_binding import KeyBindings

        from .autocomplete import AutoCompleter, SlashCommandAutoSuggest

        style = PromptStyle.from_dict(
            {
                "prompt": "#00aa00 bold",
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
                self.console.print(f"[bold green]{self._get_prompt()}[/][on grey23]{user_input}[/]")
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
                    self._print_block(content.strip(), "Reasoning", "magenta", "💭")
                    spinner.update("[bold blue]SuperCoder is thinking...[/]")
                    spinner.start()

                elif event_type == "response":
                    spinner.stop()
                    # Full response — render as Markdown
                    self.console.print(Markdown(content))
                    spinner.update("[bold blue]SuperCoder is thinking...[/]")
                    spinner.start()

                elif event_type == "tool_call":
                    spinner.stop()
                    self._display_tool_call(content)
                    self._track_files(content, touched_files)
                    name = content.get("name", "tool")
                    spinner.update(f"[bold blue]Executing {name}...[/]")
                    spinner.start()

                elif event_type == "tool_result":
                    spinner.stop()
                    self._display_tool_result(content)
                    _gen_tokens[0] = 0
                    _gen_start = time.monotonic()
                    _gen_phase[0] = "tool call"
                    spinner.update("[bold blue]SuperCoder is thinking...[/]")
                    spinner.start()

                elif event_type == "error":
                    errors.append(content)

                elif event_type == "rollback":
                    rollback_info = content

                elif event_type == "command_confirm":
                    spinner.stop()
                    if hasattr(self, "keyboard_listener"):
                        self.keyboard_listener.stop()
                    approved = self._handle_command_confirm(content.get("command", ""))
                    event["result"]["approved"] = approved
                    if hasattr(self, "keyboard_listener"):
                        self.keyboard_listener.start()
                    spinner.update("[bold blue]Running command...[/]")
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

        # === Post-processing ===
        if rollback_info:
            files = rollback_info.get("files", [])
            reason = rollback_info.get("reason", "Unknown")
            rollback_content = f"[dim]Reason: {reason}[/]\n" + "\n".join(
                f"  ✓ Restored: {f}" for f in files
            )
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
                    approved = self._handle_command_confirm(content.get("command", ""))
                    event["result"]["approved"] = approved
                    # Restart listener for the upcoming LLM turn
                    if hasattr(self, "keyboard_listener"):
                        self.keyboard_listener.start()
                    spinner.update("[bold blue]Running command...[/]")
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
            files = rollback_info.get("files", [])
            reason = rollback_info.get("reason", "Unknown")
            rollback_content = f"[dim]Reason: {reason}[/]\n" + "\n".join(
                f"  ✓ Restored: {f}" for f in files
            )
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
        """Display a status footer with mini progress bar, token usage, and active files."""
        stats = self.agent.context.get_stats()

        # Mini progress bar (8 chars)
        bar_w = 8
        filled = int(bar_w * stats.utilization_percent / 100)
        empty = bar_w - filled
        color = (
            "green"
            if stats.utilization_percent < 50
            else "yellow"
            if stats.utilization_percent < 80
            else "red"
        )
        bar = f"[{color}]{'━' * filled}[/][dim]{'━' * empty}[/]"

        parts = [f"{bar} [dim]{stats.used_tokens:,}/{stats.total_tokens:,} tokens[/]"]

        if touched_files:
            files_str = ", ".join(sorted(touched_files))
            parts.append(f"[dim]Active: {files_str}[/]")

        self.console.print(" │ ".join(parts), justify="right")

    def _render_session_history(self, messages: list) -> None:
        """Render session messages visually after restore.

        Reuses the same rendering helpers as live output (_print_block,
        _display_tool_call, _display_tool_result) to maintain visual fidelity.
        Tool calls are interleaved with their matching tool results by tool_call_id.
        """
        import json

        MAX_SHOW = 30

        # Filter out system messages
        showable = [m for m in messages if m.role != "system"]
        to_show = showable[-MAX_SHOW:] if len(showable) > MAX_SHOW else showable

        if len(showable) > MAX_SHOW:
            skipped = len(showable) - MAX_SHOW
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
                self._print_block(msg.content, "You", "cyan", "👤")

            elif dt == "thinking":
                text = msg.content[:500] + ("..." if len(msg.content) > 500 else "")
                self._print_block(text, "Reasoning", "magenta", "💭")

            elif dt in ("response", "tool_call"):
                # Render text content
                if msg.content and msg.content.strip():
                    self.console.print(Markdown(msg.content))

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
                        self._display_tool_call({"name": name, "arguments": args_obj})

                        # Find and render matching tool result
                        tc_id = tc.get("id", "")
                        j = result_index.get(tc_id)
                        if j is not None and j not in consumed:
                            result_msg = to_show[j]
                            self._display_tool_result(
                                {"name": result_msg.name or name, "result": result_msg.content}
                            )
                            consumed.add(j)

            elif dt == "tool_result":
                # Only render if not already consumed by interleaving above
                self._display_tool_result({"name": msg.name or "tool", "result": msg.content})

            elif dt == "error":
                self._print_block(msg.content, "Error", "red", "❌")

            elif dt == "compact_summary":
                text = msg.content[:200]
                self._print_block(f"[dim]{text}...[/]", "Context Summary", "dim", "📋")

            else:
                # Fallback for old sessions without display_type
                if msg.role == "user" and msg.content:
                    self._print_block(msg.content, "You", "cyan", "👤")
                elif msg.role == "assistant" and msg.content:
                    self.console.print(Markdown(msg.content))
                elif msg.role == "tool":
                    self._display_tool_result({"name": msg.name or "tool", "result": msg.content})

            i += 1

    def _print_block(self, content, title: str, color: str, icon: str = ""):
        """Print content in a panel with horizontal lines only (no vertical borders).

        Args:
            content: Rich renderable (Text, Markdown, Syntax, str)
            title: Block title (e.g. "Reasoning", "Tool Call")
            color: Color for the lines (e.g. "magenta", "yellow")
            icon: Optional emoji icon
        """
        full_title = f"[bold {color}]{icon} {title}[/]" if icon else f"[bold {color}]{title}[/]"
        self.console.print(
            Panel(content, title=full_title, border_style=color, box=box.HORIZONTALS)
        )

    def _handle_command_confirm(self, command: str) -> bool:
        """Ask user to approve or deny a shell command before it runs.

        Returns True if user approved, False otherwise.
        """
        from prompt_toolkit import prompt as pt_prompt
        from prompt_toolkit.key_binding import KeyBindings

        self._print_block(
            f"[bold]Command:[/]\n[yellow]{command}[/]",
            "Run Command?",
            "yellow",
            "⚡",
        )

        kb = KeyBindings()

        @kb.add("y")
        @kb.add("Y")
        def _(event):
            event.app.exit(result="yes")

        @kb.add("a")
        @kb.add("A")
        def _(event):
            event.app.exit(result="always")

        @kb.add("n")
        @kb.add("N")
        @kb.add("escape")
        @kb.add("enter")
        def _(event):
            event.app.exit(result="no")

        self.console.print(
            "  [bold green][[y]][/bold green] Yes   "
            "[bold cyan][[a]][/bold cyan] Always allow   "
            "[bold red][[n]][/bold red] No"
        )
        try:
            choice = pt_prompt("  > ", key_bindings=kb)
        except (KeyboardInterrupt, EOFError):
            choice = "no"

        if choice == "yes":
            self.console.print("[green]✓ Approved[/]")
            return True
        if choice == "always":
            self.console.print("[green]✓ Approved (always allowed this session)[/]")
            return True
        self.console.print("[red]✗ Cancelled[/]")
        return False

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
                args_str = json.dumps(args_obj, indent=2)
            except Exception:
                args_str = args
        else:
            import json

            args_str = json.dumps(args, indent=2)

        self._print_block(
            Syntax(args_str, "json", theme="monokai", word_wrap=True, background_color="default"),
            f"Tool Call: {name}",
            "yellow",
            "🔧",
        )

    def _display_tool_result(self, result_data):
        """Display tool result in a panel with format-aware rendering."""
        name = result_data.get("name")
        result = result_data.get("result", "")

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

    def _is_diff_result(self, result: str) -> bool:
        """Check if result contains unified diff format."""
        if not result:
            return False
        # Look for unified diff markers (--- and +++ may be at start or after newline)
        has_minus = result.startswith("---") or "\n---" in result
        has_plus = "\n+++" in result
        return has_minus and has_plus

    def _display_diff_result(self, name: str, result: str):
        """Display a result containing diff with syntax highlighting."""
        # Split result into message and diff parts
        lines = result.split("\n")
        message_lines = []
        diff_lines = []
        in_diff = False

        for line in lines:
            # Check for unified diff markers to start capturing diff
            # --- file header, +++ file header, @@ hunk header
            if line.startswith("--- ") or line.startswith("+++ ") or line.startswith("@@"):
                in_diff = True

            if in_diff:
                # Once in diff mode, capture all lines (including +/- content lines)
                diff_lines.append(line)
            else:
                message_lines.append(line)

        # Display message part (success message)
        if message_lines:
            message = "\n".join(message_lines).strip()
            if message:
                self.console.print(f"[bold green]✔ {name}[/]: {message}")

        # Display diff with syntax highlighting
        if diff_lines:
            diff_text = "\n".join(diff_lines)
            syntax = Syntax(
                diff_text, "diff", theme="monokai", line_numbers=False, background_color="default"
            )
            self._print_block(syntax, "Changes", "cyan", "📝")

    # Commands

    def _get_prompt(self) -> str:
        """Get prompt string with model tag and current mode."""
        model_tag = self.agent.llm.config.model.split("/")[-1][:15]
        mode = "ask> " if self.agent.mode == AgentMode.ASK else "You> "
        return f"[{model_tag}] {mode}"

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
            self.agent.set_mode(AgentMode.ASK)
            try:
                self._handle_chat(question)
            finally:
                self.agent.set_mode(original_mode)
        else:
            # Sticky switch to ask mode
            self.agent.set_mode(AgentMode.ASK)
            self.console.print("[cyan]Switched to ask mode[/] - questions only, no edits")
            self.console.print("[dim]Use /code to switch back to editing mode[/]")
        return False

    def cmd_code(self, user_input: str):
        """Switch to code mode (can edit files).

        /code         - Switch to code mode (sticky)
        /code <text>  - Execute one request in code mode
        """
        parts = user_input.split(maxsplit=1)
        request = parts[1].strip() if len(parts) > 1 else ""

        if request:
            # One-shot code request
            original_mode = self.agent.mode
            self.agent.set_mode(AgentMode.CODE)
            try:
                self._handle_chat(request)
            finally:
                self.agent.set_mode(original_mode)
        else:
            # Sticky switch to code mode
            self.agent.set_mode(AgentMode.CODE)
            self.console.print("[cyan]Switched to code mode[/] - can edit files")
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

        # Show spinner while compacting
        with self.console.status("[bold blue]Compacting context...[/]", spinner="dots"):
            summary, stats_before, stats_after = self.agent.compact_context()

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
                restored = self.agent.checkpoint_manager.undo_by_id(cp.id)
                if restored:
                    self.agent.handle_undo(restored)
                    self.console.print(f"[green]✓ Restored to: {cp.description}[/]")
                    for f in restored:
                        self.console.print(f"  [dim]Restored: {f}[/]")
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
        table.add_row("/code", "Code mode (can edit files)")

        # Context
        table.add_section()
        table.add_row("[bold dim]Context[/]", "")
        table.add_row("/compact", "Summarize and compress context")
        table.add_row("/stats", "Show context window stats")
        table.add_row("/clear", "Clear conversation history")

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

    def cmd_stats(self, _):
        """Show context stats with a visual progress bar."""
        stats = self.agent.context.get_stats()
        config = self.agent.llm.config

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Label", style="cyan", min_width=10)
        table.add_column("Value")

        # Progress bar
        bar_width = 20
        filled = int(bar_width * stats.utilization_percent / 100)
        empty = bar_width - filled
        color = (
            "green"
            if stats.utilization_percent < 50
            else "yellow"
            if stats.utilization_percent < 80
            else "red"
        )
        bar = f"[{color}]{'━' * filled}[/][dim]{'━' * empty}[/]"

        table.add_row(
            "Context",
            f"{bar}  {stats.utilization_percent:.1f}%   {stats.used_tokens:,} / {stats.total_tokens:,}",
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
