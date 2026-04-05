"""Interactive REPL for SuperCoder."""

import sys

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
from rich.text import Text

from . import __version__
from .abort_controller import InterruptHandler, KeyboardListener
from .agent.agent_modes import AgentMode


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
            "/sessions": self.cmd_sessions,
            "/undo": self.cmd_undo,
            "/help": self.cmd_help,
            "/tools": self.cmd_tools,
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

        from .autocomplete import AutoCompleter

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

        # Key bindings for multiline support
        kb = KeyBindings()

        @kb.add("escape", "enter")  # Alt+Enter or Escape then Enter
        def _(event):
            """Insert newline without submitting."""
            event.current_buffer.insert_text("\n")

        # History file in project-specific directory
        history_path = self.agent.repo_root / ".supercoder" / "history"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        return PromptSession(
            history=FileHistory(str(history_path)),
            lexer=PygmentsLexer(MarkdownLexer),
            style=style,
            completer=completer,
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
                prompt_prefix_len = 5  # "You> "

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
        """Handle chat interaction with streaming output.

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
        accumulated_display = ""  # All safe text received so far
        _paragraphs_printed = 0  # Number of complete paragraphs already printed

        # --- Spinner (manual start/stop) ---
        spinner = self.console.status(
            "[bold blue]SuperCoder is thinking...[/]", spinner="dots"
        )
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
            nonlocal is_streaming, display_buffer, accumulated_display, _paragraphs_printed
            flush_reasoning()
            spinner.stop()
            display_buffer = StreamingDisplayBuffer(self.agent.tool_calling_type)
            accumulated_display = ""
            _paragraphs_printed = 0
            is_streaming = True

        def print_new_paragraphs():
            """Print any newly completed paragraphs as Markdown."""
            nonlocal _paragraphs_printed
            # Split accumulated text into paragraphs by \n\n
            parts = accumulated_display.split("\n\n")
            # All except the last part are "complete" paragraphs
            complete = parts[:-1]
            # Print only NEW complete paragraphs
            new_paragraphs = complete[_paragraphs_printed:]
            for para in new_paragraphs:
                if para.strip():
                    self.console.print(Markdown(para))
            _paragraphs_printed = len(complete)

        def stop_streaming():
            """Finalize streaming, print any remaining text."""
            nonlocal is_streaming, display_buffer, accumulated_display, _paragraphs_printed
            if not is_streaming:
                spinner.stop()
                flush_reasoning()
                return

            # Flush remaining buffer content
            remaining = display_buffer.flush()
            accumulated_display += remaining

            # Clean any residual tool call tags (safety net)
            clean = self._filter_special_tokens(accumulated_display)

            # Split the CLEAN text into paragraphs and print unprinted ones
            parts = clean.split("\n\n")
            # Skip already-printed paragraphs, print the rest
            remaining_paragraphs = parts[_paragraphs_printed:]
            for para in remaining_paragraphs:
                if para.strip():
                    self.console.print(Markdown(para))

            display_buffer = None
            accumulated_display = ""
            _paragraphs_printed = 0
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
                        _paragraphs_printed = 0
                        is_streaming = False
                    spinner.stop()
                    reasoning_text = ""

                elif event_type == "rollback":
                    rollback_info = content

                elif event_type == "command_confirm":
                    stop_streaming()
                    approved = self._handle_command_confirm(content.get("command", ""))
                    content["result"]["approved"] = approved
                    spinner.update("[bold blue]Running command...[/]")
                    spinner.start()

                elif event_type == "command_waiting":
                    spinner.stop()
                    flush_reasoning()
                    self._handle_command_waiting(event)
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
        """Display a status footer with token usage and active files."""
        stats = self.agent.context.get_stats()

        parts = []
        # Token usage
        parts.append(f"[dim]Context: {stats.used_tokens:,}/{stats.total_tokens:,} tokens[/]")

        # Active files
        if touched_files:
            files_str = ", ".join(sorted(touched_files))
            parts.append(f"[dim]Active Files: {files_str}[/]")

        # Cost estimate (rough approximation)
        # Assuming generic pricing, just to show we can
        # cost = (stats.used_tokens / 1000) * 0.002 # Example
        # parts.append(f"[dim]Est. Cost: ${cost:.4f}[/]")

        self.console.print(" | ".join(parts), justify="right")

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
        import sys

        self._print_block(
            f"[bold]Command:[/]\n[yellow]{command}[/]",
            "Run Command?",
            "yellow",
            "⚡",
        )
        self.console.print("\n[bold]Allow? [[green]y[/]/[red]N[/]]>[/] ", end="")
        try:
            choice = sys.stdin.readline().strip().lower()
            if choice in ("y", "yes"):
                self.console.print("[green]✓ Approved[/]")
                return True
            self.console.print("[red]✗ Cancelled[/]")
            return False
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[red]✗ Cancelled[/]")
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
        # Clean up extra whitespace - collapse multiple newlines to single
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"  +", " ", text)
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
        """Display tool result in a panel."""
        name = result_data.get("name")
        result = result_data.get("result", "")

        # Check if result contains a diff (unified diff format)
        if self._is_diff_result(result):
            self._display_diff_result(name, result)
            return

        # Truncate long results for display
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
        """Get prompt string based on current mode."""
        if self.agent.mode == AgentMode.ASK:
            return "ask> "
        return "You> "

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

    def cmd_continue(self, _):
        """Continue a previous session."""
        sessions = self.agent.session_manager.list_sessions()

        if not sessions:
            self.console.print("[yellow]No previous sessions found[/]")
            return False

        self.console.print("\n[bold]Available Sessions:[/]")
        for i, session in enumerate(sessions, 1):
            compacted = " [dim](compacted)[/]" if session.get("is_compacted") else ""
            modified = session.get("last_modified", "")[:16].replace("T", " ")  # Format datetime
            title = session.get("title", "Untitled")
            msg_count = session.get("message_count", 0)
            self.console.print(f"  [cyan]{i}[/]. {title}{compacted}")
            self.console.print(f"      [dim]{modified} • {msg_count} messages[/]")

        self.console.print("\n[dim]Enter session number (or 'cancel'):[/]")

        try:
            choice = self.session.prompt("Select> ").strip()

            if choice.lower() == "cancel" or not choice:
                self.console.print("[dim]Cancelled[/]")
                return False

            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                session_id = sessions[idx]["id"]
                if self.agent.load_session(session_id):
                    self.console.print("[green]✓ Resumed session[/]")
                    stats = self.agent.context.get_stats()
                    self.console.print(
                        f"[dim]Loaded {stats.message_count} messages, {stats.used_tokens:,} tokens[/]"
                    )
                else:
                    self.console.print("[red]Failed to load session[/]")
            else:
                self.console.print("[red]Invalid selection[/]")
        except ValueError:
            self.console.print("[red]Invalid selection[/]")
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[dim]Cancelled[/]")

        return False

    def cmd_sessions(self, _):
        """List available sessions."""
        sessions = self.agent.session_manager.list_sessions()

        if not sessions:
            self.console.print("[yellow]No sessions found[/]")
            return False

        self.console.print("\n[bold]Saved Sessions:[/]")
        for session in sessions:
            compacted = " (compacted)" if session.get("is_compacted") else ""
            modified = session.get("last_modified", "")[:16].replace("T", " ")
            title = session.get("title", "Untitled")
            msg_count = session.get("message_count", 0)
            self.console.print(f"  • {title}{compacted}")
            self.console.print(f"    [dim]{modified} • {msg_count} messages[/]")

        self.console.print(f"\n[dim]Total: {len(sessions)} sessions (max 10)[/]")
        self.console.print("[dim]Use /continue to resume a session[/]")
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
        self.console.print("\n[bold]Available Commands:[/]")
        self.console.print("  /ask      - Switch to ask mode (Q&A without edits)")
        self.console.print("  /code     - Switch to code mode (can edit files)")
        self.console.print("  /continue - Resume a previous session")
        self.console.print("  /sessions - List saved sessions")
        self.console.print("  /undo     - Undo changes to a checkpoint")
        self.console.print("  /clear    - Clear conversation history")
        self.console.print("  /compact  - Summarize context and reduce tokens")
        self.console.print("  /stats    - Show context window stats")
        self.console.print("  /tools    - List available tools")
        self.console.print("  /models   - List available model profiles")
        self.console.print("  /model    - Switch model: /model <name>")
        self.console.print("  /config   - Show current configuration")
        self.console.print("  /debug    - Toggle debug mode")
        self.console.print("  /exit     - Quit SuperCoder")
        self.console.print()
        return False

    def cmd_tools(self, _):
        self.console.print("\n[bold]Available Tools:[/]")
        for name, tool in self.agent.tools.items():
            self.console.print(f"  [cyan]{name}[/]: {tool.definition.description[:60]}...")
        self.console.print()
        return False

    def cmd_config(self, _):
        """Show current configuration."""
        self.console.print("\n[bold]Current Configuration:[/]")
        config = self.agent.llm.config  # Access config from valid location

        self.console.print(f"  [cyan]Model[/]: {config.model}")
        self.console.print(f"  [cyan]Base URL[/]: {config.base_url}")
        self.console.print(f"  [cyan]Temperature[/]: {config.temperature}")
        self.console.print(f"  [cyan]Debug Mode[/]: {config.debug}")
        self.console.print(f"  [cyan]Context Size[/]: {config.max_context_tokens}")
        # Hide API key
        masked_key = (
            f"{config.api_key[:4]}...{config.api_key[-4:]}" if config.api_key else "Not Set"
        )
        self.console.print(f"  [cyan]API Key[/]: {masked_key}")
        self.console.print()
        return False

    def cmd_stats(self, _):
        self.console.print(f"[cyan]{self.agent.get_context_stats()}[/]")
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
        """List available model profiles."""
        config = self.agent.llm.config
        current = config.current_profile_name
        models = config.get_available_models()

        if not models:
            self.console.print("[yellow]No model profiles defined in config[/]")
            return False

        self.console.print("\n[bold]Available Model Profiles:[/]")
        for name in models:
            profile = config.get_model_profile(name)
            marker = " [green]← active[/]" if name == current else ""
            self.console.print(f"  [cyan]{name}[/]: {profile.model}{marker}")
        self.console.print()
        self.console.print("[dim]Use /model <name> to switch[/]")
        self.console.print()
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
