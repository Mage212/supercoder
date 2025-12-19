"""Interactive REPL for SuperCoder."""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PromptStyle
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.completion import WordCompleter
from pygments.lexers.python import PythonLexer
from pygments.lexers.markup import MarkdownLexer

class SuperCoderREPL:
    """Interactive Read-Eval-Print Loop for SuperCoder."""
    
    def __init__(self, agent):
        self.agent = agent
        self.console = Console()
        self.session = self._setup_session()
        self.commands = {
            "/clear": self.cmd_clear,
            "/compact": self.cmd_compact,
            "/continue": self.cmd_continue,
            "/sessions": self.cmd_sessions,
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
        
    def _setup_session(self):
        """Configure prompt_toolkit session."""
        style = PromptStyle.from_dict({
            'prompt': '#00aa00 bold',
        })
        
        command_completer = WordCompleter(
            ['/clear', '/compact', '/continue', '/sessions', '/help', '/tools', '/stats', '/debug', '/models', '/model', '/config', '/exit', '/quit'],
            ignore_case=True
        )
        
        # History file in project-specific directory
        history_path = self.agent.repo_root / ".supercoder" / "history"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        return PromptSession(
            history=FileHistory(str(history_path)),
            lexer=PygmentsLexer(MarkdownLexer),
            style=style,
            completer=command_completer,
        )


    def run(self):
        """Start the REPL loop."""
        self.console.print("[bold green]SuperCoder CLI[/] - Type /help for commands")
        self.console.print(f"[dim]Model: {self.agent.llm.model}[/]")
        
        # Start a new session on fresh start
        self.agent.start_new_session()
        
        while True:
            try:
                user_input = self.session.prompt("You> ").strip()
                
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
                import sys
                import shutil
                
                terminal_width = shutil.get_terminal_size().columns
                prompt_prefix_len = 5  # "You> "
                
                # Calculate total visual lines by accounting for terminal wrapping
                visual_lines = 0
                for line in user_input.split('\n'):
                    # First line includes prompt, subsequent lines don't (in prompt_toolkit)
                    if visual_lines == 0:
                        line_len = len(line) + prompt_prefix_len
                    else:
                        line_len = len(line)
                    
                    # Count how many terminal lines this logical line takes
                    if line_len == 0:
                        visual_lines += 1
                    else:
                        visual_lines += (line_len + terminal_width - 1) // terminal_width
                
                # Move up and clear each visual line
                for _ in range(visual_lines):
                    sys.stdout.write("\033[A\033[2K")  # Move up + clear line
                sys.stdout.flush()
                self.console.print(f"[bold green]You>[/] [on grey23]{user_input}[/]")
                self._handle_chat(user_input)
                
            except KeyboardInterrupt:
                self.console.print("\n[dim]Use 'exit' to quit[/]")
                continue
            except EOFError:
                break
        
        self.console.print("[green]Goodbye![/]")

    def _handle_chat(self, message):
        """Handle chat interaction with clean output (no streaming)."""
        # User message is already displayed by the main loop with styling

        # Collect full response first, then display
        response_text = ""
        tool_calls = []
        tool_results = []
        errors = []
        waiting_processes = []  # Track processes waiting for user decision
        
        # Track active files (files touched by tools)
        touched_files = set()
        
        # Show spinner while processing
        with self.console.status("[bold blue]SuperCoder is thinking...[/]", spinner="dots") as status:
            for event in self.agent.chat_stream(message):
                event_type = event.get("type")
                content = event.get("content")
                
                if event_type == "token":
                    response_text += content
                elif event_type == "tool_call":
                    tool_calls.append(content)
                    # Track files from tool args
                    self._track_files(content, touched_files)
                elif event_type == "tool_result":
                    tool_results.append(content)
                elif event_type == "error":
                    errors.append(content)
                elif event_type == "command_waiting":
                    # Process is waiting - need user interaction
                    status.stop()  # Stop spinner to allow interaction
                    self._handle_command_waiting(event)
                    # Continue iteration - the process has been handled
        
        # 1. Display Tool Calls & Results first (Implementation Detail)
        if tool_calls:
            self.console.print() # Spacer
            for i, tc in enumerate(tool_calls):
                self._display_tool_call(tc)
                if i < len(tool_results):
                    self._display_tool_result(tool_results[i])
            self.console.print() # Spacer

        # 2. Display Errors
        for error in errors:
            self.console.print(Panel(f"[red]{error}[/]", title="[bold red]Error[/]", border_style="red"))
        
        # 3. Display Assistant Response
        clean_text = self._filter_special_tokens(response_text)
        if clean_text:
            self.console.print(Panel(
                Markdown(clean_text),
                title="[bold blue]SuperCoder[/]",
                border_style="blue",
                box=self._get_box_style()
            ))
        
        # 4. Display Status Footer (Tokens & Files)
        self._display_status_footer(touched_files)
        
        self.console.print()

    def _track_files(self, tool_call, touched_files):
        """Extract file paths from tool arguments to track active files."""
        name = tool_call.get("name")
        args = tool_call.get("arguments", {})
        
        # Handle string args (sometimes args is a JSON string)
        if isinstance(args, str):
            try:
                import json
                args = json.loads(args)
            except:
                return

        if not isinstance(args, dict):
            return

        # Look for common file arguments
        for key in ['file', 'path', 'filename', 'target_file', 'source_file']:
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
                except:
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

    def _get_box_style(self):
        """Return a box style for panels."""
        from rich import box
        return box.ROUNDED

    def _handle_command_waiting(self, event):
        """Handle a command that appears to be waiting for input."""
        content = event.get("content", "")
        process = event.get("process")
        tool_name = event.get("tool_name", "command-exec")
        
        # Display warning
        self.console.print(Panel(
            f"[yellow]{content}[/]",
            title="[bold yellow]⚠️ Process Stalled[/]",
            border_style="yellow",
            box=self._get_box_style()

        ))
        
        # Simple stdin-based menu (most reliable across terminals)
        self.console.print("\n[bold]Options:[/]")
        self.console.print("  [cyan]k[/] - Kill the process")
        self.console.print("  [cyan]w[/] - Wait longer (continue until timeout)")
        
        try:
            import sys
            self.console.print("\n[bold cyan]Action [k/w]>[/] ", end="")
            choice = sys.stdin.readline().strip().lower()
            
            if choice.startswith("k"):
                if process and hasattr(process, 'kill'):
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
            if process and hasattr(process, 'kill'):
                try:
                    process.kill()
                    process.wait(timeout=5)
                    self.console.print("\n[green]✓ Process killed (interrupted)[/]")
                except Exception:
                    pass
            return "killed"




    
    def _filter_special_tokens(self, text: str) -> str:
        """Remove special tokens from display text while preserving normal content."""
        import re
        # Remove tool_code blocks: ```tool_code ... ```
        text = re.sub(r'```tool_code\s*\n?.*?\n?```', '', text, flags=re.DOTALL)
        # Remove our native tool call format: <@TOOL>...</@TOOL>
        text = re.sub(r'<@TOOL>.*?</@TOOL>', '', text, flags=re.DOTALL)
        # Remove model-generated TOOL_RESULT blocks (model shouldn't generate these!)
        text = re.sub(r'<@TOOL_RESULT>.*?</@TOOL_RESULT>', '', text, flags=re.DOTALL)
        # Remove complete Qwen-style blocks: <|start|>...<|call|> 
        text = re.sub(r'<\|start\|>.*?<\|call\|>', '', text, flags=re.DOTALL)
        # Remove gpt-oss format: <|channel|>...to=...<|message|>{...}
        text = re.sub(r'<\|channel\|>.*?<\|message\|>\{[^}]*\}', '', text, flags=re.DOTALL)
        # Remove simple tool call format: to=tool.name {...} or to=tool:name {...} or to=TOOL name {...}
        text = re.sub(r'to=(?:tool[:\.]|TOOL\s+)[\w-]+\s+\{[^}]*\}', '', text, flags=re.IGNORECASE)
        # Remove any remaining special markers
        text = re.sub(r'<\|[^|]+\|>', '', text)
        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'  +', ' ', text)
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
            except:
                args_str = args
        else:
            import json
            args_str = json.dumps(args, indent=2)

        self.console.print(Panel(
            Syntax(args_str, "json", theme="monokai", word_wrap=True),
            title=f"[bold yellow]🔧 Tool Call: {name}[/]",
            border_style="yellow",
            box=self._get_box_style()
        ))

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
        
        self.console.print(Panel(
            f"[dim]{display_result}[/]",
            title=f"[bold green]✔ Result: {name}[/]",
            border_style="green",
            box=self._get_box_style()
        ))
    
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
            syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=False)
            self.console.print(Panel(
                syntax, 
                title="[bold cyan]Changes[/]", 
                border_style="cyan",
                box=self._get_box_style()
            ))


    # Commands
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
        
        self.console.print(f"[dim]Current context: {stats.used_tokens:,} tokens, {stats.message_count} messages[/]")
        
        # Show spinner while compacting
        with self.console.status("[bold blue]Compacting context...[/]", spinner="dots"):
            summary, stats_before, stats_after = self.agent.compact_context()
        
        # Display results
        tokens_saved = stats_before.used_tokens - stats_after.used_tokens
        reduction_pct = (tokens_saved / stats_before.used_tokens * 100) if stats_before.used_tokens > 0 else 0
        
        self.console.print(f"\n[green]✓ Context compacted![/]")
        self.console.print(f"  [dim]Before:[/] {stats_before.used_tokens:,} tokens ({stats_before.message_count} messages)")
        self.console.print(f"  [dim]After:[/]  {stats_after.used_tokens:,} tokens ({stats_after.message_count} messages)")
        self.console.print(f"  [dim]Saved:[/]  {tokens_saved:,} tokens ({reduction_pct:.1f}% reduction)")
        
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
                    self.console.print(f"[green]✓ Resumed session[/]")
                    stats = self.agent.context.get_stats()
                    self.console.print(f"[dim]Loaded {stats.message_count} messages, {stats.used_tokens:,} tokens[/]")
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

    def cmd_help(self, _):
        self.console.print("\n[bold]Available Commands:[/]")
        self.console.print("  /continue - Resume a previous session")
        self.console.print("  /sessions - List saved sessions")
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
        masked_key = f"{config.api_key[:4]}...{config.api_key[-4:]}" if config.api_key else "Not Set"
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
        
        # Reset prompt_toolkit buffer to prevent double input issue
        # This clears any stale state that might cause the next input to be processed twice
        try:
            if hasattr(self.session, 'app') and self.session.app is not None:
                self.session.app.current_buffer.reset()
        except Exception:
            pass  # Ignore if not in active input session
        
        self.console.print(f"[green]✓ Switched to {profile_name}[/]")
        self.console.print(f"[dim]Model: {profile.model}[/]")
        self.console.print(f"[dim]Endpoint: {profile.endpoint}[/]")
        self.console.print(f"[dim]Tool calling: {profile.tool_calling_type}[/]")
        return False
