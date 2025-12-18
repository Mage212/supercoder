"""Main coding agent with context management."""

import re
import json
from pathlib import Path
from rich.console import Console

from ..llm.base import BaseLLM, Message
from ..tools.base import BaseTool
from ..context.window_manager import ContextWindowManager, ContextConfig
from ..repomap import RepoMap
from ..rules_loader import SupercoderRulesLoader
from ..logging import get_logger
from .prompts import build_system_prompt, CONTEXT_SUMMARY_PROMPT
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
        repo_root: str = "."
    ):
        self.llm = llm
        self.tools = {t.definition.name: t for t in (tools or [])}
        self.repo_root = Path(repo_root).resolve()
        
        # RepoMap setup
        self.repo_map = RepoMap(self.repo_root) if use_repo_map else None

        
        # Supercoder Rules setup
        self.rules_loader = SupercoderRulesLoader(repo_root)
        self.rules_loader.ensure_rules_dir()  # Create .supercoder/rules/ if missing
        project_rules = self.rules_loader.get_rules_for_prompt()
        
        # Build system prompt template with tools and project rules
        self.base_system_prompt = build_system_prompt(tools or [], rules=project_rules)
        
        # Setup context management
        config = context_config or ContextConfig()
        self.context = ContextWindowManager(config)
        self._update_system_prompt()
        
        # Multi-format tool call parser
        self.tool_parser = ToolCallParser(debug=False)
        
        self.debug = False

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
        
        self.context.set_system_prompt(prompt)
        # Log the updated system prompt
        get_logger().log_system_prompt(prompt)

    def chat_stream(self, user_message: str):
        """Process user message and yield response chunks.
        
        Yields:
            dict: Event dictionary with 'type' and 'content' keys.
                  Types: 'token', 'tool_call', 'tool_result', 'error', 'done'
        """
        # Update RepoMap occasionally
        if self.repo_map:
            self._update_system_prompt()
            
        # Add user message to context
        if user_message:
            self.context.add_message(Message("user", user_message))
            # Log user input
            get_logger().log_user_input(user_message)
        
        # Get messages for API
        messages = self.context.get_messages_for_api()
        
        # Log the full request payload
        get_logger().log_messages(messages)
        
        # Stream response
        response_text = ""
        
        try:
            for chunk in self.llm.chat_stream(messages):
                if not chunk.is_done:
                    # Yield token for real-time display
                    yield {"type": "token", "content": chunk.content}
                    response_text += chunk.content
            
            # Signal end of text generation
            yield {"type": "done", "content": ""}
            
        except Exception as e:
            yield {"type": "error", "content": str(e)}
            return
        
        # Add assistant response to context
        if response_text:
            self.context.add_message(Message("assistant", response_text))
            # Log model response
            get_logger().log_model_response(response_text, self.llm.model)
        
        # Check for tool calls (may be multiple)
        tool_calls = self._extract_all_tool_calls(response_text)
        if tool_calls:
            all_results = []
            
            for tool_call_data in tool_calls:
                yield {"type": "tool_call", "content": tool_call_data}
                
                # Execute tool
                name = tool_call_data.get("name", "")
                args = tool_call_data.get("arguments", "")
                
                if name not in self.tools:
                    yield {"type": "error", "content": f"Unknown tool: {name}"}
                    continue
                
                try:
                    result = self.tools[name].execute(args)
                    yield {"type": "tool_result", "content": {"name": name, "result": result}}
                    # Log tool call and result
                    get_logger().log_tool_call(name, args)
                    get_logger().log_tool_result(name, result)
                    all_results.append(f"[{name}]: {result}")
                except Exception as e:
                    result = f"Error executing tool: {e}"
                    yield {"type": "error", "content": result}
                    all_results.append(f"[{name}]: {result}")
            
            # Add combined tool results to context
            combined_results = "\n\n".join(all_results)
            self.context.add_message(
                Message("user", f"<@TOOL_RESULT>{combined_results}</@TOOL_RESULT>")
            )
            
            # Let agent continue with tool results (recursive call handling)
            yield from self.chat_stream("")

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
    
    def compact_context(self) -> tuple[str, "ContextStats", "ContextStats"]:
        """Compact the current context by summarizing it.
        
        This method:
        1. Gets all messages from history
        2. Asks the LLM to create a summary (emphasizing recent messages)
        3. Clears the history
        4. Injects the summary as the new starting context
        
        Returns:
            tuple: (summary_text, stats_before, stats_after)
        """
        from ..context.window_manager import ContextStats
        
        # Get stats before compaction
        stats_before = self.context.get_stats()
        
        # Get conversation history
        messages = self.context.get_messages()
        if not messages:
            return ("No context to compact.", stats_before, stats_before)
        
        # Format conversation for summarization
        conversation_text = "\n\n".join([
            f"[{msg.role.upper()}]: {msg.content}" 
            for msg in messages
        ])
        
        # Build summarization prompt
        summary_prompt = CONTEXT_SUMMARY_PROMPT.format(
            conversation_history=conversation_text
        )
        
        # Call LLM synchronously to get summary
        from ..llm.base import Message
        summary_messages = [Message("user", summary_prompt)]
        
        try:
            summary = self.llm.chat(summary_messages)
        except Exception as e:
            return (f"Error generating summary: {e}", stats_before, stats_before)
        
        # Clear history and set summary as initial context
        self.context.set_initial_summary(summary)
        
        # Get stats after compaction
        stats_after = self.context.get_stats()
        
        return (summary, stats_before, stats_after)
