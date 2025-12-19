"""Command execution tool with safety checks and stall detection."""

import subprocess
import shlex
import threading
import time
import os
import signal
import platform
from queue import Queue, Empty
from typing import Iterator
from .base import BaseTool, ToolDefinition


# Dangerous command patterns
DANGEROUS_PATTERNS = [
    "rm -rf /",
    "rm -rf ~",
    "rm -rf *",
    "> /dev/",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",  # fork bomb
    "chmod -R 777 /",
    "curl | sh",
    "wget | sh",
]

# Commands that need confirmation (but are allowed)
WARN_PATTERNS = [
    "rm ",
    "sudo ",
    "chmod ",
    "chown ",
    "mv /",
    "cp /",
]

# Known interactive commands that typically wait for user input
INTERACTIVE_PATTERNS = [
    "python",
    "python3",
    "node",
    "ruby",
    "irb",
    "bash",
    "sh",
    "zsh",
    "vim",
    "nano",
    "less",
    "more",
    "top",
    "htop",
]


class CommandExecutionTool(BaseTool):
    """Execute shell commands with safety checks and stall detection."""
    
    # Configuration
    STALL_THRESHOLD = 5       # Seconds without output before considered stalled (reduced from 10)
    POLL_INTERVAL = 0.5       # Seconds between output checks (reduced from 1)
    MAX_OUTPUT_LENGTH = 30000 # Maximum output length before truncation
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="command-exec",
            description='Execute shell command. Args: {"command": "ls -la", "timeout": 60}'
        )
    
    def execute(self, arguments: str) -> str:
        """Synchronous execute - collects all streaming results."""
        result_parts = []
        stall_warning = None
        
        for event in self.execute_streaming(arguments):
            if event["type"] == "output":
                result_parts.append(event["content"])
            elif event["type"] == "stalled":
                stall_warning = event["content"]
            elif event["type"] == "done":
                result_parts.append(event["content"])
            elif event["type"] == "error":
                return event["content"]
        
        result = "\n".join(result_parts)
        if stall_warning:
            result = f"{stall_warning}\n\n{result}"
        return result
    
    def execute_streaming(self, arguments: str) -> Iterator[dict]:
        """Execute command with streaming output and stall detection.
        
        Yields events:
        - {"type": "output", "content": "..."}     # Incremental stdout/stderr
        - {"type": "stalled", "content": "..."}   # Detected possible hang
        - {"type": "done", "content": "..."}      # Completion summary
        - {"type": "error", "content": "..."}     # Error occurred
        - {"type": "waiting_input", "process": proc}  # Needs user decision
        """
        args = self.parse_args(arguments)
        command = args.get("command", "")
        timeout = min(args.get("timeout", 60), 120)  # Max 2 minutes
        
        if not command:
            yield {"type": "error", "content": "Error: command is required"}
            return
        
        # Safety check for dangerous commands
        command_lower = command.lower()
        for pattern in DANGEROUS_PATTERNS:
            if pattern in command_lower:
                yield {"type": "error", "content": f"⛔ Blocked dangerous command: {command}\nThis pattern is not allowed: {pattern}"}
                return
        
        # Warning for potentially risky commands
        warnings = []
        for pattern in WARN_PATTERNS:
            if pattern in command_lower:
                warnings.append(f"⚠️ Caution: command contains '{pattern.strip()}'")
        
        if warnings:
            yield {"type": "output", "content": "\n".join(warnings)}
        
        yield {"type": "output", "content": f"$ {command}"}
        
        # Check if likely interactive
        is_interactive = any(p in command_lower for p in INTERACTIVE_PATTERNS)
        
        try:
            # Create process with stdin isolation to prevent hangs on interactive commands
            # Use start_new_session=True to create process group for proper tree killing
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                stdin=subprocess.DEVNULL,  # CRITICAL: Isolate stdin to prevent hangs
                text=True,
                cwd=".",
                start_new_session=True,  # Create new process group for tree killing
            )
            
            # Use single queue for unified stdout+stderr
            output_queue = Queue()
            
            def read_output(pipe, queue):
                """Read from pipe and put lines into queue."""
                try:
                    for line in iter(pipe.readline, ''):
                        queue.put(line)
                    pipe.close()
                except Exception:
                    pass
            
            output_thread = threading.Thread(target=read_output, args=(proc.stdout, output_queue))
            output_thread.daemon = True
            output_thread.start()

            
            # Collect output with stall detection
            output_lines = []  # Unified output (stdout+stderr merged)
            last_output_time = time.time()
            start_time = time.time()
            stall_warned = False
            
            while True:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    self.kill_process_tree(proc)
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass  # Already killed
                    
                    # Drain remaining output
                    while not output_queue.empty():
                        output_lines.append(output_queue.get_nowait())
                    
                    yield {"type": "done", "content": self._format_timeout_result(
                        command, timeout, output_lines
                    )}
                    return
                
                # Check if process finished
                if proc.poll() is not None:
                    # Process done - drain queue
                    output_thread.join(timeout=1)
                    
                    while not output_queue.empty():
                        output_lines.append(output_queue.get_nowait())
                    
                    yield {"type": "done", "content": self._format_result(
                        proc.returncode, output_lines
                    )}
                    return
                
                # Read available output
                got_output = False
                try:
                    while True:
                        line = output_queue.get_nowait()
                        output_lines.append(line)
                        got_output = True
                except Empty:
                    pass
                
                if got_output:
                    last_output_time = time.time()
                
                # Check for stall
                stall_time = time.time() - last_output_time
                if stall_time >= self.STALL_THRESHOLD and not stall_warned:
                    stall_warned = True
                    
                    current_output = "".join(output_lines[-5:])  # Last 5 lines
                    
                    if is_interactive:
                        yield {
                            "type": "waiting_input",
                            "content": f"⚠️ Process has not produced output for {int(stall_time)}s.\n"
                                       f"Command may be waiting for input.\n"
                                       f"Last output:\n{current_output}",
                            "process": proc,
                            "output": output_lines
                        }
                    else:
                        yield {
                            "type": "stalled",
                            "content": f"⚠️ Process stalled for {int(stall_time)}s (may be waiting for input)"
                        }
                
                time.sleep(self.POLL_INTERVAL)
                
        except Exception as e:
            yield {"type": "error", "content": f"Error executing command: {e}"}
    
    def _format_result(self, returncode: int, output_lines: list) -> str:
        """Format successful command result."""
        parts = [f"Exit code: {returncode}"]
        
        output = "".join(output_lines)
        if output:
            if len(output) > self.MAX_OUTPUT_LENGTH:
                output = output[:self.MAX_OUTPUT_LENGTH] + f"\n... (truncated, {len(output)} total chars)"
            parts.append(f"\n{output}")
        else:
            parts.append("(no output)")
        
        return "\n".join(parts)
    
    def _format_timeout_result(self, command: str, timeout: int, output_lines: list) -> str:
        """Format timeout result with partial output."""
        parts = [f"⏱️ Command timed out after {timeout}s: {command}"]
        
        output = "".join(output_lines)
        if output:
            truncated = output[:2000] if len(output) > 2000 else output
            parts.append(f"\n📋 Partial output before timeout:\n{truncated}")
        
        parts.append("\n💡 Tip: stdin is isolated - if command requires input, use: echo 'input' | command")
        
        return "\n".join(parts)
    
    def kill_process_tree(self, proc: subprocess.Popen) -> str:
        """Kill process and all its children (entire process tree).
        
        Uses process group killing on Unix and taskkill /T on Windows.
        """
        try:
            if platform.system() == "Windows":
                # Windows: use taskkill with /T for tree kill
                result = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return "Process tree killed"
                # Fallback to simple kill
                proc.kill()
                return "Process killed (taskkill failed)"
            else:
                # Unix: kill entire process group
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    time.sleep(0.2)  # Give processes time to terminate gracefully
                    try:
                        os.killpg(pgid, signal.SIGKILL)  # Force kill if still alive
                    except ProcessLookupError:
                        pass  # Already dead
                    return "Process tree killed"
                except ProcessLookupError:
                    return "Process already terminated"
                except OSError as e:
                    # Fallback: process might not be group leader
                    proc.kill()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                    return f"Process killed (fallback): {e}"
        except Exception as e:
            return f"Error killing process: {e}"
    
    # Backwards compatibility alias
    def kill_process(self, proc: subprocess.Popen) -> str:
        """Kill a running process (alias for kill_process_tree)."""
        return self.kill_process_tree(proc)
