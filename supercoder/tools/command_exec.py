"""Command execution tool with safety checks."""

import subprocess
import shlex
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


class CommandExecutionTool(BaseTool):
    """Execute shell commands with safety checks."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="command-exec",
            description='Execute shell command. Args: {"command": "ls -la", "timeout": 60}'
        )
    
    def execute(self, arguments: str) -> str:
        args = self.parse_args(arguments)
        command = args.get("command", "")
        timeout = args.get("timeout", 60)
        
        if not command:
            return "Error: command is required"
        
        # Safety check for dangerous commands
        command_lower = command.lower()
        for pattern in DANGEROUS_PATTERNS:
            if pattern in command_lower:
                return f"⛔ Blocked dangerous command: {command}\nThis pattern is not allowed: {pattern}"
        
        # Warning for potentially risky commands
        warnings = []
        for pattern in WARN_PATTERNS:
            if pattern in command_lower:
                warnings.append(f"⚠️ Caution: command contains '{pattern.strip()}'")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=min(timeout, 120),  # Max 2 minutes
                cwd="."
            )
            
            output_parts = []
            
            if warnings:
                output_parts.append("\n".join(warnings))
            
            output_parts.append(f"$ {command}")
            output_parts.append(f"Exit code: {result.returncode}")
            
            if result.stdout:
                stdout = result.stdout[:3000]  # Limit output
                if len(result.stdout) > 3000:
                    stdout += f"\n... (truncated, {len(result.stdout)} total chars)"
                output_parts.append(f"\n{stdout}")
            
            if result.stderr:
                stderr = result.stderr[:1000]
                output_parts.append(f"\nStderr:\n{stderr}")
            
            if not result.stdout and not result.stderr:
                output_parts.append("(no output)")
            
            return "\n".join(output_parts)
            
        except subprocess.TimeoutExpired:
            return f"⏱️ Command timed out after {timeout}s: {command}"
        except Exception as e:
            return f"Error executing command: {e}"
