"""Project structure tool."""

from pathlib import Path

from ..logging import get_logger
from ..permissions import PermissionPolicy
from .base import BaseTool, ToolDefinition
from .tool_utils import IGNORE_DIRS, IGNORE_NAMES, IGNORE_SUFFIXES, format_size, resolve_within_root


class ProjectStructureTool(BaseTool):
    """Show project directory structure."""

    def __init__(
        self,
        allowed_root: Path | None = None,
        permission_policy: PermissionPolicy | None = None,
    ):
        self.allowed_root = allowed_root
        self.permission_policy = permission_policy

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="project-structure",
            description="Show the project directory tree structure.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Root path to show structure from",
                        "default": ".",
                    },
                    "maxDepth": {
                        "type": "integer",
                        "description": "Maximum directory depth",
                        "default": 3,
                    },
                    "maxFiles": {
                        "type": "integer",
                        "description": "Maximum number of files to show",
                        "default": 50,
                    },
                },
            },
        )

    def execute(self, arguments: str) -> str:
        args = self.parse_args(arguments)
        if args.get("_parse_error"):
            return f"Error: Invalid JSON arguments: {args.get('raw', '')}"
        max_depth = args.get("maxDepth", 3)
        max_files = args.get("maxFiles", 50)
        root_path = args.get("path", ".")

        root, error = resolve_within_root(root_path, self.allowed_root)
        if error:
            return error
        if root is None:
            return "Error: Invalid path"

        if self.permission_policy:
            decision = self.permission_policy.check_path(root, "list")
            if decision.denied:
                self._log_path_denial(root, decision)
                return self.permission_policy.format_denial(root_path, decision)

        if not root.exists():
            return f"Error: Path '{root_path}' not found"

        output = ["📁 Project Structure:"]
        counter = {"files": 0, "dirs": 0}

        self._build_tree(root, output, 0, max_depth, max_files, counter)

        output.append(f"\n📊 Total: {counter['dirs']} directories, {counter['files']} files shown")

        return "\n".join(output)

    def _build_tree(
        self, path: Path, output: list, depth: int, max_depth: int, max_files: int, counter: dict
    ) -> None:
        """Recursively build directory tree."""
        if depth >= max_depth or counter["files"] >= max_files:
            return

        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return

        for item in items:
            # Skip ignored items
            if item.name in IGNORE_DIRS:
                continue
            if item.suffix in IGNORE_SUFFIXES:
                continue
            if item.name in IGNORE_NAMES:
                continue
            if item.name.startswith(".") and item.name != ".env.example":
                continue
            if not self._path_allowed(item, "list"):
                continue

            prefix = "  " * depth

            if item.is_dir():
                output.append(f"{prefix}📁 {item.name}/")
                counter["dirs"] += 1
                self._build_tree(item, output, depth + 1, max_depth, max_files, counter)
            else:
                if counter["files"] < max_files:
                    size = format_size(item.stat().st_size)
                    output.append(f"{prefix}📄 {item.name} ({size})")
                    counter["files"] += 1

    def _path_allowed(self, path: Path, operation: str) -> bool:
        if not self.permission_policy:
            return True
        decision = self.permission_policy.check_path(path, operation)
        if decision.denied:
            self._log_path_denial(path, decision)
            return False
        return True

    def _log_path_denial(self, path: Path, decision) -> None:
        if not self.permission_policy:
            return
        get_logger().log_permission_decision(
            tool_name=self.definition.name,
            subject=self.permission_policy.relative_path(path),
            action=decision.action.value,
            reason=decision.reason,
            source=decision.source,
            matched_rule=decision.matched_rule,
        )
