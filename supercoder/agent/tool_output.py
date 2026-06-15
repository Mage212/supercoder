"""Tool output masking and offload for context hygiene."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..logging import get_logger
from ..utils.atomic_writer import AtomicFileWriter
from ..utils.secret_scrubber import scrub_secrets


@dataclass(frozen=True)
class MaskedToolOutput:
    """Result of preparing tool output for the model context."""

    model_text: str
    full_text: str
    display_text: str
    offload_path: Path | None = None
    masked: bool = False
    original_size: int = 0
    omitted_chars: int = 0


class ToolOutputMasker:
    """Keep large tool outputs out of the model history."""

    MAX_INLINE_CHARS = 8000
    HEAD_CHARS = 3000
    TAIL_CHARS = 2000

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self.output_dir = self.repo_root / ".supercoder" / "tool-outputs"

    def mask(self, tool_name: str, tool_call_id: str | None, output: str) -> MaskedToolOutput:
        """Return compact model text and offload full output when needed."""
        if len(output) <= self.MAX_INLINE_CHARS:
            return MaskedToolOutput(
                model_text=output,
                full_text=output,
                display_text=output,
                original_size=len(output),
            )

        offload_path: Path | None = None
        offload_note = ""
        try:
            offload_path = self._write_full_output(tool_name, tool_call_id, output)
            rel_path = offload_path.relative_to(self.repo_root).as_posix()
            offload_note = f"Full output saved to: {rel_path}"
        except Exception as e:
            get_logger().log_error(e)
            offload_note = f"Full output could not be saved: {e}"

        omitted = len(output) - self.HEAD_CHARS - self.TAIL_CHARS
        model_text = (
            "[Tool output compacted]\n"
            f"Tool: {tool_name}\n"
            f"Original size: {len(output)} chars\n"
            f"{offload_note}\n"
            f"Omitted middle: {max(omitted, 0)} chars\n"
            "\n"
            "--- head ---\n"
            f"{output[: self.HEAD_CHARS].rstrip()}\n"
            "\n"
            "--- tail ---\n"
            f"{output[-self.TAIL_CHARS :].lstrip()}"
        )
        display_text = (
            f"Showing first {self.HEAD_CHARS} chars and last {self.TAIL_CHARS} chars.\n"
            f"Hidden middle: {max(omitted, 0)} chars\n"
            "\n"
            "Preview head:\n"
            f"{output[: self.HEAD_CHARS].rstrip()}\n"
            "\n"
            "Preview tail:\n"
            f"{output[-self.TAIL_CHARS :].lstrip()}"
        )
        return MaskedToolOutput(
            model_text=model_text,
            full_text=output,
            display_text=display_text,
            offload_path=offload_path,
            masked=True,
            original_size=len(output),
            omitted_chars=max(omitted, 0),
        )

    def _write_full_output(self, tool_name: str, tool_call_id: str | None, output: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_tool = re.sub(r"[^A-Za-z0-9_.-]+", "-", tool_name or "tool").strip("-") or "tool"
        safe_id_source = tool_call_id or uuid.uuid4().hex[:8]
        safe_id = (
            re.sub(r"[^A-Za-z0-9_.-]+", "-", safe_id_source).strip("-") or uuid.uuid4().hex[:8]
        )
        path = self.output_dir / f"{timestamp}-{safe_tool}-{safe_id}.txt"
        AtomicFileWriter.write(path, scrub_secrets(output))
        return path
