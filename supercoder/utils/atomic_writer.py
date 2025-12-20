"""Atomic file writing utilities."""

import os
import tempfile
from pathlib import Path


class AtomicFileWriter:
    """Atomic file writer using temp file + rename pattern.
    
    This ensures that file writes are atomic - either the entire
    content is written successfully, or the original file remains
    unchanged. This prevents file corruption on interrupts or errors.
    """
    
    @staticmethod
    def write(path: Path, content: str, encoding: str = "utf-8") -> None:
        """Atomically write content to a file.
        
        Args:
            path: Target file path
            content: Content to write
            encoding: File encoding (default: utf-8)
            
        Raises:
            OSError: If the write operation fails
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temp file in the same directory as target
        # This is important for atomic rename (same filesystem)
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp"
        )
        
        try:
            with os.fdopen(fd, 'w', encoding=encoding) as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk
            
            # Atomic rename (POSIX guarantee)
            os.replace(tmp_path, path)
            
        except Exception:
            # Clean up temp file on error
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise
    
    @staticmethod
    def write_bytes(path: Path, content: bytes) -> None:
        """Atomically write binary content to a file.
        
        Args:
            path: Target file path
            content: Binary content to write
            
        Raises:
            OSError: If the write operation fails
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp"
        )
        
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            
            os.replace(tmp_path, path)
            
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise
