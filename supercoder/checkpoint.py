"""Checkpoint system for safe file editing with rollback support."""

import shutil
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional


MAX_CHECKPOINTS = 10  # Maximum number of checkpoints to retain


@dataclass
class Checkpoint:
    """Represents a single checkpoint with file backups."""
    id: str
    timestamp: str
    description: str
    files: dict = field(default_factory=dict)  # original_path -> backup_path
    created_files: list = field(default_factory=list)  # list of paths created in this checkpoint


class CheckpointManager:
    """Manages checkpoints for safe file editing with rollback capability.
    
    The checkpoint system works as follows:
    1. Before editing, create() is called to start a new checkpoint
    2. For each file being modified, backup_file() saves the original
    3. If editing succeeds, commit() saves the checkpoint metadata
    4. If editing fails or is interrupted, rollback() restores all files
    5. User can manually undo with undo_by_id()
    
    Up to MAX_CHECKPOINTS are retained, older ones are automatically deleted.
    """
    
    def __init__(self, repo_root: Path):
        """Initialize the checkpoint manager.
        
        Args:
            repo_root: Root directory of the project
        """
        self.repo_root = Path(repo_root)
        self.checkpoint_dir = self.repo_root / ".supercoder" / "checkpoints"
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"ERROR: Failed to create checkpoint dir {self.checkpoint_dir}: {e}")
            
        self._cleanup_orphaned_checkpoints()
        self.current: Optional[Checkpoint] = None
    
    def create(self, description: str = "") -> Checkpoint:
        """Create a new checkpoint before starting edits.
        
        Args:
            description: Human-readable description of the changes
            
        Returns:
            The newly created Checkpoint object
        """
        checkpoint_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(exist_ok=True)
        
        self.current = Checkpoint(
            id=checkpoint_id,
            timestamp=datetime.now().isoformat(),
            description=description[:100] if description else "Unnamed checkpoint",
            files={}  # original -> backup
        )
        # Add support for tracking new files (not in dataclass yet, but as runtime attr)
        self.current.created_files = [] 
        return self.current
    
    def backup_file(self, file_path: Path) -> bool:
        """Save a backup of a file before modifying it.
        
        Args:
            file_path: Path to the file to backup
            
        Returns:
            True if backup was created, False if file doesn't exist
        """
        if not self.current:
            return False
        
        file_path = Path(file_path)
        if not file_path.exists():
            return False  # New file, nothing to backup
        
        # Already backed up in this checkpoint
        if str(file_path) in self.current.files:
            return True
        
        # Create backup with encoded path as filename
        backup_name = str(file_path.absolute()).replace("/", "__").replace("\\", "__")
        backup_path = self.checkpoint_dir / self.current.id / backup_name
        
        try:
            shutil.copy2(file_path, backup_path)
            self.current.files[str(file_path.absolute())] = str(backup_path)
            return True
        except Exception:
            return False
            
    def track_created_file(self, file_path: Path) -> None:
        """Track a newly created file to delete it on rollback.
        
        Args:
            file_path: Path to the new file
        """
        if self.current:
            path_str = str(Path(file_path).absolute())
            if path_str not in self.current.created_files:
                self.current.created_files.append(path_str)
    
    def commit(self) -> bool:
        """Commit the current checkpoint after successful edits.
        
        Saves metadata and rotates old checkpoints.
        
        Returns:
            True if committed successfully
        """
        if not self.current:
            return False
        
        # Don't save empty checkpoints (unless we created or modified files)
        if not self.current.files and not self.current.created_files:
            self._delete_checkpoint(self.current.id)
            self.current = None
            return False
        
        self._save_metadata(self.current)
        self._rotate_old_checkpoints()
        self.current = None
        return True
    
    def rollback(self) -> list[str]:
        """Rollback current uncommitted checkpoint.
        
        Restores all backed up files to their original state.
        
        Returns:
            List of restored file paths
        """
        if not self.current:
            return []
        
        restored = self._restore_files(self.current)
        self._delete_checkpoint(self.current.id)
        self.current = None
        return restored
    
    def list_checkpoints(self) -> list[Checkpoint]:
        """List all saved checkpoints, newest first.
        
        Returns:
            List of Checkpoint objects sorted by timestamp (descending)
        """
        checkpoints = []
        if self.checkpoint_dir.exists():
            for meta_file in self.checkpoint_dir.glob("*/metadata.json"):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        checkpoints.append(Checkpoint(**data))
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue  # Skip invalid checkpoints
        
        return sorted(checkpoints, key=lambda c: c.timestamp, reverse=True)
    
    def undo_by_id(self, checkpoint_id: str) -> list[str]:
        """Undo to a specific checkpoint.
        
        Restores files from the specified checkpoint and removes
        all checkpoints from that point forward.
        
        Args:
            checkpoint_id: ID of the checkpoint to restore
            
        Returns:
            List of restored file paths
        """
        meta_path = self.checkpoint_dir / checkpoint_id / "metadata.json"
        if not meta_path.exists():
            return []
        
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                checkpoint = Checkpoint(**json.load(f))
        except (json.JSONDecodeError, TypeError, KeyError):
            return []
        
        restored = self._restore_files(checkpoint)
        
        # Delete this checkpoint and all newer ones
        for cp in self.list_checkpoints():
            if cp.timestamp >= checkpoint.timestamp:
                self._delete_checkpoint(cp.id)
        
        return restored
    
    def undo_last(self) -> list[str]:
        """Undo the most recent checkpoint.
        
        Returns:
            List of restored file paths
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return []
        return self.undo_by_id(checkpoints[0].id)
    
    def _restore_files(self, checkpoint: Checkpoint) -> list[str]:
        """Restore files from a checkpoint.
        
        Args:
            checkpoint: Checkpoint to restore from
            
        Returns:
            List of successfully restored file paths
        """
        restored = []
        
        # Restore modified files
        for original, backup in checkpoint.files.items():
            backup_path = Path(backup)
            if backup_path.exists():
                try:
                    # Ensure parent directory exists (in case it was deleted)
                    Path(original).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_path, original)
                    restored.append(original)
                except Exception:
                    continue
                    
        # Delete created files
        for created_path in checkpoint.created_files:
            try:
                p = Path(created_path)
                if p.exists():
                    p.unlink()
                    restored.append(f"{created_path} (deleted)")
            except Exception:
                continue
                
        return restored
    
    def _save_metadata(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint metadata to disk."""
        meta_path = self.checkpoint_dir / checkpoint.id / "metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(checkpoint), f, indent=2, ensure_ascii=False)
    
    def _delete_checkpoint(self, checkpoint_id: str) -> None:
        """Delete a checkpoint directory."""
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path, ignore_errors=True)
    
    def _rotate_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only MAX_CHECKPOINTS."""
        checkpoints = self.list_checkpoints()
        for old in checkpoints[MAX_CHECKPOINTS:]:
            self._delete_checkpoint(old.id)
            
    def _cleanup_orphaned_checkpoints(self) -> None:
        """Remove checkpoint directories that have no metadata (failed/interrupted sessions)."""
        if not self.checkpoint_dir.exists():
            return
            
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir():
                # Check if metadata exists
                if not (path / "metadata.json").exists():
                    try:
                        shutil.rmtree(path, ignore_errors=True)
                    except Exception:
                        pass
