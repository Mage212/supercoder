"""Supercoder Rules Loader.

Loads project-specific coding rules from .supercoder/rules/*.md files
and injects them into the system prompt.
"""

from pathlib import Path


class SupercoderRulesLoader:
    """Loads coding rules from .supercoder/rules/*.md files."""
    
    RULES_DIR = ".supercoder/rules"
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.rules_dir = self.project_root / self.RULES_DIR
    
    def ensure_rules_dir(self) -> Path:
        """Create rules directory if it doesn't exist.
        
        Returns:
            Path to the rules directory.
        """
        self.rules_dir.mkdir(parents=True, exist_ok=True)
        return self.rules_dir
    
    def load_rules(self) -> str:
        """Load all .md files from rules directory.
        
        Returns:
            Combined content of all rule files, or empty string if none found.
        """
        if not self.rules_dir.exists():
            return ""
        
        rules_content = []
        
        # Get all .md files sorted by name for consistent order
        md_files = sorted(self.rules_dir.glob("*.md"))
        
        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8").strip()
                if content:
                    # Add file name as header for clarity
                    rules_content.append(f"## {md_file.stem}\n{content}")
            except Exception:
                # Skip files that can't be read
                continue
        
        if not rules_content:
            return ""
        
        return "\n\n".join(rules_content)
    
    def get_rules_for_prompt(self) -> str:
        """Get rules formatted for system prompt.
        
        Returns:
            Formatted rules section or empty string.
        """
        rules = self.load_rules()
        if not rules:
            return ""
        
        return f"""
# Project Rules
The following are project-specific coding rules you MUST follow:

{rules}
"""
