#!/usr/bin/env python3
"""Test Supercoder Rules Loader."""

import tempfile
import os
from pathlib import Path
from supercoder.rules_loader import SupercoderRulesLoader


def test_ensure_rules_dir():
    """Test that rules directory is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SupercoderRulesLoader(tmpdir)
        rules_dir = loader.ensure_rules_dir()
        
        assert rules_dir.exists()
        assert rules_dir.is_dir()
        assert rules_dir == (Path(tmpdir) / ".supercoder" / "rules").resolve()
        print("✅ ensure_rules_dir() works")


def test_load_rules_empty():
    """Test loading rules from empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SupercoderRulesLoader(tmpdir)
        loader.ensure_rules_dir()
        
        rules = loader.load_rules()
        assert rules == ""
        print("✅ load_rules() returns empty string for empty dir")


def test_load_rules_with_files():
    """Test loading rules from directory with .md files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SupercoderRulesLoader(tmpdir)
        rules_dir = loader.ensure_rules_dir()
        
        # Create test rule files
        (rules_dir / "01_style.md").write_text("Use type hints")
        (rules_dir / "02_naming.md").write_text("Use snake_case")
        (rules_dir / "ignored.txt").write_text("Should be ignored")
        
        rules = loader.load_rules()
        
        assert "Use type hints" in rules
        assert "Use snake_case" in rules
        assert "Should be ignored" not in rules
        assert "## 01_style" in rules
        assert "## 02_naming" in rules
        print("✅ load_rules() correctly loads .md files")


def test_get_rules_for_prompt():
    """Test get_rules_for_prompt() formatting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SupercoderRulesLoader(tmpdir)
        rules_dir = loader.ensure_rules_dir()
        
        (rules_dir / "test.md").write_text("Test rule")
        
        prompt_section = loader.get_rules_for_prompt()
        
        assert "# Project Rules" in prompt_section
        assert "Test rule" in prompt_section
        print("✅ get_rules_for_prompt() formats correctly")


if __name__ == "__main__":
    test_ensure_rules_dir()
    test_load_rules_empty()
    test_load_rules_with_files()
    test_get_rules_for_prompt()
    print()
    print("🎉 All SupercoderRulesLoader tests passed!")
