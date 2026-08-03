"""Tests for tool-name alias resolution.

Models (especially local/small ones) frequently call tools by names they
learned from other agents' toolsets ('read', 'edit', 'grep', 'bash') instead
of the canonical kebab-case names ('file-read', 'code-edit', ...). The
shared TOOL_ALIASES map normalizes these before the ``name not in self.tools``
check, so the call reaches the right tool instead of producing an
'Unknown tool' error.

These tests guard the alias map so a regression (e.g. an alias removed by
mistake) fails loudly.
"""

from typing import ClassVar

from supercoder.tools import ALL_TOOLS, TOOL_ALIASES


def _canonical_names() -> set[str]:
    return {t.definition.name for t in ALL_TOOLS}


class TestToolAliasesResolveToCanonicalNames:
    """Every alias value must be a real canonical tool name."""

    def test_all_alias_targets_are_canonical(self):
        canonical = _canonical_names()
        for alias, target in TOOL_ALIASES.items():
            assert target in canonical, (
                f"alias '{alias}' -> '{target}' but '{target}' is not a canonical tool name. "
                f"Canonical names: {sorted(canonical)}"
            )

    def test_no_alias_points_to_itself(self):
        for alias, target in TOOL_ALIASES.items():
            assert alias != target, f"alias '{alias}' maps to itself (useless entry)"

    def test_no_alias_is_a_canonical_name(self):
        """An alias key should never shadow a canonical name."""
        canonical = _canonical_names()
        overlap = set(TOOL_ALIASES) & canonical
        assert not overlap, (
            f"alias keys collide with canonical tool names: {overlap}. "
            "This would create an infinite-resolution risk or shadow a real tool."
        )


class TestCommonHallucinatedNamesAreCovered:
    """The names models most commonly hallucinate must have an alias."""

    REQUIRED_FILE_READ: ClassVar[list[str]] = ["read", "cat", "view", "open", "file_read"]
    REQUIRED_CODE_EDIT: ClassVar[list[str]] = [
        "edit",
        "write",
        "create-file",
        "file-write",
        "file_edit",
    ]
    REQUIRED_CODE_SEARCH: ClassVar[list[str]] = ["grep", "search", "find"]
    REQUIRED_COMMAND_EXEC: ClassVar[list[str]] = ["bash", "shell", "run", "execute"]
    REQUIRED_PROJECT_STRUCTURE: ClassVar[list[str]] = ["ls", "tree", "list"]

    def test_file_read_synonyms_covered(self):
        missing = [n for n in self.REQUIRED_FILE_READ if n not in TOOL_ALIASES]
        assert not missing, f"file-read synonyms missing from aliases: {missing}"
        for n in self.REQUIRED_FILE_READ:
            assert TOOL_ALIASES[n] == "file-read"

    def test_code_edit_synonyms_covered(self):
        missing = [n for n in self.REQUIRED_CODE_EDIT if n not in TOOL_ALIASES]
        assert not missing, f"code-edit synonyms missing from aliases: {missing}"
        for n in self.REQUIRED_CODE_EDIT:
            assert TOOL_ALIASES[n] == "code-edit"

    def test_code_search_synonyms_covered(self):
        missing = [n for n in self.REQUIRED_CODE_SEARCH if n not in TOOL_ALIASES]
        assert not missing, f"code-search synonyms missing from aliases: {missing}"
        for n in self.REQUIRED_CODE_SEARCH:
            assert TOOL_ALIASES[n] == "code-search"

    def test_command_exec_synonyms_covered(self):
        missing = [n for n in self.REQUIRED_COMMAND_EXEC if n not in TOOL_ALIASES]
        assert not missing, f"command-exec synonyms missing from aliases: {missing}"
        for n in self.REQUIRED_COMMAND_EXEC:
            assert TOOL_ALIASES[n] == "command-exec"

    def test_project_structure_synonyms_covered(self):
        missing = [n for n in self.REQUIRED_PROJECT_STRUCTURE if n not in TOOL_ALIASES]
        assert not missing, f"project-structure synonyms missing from aliases: {missing}"
        for n in self.REQUIRED_PROJECT_STRUCTURE:
            assert TOOL_ALIASES[n] == "project-structure"


class TestUnknownNamePassesThrough:
    """A name with no alias entry must pass through unchanged."""

    def test_unknown_name_returned_as_is(self):
        assert TOOL_ALIASES.get("totally-made-up-tool") is None

    def test_canonical_name_has_no_alias(self):
        """Calling a tool by its real canonical name must not be remapped."""
        for canonical in _canonical_names():
            assert canonical not in TOOL_ALIASES, (
                f"canonical name '{canonical}' is also an alias key — would shadow itself"
            )
