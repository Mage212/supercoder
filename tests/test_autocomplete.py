"""Tests for prompt autocompletion."""

from prompt_toolkit.document import Document

from supercoder.autocomplete import AutoCompleter


def _completion_texts(completer: AutoCompleter, text: str) -> list[str]:
    return [completion.text for completion in completer.get_completions(Document(text), None)]


def test_context_reference_completion_matches_basename_and_path(tmp_path):
    (tmp_path / "main.py").write_text("print('ok')\n")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "maintenance.md").write_text("# Maintenance\n")

    completer = AutoCompleter(repo_root=tmp_path, commands=[])

    completions = _completion_texts(completer, "@ma")

    assert "@main.py" in completions
    assert "@docs/maintenance.md" in completions


def test_context_reference_completion_includes_directories_with_slash(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("print('ok')\n")

    completer = AutoCompleter(repo_root=tmp_path, commands=[])

    completions = _completion_texts(completer, "@sr")

    assert "@src/" in completions


def test_context_reference_completion_ignores_runtime_dirs(tmp_path):
    (tmp_path / "main.py").write_text("print('ok')\n")
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "main.py").write_text("ignored\n")

    completer = AutoCompleter(repo_root=tmp_path, commands=[])

    completions = _completion_texts(completer, "@main")

    assert "@main.py" in completions
    assert all(".venv" not in completion for completion in completions)


def test_context_reference_completion_replaces_only_at_token(tmp_path):
    (tmp_path / "main.py").write_text("print('ok')\n")
    completer = AutoCompleter(repo_root=tmp_path, commands=[])

    completion = next(iter(completer.get_completions(Document("Check (@ma"), None)))

    assert completion.text == "@main.py"
    assert completion.start_position == -3
