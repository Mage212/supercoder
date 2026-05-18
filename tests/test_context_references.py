"""Tests for host-side @path context attachment."""

from supercoder.context.references import (
    expand_context_references,
    extract_context_references,
    summarize_context_attachment,
)


def test_extract_context_references_skips_escaped_and_email():
    message = r"Use @src/main.py, ignore \@literal and dev@example.com"

    refs = extract_context_references(message)

    assert refs == ["src/main.py"]


def test_expand_file_reference_with_line_numbers(tmp_path):
    target = tmp_path / "main.py"
    target.write_text("def main():\n    return 1\n")

    attachment = expand_context_references("Review @main.py", tmp_path)

    assert attachment is not None
    assert '<attached_file path="main.py"' in attachment.content
    assert "   1: def main():" in attachment.content
    assert "   2:     return 1" in attachment.content
    assert attachment.files == 1


def test_expand_directory_reference_lists_files_without_contents(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("SECRET_CONTENT\n")
    (src / "notes.txt").write_text("visible path only\n")

    attachment = expand_context_references("Review @src", tmp_path)

    assert attachment is not None
    assert '<attached_directory path="src"' in attachment.content
    assert "src/app.py" in attachment.content
    assert "src/notes.txt" in attachment.content
    assert "SECRET_CONTENT" not in attachment.content
    assert attachment.directories == 1


def test_expand_reference_skips_outside_root(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}_outside_reference.txt"
    outside.write_text("nope\n")

    attachment = expand_context_references("Read @../outside_reference.txt", tmp_path)

    assert attachment is not None
    assert "<skipped_reference" in attachment.content
    assert 'reason="outside repository"' in attachment.content
    outside.unlink()


def test_expand_reference_skips_binary_files(tmp_path):
    target = tmp_path / "image.bin"
    target.write_bytes(b"abc\x00def")

    attachment = expand_context_references("Read @image.bin", tmp_path)

    assert attachment is not None
    assert "<skipped_reference" in attachment.content
    assert 'reason="binary file"' in attachment.content


def test_expand_missing_reference_includes_suggestions(tmp_path):
    (tmp_path / "readable.txt").write_text("ok\n")

    attachment = expand_context_references("Read @readble.txt", tmp_path)

    assert attachment is not None
    assert "Did you mean" in attachment.content
    assert "readable.txt" in attachment.content


def test_expand_file_reference_respects_total_budget(tmp_path):
    target = tmp_path / "large.txt"
    target.write_text(("a" * 200) + "\n")

    attachment = expand_context_references(
        "Read @large.txt",
        tmp_path,
        max_file_bytes=200,
        max_total_tokens=10,
    )

    assert attachment is not None
    assert 'truncated="true"' in attachment.content
    assert attachment.truncated is True


def test_summarize_context_attachment_formats_counts(tmp_path):
    target = tmp_path / "main.py"
    target.write_text("print('ok')\n")
    attachment = expand_context_references("Read @main.py", tmp_path)

    assert attachment is not None
    summary = summarize_context_attachment(attachment.to_log_dict())

    assert "1 file(s)" in summary
    assert "tokens" in summary
