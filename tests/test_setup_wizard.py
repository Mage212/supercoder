"""Tests for setup wizard UX regressions."""

from supercoder import setup_wizard


def test_pick_provider_invalid_choice_prints_provider_count(monkeypatch):
    answers = iter(["999", "1"])
    printed = []

    monkeypatch.setattr(setup_wizard.Prompt, "ask", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr(
        setup_wizard.console, "print", lambda msg="", *args, **kwargs: printed.append(str(msg))
    )

    provider = setup_wizard._pick_provider()

    assert provider == setup_wizard.PROVIDERS[0]
    assert any(str(len(setup_wizard.PROVIDERS)) in msg for msg in printed)
    assert not any("{len(PROVIDERS)}" in msg for msg in printed)
