"""Tests for the default-on logging decision (epoch-memory-arch R3 G1).

Session logging is on by default so the recall tool can retrieve past events.
The default-on inversion (``debug or not no_log``) is easy to break in a
refactor; these tests pin the truth table without booting the REPL.
"""

from supercoder.main import resolve_log_enabled


class TestResolveLogEnabled:
    def test_default_logging_on(self):
        assert resolve_log_enabled(debug=False, no_log=False) is True

    def test_no_log_disables(self):
        assert resolve_log_enabled(debug=False, no_log=True) is False

    def test_debug_forces_on_overrides_no_log(self):
        # --debug wins over --no-log: debug runs need the log regardless.
        assert resolve_log_enabled(debug=True, no_log=True) is True

    def test_debug_alone_is_on(self):
        assert resolve_log_enabled(debug=True, no_log=False) is True
