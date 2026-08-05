"""Tests for the default-on logging decision (epoch-memory-arch R3 G1).

Session logging is on by default so the recall tool can retrieve past events.
The default-on inversion (``debug or not no_log``) is easy to break in a
refactor; these tests pin the truth table without booting the REPL.
"""

import ast

from supercoder import main as main_module
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


class TestMainDelegatesToResolveLogEnabled:
    """R4 N4: main() must actually call resolve_log_enabled.

    The unit tests above pin the function's truth table, but if someone
    reverted the one-line delegation back to an inline ``config.debug or not
    no_log``, those tests would stay green while production stopped using the
    testable function. This AST check pins the call site without booting the
    REPL (which is hard to drive to the logger-init line on an early-exit
    path). It verifies main()'s body contains a real Call to
    resolve_log_enabled — not just a mention in a docstring or string.
    """

    def _main_calls(self):
        """AST Call nodes to resolve_log_enabled within main()'s body."""
        import inspect

        # main is a click Command; .callback is the underlying function.
        source = inspect.getsource(main_module.main.callback)
        tree = ast.parse(source)
        return [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "resolve_log_enabled"
        ]

    def test_main_calls_resolve_log_enabled(self):
        calls = self._main_calls()
        assert calls, "main() does not call resolve_log_enabled — delegation reverted?"

    def test_main_passes_debug_and_no_log(self):
        """The call must wire both the debug and no_log flags through."""
        call = self._main_calls()[0]
        # Two arguments total (positional debug, no_log or matching keywords).
        assert len(call.args) + len(call.keywords) == 2
