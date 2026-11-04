"""Test-suite configuration.

Set PINPOINT_REQUIRE_ALL=1 (CI does) to make any skipped test fail the run, so a
missing optional package (astropy, python-casacore, numba, codex-africanus) can never
silently switch a check off.
"""

import os

import pytest


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("PINPOINT_REQUIRE_ALL") != "1":
        return
    tr = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = tr.stats.get("skipped", []) if tr else []
    if skipped:
        tr.write_line(f"PINPOINT_REQUIRE_ALL=1: {len(skipped)} test(s) were skipped; failing the run.", red=True)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
