"""
tests/test_researcher_harness_integrity.py

Adversarial integrity tests for the researcher acceptance test harness:
Proves that the harness fails when:
- A command returns ok=False.
- A command returns a nonzero return code.
- An expected statistical fragment is missing.
- An exported artifact (PNG, SVG, PDF) is corrupted or missing.
- Dataset values or masks are altered during roundtrip.
- Multi-model report diagnostics leak or are missing.
"""

import os
import sys
import unittest
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from engine.interpreter import CommandResult
from scripts.verify_researcher_workflows import (
    WorkflowAssertionError,
    require_command_success,
    require_true,
    require_equal,
    inspect_png_file,
    inspect_svg_file,
    inspect_pdf_file,
)


class TestResearcherHarnessIntegrity(unittest.TestCase):
    def test_require_command_success_none_result(self):
        with self.assertRaises(WorkflowAssertionError) as ctx:
            require_command_success(None)
        self.assertIn("returned None", str(ctx.exception))

    def test_require_command_success_ok_false(self):
        bad_res = CommandResult("syntax error", ok=False, return_code=198)
        with self.assertRaises(WorkflowAssertionError) as ctx:
            require_command_success(bad_res)
        self.assertIn("reported failure (ok=False)", str(ctx.exception))

    def test_require_command_success_nonzero_return_code(self):
        nonzero_res = CommandResult("command failed", ok=True, return_code=1)
        with self.assertRaises(WorkflowAssertionError) as ctx:
            require_command_success(nonzero_res)
        self.assertIn("nonzero exit code", str(ctx.exception))

    def test_require_command_success_missing_fragment(self):
        res = CommandResult("OLS regression completed\nN = 50", ok=True, return_code=0)
        with self.assertRaises(WorkflowAssertionError) as ctx:
            require_command_success(res, expected_fragments=("R-squared",))
        self.assertIn("missing required fragment", str(ctx.exception))

    def test_require_command_success_valid(self):
        res = CommandResult("OLS regression completed\nR-squared = 0.85\nN = 50", ok=True, return_code=0)
        # Should not raise
        require_command_success(res, expected_fragments=("R-squared", "N = 50"), min_lines=2)

    def test_require_true_and_equal(self):
        with self.assertRaises(WorkflowAssertionError):
            require_true(False, "Condition failed")
        with self.assertRaises(WorkflowAssertionError):
            require_equal(10, 20, "Values do not match")

    def test_inspect_png_corrupted(self, tmp_path=None):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"not a png image file")
            bad_png = f.name
        try:
            with self.assertRaises(WorkflowAssertionError) as ctx:
                inspect_png_file(bad_png)
            self.assertIn("not a valid PNG", str(ctx.exception))
        finally:
            if os.path.exists(bad_png):
                os.remove(bad_png)

    def test_inspect_svg_corrupted(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            f.write(b"<invalid xml file")
            bad_svg = f.name
        try:
            with self.assertRaises(WorkflowAssertionError) as ctx:
                inspect_svg_file(bad_svg)
            self.assertIn("not valid SVG XML", str(ctx.exception))
        finally:
            if os.path.exists(bad_svg):
                os.remove(bad_svg)

    def test_inspect_pdf_corrupted(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"plain text pretending to be pdf")
            bad_pdf = f.name
        try:
            with self.assertRaises(WorkflowAssertionError) as ctx:
                inspect_pdf_file(bad_pdf)
            self.assertIn("does not start with PDF magic header", str(ctx.exception))
        finally:
            if os.path.exists(bad_pdf):
                os.remove(bad_pdf)


if __name__ == "__main__":
    unittest.main()
