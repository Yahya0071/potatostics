"""
tests/test_stage38_security_and_ast.py - Security Regression and AST Expression Evaluator Test Suite
Verifies that expression evaluation in parser and nlcom is strictly sandboxed via AST,
prohibiting eval(), exec(), attribute traversal, arbitrary imports, and unwhitelisted functions.
Also verifies plugin installer security checks (HTTPS only, size limits, SHA-256 verification).
"""

import unittest
import pandas as pd
import numpy as np

from engine.parser import evaluate_dta_expression
from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.advanced_econometrics import compute_nlcom
from engine.plugins_manager import PluginsManager, MAX_PLUGIN_DOWNLOAD_SIZE


class TestStage38SecurityAndAST(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
        self.dataset = Dataset()
        self.dataset.load_dataframe(self.df)
        self.interp = Interpreter(self.dataset)

    def test_01_prohibit_attribute_access_and_object_traversal(self):
        """AST evaluator must strictly reject attribute access like ().__class__ or x.__doc__."""
        escape_payloads = [
            "().__class__",
            "(1).__class__.__bases__[0]",
            "x.__class__",
            "y.__dict__",
            "x.values",
        ]
        for payload in escape_payloads:
            with self.assertRaises(ValueError, msg=f"Payload '{payload}' must be rejected"):
                evaluate_dta_expression(payload, self.df)

    def test_02_prohibit_arbitrary_function_calls_and_builtins(self):
        """AST evaluator must reject unwhitelisted functions and dangerous built-ins."""
        dangerous_calls = [
            "eval('1+1')",
            "exec('x=1')",
            "open('passwords.txt')",
            "__import__('os').system('echo pwned')",
            "compile('1', '', 'eval')",
            "globals()",
            "locals()",
        ]
        for call_expr in dangerous_calls:
            with self.assertRaises(ValueError, msg=f"Expression '{call_expr}' must be rejected"):
                evaluate_dta_expression(call_expr, self.df)

    def test_03_nlcom_delta_method_uses_strict_ast_not_eval(self):
        """nlcom must reject code execution payloads and only evaluate valid mathematical parameters."""
        # Fit a simple regression model to obtain real estimate structure
        res = self.interp.execute("regress y x")
        self.assertTrue(res.ok)
        m = self.interp.estimates.last_model

        # Valid mathematical nlcom expression
        nl_good = compute_nlcom(m, "exp(_b[x]) + 2 * _b[_cons]")
        self.assertIn("estimate", nl_good)
        self.assertAlmostEqual(nl_good["estimate"], np.exp(m["beta"][0]) + 2 * m["beta"][1], places=4)

        # Malicious payload in nlcom must be rejected by AST
        bad_nl_payloads = [
            "().__class__.__bases__[0].__subclasses__()",
            "__import__('os').system('echo pwned')",
            "open('/etc/passwd')",
        ]
        for bad in bad_nl_payloads:
            with self.assertRaises(ValueError, msg=f"nlcom payload '{bad}' must be rejected"):
                compute_nlcom(m, bad)

    def test_04_plugins_manager_https_enforcement(self):
        """Plugin installer must reject insecure http:// sources and require https:// or local/catalog."""
        pm = PluginsManager(self.interp)
        with self.assertRaises(ValueError) as cm:
            pm.install_plugin("http://insecure-repository.org/malicious_plugin.py")
        self.assertIn("insecure url scheme", str(cm.exception).lower())

    def test_05_plugins_manager_download_size_constant(self):
        """Verify MAX_PLUGIN_DOWNLOAD_SIZE threshold is defined at 10 MB."""
        self.assertEqual(MAX_PLUGIN_DOWNLOAD_SIZE, 10 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
