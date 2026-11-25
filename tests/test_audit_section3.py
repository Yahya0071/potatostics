"""
tests/test_audit_section3.py - TDD Verification Suite for Section 3 (Claims & Copilot)
"""

import os
import re
import unittest
import numpy as np
import pandas as pd

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.copilot import CopilotAssistant
from engine.copilot_egm import ElephantGoldfishCopilot


class TestSection3ClaimsAndCopilot(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.dataset = Dataset()
        self.dataset.df = pd.DataFrame({
            'price': [100.0, 200.0, 300.0, 400.0, 500.0],
            'mpg': [20.0, 25.0, 18.0, 30.0, 15.0],
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'extra': [10.0, 20.0, 30.0, 40.0, 50.0],
            'tax': [5.0, 5.0, 5.0, 5.0, 5.0],
            'unit': [1, 1, 2, 2, 3]
        })
        self.interp = Interpreter(self.dataset)

    def test_3a_no_prohibited_claims_in_text(self):
        """Prohibited claims must not appear in README, GUI dialogs, banners, or docstrings."""
        prohibited = [
            r'clean-room',
            r'zero-data-leakage',
            r'60\s*fps',
            r'ast security sandbox',
            r'elephant-goldfish\s+ai',
            r'117[-–]119',
            r'1st-stage diagnostics'
        ]

        files_to_check = [
            'README.md',
            os.path.join('gui_qt', 'main_window.py'),
            os.path.join('gui_qt', 'data_grid.py'),
            os.path.join('gui', 'app.py'),
            os.path.join('gui', 'branding.py'),
            os.path.join('engine', 'dataset.py'),
            os.path.join('engine', 'copilot.py'),
            os.path.join('engine', 'copilot_egm.py'),
            os.path.join('docs', 'potatostics_cheat_sheet.md')
        ]

        violations = []
        for fpath in files_to_check:
            if not os.path.exists(fpath):
                continue
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
                content = fp.read()
            for pattern in prohibited:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    violations.append(f"{fpath}: matches prohibited claim '{matches[0]}'")

        self.assertEqual(violations, [], msg="Prohibited claims found:\n" + "\n".join(violations))

    def test_3b_assistant_name_and_associational_wording(self):
        """Assistant must be named 'Command assistant (rule-based)' and use associational wording."""
        copilot = CopilotAssistant(self.interp)
        res = copilot.ask("regress price on mpg")
        self.assertIn("Command assistant (rule-based)", res)
        self.assertIn("associated with", res.lower())
        self.assertNotIn(" increases price", res)

    def test_3c_word_boundary_variable_matching(self):
        """Word-boundary regexes must prevent matching substrings (e.g. 'x' vs 'extra' or 'tax')."""
        copilot = CopilotAssistant(self.interp)
        # Query specifies 'x', not 'extra' or 'tax'
        vars_found = copilot._find_variables("regress price on x", list(self.dataset.df.columns))
        self.assertIn("x", vars_found)
        self.assertNotIn("extra", vars_found)
        self.assertNotIn("tax", vars_found)

    def test_3d_add_robust_vce_conflict_prevention(self):
        """'add robust' must not conflict with an existing vce() option."""
        egm = ElephantGoldfishCopilot(self.interp)
        self.interp.copilot = egm

        # Run with existing cluster vce
        self.interp.execute("regress price mpg, vce(cluster unit)")
        res = egm.ask("add robust standard errors")
        self.assertIn("already", res.lower(), msg="'add robust' must reject conflict when vce() already exists")
        self.assertNotIn(", robust", res)

    def test_3e_local_endpoint_removed_or_opt_in(self):
        """Unused local_endpoint must be removed from default initialization."""
        egm = ElephantGoldfishCopilot(self.interp)
        self.assertFalse(hasattr(egm, 'local_endpoint') and egm.local_endpoint is not None,
                         msg="local_endpoint must not be present by default without explicit opt-in")


if __name__ == '__main__':
    unittest.main()
