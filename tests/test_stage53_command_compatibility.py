"""
tests/test_stage53_command_compatibility.py

Stage 53 — Priority 6: Targeted Command Compatibility & Syntax Evaluation.
Verifies the daily econometrician research script:
- clear, import delimited, describe, summarize
- generate / gen transformations with arithmetic and math builtins
- regress classical & robust HC1 [vce(robust)]
- predict xb (fitted values) & predict resid (residuals)
- estimates store, estimates table / esttab comparison
- macro evaluation ($global)
- stored scalar returns (r(N), r(mean), e(N), e(r2))
- script execution and batch/GUI equivalence
"""

import tempfile
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from engine.dataset import Dataset
from engine.interpreter import Interpreter


class TestStage53CommandCompatibility(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.ds = Dataset()
        self.interp = Interpreter(self.ds)

        # Generate sample research dataset
        rng = np.random.default_rng(12345)
        n = 500
        educ = rng.choice([10, 12, 14, 16, 18], size=n)
        exp = rng.uniform(1, 35, size=n)
        # wage DGP: log(wage) = 1.5 + 0.08*educ + 0.04*exp - 0.0005*exp^2 + e
        error = rng.normal(0, 0.25, size=n)
        log_w = 1.5 + 0.08 * educ + 0.04 * exp - 0.0005 * (exp ** 2) + error
        wage = np.exp(log_w)

        self.sample_df = pd.DataFrame({
            "wage": wage,
            "educ": educ,
            "experience": exp
        })
        self.sample_csv = self.tmp_dir / "sample.csv"
        self.sample_df.to_csv(self.sample_csv, index=False)

    def test_export_delimited_writes_csv_and_requires_replace(self):
        self.ds.load_dataframe(pd.DataFrame({"name": ['a, "b"'], "id": [2**53 + 1]}))
        base = self.tmp_dir / "sample export"
        path = base.with_suffix(".csv")
        command = f'export delimited using "{base}"'
        result = self.interp.execute(command)
        self.assertTrue(result.ok, result.output)
        self.assertTrue(path.is_file())
        self.assertFalse(base.with_suffix(".dta").exists())
        pd.testing.assert_frame_equal(pd.read_csv(path), self.ds.df)
        saved = path.read_bytes()
        self.ds.df.at[0, "name"] = "replacement"
        self.assertEqual(self.interp.execute(command).return_code, 602)
        self.assertEqual(path.read_bytes(), saved)
        self.assertTrue(self.interp.execute(command + ", replace").ok)
        pd.testing.assert_frame_equal(pd.read_csv(path), self.ds.df)
        for ext in (".txt", ".gz"):
            explicit = base.with_suffix(ext)
            self.assertTrue(self.interp.execute(f'export delimited "{explicit}"').ok)
            pd.testing.assert_frame_equal(pd.read_csv(explicit), self.ds.df)

    def test_delimited_export_error_preserves_existing_data(self):
        self.ds.load_dataframe(self.sample_df)
        binary = self.tmp_dir / "original.parquet"
        binary.write_bytes(b"saved content")
        result = self.interp.execute(f'export delimited using "{binary}", replace')
        self.assertFalse(result.ok)
        self.assertEqual(binary.read_bytes(), b"saved content")
        self.assertTrue(self.interp.execute(f'export using "{binary}", replace').ok)
        pd.testing.assert_frame_equal(pd.read_parquet(binary), self.ds.df)
        original_csv = self.sample_csv.read_bytes()
        original_filename = self.ds.filename

        def fail_write(df, path, **kwargs):
            Path(path).write_bytes(b"partial")
            raise OSError("injected write failure")

        with patch.object(pd.DataFrame, "to_csv", fail_write):
            self.assertFalse(self.interp.execute(f'export delimited "{self.sample_csv}", replace').ok)
        self.assertEqual(self.sample_csv.read_bytes(), original_csv)
        self.assertEqual(self.ds.filename, original_filename)

    def test_01_command_research_workflow_script(self):
        """Executes the standard everyday research script and verifies estimation and postestimation."""
        # 1. clear
        res = self.interp.execute("clear")
        self.assertTrue(res.ok)
        self.assertEqual(len(self.ds.df), 0)

        # 2. import delimited
        res = self.interp.execute(f'import delimited "{self.sample_csv}", clear')
        self.assertTrue(res.ok)
        self.assertEqual(len(self.ds.df), 500)
        self.assertListEqual(list(self.ds.df.columns), ["wage", "educ", "experience"])

        # 3. describe
        res = self.interp.execute("describe")
        self.assertTrue(res.ok)
        self.assertIn("wage", res.output)
        self.assertIn("educ", res.output)

        # 4. summarize
        res = self.interp.execute("summarize")
        self.assertTrue(res.ok)
        self.assertIn("Mean", res.output)
        r_n = self.interp.registers.get_r("N")
        self.assertEqual(r_n, 500.0)

        # 5. generate log_wage = log(wage)
        res = self.interp.execute("generate log_wage = log(wage)")
        self.assertTrue(res.ok)
        self.assertIn("log_wage", self.ds.df.columns)

        # 6. generate exp_sq = experience^2
        res = self.interp.execute("generate exp_sq = experience^2")
        self.assertTrue(res.ok)
        self.assertIn("exp_sq", self.ds.df.columns)

        # 7. regress log_wage educ experience exp_sq
        res_ols = self.interp.execute("regress log_wage educ experience exp_sq")
        self.assertTrue(res_ols.ok)
        self.assertEqual(res_ols.return_code, 0)
        self.assertIn("Number of obs", res_ols.output)
        e_n = self.interp.registers.get_e("N")
        self.assertEqual(e_n, 500.0)
        e_r2 = self.interp.registers.get_e("r2")
        self.assertGreater(e_r2, 0.4)

        # 8. regress log_wage educ experience exp_sq, vce(robust)
        res_robust = self.interp.execute("regress log_wage educ experience exp_sq, vce(robust)")
        self.assertTrue(res_robust.ok)
        self.assertIn("Robust", res_robust.output)

        # 9. predict fitted, xb
        res = self.interp.execute("predict fitted, xb")
        self.assertTrue(res.ok)
        self.assertIn("fitted", self.ds.df.columns)

        # 10. predict residuals, resid
        res = self.interp.execute("predict residuals, resid")
        self.assertTrue(res.ok)
        self.assertIn("residuals", self.ds.df.columns)

        # 11. estimates store baseline
        res = self.interp.execute("estimates store baseline")
        self.assertTrue(res.ok)
        self.assertIn("baseline", self.interp.estimates.store)

        # 12. Nested model comparison: regress without exp_sq
        res = self.interp.execute("regress log_wage educ experience")
        self.assertTrue(res.ok)
        self.interp.execute("estimates store restricted")

        # 13. esttab table export
        tex_file = self.tmp_dir / "models.tex"
        res_tab = self.interp.execute(f'esttab baseline restricted using "{tex_file}", replace b(3) se(3)')
        self.assertTrue(res_tab.ok)
        self.assertTrue(tex_file.exists())
        tex_text = tex_file.read_text(encoding="utf-8")
        self.assertIn("educ", tex_text)
        self.assertIn("baseline", tex_text)

    def test_02_command_macros_and_syntax_aliases(self):
        """Verifies global macro resolution and common command aliases."""
        self.ds.load_dataframe(self.sample_df.copy())
        self.interp.execute("gen lw = log(wage)")
        
        # Test global macro
        self.interp.execute("global X_VARS educ experience")
        self.assertEqual(self.interp.macro_mgr.get_global("X_VARS"), "educ experience")

        # Execute command using $macro
        res = self.interp.execute("reg lw $X_VARS")
        self.assertTrue(res.ok)
        self.assertIn("educ", res.output)
        self.assertIn("experience", res.output)

    def test_03_do_file_script_execution(self):
        """Verifies multi-line do-file script execution via interpreter."""
        do_file = self.tmp_dir / "analysis.do"
        do_file.write_text(
            f'import delimited "{self.sample_csv}", clear\n'
            'gen l_wage = log(wage)\n'
            'reg l_wage educ experience\n'
            'predict y_hat, xb\n',
            encoding="utf-8"
        )
        res = self.interp.execute(f'do "{do_file}"')
        self.assertTrue(res.ok)
        self.assertIn("y_hat", self.ds.df.columns)
        self.assertEqual(len(self.ds.df), 500)


if __name__ == "__main__":
    unittest.main()
