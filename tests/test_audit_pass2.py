"""
tests/test_audit_pass2.py
=========================
Failing tests written FIRST for deep testing pass:
1. summarize, detail (and su, d):
   - Computes percentiles (p1, p5, p10, p25, p50, p75, p90, p95, p99), smallest 4, largest 4,
     mean, std dev, variance, skewness, and kurtosis (normal=3).
   - Sets r(p1), r(p5), r(p10), r(p25), r(p50), r(p75), r(p90), r(p95), r(p99),
     r(skewness), r(kurtosis), r(Var), r(sum_w), r(sum).
2. predict newvar overwrite protection:
   - predict existingvar must raise return_code 110 ("variable already defined") unless replace option is used.
3. generate / replace dtype safety & type mismatch (r(109)):
   - Boolean expressions in generate evaluate to numeric 1 and 0 (with no pandas dtype warnings).
   - replace numeric_var = "string" raises type mismatch (r(109)).
   - replace string_var = 123 raises type mismatch (r(109)).
4. correlate lower triangle & covariance option:
   - correlate displays lower-triangular matrix.
   - correlate, covariance computes sample covariance with (N-1) divisor.
"""

import os
import unittest
import numpy as np
import pandas as pd
from engine.dataset import Dataset
from engine.interpreter import Interpreter

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
AUTO_PATH = os.path.join(DATA_DIR, "auto.dta").replace("\\", "/")

class TestAuditPass2(unittest.TestCase):
    def setUp(self):
        self.ds = Dataset()
        # Create standard test dataframe
        np.random.seed(42)
        n = 100
        x = np.random.normal(50, 10, n)
        y = 2.0 * x + np.random.normal(0, 5, n)
        str_col = ["A" if i % 2 == 0 else "B" for i in range(n)]
        df = pd.DataFrame({"x": x, "y": y, "group": str_col, "year": np.random.choice([2000, 2005, 2010], n)})
        self.ds.load_dataframe(df)
        self.interp = Interpreter(self.ds)

    def test_01_summarize_detail(self):
        """summarize, detail must print percentiles/skewness/kurtosis and populate r() registers"""
        res = self.interp.execute("summarize x, detail")
        self.assertTrue(res.ok, f"summarize x, detail failed: {res.error}")
        out = res.output
        self.assertIn("Percentiles", out)
        self.assertIn("Smallest", out)
        self.assertIn("Largest", out)
        self.assertIn("Skewness", out)
        self.assertIn("Kurtosis", out)
        self.assertIn("Variance", out)

        # Check return registers
        r = self.interp.registers.r_scalars
        self.assertIn("p50", r)
        self.assertIn("p25", r)
        self.assertIn("p75", r)
        self.assertIn("p1", r)
        self.assertIn("p99", r)
        self.assertIn("skewness", r)
        self.assertIn("kurtosis", r)
        self.assertIn("Var", r)
        self.assertAlmostEqual(r["p50"], float(np.median(self.ds.df["x"])), delta=1.0)
        # Normal kurtosis reference should be around 3.0 (not excess kurtosis 0.0)
        self.assertGreater(r["kurtosis"], 1.0)

    def test_02_predict_overwrite_protection(self):
        """predict newvar must raise r(110) if newvar exists without replace option"""
        self.interp.execute("regress y x")
        res1 = self.interp.execute("predict yhat")
        self.assertTrue(res1.ok)
        self.assertIn("yhat", self.ds.df.columns)

        # Attempting to predict yhat again without replace must fail with r(110)
        res2 = self.interp.execute("predict yhat")
        self.assertFalse(res2.ok)
        self.assertEqual(res2.return_code, 110)
        self.assertIn("already defined", res2.output.lower())

        # With replace option, it should succeed
        res3 = self.interp.execute("predict yhat, replace")
        self.assertTrue(res3.ok)

    def test_03_generate_boolean_and_replace_type_mismatch(self):
        """generate boolean produces numeric 0/1; replace across mismatched types raises r(109)"""
        # Boolean comparison in generate
        res_gen = self.interp.execute("generate treated = (year >= 2005)")
        self.assertTrue(res_gen.ok)
        self.assertTrue(pd.api.types.is_numeric_dtype(self.ds.df["treated"]))
        self.assertTrue(set(self.ds.df["treated"].dropna().unique()).issubset({0.0, 1.0}))

        # Type mismatch: replace numeric with string
        res_mismatch1 = self.interp.execute('replace x = "text" if year == 2005')
        self.assertFalse(res_mismatch1.ok)
        self.assertEqual(res_mismatch1.return_code, 109)
        self.assertIn("type mismatch", res_mismatch1.output.lower())

        # Type mismatch: replace string with numeric
        res_mismatch2 = self.interp.execute('replace group = 99 if year == 2005')
        self.assertFalse(res_mismatch2.ok)
        self.assertEqual(res_mismatch2.return_code, 109)
        self.assertIn("type mismatch", res_mismatch2.output.lower())

    def test_04_correlate_lower_triangle_and_covariance(self):
        """correlate must output lower triangle; covariance option must compute sample covariance"""
        res_corr = self.interp.execute("correlate x y")
        self.assertTrue(res_corr.ok)
        lines = res_corr.output.strip().splitlines()
        # Verify lower-triangular structure: first row has 1 number, second row has 2 numbers
        # Find the line starting with x and line starting with y
        x_line = next(l for l in lines if l.strip().startswith("x"))
        y_line = next(l for l in lines if l.strip().startswith("y"))
        # In x_line, after |, only 1.0000 should appear
        x_vals = x_line.split("|")[1].split()
        self.assertEqual(len(x_vals), 1)
        # In y_line, after |, 2 values should appear (corr and 1.0000)
        y_vals = y_line.split("|")[1].split()
        self.assertEqual(len(y_vals), 2)

        # Covariance option
        res_cov = self.interp.execute("correlate x y, covariance")
        self.assertTrue(res_cov.ok)
        self.assertIn("cov", res_cov.output.lower())
        # Covariance of x with x is Var(x)
        sample_var_x = float(self.ds.df["x"].var(ddof=1))
        # Formatted with 4 decimal places matching Reference
        self.assertTrue(any(f"{sample_var_x:.4f}" in l for l in res_cov.output.splitlines()))

    def test_05_test_varlist_joint_null(self):
        """test var1 var2 must test joint hypothesis that all coefficients are zero"""
        self.interp.execute("regress y x year")
        res = self.interp.execute("test x year")
        self.assertTrue(res.ok, f"test x year failed: {res.error}")
        self.assertIn("F(", res.output)
        self.assertIn("Prob > F", res.output)
        # Check return registers
        r = self.interp.registers.r_scalars
        self.assertIn("F", r)
        self.assertIn("p", r)
        self.assertIn("df", r)
        self.assertEqual(r["df"], 2.0)
        self.assertIn("df_r", r)

    def test_06_regress_noconstant(self):
        """regress y x, noconstant must omit _cons from X and coef table, and use df_r = n - k"""
        res = self.interp.execute("regress y x, noconstant")
        self.assertTrue(res.ok, f"regress noconstant failed: {res.error}")
        out = res.output
        # _cons must NOT appear in the coefficient table
        self.assertNotIn("_cons", out)
        # Model object in estimates must NOT have _cons
        last_m = self.interp.estimates.last_model
        self.assertNotIn("_cons", last_m["var_names"])
        self.assertEqual(len(last_m["coefs"]), 1)
        self.assertEqual(last_m["coefs"][0]["name"], "x")
        # df_r should be n - k = 100 - 1 = 99 (not 98)
        self.assertEqual(last_m["df_r"], 99)
        self.assertEqual(last_m["df_m"], 1)

    def test_07_margins_with_factor_variables(self):
        """margins, dydx(*) must successfully compute AME when model contains factor variables"""
        self.interp.execute(f'use "{AUTO_PATH}", clear')
        self.interp.execute("logit foreign i.rep78 weight")
        res = self.interp.execute("margins, dydx(*)")
        self.assertTrue(res.ok, f"margins with factor variables failed: {res.error}")
        out = res.output
        self.assertIn("Average marginal effects", out)
        # Should include expanded levels of rep78 (e.g. 2.rep78, 3.rep78) and weight
        self.assertIn("weight", out)
        self.assertIn("rep78", out)

    def test_08_test_factor_variable_expansion(self):
        """test i.varname must jointly test all levels of the factor variable"""
        self.interp.execute(f'use "{AUTO_PATH}", clear')
        self.interp.execute("regress price i.rep78 weight")
        res = self.interp.execute("test i.rep78")
        self.assertTrue(res.ok, f"test i.rep78 failed: {res.error}")
        self.assertIn("F(  4,", res.output)
        r = self.interp.registers.r_scalars
        self.assertEqual(r["df"], 4.0)

if __name__ == "__main__":
    unittest.main()
