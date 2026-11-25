"""
tests/test_stage36_session_fidelity_and_security.py - Session Data Fidelity,
Estimates Restoration, and Security Hardening Test Suite.
"""

import os
import json
import shutil
import zipfile
import tempfile
import unittest
import numpy as np
import pandas as pd

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.parser import evaluate_dta_expression
from engine.session_manager import save_session_bundle, load_session_bundle


class TestStage36SessionFidelityAndSecurity(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.session_path = os.path.join(self.temp_dir, "fidelity_test.potproj")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_session_data_fidelity_parquet_and_dtypes(self):
        """Verify dataset in .potproj is stored as Parquet (pyarrow) and round-trips bit-identically."""
        df = pd.DataFrame({
            "small_float": [1e-12, 2.5e-12, np.nan],
            "large_float": [1e12, 3.141592653589793e12, 1e-12],
            "large_int": np.array([9007199254740993, 9007199254740994, 9007199254740995], dtype=np.int64),
            "bool_col": [True, False, True],
            "str_col": ["alpha", "beta", "gamma"],
            "dt_col": pd.to_datetime(["2026-01-01 00:00:00", "2026-05-15 12:30:00", "2026-11-17 00:00:00"])
        })
        ds = Dataset()
        ds.load_dataframe(df)
        interp = Interpreter(ds)

        save_session_bundle(self.session_path, interp=interp)

        # 1. Archive must contain data.parquet, NOT data.json
        with zipfile.ZipFile(self.session_path, "r") as zf:
            namelist = zf.namelist()
            self.assertIn("data.parquet", namelist, f"data.parquet missing from archive: {namelist}")
            self.assertNotIn("data.json", namelist, "Legacy data.json should not be stored when pyarrow is available")

        # 2. Load into clean interpreter
        clean_ds = Dataset()
        clean_interp = Interpreter(clean_ds)
        load_session_bundle(self.session_path, clean_interp)

        # 3. Bit-identical equality and exact dtypes
        pd.testing.assert_frame_equal(ds.df, clean_interp.dataset.df, check_exact=True)

    def test_02_session_estimates_postestimation_without_reestimating(self):
        """Verify stored estimates restore with numpy arrays allowing post-estimation commands without re-fitting."""
        df = pd.DataFrame({
            "price": [5000.0, 6000.0, 7200.0, 8100.0, 9500.0, 11000.0],
            "mpg": [24.0, 22.0, 20.0, 18.0, 16.0, 14.0],
            "weight": [2200.0, 2500.0, 2800.0, 3100.0, 3400.0, 3800.0]
        })
        ds = Dataset()
        ds.load_dataframe(df)
        interp = Interpreter(ds)

        # Fit model and store
        r_fit = interp.execute("regress price mpg weight")
        self.assertTrue(r_fit.ok)
        r_store = interp.execute("estimates store m_full")
        self.assertTrue(r_store.ok)

        # Record original coefficient and SE
        orig_b_mpg = interp.registers.b["mpg"]
        orig_se_mpg = interp.registers.se["mpg"]

        save_session_bundle(self.session_path, interp=interp)

        # Load into brand new interpreter
        clean_interp = Interpreter(Dataset())
        load_session_bundle(self.session_path, clean_interp)

        # Stored model arrays must be numpy ndarrays
        m = clean_interp.estimates.last_model
        self.assertIsNotNone(m)
        self.assertIsInstance(m["beta"], np.ndarray)
        self.assertIsInstance(m["vcov"], np.ndarray)
        self.assertIsInstance(m["resid"], np.ndarray)
        self.assertIsInstance(m["y_pred"], np.ndarray)

        # 1. estimates table
        r_table = clean_interp.execute("estimates table m_full")
        self.assertTrue(r_table.ok, f"estimates table failed: {r_table.output}")
        self.assertIn("mpg", r_table.output)

        # 2. test (Wald test)
        r_test = clean_interp.execute("test mpg = 0")
        self.assertTrue(r_test.ok, f"test failed: {r_test.output}")
        self.assertIn("F(", r_test.output)

        # 3. predict
        r_pred = clean_interp.execute("predict yhat, xb")
        self.assertTrue(r_pred.ok, f"predict failed: {r_pred.output}")
        self.assertIn("yhat", clean_interp.dataset.df.columns)

        # 4. margins
        r_margins = clean_interp.execute("margins, dydx(*)")
        self.assertTrue(r_margins.ok, f"margins failed: {r_margins.output}")
        self.assertIn("marginal effects", r_margins.output.lower())

        # 5. estat hettest
        r_hettest = clean_interp.execute("estat hettest")
        self.assertTrue(r_hettest.ok, f"estat hettest failed: {r_hettest.output}")
        self.assertIn("chi2", r_hettest.output)

        # 6. Expression evaluator with _b[x] and _se[x]
        ctx = clean_interp.registers.get_context_dict()
        val_expr = evaluate_dta_expression("_b[mpg] + 2 * _se[mpg]", df=clean_interp.dataset.df, scalar_context=ctx)
        self.assertAlmostEqual(float(val_expr.iloc[0] if isinstance(val_expr, pd.Series) else val_expr), orig_b_mpg + 2 * orig_se_mpg, places=5)

    def test_03_restore_registers_macros_and_labels(self):
        """Verify e() and r() scalars, macros, and variable/value labels round-trip exactly."""
        ds = Dataset()
        ds.load_dataframe(pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
        ds.var_labels = {"x": "Predictor X", "y": "Response Y"}
        ds.val_labels = {"x": {1: "Low", 2: "Medium", 3: "High"}}
        ds.val_label_names = {"x": "x_lbl"}
        ds.dataset_label = "Fidelity Dataset Benchmark"

        interp = Interpreter(ds)
        interp.registers.set_r_scalar("N", 3.0)
        interp.registers.set_r_scalar("mean", 5.0)
        interp.registers.set_e_scalar("r2", 0.985)
        interp.registers.set_r_macro("title", "Summary Stats")
        interp.registers.set_e_macro("cmd", "regress")
        interp.execute("global study_id = \"EXP-2026-X\"")
        interp.macro_mgr.set_local("temp_flag", "active")

        save_session_bundle(self.session_path, interp=interp)

        clean_interp = Interpreter(Dataset())
        load_session_bundle(self.session_path, clean_interp)

        # Dataset labels
        self.assertEqual(clean_interp.dataset.var_labels, {"x": "Predictor X", "y": "Response Y"})
        self.assertEqual(clean_interp.dataset.val_labels, {"x": {1: "Low", 2: "Medium", 3: "High"}})
        self.assertEqual(clean_interp.dataset.dataset_label, "Fidelity Dataset Benchmark")

        # Registers
        self.assertEqual(clean_interp.registers.r_scalars.get("N"), 3.0)
        self.assertEqual(clean_interp.registers.r_scalars.get("mean"), 5.0)
        self.assertEqual(clean_interp.registers.e_scalars.get("r2"), 0.985)
        self.assertEqual(clean_interp.registers.r_macros.get("title"), "Summary Stats")
        self.assertEqual(clean_interp.registers.e_macros.get("cmd"), "regress")

        # Macros
        self.assertEqual(clean_interp.global_macros.get("study_id"), "EXP-2026-X")
        self.assertEqual(clean_interp.macro_mgr.locals.get("temp_flag"), "active")

    def test_04_reject_malformed_or_hostile_archives(self):
        """Verify that zip slips, oversized members, unexpected names, and bad JSON produce clear errors."""
        interp = Interpreter(Dataset())

        # 1. Unexpected member name
        bad_name_zip = os.path.join(self.temp_dir, "bad_name.potproj")
        with zipfile.ZipFile(bad_name_zip, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"version": 1}))
            zf.writestr("unauthorized_script.py", "print('malicious')")
        with self.assertRaises(ValueError) as cm:
            load_session_bundle(bad_name_zip, interp)
        self.assertIn("unexpected member", str(cm.exception).lower())

        # 2. Path traversal / Zip Slip
        bad_traversal_zip = os.path.join(self.temp_dir, "bad_traversal.potproj")
        with zipfile.ZipFile(bad_traversal_zip, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"version": 1}))
            zf.writestr("../escape.txt", "escape")
        with self.assertRaises(ValueError) as cm:
            load_session_bundle(bad_traversal_zip, interp)
        self.assertIn("traversal", str(cm.exception).lower())

        # 3. Invalid / Corrupted JSON
        corrupted_json_zip = os.path.join(self.temp_dir, "corrupt.potproj")
        with zipfile.ZipFile(corrupted_json_zip, "w") as zf:
            zf.writestr("manifest.json", "{ invalid json content ...")
        with self.assertRaises(ValueError) as cm:
            load_session_bundle(corrupted_json_zip, interp)
        self.assertIn("invalid json", str(cm.exception).lower())

        # 4. CLI command returns clean r(198) without traceback
        res = interp.execute(f'session load "{corrupted_json_zip}"')
        self.assertFalse(res.ok)
        self.assertEqual(res.return_code, 198)
        self.assertIn("Session load error", res.output)


if __name__ == "__main__":
    unittest.main()
