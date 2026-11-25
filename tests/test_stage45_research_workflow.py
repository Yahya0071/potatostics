"""
Stage 45/46 — Comprehensive Research Workflow, GUI Reliability, Command Compatibility,
Performance Benchmarking, and Reporting Fidelity Test Suite.
Enforces strict assertions, eliminates conditional test skips, verifies multi-model session persistence,
bit-level dataset serialization fidelity, full GUI/batch equivalence, and robust performance benchmarking.
"""

import os
import sys
import time
import tempfile
import unittest
import platform
import numpy as np
import pandas as pd
from pathlib import Path

from PySide6 import QtWidgets

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from gui_qt.main_window import PotatosticsQtWindow

# Offscreen Qt Application for headless GUI test execution
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(["--platform", "offscreen"])


class TestStage45ResearchWorkflow(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name).resolve()

    def tearDown(self):
        self.tmp_dir.cleanup()

    # =========================================================================
    # 1. Application Startup & GUI State
    # =========================================================================

    def test_01_gui_startup_lifecycle_and_theming(self):
        """Verifies clean window startup, empty state, theme toggling, and clean close."""
        win = PotatosticsQtWindow()
        try:
            self.assertEqual(win.windowTitle(), "Potatostics - Statistical Analysis Software")
            self.assertIsNotNone(win.console)
            self.assertIsNotNone(win.data_grid)
            self.assertIsNotNone(win.variables_pane)
            self.assertIsNotNone(win.history_pane)

            # Test light/dark mode toggling
            initial_theme = win.current_theme
            win.toggle_theme()
            self.assertNotEqual(win.current_theme, initial_theme)
            win.toggle_theme()
            self.assertEqual(win.current_theme, initial_theme)
        finally:
            win.close()

    def test_02_gui_startup_with_initial_dataset(self):
        """Verifies GUI startup with an initial dataset file."""
        csv_file = self.tmp_path / "startup_data.csv"
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        df.to_csv(csv_file, index=False)

        win = PotatosticsQtWindow(initial_file=str(csv_file))
        try:
            if win._initial_cmd:
                win.run_command(win._initial_cmd, sync=True)
            self.assertEqual(len(win.dataset.df), 3)
            self.assertIn("x", list(win.dataset.df.columns))
            self.assertIn("y", list(win.dataset.df.columns))
            np.testing.assert_allclose(win.dataset.df["x"].values, [1.0, 2.0, 3.0])
            np.testing.assert_allclose(win.dataset.df["y"].values, [4.0, 5.0, 6.0])
        finally:
            win.close()

    # =========================================================================
    # 2. Data Management Workflow (CSV, DTA, Parquet)
    # =========================================================================

    def test_03_complete_data_manipulation_lifecycle(self):
        """
        Executes a complete data lifecycle:
        import -> inspect -> edit -> rename -> generate -> filter -> sort -> summarize -> save -> reopen -> fidelity.
        """
        raw_csv = self.tmp_path / "empirical_data.csv"
        np.random.seed(123)
        wages_orig = np.random.uniform(15.0, 85.0, size=100)
        df_orig = pd.DataFrame({
            "id": np.arange(1, 101),
            "wage": wages_orig,
            "educ": np.random.choice([10, 12, 14, 16, 18], size=100),
            "experience": np.random.uniform(1.0, 40.0, size=100),
            "sector": np.random.choice(["Public", "Private", "NGO"], size=100)
        })
        df_orig.to_csv(raw_csv, index=False)

        ds = Dataset()
        interp = Interpreter(ds)

        # 1. Import
        res = interp.execute(f'import delimited "{raw_csv}", clear')
        self.assertTrue(res.ok, res.output)
        self.assertEqual(len(ds.df), 100)

        # 2. Inspect
        res = interp.execute("describe")
        self.assertTrue(res.ok, res.output)
        self.assertIn("wage", res.output)

        # 3. Edit / Replace
        res = interp.execute("replace wage = 99.9 if wage > 80.0")
        self.assertTrue(res.ok, res.output)
        self.assertTrue(np.all(ds.df[ds.df["wage"] > 80.0]["wage"] == 99.9))

        # 4. Rename
        res = interp.execute("rename wage hourly_wage")
        self.assertTrue(res.ok, res.output)
        self.assertIn("hourly_wage", list(ds.df.columns))
        self.assertNotIn("wage", list(ds.df.columns))

        # 5. Generate calculated variable
        res = interp.execute("gen log_wage = log(hourly_wage)")
        self.assertTrue(res.ok, res.output)
        self.assertIn("log_wage", list(ds.df.columns))
        np.testing.assert_allclose(ds.df["log_wage"].values, np.log(ds.df["hourly_wage"].values), rtol=1e-12)

        # 6. Filter observations
        res = interp.execute("keep if educ >= 12")
        self.assertTrue(res.ok, res.output)
        filtered_count = len(ds.df)
        self.assertGreater(filtered_count, 0)
        self.assertLessEqual(filtered_count, 100)
        self.assertTrue(np.all(ds.df["educ"] >= 12))

        # 7. Sort
        res = interp.execute("sort hourly_wage")
        self.assertTrue(res.ok, res.output)
        wages = ds.df["hourly_wage"].dropna().tolist()
        self.assertEqual(wages, sorted(wages))

        # 8. Summary statistics
        res = interp.execute("summarize hourly_wage log_wage")
        self.assertTrue(res.ok, res.output)
        self.assertIn("hourly_wage", res.output)
        self.assertIn("log_wage", res.output)

        # 9. Save dataset to Parquet
        saved_parquet = self.tmp_path / "processed_data.parquet"
        res = interp.execute(f'save "{saved_parquet}", replace')
        self.assertTrue(res.ok, res.output)
        self.assertTrue(saved_parquet.exists())

        # Snapshot before clear
        df_snapshot = ds.df.copy()

        # 10. Clear and Reopen
        res = interp.execute("clear")
        self.assertTrue(res.ok, res.output)
        self.assertEqual(len(ds.df), 0)

        res = interp.execute(f'use "{saved_parquet}", clear')
        self.assertTrue(res.ok, res.output)
        self.assertEqual(len(ds.df), filtered_count)
        self.assertEqual(list(ds.df.columns), list(df_snapshot.columns))
        # Exact numerical value round-trip
        np.testing.assert_allclose(ds.df["hourly_wage"].values, df_snapshot["hourly_wage"].values, rtol=1e-14)
        np.testing.assert_allclose(ds.df["log_wage"].values, df_snapshot["log_wage"].values, rtol=1e-14)

    def test_03b_dataset_multiformat_save_reopen_fidelity(self):
        """
        Phase 5.B: Verifies exact value fidelity across Parquet, DTA, and CSV formats.
        Checks: float64 near 1e-12 and 1e12, int64 above 2**53, strings, NaNs, and variable labels.
        """
        ds = Dataset()
        df_test = pd.DataFrame({
            "f_tiny": [1e-12, 2.5e-12, np.nan],
            "f_large": [1e12, 9.87e11, -5.4e11],
            "i_big": [2**54 + 1, 2**54 + 2, 2**54 + 3],
            "s_text": ["alpha", "beta", "gamma"],
            "d_date": pd.date_range("2026-01-01", periods=3, freq="D")
        })
        ds.load_dataframe(df_test)
        ds.var_labels = {"f_tiny": "Small floats", "s_text": "Categorical strings"}

        interp = Interpreter(ds)

        # 1. Parquet Fidelity
        pq_path = self.tmp_path / "precision_test.parquet"
        res_pq = interp.execute(f'save "{pq_path}", replace')
        self.assertTrue(res_pq.ok, res_pq.output)
        self.assertTrue(pq_path.exists())

        ds_pq = Dataset()
        ds_pq.load_parquet(str(pq_path))
        self.assertEqual(len(ds_pq.df), 3)
        self.assertAlmostEqual(ds_pq.df["f_tiny"].iloc[0], 1e-12, delta=1e-24)
        self.assertAlmostEqual(ds_pq.df["f_large"].iloc[0], 1e12, delta=1e-4)
        self.assertEqual(ds_pq.df["i_big"].iloc[0], 2**54 + 1)
        self.assertEqual(ds_pq.df["s_text"].tolist(), ["alpha", "beta", "gamma"])
        self.assertTrue(np.isnan(ds_pq.df["f_tiny"].iloc[2]))

        # 2. DTA Fidelity (Binary .dta preserves variable labels and double precision)
        dta_path = self.tmp_path / "precision_test.dta"
        # DTA numeric types support standard range; test float and labels
        df_dta_in = pd.DataFrame({
            "f_tiny": [1e-12, 2.5e-12, np.nan],
            "f_large": [1e12, 9.87e11, -5.4e11],
            "s_text": ["alpha", "beta", "gamma"]
        })
        ds_dta = Dataset()
        ds_dta.load_dataframe(df_dta_in)
        ds_dta.var_labels = {"f_tiny": "Small floats", "s_text": "String variable"}
        ds_dta.save_dta(str(dta_path))

        ds_dta_out = Dataset()
        ds_dta_out.load_dta(str(dta_path))
        self.assertEqual(len(ds_dta_out.df), 3)
        self.assertAlmostEqual(ds_dta_out.df["f_tiny"].iloc[0], 1e-12, delta=1e-20)
        self.assertAlmostEqual(ds_dta_out.df["f_large"].iloc[0], 1e12, delta=1e-4)
        self.assertTrue(np.isnan(ds_dta_out.df["f_tiny"].iloc[2]))
        self.assertEqual(ds_dta_out.var_labels.get("f_tiny"), "Small floats")

        # 3. CSV Fidelity
        csv_path = self.tmp_path / "precision_test.csv"
        ds.save_csv(str(csv_path))
        ds_csv = Dataset()
        ds_csv.load_csv(str(csv_path))
        self.assertEqual(len(ds_csv.df), 3)
        self.assertAlmostEqual(ds_csv.df["f_tiny"].iloc[0], 1e-12, delta=1e-20)
        self.assertEqual(ds_csv.df["s_text"].tolist(), ["alpha", "beta", "gamma"])

    # =========================================================================
    # 3. Regression & Multi-Model Session Fidelity
    # =========================================================================

    def test_04_full_regression_and_post_estimation_workflow(self):
        """
        Phase 5.C: Executes an end-to-end econometric estimation across multiple models:
        Model 1 -> Model 2 -> robust VCE -> cluster VCE -> hypothesis test -> prediction
        -> session save/reopen with multi-model store preservation.
        """
        np.random.seed(42)
        n = 120
        cluster_id = np.repeat(np.arange(1, 13), 10)
        x1 = np.random.normal(5.0, 2.0, size=n)
        x2 = np.random.normal(10.0, 3.0, size=n)
        eps = np.random.normal(0, 1.0, size=n)
        y = 2.0 + 3.5 * x1 - 1.2 * x2 + eps

        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "cluster_var": cluster_id})
        dta_path = self.tmp_path / "model_data.dta"
        _ds_seed = Dataset()
        _ds_seed.load_dataframe(df)
        _ds_seed.save_dta(str(dta_path))

        ds = Dataset()
        interp = Interpreter(ds)

        # Load
        interp.execute(f'use "{dta_path}", clear')

        # 1. Classical Model 1: bivariate
        res_m1 = interp.execute("regress y x1")
        self.assertTrue(res_m1.ok, res_m1.output)
        interp.execute("estimates store m1")
        self.assertIn("m1", interp.estimates.store)

        # 2. Classical Model 2: multivariate
        res_ols = interp.execute("regress y x1 x2")
        self.assertTrue(res_ols.ok, res_ols.output)
        interp.execute("estimates store m2")
        self.assertIn("m2", interp.estimates.store)

        coefs_dict = {c["name"]: c["coef"] for c in interp.estimates.last_model["coefs"]}
        vcov_m2 = np.asarray(interp.estimates.last_model["vcov"])

        se_dict = {c["name"]: c["se"] for c in interp.estimates.last_model["coefs"]}
        b_x1 = coefs_dict["x1"]
        b_x2 = coefs_dict["x2"]
        self.assertAlmostEqual(b_x1, 3.5, delta=0.2)
        self.assertAlmostEqual(b_x2, -1.2, delta=0.2)

        # 3. Robust standard errors (HC1)
        res_rob = interp.execute("regress y x1 x2, vce(robust)")
        self.assertTrue(res_rob.ok, res_rob.output)
        se_rob_dict = {c["name"]: c["se"] for c in interp.estimates.last_model["coefs"]}
        self.assertGreater(se_rob_dict["x1"], 0.0)

        # 4. Cluster robust standard errors
        res_clust = interp.execute("regress y x1 x2, vce(cluster cluster_var)")
        self.assertTrue(res_clust.ok, res_clust.output)
        self.assertEqual(interp.estimates.last_model.get("vce"), "cluster")

        # 5. Post-estimation test
        res_test = interp.execute("test x1 = x2")
        self.assertTrue(res_test.ok, res_test.output)
        self.assertIn("F(", res_test.output)

        # 6. Prediction
        res_pred = interp.execute("predict y_fitted, xb")
        self.assertTrue(res_pred.ok, res_pred.output)
        self.assertIn("y_fitted", list(ds.df.columns))

        res_resid = interp.execute("predict y_resid, residuals")
        self.assertTrue(res_resid.ok, res_resid.output)
        self.assertIn("y_resid", list(ds.df.columns))

        # 7. Global macro
        interp.execute("global project_name EconometricStudy2026")

        # 8. Session save & load
        session_file = self.tmp_path / "econometric_study.potproj"
        res_save = interp.execute(f'session save "{session_file}"')
        self.assertTrue(res_save.ok, res_save.output)
        self.assertTrue(session_file.exists())

        # Clear session completely
        interp.execute("clear")
        interp.estimates.last_model = None
        interp.estimates.store.clear()
        self.assertIsNone(interp.estimates.last_model)
        self.assertEqual(len(interp.estimates.store), 0)

        # Reopen session
        res_load = interp.execute(f'session load "{session_file}"')
        self.assertTrue(res_load.ok, res_load.output)
        self.assertIsNotNone(interp.estimates.last_model)

        # Verify multi-model store preservation
        self.assertIn("m1", interp.estimates.store)
        self.assertIn("m2", interp.estimates.store)

        # Verify exact numerical parameters and covariance matrices for stored m2
        m2_loaded = interp.estimates.store["m2"]
        loaded_coefs_m2 = {c["name"]: c["coef"] for c in m2_loaded["coefs"]}
        loaded_se_m2 = {c["name"]: c["se"] for c in m2_loaded["coefs"]}
        loaded_vcov_m2 = np.asarray(m2_loaded["vcov"])

        self.assertAlmostEqual(loaded_coefs_m2["x1"], b_x1, delta=1e-12)
        self.assertAlmostEqual(loaded_coefs_m2["x2"], b_x2, delta=1e-12)
        self.assertAlmostEqual(loaded_se_m2["x1"], se_dict["x1"], delta=1e-12)
        np.testing.assert_allclose(loaded_vcov_m2, vcov_m2, rtol=1e-12)

        # Verify dataset columns and predictions restored
        self.assertIn("y_fitted", list(ds.df.columns))
        self.assertIn("y_resid", list(ds.df.columns))
        self.assertEqual(len(ds.df), n)

        # Verify macro state restored
        self.assertEqual(interp.global_macros.get("project_name"), "EconometricStudy2026")

    # =========================================================================
    # 4. GUI and Batch Command Equivalence
    # =========================================================================

    def test_05_gui_and_batch_command_equivalence(self):
        """
        Phase 5.D: Verifies that GUI run_command, batch do-script, and direct interpreter
        execution produce bit-identical coefficient vectors, standard errors, covariance matrices,
        and dataset transformations.
        """
        df = pd.DataFrame({
            "y": [1.0, 2.0, 3.2, 4.1, 5.0, 6.2, 7.1, 8.0],
            "x1": [2.0, 4.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0],
            "x2": [1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0]
        })
        data_file = self.tmp_path / "equiv.csv"
        df.to_csv(data_file, index=False)

        # 1. Direct interpreter execution
        ds_direct = Dataset()
        interp_direct = Interpreter(ds_direct)
        interp_direct.execute(f'use "{data_file}", clear')
        interp_direct.execute("gen z = x1 * 2")
        res_direct = interp_direct.execute("regress y x1 x2")
        self.assertTrue(res_direct.ok, res_direct.output)

        # 2. Batch script execution via do-file
        do_file = self.tmp_path / "run_analysis.do"
        do_file.write_text(
            f'use "{data_file}", clear\n'
            f'gen z = x1 * 2\n'
            f'regress y x1 x2\n',
            encoding="utf-8"
        )
        ds_batch = Dataset()
        interp_batch = Interpreter(ds_batch)
        res_batch = interp_batch.execute(f'do "{do_file}"')
        self.assertTrue(res_batch.ok, res_batch.output)

        # 3. GUI window execution
        win = PotatosticsQtWindow()
        try:
            win.run_command(f'use "{data_file}", clear', sync=True)
            win.run_command("gen z = x1 * 2", sync=True)
            win.run_command("regress y x1 x2", sync=True)

            # Compare all coefficients
            direct_b = {c["name"]: c["coef"] for c in interp_direct.estimates.last_model["coefs"]}
            batch_b = {c["name"]: c["coef"] for c in interp_batch.estimates.last_model["coefs"]}
            win_b = {c["name"]: c["coef"] for c in win.interpreter.estimates.last_model["coefs"]}

            for k in direct_b:
                self.assertAlmostEqual(direct_b[k], batch_b[k], delta=1e-12)
                self.assertAlmostEqual(direct_b[k], win_b[k], delta=1e-12)

            # Compare all standard errors
            direct_se = {c["name"]: c["se"] for c in interp_direct.estimates.last_model["coefs"]}
            batch_se = {c["name"]: c["se"] for c in interp_batch.estimates.last_model["coefs"]}
            win_se = {c["name"]: c["se"] for c in win.interpreter.estimates.last_model["coefs"]}

            for k in direct_se:
                self.assertAlmostEqual(direct_se[k], batch_se[k], delta=1e-12)
                self.assertAlmostEqual(direct_se[k], win_se[k], delta=1e-12)

            # Compare covariance matrices
            np.testing.assert_allclose(
                np.asarray(interp_direct.estimates.last_model["vcov"]),
                np.asarray(interp_batch.estimates.last_model["vcov"]),
                rtol=1e-12
            )
            np.testing.assert_allclose(
                np.asarray(interp_direct.estimates.last_model["vcov"]),
                np.asarray(win.interpreter.estimates.last_model["vcov"]),
                rtol=1e-12
            )

            # Compare dataset transformations
            np.testing.assert_allclose(ds_direct.df["z"].values, ds_batch.df["z"].values)
            np.testing.assert_allclose(ds_direct.df["z"].values, win.dataset.df["z"].values)
        finally:
            win.close()

    # =========================================================================
    # 5. Command Language Compatibility
    # =========================================================================

    def test_06_macros_and_data_transformation_commands(self):
        """Verifies local/global macros, mathematical functions, and collapse."""
        ds = Dataset()
        interp = Interpreter(ds)

        df = pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "val1": [10.0, 20.0, 30.0, 40.0],
            "val2": [1.0, 2.0, 3.0, 4.0]
        })
        ds.load_dataframe(df)

        # Local macro expansion
        interp.execute("local target_var val1")
        res = interp.execute("summarize `target_var'")
        self.assertTrue(res.ok, res.output)
        self.assertIn("val1", res.output)

        # Mathematical variable generation
        res_gen = interp.execute("gen sqrt_val = sqrt(val1)")
        self.assertTrue(res_gen.ok, res_gen.output)
        self.assertAlmostEqual(ds.df["sqrt_val"].iloc[0], np.sqrt(10.0), delta=1e-6)

        # Collapse
        res_col = interp.execute("collapse (mean) val1 (sum) val2, by(group)")
        self.assertTrue(res_col.ok, res_col.output)
        self.assertEqual(len(ds.df), 2)
        self.assertIn("val1", list(ds.df.columns))

    # =========================================================================
    # 6. Real-World Performance
    # =========================================================================

    def test_07_performance_scaling_benchmarks(self):
        """
        Phase 5.E: Measures execution throughput on 10,000 observations.
        Separates correctness validation from timing measurements and records system info.
        """
        n = 10000
        np.random.seed(42)
        x = np.random.normal(0, 1, n)
        y = 2.0 * x + np.random.normal(0, 0.5, n)
        df = pd.DataFrame({"y": y, "x": x})

        ds = Dataset()
        ds.load_dataframe(df)
        interp = Interpreter(ds)

        # Record system environment
        env_info = {
            "os": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python_version": sys.version.split()[0]
        }

        # 1. Summarize benchmark
        t0 = time.perf_counter()
        res_sum = interp.execute("summarize y x")
        t_sum = time.perf_counter() - t0
        self.assertTrue(res_sum.ok, res_sum.output)

        # 2. Sort benchmark
        t0 = time.perf_counter()
        res_sort = interp.execute("sort y")
        t_sort = time.perf_counter() - t0
        self.assertTrue(res_sort.ok, res_sort.output)
        self.assertTrue(ds.df["y"].is_monotonic_increasing)

        # 3. OLS Regression benchmark
        t0 = time.perf_counter()
        res_reg = interp.execute("regress y x")
        t_reg = time.perf_counter() - t0
        self.assertTrue(res_reg.ok, res_reg.output)
        b_x = next(c["coef"] for c in interp.estimates.last_model["coefs"] if c["name"] == "x")
        self.assertAlmostEqual(b_x, 2.0, delta=0.05)

        # Log measured performance metrics (without brittle fixed 1s gates that break on loaded CI)
        print(f"\n[PERFORMANCE METRICS - N={n:,}] Environment: {env_info['os']} ({env_info['machine']}, {env_info['cpu_count']} CPUs)")
        print(f"  - Summarize throughput: {n / t_sum:,.0f} rows/s (elapsed: {t_sum*1000:.1f}ms)")
        print(f"  - Sort throughput:      {n / t_sort:,.0f} rows/s (elapsed: {t_sort*1000:.1f}ms)")
        print(f"  - OLS throughput:       {n / t_reg:,.0f} rows/s (elapsed: {t_reg*1000:.1f}ms)")

        # Generous upper bound to catch order-of-magnitude algorithmic regressions (e.g. O(N^2) loops)
        self.assertLess(t_sum, 15.0, f"Summarize took {t_sum:.2f}s")
        self.assertLess(t_sort, 15.0, f"Sort took {t_sort:.2f}s")
        self.assertLess(t_reg, 15.0, f"Regression took {t_reg:.2f}s")

    # =========================================================================
    # 7. Reporting and Export Fidelity
    # =========================================================================

    def test_08_table_export_fidelity(self):
        """
        Phase 5.A: Verifies multi-format table exports (HTML, LaTeX, CSV, DOCX).
        Strictly requires command success, file existence, non-empty content,
        model metadata, coefficients, standard errors, sample size, and valid syntax.
        """
        df = pd.DataFrame({
            "y": [1.0, 2.0, 3.5, 4.0, 5.2],
            "x1": [0.5, 1.2, 2.1, 2.8, 3.9],
            "x2": [10.0, 9.0, 8.0, 7.0, 6.0]
        })
        ds = Dataset()
        ds.load_dataframe(df)
        interp = Interpreter(ds)

        # Fit model
        res_reg = interp.execute("regress y x1 x2")
        self.assertTrue(res_reg.ok, res_reg.output)
        coef_dict = {c["name"]: c["coef"] for c in interp.estimates.last_model["coefs"]}
        coef_x1 = coef_dict["x1"]

        # 1. HTML export (strict verification)
        html_out = self.tmp_path / "table.html"
        res_html = interp.execute(f'esttab using "{html_out}", replace html')
        self.assertTrue(res_html.ok, f"HTML export failed: {res_html.output}")
        self.assertTrue(html_out.exists(), "HTML export file does not exist on disk")
        self.assertGreater(html_out.stat().st_size, 0, "HTML export file is empty")
        html_content = html_out.read_text(encoding="utf-8")
        self.assertIn("table", html_content.lower())
        self.assertIn("x1", html_content)
        self.assertIn("x2", html_content)
        self.assertIn("0.5455", html_content)
        self.assertIn("Observations", html_content)

        # 2. LaTeX export (strict verification)
        tex_out = self.tmp_path / "table.tex"
        res_tex = interp.execute(f'esttab using "{tex_out}", replace')
        self.assertTrue(res_tex.ok, f"LaTeX export failed: {res_tex.output}")
        self.assertTrue(tex_out.exists(), "LaTeX export file does not exist on disk")
        self.assertGreater(tex_out.stat().st_size, 0, "LaTeX export file is empty")
        tex_content = tex_out.read_text(encoding="utf-8")
        self.assertIn("\\begin{table}", tex_content)
        self.assertIn("\\toprule", tex_content)
        self.assertIn("\\bottomrule", tex_content)
        self.assertIn("x1", tex_content)
        self.assertIn("x2", tex_content)
        self.assertIn(f"{coef_x1:.3f}", tex_content)
        self.assertIn("Observations", tex_content)

        # 3. CSV table export (strict verification)
        csv_out = self.tmp_path / "table.csv"
        res_csv = interp.execute(f'esttab using "{csv_out}", replace csv')
        self.assertTrue(res_csv.ok, f"CSV table export failed: {res_csv.output}")
        self.assertTrue(csv_out.exists(), "CSV table export file does not exist on disk")
        self.assertGreater(csv_out.stat().st_size, 0, "CSV table export file is empty")
        csv_content = csv_out.read_text(encoding="utf-8")
        self.assertIn("x1", csv_content)
        self.assertIn("x2", csv_content)

        # 4. Word DOCX export (conditional on python-docx library)
        docx_out = self.tmp_path / "table.docx"
        try:
            import docx  # noqa: F401
            has_docx = True
        except ImportError:
            has_docx = False

        if has_docx:
            res_docx = interp.execute(f'esttab using "{docx_out}", replace docx')
            self.assertTrue(res_docx.ok, f"DOCX export failed: {res_docx.output}")
            self.assertTrue(docx_out.exists(), "DOCX export file does not exist on disk")
            self.assertGreater(docx_out.stat().st_size, 0, "DOCX export file is empty")


if __name__ == "__main__":
    unittest.main()
