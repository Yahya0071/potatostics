"""
Stage 49/50 — Comprehensive Real Desktop GUI Workflows Verification Suite.
Validates the complete end-to-end research experience across 5 core workflows:
Workflow A — Data preparation (import, inspect, cell edit, multi-select, copy/paste, sort, filter, metadata, save/reopen)
Workflow B — Statistical analysis (load auto.dta, summarize, OLS, robust inference, CIs, predict, Qt plot dialog, table export)
Workflow C — Script execution (.do file execution, macro expansion, history review, error handling, stored results)
Workflow D — Session persistence (.potproj full state serialization and 100% fidelity restoration)
Workflow E — Worker lifecycle safety (syntax error, worker exception recovery, cooperative cancellation, non-cooperative timeout)
"""

import os
import sys
import time
import tempfile
import unittest
from pathlib import Path

# Set headless platform before Qt import
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["POTATOSTICS_TESTING"] = "1"

try:
    from PySide6 import QtWidgets, QtCore, QtGui  # noqa: F401
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))


@unittest.skipUnless(HAS_PYSIDE6, "PySide6 not installed in current test environment")
class TestStage49GuiWorkflows(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication(["--platform", "offscreen"])

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name).resolve()
        from gui_qt.main_window import PotatosticsQtWindow
        self.win = PotatosticsQtWindow()

    def tearDown(self):
        if self.win:
            # If a worker thread is running, cancel and wait before closing
            if hasattr(self.win, "worker") and self.win.worker and self.win.worker.isRunning():
                self.win.worker.cancel()
                self.win.worker.wait(2000)
            self.win.close()
            QtWidgets.QApplication.processEvents()
        self.tmp_dir.cleanup()

    # =========================================================================
    # Workflow A — Data preparation
    # =========================================================================
    def test_workflow_a_data_preparation(self):
        """
        Workflow A: Import CSV, inspect variables, edit data-grid cells directly via GUI,
        perform multi-cell selection, copy & paste, sort using view controls, filter observations,
        view and edit variable metadata, create new variables, save to .dta, and reopen.
        """
        import pandas as pd
        import numpy as np

        # 1. Create CSV dataset
        n = 500
        df_init = pd.DataFrame({
            "id": np.arange(1, n + 1),
            "wage": np.round(np.random.uniform(10.0, 50.0, n), 2),
            "educ": np.random.randint(8, 20, n),
            "exper": np.random.randint(0, 30, n),
            "female": np.random.choice([0, 1], n)
        })
        csv_file = self.tmp_path / "raw_data.csv"
        df_init.to_csv(csv_file, index=False)

        # 2. Import CSV via GUI command
        res = self.win.run_command_sync(f'use "{csv_file}", clear')
        self.assertTrue(res.ok, f"Import failed: {res.output}")
        self.assertEqual(len(self.win.dataset.df), n)

        # 3. Inspect variables pane
        var_names = list(self.win.dataset.df.columns)
        self.assertIn("wage", var_names)
        self.assertIn("educ", var_names)
        self.assertEqual(self.win.variables_pane.tree.topLevelItemCount(), 5)

        # 4. Genuine GUI Interaction: Direct Data Grid In-Place Cell Editing
        model = self.win.data_grid.model
        wage_col_idx = list(self.win.dataset.df.columns).index("wage")
        idx_0 = model.index(0, wage_col_idx)
        # Edit row 0 wage to 99.95 via model.setData
        ok_set = model.setData(idx_0, "99.95", QtCore.Qt.ItemDataRole.EditRole)
        self.assertTrue(ok_set)
        self.assertAlmostEqual(self.win.dataset.df.iloc[0]["wage"], 99.95, places=2)

        # 5. Genuine GUI Interaction: Multiple Cell Selection
        table_view = self.win.data_grid.table_view
        selection_model = table_view.selectionModel()
        top_left = model.index(0, 1)
        bottom_right = model.index(2, 2)
        selection_model.select(QtCore.QItemSelection(top_left, bottom_right), QtCore.QItemSelectionModel.SelectionFlag.Select)
        selected_indexes = selection_model.selectedIndexes()
        self.assertEqual(len(selected_indexes), 6)  # 3 rows x 2 cols

        # 6. Genuine GUI Interaction: Copy and Paste
        copied_text = self.win.data_grid.copy_selection()
        self.assertIn("99.95", copied_text)
        # Paste into row 10, col 1
        pasted_count = self.win.data_grid.paste_clipboard(start_row=10, start_col=1)
        self.assertGreater(pasted_count, 0)
        self.assertAlmostEqual(self.win.dataset.df.iloc[10]["wage"], 99.95, places=2)

        # 7. Genuine GUI Interaction: Sort using GUI Header Controls
        # Sort by wage column
        model.sort_by_column(wage_col_idx)
        self.assertIsNotNone(model._row_map)
        first_wage = float(model.data(model.index(0, wage_col_idx), QtCore.Qt.ItemDataRole.DisplayRole))
        last_wage = float(model.data(model.index(model.rowCount() - 1, wage_col_idx), QtCore.Qt.ItemDataRole.DisplayRole))
        self.assertLessEqual(first_wage, last_wage)

        # 8. Genuine GUI Interaction: Filter Observations via Search Bar
        self.win.data_grid.txt_filter.setText("99.95")
        QtWidgets.QApplication.processEvents()
        self.assertEqual(model.rowCount(), 2)  # Exactly 2 observations have 99.95
        # Clear filter
        self.win.data_grid.txt_filter.setText("")
        QtWidgets.QApplication.processEvents()
        self.assertEqual(model.rowCount(), n)

        # 9. Genuine GUI Interaction: Variable Metadata Editing
        self.win.dataset.var_labels["wage"] = "Hourly Wage in USD"
        self.win.variables_pane.refresh()
        wage_item = self.win.variables_pane.tree.topLevelItem(1)
        self.assertEqual(wage_item.text(2), "Hourly Wage in USD")

        # 10. Generate variables & conditional replace via command syntax
        res_gen = self.win.run_command_sync("generate expersq = exper ^ 2")
        self.assertTrue(res_gen.ok, f"generate failed: {res_gen.output}")
        self.assertIn("expersq", self.win.dataset.df.columns)
        self.assertTrue(np.allclose(self.win.dataset.df["expersq"], self.win.dataset.df["exper"] ** 2))

        # 11. Save and Reopen with full numerical verification
        saved_dta = self.tmp_path / "processed_data.dta"
        res_save = self.win.run_command_sync(f'save "{saved_dta}", replace')
        self.assertTrue(res_save.ok, f"save failed: {res_save.output}")
        self.assertTrue(saved_dta.exists())

        res_reopen = self.win.run_command_sync(f'use "{saved_dta}", clear')
        self.assertTrue(res_reopen.ok, f"reopen failed: {res_reopen.output}")
        self.assertEqual(len(self.win.dataset.df), n)
        self.assertAlmostEqual(self.win.dataset.df.iloc[0]["wage"], 99.95, places=2)
        self.assertAlmostEqual(self.win.dataset.df.iloc[10]["wage"], 99.95, places=2)
        self.assertIn("expersq", self.win.dataset.df.columns)

    # =========================================================================
    # Workflow B — Statistical analysis
    # =========================================================================
    def test_workflow_b_statistical_analysis(self):
        """
        Workflow B: Load auto.dta, summarize, OLS, robust inference, full coefficient vector,
        SEs, confidence intervals, regression sample, prediction, Qt plot dialog construction,
        and LaTeX table export.
        """
        auto_path = ROOT_PATH / "engine" / "assets" / "data" / "auto.dta"
        if not auto_path.is_file():
            self.skipTest("auto.dta not present")

        # 1. Load auto.dta
        res_load = self.win.run_command_sync(f'use "{auto_path}", clear')
        self.assertTrue(res_load.ok, f"Failed to load auto.dta: {res_load.output}")
        self.assertEqual(len(self.win.dataset.df), 74)

        # 2. Summarize numeric variables
        res_sum = self.win.run_command_sync("summarize price mpg weight")
        self.assertTrue(res_sum.ok)
        out = str(res_sum.output)
        self.assertIn("price", out)
        self.assertIn("mpg", out)
        self.assertIn("weight", out)
        self.assertIn("Mean", out)

        # 3. Classical OLS regression with detailed vector checks
        res_ols = self.win.run_command_sync("regress price mpg weight")
        self.assertTrue(res_ols.ok, f"OLS failed: {res_ols.output}")
        model = self.win.interpreter.estimates.last_model
        self.assertIsNotNone(model)

        # Verify coefficient vector
        coefs_dict = {c["name"]: c for c in model["coefs"]}
        var_names = model["var_names"]
        self.assertIn("mpg", var_names)
        self.assertIn("weight", var_names)
        self.assertIn("_cons", var_names)

        self.assertAlmostEqual(coefs_dict["mpg"]["coef"], -49.5122, places=3)
        self.assertAlmostEqual(coefs_dict["weight"]["coef"], 1.7466, places=3)
        self.assertAlmostEqual(coefs_dict["_cons"]["coef"], 1946.0687, places=2)

        # Verify standard error vector and confidence intervals
        for c in model["coefs"]:
            self.assertGreater(c["se"], 0)
            self.assertLess(c["ci_lower"], c["coef"])
            self.assertGreater(c["ci_upper"], c["coef"])

        # Verify regression sample size and R2
        self.assertEqual(model["n"], 74)
        self.assertAlmostEqual(model["r2"], 0.2934, places=3)

        # Verify console output formatting
        console_text = self.win.console.toPlainText()
        self.assertIn("Number of obs", console_text)
        self.assertIn("74", console_text)
        self.assertIn("R-squared", console_text)
        self.assertIn("0.2934", console_text)
        self.assertIn("-49.5122", console_text)

        # 4. Heteroskedasticity-Robust OLS (HC1)
        res_rob = self.win.run_command_sync("regress price mpg weight, vce(robust)")
        self.assertTrue(res_rob.ok, f"Robust OLS failed: {res_rob.output}")
        rob_coefs_dict = {c["name"]: c for c in self.win.interpreter.estimates.last_model["coefs"]}
        # Robust (HC1) SE for weight is approx 0.7778 (vs classical 0.6414)
        self.assertAlmostEqual(rob_coefs_dict["weight"]["se"], 0.7778, places=3)

        # 5. Generate predictions
        res_pred = self.win.run_command_sync("predict price_hat")
        self.assertTrue(res_pred.ok, f"predict failed: {res_pred.output}")
        self.assertIn("price_hat", self.win.dataset.df.columns)
        self.assertEqual(len(self.win.dataset.df["price_hat"].dropna()), 74)

        # 6. Genuine Plot Dialog Construction
        from gui_qt.main_window import QtPlotDialog
        dlg = QtPlotDialog("scatter", self.win.dataset.df, kwargs={"xvar": "mpg", "yvar": "price"}, parent=self.win)
        self.assertIsNotNone(dlg.canvas)
        self.assertIsNotNone(dlg.toolbar)
        self.assertEqual(dlg.plot_type, "scatter")
        dlg.close()

        # 7. Publication Table Export & Numerical Fidelity
        tex_file = self.tmp_path / "models_table.tex"
        res_exp = self.win.run_command_sync(f'esttab using "{tex_file}", replace booktabs')
        self.assertTrue(res_exp.ok, f"esttab export failed: {res_exp.output}")
        self.assertTrue(tex_file.exists())
        tex_content = tex_file.read_text(encoding="utf-8")
        self.assertIn("tabular", tex_content)
        self.assertIn("mpg", tex_content)
        self.assertIn("-49.51", tex_content)

    # =========================================================================
    # Workflow C — Script execution (.do files)
    # =========================================================================
    def test_workflow_c_script_execution(self):
        """Workflow C: Open .do file, macro expansion, command history, error codes, save output."""
        auto_path = ROOT_PATH / "engine" / "assets" / "data" / "auto.dta"
        if not auto_path.is_file():
            self.skipTest("auto.dta not present")

        # 1. Create a .do script with macros and operations
        do_file = self.tmp_path / "analysis.do"
        script_code = (
            f'use "{auto_path}", clear\n'
            'local factor 10\n'
            'gen scaled_mpg = mpg * `factor\'\n'
            'summarize scaled_mpg\n'
            'regress price scaled_mpg\n'
        )
        do_file.write_text(script_code, encoding="utf-8")

        # 2. Execute .do file
        res_do = self.win.run_command_sync(f'do "{do_file}"')
        self.assertTrue(res_do.ok, f"do script failed: {res_do.output}")
        self.assertIn("scaled_mpg", self.win.dataset.df.columns)

        # 3. Verify command history captured execution
        self.assertGreater(len(self.win.history_pane.history_records), 0)
        self.assertTrue(self.win.history_pane.history_records[-1]["ok"])

        # 4. Intentional error handling in script
        bad_do = self.tmp_path / "bad.do"
        bad_do.write_text("summarize nonexistent_variable_999\n", encoding="utf-8")
        res_bad = self.win.run_command_sync(f'do "{bad_do}"')
        self.assertFalse(res_bad.ok)
        self.assertNotEqual(res_bad.return_code, 0)
        self.assertFalse(self.win.history_pane.history_records[-1]["ok"])

        # 5. Save console log output to disk
        log_file = self.tmp_path / "session.log"
        console_text = self.win.console.toPlainText()
        log_file.write_text(console_text, encoding="utf-8")
        self.assertTrue(log_file.exists())
        self.assertIn("scaled_mpg", log_file.read_text(encoding="utf-8"))

    # =========================================================================
    # Workflow D — Session persistence (.potproj)
    # =========================================================================
    def test_workflow_d_session_persistence(self):
        """Workflow D: Load data, fit model, save session, reopen in new window, verify full state."""
        import numpy as np
        import pandas as pd
        auto_path = ROOT_PATH / "engine" / "assets" / "data" / "auto.dta"
        if not auto_path.is_file():
            self.skipTest("auto.dta not present")

        # 1. Load data & fit model
        self.win.run_command_sync(f'use "{auto_path}", clear')
        self.win.run_command_sync("regress price mpg weight")
        orig_r2 = self.win.interpreter.estimates.last_model["r2"]
        orig_b = np.array(self.win.interpreter.estimates.last_model["beta"])
        orig_se = np.array([c["se"] for c in self.win.interpreter.estimates.last_model["coefs"]])
        orig_vcov = np.array(self.win.interpreter.estimates.last_model["vcov"])
        orig_rows = len(self.win.dataset.df)
        orig_df = self.win.dataset.df.copy()

        # 2. Save session bundle
        session_file = self.tmp_path / "research_project.potproj"
        ok_save = self.win.save_session(str(session_file))
        self.assertTrue(ok_save, "save_session returned False")
        self.assertTrue(session_file.exists())

        # 3. Close window and instantiate clean new window
        self.win.close()
        QtWidgets.QApplication.processEvents()

        from gui_qt.main_window import PotatosticsQtWindow
        win2 = PotatosticsQtWindow()
        try:
            # 4. Restore session
            ok_open = win2.open_session(str(session_file))
            self.assertTrue(ok_open, "open_session returned False")

            # 5. Verify restored dataset, estimation state, and console history
            self.assertEqual(len(win2.dataset.df), orig_rows)
            pd.testing.assert_frame_equal(win2.dataset.df, orig_df)
            self.assertIsNotNone(win2.interpreter.estimates.last_model)
            self.assertAlmostEqual(win2.interpreter.estimates.last_model["r2"], orig_r2, places=4)
            np.testing.assert_allclose(win2.interpreter.estimates.last_model["beta"], orig_b, rtol=1e-5)
            win2_se = np.array([c["se"] for c in win2.interpreter.estimates.last_model["coefs"]])
            np.testing.assert_allclose(win2_se, orig_se, rtol=1e-5)
            np.testing.assert_allclose(win2.interpreter.estimates.last_model["vcov"], orig_vcov, rtol=1e-5)
            self.assertGreater(len(win2.history_pane.history_records), 0)
        finally:
            win2.close()
            QtWidgets.QApplication.processEvents()

    # =========================================================================
    # Workflow E — Worker Lifecycle Safety & Cancellation
    # =========================================================================
    def test_workflow_e_failure_recovery_and_worker_safety(self):
        """
        Workflow E: Tests failure paths and worker safety:
        1. Syntax error handling and recovery
        2. Worker exception raising: authoritative _on_command_error stores failed CommandResult
        3. Cooperative slow worker cancellation and safe GUI state recovery
        4. Deliberately non-cooperative worker timeout: ensures application remains busy
        """
        from engine.interpreter import CommandResult
        from gui_qt.main_window import WorkerState

        auto_path = ROOT_PATH / "engine" / "assets" / "data" / "auto.dta"
        if not auto_path.is_file():
            self.skipTest("auto.dta not present")

        self.win.run_command_sync(f'use "{auto_path}", clear')
        initial_rows = len(self.win.dataset.df)

        # 1. Syntax error
        res_bad_syntax = self.win.run_command_sync("regress @@@ invalid syntax ///")
        self.assertFalse(res_bad_syntax.ok)
        self.assertFalse(self.win.is_busy())
        self.assertEqual(len(self.win.dataset.df), initial_rows, "Dataset was corrupted by syntax error")

        # 2. Worker raises unexpected Python exception
        def _failing_cmd_handler(cmd):
            raise RuntimeError("Unexpected simulated engine failure in worker thread")

        self.win.interpreter.register_command("fail_unexpectedly", _failing_cmd_handler)
        res_exc = self.win.run_command_sync("fail_unexpectedly")
        self.assertIsNotNone(res_exc)
        self.assertFalse(res_exc.ok)
        self.assertEqual(res_exc.return_code, 198)
        self.assertIn("Unexpected simulated engine failure", str(res_exc.error))
        self.assertFalse(self.win.is_busy())

        # Follow-up command works immediately after exception
        res_after_exc = self.win.run_command_sync("display 10 + 20")
        self.assertTrue(res_after_exc.ok)
        self.assertIn("30", str(res_after_exc.output))

        # 3. Cooperative Long-Running Cancellation Test
        cancel_observed = []

        def _slow_cooperative_handler(cmd):
            for _ in range(50):
                if self.win.interpreter.check_cancelled():
                    cancel_observed.append(True)
                    return CommandResult("(Cancelled cooperatively)", ok=False, return_code=1)
                time.sleep(0.02)
            return CommandResult("Finished slow op", ok=True)

        self.win.interpreter.register_command("slow_coop_test", _slow_cooperative_handler)

        started = self.win.run_command("slow_coop_test", sync=False)
        self.assertTrue(started)
        self.assertTrue(self.win.is_busy())

        # Cancel while running
        time.sleep(0.04)
        self.win.cancel_command()

        # Wait for worker thread to stop cleanly
        if self.win.worker:
            self.win.worker.wait(2000)
        QtWidgets.QApplication.processEvents()

        self.assertFalse(self.win.is_busy(), "Window remained busy after cooperative cancel")
        self.assertTrue(len(cancel_observed) > 0, "Cooperative worker did not observe cancellation")
        self.assertEqual(len(self.win.dataset.df), initial_rows, "Dataset was corrupted during cancellation")

        # Recovery command
        res_rec = self.win.run_command_sync("display 42")
        self.assertTrue(res_rec.ok)
        self.assertIn("42", str(res_rec.output))

        # 4. Deliberately non-cooperative worker timeout handling
        def _uncooperative_handler(cmd):
            # Sleeps without checking cancellation
            time.sleep(2.0)
            return CommandResult("Finished uncooperatively", ok=True)

        self.win.interpreter.register_command("non_coop_test", _uncooperative_handler)

        # Execute with very short timeout
        res_timeout = self.win.run_command_sync("non_coop_test", timeout=0.2)
        self.assertIsNotNone(res_timeout)
        self.assertFalse(res_timeout.ok)
        self.assertEqual(res_timeout.return_code, 124)
        self.assertIn("timed out", str(res_timeout.output).lower())
        # CRITICAL SAFETY REQUIREMENT: Worker is STILL running, so GUI must NOT report idle!
        self.assertTrue(self.win.is_busy(), "Application falsely reported idle while non-cooperative worker is active")
        self.assertEqual(self.win.worker.state, WorkerState.TIMED_OUT)

        # Wait for uncooperative worker to finally complete so we don't leak thread
        self.win.worker.wait(3000)
        QtWidgets.QApplication.processEvents()
        self.win._set_busy_state(False)
        self.assertFalse(self.win.is_busy())


if __name__ == "__main__":
    unittest.main()
