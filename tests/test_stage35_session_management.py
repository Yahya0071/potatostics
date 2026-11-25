"""
tests/test_stage35_session_management.py - Point-and-Click Session Persistence Suite
Verifies saving and restoring complete Potatostics analytical sessions (.potproj)
including dataset, variable labels, command review history, stored estimates,
macros, and console transcript output.
"""

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
import pandas as pd
from PySide6 import QtWidgets

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.session_manager import save_session_bundle, load_session_bundle
from gui_qt.main_window import PotatosticsQtWindow

app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(["--platform", "offscreen"])


class TestStage35SessionManagement(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.session_path = os.path.join(self.temp_dir, "analysis_session.potproj")

        # Set up a dataset with variables, labels, and estimates
        self.ds = Dataset()
        df = pd.DataFrame({
            "price": [5000.0, 6500.0, 7200.0, 8900.0, 10500.0],
            "mpg": [22.0, 20.0, 18.0, 16.0, 14.0],
            "foreign": [0, 1, 0, 1, 0]
        })
        self.ds.load_dataframe(df)
        self.ds.var_labels = {"price": "Car Price in USD", "mpg": "Miles per Gallon", "foreign": "Domestic/Foreign"}
        self.ds.val_labels = {"foreign": {0: "Domestic", 1: "Foreign"}}
        self.ds.dataset_label = "1978 Automobile Clean Sample"

        self.interp = Interpreter(self.ds)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_save_and_load_session_bundle_engine(self):
        """Verify that save_session_bundle creates a bundle and load_session_bundle restores all state."""
        # 1. Run commands and store estimates
        self.interp.execute("regress price mpg")
        self.interp.execute("estimates store m1")
        self.interp.execute("global author = \"Prof. Jane Doe\"")

        console_transcript = "Potatostics Console Log...\n. regress price mpg\nModel estimated successfully."
        history_records = [
            {"command": "summarize price", "status": "ok", "time": "12:00:00"},
            {"command": "regress price mpg", "status": "ok", "time": "12:01:00"},
            {"command": "estimates store m1", "status": "ok", "time": "12:02:00"}
        ]

        # 2. Save session bundle
        save_session_bundle(
            self.session_path,
            interp=self.interp,
            console_text=console_transcript,
            history_records=history_records
        )

        self.assertTrue(os.path.exists(self.session_path))
        self.assertGreater(os.path.getsize(self.session_path), 100)

        # 3. Create a clean interpreter with empty dataset
        clean_ds = Dataset()
        clean_interp = Interpreter(clean_ds)
        self.assertTrue(clean_ds.is_empty)
        self.assertEqual(len(clean_interp.estimates.store), 0)

        # 4. Load session bundle
        loaded_data = load_session_bundle(self.session_path, clean_interp)

        # Verify dataset restoration
        self.assertFalse(clean_ds.is_empty)
        self.assertEqual(len(clean_ds.df), 5)
        self.assertEqual(list(clean_ds.df.columns), ["price", "mpg", "foreign"])
        self.assertEqual(clean_ds.var_labels.get("price"), "Car Price in USD")
        self.assertEqual(clean_ds.val_labels.get("foreign"), {0: "Domestic", 1: "Foreign"})
        self.assertEqual(clean_ds.dataset_label, "1978 Automobile Clean Sample")

        # Verify stored model estimates restoration
        self.assertIn("m1", clean_interp.estimates.store)
        self.assertIn("mpg", clean_interp.estimates.store["m1"]["var_names"])

        # Verify macro restoration
        self.assertEqual(clean_interp.global_macros.get("author"), "Prof. Jane Doe")

        # Verify console and history extraction
        self.assertEqual(loaded_data["console_text"], console_transcript)
        self.assertEqual(len(loaded_data["history_records"]), 3)
        self.assertEqual(loaded_data["history_records"][0]["command"], "summarize price")

    def test_02_interpreter_session_command_syntax(self):
        """Verify CLI 'session save' and 'session load' commands."""
        self.interp.execute("regress price mpg")
        self.interp.execute("estimates store base_model")

        # CLI Session Save
        res_save = self.interp.execute(f'session save "{self.session_path}"')
        self.assertTrue(res_save.ok)
        self.assertTrue(os.path.exists(self.session_path))
        self.assertIn("saved successfully", res_save.output)

        # CLI Session Load
        new_interp = Interpreter(Dataset())
        res_load = new_interp.execute(f'session load "{self.session_path}"')
        self.assertTrue(res_load.ok)
        self.assertIn("loaded successfully", res_load.output)
        self.assertEqual(len(new_interp.dataset.df), 5)
        self.assertIn("base_model", new_interp.estimates.store)

    def test_03_gui_file_menu_actions_and_dialog_integration(self):
        """Verify GUI has point-and-click Save Session and Open Session actions in File menu."""
        win = PotatosticsQtWindow()
        try:
            # Check File menu actions
            file_actions = [a.text() for a in win.file_menu.actions() if a.text()]
            self.assertTrue(
                any("save session" in a.lower() for a in file_actions),
                f"Missing 'Save Session' in File menu: {file_actions}"
            )
            self.assertTrue(
                any("open session" in a.lower() for a in file_actions),
                f"Missing 'Open Session' in File menu: {file_actions}"
            )

            # Check GUI save and open methods
            win.dataset.load_dataframe(self.ds.df)
            win.dataset.var_labels = self.ds.var_labels
            win.run_command("regress price mpg", sync=True)

            # Save session via GUI method
            saved_ok = win.save_session(self.session_path)
            self.assertTrue(saved_ok)
            self.assertTrue(os.path.exists(self.session_path))

            # Open session into a fresh window
            win2 = PotatosticsQtWindow()
            try:
                self.assertTrue(win2.dataset.is_empty)
                loaded_ok = win2.open_session(self.session_path)
                self.assertTrue(loaded_ok)
                self.assertFalse(win2.dataset.is_empty)
                self.assertEqual(len(win2.dataset.df), 5)

                # Variables pane updated
                self.assertGreater(win2.variables_pane.tree.topLevelItemCount(), 0)

                # History pane updated
                self.assertGreater(len(win2.history_pane.history_records), 0)
            finally:
                win2.close()
        finally:
            win.close()

    def test_command_bar_and_do_file_preserve_gui_results(self):
        win = PotatosticsQtWindow(initial_file=False)
        try:
            win.dataset.load_dataframe(self.ds.df)
            self.assertTrue(win.run_command_sync("regress price mpg").ok)
            failed = win.run_command_sync("summarize absent")
            self.assertFalse(failed.ok)
            captured = win.run_command_sync("capture summarize absent")
            self.assertTrue(captured.ok)
            self.assertEqual(captured.return_code, failed.return_code)
            self.assertEqual(win.history_pane.history_records[-1]["rc"], captured.return_code)
            transcript = win.console.toPlainText()
            records = list(win.history_pane.history_records)
            ranges = win.console.session_ranges()
            script = Path(self.temp_dir) / "archive.do"
            script.write_text(f'session save "{self.session_path}"\n', encoding="utf-8")
            self.assertTrue(win.run_command_sync(f'do "{script}"').ok)
            saved = load_session_bundle(self.session_path, Interpreter(Dataset()))
            self.assertEqual(saved["console_text"], transcript)
            self.assertEqual(saved["history_records"], records)
            self.assertEqual(saved["manifest"]["console_ranges_v1"], ranges)
            self.assertTrue(win.run_command_sync("summarize foreign").ok)
            self.assertTrue(win.run_command_sync(f'session open "{self.session_path}"').ok)
            self.assertEqual(win.history_pane.history_records[:-1], records)
            self.assertTrue(win.console.toPlainText().startswith(transcript))
            self.assertNotIn("summarize foreign", win.console.toPlainText())
            self.assertEqual(win.console.session_ranges()[:len(ranges)], ranges)
        finally:
            win.close()

    def test_startup_project_restores_menu_saved_history_and_console(self):
        win = PotatosticsQtWindow(initial_file=False)
        try:
            win.dataset.load_dataframe(self.ds.df)
            self.assertTrue(win.run_command_sync("summarize mpg").ok)
            self.assertFalse(win.run_command_sync("summarize absent").ok)
            records = list(win.history_pane.history_records)
            transcript = win.console.toPlainText()
            self.assertTrue(win.save_session(self.session_path))
        finally:
            win.close()

        reopened = PotatosticsQtWindow(initial_file=self.session_path)
        try:
            deadline = time.monotonic() + 10
            while (reopened.last_command_result is None or reopened.is_busy()) and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)
            self.assertIsNotNone(reopened.last_command_result)
            self.assertTrue(reopened.last_command_result.ok)
            self.assertEqual(reopened.history_pane.history_records[:-1], records)
            self.assertTrue(reopened.console.toPlainText().startswith(transcript))
            self.assertEqual(list(reopened.dataset.df.columns), list(self.ds.df.columns))
        finally:
            reopened.close()


if __name__ == "__main__":
    unittest.main()
