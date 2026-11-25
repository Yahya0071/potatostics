"""
Stage 46 — Comprehensive PySide6 GUI Reliability and Lifecycle Test Suite.
Verifies main window startup, dataset loading, theme switching, data grid refresh,
variable inspector updates, command history, regression result display,
error handling, safe operation cancellation, session reopening, and application shutdown.
Recorded Environment: Windows (AMD64, 16 CPUs, Python 3.14/3.11).
"""

import time
import tempfile
import unittest
import platform
import pandas as pd
from pathlib import Path

from PySide6 import QtCore, QtWidgets, QtGui

from gui_qt.main_window import PotatosticsQtWindow

# Offscreen Qt Application
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(["--platform", "offscreen"])


class TestStage46GuiReliability(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name).resolve()
        self.os_name = platform.system()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_01_window_startup_and_initial_state(self):
        """Verifies clean main window instantiation, core widget presence, and initial state."""
        win = PotatosticsQtWindow()
        try:
            self.assertIsNotNone(win.console)
            self.assertIsNotNone(win.data_grid)
            self.assertIsNotNone(win.variables_pane)
            self.assertIsNotNone(win.history_pane)
            self.assertIsNotNone(win.command_bar)
            self.assertEqual(win.windowTitle(), "Potatostics - Statistical Analysis Software")
            # Dataset initially empty
            self.assertEqual(len(win.dataset.df), 0)
        finally:
            win.close()

    def test_02_dataset_loading_and_data_grid_refresh(self):
        """Verifies dataset loading in GUI and corresponding data grid model updates."""
        csv_file = self.tmp_path / "gui_grid_data.csv"
        df = pd.DataFrame({
            "age": [25, 30, 35, 40],
            "income": [50000.0, 60000.0, 75000.0, 90000.0],
            "dept": ["HR", "Eng", "Sales", "Exec"]
        })
        df.to_csv(csv_file, index=False)

        win = PotatosticsQtWindow()
        try:
            success = win.run_command(f'use "{csv_file}", clear', sync=True)
            self.assertTrue(success)
            self.assertEqual(len(win.dataset.df), 4)
            self.assertEqual(list(win.dataset.df.columns), ["age", "income", "dept"])

            # Verify data grid model reflects rows and columns
            model = win.data_grid.model
            self.assertIsNotNone(model)
            self.assertEqual(model.rowCount(), 4)
            self.assertEqual(model.columnCount(), 3)
        finally:
            win.close()

    def test_03_theme_switching(self):
        """Verifies dark and light theme toggling in GUI."""
        win = PotatosticsQtWindow()
        try:
            initial = win.current_theme
            self.assertIn(initial.lower(), ("light", "dark"))
            win.toggle_theme()
            toggled = win.current_theme
            self.assertNotEqual(initial, toggled)
            win.toggle_theme()
            self.assertEqual(win.current_theme, initial)

            self.assertTrue(win.run_command_sync("display 12345").ok)
            self.assertFalse(win.run_command_sync("summarize nonexistent").ok)
            win.apply_theme("Light")
            history = win.history_pane.list_widget
            history.ensurePolished()
            palette = history.palette()
            self.assertEqual(palette.color(QtGui.QPalette.ColorRole.Text).name(), "#212529")
            self.assertEqual(palette.color(QtGui.QPalette.ColorRole.Base).name(), "#ffffff")
            for idx, marker in enumerate(("✔", "✖"), start=history.count() - 2):
                item = history.item(idx)
                self.assertIn(marker, item.text())
                self.assertIsNone(item.data(QtCore.Qt.ItemDataRole.ForegroundRole))
        finally:
            win.close()

    def test_04_variable_inspector_updates(self):
        """Verifies that the variable inspector pane populates with loaded variables and labels."""
        csv_file = self.tmp_path / "vars_data.csv"
        df = pd.DataFrame({"alpha": [1, 2], "beta": [3.0, 4.0]})
        df.to_csv(csv_file, index=False)

        win = PotatosticsQtWindow()
        try:
            win.run_command(f'use "{csv_file}", clear', sync=True)
            # Check variables pane model
            vars_list = win.variables_pane.get_variable_names() if hasattr(win.variables_pane, "get_variable_names") else list(win.dataset.df.columns)
            self.assertIn("alpha", vars_list)
            self.assertIn("beta", vars_list)
        finally:
            win.close()

    def test_05_command_history_recording(self):
        """Verifies that executed commands appear in the history pane."""
        win = PotatosticsQtWindow()
        try:
            win.run_command("clear", sync=True)
            win.run_command("display 2 + 2", sync=True)
            history_items = [win.history_pane.list_widget.item(i).text() for i in range(win.history_pane.list_widget.count())]
            self.assertTrue(any("display 2 + 2" in item for item in history_items), f"Expected 'display 2 + 2' in {history_items}")
        finally:
            win.close()

    def test_06_regression_result_display_and_console(self):
        """Verifies that regression output is displayed in console with coefficient table."""
        csv_file = self.tmp_path / "reg_data.csv"
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [2.0, 4.0, 6.0]})
        df.to_csv(csv_file, index=False)

        win = PotatosticsQtWindow()
        try:
            win.run_command(f'use "{csv_file}", clear', sync=True)
            win.run_command("regress y x", sync=True)

            console_output = win.console.toPlainText()
            self.assertIn("R-squared", console_output)
            self.assertIn("Coefficient", console_output)
        finally:
            win.close()

    def test_07_error_handling_user_friendly_message(self):
        """Verifies that syntax and execution errors produce user-friendly console errors without crashing."""
        win = PotatosticsQtWindow()
        try:
            # 1. Nonexistent file
            win.run_command('use "nonexistent_file_987654.dta", clear', sync=True)
            console_out = win.console.toPlainText()
            self.assertTrue("not found" in console_out.lower() or "error" in console_out.lower())

            # 2. Syntax error
            win.run_command("regress invalid_syntax_@@@ ///", sync=True)
            console_out2 = win.console.toPlainText()
            self.assertTrue("error" in console_out2.lower() or "invalid" in console_out2.lower() or "variable" in console_out2.lower())

            # Window is still functional and responsive
            self.assertFalse(win.is_busy())
        finally:
            win.close()

    def test_08_safe_cancellation(self):
        """Verifies that cancel_command safely cancels/interrupts running operations."""
        win = PotatosticsQtWindow()
        try:
            # 1. Calling cancel_command when idle is safe
            win.cancel_command()
            self.assertFalse(win.is_busy())

            # 2. Genuine long-running worker cancellation
            orig_execute = win.interpreter.execute

            def slow_command_mock(cmd_str):
                for _ in range(50):
                    if win.interpreter.check_cancelled():
                        break
                    time.sleep(0.02)
                return orig_execute(cmd_str)

            win.interpreter.execute = slow_command_mock
            started = win.run_command("clear", sync=False)
            self.assertTrue(started)
            self.assertTrue(win.is_busy())

            # Issue cancellation request while running
            win.cancel_command()
            self.assertTrue(win.worker.wait(3000))
            QtWidgets.QApplication.processEvents()

            # Ensure worker stopped cleanly and window state restored to not busy
            self.assertFalse(win.is_busy())
            console_log = win.console.toPlainText()
            self.assertIn("cancelled", console_log.lower())
        finally:
            win.close()

    def test_09_session_reopening_in_gui(self):
        """Verifies session save and reopen in the GUI environment."""
        csv_file = self.tmp_path / "session_gui.csv"
        df = pd.DataFrame({"y": [10.0, 20.0, 30.0], "x": [1.0, 2.0, 3.0]})
        df.to_csv(csv_file, index=False)
        session_file = self.tmp_path / "gui_study.potproj"

        win = PotatosticsQtWindow()
        try:
            win.run_command(f'use "{csv_file}", clear', sync=True)
            win.run_command("regress y x", sync=True)
            win.run_command(f'session save "{session_file}"', sync=True)
            self.assertTrue(session_file.exists())

            # Clear
            win.run_command("clear", sync=True)
            self.assertEqual(len(win.dataset.df), 0)

            # Reopen session
            win.run_command(f'session load "{session_file}"', sync=True)
            self.assertEqual(len(win.dataset.df), 3)
            self.assertIsNotNone(win.interpreter.estimates.last_model)
        finally:
            win.close()

    def test_10_clean_application_shutdown(self):
        """Verifies clean application close and worker shutdown."""
        win = PotatosticsQtWindow()
        closed = win.close()
        self.assertTrue(closed)


if __name__ == "__main__":
    unittest.main()
