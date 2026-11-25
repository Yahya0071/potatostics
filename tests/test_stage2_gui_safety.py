"""
tests/test_stage2_gui_safety.py - Tests for Stage 2 P1 GUI State Safety & Serialization
"""

import time
import unittest
import pandas as pd

from PySide6 import QtGui, QtWidgets

from gui_qt.main_window import PotatosticsQtWindow, CommandWorker

app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(["--platform", "offscreen"])


class TestStage2GuiSafety(unittest.TestCase):
    def setUp(self):
        self.win = PotatosticsQtWindow()
        self.df = pd.DataFrame({
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x": [2.0, 4.0, 6.0, 8.0, 10.0]
        })
        self.win.dataset.load_dataframe(self.df.copy())

    def tearDown(self):
        if self.win.worker and self.win.worker.isRunning():
            self.win.worker.cancel()
            self.win.worker.wait(1000)
        self.win.close()

    def test_rapid_double_submit_prevention(self):
        """Second command submitted while a worker is running must be rejected or queued, not run concurrently."""
        def slow_cmd(cmd):
            for _ in range(20):
                time.sleep(0.01)
            from engine.parser import CommandResult
            return CommandResult("slow done", ok=True)
        
        self.win.interpreter._cmd_slow = slow_cmd
        
        # Start first command
        started_first = self.win.run_command("_cmd_slow")
        self.assertTrue(started_first)
        self.assertTrue(self.win.is_busy(), "Window should report is_busy()=True while worker is running")
        
        # Try to launch second command immediately
        started_second = self.win.run_command("summarize y")
        self.assertFalse(started_second, "run_command must return False and not start a second concurrent worker")
        
        # Wait for worker to finish
        self.win.worker.wait(1000)
        QtWidgets.QApplication.processEvents()
        self.assertFalse(self.win.is_busy(), "Window should report is_busy()=False after worker finishes")

    def test_menu_and_toolbar_disabled_while_busy(self):
        """Toolbar and command menus must be disabled while a command is active."""
        def slow_cmd(cmd):
            for _ in range(15):
                time.sleep(0.01)
            from engine.parser import CommandResult
            return CommandResult("done", ok=True)
            
        self.win.interpreter._cmd_slow = slow_cmd
        self.win.run_command("_cmd_slow")
        
        # Verify menus and toolbars are disabled
        self.assertFalse(self.win.toolbar.isEnabled(), "Main toolbar must be disabled while running")
        self.assertFalse(self.win.data_menu.isEnabled(), "Data menu must be disabled while running")
        self.assertFalse(self.win.stat_menu.isEnabled(), "Statistics menu must be disabled while running")
        
        self.win.worker.wait(1000)
        QtWidgets.QApplication.processEvents()
        
        # Verify menus and toolbars are restored
        self.assertTrue(self.win.toolbar.isEnabled(), "Main toolbar must be re-enabled after run")
        self.assertTrue(self.win.data_menu.isEnabled(), "Data menu must be re-enabled after run")
        self.assertTrue(self.win.stat_menu.isEnabled(), "Statistics menu must be re-enabled after run")

    def test_plot_and_cancellation_callback_restoration_on_exception(self):
        """Worker must restore interpreter.plot_callback and is_cancelled even if command raises an exception."""
        sentinel_callback = lambda *a, **k: "sentinel"
        self.win.interpreter.plot_callback = sentinel_callback
        self.win.interpreter.is_cancelled = None

        def explode_cmd(cmd):
            raise RuntimeError("Fatal command error")
            
        self.win.interpreter._cmd_explode = explode_cmd
        
        worker = CommandWorker(self.win.interpreter, "_cmd_explode")
        worker.start()
        worker.wait(1000)
        QtWidgets.QApplication.processEvents()
        
        # Callbacks must be restored to original values via finally block
        self.assertEqual(self.win.interpreter.plot_callback, sentinel_callback)
        self.assertIsNone(self.win.interpreter.is_cancelled)

    def test_cancellation_semantics_and_signal(self):
        """Cancellation must emit cancelled_signal and cleanly stop before showing cancelled."""
        def slow_loop(cmd):
            for _ in range(50):
                if self.win.interpreter.is_cancelled and self.win.interpreter.is_cancelled():
                    break
                time.sleep(0.01)
            from engine.parser import CommandResult
            return CommandResult("loop done", ok=True)
            
        self.win.interpreter._cmd_slow_loop = slow_loop
        self.win.run_command("_cmd_slow_loop")
        
        self.assertTrue(self.win.is_busy())
        self.win.cancel_command()
        
        # Wait for worker termination
        self.win.worker.wait(1000)
        QtWidgets.QApplication.processEvents()
        
        self.assertFalse(self.win.is_busy())
        # History pane must log cancellation
        records = self.win.history_pane.history_records
        last_entry = records[-1] if records else None
        self.assertIsNotNone(last_entry)
        self.assertFalse(last_entry.get("ok", True))

    def test_close_event_clean_worker_termination(self):
        """Closing the window while worker is active must safely cancel and wait for the thread."""
        def slow_work(cmd):
            for _ in range(30):
                time.sleep(0.01)
            from engine.parser import CommandResult
            return CommandResult("closed", ok=True)
            
        self.win.interpreter._cmd_slow_work = slow_work
        self.win.run_command("_cmd_slow_work")
        
        # Simulate close event
        event = QtGui.QCloseEvent()
        self.win.closeEvent(event)
        self.assertTrue(event.isAccepted())
        self.assertFalse(self.win.worker.isRunning())


if __name__ == "__main__":
    unittest.main()
