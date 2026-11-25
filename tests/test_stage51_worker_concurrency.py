"""
tests/test_stage51_worker_concurrency.py

Comprehensive tests for Potatostics Stage 51 & 52:
- Phase 1 & 2: Genuine Concurrency Testing & Controlled Interpreter Execution
  (Deterministic threading.Event synchronization, asserting no test command fails with rc=199)
- Phase 3: Complete 10-State Worker Lifecycle Verification & Execution ID Token Isolation
- Phase 4: Concurrent Session Mutation & Data Grid Protection Serialization
"""

import os
import time
import threading
import unittest
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from PySide6 import QtWidgets
from gui_qt.main_window import PotatosticsQtWindow, CommandWorker, WorkerState
from engine.interpreter import CommandResult


# Initialize single persistent QApplication for headless Qt tests
app = QtWidgets.QApplication.instance()
if app is None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["POTATOSTICS_TESTING"] = "1"
    app = QtWidgets.QApplication([])


class TestStage51WorkerConcurrency(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path("./scratch/test_stage51_scratch").resolve()
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.win = PotatosticsQtWindow()

    def tearDown(self):
        try:
            if self.win.worker and self.win.worker.isRunning():
                self.win.worker.cancel()
                self.win.worker.wait(1000)
            self.win.close()
        except Exception:
            pass

    # =========================================================================
    # Phase 1 & 3: Worker Lifecycle States, Concurrency & Generation ID Safety
    # =========================================================================

    def test_01_successful_command_completion(self):
        """State 1: Successful command execution transitions worker to COMPLETED."""
        res = self.win.run_command_sync("display 2 + 2", timeout=5.0)
        self.assertTrue(res.ok)
        self.assertEqual(res.return_code, 0)
        self.assertNotEqual(res.return_code, 199, "Must not return unrecognized command rc=199")
        self.assertIn("4", res.output)
        self.assertEqual(self.win.worker.state, WorkerState.COMPLETED)
        self.assertFalse(self.win.is_busy())
        # History pane recorded success
        records = self.win.history_pane.history_records
        self.assertTrue(any(r["cmd"] == "display 2 + 2" and r["ok"] and r["rc"] == 0 for r in records))

    def test_02_ordinary_interpreter_error(self):
        """State 2: Normal interpreter error produces rc != 0, preserves error state."""
        res = self.win.run_command_sync("regress non_existent_variable_xyz", timeout=5.0)
        self.assertFalse(res.ok)
        self.assertNotEqual(res.return_code, 0)
        self.assertNotEqual(res.return_code, 199, "Must be valid regression syntax error, not unrecognized command 199")
        self.assertEqual(self.win.worker.state, WorkerState.COMPLETED)
        self.assertFalse(self.win.is_busy())
        # History pane recorded failure with nonzero rc
        records = self.win.history_pane.history_records
        last = records[-1]
        self.assertFalse(last["ok"])
        self.assertEqual(last["rc"], res.return_code)

    def test_03_unexpected_python_exception_handling(self):
        """State 3: Worker entering FAILED does NOT get overwritten to COMPLETED by _on_worker_finished."""
        orig_execute = self.win.interpreter.execute
        def crashing_execute(cmd):
            if cmd == "_trigger_crash":
                raise RuntimeError("Fatal numerical solver crash in native library")
            return orig_execute(cmd)

        self.win.interpreter.execute = crashing_execute
        try:
            res = self.win.run_command_sync("_trigger_crash", timeout=5.0)
            self.assertFalse(res.ok)
            self.assertEqual(res.return_code, 198)
            self.assertNotEqual(res.return_code, 199, "Must be caught exception rc=198, not rc=199")
            self.assertIn("Fatal numerical solver crash", res.output)
            # CRITICAL TEST: Worker terminal state must remain FAILED, NOT overwritten to COMPLETED!
            self.assertEqual(self.win.worker.state, WorkerState.FAILED)
            self.assertFalse(self.win.is_busy())

            # Verify immediate recovery with follow-up command
            res2 = self.win.run_command_sync("display 100", timeout=5.0)
            self.assertTrue(res2.ok)
            self.assertEqual(res2.return_code, 0)
            self.assertNotEqual(res2.return_code, 199)
            self.assertEqual(self.win.worker.state, WorkerState.COMPLETED)
        finally:
            self.win.interpreter.execute = orig_execute

    def test_04_cooperative_cancellation(self):
        """State 5: Cooperative cancellation during active execution transitions worker to CANCELLED and returns rc=1."""
        orig_execute = self.win.interpreter.execute
        started_event = threading.Event()

        def slow_cancellable(cmd_text):
            if cmd_text == "_test_slow_cancel":
                started_event.set()
                for _ in range(100):
                    if self.win.interpreter.check_cancelled():
                        return CommandResult("(Command cancelled by user)", ok=False, return_code=1)
                    time.sleep(0.02)
                return CommandResult("finished", ok=True)
            return orig_execute(cmd_text)

        self.win.interpreter.execute = slow_cancellable
        try:
            self.win.run_command("_test_slow_cancel")
            # Deterministic synchronization: wait until background loop is actively running
            self.assertTrue(started_event.wait(timeout=2.0), "Background worker must genuinely start")
            self.assertTrue(self.win.is_busy())

            # Request cancellation while actively running
            self.win.cancel_command()
            self.win.worker.wait(3000)
            QtWidgets.QApplication.processEvents()

            self.assertFalse(self.win.is_busy())
            self.assertEqual(self.win.worker.state, WorkerState.CANCELLED)
            self.assertEqual(self.win.last_command_result.return_code, 1)
            self.assertNotEqual(self.win.last_command_result.return_code, 199, "Must not return unrecognized command rc=199")
            self.assertIn("cancelled", self.win.last_command_result.output.lower())
        finally:
            self.win.interpreter.execute = orig_execute

    def test_04b_cancellation_before_worker_starts(self):
        """State 4: Worker pre-emptively cancelled before thread start transitions directly to CANCELLED."""
        exec_id = self.win._current_execution_id + 1
        worker = CommandWorker(self.win.interpreter, "display 42", execution_id=exec_id)
        # Cancel before starting
        worker.cancel()
        self.assertEqual(worker.state, WorkerState.CANCELLING)
        self.assertTrue(worker._is_cancelled)

        # Start worker and wait for it
        worker.start()
        worker.wait(2000)
        QtWidgets.QApplication.processEvents()

        self.assertEqual(worker.state, WorkerState.CANCELLED)
        self.assertIsNone(worker.result, "Command must not have executed")

    def test_cancel_requested_during_mutation_reports_actual_completion(self):
        from engine import interpreter

        self.win.dataset.load_dataframe(pd.DataFrame({"x": [1, 2]}))
        entered = threading.Event()
        release = threading.Event()
        evaluate = interpreter.evaluate_dta_expression

        def delayed_evaluate(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release expression evaluation")
            return evaluate(*args, **kwargs)

        with patch.object(interpreter, "evaluate_dta_expression", delayed_evaluate):
            self.assertTrue(self.win.run_command("generate y = x * 2"))
            try:
                self.assertTrue(entered.wait(2))
                self.win.cancel_command()
                self.assertTrue(self.win.is_busy())
                self.assertNotIn("y", self.win.dataset.df.columns)
            finally:
                release.set()
                self.assertTrue(self.win.worker.wait(5000))
            QtWidgets.QApplication.processEvents()

        self.assertEqual(self.win.dataset.df["y"].tolist(), [2, 4])
        self.assertTrue(self.win.last_command_result.ok)
        self.assertIn("generated", self.win.last_command_result.output)
        self.assertEqual(self.win.worker.state, WorkerState.COMPLETED)
        self.assertFalse(self.win.is_busy())
        entry = self.win.history_pane.history_records[-1]
        self.assertTrue(entry["ok"])
        self.assertEqual(entry["rc"], 0)
        self.assertIn("completed after cancellation was requested", self.win.console.toPlainText())

    def test_05_non_cooperative_timeout_and_late_completion(self):
        """Keep rc=124 and refresh the dataset when the active worker finally stops."""
        from engine import interpreter

        self.win.dataset.load_dataframe(pd.DataFrame({"x": [1, 2]}))
        self.win.data_grid.refresh()
        entered = threading.Event()
        release = threading.Event()
        evaluate = interpreter.evaluate_dta_expression

        def delayed_evaluate(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release expression evaluation")
            return evaluate(*args, **kwargs)

        with patch.object(interpreter, "evaluate_dta_expression", delayed_evaluate):
            try:
                res_timeout = self.win.run_command_sync("generate y = x * 2", timeout=0.25)
                self.assertTrue(entered.is_set())
                self.assertFalse(res_timeout.ok)
                self.assertEqual(res_timeout.return_code, 124)
                self.assertEqual(self.win.worker.state, WorkerState.TIMED_OUT)
                self.assertTrue(self.win.is_busy())
                self.assertEqual(self.win.data_grid.model.columnCount(), 1)
            finally:
                release.set()
                self.assertTrue(self.win.worker.wait(5000))
            QtWidgets.QApplication.processEvents()

        self.assertFalse(self.win.is_busy())
        self.assertEqual(self.win.worker.state, WorkerState.TIMED_OUT)
        self.assertIs(self.win.last_command_result, res_timeout)
        self.assertEqual(self.win.history_pane.history_records[-1]["rc"], 124)
        self.assertEqual(self.win.dataset.df["y"].tolist(), [2, 4])
        self.assertEqual(self.win.data_grid.model.columnCount(), 2)
        self.assertIn("may have applied changes", self.win.console.toPlainText())

    def test_06_late_successful_result_after_cancellation_is_discarded(self):
        """Ignore a success signal from a worker that already acknowledged cancellation."""
        # Setup active worker that was cancelled
        self.win.worker = CommandWorker(self.win.interpreter, "display 1", execution_id=self.win._current_execution_id)
        self.win.worker.state = WorkerState.CANCELLED
        self.win.worker._is_cancelled = True
        self.win.last_command_result = CommandResult("(Command cancelled by user)", ok=False, return_code=1)

        # Directly invoke slot with a mock late successful result
        self.win._on_command_finished("display 1", CommandResult("late success", ok=True, return_code=0))
        # Result should not be accepted as current last_command_result
        self.assertEqual(self.win.last_command_result.return_code, 1)
        self.assertNotEqual(self.win.last_command_result.return_code, 199)
        self.assertFalse(self.win.last_command_result.ok)

    def test_07_repeated_cancellation_requests_are_idempotent(self):
        """State 6: Verifies calling cancel_command multiple times is completely idempotent."""
        self.win.cancel_command()
        self.win.cancel_command()
        self.win.cancel_command()
        self.assertFalse(self.win.is_busy())

    def test_08_close_during_computation(self):
        """State 9: Verifies clean window shutdown when closed during genuine worker execution."""
        orig_execute = self.win.interpreter.execute
        started_event = threading.Event()

        def long_loop(cmd_text):
            if cmd_text == "_test_long_cmd":
                started_event.set()
                for _ in range(100):
                    if self.win.interpreter.check_cancelled():
                        break
                    time.sleep(0.02)
                return CommandResult("done", ok=True)
            return orig_execute(cmd_text)

        self.win.interpreter.execute = long_loop
        try:
            self.win.run_command("_test_long_cmd")
            self.assertTrue(started_event.wait(timeout=2.0), "Worker must start before window close")
            self.assertTrue(self.win.is_busy())

            closed = self.win.close()
            self.assertTrue(closed)
            if self.win.worker and self.win.worker.isRunning():
                self.win.worker.wait(3000)
            QtWidgets.QApplication.processEvents()
            self.assertFalse(self.win.is_busy())
        finally:
            self.win.interpreter.execute = orig_execute

    def test_09_command_execution_immediately_after_worker_cleanup(self):
        """State 10: Verifies new command executes cleanly immediately after previous worker finishes."""
        res1 = self.win.run_command_sync("display 1 + 1", timeout=5.0)
        self.assertTrue(res1.ok)
        self.assertEqual(res1.return_code, 0)
        self.assertNotEqual(res1.return_code, 199)
        exec_id_1 = self.win.worker.execution_id

        res2 = self.win.run_command_sync("display 2 + 2", timeout=5.0)
        self.assertTrue(res2.ok)
        self.assertEqual(res2.return_code, 0)
        self.assertNotEqual(res2.return_code, 199)
        exec_id_2 = self.win.worker.execution_id

        self.assertGreater(exec_id_2, exec_id_1)
        self.assertEqual(self.win.worker.state, WorkerState.COMPLETED)

    def test_15_execution_id_token_isolation(self):
        """Verifies signal handlers ignore stale signals from superseded execution IDs."""
        self.win._current_execution_id = 42
        obsolete_worker = CommandWorker(self.win.interpreter, "display 100", execution_id=41)
        obsolete_worker.state = WorkerState.COMPLETED
        current_result = CommandResult("active valid result", ok=True, return_code=0)
        self.win.last_command_result = current_result

        # Obsolete worker emits finished_signal with error/success
        obsolete_worker.finished_signal.connect(self.win._on_command_finished)
        obsolete_res = CommandResult("stale obsolete result", ok=False, return_code=99)
        obsolete_worker.finished_signal.emit("display 100", obsolete_res)
        QtWidgets.QApplication.processEvents()

        # Active result must not be overwritten
        self.assertEqual(self.win.last_command_result.output, "active valid result")
        self.assertEqual(self.win.last_command_result.return_code, 0)

    # =========================================================================
    # Phase 2 & 4: Concurrent Session Mutation & Data Guard Serialization
    # =========================================================================

    def test_10_concurrent_session_save_is_blocked_during_execution(self):
        """Verifies save_session() is strictly blocked while a command is active."""
        orig_execute = self.win.interpreter.execute
        started_event = threading.Event()

        def slow_execute(cmd_text):
            if cmd_text == "_test_pause_cmd":
                started_event.set()
                for _ in range(40):
                    if self.win.interpreter.check_cancelled():
                        break
                    time.sleep(0.02)
                return CommandResult("pause over", ok=True)
            return orig_execute(cmd_text)

        self.win.interpreter.execute = slow_execute
        try:
            self.win.run_command("_test_pause_cmd")
            self.assertTrue(started_event.wait(timeout=2.0), "Worker must be running before save attempt")
            self.assertTrue(self.win.is_busy())

            # Attempt to save session while busy
            target_proj = str(self.tmp_dir / "blocked_save.potproj")
            if os.path.exists(target_proj):
                os.remove(target_proj)
            save_success = self.win.save_session(target_proj)
            self.assertFalse(save_success, "save_session must return False while command is executing")
            self.assertFalse(os.path.exists(target_proj), "Blocked session file must not be created")

            # Let command finish
            self.win.worker.wait(3000)
            QtWidgets.QApplication.processEvents()
            self.assertFalse(self.win.is_busy())
            self.assertEqual(self.win.last_command_result.return_code, 0)
            self.assertNotEqual(self.win.last_command_result.return_code, 199)

            # Now save session should succeed
            save_success_after = self.win.save_session(target_proj)
            self.assertTrue(save_success_after)
            self.assertTrue(os.path.exists(target_proj))
        finally:
            self.win.interpreter.execute = orig_execute

    def test_11_concurrent_session_open_is_blocked_during_execution(self):
        """Verifies open_session() is strictly blocked while a command is active."""
        # Create a valid session first
        self.win.run_command_sync("clear")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        self.win.interpreter.dataset.load_dataframe(df)
        proj_path = str(self.tmp_dir / "valid_session.potproj")
        self.win.save_session(proj_path)
        self.assertTrue(os.path.exists(proj_path))

        orig_execute = self.win.interpreter.execute
        started_event = threading.Event()

        def slow_calc(cmd_text):
            if cmd_text == "_test_slow_calc":
                started_event.set()
                for _ in range(40):
                    if self.win.interpreter.check_cancelled():
                        break
                    time.sleep(0.02)
                return CommandResult("done", ok=True)
            return orig_execute(cmd_text)

        self.win.interpreter.execute = slow_calc
        try:
            self.win.run_command("_test_slow_calc")
            self.assertTrue(started_event.wait(timeout=2.0), "Worker must be running before open attempt")
            self.assertTrue(self.win.is_busy())

            # Attempt to open session while busy
            open_success = self.win.open_session(proj_path)
            self.assertFalse(open_success, "open_session must return False while command is executing")

            # Let command finish
            self.win.worker.wait(3000)
            QtWidgets.QApplication.processEvents()
            self.assertFalse(self.win.is_busy())
            self.assertEqual(self.win.last_command_result.return_code, 0)
            self.assertNotEqual(self.win.last_command_result.return_code, 199)

            # Open session succeeds after worker termination
            open_success_after = self.win.open_session(proj_path)
            self.assertTrue(open_success_after)
        finally:
            self.win.interpreter.execute = orig_execute

    def test_12_concurrent_command_execution_is_blocked(self):
        """Verifies attempting to run a second command while busy returns False or rc=198."""
        orig_execute = self.win.interpreter.execute
        started_event = threading.Event()

        def pause_cmd(cmd_text):
            if cmd_text == "_test_pause_cmd2":
                started_event.set()
                for _ in range(40):
                    if self.win.interpreter.check_cancelled():
                        break
                    time.sleep(0.02)
                return CommandResult("pause over", ok=True)
            return orig_execute(cmd_text)

        self.win.interpreter.execute = pause_cmd
        try:
            started = self.win.run_command("_test_pause_cmd2")
            self.assertTrue(started)
            self.assertTrue(started_event.wait(timeout=2.0), "Worker must be running")
            self.assertTrue(self.win.is_busy())

            # Attempt asynchronous run_command
            started_second = self.win.run_command("display 999")
            self.assertFalse(started_second)

            # Attempt synchronous run_command_sync
            res_sync = self.win.run_command_sync("display 888")
            self.assertFalse(res_sync.ok)
            self.assertEqual(res_sync.return_code, 198)
            self.assertNotEqual(res_sync.return_code, 199)
            self.assertIn("another command is currently running", res_sync.output)

            self.win.worker.wait(3000)
            QtWidgets.QApplication.processEvents()
            self.assertFalse(self.win.is_busy())
            self.assertEqual(self.win.last_command_result.return_code, 0)
            self.assertNotEqual(self.win.last_command_result.return_code, 199)
        finally:
            self.win.interpreter.execute = orig_execute

    def test_13_concurrent_data_grid_edit_is_blocked(self):
        """Verifies VirtualPandasModel.setData() refuses edits while busy (_is_readonly=True)."""
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0]})
        self.win.interpreter.dataset.load_dataframe(df)
        self.win.data_grid.refresh()
        model = self.win.data_grid.model
        self.assertIsNotNone(model)

        orig_execute = self.win.interpreter.execute
        started_event = threading.Event()

        def slow_step(cmd_text):
            if cmd_text == "_test_slow_step":
                started_event.set()
                for _ in range(40):
                    if self.win.interpreter.check_cancelled():
                        break
                    time.sleep(0.02)
                return CommandResult("done", ok=True)
            return orig_execute(cmd_text)

        self.win.interpreter.execute = slow_step
        try:
            self.win.run_command("_test_slow_step")
            self.assertTrue(started_event.wait(timeout=2.0), "Worker must be running before edit attempt")
            self.assertTrue(self.win.is_busy())
            self.assertTrue(model._is_readonly)

            # Attempt in-place model edit during execution
            idx = model.index(0, 0)
            edit_result = model.setData(idx, "999.0")
            self.assertFalse(edit_result, "setData must return False when grid is locked/readonly during computation")
            self.assertEqual(self.win.dataset.df.iloc[0, 0], 10.0, "Canonical dataset must not be modified")

            # Finish computation
            self.win.worker.wait(3000)
            QtWidgets.QApplication.processEvents()
            self.assertFalse(self.win.is_busy())
            self.assertEqual(self.win.last_command_result.return_code, 0)
            self.assertNotEqual(self.win.last_command_result.return_code, 199)
            self.assertFalse(model._is_readonly)

            # Edit now succeeds
            edit_result_after = model.setData(idx, "999.0")
            self.assertTrue(edit_result_after)
            self.assertEqual(self.win.dataset.df.iloc[0, 0], 999.0)
        finally:
            self.win.interpreter.execute = orig_execute

    def test_14_file_menu_session_actions_disabled_during_busy_state(self):
        """Verifies session actions on File menu are disabled when busy and re-enabled when idle."""
        self.win._set_busy_state(True)
        for act in self.win._session_actions:
            self.assertFalse(act.isEnabled(), f"Action {act.text()} should be disabled during busy state")

        self.win._set_busy_state(False, force=True)
        for act in self.win._session_actions:
            self.assertTrue(act.isEnabled(), f"Action {act.text()} should be enabled when idle")


if __name__ == "__main__":
    unittest.main()
