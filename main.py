"""
main.py - Potatostics Application Launcher
Primary desktop entry point for Potatostics statistical workstation.
Defaults to the PySide6 dockable desktop workstation.
Supports optional CustomTkinter extra via '--tk' flag.
"""

import sys
import os
import traceback
import datetime

STARTUP_LOG_FILE = "potatostics_startup.log"

def _log_and_show_error(err_title, err_msg, exc=None):
    """Writes error to startup log file and attempts to show a Qt graphical error box."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if exc:
        full_trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    else:
        full_trace = traceback.format_exc() if sys.exc_info()[0] else ""
    log_entry = (
        f"[{timestamp}] {err_title}: {err_msg}\n"
        f"{full_trace}\n"
        f"{'-'*60}\n"
    )
    try:
        with open(STARTUP_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as log_err:
        sys.stderr.write(f"Failed to write to {STARTUP_LOG_FILE}: {log_err}\n")

    sys.stderr.write(f"[Potatostics] {err_title}: {err_msg}\n")
    if full_trace:
        sys.stderr.write(f"{full_trace}\n")

    # Attempt to show Qt graphical error box if PySide6 is available
    try:
        from PySide6 import QtWidgets, QtCore
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        
        box = QtWidgets.QMessageBox()
        box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        box.setWindowTitle(err_title)
        box.setText(f"<b>{err_title}</b>")
        box.setInformativeText(f"{err_msg}\n\nDetails have been logged to: {os.path.abspath(STARTUP_LOG_FILE)}")
        if full_trace:
            box.setDetailedText(full_trace)
        
        # In headless test or offscreen mode, auto-dismiss rather than blocking indefinitely
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen" or os.environ.get("POTATOSTICS_TESTING") == "1":
            box.show()
            QtCore.QTimer.singleShot(50, box.accept)
        else:
            box.exec()
    except Exception:
        pass


def main():
    if any(a in sys.argv[1:] for a in ("--version", "-v")):
        print("Potatostics 1.0.0")
        sys.exit(0)
    if any(a in sys.argv[1:] for a in ("--help", "-h")):
        print("Potatostics 1.0.0 — Statistical and Econometric Workstation")
        print("Usage: potatostics [options] [script.do | dataset.dta]")
        print("Options:")
        print("  --batch, -b      Run headless batch script")
        print("  --headless       Headless execution")
        print("  --gui-smoke      Noninteractive Qt GUI component and execution smoke test")
        print("  --gui-smoke-visible  Run the GUI smoke test in a native desktop window")
        print("  --tk             Launch legacy CustomTkinter GUI")
        print("  --version, -v    Display version information")
        print("  --help, -h       Display this help message")
        sys.exit(0)

    if any(a in sys.argv[1:] for a in ("--gui-smoke", "--gui-smoke-test", "--gui-smoke-visible")):
        if not __debug__:
            sys.exit("GUI smoke verification requires Python assertions to be enabled")
        visible = "--gui-smoke-visible" in sys.argv[1:]
        if not visible:
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ["POTATOSTICS_TESTING"] = "1"
        try:
            from PySide6 import QtWidgets, QtCore
            app = QtWidgets.QApplication.instance()
            if app is None:
                app = QtWidgets.QApplication([sys.argv[0]])
            from gui_qt.main_window import PotatosticsQtWindow
            win = PotatosticsQtWindow()
            if visible:
                assert app.platformName() not in ("offscreen", "minimal"), "A native desktop is required"
                win.show()
                timer = QtCore.QElapsedTimer()
                timer.start()
                while not win.windowHandle().isExposed() and timer.elapsed() < 5000:
                    app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
                assert win.windowHandle().isExposed(), "Desktop window was not exposed"
            assert win.console is not None, "Console component missing"
            assert win.data_grid is not None, "Data grid component missing"
            assert win.variables_pane is not None, "Variables pane missing"
            assert win.history_pane is not None, "History pane missing"
            assert win.command_bar is not None, "Command bar missing"

            # Locate packaged data asset
            import pandas as pd
            from pathlib import Path
            base_dir = getattr(sys, "_MEIPASS", None)
            if base_dir:
                auto_path = Path(base_dir) / "engine" / "assets" / "data" / "auto.dta"
            else:
                import engine
                auto_path = Path(engine.__file__).parent / "assets" / "data" / "auto.dta"

            if not auto_path.is_file():
                # Fallback to local sample_data.csv if auto.dta is not present
                if base_dir and (Path(base_dir) / "gui_qt" / "sample_data.csv").is_file():
                    auto_path = Path(base_dir) / "gui_qt" / "sample_data.csv"
                else:
                    auto_path = Path(__file__).parent / "gui_qt" / "sample_data.csv"

            if auto_path.suffix == ".dta":
                from engine.dataset import Dataset
                _ds_tmp = Dataset()
                _ds_tmp.load_dta(str(auto_path))
                df = _ds_tmp.df
            else:
                df = pd.read_csv(auto_path)

            assert len(df) > 0, f"Sample dataset {auto_path.name} loaded empty"
            win.dataset.load_dataframe(df)
            win.data_grid.refresh()
            win.variables_pane.refresh()

            # 1. Identify explicitly known numeric variable
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            assert len(numeric_cols) > 0, "No numeric variables found in dataset"
            target_var = "price" if "price" in numeric_cols else str(numeric_cols[0])
            assert target_var in df.columns, f"Target variable '{target_var}' missing from dataset"

            win.dataset.load_dataframe(df)
            win.data_grid.refresh()
            win.variables_pane.refresh()

            # 2. Execute statistical command through genuine GUI command path
            res = win.run_command_sync(f"summarize {target_var}", timeout=15.0)
            assert res is not None, "run_command_sync returned None"
            assert res.ok is True, f"Command 'summarize {target_var}' reported failure: {res.output}"
            assert res.return_code == 0, f"Command 'summarize {target_var}' returned non-zero exit code: {res.return_code}"

            # 3. Verify actual numerical output in result
            out_str = str(res.output)
            assert "Mean" in out_str or "Obs" in out_str, f"Expected numerical output missing from summary: {out_str}"
            assert target_var in out_str, f"Target variable name missing from summary output: {out_str}"

            # 3b. Verify OLS regression estimation and numerical estimates in GUI
            second_var = [c for c in numeric_cols if c != target_var]
            if second_var:
                reg_var = second_var[0]
                reg_res = win.run_command_sync(f"regress {target_var} {reg_var}", timeout=15.0)
                assert reg_res is not None and reg_res.ok is True, f"Regression failed: {reg_res}"
                assert win.interpreter.estimates.last_model is not None, "Estimation model not stored"
                assert "R-squared" in str(reg_res.output), "R-squared missing from regression table"

            # 4. Verify GUI history recorded success
            assert len(win.history_pane.history_records) >= 1, "History record was not created"
            last_hist = win.history_pane.history_records[-1]
            assert last_hist["ok"] is True, f"History record marked failure: {last_hist}"
            assert last_hist["rc"] == 0, f"History return code was not 0: {last_hist}"

            # 5. Verify interface responsiveness & no active worker
            assert not win.is_busy(), "Window remained in busy state after command completion"
            assert win.worker is None or not win.worker.isRunning(), "Background worker remained active after completion"

            # 5b. Verify Session saving and reopening
            import tempfile
            with tempfile.TemporaryDirectory() as smoke_dir:
                session_file = Path(smoke_dir) / "session.potproj"
                saved_ok = win.save_session(str(session_file))
                assert saved_ok, "Session save failed in GUI smoke test"
                assert session_file.exists(), "Session file not created on disk"
                win.run_command_sync("clear", timeout=5.0)
                assert len(win.dataset.df) == 0, "Dataset not cleared"
                opened_ok = win.open_session(str(session_file))
                assert opened_ok, "Session open failed in GUI smoke test"
                assert len(win.dataset.df) > 0, "Dataset not restored from session"

            # 6. Mandatory negative test: execute intentionally invalid command
            bad_cmd = "summarize __non_existent_numeric_var_9999__"
            bad_res = win.run_command_sync(bad_cmd, timeout=15.0)
            assert bad_res is not None, "Negative test run_command_sync returned None"
            assert bad_res.ok is False, f"Invalid command was expected to fail, but succeeded: {bad_res}"
            assert bad_res.return_code != 0, f"Invalid command returned 0 return_code: {bad_res.return_code}"

            # Verify history records failure
            bad_hist = win.history_pane.history_records[-1]
            assert bad_hist["ok"] is False, f"History record for invalid command was not marked false: {bad_hist}"
            assert bad_hist["rc"] != 0, f"History record return code was 0 for failed command: {bad_hist}"

            # 6b. Verify worker cancellation in GUI
            started = win.run_command("display 12345")
            assert started is True
            win.cancel_command()
            if win.worker:
                win.worker.wait(1000)
            QtWidgets.QApplication.processEvents()
            assert not win.is_busy(), "Window busy after cancellation"

            # Verify app remains operational by executing a follow-up valid command
            post_res = win.run_command_sync("display 2 + 2", timeout=10.0)
            assert post_res is not None and post_res.ok is True, "GUI unresponsive after failed command"
            assert "4" in str(post_res.output), f"Expected 4 in display output: {post_res.output}"
            assert not win.is_busy(), "Window busy after failure recovery"
            assert win.worker is None or not win.worker.isRunning(), "Worker active after recovery"

            # 7. Clean shutdown
            win.close()
            win.deleteLater()
            app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            QtWidgets.QApplication.processEvents()
            print("GUI_SMOKE_TEST_PASSED", flush=True)
            os._exit(0)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            sys.stderr.write(f"GUI Smoke Test Failed: {exc}\n")
            sys.exit(1)

    args_flags = {a.lower() for a in sys.argv[1:] if a.startswith("-")}
    is_batch = bool(args_flags.intersection({"--batch", "-b", "--headless"}))

    target_file = None
    # Check for --batch=filename or -b=filename
    for a in sys.argv[1:]:
        if a.lower().startswith("--batch=") or a.lower().startswith("-b="):
            candidate = a.split("=", 1)[1].strip('"\'')
            if os.path.exists(candidate):
                target_file = os.path.abspath(candidate)
                is_batch = True

    if not target_file:
        pos_args = [a for a in sys.argv[1:] if not a.startswith("-")]
        if pos_args and os.path.exists(pos_args[0]):
            target_file = os.path.abspath(pos_args[0])

    is_do_file = bool(target_file and target_file.lower().endswith(".do"))

    # Headless Batch Execution Mode
    if is_batch or (is_do_file and bool(args_flags.intersection({"--batch", "-b", "--headless"}))):
        if not target_file:
            sys.stderr.write("Error: Batch execution requires a target script file (e.g. .do file).\n")
            sys.exit(1)
        if not os.path.exists(target_file):
            sys.stderr.write(f"Error: Target file '{target_file}' not found.\n")
            sys.exit(1)

        from engine.dataset import Dataset
        from engine.interpreter import Interpreter
        ds = Dataset()
        interp = Interpreter(ds)
        interp.cli_script_mode = True

        run_cmd = f'do "{target_file}"' if is_do_file else f'use "{target_file}", clear'
        res = interp.execute(run_cmd)

        if res.output:
            sys.stdout.write(res.output + "\n")
        if not res.ok:
            if res.error:
                sys.stderr.write(f"Error: {res.error}\n")
            sys.exit(res.return_code if res.return_code != 0 else 1)
        sys.exit(0)

    use_tk = "--tk" in sys.argv

    if not use_tk:
        try:
            from gui_qt.main_window import launch_qt_app
            sys.exit(launch_qt_app(initial_file=target_file))
        except Exception as e:
            _log_and_show_error("PySide6 Desktop Initialization Error", str(e), exc=e)
            print("[Potatostics] Falling back to lightweight CustomTkinter interface...")

    # Optional CustomTkinter UI extra
    try:
        from gui.app import PotatosticsApp
        app = PotatosticsApp()
        if target_file:
            app.after(200, lambda: app.run_command(f'use "{target_file}", clear'))
        else:
            sample_path = os.path.join(os.path.dirname(__file__), "sample_data.csv")
            if os.path.exists(sample_path):
                app.after(200, lambda: app.run_command(f'use "{sample_path}", clear'))
        app.mainloop()
    except Exception as e:
        _log_and_show_error("CustomTkinter Launch Error", str(e), exc=e)

if __name__ == "__main__":
    main()
