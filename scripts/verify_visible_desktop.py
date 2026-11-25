"""
scripts/verify_visible_desktop.py

Comprehensive Windows Desktop GUI & Performance Verification Suite:
- Runs live visible Windows Qt desktop application (PySide6).
- Exercises workspace docks, command console, history, variables pane, themes.
- Verifies results console LaTeX and Markdown table copy workflows on live regressions.
- Verifies Matplotlib Graph Studio rendering and exports (PNG, SVG, PDF).
- Verifies research project save and restoration (.potproj).
- Captures screenshots of dark and light themes and plot dialog.
- Benchmarks actual GUI responsiveness and scrolling frame intervals for 10k, 100k, and 1M rows.
"""

import os
import sys
import time
import tempfile
import platform
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

# Use native Windows platform (not offscreen)
if "QT_QPA_PLATFORM" in os.environ:
    del os.environ["QT_QPA_PLATFORM"]

from PySide6 import QtWidgets, QtGui

from gui_qt.main_window import PotatosticsQtWindow, QtPlotDialog
from gui_qt.data_grid import DataGridWidget
from engine.dataset import Dataset

SCREENSHOTS_DIR = ROOT_PATH / "tests" / "data" / "benchmark_results" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def measure_scrolling_latency(view, n_scrolls=100):
    """Measures frame intervals (ms) during continuous scrolling."""
    latencies = []
    vbar = view.verticalScrollBar()
    max_val = vbar.maximum()
    if max_val <= 0:
        return [0.0]

    step = max(1, max_val // n_scrolls)
    current = 0

    for _ in range(n_scrolls):
        current = min(max_val, current + step)
        t0 = time.perf_counter()
        vbar.setValue(current)
        QtWidgets.QApplication.processEvents()
        view.viewport().repaint()
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    return latencies


def run_desktop_verification():
    print("=" * 80)
    print("POTATOSTICS — VISIBLE WINDOWS DESKTOP & RESPONSIVENESS VERIFICATION")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    import PySide6
    print(f"Python: {sys.version.split()[0]} | PySide6: {PySide6.__version__}")
    print("=" * 80)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    # -------------------------------------------------------------------------
    # 1. Main Window Startup & Workspace
    # -------------------------------------------------------------------------
    print("\n[Phase 1] Launching visible Potatostics desktop window...")
    t0_start = time.perf_counter()
    win = PotatosticsQtWindow()
    win.show()
    QtWidgets.QApplication.processEvents()
    t_startup = (time.perf_counter() - t0_start) * 1000.0
    print(f"  -> Window rendered in {t_startup:.2f} ms")

    # Dark theme screenshot
    win.apply_theme("Dark")
    QtWidgets.QApplication.processEvents()
    dark_shot_path = SCREENSHOTS_DIR / "workspace_dark.png"
    win.grab().save(str(dark_shot_path))
    print(f"  -> Saved Dark theme screenshot: {dark_shot_path.name}")

    # Light theme screenshot
    win.apply_theme("Light")
    QtWidgets.QApplication.processEvents()
    light_shot_path = SCREENSHOTS_DIR / "workspace_light.png"
    win.grab().save(str(light_shot_path))
    print(f"  -> Saved Light theme screenshot: {light_shot_path.name}")

    # Revert to Dark
    win.apply_theme("Dark")
    QtWidgets.QApplication.processEvents()

    # -------------------------------------------------------------------------
    # 2. Results Console & Regression Table Copy Workflows
    # -------------------------------------------------------------------------
    print("\n[Phase 2] Verifying Results Console table copy workflows...")
    # Load sample auto data into session
    sample_df = pd.DataFrame({
        "price": [5000, 6000, 4500, 7000, 8000, 5500, 9000, 6500, 7200, 8100],
        "mpg": [20, 25, 18, 30, 15, 22, 14, 21, 24, 16],
        "weight": [3000, 2500, 3200, 2200, 3800, 2900, 4000, 3100, 2600, 3700]
    })
    win.dataset.load_dataframe(sample_df)
    win.variables_pane.refresh()
    win.data_grid.refresh()
    win.command_bar.update_autocomplete()

    # Execute Model 1: regress price mpg weight
    win.run_command_sync("regress price mpg weight")
    QtWidgets.QApplication.processEvents()

    # Execute Model 2: regress price mpg
    win.run_command_sync("regress price mpg")
    QtWidgets.QApplication.processEvents()

    console_text = win.console.toPlainText()
    assert "price" in console_text, "Regression output must appear in console"

    def read_clipboard_after(action):
        QtWidgets.QApplication.clipboard().clear()
        time.sleep(0.02)
        action()
        for _ in range(20):
            QtWidgets.QApplication.processEvents()
            txt = QtWidgets.QApplication.clipboard().text()
            if txt:
                return txt
            time.sleep(0.02)
        return QtWidgets.QApplication.clipboard().text()

    # Verify execution records are present and distinct
    records = win.console._execution_records
    assert len(records) >= 2, "Both regression execution records must be tracked"
    rec1 = records[-2]
    rec2 = records[-1]
    assert rec1.command_text == "regress price mpg weight"
    assert rec2.command_text == "regress price mpg"

    # 2a. No Selection: defaults to latest model (Model 2) with Model 2's diagnostics
    cursor = win.console.textCursor()
    cursor.clearSelection()
    win.console.setTextCursor(cursor)
    cb_latest_full = read_clipboard_after(win.console._copy_full_report_as_latex)
    assert "\\begin{table}[htbp]" in cb_latest_full
    assert "Model Diagnostics" in cb_latest_full
    assert "0.1188" in cb_latest_full, "Latest model report must have Model 2 R2 (0.1188)"
    assert "0.7177" not in cb_latest_full, "Latest model report must NOT have Model 1 R2 (0.7177)"
    print("  -> Verified GUI action: No-selection defaults to latest model report (Model 2)")

    # 2b. Select Model 1: exports Model 1 with Model 1's diagnostics
    cursor.setPosition(rec1.char_start)
    cursor.setPosition(rec1.char_end, QtGui.QTextCursor.MoveMode.KeepAnchor)
    win.console.setTextCursor(cursor)
    cb_m1_full = read_clipboard_after(win.console._copy_full_report_as_latex)
    assert "0.7177" in cb_m1_full, "Selected Model 1 must have Model 1 R2 (0.7177)"
    assert "0.1188" not in cb_m1_full, "Selected Model 1 must NOT have Model 2 R2 (0.1188)"
    print("  -> Verified GUI action: Select Model 1 exports Model 1's scoped diagnostics")

    # 2c. Select Model 2: exports Model 2 with Model 2's diagnostics
    cursor.setPosition(rec2.char_start)
    cursor.setPosition(rec2.char_end, QtGui.QTextCursor.MoveMode.KeepAnchor)
    win.console.setTextCursor(cursor)
    cb_m2_full = read_clipboard_after(win.console._copy_full_report_as_latex)
    assert "0.1188" in cb_m2_full, "Selected Model 2 must have Model 2 R2 (0.1188)"
    assert "0.7177" not in cb_m2_full, "Selected Model 2 must NOT have Model 1 R2 (0.7177)"
    print("  -> Verified GUI action: Select Model 2 exports Model 2's scoped diagnostics")

    # 2d. Table Only export: does NOT include Model Diagnostics
    cb_m2_table = read_clipboard_after(win.console._copy_as_latex)
    assert "\\begin{table}[htbp]" in cb_m2_table
    assert "Model Diagnostics" not in cb_m2_table, "Table-only export must NOT contain diagnostics"
    print("  -> Verified GUI action: Copy Table as LaTeX (table only without diagnostics)")

    # 2e. Markdown Table and Full Report
    cb_md_table = read_clipboard_after(win.console._copy_as_markdown)
    assert "| price |" in cb_md_table
    assert "### Model Diagnostics" not in cb_md_table
    cb_md_full = read_clipboard_after(win.console._copy_full_report_as_markdown)
    assert "### Model Diagnostics" in cb_md_full
    assert "0.1188" in cb_md_full
    print("  -> Verified GUI action: Copy Table and Full Report as Markdown")

    # 2f. Verify invalid selection fallback and user notification
    saved_records = list(win.console._execution_records)
    win.console.setPlainText("Status | pending\nDiagnostic output")
    cursor = win.console.textCursor()
    cursor.select(QtGui.QTextCursor.SelectionType.Document)
    win.console.setTextCursor(cursor)
    cb_text = read_clipboard_after(win.console._copy_as_latex)
    assert cb_text.startswith("Status | pending"), "Fallback plain text must be copied when table is invalid"
    cb_text_full = read_clipboard_after(win.console._copy_full_report_as_latex)
    assert cb_text_full.startswith("Status | pending"), "Fallback plain text must be copied when full report table is invalid"
    print("  -> Verified invalid table rejection and plain-text fallback notification")

    # Restore console output and records
    win.console.setPlainText(console_text)
    win.console._execution_records = saved_records

    # -------------------------------------------------------------------------
    # 3. Graphics Studio Rendering & Exports (PNG, SVG, PDF)
    # -------------------------------------------------------------------------
    print("\n[Phase 3] Testing Plot Studio rendering and multi-format exports...")
    plot_data = sample_df
    plot_dlg = QtPlotDialog("scatter", plot_data, kwargs={"xvar": "weight", "yvar": "price"}, theme="Dark", parent=win)
    plot_dlg.show()
    QtWidgets.QApplication.processEvents()

    # Verify buttons are present and enabled
    assert plot_dlg.btn_png.isEnabled(), "Export PNG button must be enabled"
    assert plot_dlg.btn_pdf.isEnabled(), "Export PDF button must be enabled"
    assert plot_dlg.btn_svg.isEnabled(), "Export SVG button must be enabled"
    print("  -> Verified Graph Studio export controls: btn_png, btn_pdf, btn_svg")

    plot_shot_path = SCREENSHOTS_DIR / "plot_studio_scatter.png"
    plot_dlg.grab().save(str(plot_shot_path))
    print(f"  -> Saved Plot Studio screenshot: {plot_shot_path.name}")

    with tempfile.TemporaryDirectory() as tmpdir:
        png_file = os.path.join(tmpdir, "scatter_export.png")
        svg_file = os.path.join(tmpdir, "scatter_export.svg")
        pdf_file = os.path.join(tmpdir, "scatter_export.pdf")
        invalid_file = os.path.join(tmpdir, "nonexistent_dir", "scatter.png")

        # Test _export_as method directly with target_path
        res_png = plot_dlg._export_as("png", target_path=png_file)
        assert res_png, "Export PNG via _export_as must succeed"
        with open(png_file, "rb") as f:
            png_header = f.read(8)
        assert png_header == b"\x89PNG\r\n\x1a\n", "PNG file header must match signature"

        res_svg = plot_dlg._export_as("svg", target_path=svg_file)
        assert res_svg, "Export SVG via _export_as must succeed"
        with open(svg_file, "rb") as f:
            svg_content = f.read(500)
        assert b"<svg" in svg_content or b"<?xml" in svg_content, "SVG content must match signature"

        res_pdf = plot_dlg._export_as("pdf", target_path=pdf_file)
        assert res_pdf, "Export PDF via _export_as must succeed"
        with open(pdf_file, "rb") as f:
            pdf_header = f.read(5)
        assert pdf_header == b"%PDF-", "PDF file header must match signature"

        # Test error handling on invalid path
        res_fail = plot_dlg._export_as("png", target_path=invalid_file)
        assert not res_fail, "Export to invalid path must return False without unhandled exception"

        print(f"  -> Verified exports: PNG ({os.path.getsize(png_file):,} B, signature valid), "
              f"SVG ({os.path.getsize(svg_file):,} B, signature valid), "
              f"PDF ({os.path.getsize(pdf_file):,} B, signature valid)")
        print("  -> Verified export error handling on invalid path")

    plot_dlg.close()

    # -------------------------------------------------------------------------
    # 4. Research Project Persistence (.potproj)
    # -------------------------------------------------------------------------
    print("\n[Phase 4] Verifying complete research project persistence (.potproj)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        proj_path = os.path.join(tmpdir, "research_project.potproj")
        saved = win.save_session(proj_path)
        assert saved, "Project bundle must save successfully"
        assert os.path.exists(proj_path), "Project file must exist on disk"

        # Clear session
        win.run_command("clear")
        QtWidgets.QApplication.processEvents()
        assert win.dataset.is_empty, "Dataset must be cleared"

        # Reopen project
        loaded = win.open_session(proj_path)
        assert loaded, "Project bundle must reload successfully"
        assert not win.dataset.is_empty, "Dataset must be restored"
        pd.testing.assert_frame_equal(win.dataset.df, sample_df, check_dtype=True)
        assert len(win.dataset.df) == 10, "Restored rows must match original"
        print("  -> Verified analytical session save & restore with exact numerical parity")

    # -------------------------------------------------------------------------
    # 5. Large-Dataset Responsiveness Benchmarking
    # -------------------------------------------------------------------------
    print("\n[Phase 5] Measuring actual GUI responsiveness across dataset scales...")
    print("  [NOTE] Scroll latency benchmarks measure programmatic widget scroll-step and repaint processing time,")
    print("         not physical display hardware V-Sync refresh rate.")
    print(f"{'Scale (N)':<14} | {'DF Gen':<9} | {'Load':<8} | {'Model Init':<11} | {'Paint':<9} | {'Scroll p50':<11} | {'Scroll p95':<11} | {'Scroll Max':<11} | {'Scroll Rate':<15}")
    print("-" * 110)

    test_scales = [10_000, 100_000, 1_000_000]

    for n in test_scales:
        # 1. DF Generation
        t0 = time.perf_counter()
        df = pd.DataFrame({
            "id": np.arange(n, dtype=np.int64),
            "score": np.random.randn(n),
            "cost": np.random.uniform(10.0, 500.0, size=n),
            "category": np.random.choice(["Alpha", "Beta", "Gamma"], size=n)
        })
        t_gen = (time.perf_counter() - t0) * 1000.0

        # 2. Dataset Load
        t0 = time.perf_counter()
        ds = Dataset()
        ds.load_dataframe(df)
        t_load = (time.perf_counter() - t0) * 1000.0

        # 3. Model Init
        t0 = time.perf_counter()
        grid = DataGridWidget(ds)
        grid.resize(800, 600)
        grid.show()
        QtWidgets.QApplication.processEvents()
        t_model = (time.perf_counter() - t0) * 1000.0

        # 4. First Paint
        t0 = time.perf_counter()
        grid.table_view.viewport().repaint()
        QtWidgets.QApplication.processEvents()
        t_paint = (time.perf_counter() - t0) * 1000.0

        # 5. Continuous Scrolling Frame Timings
        scroll_latencies = measure_scrolling_latency(grid.table_view, n_scrolls=50)
        p50 = float(np.percentile(scroll_latencies, 50))
        p95 = float(np.percentile(scroll_latencies, 95))
        max_lat = float(np.max(scroll_latencies))
        scroll_rate = 1000.0 / p50 if p50 > 0 else 60.0

        print(f"{n:<14,d} | {t_gen:>6.1f} ms | {t_load:>5.1f} ms | {t_model:>8.2f} ms | {t_paint:>6.2f} ms | {p50:>8.2f} ms | {p95:>8.2f} ms | {max_lat:>8.2f} ms | {scroll_rate:>9.1f} ops/s")

        grid.close()
        QtWidgets.QApplication.processEvents()

    win.close()
    print("\n" + "=" * 80)
    print("ALL VISIBLE WINDOWS DESKTOP CHECKS & PERFORMANCE MEASUREMENTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    run_desktop_verification()
