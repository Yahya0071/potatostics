"""
scripts/verify_researcher_workflows.py

Deterministic Acceptance Test Suite for Core Researcher Workflows in Potatostics.
Executes genuine end-to-end verification of six practical econometrician workflows:

- Workflow A: CSV import, cell edit, multi-cell paste, sort, filter, generate, save/reopen binary .dta
              with complete dataset fidelity (missingness, unicode, large integers).
- Workflow B: Summarize, classical OLS, robust OLS, estimates store, predict fitted values & residuals.
- Workflow C: Multi-model do-file through GUI, independent OutputSections, scoped diagnostics,
              LaTeX/Markdown exports, esttab publication tables.
- Workflow D: Interactive graphs (scatter, line, histogram, boxplot), dark/light accessibility,
              PNG/SVG/PDF exports with artifact inspection and cancel safety.
- Workflow E: Error command recovery, cooperative worker cancellation, clean resumption.
- Workflow F: Complete analytical project state persistence (.potproj) fidelity with model parity.

Supports explicit execution modes:
  --headless : Automated CI / regression testing via Qt offscreen platform.
  --visible  : Live native Windows desktop session with UI interaction & screenshot capture.

Exits with code 0 on complete pass, 1 on any failure.
"""

import os
import sys
import time
import json
import struct
import argparse
import platform
import tempfile
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

SCREENSHOTS_DIR = ROOT_PATH / "tests" / "data" / "benchmark_results" / "screenshots"


class WorkflowAssertionError(RuntimeError):
    """Explicit assertion error that cannot be disabled by Python optimization flags (-O)."""
    pass


def require_true(condition: bool, message: str) -> None:
    """Explicit condition verification helper."""
    if not condition:
        raise WorkflowAssertionError(message)


def require_equal(actual: Any, expected: Any, message: str) -> None:
    """Explicit equality verification helper."""
    if actual != expected:
        raise WorkflowAssertionError(f"{message} (Expected: {expected!r}, Got: {actual!r})")


def require_command_success(result, expected_fragments: Tuple[str, ...] = (), min_lines: int = 0) -> None:
    """Requires completed command execution with ok=True and return_code==0."""
    if result is None:
        raise WorkflowAssertionError("Command execution returned None (no result produced)")

    if not getattr(result, "ok", False):
        err_msg = getattr(result, "error", "") or getattr(result, "output", "")
        raise WorkflowAssertionError(f"Command execution reported failure (ok=False): {err_msg}")

    ret_code = getattr(result, "return_code", 0)
    if ret_code != 0:
        raise WorkflowAssertionError(f"Command execution returned nonzero exit code {ret_code}: {getattr(result, 'output', '')}")

    output = getattr(result, "output", "") or ""
    if min_lines > 0:
        lines = [line for line in output.splitlines() if line.strip()]
        if len(lines) < min_lines:
            raise WorkflowAssertionError(f"Command output had {len(lines)} lines, expected at least {min_lines}")

    for fragment in expected_fragments:
        if fragment not in output:
            raise WorkflowAssertionError(f"Command output missing required fragment: {fragment!r}\nActual output: {output[:300]}")


def inspect_png_file(path: str) -> Tuple[int, int]:
    """Validates PNG magic bytes and reads image dimensions from IHDR."""
    with open(path, "rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise WorkflowAssertionError(f"File {path} is not a valid PNG image")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise WorkflowAssertionError(f"Invalid PNG dimensions in {path}: {width}x{height}")
    return width, height


def inspect_svg_file(path: str) -> int:
    """Validates SVG XML structure and counts graphical elements."""
    try:
        tree = ET.parse(path)
    except Exception as e:
        raise WorkflowAssertionError(f"File {path} is not valid SVG XML: {e}")
    root = tree.getroot()
    if not root.tag.lower().endswith("svg"):
        raise WorkflowAssertionError(f"Root tag of {path} is not svg: {root.tag}")
    elements_count = len(list(root.iter()))
    if elements_count < 5:
        raise WorkflowAssertionError(f"SVG file {path} contains insufficient graphical elements ({elements_count})")
    return elements_count


def inspect_pdf_file(path: str) -> int:
    """Validates PDF header and counts /Page objects."""
    with open(path, "rb") as f:
        content = f.read()
    if not content.startswith(b"%PDF-"):
        raise WorkflowAssertionError(f"File {path} does not start with PDF magic header")
    page_count = content.count(b"/Type /Page") + content.count(b"/Type/Page")
    if page_count == 0:
        # Fallback to checking for Page dictionary
        if b"/Page" not in content:
            raise WorkflowAssertionError(f"PDF file {path} contains no valid pages")
        page_count = 1
    return page_count


def run_researcher_workflows(
    mode: str = "headless",
    seed: int = 42,
    json_out: Optional[str] = None
) -> Dict[str, Any]:
    is_visible = (mode.lower() == "visible")

    print("=" * 80)
    print("POTATOSTICS — DETERMINISTIC RESEARCHER WORKFLOW ACCEPTANCE SUITE")
    print(f"Execution Mode: {'VISIBLE DESKTOP (Native Windows UI)' if is_visible else 'HEADLESS INTEGRATION (Offscreen)'}")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print("=" * 80)

    # Configure Qt Platform before creating QApplication
    if is_visible:
        if "QT_QPA_PLATFORM" in os.environ:
            del os.environ["QT_QPA_PLATFORM"]
    else:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["POTATOSTICS_TESTING"] = "1"

    import numpy as np
    import pandas as pd
    import scipy
    import statsmodels
    import PySide6
    from PySide6 import QtWidgets, QtCore

    app = QtWidgets.QApplication.instance()
    if app is None:
        if is_visible:
            app = QtWidgets.QApplication(sys.argv)
        else:
            app = QtWidgets.QApplication(["--platform", "offscreen"])

    from gui_qt.main_window import PotatosticsQtWindow, WorkerState, QtPlotDialog
    from gui_qt.console import (
        format_full_report_as_latex,
        format_full_report_as_markdown,
    )
    from engine.interpreter import CommandResult

    rng = np.random.default_rng(seed)
    t_start_total = time.perf_counter()

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "visible" if is_visible else "headless",
        "random_seed": seed,
        "environment": {
            "platform": platform.platform(),
            "os_name": platform.system(),
            "os_release": platform.release(),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "statsmodels_version": statsmodels.__version__,
            "pyside6_version": PySide6.__version__,
            "qt_platform": app.platformName(),
        },
        "verified_gui_actions": [],
        "workflows": {},
        "overall_status": "PENDING"
    }

    temp_dir_obj = tempfile.TemporaryDirectory()
    td = Path(temp_dir_obj.name).resolve()

    try:
        t0 = time.perf_counter()
        win = PotatosticsQtWindow()
        if is_visible:
            win.show()
            win.raise_()
            win.activateWindow()
            QtWidgets.QApplication.processEvents()
            SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            shot_path = SCREENSHOTS_DIR / "researcher_visible_desktop.png"
            win.grab().save(str(shot_path))
            report["visible_screenshot"] = str(shot_path)
            report["verified_gui_actions"].append(f"Window rendered on native desktop: {shot_path.name}")
            print(f"[Visible UI] Native desktop window rendered; saved screenshot: {shot_path.name}")

        t_startup = time.perf_counter() - t0
        report["startup_latency_s"] = t_startup
        print(f"\n[Init] Application Window Initialized: {t_startup*1000:.1f} ms")

        # ---------------------------------------------------------------------
        # Workflow A: Data Management & Binary .dta Persistence Fidelity
        # ---------------------------------------------------------------------
        print("\n--- Workflow A: Data Management & .dta Persistence ---")
        wf_a = {"name": "Data Management & DTA Persistence", "status": "PENDING", "steps": {}}
        t0_wf = time.perf_counter()

        n_rows = 1_000
        # Include comprehensive fixtures:
        # 1. Numeric, 2. Missing numeric values (NaN), 3. Large integers (near 2^53),
        # 4. Standard strings, 5. Multilingual/Arabic strings, 6. Categoricals, 7. Dates
        wage_vals = rng.lognormal(3.0, 0.4, size=n_rows)
        # Inject realistic missing values
        wage_vals[rng.choice(n_rows, size=30, replace=False)] = np.nan

        exp_vals = rng.uniform(1, 40, size=n_rows)
        exp_vals[rng.choice(n_rows, size=20, replace=False)] = np.nan

        # Large 64-bit integer identifiers
        large_ids = np.arange(9007199254740000, 9007199254740000 + n_rows, dtype=np.int64)

        # Multilingual & Arabic text notes
        notes_corpus = [
            "Baseline survey observation",
            "ملاحظة مسح اقتصادي قياسي",
            "Économétrie et statistiques appliquées",
            "Verified field interview"
        ]
        notes_col = [notes_corpus[i % len(notes_corpus)] for i in range(n_rows)]

        df_raw = pd.DataFrame({
            "id": large_ids,
            "wage": wage_vals,
            "educ": rng.choice([10, 12, 14, 16, 18], size=n_rows),
            "experience": exp_vals,
            "region": rng.choice(["North", "South", "East", "West"], size=n_rows),
            "notes": notes_col,
            "union": rng.choice([0, 1], size=n_rows)
        })
        df_raw["id"] = df_raw["id"].astype("Int64")
        df_raw.loc[1, "id"] = pd.NA
        csv_path = td / "workflow_a_data.csv"
        df_raw.to_csv(csv_path, index=False, encoding="utf-8")

        # 1. CSV Import
        res = win.run_command_sync(f'import delimited "{csv_path}", clear', timeout=10.0)
        require_command_success(res)
        require_equal(len(win.dataset.df), n_rows, "Imported dataset row count")
        pd.testing.assert_series_equal(win.dataset.df["id"], df_raw["id"])
        require_true("wage" in win.dataset.df.columns and "notes" in win.dataset.df.columns, "Columns present")
        wf_a["steps"]["import_csv"] = "PASS"
        report["verified_gui_actions"].append("Import CSV with multilingual text and large integers")

        # 2. In-place Cell Edit with Validation
        grid_model = win.data_grid.model
        idx_educ = grid_model.index(0, 2)
        res_bad = grid_model.setData(idx_educ, "invalid_float_string")
        require_true(not res_bad, "Grid must reject invalid string edit in numeric column")
        res_good = grid_model.setData(idx_educ, "21")
        require_true(res_good, "Grid must accept valid numeric edit")
        require_equal(float(win.dataset.df["educ"].iloc[0]), 21.0, "Edited cell value")
        wf_a["steps"]["cell_edit"] = "PASS"
        report["verified_gui_actions"].append("In-place Data Grid cell edit with type protection")

        # 3. Multi-cell Paste
        paste_text = "22\t15.5\n24\t18.0"
        QtWidgets.QApplication.clipboard().setText(paste_text)
        pasted = win.data_grid.paste_clipboard(start_row=1, start_col=2)
        require_equal(pasted, 4, "Number of pasted cells")
        require_equal(float(win.dataset.df["educ"].iloc[1]), 22.0, "First pasted cell value")
        wf_a["steps"]["multicell_paste"] = "PASS"
        report["verified_gui_actions"].append("Multi-cell clipboard rectangular paste")

        # 4. Sorting & Filtering
        grid_model.sort_by_column(1)  # wage
        require_equal(grid_model._sort_col, "wage", "Sort column name")
        grid_model.set_filter("North")
        filtered_cnt = grid_model.rowCount()
        require_true(0 < filtered_cnt < n_rows, "Filtered count must be subset of rows")
        grid_model.set_filter("")
        require_equal(grid_model.rowCount(), n_rows, "Restored filter row count")
        wf_a["steps"]["sort_and_filter"] = "PASS"
        report["verified_gui_actions"].append("Data Grid column sort and case-insensitive filter")

        # 5. Variable Generation
        res = win.run_command_sync("gen log_wage = log(wage)", timeout=5.0)
        require_command_success(res)
        require_true("log_wage" in win.dataset.df.columns, "log_wage in columns")
        res = win.run_command_sync("gen exp_sq = experience^2", timeout=5.0)
        require_command_success(res)
        require_true("exp_sq" in win.dataset.df.columns, "exp_sq in columns")
        wf_a["steps"]["generate_variables"] = "PASS"

        # 6. Save & Reopen Binary .dta with Deep Fidelity Verification
        # Set variable label
        win.dataset.var_labels["wage"] = "Hourly wage in USD"
        win.dataset.var_labels["notes"] = "Multilingual interviewer notes"

        dta_file = td / "workflow_a_saved.dta"
        dta_str = str(dta_file).replace("\\", "/")
        rejected = win.run_command_sync(f'save "{dta_str}", replace', timeout=10.0)
        require_true(not rejected.ok and "column 'id'" in str(rejected), "Lossy numeric ID export rejected")
        require_true(not dta_file.exists(), "Rejected numeric export left no partial DTA")
        # DTA has no 64-bit integer storage. Store the same identifiers as
        # decimal strings, with the DTA empty-string convention for missing.
        pd.testing.assert_series_equal(win.dataset.df["id"], df_raw["id"])
        id_text = ["" if pd.isna(value) else str(int(value)) for value in df_raw["id"]]
        win.dataset.df["id"] = id_text
        res = win.run_command_sync(f'save "{dta_str}", replace', timeout=10.0)
        require_command_success(res)
        require_true(dta_file.exists(), ".dta file must exist on disk")

        pre_save_df = win.dataset.df.copy()
        pre_save_cols = list(pre_save_df.columns)
        pre_save_wage_na_mask = pre_save_df["wage"].isna().to_numpy()

        res = win.run_command_sync("clear", timeout=5.0)
        require_command_success(res)
        require_equal(len(win.dataset.df), 0, "Cleared dataset row count")

        res = win.run_command_sync(f'use "{dta_str}", clear', timeout=10.0)
        require_command_success(res)
        post_df = win.dataset.df

        # Verify full fidelity:
        # A. Dimensions & column ordering
        require_equal(len(post_df), n_rows, "Reloaded observation count")
        require_equal(list(post_df.columns), pre_save_cols, "Column ordering parity")
        require_equal(post_df["id"].tolist(), id_text, "Every identifier and missing ID survived DTA")

        # B. Missing value mask preservation
        post_wage_na_mask = post_df["wage"].isna().to_numpy()
        require_true(np.array_equal(pre_save_wage_na_mask, post_wage_na_mask), "Missing value mask preserved exactly")

        # C. Numeric value preservation on non-missing observations
        valid_mask = ~pre_save_wage_na_mask
        np.testing.assert_allclose(
            post_df["wage"].to_numpy(dtype=float)[valid_mask],
            pre_save_df["wage"].to_numpy(dtype=float)[valid_mask],
            rtol=1e-6,
            err_msg="Wage numeric values preserved within DTA float serialization tolerance"
        )

        # D. Multilingual/Arabic string preservation
        sample_note = pre_save_df["notes"].iloc[1]
        reloaded_note = post_df["notes"].iloc[1]
        require_equal(reloaded_note, sample_note, "Arabic and multilingual string preservation")

        # E. Variable labels preservation
        require_equal(win.dataset.var_labels.get("wage"), "Hourly wage in USD", "Variable label preserved")
        wf_a["steps"]["dta_save_and_reopen"] = "PASS"
        report["verified_gui_actions"].append("DTA rejects lossy integer export and preserves exact text IDs, missing values, wages and Arabic text")

        wf_a["status"] = "PASS"
        wf_a["elapsed_s"] = time.perf_counter() - t0_wf
        report["workflows"]["workflow_a"] = wf_a
        print(f"  -> Workflow A PASS ({wf_a['elapsed_s']:.2f}s)")

        # ---------------------------------------------------------------------
        # Workflow B: Econometric Estimation & Inference
        # ---------------------------------------------------------------------
        print("\n--- Workflow B: Econometric Analysis & Inference ---")
        wf_b = {"name": "Econometric Analysis & Inference", "status": "PENDING", "steps": {}}
        t0_wf = time.perf_counter()

        # 1. Summarize
        res = win.run_command_sync("summarize wage educ experience", timeout=5.0)
        require_command_success(res, min_lines=2)
        if "Mean" not in res.output and "mean" not in res.output.lower():
            raise WorkflowAssertionError(f"Missing mean summary in summarize output: {res.output}")
        wf_b["steps"]["summarize"] = "PASS"

        # 2. Classical OLS
        res_ols = win.run_command_sync("regress log_wage educ experience exp_sq", timeout=10.0)
        require_command_success(res_ols, ("Number of obs", "R-squared"))
        res_m1 = win.interpreter.estimates.last_model
        require_true(res_m1 is not None, "Last model estimate record must exist")
        res_store = win.run_command_sync("estimates store m1", timeout=5.0)
        require_command_success(res_store)
        wf_b["steps"]["ols_classical"] = "PASS"

        # 3. Robust OLS (HC1)
        res_rob = win.run_command_sync("regress log_wage educ experience exp_sq, vce(robust)", timeout=10.0)
        require_command_success(res_rob, ("Number of obs",))
        require_true("Robust" in res_rob.output or "HC1" in res_rob.output, "Robust VCE in output")
        res_m2 = win.interpreter.estimates.last_model
        require_true(res_m2 is not None, "Robust model estimate record must exist")

        # Coefficients must be analytically identical; standard errors must differ
        np.testing.assert_allclose(res_m1["beta"], res_m2["beta"], rtol=1e-10)
        se_m1 = np.sqrt(np.diag(res_m1["vcov"]))
        se_m2 = np.sqrt(np.diag(res_m2["vcov"]))
        require_true(not np.allclose(se_m1, se_m2, rtol=1e-5), "Robust standard errors must differ from classical SEs")
        res_store = win.run_command_sync("estimates store m2", timeout=5.0)
        require_command_success(res_store)
        wf_b["steps"]["ols_robust"] = "PASS"

        # 4. Predict fitted values & residuals
        res = win.run_command_sync("predict yhat, xb", timeout=5.0)
        require_command_success(res)
        require_true("yhat" in win.dataset.df.columns, "Fitted values column generated")

        res = win.run_command_sync("predict ehat, resid", timeout=5.0)
        require_command_success(res)
        require_true("ehat" in win.dataset.df.columns, "Residuals column generated")
        wf_b["steps"]["predict_xb_and_resid"] = "PASS"

        wf_b["status"] = "PASS"
        wf_b["elapsed_s"] = time.perf_counter() - t0_wf
        report["workflows"]["workflow_b"] = wf_b
        print(f"  -> Workflow B PASS ({wf_b['elapsed_s']:.2f}s)")

        # ---------------------------------------------------------------------
        # Workflow C: Multi-Model Reporting & Publication Exports
        # ---------------------------------------------------------------------
        print("\n--- Workflow C: Multi-Model Reporting & Publication Exports ---")
        wf_c = {"name": "Multi-Model Reporting & Publication Exports", "status": "PENDING", "steps": {}}
        t0_wf = time.perf_counter()

        # 1. Multi-model do-file execution through the GUI command path
        dofile_path = td / "multimodel_script.do"
        dofile_content = (
            "regress log_wage educ experience exp_sq\n"
            "regress log_wage educ\n"
        )
        dofile_path.write_text(dofile_content, encoding="utf-8")

        # Execute do-file via GUI command runner
        dofile_cmd_str = str(dofile_path).replace("\\", "/")
        res_dofile = win.run_command_sync(f'do "{dofile_cmd_str}"', timeout=15.0)
        require_command_success(res_dofile)
        require_true(hasattr(res_dofile, "output_sections") and len(res_dofile.output_sections) >= 2,
                     "Do-file execution must yield independent OutputSections")
        report["verified_gui_actions"].append("GUI command execution of multi-model do-file with sections")

        # Verify structured export retains both models with independent diagnostics
        md_multi = format_full_report_as_markdown(res_dofile)
        require_true("log_wage" in md_multi, "Markdown contains depvar log_wage")
        require_true("experience" in md_multi, "Markdown contains regressor experience")
        require_true("Model Diagnostics" in md_multi, "Markdown contains Model Diagnostics")

        # Also LaTeX export
        tex_multi = format_full_report_as_latex(res_dofile)
        require_true("\\begin{table}" in tex_multi, "LaTeX contains table environment")
        require_true("Model Diagnostics" in tex_multi, "LaTeX contains Model Diagnostics")

        # Compare model-specific statistics
        r2_m1 = res_dofile.output_sections[0].text
        r2_m2 = res_dofile.output_sections[1].text
        require_true("R-squared" in r2_m1 and "R-squared" in r2_m2, "Both sections contain R-squared")
        require_true("experience" in r2_m1 and "experience" not in r2_m2, "Model 1 has experience, Model 2 does not")
        wf_c["steps"]["multimodel_dofile_export"] = "PASS"

        # Publication esttab table export
        esttab_tex = td / "models_table.tex"
        esttab_tex_str = str(esttab_tex).replace("\\", "/")
        res = win.run_command_sync(f'esttab m1 m2 using "{esttab_tex_str}", replace b(3) se(3)', timeout=10.0)
        require_command_success(res)
        require_true(esttab_tex.exists(), "esttab LaTeX table exists on disk")
        tex_content = esttab_tex.read_text(encoding="utf-8")
        require_true("educ" in tex_content and "experience" in tex_content, "esttab table contains variables")
        wf_c["steps"]["esttab_latex"] = "PASS"

        wf_c["status"] = "PASS"
        wf_c["elapsed_s"] = time.perf_counter() - t0_wf
        report["workflows"]["workflow_c"] = wf_c
        print(f"  -> Workflow C PASS ({wf_c['elapsed_s']:.2f}s)")

        # ---------------------------------------------------------------------
        # Workflow D: Interactive Visualizations & Graph Export
        # ---------------------------------------------------------------------
        print("\n--- Workflow D: Interactive Visualizations & Graph Export ---")
        wf_d = {"name": "Interactive Visualizations & Graph Export", "status": "PENDING", "steps": {}}
        t0_wf = time.perf_counter()

        plot_df = win.dataset.df.iloc[:200].dropna(subset=["wage", "experience"]).copy()

        # 1. Scatter Plot & Theme Accessibility
        dlg_scatter = QtPlotDialog("scatter", plot_df, kwargs={"yvar": "wage", "xvar": "experience"}, theme="Dark")
        require_true(dlg_scatter.figure is not None, "Plot figure initialized")
        require_true("#252526" in dlg_scatter.toolbar.styleSheet(), "Dark theme toolbar contrast verified")

        png_out = str(td / "plot_export.png")
        svg_out = str(td / "plot_export.svg")
        pdf_out = str(td / "plot_export.pdf")

        # PNG Export & Header Inspection
        export_ok = dlg_scatter._export_as("png", target_path=png_out)
        require_true(export_ok and os.path.exists(png_out), "PNG export created file")
        w, h = inspect_png_file(png_out)
        require_true(w > 100 and h > 100, f"Valid PNG resolution: {w}x{h}")

        # SVG Export & XML Inspection
        export_ok = dlg_scatter._export_as("svg", target_path=svg_out)
        require_true(export_ok and os.path.exists(svg_out), "SVG export created file")
        svg_elems = inspect_svg_file(svg_out)
        require_true(svg_elems >= 10, f"Valid SVG graphical elements: {svg_elems}")

        # PDF Export & Document Inspection
        export_ok = dlg_scatter._export_as("pdf", target_path=pdf_out)
        require_true(export_ok and os.path.exists(pdf_out), "PDF export created file")
        pdf_pages = inspect_pdf_file(pdf_out)
        require_true(pdf_pages >= 1, f"Valid PDF page count: {pdf_pages}")

        # Test cancelled export via simulated cancel in getSaveFileName
        from unittest import mock
        with mock.patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=("", "")):
            with mock.patch.object(QtWidgets.QMessageBox, "information") as mock_info:
                dlg_scatter.btn_png.click()
                require_true(not mock_info.called, "Cancelled export must not show success box")

        # Test invalid destination directory error handling
        bad_dest = "/non_existent_directory_987654/plot.png"
        with mock.patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(bad_dest, "PNG Files (*.png)")):
            with mock.patch.object(QtWidgets.QMessageBox, "critical") as mock_crit:
                dlg_scatter.btn_png.click()
                require_true(mock_crit.called, "Invalid destination must show critical error box")

        # Also verify Light theme toolbar accessibility
        dlg_light = QtPlotDialog("scatter", plot_df, kwargs={"yvar": "wage", "xvar": "experience"}, theme="Light")
        require_true("#F3F3F3" in dlg_light.toolbar.styleSheet(), "Light theme toolbar contrast verified")
        dlg_light.close()

        dlg_scatter.close()
        wf_d["steps"]["scatter_and_export"] = "PASS"
        report["verified_gui_actions"].append("QtPlotDialog PNG, SVG, PDF button exports, cancel, and invalid path safety")

        # 2. Distributional & Category Plots
        dlg_line = QtPlotDialog("line", plot_df, kwargs={"yvar": "wage", "xvar": "experience"})
        dlg_line.close()
        dlg_hist = QtPlotDialog("histogram", plot_df, kwargs={"var": "wage"})
        dlg_hist.close()
        dlg_box = QtPlotDialog("boxplot", plot_df, kwargs={"var": "wage", "byvar": "region"})
        dlg_box.close()
        wf_d["steps"]["distribution_plots"] = "PASS"

        wf_d["status"] = "PASS"
        wf_d["elapsed_s"] = time.perf_counter() - t0_wf
        report["workflows"]["workflow_d"] = wf_d
        print(f"  -> Workflow D PASS ({wf_d['elapsed_s']:.2f}s)")

        # ---------------------------------------------------------------------
        # Workflow E: Error Recovery & Cancellation Safety
        # ---------------------------------------------------------------------
        print("\n--- Workflow E: Error Recovery and Cancellation Safety ---")
        wf_e = {"name": "Error Recovery and Cancellation Safety", "status": "PENDING", "steps": {}}
        t0_wf = time.perf_counter()

        # 1. Invalid command handling
        res_bad = win.run_command_sync("regress totally_non_existent_var", timeout=5.0)
        require_true(not res_bad.ok, "Invalid command reported ok=False")
        require_true(res_bad.return_code != 0, "Invalid command returned nonzero return code")
        wf_e["steps"]["error_detection"] = "PASS"

        # 2. Subsequent command succeeds immediately
        res_ok = win.run_command_sync("display 100 * 2", timeout=5.0)
        require_command_success(res_ok, ("200",))
        wf_e["steps"]["clean_recovery"] = "PASS"

        # 3. Cooperative cancellation
        import threading
        started_event = threading.Event()
        orig_execute = win.interpreter.execute

        def cancellable_stub(cmd_text):
            if cmd_text == "_slow_test_cancel":
                started_event.set()
                for _ in range(100):
                    if win.interpreter.check_cancelled():
                        return CommandResult("(Cancelled)", ok=False, return_code=1)
                    time.sleep(0.02)
                return CommandResult("done", ok=True)
            return orig_execute(cmd_text)

        win.interpreter.execute = cancellable_stub
        try:
            win.run_command("_slow_test_cancel")
            require_true(started_event.wait(timeout=2.0), "Background thread started")
            require_true(win.is_busy(), "Window reports busy state while worker runs")
            win.cancel_command()
            win.worker.wait(3000)
            QtWidgets.QApplication.processEvents()
            require_true(not win.is_busy(), "Window clears busy state after cancellation")
            require_equal(win.worker.state, WorkerState.CANCELLED, "Worker transitioned to CANCELLED")
            wf_e["steps"]["cooperative_cancellation"] = "PASS"
            report["verified_gui_actions"].append("Cooperative QThread cancellation via Cancel button")
        finally:
            win.interpreter.execute = orig_execute

        wf_e["status"] = "PASS"
        wf_e["elapsed_s"] = time.perf_counter() - t0_wf
        report["workflows"]["workflow_e"] = wf_e
        print(f"  -> Workflow E PASS ({wf_e['elapsed_s']:.2f}s)")

        # ---------------------------------------------------------------------
        # Workflow F: Complete Project State Persistence (.potproj)
        # ---------------------------------------------------------------------
        print("\n--- Workflow F: Analytical Project Persistence (.potproj) ---")
        wf_f = {"name": "Analytical Project Persistence", "status": "PENDING", "steps": {}}
        t0_wf = time.perf_counter()

        proj_path = str(td / "researcher_acceptance_project.potproj")
        save_ok = win.save_session(proj_path)
        require_true(save_ok, "save_session reported True")
        require_true(os.path.exists(proj_path), "Project file written to disk")

        # Clear session
        res = win.run_command_sync("clear", timeout=5.0)
        require_command_success(res)
        require_true(win.dataset.is_empty, "Dataset cleared")

        # Reload session
        open_ok = win.open_session(proj_path)
        require_true(open_ok, "open_session reported True")
        require_true(not win.dataset.is_empty, "Dataset restored")
        require_equal(len(win.dataset.df), n_rows, "Restored dataset row count")
        require_true("log_wage" in win.dataset.df.columns, "Restored created variable log_wage")
        require_true("yhat" in win.dataset.df.columns, "Restored fitted values yhat")
        require_true("m1" in win.interpreter.estimates.store, "Model m1 restored in estimates store")
        require_true("m2" in win.interpreter.estimates.store, "Model m2 restored in estimates store")

        # Check covariance & coefficient exact preservation for both stored models
        reloaded_m1 = win.interpreter.estimates.store["m1"]
        reloaded_m2 = win.interpreter.estimates.store["m2"]
        np.testing.assert_allclose(reloaded_m1["beta"], res_m1["beta"], rtol=1e-12,
                                  err_msg="Model m1 coefficients preserved across project save/reopen")
        np.testing.assert_allclose(reloaded_m1["vcov"], res_m1["vcov"], rtol=1e-12,
                                  err_msg="Model m1 covariance preserved across project save/reopen")
        np.testing.assert_allclose(reloaded_m2["beta"], res_m2["beta"], rtol=1e-12,
                                  err_msg="Model m2 coefficients preserved across project save/reopen")
        np.testing.assert_allclose(reloaded_m2["vcov"], res_m2["vcov"], rtol=1e-12,
                                  err_msg="Model m2 covariance preserved across project save/reopen")
        wf_f["steps"]["persistence_parity"] = "PASS"
        report["verified_gui_actions"].append("Full session save/reopen with multi-model coefficient & vcov parity")

        wf_f["status"] = "PASS"
        wf_f["elapsed_s"] = time.perf_counter() - t0_wf
        report["workflows"]["workflow_f"] = wf_f
        print(f"  -> Workflow F PASS ({wf_f['elapsed_s']:.2f}s)")

        win.close()
        win.deleteLater()
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        app.processEvents()

        report["overall_status"] = "PASS"
        report["total_elapsed_s"] = time.perf_counter() - t_start_total
        print("\n" + "=" * 80)
        print(f"ALL 6 RESEARCHER WORKFLOWS PASSED in mode '{report['mode']}' (Total: {report['total_elapsed_s']:.2f}s)")
        print("=" * 80)

    except Exception as ex:
        report["overall_status"] = "FAIL"
        report["error"] = {
            "message": str(ex),
            "traceback": traceback.format_exc()
        }
        print(f"\n[ERROR] Researcher Workflow Failure: {ex}")
        traceback.print_exc()

    finally:
        try:
            temp_dir_obj.cleanup()
        except Exception:
            pass

    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved structured report to: {json_out}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Deterministic Researcher Acceptance Suite for Potatostics")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--headless", action="store_true", help="Run automated headless offscreen integration tests (default)")
    mode_group.add_argument("--visible", action="store_true", help="Run visible desktop tests on native Windows UI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data generation")
    parser.add_argument("--json-out", type=str, default=None, help="Path to write structured JSON results")
    args = parser.parse_args()

    mode = "visible" if args.visible else "headless"
    report = run_researcher_workflows(mode=mode, seed=args.seed, json_out=args.json_out)
    exit_code = 0 if report.get("overall_status") == "PASS" else 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
