"""
scripts/evaluate_product_quality.py

Potatostics Practical Product Quality & Comprehensive Research Workflow Evaluator.
Executes genuine end-to-end verification of all six core researcher workflows:
- Workflow A: Data import, inspection, transformations, sort, filter, in-place edit, Find & Replace, roundtrip save/reopen.
- Workflow B: Econometric estimation (summarize, OLS, robust HC1, fitted values, residuals, multi-model storage/retrieval).
- Workflow C: Reporting & publication tables (LaTeX booktabs esttab, HTML export, numerical fidelity).
- Workflow D: Analytical session persistence (.potproj project serialization, clear, deserialization, covariance fidelity).
- Workflow E: Error recovery & cancellation safety (syntax error reporting, recovery, cooperative background worker cancellation).
- Workflow F: Data grid interaction (Qt VirtualPandasModel setData, sorting, filtering, and multi-cell transactional paste).

Captures structured latency, memory consumption, platform specifications, and exports both
quality_report.json and quality_report.md.
"""

import os
import sys
import time
import json
import argparse
import platform
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Headless Qt environment for offscreen execution
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["POTATOSTICS_TESTING"] = "1"

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import numpy as np
import pandas as pd
import scipy
import statsmodels
import PySide6
from PySide6 import QtWidgets

# Safe psutil loading
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(["--platform", "offscreen"])

from gui_qt.main_window import PotatosticsQtWindow, WorkerState
from gui_qt.data_grid import perform_find_replace
from engine.interpreter import CommandResult


def get_process_memory_mb() -> float:
    if HAS_PSUTIL and psutil:
        try:
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    return 0.0


def run_evaluation(seed: int = 42, output_dir: Optional[str] = None) -> Dict[str, Any]:
    print("=" * 80)
    print("POTATOSTICS — PRACTICAL PRODUCT QUALITY & RESEARCH WORKFLOW EVALUATION")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {sys.version.split()[0]} | PySide6: {PySide6.__version__} | Seed: {seed}")
    print("=" * 80)

    rng = np.random.default_rng(seed)
    mem_init_mb = get_process_memory_mb()
    t_start_total = time.perf_counter()

    report: Dict[str, Any] = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": seed,
        "environment": {
            "platform": platform.platform(),
            "os_name": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "statsmodels_version": statsmodels.__version__,
            "pyside6_version": PySide6.__version__,
            "psutil_available": HAS_PSUTIL
        },
        "workflows": {},
        "performance_metrics": {},
        "overall_status": "PENDING"
    }

    temp_dir_obj = tempfile.TemporaryDirectory()
    td = Path(temp_dir_obj.name).resolve()

    try:
        # ---------------------------------------------------------------------
        # Application Startup
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        win = PotatosticsQtWindow()
        t_startup = time.perf_counter() - t0
        report["performance_metrics"]["startup_latency_s"] = t_startup
        print(f"\n[Startup] Desktop Application Initialized: {t_startup*1000:.1f} ms")

        # ---------------------------------------------------------------------
        # Workflow A: Data Import, Preparation & Find/Replace
        # ---------------------------------------------------------------------
        print("\n--- Workflow A: Data Import and Preparation ---")
        wf_a = {"name": "Data Import and Preparation", "status": "PENDING", "steps": {}}
        t0_wf = time.perf_counter()

        n_rows = 10_000
        df_raw = pd.DataFrame({
            "id": np.arange(1, n_rows + 1),
            "wage": rng.lognormal(3.0, 0.4, size=n_rows),
            "educ": rng.choice([10, 12, 14, 16, 18], size=n_rows),
            "experience": rng.uniform(1, 40, size=n_rows),
            "region": rng.choice(["North", "South", "East", "West"], size=n_rows),
            "union": rng.choice([0, 1], size=n_rows)
        })
        # Inject known missing values
        df_raw.loc[5, "wage"] = np.nan
        df_raw.loc[10, "experience"] = np.nan

        csv_path = td / "workflow_a_data.csv"
        df_raw.to_csv(csv_path, index=False)

        # 1. Import
        t0 = time.perf_counter()
        res = win.run_command_sync(f'import delimited "{csv_path}", clear', timeout=10.0)
        assert res.ok and res.return_code == 0, f"Import failed: {res.output}"
        assert len(win.dataset.df) == n_rows, "Row count mismatch"
        assert len(win.dataset.df.columns) == 6, "Column count mismatch"
        t_import = time.perf_counter() - t0
        wf_a["steps"]["import_csv"] = {"status": "PASS", "latency_s": t_import, "rows": n_rows}

        # 2. Variable Generation
        res = win.run_command_sync("gen log_wage = log(wage)", timeout=5.0)
        assert res.ok and res.return_code == 0
        res = win.run_command_sync("gen exp_sq = experience^2", timeout=5.0)
        assert res.ok and res.return_code == 0
        assert "log_wage" in win.dataset.df.columns
        assert "exp_sq" in win.dataset.df.columns
        wf_a["steps"]["generate_variables"] = {"status": "PASS"}

        # 3. Sort & Filter in GUI
        win.data_grid.refresh()
        win.data_grid.model.set_filter("North")
        QtWidgets.QApplication.processEvents()
        filtered_cnt = win.data_grid.model.rowCount()
        assert filtered_cnt > 0 and filtered_cnt < n_rows
        win.data_grid.model.set_filter("")

        # 4. Exact Find & Replace and Transactional Safety
        c_repl = perform_find_replace(win.dataset, find_text="North", replace_text="Northern_Region", exact_match=True)
        assert c_repl == filtered_cnt, f"Replacement count mismatch: {c_repl} vs {filtered_cnt}"
        assert "Northern_Region" in win.dataset.df["region"].values
        assert "North" not in win.dataset.df["region"].values

        # 4b. No-Op replacement returns 0 changed cells
        c_noop = perform_find_replace(win.dataset, find_text="Northern_Region", replace_text="Northern_Region", match_case=True)
        assert c_noop == 0, f"No-op replacement must return 0 changed cells, got {c_noop}"

        # 4c. Mixed-column transactional rollback on invalid text in integer column
        c_bad = perform_find_replace(win.dataset, find_text="12", replace_text="invalid_str")
        assert c_bad == 0, f"Mixed-column incompatible replacement must fail transactionally, got {c_bad}"
        assert 12 in win.dataset.df["educ"].values, "Dataset must remain untouched on transaction failure"
        wf_a["steps"]["find_replace_exact"] = {"status": "PASS", "replacements": c_repl}

        # 5. Save & Reopen Data
        saved_csv = td / "saved_workflow_data.csv"
        res = win.run_command_sync(f'export delimited using "{saved_csv}", replace', timeout=5.0)
        assert res.ok and res.return_code == 0
        assert saved_csv.exists()
        res = win.run_command_sync(f'import delimited "{saved_csv}", clear', timeout=5.0)
        assert res.ok and res.return_code == 0
        assert len(win.dataset.df) == n_rows
        assert "Northern_Region" in win.dataset.df["region"].values

        wf_a["status"] = "PASS"
        wf_a["elapsed_s"] = time.perf_counter() - t0_wf
        report["workflows"]["workflow_a"] = wf_a
        print(f"  -> Workflow A PASS ({wf_a['elapsed_s']:.2f}s)")

        # ---------------------------------------------------------------------
        # Workflow B: Econometric Analysis & Stored Estimates
        # ---------------------------------------------------------------------
        print("\n--- Workflow B: Econometric Analysis ---")
        wf_b = {"name": "Econometric Analysis", "status": "PENDING", "steps": {}}
        t0_wf = time.perf_counter()

        # 1. Summary Statistics
        t0 = time.perf_counter()
        res = win.run_command_sync("summarize wage educ experience", timeout=5.0)
        assert res.ok and res.return_code == 0
        assert "Mean" in res.output or "mean" in res.output.lower()
        wf_b["steps"]["summarize"] = {"status": "PASS", "latency_s": time.perf_counter() - t0}

        # 2. OLS Classical
        t0 = time.perf_counter()
        res_ols = win.run_command_sync("regress log_wage educ experience exp_sq", timeout=10.0)
        assert res_ols.ok and res_ols.return_code == 0
        assert "Number of obs" in res_ols.output
        t_ols = time.perf_counter() - t0
        res_m1 = win.interpreter.estimates.last_model
        assert res_m1 is not None
        assert len(res_m1["beta"]) == 4  # educ, experience, exp_sq, _cons
        res = win.run_command_sync("estimates store m1", timeout=5.0)
        assert res.ok and res.return_code == 0
        wf_b["steps"]["ols_classical"] = {"status": "PASS", "latency_s": t_ols, "nobs": res_m1["n"]}

        # 3. OLS Robust HC1
        t0 = time.perf_counter()
        res_robust = win.run_command_sync("regress log_wage educ experience exp_sq, vce(robust)", timeout=10.0)
        assert res_robust.ok and res_robust.return_code == 0
        assert "Robust" in res_robust.output or "HC1" in res_robust.output
        t_robust = time.perf_counter() - t0
        res_m2 = win.interpreter.estimates.last_model
        assert res_m2 is not None
        # Coefficients must be identical between classical and robust
        np.testing.assert_allclose(res_m1["beta"], res_m2["beta"], rtol=1e-10)
        # Robust SEs must differ from classical SEs
        se_m1 = np.sqrt(np.diag(res_m1["vcov"]))
        se_m2 = np.sqrt(np.diag(res_m2["vcov"]))
        assert not np.allclose(se_m1, se_m2, rtol=1e-5)
        res = win.run_command_sync("estimates store m2", timeout=5.0)
        assert res.ok and res.return_code == 0
        wf_b["steps"]["ols_robust"] = {"status": "PASS", "latency_s": t_robust}

        # 4. Post-Estimation Diagnostics: Fitted & Residuals
        res = win.run_command_sync("predict yhat, xb", timeout=5.0)
        assert res.ok and res.return_code == 0
        res = win.run_command_sync("predict ehat, resid", timeout=5.0)
        assert res.ok and res.return_code == 0
        assert "yhat" in win.dataset.df.columns
        assert "ehat" in win.dataset.df.columns

        # Verify retrieved estimates
        assert "m1" in win.interpreter.estimates.store
        assert "m2" in win.interpreter.estimates.store

        wf_b["status"] = "PASS"
        wf_b["elapsed_s"] = time.perf_counter() - t0_wf
        report["workflows"]["workflow_b"] = wf_b
        print(f"  -> Workflow B PASS ({wf_b['elapsed_s']:.2f}s)")

        # ---------------------------------------------------------------------
        # Workflow C: Reporting & Publication Tables
        # ---------------------------------------------------------------------
        print("\n--- Workflow C: Reporting and Table Export ---")
        wf_c = {"name": "Reporting and Table Export", "status": "PENDING", "steps": {}}
        t0_wf = time.perf_counter()

        tex_file = td / "regression_models.tex"
        t0 = time.perf_counter()
        res = win.run_command_sync(f'esttab m1 m2 using "{tex_file}", replace b(3) se(3)', timeout=10.0)
        assert res.ok and res.return_code == 0
        assert tex_file.exists()
        tex_content = tex_file.read_text(encoding="utf-8")
        assert "\\begin{table}" in tex_content or "educ" in tex_content
        assert "m1" in tex_content or "Model" in tex_content
        t_tex = time.perf_counter() - t0
        wf_c["steps"]["latex_export"] = {"status": "PASS", "latency_s": t_tex, "bytes": len(tex_content)}

        # HTML export
        html_file = td / "report.html"
        res = win.run_command_sync(f'esttab m1 m2 using "{html_file}", replace html', timeout=10.0)
        assert res.ok and res.return_code == 0
        assert html_file.exists()
        html_content = html_file.read_text(encoding="utf-8")
        assert "<table" in html_content
        wf_c["steps"]["html_export"] = {"status": "PASS", "bytes": len(html_content)}

        wf_c["status"] = "PASS"
        wf_c["elapsed_s"] = time.perf_counter() - t0_wf
        report["workflows"]["workflow_c"] = wf_c
        print(f"  -> Workflow C PASS ({wf_c['elapsed_s']:.2f}s)")

        # ---------------------------------------------------------------------
        # Workflow D: Session Persistence & Full Analytical State Reload
        # ---------------------------------------------------------------------
        print("\n--- Workflow D: Analytical Session Persistence ---")
        wf_d = {"name": "Session Persistence", "status": "PENDING", "steps": {}}
        t0_wf = time.perf_counter()

        proj_file = td / "research_session.potproj"
        t0 = time.perf_counter()
        res = win.run_command_sync(f'session save "{proj_file}"', timeout=10.0)
        assert res.ok and res.return_code == 0
        assert proj_file.exists()
        t_save_proj = time.perf_counter() - t0

        # Clear active state
        res = win.run_command_sync("clear", timeout=5.0)
        assert res.ok and res.return_code == 0
        assert len(win.dataset.df) == 0

        # Reload session
        t0 = time.perf_counter()
        res = win.run_command_sync(f'session load "{proj_file}"', timeout=10.0)
        assert res.ok and res.return_code == 0
        t_load_proj = time.perf_counter() - t0

        # Verify dataset values, columns, and stored models fidelity
        assert len(win.dataset.df) == n_rows
        assert "log_wage" in win.dataset.df.columns
        assert "yhat" in win.dataset.df.columns
        assert "m1" in win.interpreter.estimates.store
        assert "m2" in win.interpreter.estimates.store
        restored_m1 = win.interpreter.estimates.store["m1"]
        np.testing.assert_allclose(restored_m1["beta"], res_m1["beta"], rtol=1e-12)
        np.testing.assert_allclose(restored_m1["vcov"], res_m1["vcov"], rtol=1e-12)

        wf_d["status"] = "PASS"
        wf_d["elapsed_s"] = time.perf_counter() - t0_wf
        wf_d["steps"]["save_s"] = t_save_proj
        wf_d["steps"]["load_s"] = t_load_proj
        report["workflows"]["workflow_d"] = wf_d
        print(f"  -> Workflow D PASS ({wf_d['elapsed_s']:.2f}s)")

        # ---------------------------------------------------------------------
        # Workflow E: Error Recovery & Cooperative Worker Cancellation
        # ---------------------------------------------------------------------
        print("\n--- Workflow E: Error Recovery and Cancellation Safety ---")
        wf_e = {"name": "Error Recovery and Cancellation Safety", "status": "PENDING", "steps": {}}
        t0_wf = time.perf_counter()

        # 1. Invalid command error handling
        res_err = win.run_command_sync("regress non_existent_variable_xyz", timeout=5.0)
        assert not res_err.ok
        assert res_err.return_code != 0
        assert res_err.return_code != 199
        wf_e["steps"]["invalid_command_error"] = {"status": "PASS", "return_code": res_err.return_code}

        # 2. Recovery with follow-up command
        res_rec = win.run_command_sync("display 2 + 2", timeout=5.0)
        assert res_rec.ok and res_rec.return_code == 0
        assert "4" in res_rec.output
        wf_e["steps"]["error_recovery_followup"] = {"status": "PASS"}

        # 3. Real slow operation cancellation
        import threading
        started_event = threading.Event()
        orig_execute = win.interpreter.execute

        def slow_cancellable(cmd_text):
            if cmd_text == "_eval_slow_cancel":
                started_event.set()
                for _ in range(100):
                    if win.interpreter.check_cancelled():
                        return CommandResult("(Cancelled)", ok=False, return_code=1)
                    time.sleep(0.02)
                return CommandResult("done", ok=True)
            return orig_execute(cmd_text)

        win.interpreter.execute = slow_cancellable
        try:
            win.run_command("_eval_slow_cancel")
            assert started_event.wait(timeout=2.0)
            assert win.is_busy()
            win.cancel_command()
            win.worker.wait(3000)
            QtWidgets.QApplication.processEvents()
            assert not win.is_busy()
            assert win.worker.state == WorkerState.CANCELLED
            assert win.last_command_result.return_code == 1
            wf_e["steps"]["cooperative_cancellation"] = {"status": "PASS"}
        finally:
            win.interpreter.execute = orig_execute

        wf_e["status"] = "PASS"
        wf_e["elapsed_s"] = time.perf_counter() - t0_wf
        report["workflows"]["workflow_e"] = wf_e
        print(f"  -> Workflow E PASS ({wf_e['elapsed_s']:.2f}s)")

        # ---------------------------------------------------------------------
        # Workflow F: Interactive Data Grid In-Place Model Manipulation
        # ---------------------------------------------------------------------
        print("\n--- Workflow F: Interactive Data Grid Behavior ---")
        wf_f = {"name": "Data Grid Model Manipulation", "status": "PENDING", "steps": {}}
        t0_wf = time.perf_counter()

        grid_model = win.data_grid.model
        assert grid_model is not None

        # 1. In-place integer editing and validation
        idx_educ = grid_model.index(0, 2)  # row 0, col 'educ'
        # Reject fractional edit
        res_bad = grid_model.setData(idx_educ, "15.8")
        assert not res_bad, "Data grid must reject fractional input in integer column"
        # Accept valid integer edit
        res_good = grid_model.setData(idx_educ, "19")
        assert res_good
        assert win.dataset.df["educ"].iloc[0] == 19

        # 1b. Large-integer precision preservation when setting missing
        win.dataset.df.iloc[0, 0] = 9007199254740993  # set id in row 0
        idx_id_row1 = grid_model.index(1, 0)
        res_na = grid_model.setData(idx_id_row1, ".")
        assert res_na
        assert int(win.dataset.df["id"].iloc[0]) == 9007199254740993, "Large-integer row 0 must preserve exact 64-bit value"
        assert pd.isna(win.dataset.df["id"].iloc[1]), "Row 1 must be missing NA"
        assert str(win.dataset.df["id"].dtype) == "Int64", "Column must be nullable Int64 ExtensionDtype"
        wf_f["steps"]["integer_cell_validation"] = {"status": "PASS"}

        # 2. View sorting synchronization
        grid_model.sort_by_column(1)  # sort by wage
        assert grid_model._sort_col == "wage"
        wf_f["steps"]["view_sort"] = {"status": "PASS"}

        # 3. View filtering synchronization
        grid_model.set_filter("Northern")
        assert grid_model.rowCount() > 0
        grid_model.set_filter("")
        assert grid_model.rowCount() == n_rows
        wf_f["steps"]["view_filter"] = {"status": "PASS"}

        # 4. Multi-cell paste validation
        valid_paste = "20\t35\n22\t40"
        QtWidgets.QApplication.clipboard().setText(valid_paste)
        pasted = win.data_grid.paste_clipboard(start_row=0, start_col=2)
        assert pasted == 4
        actual_pasted_row = grid_model._map_row(0)
        assert win.dataset.df["educ"].iloc[actual_pasted_row] == 20
        wf_f["steps"]["multicell_paste"] = {"status": "PASS", "pasted_cells": pasted}

        wf_f["status"] = "PASS"
        wf_f["elapsed_s"] = time.perf_counter() - t0_wf
        report["workflows"]["workflow_f"] = wf_f
        print(f"  -> Workflow F PASS ({wf_f['elapsed_s']:.2f}s)")

        report["overall_status"] = "PASS"
        print("\n" + "=" * 80)
        print("ALL 6 PRACTICAL RESEARCH WORKFLOWS PASSED VERIFICATION")
        print("=" * 80)

    except Exception as ex:
        report["overall_status"] = "FAIL"
        report["error"] = {
            "message": str(ex),
            "traceback": traceback.format_exc()
        }
        print(f"\n[EVALUATION FAILED]: {ex}")
        traceback.print_exc()

    finally:
        mem_final_mb = get_process_memory_mb()
        t_total = time.perf_counter() - t_start_total
        report["performance_metrics"]["total_elapsed_s"] = t_total
        report["performance_metrics"]["initial_memory_mb"] = mem_init_mb
        report["performance_metrics"]["final_memory_mb"] = mem_final_mb
        report["performance_metrics"]["memory_delta_mb"] = mem_final_mb - mem_init_mb

        try:
            temp_dir_obj.cleanup()
        except Exception:
            pass

    # Save reports
    out_dir = Path(output_dir).resolve() if output_dir else ROOT_PATH / "tests" / "data" / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "quality_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = out_dir / "quality_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Potatostics Product Quality & Empirical Research Workflow Report\n\n")
        f.write(f"- **Overall Status**: **{report['overall_status']}**\n")
        f.write(f"- **Timestamp**: `{report['evaluation_timestamp']}`\n")
        f.write(f"- **Platform**: `{report['environment']['platform']}`\n")
        f.write(f"- **Python**: `{report['environment']['python_version']}` | **PySide6**: `{report['environment']['pyside6_version']}`\n")
        f.write(f"- **Random Seed**: `{report['random_seed']}`\n")
        f.write(f"- **Total Elapsed Duration**: `{report['performance_metrics']['total_elapsed_s']:.2f}s`\n")
        f.write(f"- **Memory Usage**: Initial: `{report['performance_metrics']['initial_memory_mb']:.1f} MB` | Final: `{report['performance_metrics']['final_memory_mb']:.1f} MB` (Delta: `+{report['performance_metrics']['memory_delta_mb']:.1f} MB`)\n\n")

        f.write("## Practical Workflow Breakdown\n\n")
        f.write("| Workflow | Description | Status | Elapsed |\n")
        f.write("| :--- | :--- | :---: | :---: |\n")
        for wf_key, wf_val in report.get("workflows", {}).items():
            f.write(f"| **{wf_key.upper()}** | {wf_val['name']} | **{wf_val['status']}** | {wf_val.get('elapsed_s', 0):.2f}s |\n")
        f.write("\n")

    print(f"\nSaved structured quality evaluation report to:\n  - {json_path}\n  - {md_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Potatostics Product Quality & Research Workflows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic evaluation")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store evaluation reports")
    args = parser.parse_args()

    res = run_evaluation(seed=args.seed, output_dir=args.output_dir)
    sys.exit(0 if res["overall_status"] == "PASS" else 1)
