"""
Potatostics — Release Acceptance Pipeline
Executes the complete 12-stage release verification sequence:
1. Source Compilation
2. Module Import Smoke Tests
3. Scientific Regression Tests
4. Complete Automated Test Suite
5. GUI Integration Tests
6. Econometric Benchmark Suite
7. Clean Source-Release Generation
8. Independent Archive Verification
9. Wheel Build
10. Isolated Wheel Installation & Unpack Check
11. Installed-Package Resource Checks
12. Windows Executable Verification

Outputs a unified acceptance matrix with status: PASS, FAIL, SKIPPED, NOT VERIFIED.
"""

from __future__ import annotations

import os
import sys
import re
import time
import json
import hashlib
import py_compile
import platform
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

# Ensure repository root is on sys.path
ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))


from enum import Enum


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"
    NOT_VERIFIED = "NOT VERIFIED"


VALID_GATE_STATUSES = {s.value for s in GateStatus}


class ReleaseGateResult:
    def __init__(
        self,
        gate_num: int,
        name: str,
        status: str,
        details: str = "",
        elapsed_s: float = 0.0,
        is_clean_install: bool = True,
        build_provenance: Optional[Dict[str, Any]] = None
    ):
        self.gate_num = gate_num
        self.name = name
        self.status = status.value if isinstance(status, GateStatus) else str(status)
        self.details = details
        self.elapsed_s = elapsed_s
        self.is_clean_install = is_clean_install
        self.build_provenance = build_provenance


def extract_release_version(root_path: Path) -> str:
    """Extracts project version dynamically from pyproject.toml."""
    pyproject = root_path / "pyproject.toml"
    if pyproject.exists():
        try:
            content = pyproject.read_text(encoding="utf-8")
            m = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if m:
                return m.group(1).strip()
        except Exception:
            pass
    return "1.0.0"


def compute_source_tree_fingerprint(root_path: Path, manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Computes a content-based SHA-256 fingerprint across all approved release files.
    Normalizes relative paths, reads exact file bytes and sizes, and builds a
    deterministic source-tree content digest.
    """
    if manifest_path is None:
        manifest_path = root_path / "release_manifest.json"

    if not manifest_path.exists():
        return {
            "source_tree_content_sha256": "missing_manifest",
            "manifest_policy_sha256": "missing_manifest",
            "total_files": 0,
            "total_bytes": 0,
            "git_commit": "unknown",
            "git_is_dirty": False
        }

    manifest_bytes = manifest_path.read_bytes()
    manifest_policy_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_data = json.loads(manifest_bytes.decode("utf-8"))
    approved_files = manifest_data.get("approved_files", [])

    file_entries = []
    total_bytes = 0

    for rel_path in sorted(approved_files):
        norm_path = rel_path.replace("\\", "/")
        full_path = root_path / norm_path
        if full_path.exists() and full_path.is_file():
            data = full_path.read_bytes()
            size = len(data)
            total_bytes += size
            f_sha = hashlib.sha256(data).hexdigest()
            file_entries.append((norm_path, size, f_sha))
        else:
            file_entries.append((norm_path, -1, "missing"))

    hasher = hashlib.sha256()
    for norm_path, size, f_sha in file_entries:
        hasher.update(f"{norm_path}:{size}:{f_sha}\n".encode("utf-8"))
    source_tree_content_sha256 = hasher.hexdigest()

    git_commit = "unknown"
    git_is_dirty = False
    try:
        proc_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root_path), capture_output=True, text=True)
        if proc_rev.returncode == 0:
            git_commit = proc_rev.stdout.strip()
        proc_diff = subprocess.run(["git", "status", "--porcelain"], cwd=str(root_path), capture_output=True, text=True)
        if proc_diff.returncode == 0:
            git_is_dirty = len(proc_diff.stdout.strip()) > 0
    except Exception:
        pass

    return {
        "source_tree_content_sha256": source_tree_content_sha256,
        "manifest_policy_sha256": manifest_policy_sha256,
        "total_files": len(approved_files),
        "total_bytes": total_bytes,
        "git_commit": git_commit,
        "git_is_dirty": git_is_dirty
    }


def evaluate_release_readiness(
    results: Union[List[ReleaseGateResult], Dict[int, ReleaseGateResult]],
    target: str = "source",
    install_mode: str = "auto",
    build_provenance: Optional[Dict[str, Any]] = None,
    gui_smoke_executed: bool = True
) -> Dict[str, Any]:
    """
    Pure evaluation function assessing overall release readiness across all 12 gates.
    Enforces strict release policy:
    - Target 'source': Gates 1-11 mandatory. Gate 10 must be clean Mode A.
    - Target 'windows': Gates 1-12 mandatory. Gate 12 must have valid provenance and gui_smoke.
    - Target 'all': Gates 1-12 mandatory.
    - Status MUST strictly equal 'PASS' for all mandatory gates.
    - Unrecognized, pending, or malformed statuses immediately reject release.
    - Duplicate gate numbers reject release.
    - Mode B preinstalled dependencies cannot substitute for clean install.
    - Disabling the GUI smoke test cannot certify a Windows desktop release.
    """
    reasons_for_rejection: List[str] = []
    conditional_checks: List[str] = []

    # 1. Structure validation & duplicate gate number detection
    gates_dict: Dict[int, ReleaseGateResult] = {}
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, ReleaseGateResult) or not hasattr(item, "gate_num") or not hasattr(item, "status"):
                reasons_for_rejection.append(f"Malformed gate record: expected ReleaseGateResult, got {type(item).__name__}")
                continue
            if item.gate_num in gates_dict:
                reasons_for_rejection.append(f"Duplicate gate number detected: Gate {item.gate_num} defined more than once")
            gates_dict[item.gate_num] = item
    elif isinstance(results, dict):
        for k, item in results.items():
            if not isinstance(item, ReleaseGateResult) or not hasattr(item, "gate_num") or not hasattr(item, "status"):
                reasons_for_rejection.append(f"Malformed gate record for key {k}: expected ReleaseGateResult, got {type(item).__name__}")
                continue
            if item.gate_num != k:
                reasons_for_rejection.append(f"Gate number mismatch: key {k} != gate_num {item.gate_num}")
            gates_dict[item.gate_num] = item
    else:
        reasons_for_rejection.append(f"Invalid results container: expected list or dict, got {type(results).__name__}")

    if target == "source":
        mandatory_gate_nums = list(range(1, 12))
    elif target in ("windows", "all"):
        mandatory_gate_nums = list(range(1, 13))
    else:
        mandatory_gate_nums = list(range(1, 13))

    # 2. Strict status schema validation on all submitted gates
    for g_num, r in gates_dict.items():
        if r.status not in VALID_GATE_STATUSES:
            reasons_for_rejection.append(f"Gate {g_num} ({r.name}) has unrecognized or invalid status: '{r.status}'")

    # 3. Mandatory gate PASS requirement
    for g_num in mandatory_gate_nums:
        if g_num not in gates_dict:
            reasons_for_rejection.append(f"Mandatory Gate {g_num} was not executed")
            continue
        gate_res = gates_dict[g_num]
        if gate_res.status != GateStatus.PASS.value:
            if gate_res.status in (GateStatus.FAIL.value, GateStatus.ERROR.value):
                reasons_for_rejection.append(f"Mandatory Gate {g_num} ({gate_res.name}) FAILED: {gate_res.details}")
            elif gate_res.status == GateStatus.NOT_VERIFIED.value:
                reasons_for_rejection.append(f"Mandatory Gate {g_num} ({gate_res.name}) NOT VERIFIED: {gate_res.details}")
            elif gate_res.status == GateStatus.SKIPPED.value:
                reasons_for_rejection.append(f"Mandatory Gate {g_num} ({gate_res.name}) was SKIPPED, but is required for target '{target}'")
            else:
                reasons_for_rejection.append(f"Mandatory Gate {g_num} ({gate_res.name}) did not PASS (status='{gate_res.status}')")

    # 4. Gate 10 Mode A Clean Installation Policy
    if 10 in gates_dict:
        g10 = gates_dict[10]
        is_clean = getattr(g10, "is_clean_install", True) and ("Mode B" not in g10.details) and ("Clean dependency resolution NOT validated" not in g10.details)
        if not is_clean:
            conditional_checks.append("Gate 10 used Mode B (preinstalled host dependencies; clean dependency resolution not verified)")
            if install_mode == "isolated" or target in ("source", "windows", "all"):
                reasons_for_rejection.append(
                    "Gate 10 verified preinstalled dependencies only (Mode B); "
                    "certified distributable release requires clean isolated dependency installation (Mode A)"
                )

    # 5. Windows Executable & GUI Smoke Policy
    if target in ("windows", "all"):
        if not gui_smoke_executed:
            conditional_checks.append("GUI smoke test skipped via --no-gui-smoke")
            reasons_for_rejection.append("GUI smoke test was disabled (--no-gui-smoke); cannot certify Windows desktop release without full desktop GUI smoke test")

        if build_provenance:
            if build_provenance.get("is_stale"):
                reasons_for_rejection.append(f"Windows executable is STALE: {build_provenance.get('reason')}")
            if build_provenance.get("unverified_provenance"):
                reasons_for_rejection.append(f"Windows executable provenance unverified: {build_provenance.get('reason')}")

    passed_gates = [g for g, r in gates_dict.items() if r.status == GateStatus.PASS.value]
    failed_gates = [g for g, r in gates_dict.items() if r.status in (GateStatus.FAIL.value, GateStatus.ERROR.value)]
    unverified_gates = [g for g, r in gates_dict.items() if r.status == GateStatus.NOT_VERIFIED.value]
    skipped_gates = [g for g, r in gates_dict.items() if r.status == GateStatus.SKIPPED.value]

    is_ready = len(reasons_for_rejection) == 0
    overall_status = "RELEASE READY" if is_ready else "RELEASE BLOCKED"

    return {
        "overall_status": overall_status,
        "target": target,
        "mandatory_gates": mandatory_gate_nums,
        "passed_gates": passed_gates,
        "failed_gates": failed_gates,
        "unverified_gates": unverified_gates,
        "skipped_gates": skipped_gates,
        "conditional_checks": conditional_checks,
        "reasons_for_rejection": reasons_for_rejection,
        "is_ready": is_ready
    }


def run_gate_01_compilation(root_path: Path) -> ReleaseGateResult:
    t0 = time.perf_counter()
    compiled_count = 0
    errors = []
    for p in root_path.rglob("*.py"):
        if any(part in (".git", "__pycache__", ".pytest_cache", "dist", "build", ".venv", ".vscode") for part in p.parts):
            continue
        try:
            py_compile.compile(str(p), doraise=True)
            compiled_count += 1
        except Exception as ex:
            errors.append(f"{p.name}: {ex}")

    t = time.perf_counter() - t0
    if errors:
        return ReleaseGateResult(1, "Source Compilation", "FAIL", f"{len(errors)} compilation errors: {'; '.join(errors[:3])}", t)
    return ReleaseGateResult(1, "Source Compilation", "PASS", f"Compiled {compiled_count} Python source files without errors", t)


def run_gate_02_import_smoke(root_path: Path) -> ReleaseGateResult:
    t0 = time.perf_counter()
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_stage45_module_import_smoke.py"]
    res = subprocess.run(cmd, cwd=str(root_path), capture_output=True, text=True)
    t = time.perf_counter() - t0
    if res.returncode == 0:
        match = re.search(r"(\d+)\s+passed", res.stdout)
        count_str = f"{match.group(1)} modules" if match else "All modules"
        return ReleaseGateResult(2, "Module Import Smoke Tests", "PASS", f"{count_str} imported cleanly without missing symbols or syntax errors", t)
    return ReleaseGateResult(2, "Module Import Smoke Tests", "FAIL", res.stderr[-500:] or res.stdout[-500:], t)


def run_gate_03_scientific_regressions(root_path: Path) -> ReleaseGateResult:
    t0 = time.perf_counter()
    cmd = [
        sys.executable, "-m", "unittest",
        "tests/test_stage43_ordered_logit_inference.py",
        "tests/test_stage46_benchmark_alias.py"
    ]
    res = subprocess.run(cmd, cwd=str(root_path), capture_output=True, text=True)
    t = time.perf_counter() - t0
    if res.returncode == 0:
        match = re.search(r"Ran\s+(\d+)\s+tests?", res.stderr or res.stdout)
        count_str = f"{match.group(1)} regression tests" if match else "Targeted regression tests"
        return ReleaseGateResult(3, "Scientific Regression Tests", "PASS", f"{count_str} passed: Ordered logit cutpoint Delta method & custom aliases verified", t)
    return ReleaseGateResult(3, "Scientific Regression Tests", "FAIL", res.stderr[-500:], t)


def run_gate_04_full_test_suite(root_path: Path) -> ReleaseGateResult:
    t0 = time.perf_counter()
    cmd = [sys.executable, "-m", "pytest", "-q"]
    res = subprocess.run(cmd, cwd=str(root_path), capture_output=True, text=True)
    t = time.perf_counter() - t0
    last_line = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
    if res.returncode == 0:
        return ReleaseGateResult(4, "Complete Automated Test Suite", "PASS", f"Pytest: {last_line}", t)
    return ReleaseGateResult(4, "Complete Automated Test Suite", "FAIL", f"Pytest failed: {last_line}\n{res.stderr[-400:]}", t)


def run_gate_05_gui_integration(root_path: Path) -> ReleaseGateResult:
    t0 = time.perf_counter()
    pyside6_ver = "unknown"
    try:
        import PySide6
        pyside6_ver = getattr(PySide6, "__version__", "installed")
    except Exception:
        pyside6_ver = "not installed"

    platform_str = f"{platform.system()} {platform.release()}"
    qpa_plugin = os.environ.get("QT_QPA_PLATFORM", "offscreen")

    cmd = [
        sys.executable, "-m", "pytest", "-v",
        "tests/test_stage46_gui_reliability.py",
        "tests/test_stage45_research_workflow.py",
        "tests/test_stage49_gui_workflows.py",
        "tests/test_stage51_worker_concurrency.py"
    ]
    res = subprocess.run(cmd, cwd=str(root_path), capture_output=True, text=True)
    t = time.perf_counter() - t0

    # Parse individual test results
    passed_tests = set()
    skipped_tests = set()
    failed_tests = set()

    for line in res.stdout.splitlines():
        line_clean = line.strip()
        m_test = re.match(r"^(\S+::\S+)\s+(PASSED|SKIPPED|FAILED|ERROR)", line_clean)
        if m_test:
            test_id = m_test.group(1)
            outcome = m_test.group(2)
            if outcome == "PASSED":
                passed_tests.add(test_id)
            elif outcome == "SKIPPED":
                skipped_tests.add(test_id)
            elif outcome in ("FAILED", "ERROR"):
                failed_tests.add(test_id)

    last_line = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
    passed_m = re.search(r"(\d+)\s+passed", res.stdout)
    skipped_m = re.search(r"(\d+)\s+skipped", res.stdout)
    failed_m = re.search(r"(\d+)\s+failed", res.stdout)
    errors_m = re.search(r"(\d+)\s+error", res.stdout)

    passed_cnt = int(passed_m.group(1)) if passed_m else len(passed_tests)
    skipped_cnt = int(skipped_m.group(1)) if skipped_m else len(skipped_tests)
    failed_cnt = int(failed_m.group(1)) if failed_m else len(failed_tests)
    errors_cnt = int(errors_m.group(1)) if errors_m else 0
    total_cnt = passed_cnt + skipped_cnt + failed_cnt + errors_cnt

    if res.returncode != 0 or failed_cnt > 0 or errors_cnt > 0:
        return ReleaseGateResult(5, "GUI Integration Tests", "FAIL", f"GUI test suite failed ({last_line})\n{res.stderr[-400:]}", t)

    if passed_cnt == 0:
        return ReleaseGateResult(5, "GUI Integration Tests", "FAIL", f"All GUI tests were skipped ({skipped_cnt} skipped). Mandatory GUI verification requires at least one passed GUI test.", t)

    # Enforce mandatory GUI test groups
    mandatory_gui_groups = {
        "1. Application startup": [
            "test_01_window_startup_and_initial_state",
            "test_01_gui_startup_lifecycle_and_theming",
            "test_02_gui_startup_with_initial_dataset",
        ],
        "2. Data preparation": [
            "test_02_dataset_loading_and_data_grid_refresh",
            "test_03_complete_data_manipulation_lifecycle",
            "test_workflow_a_data_preparation",
        ],
        "3. Statistical analysis": [
            "test_06_regression_result_display_and_console",
            "test_04_full_regression_and_post_estimation_workflow",
            "test_workflow_b_statistical_analysis",
        ],
        "4. Do-file execution": [
            "test_06_macros_and_data_transformation_commands",
            "test_workflow_c_script_execution",
        ],
        "5. Session persistence": [
            "test_09_session_reopening_in_gui",
            "test_03b_dataset_multiformat_save_reopen_fidelity",
            "test_workflow_d_session_persistence",
        ],
        "6. Worker cancellation": [
            "test_08_safe_cancellation",
            "test_workflow_e_failure_recovery_and_worker_safety",
        ],
        "7. Error recovery": [
            "test_07_error_handling_user_friendly_message",
            "test_workflow_e_failure_recovery_and_worker_safety",
        ],
        "8. GUI shutdown": [
            "test_10_clean_application_shutdown",
        ],
        "9. Concurrency safety": [
            "test_03_unexpected_python_exception_handling",
            "test_04_cooperative_cancellation",
            "test_05_non_cooperative_timeout_and_late_completion",
            "test_10_concurrent_session_save_is_blocked_during_execution",
            "test_13_concurrent_data_grid_edit_is_blocked",
        ],
    }

    missing_groups = []
    for grp_name, cand_patterns in mandatory_gui_groups.items():
        has_pass = any(
            any(pat in t_id for pat in cand_patterns)
            for t_id in passed_tests
        )
        if not has_pass:
            missing_groups.append(grp_name)

    if missing_groups:
        return ReleaseGateResult(
            5, "GUI Integration Tests", "FAIL",
            f"Mandatory GUI test groups were skipped or missing: {', '.join(missing_groups)} "
            f"({passed_cnt} passed, {skipped_cnt} skipped of {total_cnt} tests).",
            t
        )

    # Also run the noninteractive GUI smoke test against the live source
    cmd_smoke = [sys.executable, "main.py", "--gui-smoke"]
    res_smoke = subprocess.run(cmd_smoke, cwd=str(root_path), capture_output=True, text=True)
    if res_smoke.returncode != 0 or "GUI_SMOKE_TEST_PASSED" not in res_smoke.stdout:
        return ReleaseGateResult(5, "GUI Integration Tests", "FAIL", f"GUI smoke test failed: {res_smoke.stderr or res_smoke.stdout}", time.perf_counter() - t0)

    # Run deterministic researcher acceptance workflows (Headless)
    cmd_researcher = [sys.executable, "scripts/verify_researcher_workflows.py", "--headless"]
    res_researcher = subprocess.run(cmd_researcher, cwd=str(root_path), capture_output=True, text=True)
    if res_researcher.returncode != 0:
        return ReleaseGateResult(5, "GUI Integration Tests", "FAIL", f"Researcher workflows failed:\n{res_researcher.stderr or res_researcher.stdout}", time.perf_counter() - t0)

    t_total = time.perf_counter() - t0
    details = (
        f"{passed_cnt} passed, {skipped_cnt} skipped of {total_cnt} tests "
        f"(all 9 mandatory GUI groups verified) "
        f"[PySide6={pyside6_ver}, OS={platform_str}, QPA={qpa_plugin}] "
        f"+ live desktop smoke test verified + 6 researcher workflows verified (Headless)"
    )
    return ReleaseGateResult(5, "GUI Integration Tests", "PASS", details, t_total)


def run_gate_06_benchmarks(root_path: Path) -> ReleaseGateResult:
    t0 = time.perf_counter()
    try:
        from engine.benchmarks import BenchmarkRunner
        runner = BenchmarkRunner()
        runner.run_all()
        passed = sum(1 for b in runner.results if b["status"] == "PASS")
        total = len(runner.results)
        t = time.perf_counter() - t0
        if passed == total and total == 19:
            return ReleaseGateResult(6, "Econometric Benchmark Suite", "PASS", f"All {passed}/{total} benchmarks PASS with 0 fabricated records", t)
        return ReleaseGateResult(6, "Econometric Benchmark Suite", "FAIL", f"{passed}/{total} passed", t)
    except Exception as ex:
        t = time.perf_counter() - t0
        return ReleaseGateResult(6, "Econometric Benchmark Suite", "FAIL", str(ex), t)


def run_gate_07_source_release_generation(root_path: Path) -> Tuple[ReleaseGateResult, Optional[str]]:
    t0 = time.perf_counter()
    try:
        from scripts.create_clean_release_archive import create_release_archive
        archive_path = create_release_archive(root_dir=str(root_path), strict=True)
        t = time.perf_counter() - t0
        size_kb = os.path.getsize(archive_path) / 1024
        return ReleaseGateResult(7, "Clean Source-Release Generation", "PASS", f"Created {Path(archive_path).name} ({size_kb:.1f} KB)", t), archive_path
    except Exception as ex:
        t = time.perf_counter() - t0
        return ReleaseGateResult(7, "Clean Source-Release Generation", "FAIL", str(ex), t), None


def run_gate_08_archive_verification(archive_path: Optional[str], root_path: Path) -> ReleaseGateResult:
    t0 = time.perf_counter()
    if not archive_path or not os.path.exists(archive_path):
        return ReleaseGateResult(8, "Independent Archive Verification", "FAIL", "Archive not available from Gate 7", 0.0)

    try:
        from scripts.create_clean_release_archive import verify_release_archive
        report = verify_release_archive(archive_path, root_path=root_path, test_extraction=True)
        t = time.perf_counter() - t0
        if report["verified"]:
            return ReleaseGateResult(8, "Independent Archive Verification", "PASS", f"Verified {report['total_files']} files, SHA-256 hashes & extraction", t)
        return ReleaseGateResult(8, "Independent Archive Verification", "FAIL", f"Mismatches: {'; '.join(report.get('mismatches', []))}", t)
    except Exception as ex:
        t = time.perf_counter() - t0
        return ReleaseGateResult(8, "Independent Archive Verification", "FAIL", str(ex), t)


def run_gate_09_wheel_build(root_path: Path) -> Tuple[ReleaseGateResult, Optional[Path]]:
    t0 = time.perf_counter()
    dist_dir = root_path / "dist"
    dist_dir.mkdir(exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(dist_dir), str(root_path)]
    res = subprocess.run(cmd, cwd=str(root_path), capture_output=True, text=True)
    t = time.perf_counter() - t0

    wheels = list(dist_dir.glob("potatostics-*.whl"))
    if res.returncode == 0 and wheels:
        latest_wheel = max(wheels, key=lambda p: p.stat().st_mtime)
        return ReleaseGateResult(9, "Wheel Build", "PASS", f"Built {latest_wheel.name} ({latest_wheel.stat().st_size / 1024:.1f} KB)", t), latest_wheel
    return ReleaseGateResult(9, "Wheel Build", "FAIL", res.stderr[-500:], t), None


def run_gate_10_isolated_wheel_install(
    wheel_path: Optional[Union[Path, str]],
    install_mode: str = "auto",
    wheelhouse: Optional[Union[Path, str]] = None
) -> Tuple[ReleaseGateResult, Optional[tempfile.TemporaryDirectory], Optional[Path]]:
    t0 = time.perf_counter()
    if not wheel_path:
        return ReleaseGateResult(10, "Isolated Wheel Installation & Site-Packages Provenance", "FAIL", "Wheel not found", 0.0, is_clean_install=False), None, None

    # Problem C: Normalize relative wheel paths to absolute path before invoking pip from temporary directories
    wheel_path = Path(wheel_path).resolve()
    if not wheel_path.exists():
        return ReleaseGateResult(10, "Isolated Wheel Installation & Site-Packages Provenance", "FAIL", f"Wheel not found at {wheel_path}", 0.0, is_clean_install=False), None, None

    if wheelhouse:
        wheelhouse = Path(wheelhouse).resolve()

    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = Path(temp_dir_obj.name).resolve()
    venv_dir = temp_dir / "test_venv"

    use_system_site_packages = (install_mode == "preinstalled")
    is_mode_a = (install_mode != "preinstalled")

    try:
        # Step 1: Create isolated virtual environment
        venv_cmd = [sys.executable, "-m", "venv"]
        if use_system_site_packages:
            venv_cmd.append("--system-site-packages")
        venv_cmd.append(str(venv_dir))

        res_venv = subprocess.run(venv_cmd, cwd=str(temp_dir), capture_output=True, text=True)
        if res_venv.returncode != 0:
            t = time.perf_counter() - t0
            temp_dir_obj.cleanup()
            return ReleaseGateResult(10, "Isolated Wheel Installation & Site-Packages Provenance", "FAIL", f"venv creation failed: {res_venv.stderr}", t, is_clean_install=False), None, None

        venv_python = venv_dir / "Scripts" / "python.exe" if platform.system() == "Windows" else venv_dir / "bin" / "python"
        venv_bin_dir = venv_dir / "Scripts" if platform.system() == "Windows" else venv_dir / "bin"
        if not venv_python.exists():
            t = time.perf_counter() - t0
            temp_dir_obj.cleanup()
            return ReleaseGateResult(10, "Isolated Wheel Installation & Site-Packages Provenance", "FAIL", f"venv python not found: {venv_python}", t, is_clean_install=False), None, None

        # Step 2: Install wheel into the venv (copy wheel locally to avoid non-ASCII path encoding issues)
        import shutil
        local_wheel = temp_dir / wheel_path.name
        shutil.copy2(wheel_path, local_wheel)

        pip_cmd = [str(venv_python), "-m", "pip", "install"]
        if is_mode_a:
            if wheelhouse:
                pip_cmd.extend(["--no-index", "--find-links", str(wheelhouse)])
            pip_cmd.append(local_wheel.name)
        else:
            pip_cmd.extend(["--no-deps", local_wheel.name])

        res_pip = subprocess.run(pip_cmd, cwd=str(temp_dir), capture_output=True, text=True)
        if res_pip.returncode != 0:
            # In auto mode, fallback to Mode B if isolated pip install fails (e.g. offline environment without wheelhouse)
            if install_mode == "auto":
                try:
                    temp_dir_obj.cleanup()
                except Exception:
                    pass
                temp_dir_obj = tempfile.TemporaryDirectory()
                temp_dir = Path(temp_dir_obj.name).resolve()
                local_wheel = temp_dir / wheel_path.name
                shutil.copy2(wheel_path, local_wheel)
                venv_dir = temp_dir / "test_venv"
                subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)], cwd=str(temp_dir), check=True)
                venv_python = venv_dir / "Scripts" / "python.exe" if platform.system() == "Windows" else venv_dir / "bin" / "python"
                venv_bin_dir = venv_dir / "Scripts" if platform.system() == "Windows" else venv_dir / "bin"
                res_pip_b = subprocess.run([str(venv_python), "-m", "pip", "install", "--no-deps", local_wheel.name], cwd=str(temp_dir), capture_output=True, text=True)
                if res_pip_b.returncode != 0:
                    t = time.perf_counter() - t0
                    temp_dir_obj.cleanup()
                    return ReleaseGateResult(10, "Isolated Wheel Installation & Site-Packages Provenance", "FAIL", f"Mode B pip install failed: {res_pip_b.stderr}", t, is_clean_install=False), None, None
                is_mode_a = False
            else:
                t = time.perf_counter() - t0
                temp_dir_obj.cleanup()
                return ReleaseGateResult(10, "Isolated Wheel Installation & Site-Packages Provenance", "FAIL", f"Mode A pip install failed: {res_pip.stderr}. (Tip: specify --wheelhouse <dir> or use --install-mode preinstalled for offline environments)", t, is_clean_install=False), None, None

        # Step 3: Run pip check (strictly required for Mode A; informational for Mode B)
        res_check_deps = subprocess.run([str(venv_python), "-m", "pip", "check"], cwd=str(temp_dir), capture_output=True, text=True)
        if is_mode_a and res_check_deps.returncode != 0:
            t = time.perf_counter() - t0
            temp_dir_obj.cleanup()
            return ReleaseGateResult(10, "Isolated Wheel Installation & Site-Packages Provenance", "FAIL", f"Mode A pip check dependency conflict: {res_check_deps.stdout or res_check_deps.stderr}", t, is_clean_install=False), None, None

        # Step 4: Execute module provenance and import verification strictly from outside repo
        check_script = (
            "import sys, os\n"
            "from pathlib import Path\n"
            "import engine\n"
            "import gui_qt\n"
            "engine_file = Path(engine.__file__).resolve()\n"
            "gui_qt_file = Path(gui_qt.__file__).resolve()\n"
            "venv_root = Path(sys.prefix).resolve()\n"
            "if not (engine_file.is_relative_to(venv_root) and gui_qt_file.is_relative_to(venv_root)):\n"
            "    sys.stderr.write(f'Provenance check failed: engine={engine_file}, venv={venv_root}\\n')\n"
            "    sys.exit(1)\n"
            "from engine.dataset import Dataset\n"
            "from engine.interpreter import Interpreter\n"
            "from gui_qt.main_window import PotatosticsQtWindow\n"
            "ds = Dataset()\n"
            "interp = Interpreter(ds)\n"
            "print('PROVENANCE_OK')\n"
        )
        res_check = subprocess.run(
            [str(venv_python), "-c", check_script],
            cwd=str(temp_dir),
            capture_output=True,
            text=True
        )
        if res_check.returncode != 0:
            t = time.perf_counter() - t0
            temp_dir_obj.cleanup()
            return ReleaseGateResult(10, "Isolated Wheel Installation & Site-Packages Provenance", "FAIL", f"Provenance check failed: {res_check.stderr}", t, is_clean_install=False), None, None

        # Step 5: Check console entry point in venv (Problem B: missing or failing entry point causes FAIL)
        potatostics_bin = venv_bin_dir / ("potatostics.exe" if platform.system() == "Windows" else "potatostics")
        if not potatostics_bin.exists():
            t = time.perf_counter() - t0
            temp_dir_obj.cleanup()
            return ReleaseGateResult(10, "Isolated Wheel Installation & Site-Packages Provenance", "FAIL", f"Console entry point executable missing: {potatostics_bin.name}", t, is_clean_install=False), None, None

        expected_ver = extract_release_version(ROOT_PATH)
        res_entry = subprocess.run([str(potatostics_bin), "--version"], cwd=str(temp_dir), capture_output=True, text=True, timeout=15)
        if res_entry.returncode != 0:
            t = time.perf_counter() - t0
            temp_dir_obj.cleanup()
            return ReleaseGateResult(10, "Isolated Wheel Installation & Site-Packages Provenance", "FAIL", f"Console entry point failed (code {res_entry.returncode}): {res_entry.stderr or res_entry.stdout}", t, is_clean_install=False), None, None

        if expected_ver not in res_entry.stdout:
            t = time.perf_counter() - t0
            temp_dir_obj.cleanup()
            return ReleaseGateResult(10, "Isolated Wheel Installation & Site-Packages Provenance", "FAIL", f"Console entry point version mismatch: expected '{expected_ver}', got '{res_entry.stdout.strip()}'", t, is_clean_install=False), None, None

        try:
            if not potatostics_bin.resolve().is_relative_to(venv_dir.resolve()):
                t = time.perf_counter() - t0
                temp_dir_obj.cleanup()
                return ReleaseGateResult(10, "Isolated Wheel Installation & Site-Packages Provenance", "FAIL", f"Entry point binary {potatostics_bin} not inside venv {venv_dir}", t, is_clean_install=False), None, None
        except Exception:
            pass

        t = time.perf_counter() - t0
        if is_mode_a:
            detail_msg = f"Mode A (Clean Isolated Install): Installed with declared runtime dependencies; pip check passed; site-packages provenance verified; console entry=True (v{expected_ver})"
        else:
            detail_msg = f"Mode B (Preinstalled Dependencies): Wheel installed with --no-deps into host-shared venv; site-packages provenance verified; console entry=True (v{expected_ver}) (Clean dependency resolution NOT validated)"

        return ReleaseGateResult(10, "Isolated Wheel Installation & Site-Packages Provenance", "PASS", detail_msg, t, is_clean_install=is_mode_a), temp_dir_obj, venv_python
    except Exception as ex:
        t = time.perf_counter() - t0
        try:
            temp_dir_obj.cleanup()
        except Exception:
            pass
        return ReleaseGateResult(10, "Isolated Wheel Installation & Site-Packages Provenance", "FAIL", str(ex), t, is_clean_install=False), None, None


def run_gate_11_installed_package_resources(venv_python: Optional[Path], temp_dir_obj: Optional[tempfile.TemporaryDirectory]) -> ReleaseGateResult:
    t0 = time.perf_counter()
    if not venv_python or not temp_dir_obj:
        return ReleaseGateResult(11, "Installed-Package Resource Loading", "FAIL", "Isolated virtual environment not available from Gate 10", 0.0)

    try:
        temp_dir = Path(temp_dir_obj.name).resolve()
        resource_script = (
            "import sys\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import engine\n"
            "import gui_qt\n"
            "engine_dir = Path(engine.__file__).resolve().parent\n"
            "gui_qt_dir = Path(gui_qt.__file__).resolve().parent\n"
            "auto_path = engine_dir / 'assets' / 'data' / 'auto.dta'\n"
            "basque_path = engine_dir / 'assets' / 'data' / 'basque.csv'\n"
            "mpdta_path = engine_dir / 'assets' / 'data' / 'mpdta.csv'\n"
            "sample_path = gui_qt_dir / 'sample_data.csv'\n"
            "for p in (auto_path, basque_path, mpdta_path, sample_path):\n"
            "    if not p.is_file():\n"
            "        sys.stderr.write(f'Missing installed asset: {p}\\n')\n"
            "        sys.exit(1)\n"
            "from engine.dataset import Dataset\n"
            "_ds_auto = Dataset()\n"
            "_ds_auto.load_dta(str(auto_path))\n"
            "df_auto = _ds_auto.df\n"
            "df_basque = pd.read_csv(basque_path)\n"
            "df_mpdta = pd.read_csv(mpdta_path)\n"
            "df_sample = pd.read_csv(sample_path)\n"
            "if len(df_auto) == 0 or len(df_basque) == 0 or len(df_mpdta) == 0 or len(df_sample) == 0:\n"
            "    sys.stderr.write('One or more installed datasets loaded empty\\n')\n"
            "    sys.exit(1)\n"
            "print(f'auto:{len(df_auto)}, basque:{len(df_basque)}, mpdta:{len(df_mpdta)}, sample:{len(df_sample)}')\n"
        )
        res_resource = subprocess.run(
            [str(venv_python), "-c", resource_script],
            cwd=str(temp_dir),
            capture_output=True,
            text=True
        )
        t = time.perf_counter() - t0
        if res_resource.returncode != 0:
            return ReleaseGateResult(11, "Installed-Package Resource Loading", "FAIL", res_resource.stderr[-500:], t)
        return ReleaseGateResult(11, "Installed-Package Resource Loading", "PASS", f"Successfully loaded installed assets: {res_resource.stdout.strip()}", t)
    except Exception as ex:
        t = time.perf_counter() - t0
        return ReleaseGateResult(11, "Installed-Package Resource Loading", "FAIL", str(ex), t)


def run_gate_12_windows_executable(root_path: Path, target: str = "source", build_exe: bool = False, gui_smoke: bool = True) -> ReleaseGateResult:
    t0 = time.perf_counter()
    if platform.system() != "Windows":
        if target in ("windows", "all"):
            return ReleaseGateResult(12, "Windows Executable Build & Smoke Test", "FAIL", "Target requires Windows executable, but running on non-Windows OS", 0.0)
        return ReleaseGateResult(12, "Windows Executable Build & Smoke Test", "SKIPPED", "Non-Windows platform", 0.0)

    if target == "source" and not build_exe:
        return ReleaseGateResult(12, "Windows Executable Build & Smoke Test", "SKIPPED", "Target is 'source'; Windows executable build not required for source distribution", 0.0)

    exe_path = root_path / "dist" / "Potatostics" / "Potatostics.exe"
    meta_path = root_path / "dist" / "Potatostics" / "build_metadata.json"
    manifest_file = root_path / "release_manifest.json"
    fingerprint = compute_source_tree_fingerprint(root_path, manifest_file)
    current_tree_content_sha = fingerprint["source_tree_content_sha256"]
    current_policy_manifest_sha = fingerprint["manifest_policy_sha256"]
    expected_ver = extract_release_version(root_path)

    build_meta: Dict[str, Any] = {}
    build_prov: Optional[Dict[str, Any]] = None

    if build_exe:
        # Build executable using PyInstaller
        try:
            cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "potatostics.spec"]
            res = subprocess.run(cmd, cwd=str(root_path), capture_output=True, text=True)
            t = time.perf_counter() - t0
            if res.returncode != 0 or not exe_path.exists():
                return ReleaseGateResult(12, "Windows Executable Build & Smoke Test", "FAIL", res.stderr[-500:] or "Executable missing after build", t)

            exe_sha = hashlib.sha256(exe_path.read_bytes()).hexdigest()
            pyinstaller_ver = "unknown"
            try:
                import PyInstaller
                pyinstaller_ver = getattr(PyInstaller, "__version__", "unknown")
            except Exception:
                pass

            build_meta = {
                "build_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "exe_sha256": exe_sha,
                "source_tree_content_sha256": current_tree_content_sha,
                "source_manifest_sha256": current_policy_manifest_sha,
                "pyinstaller_version": pyinstaller_ver,
                "python_version": platform.python_version(),
                "platform": f"{platform.system()} ({platform.machine()})",
                "version": expected_ver,
                "git_commit": fingerprint.get("git_commit", "unknown"),
                "git_is_dirty": fingerprint.get("git_is_dirty", False),
                "is_newly_built": True
            }
            meta_path.write_text(json.dumps(build_meta, indent=2), encoding="utf-8")
            build_prov = build_meta
        except Exception as ex:
            t = time.perf_counter() - t0
            return ReleaseGateResult(12, "Windows Executable Build & Smoke Test", "FAIL", str(ex), t)

    if not exe_path.exists():
        return ReleaseGateResult(12, "Windows Executable Build & Smoke Test", "NOT VERIFIED", "Executable binary not found at dist/Potatostics/Potatostics.exe; pass --build-exe to compile", 0.0)

    # Provenance verification for pre-existing executable
    if not build_exe:
        if not meta_path.exists():
            build_prov = {"unverified_provenance": True, "reason": "Missing dist/Potatostics/build_metadata.json sidecar"}
            return ReleaseGateResult(12, "Windows Executable Build & Smoke Test", "NOT VERIFIED", "Pre-existing executable lacks build_metadata.json sidecar; source provenance unverified", 0.0, build_provenance=build_prov)

        try:
            build_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            build_meta["is_newly_built"] = False
            actual_exe_sha = hashlib.sha256(exe_path.read_bytes()).hexdigest()
            if build_meta.get("exe_sha256") != actual_exe_sha:
                build_prov = {"unverified_provenance": True, "reason": "Executable SHA-256 does not match build_metadata.json"}
                return ReleaseGateResult(12, "Windows Executable Build & Smoke Test", "FAIL", "Executable SHA-256 mismatch with build metadata (binary modified or corrupt)", 0.0, build_provenance=build_prov)

            # Check source tree content fingerprint
            built_content_sha = build_meta.get("source_tree_content_sha256")
            if not built_content_sha or built_content_sha != current_tree_content_sha:
                build_prov = {"is_stale": True, "reason": f"Source tree content modified since build (built={str(built_content_sha)[:8]}... vs current={current_tree_content_sha[:8]}...)"}
                return ReleaseGateResult(12, "Windows Executable Build & Smoke Test", "NOT VERIFIED", f"Executable is STALE: source content modified since build ({str(built_content_sha)[:8]} != {current_tree_content_sha[:8]})", 0.0, build_provenance=build_prov)

            # Check policy manifest sha
            if build_meta.get("source_manifest_sha256") != current_policy_manifest_sha:
                build_prov = {"is_stale": True, "reason": f"Built from manifest {build_meta.get('source_manifest_sha256', '')[:8]}... but current tree is {current_policy_manifest_sha[:8]}..."}
                return ReleaseGateResult(12, "Windows Executable Build & Smoke Test", "NOT VERIFIED", f"Executable is STALE: current source manifest differs from build source ({build_meta.get('source_manifest_sha256', '')[:8]} != {current_policy_manifest_sha[:8]})", 0.0, build_provenance=build_prov)

            build_prov = build_meta
        except Exception as ex:
            build_prov = {"unverified_provenance": True, "reason": str(ex)}
            return ReleaseGateResult(12, "Windows Executable Build & Smoke Test", "NOT VERIFIED", f"Invalid build metadata: {ex}", 0.0, build_provenance=build_prov)

    # 1. Version command test
    try:
        proc_ver = subprocess.run([str(exe_path), "--version"], cwd=str(root_path), capture_output=True, text=True, timeout=15)
        if proc_ver.returncode != 0 or expected_ver not in proc_ver.stdout:
            t = time.perf_counter() - t0
            return ReleaseGateResult(12, "Windows Executable Build & Smoke Test", "FAIL", f"Executable --version failed (code {proc_ver.returncode}): {proc_ver.stderr or proc_ver.stdout}", t, build_provenance=build_prov)
    except Exception as ex:
        t = time.perf_counter() - t0
        return ReleaseGateResult(12, "Windows Executable Build & Smoke Test", "FAIL", f"Executable version check error: {ex}", t, build_provenance=build_prov)

    # 2. Comprehensive noninteractive GUI smoke test
    gui_smoke_ok = False
    if gui_smoke:
        try:
            proc_gui = subprocess.run([str(exe_path), "--gui-smoke"], cwd=str(root_path), capture_output=True, text=True, timeout=30)
            if proc_gui.returncode == 0 and "GUI_SMOKE_TEST_PASSED" in proc_gui.stdout:
                gui_smoke_ok = True
            else:
                t = time.perf_counter() - t0
                return ReleaseGateResult(12, "Windows Executable Build & Smoke Test", "FAIL", f"GUI smoke test failed (code {proc_gui.returncode}): {proc_gui.stderr or proc_gui.stdout}", t, build_provenance=build_prov)
        except Exception as ex:
            t = time.perf_counter() - t0
            return ReleaseGateResult(12, "Windows Executable Build & Smoke Test", "FAIL", f"Executable GUI smoke error: {ex}", t, build_provenance=build_prov)

    # 3. Record executable metadata and provenance
    exe_size_mb = exe_path.stat().st_size / (1024 * 1024)
    exe_sha = hashlib.sha256(exe_path.read_bytes()).hexdigest()
    t = time.perf_counter() - t0
    is_new = build_meta.get("is_newly_built", False)
    built_desc = "newly built" if is_new else "pre-existing verified"
    smoke_desc = "PASS" if gui_smoke_ok else "SKIPPED (diagnostic run only)"
    detail = f"Executable verified ({built_desc}): {exe_path.name} ({exe_size_mb:.1f} MB, sha256={exe_sha[:16]}..., version={expected_ver}, gui_smoke={smoke_desc})"
    status = "PASS" if gui_smoke_ok else "NOT VERIFIED"
    return ReleaseGateResult(12, "Windows Executable Build & Smoke Test", status, detail, t, build_provenance=build_prov)


def print_usage():
    print("Potatostics — Release Acceptance Pipeline (Stage 49)")
    print("Usage: python scripts/run_release_acceptance.py [options]")
    print()
    print("Options:")
    print("  --target, -t <target>  Release target: 'source' (default), 'windows', or 'all'")
    print("  --install-mode <mode>  Gate 10 install mode: 'isolated' (Mode A), 'preinstalled' (Mode B), or 'auto' (default)")
    print("  --wheelhouse <dir>     Path to directory of local dependency wheels for offline clean install")
    print("  --build-exe            Compile Windows executable via PyInstaller in Gate 12")
    print("  --no-gui-smoke         Skip GUI smoke test during executable verification (diagnostic only)")
    print("  --help, -h             Display this usage help message and exit")


def main():
    if any(arg in sys.argv[1:] for arg in ("--help", "-h")):
        print_usage()
        sys.exit(0)

    target = "source"
    install_mode = "auto"
    wheelhouse = None
    build_exe = "--build-exe" in sys.argv
    gui_smoke = "--no-gui-smoke" not in sys.argv

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith("--target="):
            target = arg.split("=", 1)[1].lower()
        elif arg in ("-t", "--target") and i + 1 < len(sys.argv):
            target = sys.argv[i + 1].lower()
            i += 1
        elif arg.startswith("--install-mode="):
            install_mode = arg.split("=", 1)[1].lower()
        elif arg == "--install-mode" and i + 1 < len(sys.argv):
            install_mode = sys.argv[i + 1].lower()
            i += 1
        elif arg.startswith("--wheelhouse="):
            wheelhouse = Path(arg.split("=", 1)[1]).resolve()
        elif arg == "--wheelhouse" and i + 1 < len(sys.argv):
            wheelhouse = Path(sys.argv[i + 1]).resolve()
            i += 1
        i += 1

    if target not in ("source", "windows", "all"):
        print(f"Error: Unknown target '{target}'. Valid targets: source, windows, all", file=sys.stderr)
        sys.exit(1)

    if install_mode not in ("isolated", "preinstalled", "auto"):
        print(f"Error: Unknown install-mode '{install_mode}'. Valid modes: isolated, preinstalled, auto", file=sys.stderr)
        sys.exit(1)

    root_path = Path(".").resolve()

    print("=" * 80)
    print("POTATOSTICS — OFFICIAL RELEASE ACCEPTANCE PIPELINE (STAGE 48)")
    print(f"Platform: {platform.system()} ({platform.machine()}) | Python: {platform.python_version()}")
    print(f"Release Target: '{target}' | Install Mode: '{install_mode}' | Build Exe Flag: {build_exe}")
    print("=" * 80)

    results: List[ReleaseGateResult] = []

    # Gate 1: Compilation
    print("\n[Gate 1/12] Source Compilation...")
    g1 = run_gate_01_compilation(root_path)
    results.append(g1)
    print(f"  -> {g1.status}: {g1.details} ({g1.elapsed_s:.2f}s)")

    # Gate 2: Module Smoke
    print("\n[Gate 2/12] Module Import Smoke Tests...")
    g2 = run_gate_02_import_smoke(root_path)
    results.append(g2)
    print(f"  -> {g2.status}: {g2.details} ({g2.elapsed_s:.2f}s)")

    # Gate 3: Scientific Regressions
    print("\n[Gate 3/12] Scientific Regression Tests...")
    g3 = run_gate_03_scientific_regressions(root_path)
    results.append(g3)
    print(f"  -> {g3.status}: {g3.details} ({g3.elapsed_s:.2f}s)")

    # Gate 4: Full Test Suite
    print("\n[Gate 4/12] Complete Automated Pytest Suite...")
    g4 = run_gate_04_full_test_suite(root_path)
    results.append(g4)
    print(f"  -> {g4.status}: {g4.details} ({g4.elapsed_s:.2f}s)")

    # Gate 5: GUI Integration
    print("\n[Gate 5/12] GUI Integration & Research Workflow...")
    g5 = run_gate_05_gui_integration(root_path)
    results.append(g5)
    print(f"  -> {g5.status}: {g5.details} ({g5.elapsed_s:.2f}s)")

    # Gate 6: Benchmarks
    print("\n[Gate 6/12] Econometric Benchmark Suite...")
    g6 = run_gate_06_benchmarks(root_path)
    results.append(g6)
    print(f"  -> {g6.status}: {g6.details} ({g6.elapsed_s:.2f}s)")

    # Gate 7: Source Release Generation
    print("\n[Gate 7/12] Clean Source-Release Generation...")
    g7, archive_path = run_gate_07_source_release_generation(root_path)
    results.append(g7)
    print(f"  -> {g7.status}: {g7.details} ({g7.elapsed_s:.2f}s)")

    # Gate 8: Archive Verification
    print("\n[Gate 8/12] Independent Post-Build Archive Verification...")
    g8 = run_gate_08_archive_verification(archive_path, root_path)
    results.append(g8)
    print(f"  -> {g8.status}: {g8.details} ({g8.elapsed_s:.2f}s)")

    # Gate 9: Wheel Build
    print("\n[Gate 9/12] Wheel Build...")
    g9, wheel_path = run_gate_09_wheel_build(root_path)
    results.append(g9)
    print(f"  -> {g9.status}: {g9.details} ({g9.elapsed_s:.2f}s)")

    # Gate 10: Wheel Installation & Site-Packages Provenance
    print("\n[Gate 10/12] Isolated Wheel Installation & Site-Packages Provenance...")
    g10, temp_dir_obj, venv_python = run_gate_10_isolated_wheel_install(
        wheel_path,
        install_mode=install_mode,
        wheelhouse=wheelhouse
    )
    results.append(g10)
    print(f"  -> {g10.status}: {g10.details} ({g10.elapsed_s:.2f}s)")

    # Gate 11: Package Resources Loading from Isolated venv
    print("\n[Gate 11/12] Installed-Package Resource Loading...")
    g11 = run_gate_11_installed_package_resources(venv_python, temp_dir_obj)
    results.append(g11)
    print(f"  -> {g11.status}: {g11.details} ({g11.elapsed_s:.2f}s)")

    if temp_dir_obj:
        try:
            temp_dir_obj.cleanup()
        except Exception:
            pass

    # Gate 12: Windows Executable
    print("\n[Gate 12/12] Windows Executable Verification...")
    g12 = run_gate_12_windows_executable(
        root_path,
        target=target,
        build_exe=build_exe,
        gui_smoke=gui_smoke
    )
    results.append(g12)
    print(f"  -> {g12.status}: {g12.details} ({g12.elapsed_s:.2f}s)")

    # Acceptance Summary Table
    print("\n" + "=" * 90)
    print(f"{'Gate #':<8} | {'Gate Name':<42} | {'Status':<12} | {'Elapsed':<8}")
    print("-" * 90)
    for res in results:
        print(f"{res.gate_num:<8} | {res.name:<42} | {res.status:<12} | {res.elapsed_s:>6.2f}s")
    print("=" * 90)

    # Evaluate overall release readiness via pure production policy function
    readiness = evaluate_release_readiness(
        results=results,
        target=target,
        install_mode=install_mode,
        build_provenance=getattr(g12, "build_provenance", None),
        gui_smoke_executed=gui_smoke
    )

    if readiness["conditional_checks"]:
        print("\n[Conditional Checks / Diagnostic Notices]")
        for note in readiness["conditional_checks"]:
            print(f"  * {note}")

    if readiness["is_ready"]:
        print(f"\nOVERALL STATUS: [RELEASE READY] (All mandatory acceptance gates for target '{target}' successfully passed)")
        sys.exit(0)
    else:
        print(f"\nOVERALL STATUS: [RELEASE BLOCKED] (Release gates rejected for target '{target}'):")
        for reason in readiness["reasons_for_rejection"]:
            print(f"  * {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
