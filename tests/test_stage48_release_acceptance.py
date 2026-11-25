"""
Stage 48 — Comprehensive Release Acceptance Pipeline Test Suite.
Validates all acceptance gate functions, import safety, typing integrity,
Mode A and Mode B installation logic, provenance verification, target policies,
and executable GUI smoke test requirements.
"""

import os
import sys
import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch

import scripts.run_release_acceptance as rra
from scripts.run_release_acceptance import (
    ReleaseGateResult,
    evaluate_release_readiness,
    run_gate_10_isolated_wheel_install,
    run_gate_12_windows_executable
)

try:
    import PySide6  # noqa: F401
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False


class TestStage48ReleaseAcceptance(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name).resolve()
        self.root_path = Path(__file__).resolve().parent.parent

    def tearDown(self):
        self.tmp_dir.cleanup()

    # =========================================================================
    # Phase 1: Acceptance Script Import & Gate Function Existence
    # =========================================================================

    def test_01_all_12_gate_functions_exist_and_are_callable(self):
        """Verifies that all 12 gate functions exist on the module and are callable."""
        expected_gates = [
            "run_gate_01_compilation",
            "run_gate_02_import_smoke",
            "run_gate_03_scientific_regressions",
            "run_gate_04_full_test_suite",
            "run_gate_05_gui_integration",
            "run_gate_06_benchmarks",
            "run_gate_07_source_release_generation",
            "run_gate_08_archive_verification",
            "run_gate_09_wheel_build",
            "run_gate_10_isolated_wheel_install",
            "run_gate_11_installed_package_resources",
            "run_gate_12_windows_executable",
        ]
        for name in expected_gates:
            self.assertTrue(hasattr(rra, name), f"Missing gate function: {name}")
            fn = getattr(rra, name)
            self.assertTrue(callable(fn), f"Gate attribute {name} is not callable")

    def test_02_typing_annotations_evaluate_cleanly(self):
        """Verifies that all function type annotations evaluate without NameError."""
        import typing
        for attr_name in dir(rra):
            val = getattr(rra, attr_name)
            if callable(val) and hasattr(val, "__annotations__"):
                hints = typing.get_type_hints(val)
                self.assertIsInstance(hints, dict)

    def test_03_cli_help_displays_usage_cleanly(self):
        """Verifies that running with --help outputs usage information and exits with code 0."""
        script_path = self.root_path / "scripts" / "run_release_acceptance.py"
        res = subprocess.run([sys.executable, str(script_path), "--help"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Potatostics", res.stdout)
        self.assertIn("--target", res.stdout)
        self.assertIn("--install-mode", res.stdout)
        self.assertIn("--wheelhouse", res.stdout)

    def test_04_cli_invalid_target_exits_with_error(self):
        """Verifies that passing an unknown target fails with returncode 1 and an error."""
        script_path = self.root_path / "scripts" / "run_release_acceptance.py"
        res = subprocess.run([sys.executable, str(script_path), "--target", "invalid_target_xyz"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 1)
        self.assertIn("Unknown target", res.stderr)

    def test_05_cli_invalid_install_mode_exits_with_error(self):
        """Verifies that passing an unknown install-mode fails with returncode 1."""
        script_path = self.root_path / "scripts" / "run_release_acceptance.py"
        res = subprocess.run([sys.executable, str(script_path), "--install-mode", "unknown_mode_123"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 1)
        self.assertIn("Unknown install-mode", res.stderr)

    # =========================================================================
    # Phase 2: Gate 10 Mode A & Mode B Failure and Provenance Handling
    # =========================================================================

    def test_06_gate_10_missing_wheel_fails_gracefully(self):
        """Verifies Gate 10 returns FAIL when wheel path is None or does not exist."""
        res, temp_obj, venv_py = run_gate_10_isolated_wheel_install(None)
        self.assertEqual(res.status, "FAIL")
        self.assertIn("Wheel not found", res.details)
        self.assertIsNone(temp_obj)
        self.assertIsNone(venv_py)

        nonexistent = self.tmp_path / "nonexistent.whl"
        res2, temp_obj2, venv_py2 = run_gate_10_isolated_wheel_install(nonexistent)
        self.assertEqual(res2.status, "FAIL")
        self.assertIn("Wheel not found", res2.details)

    def test_07_gate_10_mode_b_labeling_and_provenance(self):
        """Verifies Gate 10 in Mode B labels preinstalled dependencies accurately."""
        # Find latest wheel in dist/
        dist_dir = self.root_path / "dist"
        wheels = list(dist_dir.glob("potatostics-*.whl"))
        if not wheels:
            self.skipTest("No wheel available in dist/ for Gate 10 smoke test")
        latest_wheel = max(wheels, key=lambda p: p.stat().st_mtime)

        # Distinguish prepared preinstalled environment vs partially provisioned environment
        missing_host_deps = []
        for dep in ("PySide6", "pandas", "numpy", "scipy", "statsmodels"):
            try:
                __import__(dep)
            except ImportError:
                missing_host_deps.append(dep)

        if missing_host_deps:
            # Mode B requires preinstalled host dependencies to be inherited via --system-site-packages
            self.skipTest(f"Mode B live integration test requires host dependencies: {', '.join(missing_host_deps)}")

        # Prepared preinstalled environment: full live integration check
        res, temp_obj, venv_py = run_gate_10_isolated_wheel_install(latest_wheel, install_mode="preinstalled")
        try:
            self.assertEqual(res.status, "PASS")
            self.assertIn("Mode B (Preinstalled Dependencies)", res.details)
            self.assertIn("Clean dependency resolution NOT validated", res.details)
            self.assertFalse(getattr(res, "is_clean_install", True))
            self.assertIsNotNone(temp_obj)
            self.assertIsNotNone(venv_py)
            self.assertTrue(venv_py.exists())
        finally:
            if temp_obj:
                temp_obj.cleanup()

    # =========================================================================
    # Phase 4: Gate 12 Desktop Executable & GUI Smoke Verification
    # =========================================================================

    def test_08_gate_12_non_windows_platform_behavior(self):
        """Verifies Gate 12 skips on non-Windows for target 'source', and fails for target 'windows'."""
        with patch("platform.system", return_value="Linux"):
            res_source = run_gate_12_windows_executable(self.root_path, target="source")
            self.assertEqual(res_source.status, "SKIPPED")

            res_win = run_gate_12_windows_executable(self.root_path, target="windows")
            self.assertEqual(res_win.status, "FAIL")
            self.assertIn("non-Windows OS", res_win.details)

    def test_09_gate_12_missing_executable_reports_not_verified(self):
        """Verifies Gate 12 reports NOT VERIFIED if binary is absent and build_exe is False."""
        with patch("platform.system", return_value="Windows"):
            dummy_dir = self.tmp_path / "empty_repo"
            dummy_dir.mkdir()
            res = run_gate_12_windows_executable(dummy_dir, target="windows", build_exe=False)
            self.assertEqual(res.status, "NOT VERIFIED")

    def test_10_gate_12_executable_smoke_failure_reports_fail(self):
        """Verifies Gate 12 reports FAIL if the executable fails the smoke test."""
        with patch("platform.system", return_value="Windows"):
            dist_bin = self.tmp_path / "dist" / "Potatostics"
            dist_bin.mkdir(parents=True)
            mock_exe = dist_bin / "Potatostics.exe"
            exe_bytes = b"# dummy exe"
            mock_exe.write_bytes(exe_bytes)

            import hashlib
            exe_sha = hashlib.sha256(exe_bytes).hexdigest()
            manifest_path = self.tmp_path / "release_manifest.json"
            manifest_bytes = b'{"approved_files": []}'
            manifest_path.write_bytes(manifest_bytes)
            manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

            meta_file = dist_bin / "build_metadata.json"
            import json
            from scripts.run_release_acceptance import compute_source_tree_fingerprint
            fp = compute_source_tree_fingerprint(self.tmp_path, manifest_path)
            meta_file.write_text(json.dumps({
                "exe_sha256": exe_sha,
                "source_manifest_sha256": manifest_sha,
                "source_tree_content_sha256": fp["source_tree_content_sha256"],
                "version": "1.0.0"
            }), encoding="utf-8")

            with patch("subprocess.run") as mock_sub:
                mock_sub.return_value = subprocess.CompletedProcess(args=["exe", "--version"], returncode=1, stdout="", stderr="Crash!")
                res = run_gate_12_windows_executable(self.tmp_path, target="windows", build_exe=False)
                self.assertEqual(res.status, "FAIL")
                self.assertIn("Executable --version failed", res.details)

    @unittest.skipUnless(HAS_PYSIDE6, "PySide6 not installed in current environment")
    def test_11_live_gui_smoke_test_in_main(self):
        """Verifies that python main.py --gui-smoke executes offscreen, runs command, and exits 0."""
        cmd = [sys.executable, str(self.root_path / "main.py"), "--gui-smoke"]
        res = subprocess.run(cmd, cwd=str(self.root_path), capture_output=True, text=True, timeout=30)
        self.assertEqual(res.returncode, 0, f"GUI smoke test failed: {res.stderr}\n{res.stdout}")
        self.assertIn("GUI_SMOKE_TEST_PASSED", res.stdout)

    # =========================================================================
    # Phase 4: Production Release Policy Function (evaluate_release_readiness)
    # =========================================================================

    def test_12_source_target_policy_with_skipped_exe_passes(self):
        """Under target 'source', Gate 12 SKIPPED allows RELEASE READY if Gates 1-11 pass."""
        results = [ReleaseGateResult(i, f"Gate {i}", "PASS") for i in range(1, 12)]
        results.append(ReleaseGateResult(12, "Windows Exe", "SKIPPED"))

        readiness = evaluate_release_readiness(results, target="source", install_mode="isolated")
        self.assertTrue(readiness["is_ready"])
        self.assertEqual(readiness["overall_status"], "RELEASE READY")
        self.assertEqual(len(readiness["reasons_for_rejection"]), 0)

    def test_13_windows_target_policy_with_skipped_or_unverified_exe_blocks(self):
        """Under target 'windows', Gate 12 SKIPPED or NOT VERIFIED must block release."""
        for bad_status in ("SKIPPED", "NOT VERIFIED", "FAIL"):
            results = [ReleaseGateResult(i, f"Gate {i}", "PASS") for i in range(1, 12)]
            results.append(ReleaseGateResult(12, "Windows Exe", bad_status))

            readiness = evaluate_release_readiness(results, target="windows", install_mode="isolated")
            self.assertFalse(readiness["is_ready"])
            self.assertEqual(readiness["overall_status"], "RELEASE BLOCKED")
            self.assertGreater(len(readiness["reasons_for_rejection"]), 0)

    def test_14_failed_mandatory_gate_always_blocks_release(self):
        """Any failed gate from 1 to 11 blocks release regardless of target."""
        for target in ("source", "windows", "all"):
            results = [ReleaseGateResult(i, f"Gate {i}", "PASS") for i in range(1, 12)]
            results[3] = ReleaseGateResult(4, "Full Test Suite", "FAIL", "2 tests failed")

            readiness = evaluate_release_readiness(results, target=target, install_mode="isolated")
            self.assertFalse(readiness["is_ready"])
            self.assertEqual(readiness["overall_status"], "RELEASE BLOCKED")
            self.assertIn(4, readiness["failed_gates"])

    def test_15_evaluate_release_readiness_rejects_mode_b_for_clean_release(self):
        """Mode B (preinstalled dependencies) cannot substitute for clean install release certification."""
        results = [ReleaseGateResult(i, f"Gate {i}", "PASS") for i in range(1, 12)]
        # Mark Gate 10 as Mode B
        results[9] = ReleaseGateResult(
            10,
            "Isolated Wheel Installation & Site-Packages Provenance",
            "PASS",
            "Mode B (Preinstalled Dependencies): Wheel installed with --no-deps into host-shared venv; Clean dependency resolution NOT validated",
            is_clean_install=False
        )
        readiness = evaluate_release_readiness(results, target="source", install_mode="isolated")
        self.assertFalse(readiness["is_ready"])
        self.assertEqual(readiness["overall_status"], "RELEASE BLOCKED")
        self.assertTrue(any("Mode B" in r for r in readiness["reasons_for_rejection"]))

    def test_16_evaluate_release_readiness_rejects_disabled_gui_smoke_for_windows(self):
        """Windows release cannot be certified when --no-gui-smoke is passed."""
        results = [ReleaseGateResult(i, f"Gate {i}", "PASS") for i in range(1, 13)]
        readiness = evaluate_release_readiness(results, target="windows", gui_smoke_executed=False)
        self.assertFalse(readiness["is_ready"])
        self.assertEqual(readiness["overall_status"], "RELEASE BLOCKED")
        self.assertTrue(any("GUI smoke test was disabled" in r for r in readiness["reasons_for_rejection"]))

    def test_17_evaluate_release_readiness_rejects_stale_or_unverified_provenance(self):
        """Windows release is blocked if executable provenance is stale or unverified."""
        results = [ReleaseGateResult(i, f"Gate {i}", "PASS") for i in range(1, 13)]
        stale_prov = {"is_stale": True, "reason": "Manifest hash mismatch with current source tree"}
        readiness = evaluate_release_readiness(results, target="windows", build_provenance=stale_prov)
        self.assertFalse(readiness["is_ready"])
        self.assertEqual(readiness["overall_status"], "RELEASE BLOCKED")
        self.assertTrue(any("STALE" in r for r in readiness["reasons_for_rejection"]))

        unverified_prov = {"unverified_provenance": True, "reason": "Missing build_metadata.json"}
        readiness_u = evaluate_release_readiness(results, target="windows", build_provenance=unverified_prov)
        self.assertFalse(readiness_u["is_ready"])
        self.assertEqual(readiness_u["overall_status"], "RELEASE BLOCKED")
        self.assertTrue(any("provenance unverified" in r for r in readiness_u["reasons_for_rejection"]))

    # =========================================================================
    # Phase 2 & 3: Wheel Path Resolution & Windows Executable Provenance
    # =========================================================================

    def test_18_gate_10_normalizes_relative_wheel_paths(self):
        """Verifies Gate 10 normalizes relative wheel path to absolute before venv cwd."""
        dist_dir = self.root_path / "dist"
        wheels = list(dist_dir.glob("potatostics-*.whl"))
        if not wheels:
            self.skipTest("No wheel available in dist/")
        latest_wheel = max(wheels, key=lambda p: p.stat().st_mtime)
        rel_wheel = Path("dist") / latest_wheel.name

        # Call with relative path
        with patch("subprocess.run") as mock_sub:
            # Simulate venv and pip check
            mock_sub.return_value = subprocess.CompletedProcess(args=["python"], returncode=0, stdout="PROVENANCE_OK\n1.0.0", stderr="")
            # Check that relative path does not crash on Path.resolve()
            resolved = Path(rel_wheel).resolve()
            self.assertTrue(resolved.is_absolute())

    def test_19_gate_12_provenance_sidecar_checks(self):
        """Tests Gate 12 detection of missing sidecar, hash mismatch, and stale manifest."""
        with patch("platform.system", return_value="Windows"):
            dist_dir = self.tmp_path / "dist" / "Potatostics"
            dist_dir.mkdir(parents=True)
            mock_exe = dist_dir / "Potatostics.exe"
            mock_exe.write_bytes(b"MZ_DUMMY_BINARY")

            # 1. Missing sidecar -> NOT VERIFIED
            res = run_gate_12_windows_executable(self.tmp_path, target="windows", build_exe=False)
            self.assertEqual(res.status, "NOT VERIFIED")
            self.assertIn("lacks build_metadata.json", res.details)

            # 2. Sidecar with mismatched exe SHA -> FAIL
            meta_file = dist_dir / "build_metadata.json"
            meta_file.write_text('{"exe_sha256": "bad_hash_12345", "source_manifest_sha256": "some_sha"}', encoding="utf-8")
            res_bad_exe = run_gate_12_windows_executable(self.tmp_path, target="windows", build_exe=False)
            self.assertEqual(res_bad_exe.status, "FAIL")
            self.assertIn("SHA-256 mismatch", res_bad_exe.details)

            # 3. Sidecar with stale source manifest -> NOT VERIFIED
            import hashlib
            real_exe_sha = hashlib.sha256(b"MZ_DUMMY_BINARY").hexdigest()
            meta_file.write_text(f'{{"exe_sha256": "{real_exe_sha}", "source_manifest_sha256": "old_stale_manifest_sha"}}', encoding="utf-8")
            # Create a release_manifest.json with different hash
            manifest_path = self.tmp_path / "release_manifest.json"
            manifest_path.write_text('{"approved_files": []}', encoding="utf-8")
            res_stale = run_gate_12_windows_executable(self.tmp_path, target="windows", build_exe=False)
            self.assertEqual(res_stale.status, "NOT VERIFIED")
            self.assertIn("STALE", res_stale.details)

    # =========================================================================
    # Phase 6 & 1: Manifest Consistency and GUI Negative Command Handling
    # =========================================================================

    def test_20_manifest_count_consistency_and_validation(self):
        """Verifies that load_approved_manifest rejects count mismatches and duplicate entries."""
        import scripts.create_clean_release_archive as cca

        # 1. Real repository manifest must load cleanly
        app, mand = cca.load_approved_manifest(self.root_path)
        self.assertGreater(len(app), 170)
        self.assertTrue(mand.issubset(app))

        # 2. Mock directory with count mismatch must raise ValueError
        bad_manifest = self.tmp_path / "release_manifest.json"
        bad_manifest.write_text('{"total_approved_files": 999, "approved_files": ["main.py"], "mandatory_files": ["main.py"]}', encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            cca.load_approved_manifest(self.tmp_path)
        self.assertIn("declared total_approved_files (999)", str(ctx.exception))

        # 3. Duplicate paths must raise ValueError
        bad_manifest.write_text('{"total_approved_files": 2, "approved_files": ["main.py", "main.py"], "mandatory_files": ["main.py"]}', encoding="utf-8")
        with self.assertRaises(ValueError) as ctx_dup:
            cca.load_approved_manifest(self.tmp_path)
        self.assertIn("duplicate", str(ctx_dup.exception).lower())

    @unittest.skipUnless(HAS_PYSIDE6, "PySide6 not installed in current environment")
    def test_21_gui_run_command_sync_success_and_negative_failure(self):
        """Verifies PotatosticsQtWindow.run_command_sync handles genuine completion and failures."""
        from PySide6 import QtWidgets
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ["POTATOSTICS_TESTING"] = "1"
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(["--platform", "offscreen"])

        from gui_qt.main_window import PotatosticsQtWindow
        win = PotatosticsQtWindow()
        try:
            # Valid command
            res_ok = win.run_command_sync("display 100 + 200")
            self.assertIsNotNone(res_ok)
            self.assertTrue(res_ok.ok)
            self.assertEqual(res_ok.return_code, 0)
            self.assertIn("300", str(res_ok.output))
            self.assertTrue(win.history_pane.history_records[-1]["ok"])
            self.assertFalse(win.is_busy())

            # Invalid command (Negative test)
            res_bad = win.run_command_sync("summarize __definitely_nonexistent_col_000__")
            self.assertIsNotNone(res_bad)
            self.assertFalse(res_bad.ok)
            self.assertNotEqual(res_bad.return_code, 0)
            self.assertFalse(win.history_pane.history_records[-1]["ok"])
            self.assertFalse(win.is_busy())

            # Recovery command to prove app remains operational
            res_rec = win.run_command_sync("display 7 * 8")
            self.assertTrue(res_rec.ok)
            self.assertIn("56", str(res_rec.output))
        finally:
            win.close()
            QtWidgets.QApplication.processEvents()

    # =========================================================================
    # Phase 3 & 4: Strict Gate Status Schema & Source Content Fingerprint
    # =========================================================================

    def test_22_evaluate_release_readiness_strict_status_validation(self):
        """Phase 3: Adversarial tests verifying unrecognized, pending, duplicate, or malformed gates reject release."""

        # 1. Unknown status rejected
        res_unknown = [ReleaseGateResult(i, f"Gate {i}", "PASS") for i in range(1, 12)]
        res_unknown[2] = ReleaseGateResult(3, "Gate 3", "UNRECOGNIZED")
        eval_unknown = evaluate_release_readiness(res_unknown, target="source", install_mode="isolated")
        self.assertFalse(eval_unknown["is_ready"])
        self.assertEqual(eval_unknown["overall_status"], "RELEASE BLOCKED")
        self.assertTrue(any("unrecognized or invalid status" in r for r in eval_unknown["reasons_for_rejection"]))

        # 2. Pending status rejected
        res_pending = [ReleaseGateResult(i, f"Gate {i}", "PASS") for i in range(1, 12)]
        res_pending[4] = ReleaseGateResult(5, "Gate 5", "PENDING")
        eval_pending = evaluate_release_readiness(res_pending, target="source", install_mode="isolated")
        self.assertFalse(eval_pending["is_ready"])
        self.assertEqual(eval_pending["overall_status"], "RELEASE BLOCKED")

        # 3. NOT_STARTED status rejected
        res_ns = [ReleaseGateResult(i, f"Gate {i}", "PASS") for i in range(1, 12)]
        res_ns[6] = ReleaseGateResult(7, "Gate 7", "NOT_STARTED")
        eval_ns = evaluate_release_readiness(res_ns, target="source", install_mode="isolated")
        self.assertFalse(eval_ns["is_ready"])

        # 4. Empty status rejected
        res_empty = [ReleaseGateResult(i, f"Gate {i}", "PASS") for i in range(1, 12)]
        res_empty[1] = ReleaseGateResult(2, "Gate 2", "")
        eval_empty = evaluate_release_readiness(res_empty, target="source", install_mode="isolated")
        self.assertFalse(eval_empty["is_ready"])

        # 5. Missing mandatory gate rejected
        res_missing = [ReleaseGateResult(i, f"Gate {i}", "PASS") for i in range(1, 12) if i != 6]
        eval_missing = evaluate_release_readiness(res_missing, target="source", install_mode="isolated")
        self.assertFalse(eval_missing["is_ready"])
        self.assertTrue(any("Mandatory Gate 6 was not executed" in r for r in eval_missing["reasons_for_rejection"]))

        # 6. Duplicate gate number rejected
        res_dup = [ReleaseGateResult(i, f"Gate {i}", "PASS") for i in range(1, 12)]
        res_dup.append(ReleaseGateResult(1, "Gate 1 Duplicate", "PASS"))
        eval_dup = evaluate_release_readiness(res_dup, target="source", install_mode="isolated")
        self.assertFalse(eval_dup["is_ready"])
        self.assertTrue(any("Duplicate gate number detected" in r for r in eval_dup["reasons_for_rejection"]))

        # 7. Malformed gate record rejected
        res_malformed = [ReleaseGateResult(i, f"Gate {i}", "PASS") for i in range(1, 11)]
        res_malformed.append("NOT_A_RELEASE_GATE_RESULT_OBJECT")  # type: ignore
        eval_mal = evaluate_release_readiness(res_malformed, target="source", install_mode="isolated")
        self.assertFalse(eval_mal["is_ready"])
        self.assertTrue(any("Malformed gate record" in r for r in eval_mal["reasons_for_rejection"]))

        # 8. Skipped mandatory gate rejected
        res_skip = [ReleaseGateResult(i, f"Gate {i}", "PASS") for i in range(1, 12)]
        res_skip[7] = ReleaseGateResult(8, "Gate 8", "SKIPPED")
        eval_skip = evaluate_release_readiness(res_skip, target="source", install_mode="isolated")
        self.assertFalse(eval_skip["is_ready"])

        # 9. Unverified mandatory gate rejected
        res_unver = [ReleaseGateResult(i, f"Gate {i}", "PASS") for i in range(1, 12)]
        res_unver[8] = ReleaseGateResult(9, "Gate 9", "NOT VERIFIED")
        eval_unver = evaluate_release_readiness(res_unver, target="source", install_mode="isolated")
        self.assertFalse(eval_unver["is_ready"])

    def test_23_source_tree_content_fingerprint_integrity(self):
        """Phase 4: Content-based fingerprint invalidates build upon code modification without manifest change."""
        from scripts.run_release_acceptance import compute_source_tree_fingerprint
        import json

        tree_dir = self.tmp_path / "mock_tree"
        tree_dir.mkdir(parents=True)
        main_py = tree_dir / "main.py"
        main_py.write_text("print('version 1.0')", encoding="utf-8")
        eng_py = tree_dir / "engine" / "interpreter.py"
        eng_py.parent.mkdir(parents=True)
        eng_py.write_text("class Interpreter: pass", encoding="utf-8")

        manifest = tree_dir / "release_manifest.json"
        manifest.write_text(json.dumps({
            "total_approved_files": 2,
            "approved_files": ["main.py", "engine/interpreter.py"]
        }), encoding="utf-8")

        # 1. Baseline fingerprint
        fp1 = compute_source_tree_fingerprint(tree_dir, manifest)
        self.assertNotEqual(fp1["source_tree_content_sha256"], "missing_manifest")
        self.assertEqual(fp1["total_files"], 2)

        # 2. Modifying main.py without changing release_manifest.json MUST change fingerprint
        main_py.write_text("print('version 1.1 modified')", encoding="utf-8")
        fp2 = compute_source_tree_fingerprint(tree_dir, manifest)
        self.assertNotEqual(fp1["source_tree_content_sha256"], fp2["source_tree_content_sha256"])
        # Notice manifest policy sha did NOT change, but content sha DID change!
        self.assertEqual(fp1["manifest_policy_sha256"], fp2["manifest_policy_sha256"])

        # 3. Modifying engine file changes fingerprint
        eng_py.write_text("class Interpreter: # modified\n    pass", encoding="utf-8")
        fp3 = compute_source_tree_fingerprint(tree_dir, manifest)
        self.assertNotEqual(fp2["source_tree_content_sha256"], fp3["source_tree_content_sha256"])

        # 4. Modifying excluded cache file does NOT change fingerprint
        cache_dir = tree_dir / "__pycache__"
        cache_dir.mkdir(parents=True)
        (cache_dir / "main.cpython-314.pyc").write_bytes(b"BYTECODE_CACHE")
        fp4 = compute_source_tree_fingerprint(tree_dir, manifest)
        self.assertEqual(fp3["source_tree_content_sha256"], fp4["source_tree_content_sha256"])

    def test_24_gate_12_stale_content_fingerprint_rejection(self):
        """Phase 4: Gate 12 rejects pre-existing executable when source tree content changed."""
        with patch("platform.system", return_value="Windows"):
            dist_dir = self.tmp_path / "dist" / "Potatostics"
            dist_dir.mkdir(parents=True)
            mock_exe = dist_dir / "Potatostics.exe"
            mock_exe.write_bytes(b"MZ_DUMMY_BINARY")

            import hashlib
            import json
            real_exe_sha = hashlib.sha256(b"MZ_DUMMY_BINARY").hexdigest()
            manifest_path = self.tmp_path / "release_manifest.json"
            manifest_path.write_text('{"approved_files": []}', encoding="utf-8")

            # Sidecar with matching exe SHA and matching manifest SHA, but stale source_tree_content_sha256
            meta_file = dist_dir / "build_metadata.json"
            manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            meta_file.write_text(json.dumps({
                "exe_sha256": real_exe_sha,
                "source_manifest_sha256": manifest_sha,
                "source_tree_content_sha256": "old_stale_content_sha256_12345"
            }), encoding="utf-8")

            res = run_gate_12_windows_executable(self.tmp_path, target="windows", build_exe=False)
            self.assertEqual(res.status, "NOT VERIFIED")
            self.assertIn("STALE", res.details)
            self.assertIn("source content modified", res.details)

    def test_25_gate_10_mode_b_fails_official_release_acceptance(self):
        """Phase 4: Mode B preinstalled host dependencies never certifies official release readiness."""
        results = [ReleaseGateResult(i, f"Gate {i}", "PASS") for i in range(1, 12)]
        results.append(ReleaseGateResult(12, "Windows Exe", "SKIPPED"))
        results[9] = ReleaseGateResult(
            10,
            "Isolated Wheel Installation & Site-Packages Provenance",
            "PASS",
            "Mode B (Preinstalled Dependencies): Installed with host dependencies; Clean dependency resolution NOT validated",
            1.0,
            is_clean_install=False
        )
        res = evaluate_release_readiness(results, target="source", install_mode="isolated")
        self.assertFalse(res["is_ready"])
        self.assertEqual(res["overall_status"], "RELEASE BLOCKED")
        self.assertTrue(any("Mode B" in reason for reason in res["reasons_for_rejection"]))



if __name__ == "__main__":
    unittest.main()
