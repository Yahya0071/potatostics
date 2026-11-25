"""
tests/test_stage37_batch_and_extensions.py - Test Suite for Batch Mode, Seeded Bootstrap,
and Advanced Econometric Extensions (Bacon Decomposition & Doubly Robust DiD).
"""

import os
import sys
import shutil
import tempfile
import unittest
import subprocess
import numpy as np
import pandas as pd

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.causal_modern import compute_did_multiplegt


class TestStage37BatchAndExtensions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_main_batch_execution_success(self):
        """Verify python main.py --batch <file.do> executes headlessly and returns code 0."""
        do_file = os.path.join(self.temp_dir, "batch_test_success.do")
        with open(do_file, "w", encoding="utf-8") as f:
            f.write("clear\n")
            f.write("set obs 10\n")
            f.write("gen x = 42\n")
            f.write("summarize x\n")

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cmd = [sys.executable, os.path.join(repo_root, "main.py"), "--batch", do_file]
        res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)

        self.assertEqual(res.returncode, 0, f"Batch script failed with code {res.returncode}. Stderr: {res.stderr}")
        self.assertIn("Variable", res.stdout)
        self.assertIn("x", res.stdout)

    def test_02_main_batch_execution_error(self):
        """Verify python main.py -b <file.do> returns non-zero code on syntax/runtime error."""
        do_file = os.path.join(self.temp_dir, "batch_test_error.do")
        with open(do_file, "w", encoding="utf-8") as f:
            f.write("clear\n")
            f.write("nonexistent_syntax_error_command_xyz\n")

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cmd = [sys.executable, os.path.join(repo_root, "main.py"), "-b", do_file]
        res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)

        self.assertNotEqual(res.returncode, 0, "Batch script with error must exit with non-zero code")
        combined_output = (res.stdout + res.stderr).lower()
        self.assertTrue("error" in combined_output or "unrecognized" in combined_output or "command not found" in combined_output)

    def test_03_gui_initial_file_command_type(self):
        """Verify gui_qt/main_window schedules 'do' for .do files and 'session load' for .potproj."""
        from gui_qt.main_window import PotatosticsQtWindow
        from PySide6 import QtWidgets
        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["--platform", "offscreen"])

        # Test .do file command selection
        do_path = os.path.join(self.temp_dir, "script.do")
        with open(do_path, "w") as f:
            f.write("// do file\n")
        win_do = PotatosticsQtWindow(initial_file=do_path)
        self.assertEqual(win_do._initial_cmd, f'do "{do_path}"')
        win_do.close()

        # Test .potproj session file command selection
        proj_path = os.path.join(self.temp_dir, "session.potproj")
        with open(proj_path, "w") as f:
            f.write("PK")
        win_proj = PotatosticsQtWindow(initial_file=proj_path)
        self.assertEqual(win_proj._initial_cmd, f'session load "{proj_path}"')
        win_proj.close()

    def test_04_did_multiplegt_seeded_bootstrap(self):
        """Verify did_multiplegt bootstrap is 100% deterministic with seed and avoids indexing bugs."""
        # Create panel data with 8 units, 3 time periods, with switchers and heterogeneous effects
        sim_rng = np.random.RandomState(99)
        rows = []
        for uid in range(1, 9):
            effect_i = 1.5 if uid % 2 == 0 else 2.5
            for tid in [1, 2, 3]:
                treated = 1 if (uid <= 4 and tid >= 2) else 0
                val = 10.0 + effect_i * treated + 0.5 * tid + 0.1 * uid + float(sim_rng.normal(0, 0.5))
                rows.append({"id": uid, "t": tid, "d": treated, "y": val})
        df = pd.DataFrame(rows)

        # Call with seed=42 twice
        r1 = compute_did_multiplegt(df, "y", "id", "t", "d", reps=30, seed=42)
        r2 = compute_did_multiplegt(df, "y", "id", "t", "d", reps=30, seed=42)
        self.assertEqual(r1["se"], r2["se"], "Consecutive calls with identical seed must produce bit-identical SE")

        # Call with different seed
        r3 = compute_did_multiplegt(df, "y", "id", "t", "d", reps=30, seed=123)
        self.assertNotEqual(r1["se"], r3["se"], "Calls with different seed should produce distinct bootstrap draws")

        # Interpreter command execution with seed() option
        ds = Dataset()
        ds.load_dataframe(df)
        interp = Interpreter(ds)
        res1 = interp.execute("did_multiplegt y id t d, reps(20) seed(42)")
        self.assertTrue(res1.ok, f"did_multiplegt failed: {res1.error}")
        se_interp_1 = interp.registers.get_e("se_did_m")

        res2 = interp.execute("did_multiplegt y id t d, reps(20) seed(42)")
        self.assertTrue(res2.ok)
        se_interp_2 = interp.registers.get_e("se_did_m")
        self.assertEqual(se_interp_1, se_interp_2, "Interpreter did_multiplegt must reproduce exact same SE with seed(42)")

    def test_05_bacondecomp_analytical_benchmark(self):
        """Verify Goodman-Bacon decomposition 2x2 estimates and exact TWFE decomposition."""
        from engine.causal_modern import compute_bacondecomp
        # 12 units, 4 periods: 4 never treated, 4 treated at t=2, 4 treated at t=3
        rows = []
        for uid in range(1, 13):
            k = None
            if 1 <= uid <= 4:
                k = 2
            elif 5 <= uid <= 8:
                k = 3
            for tid in [1, 2, 3, 4]:
                treated = 1 if (k is not None and tid >= k) else 0
                eff = 2.0 if k == 2 else 4.0
                y = 10.0 + eff * treated + 0.5 * tid + 0.2 * uid
                rows.append({"id": uid, "t": tid, "d": treated, "y": y})
        df = pd.DataFrame(rows)

        # 1. Engine calculation
        r = compute_bacondecomp(df, "y", "d", "id", "t")
        self.assertIn("twfe", r)
        self.assertIn("weighted_bacon", r)
        self.assertAlmostEqual(r["twfe"], r["weighted_bacon"], places=6)
        self.assertAlmostEqual(r["weights_sum"], 1.0, places=6)

        # Verify summary breakdown across all 3 comparison categories
        summary = r["summary"]
        comp_types = [row["comparison_type"] for row in summary]
        self.assertIn("Treated vs Never Treated", comp_types)
        self.assertIn("Earlier vs Later Treated", comp_types)
        self.assertIn("Later vs Earlier Treated", comp_types)

        # 2. Interpreter command execution
        ds = Dataset()
        ds.load_dataframe(df)
        interp = Interpreter(ds)
        res = interp.execute("bacondecomp y d, ivar(id) time(t)")
        self.assertTrue(res.ok, f"bacondecomp command failed: {res.error}")
        self.assertIn("Goodman-Bacon", res.output)
        self.assertAlmostEqual(interp.registers.get_r("twfe"), r["twfe"], places=5)

    def test_06_drdid_analytical_benchmark(self):
        """Verify Sant'Anna & Zhao (2020) Doubly Robust Difference-in-Differences."""
        from engine.causal_modern import compute_drdid
        rng = np.random.RandomState(42)
        rows = []
        # 200 units, 2 periods (0 and 1)
        for uid in range(1, 201):
            x = float(rng.normal(1.0, 0.5))
            ps = 1.0 / (1.0 + np.exp(-( -0.5 + 0.8 * x)))
            is_treated = 1 if float(rng.uniform(0, 1)) < ps else 0
            if uid <= 20: is_treated = 1
            elif uid <= 40: is_treated = 0

            for tid in [0, 1]:
                d = is_treated if tid == 1 else 0
                y = 5.0 + 3.0 * d + 1.5 * x + 0.8 * tid + float(rng.normal(0, 0.3))
                rows.append({"id": uid, "t": tid, "treated": is_treated, "y": y, "x": x})
        df = pd.DataFrame(rows)

        # 1. Engine calculation
        r = compute_drdid(df, "y", "treated", "t", covars=["x"], unitvar="id")
        self.assertAlmostEqual(r["att"], 3.0, delta=0.35)
        self.assertGreater(r["se"], 0.0)
        self.assertLess(r["p"], 0.05)

        # 2. Interpreter command execution
        ds = Dataset()
        ds.load_dataframe(df)
        interp = Interpreter(ds)
        res = interp.execute("drdid y treated t x, ivar(id)")
        self.assertTrue(res.ok, f"drdid command failed: {res.error}")
        self.assertIn("Doubly Robust Difference-in-Differences", res.output)
        self.assertAlmostEqual(interp.registers.get_r("att"), r["att"], places=5)
        self.assertAlmostEqual(interp.registers.b["ATT"], r["att"], places=5)


if __name__ == "__main__":
    unittest.main()
