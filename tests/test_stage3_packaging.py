"""
tests/test_stage3_packaging.py - Verification tests for Stage 3 Packaging & Installability
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

from assets import get_asset_path
import gui_qt
from gui_qt.main_window import PotatosticsQtWindow
import main
from scripts.create_clean_release_archive import get_candidate_files, is_file_allowed


class TestStage3Packaging(unittest.TestCase):
    def test_gui_qt_package_structure(self):
        """gui_qt must be a valid package exposing PotatosticsQtWindow."""
        self.assertTrue(hasattr(gui_qt, "PotatosticsQtWindow"))
        self.assertEqual(gui_qt.PotatosticsQtWindow, PotatosticsQtWindow)

    def test_asset_resolution(self):
        """All critical assets must resolve to existing files on disk."""
        ico = get_asset_path("potatostics.ico")
        self.assertIsNotNone(ico, "potatostics.ico must be found via get_asset_path")
        self.assertTrue(os.path.exists(ico))

        png = get_asset_path("potatostics-app-icon-32.png")
        self.assertIsNotNone(png, "potatostics-app-icon-32.png must be found")
        self.assertTrue(os.path.exists(png))

        sample_csv = get_asset_path("sample_data.csv")
        self.assertIsNotNone(sample_csv, "sample_data.csv must be found")
        self.assertTrue(os.path.exists(sample_csv))

    def test_entry_point_module(self):
        """main.py must expose main() callable entry point."""
        self.assertTrue(hasattr(main, "main"))
        self.assertTrue(callable(main.main))

    def test_built_wheel_archive_contents(self):
        """Built distribution wheel must contain main.py, gui_qt, assets, and entry_points."""
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "source"
            wheels = Path(tmp) / "wheels"
            for relative in get_candidate_files(repo_root):
                self.assertFalse(relative.is_absolute())
                self.assertTrue(is_file_allowed(relative, repo_root, strict=True), relative)
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repo_root / relative, target)

            result = subprocess.run(
                [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(wheels), "."],
                cwd=stage, capture_output=True, text=True, errors="replace",
            )
            self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}")
            whls = list(wheels.glob("*.whl"))
            self.assertEqual(len(whls), 1, "Expected one distribution wheel")

            with zipfile.ZipFile(whls[0], 'r') as z:
                names = z.namelist()
                self.assertIn("main.py", names, "main.py must be in root of wheel")
                self.assertTrue(any(n.startswith("gui_qt/") for n in names), "gui_qt must be in wheel")
                self.assertTrue(any("potatostics.ico" in n for n in names), "potatostics.ico must be in wheel")
                self.assertTrue(any("sample_data.csv" in n for n in names), "sample_data.csv must be in wheel")

                ep = [n for n in names if "entry_points.txt" in n]
                self.assertTrue(len(ep) > 0, "entry_points.txt must be present")
                ep_content = z.read(ep[0]).decode('utf-8')
                self.assertIn("potatostics = main:main", ep_content)

if __name__ == "__main__":
    unittest.main()
