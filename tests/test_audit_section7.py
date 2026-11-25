"""
tests/test_audit_section7.py - TDD Verification Suite for Section 7 (Hygiene, Cleanup & Packaging)
"""

import os
import unittest
import platformdirs
import pandas as pd

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.plugins_manager import PluginsManager


class TestSection7Hygiene(unittest.TestCase):
    def setUp(self):
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset = Dataset()
        self.interp = Interpreter(self.dataset)

    def test_7a_temp_dir_deleted_and_gitignore_exists(self):
        """Literal %TEMP% directory must be deleted, and .gitignore must exist with build/dist/pycache."""
        temp_dir = os.path.join(self.root, "%TEMP%")
        self.assertFalse(os.path.exists(temp_dir), msg="Literal '%TEMP%' directory must be deleted")

        gitignore_path = os.path.join(self.root, ".gitignore")
        self.assertTrue(os.path.exists(gitignore_path), msg=".gitignore must exist")
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            content = f.read()
        for item in ["build/", "dist/", "__pycache__/"]:
            self.assertIn(item, content, msg=f".gitignore must contain {item}")

    def test_7b_pyproject_toml_exists_with_dependencies(self):
        """pyproject.toml must exist with project metadata and dependencies."""
        pyproject_path = os.path.join(self.root, "pyproject.toml")
        self.assertTrue(os.path.exists(pyproject_path), msg="pyproject.toml must exist")
        with open(pyproject_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn("potatostics", content.lower())
        self.assertIn("dependencies", content.lower())
        self.assertIn("pyside6", content.lower())

    def test_7c_plugins_loaded_from_platformdirs(self):
        """PluginsManager must load plugins from platformdirs.user_config_dir('potatostics')."""
        pm = PluginsManager(self.interp)
        expected_base = platformdirs.user_config_dir("potatostics")
        self.assertTrue(pm.plugins_dir.startswith(expected_base),
                        msg=f"Plugins directory {pm.plugins_dir} must be under {expected_base}")

    def test_7d_estimation_py_renamed(self):
        """engine/estimation.py must exist and provide estimation functions."""
        from engine.estimation import compute_regression, compute_logit, compute_ivregress_2sls
        self.assertTrue(callable(compute_regression))
        self.assertTrue(callable(compute_logit))
        self.assertTrue(callable(compute_ivregress_2sls))

    def test_7e_estimates_export_uses_exporters_and_old_deleted(self):
        """estimates export must use engine/exporters.py, and old export_latex in estimates.py is deleted."""
        from engine.estimates import EstimatesStore
        store = EstimatesStore()
        self.assertFalse(hasattr(store, 'export_latex'),
                         msg="Old export_latex method must be deleted from EstimatesStore")

        self.dataset.df = pd.DataFrame({'y': [1, 2, 3, 4], 'x': [2, 4, 6, 8]})
        self.interp.execute("regress y x")
        out_dir = os.path.join(self.root, "tests", "test_outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_tex = os.path.join(out_dir, "test_export.tex")
        res = self.interp.execute(f'estimates export "{out_tex}"')
        self.assertTrue(res.ok)
        self.assertTrue(os.path.exists(out_tex))
        with open(out_tex, 'r', encoding='utf-8') as f:
            tex_content = f.read()
        self.assertIn(r"\toprule", tex_content)
        if os.path.exists(out_tex):
            os.remove(out_tex)
        if os.path.exists(out_dir) and not os.listdir(out_dir):
            os.rmdir(out_dir)


if __name__ == '__main__':
    unittest.main()
