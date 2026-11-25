"""
tests/test_stage25_package_hub.py - Community Package & Plugin Hub (Package Hub / net install Equivalent)
Tests:
1. plugin new <name>: Scaffolds a starter Python plugin template.
2. Dynamic Command Registration: Custom plugins register verbs callable from Potatostics command line.
3. pkg search: Searches community econometric packages.
4. pkg install / ssc install: Downloads/installs plugin, verifies AST syntax, hot-loads into interpreter.
5. pkg list & pkg uninstall: Lists and removes installed packages.
"""

import ast
import os
import shutil
import tempfile
import unittest
import pandas as pd
from unittest import mock

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.plugins_manager import PluginsManager, COMMUNITY_CATALOG

TEST_CATALOG = {
    "winsorize": {
        "name": "winsorize",
        "version": "1.0.0",
        "author": "Test Community",
        "description": "Winsorizes numeric variables at specified percentiles.",
        "when_to_use": "Use to cap outliers.",
        "syntax": "winsorize varlist [, cuts(lower upper)]",
        "example": "winsorize price, cuts(10 90)",
        "code": (
            "import numpy as np\n"
            "from engine.parser import CommandResult\n"
            "def winsorize_cmd(interp, cmd):\n"
            "    tokens = cmd.body.split()\n"
            "    for v in tokens:\n"
            "        if v in interp.dataset.df.columns:\n"
            "            lo, hi = np.percentile(interp.dataset.df[v].dropna(), [10, 90])\n"
            "            interp.dataset.df[v] = np.clip(interp.dataset.df[v], lo, hi)\n"
            "    return CommandResult(f'Variables {tokens} winsorized.', ok=True)\n"
            "def register_plugin(interp):\n"
            "    interp.register_command('winsorize', winsorize_cmd)\n"
        ),
    },
    "bacondecomp": {
        "name": "bacondecomp",
        "version": "1.0.0",
        "author": "Test Community",
        "description": "Test DiD decomposition package.",
        "code": "def register_plugin(interp): pass\n",
    },
}


class TestPackageHub(unittest.TestCase):
    def setUp(self):
        self._cat_patch = mock.patch.dict(COMMUNITY_CATALOG, TEST_CATALOG, clear=True)
        self._cat_patch.start()
        self.addCleanup(self._cat_patch.stop)
        self.temp_dir = tempfile.mkdtemp()
        self.df = pd.DataFrame({
            'price': [100.0, 200.0, 300.0, 400.0, 500.0],
            'mpg': [20.0, 25.0, 18.0, 30.0, 22.0]
        })
        self.ds = Dataset()
        self.ds.load_dataframe(self.df)
        self.interp = Interpreter(self.ds)
        self.interp.custom_commands.clear()
        # Point plugins manager to our isolated temp directory
        self.pm = PluginsManager(self.interp, plugins_dir=self.temp_dir)
        self.interp.plugins_mgr = self.pm

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_create_plugin_template(self):
        """Verify 'plugin new <name>' scaffolds a syntactically valid Python plugin template."""
        res = self.interp.execute("plugin new shrinkage")
        self.assertTrue(res.ok, f"plugin new failed: {res.error}")
        self.assertIn("created", res.output.lower())

        target_file = os.path.join(self.temp_dir, "shrinkage.py")
        self.assertTrue(os.path.exists(target_file), f"File {target_file} was not created.")

        with open(target_file, "r", encoding="utf-8") as f:
            code = f.read()

        # Must parse cleanly as Python AST
        parsed = ast.parse(code)
        self.assertIsNotNone(parsed)
        self.assertIn("register_plugin", code)

    def test_02_dynamic_command_execution(self):
        """Verify custom Python plugin registers a new command verb that can read data and set registers."""
        plugin_code = (
            "from engine.parser import CommandResult\n"
            "def my_mean(interp, cmd):\n"
            "    tokens = cmd.body.split()\n"
            "    if not tokens:\n"
            "        return CommandResult('syntax: my_mean varname', ok=False, return_code=198)\n"
            "    v = tokens[0]\n"
            "    val = float(interp.dataset.df[v].mean())\n"
            "    interp.registers.clear_r()\n"
            "    interp.registers.set_r_scalar('mean', val)\n"
            "    return CommandResult(f'Custom mean of {v} = {val:.2f}', ok=True)\n"
            "\n"
            "def register_plugin(interp):\n"
            "    interp.register_command('my_mean', my_mean)\n"
        )
        plugin_path = os.path.join(self.temp_dir, "my_mean.py")
        with open(plugin_path, "w", encoding="utf-8") as f:
            f.write(plugin_code)

        # Enable and load
        self.pm.enable_plugin("my_mean")
        self.pm.load_all_plugins()

        # Execute custom command
        res = self.interp.execute("my_mean price")
        self.assertTrue(res.ok, f"my_mean failed: {res.error}")
        self.assertIn("Custom mean of price = 300.00", res.output)
        self.assertEqual(self.interp.registers.get_r('mean'), 300.0)

    def test_03_pkg_search_and_list(self):
        """Verify 'pkg search' queries community packages and 'pkg list' lists installed."""
        res_search = self.interp.execute("pkg search winsor")
        self.assertTrue(res_search.ok, f"pkg search failed: {res_search.error}")
        self.assertIn("winsor", res_search.output.lower())

        res_list = self.interp.execute("pkg list")
        self.assertTrue(res_list.ok, f"pkg list failed: {res_list.error}")
        self.assertIn("Plugin Name", res_list.output)

    def test_04_pkg_install_and_explicit_enable(self):
        """Verify 'pkg install <name>' installs without enabling, and 'pkg enable <name>' activates command."""
        # Install 'winsorize' from built-in community catalog
        res_inst = self.interp.execute("pkg install winsorize")
        self.assertTrue(res_inst.ok, f"pkg install failed: {res_inst.error}")
        self.assertIn("installed (disabled)", res_inst.output.lower())
        self.assertFalse(self.pm.is_enabled("winsorize"))

        # Explicitly enable
        res_en = self.interp.execute("pkg enable winsorize")
        self.assertTrue(res_en.ok, f"pkg enable failed: {res_en.error}")
        self.assertTrue(self.pm.is_enabled("winsorize"))

        # 'winsorize' should now be callable!
        res_run = self.interp.execute("winsorize price, cuts(10 90)")
        self.assertTrue(res_run.ok, f"winsorize failed: {res_run.error}")
        self.assertIn("winsorized", res_run.output.lower())

    def test_05_package_compat_ssc_and_net_install(self):
        """Verify Reference 'ssc install' and 'net install' syntax work as aliases."""
        res_ssc = self.interp.execute("ssc install bacondecomp")
        self.assertTrue(res_ssc.ok, f"ssc install failed: {res_ssc.error}")
        self.assertIn("bacondecomp", res_ssc.output.lower())

        # Test net install from local file
        sample_pkg = os.path.join(self.temp_dir, "custom_net.py")
        with open(sample_pkg, "w", encoding="utf-8") as f:
            f.write("def register_plugin(interp): pass\n")

        res_net = self.interp.execute(f"net install custom_net, from('{sample_pkg}')")
        self.assertTrue(res_net.ok, f"net install failed: {res_net.error}")
        self.assertIn("custom_net", res_net.output.lower())

    def test_06_pkg_uninstall(self):
        """Verify 'pkg uninstall <name>' removes plugin and unregisters commands."""
        # First install
        self.interp.execute("pkg install winsorize")
        self.assertIn("winsorize", self.pm.discover_plugins())

        # Uninstall
        res_un = self.interp.execute("pkg uninstall winsorize")
        self.assertTrue(res_un.ok, f"pkg uninstall failed: {res_un.error}")
        self.assertNotIn("winsorize", self.pm.discover_plugins())

    def test_07_package_manager_gui_dialog(self):
        """Verify PackageManagerDialog builds and renders all 4 tabs, usage card, and paste signal."""
        from PySide6 import QtWidgets
        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from gui_qt.package_dialog import PackageManagerDialog
        dlg = PackageManagerDialog(self.interp)
        self.assertEqual(dlg.tab_widget.count(), 4)
        self.assertGreaterEqual(dlg.catalog_table.rowCount(), 1)
        self.assertTrue(len(dlg.edt_pkg_example.text()) > 0)

        pasted = []
        dlg.command_pasted.connect(pasted.append)
        from unittest import mock
        with mock.patch.object(QtWidgets.QMessageBox, "information"):
            dlg.btn_paste_example.click()
        self.assertEqual(pasted, [dlg.edt_pkg_example.text()])

        dlg.cmb_template.setCurrentIndex(1)
        self.assertIn("_z", dlg.code_edit.toPlainText())
        dlg.close()


if __name__ == "__main__":
    unittest.main()
