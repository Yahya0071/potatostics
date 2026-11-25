"""
tests/test_stage6_plugin_trust.py - Tests for Stage 6 Plugin Trust Model & Opt-out Controls
"""

import os
import shutil
import tempfile
import unittest

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.plugins_manager import PluginsManager, SECURITY_WARNING


class TestStage6PluginTrust(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ds = Dataset()
        self.interp = Interpreter(self.ds)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_plugin_security_warning_present(self):
        """PluginsManager must declare an explicit security warning regarding un-sandboxed execution."""
        self.assertIn("SECURITY WARNING", SECURITY_WARNING)
        self.assertIn("full user privileges", SECURITY_WARNING)
        self.assertIn("arbitrary Python", SECURITY_WARNING)

    def test_plugin_explicit_enable_and_disable(self):
        """Newly created plugins must not load unless explicitly activated."""
        plugin_file = os.path.join(self.temp_dir, "my_custom_plugin.py")
        with open(plugin_file, "w", encoding="utf-8") as f:
            f.write(
                "def register_plugin(interp):\n"
                "    interp.custom_registered = True\n"
            )

        pm = PluginsManager(self.interp, plugins_dir=self.temp_dir)
        discovered = pm.discover_plugins()
        self.assertIn("my_custom_plugin", discovered)
        self.assertFalse(discovered["my_custom_plugin"]["enabled"])

        # Not enabled -> not loaded
        loaded = pm.load_all_plugins()
        self.assertEqual(len(loaded), 0)
        self.assertFalse(hasattr(self.interp, "custom_registered"))

        # Explicitly enable -> now loaded
        pm.enable_plugin("my_custom_plugin")
        self.assertTrue(pm.is_enabled("my_custom_plugin"))
        loaded = pm.load_all_plugins()
        self.assertIn("my_custom_plugin", loaded)
        self.assertTrue(getattr(self.interp, "custom_registered", False))

        # Disable
        pm.disable_plugin("my_custom_plugin")
        self.assertFalse(pm.is_enabled("my_custom_plugin"))

    def test_plugin_opt_out_setting(self):
        """PluginsManager with enabled=False must not load any plugins."""
        plugin_file = os.path.join(self.temp_dir, "auto_plugin.py")
        with open(plugin_file, "w", encoding="utf-8") as f:
            f.write("def register_plugin(interp): interp.auto_reg = True\n")

        pm = PluginsManager(self.interp, plugins_dir=self.temp_dir, enabled=False)
        pm.enable_plugin("auto_plugin")
        loaded = pm.load_all_plugins()
        self.assertEqual(len(loaded), 0)
        self.assertFalse(hasattr(self.interp, "auto_reg"))

    def test_plugin_robust_loader_handling(self):
        """Corrupted plugin file with syntax error must not crash the loader."""
        bad_plugin = os.path.join(self.temp_dir, "bad_syntax.py")
        with open(bad_plugin, "w", encoding="utf-8") as f:
            f.write("def def bad syntax ::::\n")

        pm = PluginsManager(self.interp, plugins_dir=self.temp_dir)
        pm.enable_plugin("bad_syntax")
        loaded = pm.load_all_plugins()
        self.assertNotIn("bad_syntax", loaded)
        info = pm.discover_plugins()
        self.assertIn("bad_syntax", info)

    def test_interpreter_plugin_commands(self):
        """Interpreter must support 'plugin list' displaying the security warning banner."""
        self.interp.plugins_mgr.plugins_dir = self.temp_dir
        res_list = self.interp.execute("plugin list")
        self.assertTrue(res_list.ok)
        self.assertIn("SECURITY WARNING", res_list.output)
        self.assertIn("Plugin Name", res_list.output)


if __name__ == "__main__":
    unittest.main()
