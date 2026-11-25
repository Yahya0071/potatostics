import os
import shutil
import tempfile
import unittest
import pandas as pd

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.plugins_manager import PluginsManager


class TestScriptSafety(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = Dataset()
        self.dataset.load_dataframe(pd.DataFrame({"price": [100.0, 200.0, 300.0, 400.0, 500.0]}))
        self.interp = Interpreter(self.dataset)
        self.interp.custom_commands.clear()
        self.pm = PluginsManager(self.interp, plugins_dir=self.temp_dir, enabled=True)
        self.interp.plugins_mgr = self.pm

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_dofile_cannot_enable_or_install_plugins_by_default(self):
        """A do-file containing pkg install followed by pkg enable must refuse to run and not execute plugin."""
        # Create a local plugin file to install
        sample_code = (
            "from engine.parser import CommandResult\n"
            "def secret_tool_cmd(interp, cmd):\n"
            "    return CommandResult('TOOL_ACTIVATED', ok=True)\n"
            "def register_plugin(interp):\n"
            "    interp.register_command('secret_tool', secret_tool_cmd)\n"
        )
        sample_path = os.path.join(self.temp_dir, "secret_tool.py").replace("\\", "/")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(sample_code)

        # Create do-file trying to install and enable
        dofile_path = os.path.join(self.temp_dir, "test_install.do").replace("\\", "/")
        with open(dofile_path, "w", encoding="utf-8") as f:
            f.write(
                f'pkg install "{sample_path}"\n'
                f'pkg enable secret_tool\n'
                f'secret_tool\n'
            )

        res = self.interp.execute(f'do "{dofile_path}"')
        
        # Do-file execution must fail on pkg install / enable
        self.assertFalse(res.ok)
        self.assertIn("cannot be executed from a do-file or script", res.output.lower())
        
        # Plugin must NOT have been enabled or registered
        self.assertFalse(self.pm.is_enabled("secret_tool"))
        self.assertNotIn("secret_tool", self.interp.custom_commands)

        # secret_tool must not run
        res_cmd = self.interp.execute("secret_tool")
        self.assertFalse(res_cmd.ok)

    def test_02_interactive_install_and_enable_still_works(self):
        """Interactive console execution of pkg install and pkg enable must still work."""
        sample_code = (
            "from engine.parser import CommandResult\n"
            "def interact_tool_cmd(interp, cmd):\n"
            "    return CommandResult('INTERACTIVE_SUCCESS', ok=True)\n"
            "def register_plugin(interp):\n"
            "    interp.register_command('interact_tool', interact_tool_cmd)\n"
        )
        sample_path = os.path.join(self.temp_dir, "interact_tool.py").replace("\\", "/")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(sample_code)

        # Interactive command (not inside do-file)
        res_inst = self.interp.execute(f'pkg install "{sample_path}"')
        self.assertTrue(res_inst.ok, f"Interactive pkg install failed: {res_inst.error}")

        res_en = self.interp.execute("pkg enable interact_tool")
        self.assertTrue(res_en.ok, f"Interactive pkg enable failed: {res_en.error}")

        # Command is now callable
        res_run = self.interp.execute("interact_tool")
        self.assertTrue(res_run.ok)
        self.assertIn("INTERACTIVE_SUCCESS", res_run.output)

    def test_03_setting_allow_script_plugins_stored_in_config(self):
        """Setting 'allow plugin enabling from scripts' allows scripts to install/enable when explicitly turned on."""
        # By default it is off
        self.assertFalse(self.pm.get_allow_script_plugins())

        # Turn it on
        res_set = self.interp.execute("set allow_script_plugins on")
        self.assertTrue(res_set.ok)
        self.assertTrue(self.pm.get_allow_script_plugins())
        self.assertIn("SECURITY WARNING", res_set.output)

        # Create plugin and do-file
        sample_code = (
            "from engine.parser import CommandResult\n"
            "def allowed_script_cmd(interp, cmd):\n"
            "    return CommandResult('SCRIPT_ALLOWED_SUCCESS', ok=True)\n"
            "def register_plugin(interp):\n"
            "    interp.register_command('allowed_script', allowed_script_cmd)\n"
        )
        sample_path = os.path.join(self.temp_dir, "allowed_script.py").replace("\\", "/")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(sample_code)

        dofile_path = os.path.join(self.temp_dir, "run_allowed.do").replace("\\", "/")
        with open(dofile_path, "w", encoding="utf-8") as f:
            f.write(
                f'pkg install "{sample_path}"\n'
                f'pkg enable allowed_script\n'
                f'allowed_script\n'
            )

        res_do = self.interp.execute(f'do "{dofile_path}"')
        self.assertTrue(res_do.ok, f"Do-file with allow_script_plugins on failed: {res_do.error}")
        self.assertIn("SCRIPT_ALLOWED_SUCCESS", res_do.output)

        # Turn it back off and verify it persists
        res_off = self.interp.execute("set allow_script_plugins off")
        self.assertTrue(res_off.ok)
        self.assertFalse(self.pm.get_allow_script_plugins())

    def test_04_script_runner_and_cli_mode_blocked(self):
        """Direct script_runner.run_script and CLI script mode block plugin enable/install."""
        script_code = "pkg install winsorize\npkg enable winsorize\n"
        res = self.interp.script_runner.run_script(script_code)
        self.assertFalse(res.ok)
        self.assertIn("Security restriction", res.output)

        # CLI script mode
        self.interp.cli_script_mode = True
        try:
            res_cli = self.interp.execute("pkg install winsorize")
            self.assertFalse(res_cli.ok)
            self.assertIn("Security restriction", res_cli.output)
        finally:
            self.interp.cli_script_mode = False
