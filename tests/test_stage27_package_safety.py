import os
import shutil
import tempfile
import hashlib
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
        "description": "Test winsorize plugin.",
        "code": (
            "from engine.parser import CommandResult\n"
            "def winsorize_cmd(interp, cmd):\n"
            "    return CommandResult('winsorized', ok=True)\n"
            "def register_plugin(interp):\n"
            "    interp.register_command('winsorize', winsorize_cmd)\n"
        ),
    }
}

class TestPackageSafety(unittest.TestCase):
    def setUp(self):
        self._cat_patch = mock.patch.dict(COMMUNITY_CATALOG, TEST_CATALOG, clear=True)
        self._cat_patch.start()
        self.addCleanup(self._cat_patch.stop)
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = Dataset()
        self.dataset.load_dataframe(pd.DataFrame({"price": [100, 200, 300, 400, 500]}))
        self.interp = Interpreter(self.dataset)
        self.interp.custom_commands.clear()
        self.pm = PluginsManager(self.interp, plugins_dir=self.temp_dir, enabled=True)
        self.interp.plugins_mgr = self.pm

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_pkg_install_default_not_enabled(self):
        """pkg install and ssc install must install-without-enabling by default."""
        res = self.interp.execute("pkg install winsorize")
        self.assertTrue(res.ok, f"pkg install failed: {res.error}")
        
        # File must exist
        target_file = os.path.join(self.temp_dir, "winsorize.py")
        self.assertTrue(os.path.exists(target_file))

        # Must NOT be enabled by default
        self.assertFalse(self.pm.is_enabled("winsorize"))

        # Command must not be callable yet
        res_cmd = self.interp.execute("winsorize price")
        self.assertFalse(res_cmd.ok)
        self.assertEqual(res_cmd.return_code, 199)

        # Output must show file size, SHA-256, and typed enable command
        self.assertIn("SHA-256", res_inst_out := res.output)
        self.assertIn("pkg enable winsorize", res_inst_out)

        # Now explicitly enable via typed command
        res_enable = self.interp.execute("pkg enable winsorize")
        self.assertTrue(res_enable.ok)
        self.assertTrue(self.pm.is_enabled("winsorize"))

        # Command should now execute successfully
        res_cmd2 = self.interp.execute("winsorize price")
        self.assertTrue(res_cmd2.ok, f"winsorize failed: {res_cmd2.error}")

    def test_02_pkg_install_rejects_insecure_urls(self):
        """Only https:// URLs are allowed. http:// must be rejected."""
        res_http = self.interp.execute("pkg install http://example.com/plugin.py")
        self.assertFalse(res_http.ok)
        self.assertIn("https", res_http.error.lower())

    def test_03_pkg_install_sha256_verification(self):
        """Optional sha256(...) option must verify file digest and reject mismatches."""
        code = COMMUNITY_CATALOG["winsorize"]["code"]
        expected_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()

        # Bad hash must fail and not install
        res_bad = self.interp.execute("pkg install winsorize, sha256(0000000000000000000000000000000000000000000000000000000000000000)")
        self.assertFalse(res_bad.ok)
        self.assertIn("sha", res_bad.error.lower())
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir, "winsorize.py")))

        # Correct hash must succeed
        res_good = self.interp.execute(f"pkg install winsorize, sha256({expected_sha})")
        self.assertTrue(res_good.ok, f"pkg install failed: {res_good.error}")
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "winsorize.py")))

    def test_04_no_code_execution_during_install(self):
        """Downloaded/installed plugin must never execute Python code during download or install."""
        bomb_code = (
            "raise RuntimeError('MALICIOUS_EXECUTION_TRIGGERED')\n"
            "def register_plugin(interp):\n"
            "    pass\n"
        )
        bomb_path = os.path.join(self.temp_dir, "remote_bomb.py")
        with open(bomb_path, "w", encoding="utf-8") as f:
            f.write(bomb_code)

        # Installing the file must parse AST but NOT execute module code
        try:
            res = self.pm.install_plugin(bomb_path, target_name="safe_bomb", enable=False)
            self.assertEqual(res["name"], "safe_bomb")
        except RuntimeError:
            self.fail("Code was executed during install_plugin!")

    def test_05_path_traversal_prevention(self):
        """Sanitize plugin names to prevent directory traversal across install, enable, disable, uninstall."""
        # Create a victim file outside plugins directory
        victim_path = os.path.join(self.temp_dir, "..", "victim.py")
        with open(victim_path, "w", encoding="utf-8") as f:
            f.write("# victim file")

        try:
            # Attempt uninstall ../victim
            res = self.interp.execute("pkg uninstall ../victim")
            self.assertFalse(res.ok)
            self.assertTrue(os.path.exists(victim_path), "Victim file was deleted by traversal attack!")

            # Attempt enable ../victim
            res_en = self.interp.execute("pkg enable ../victim")
            self.assertFalse(res_en.ok)

            # Attempt plugin new ../victim
            res_new = self.interp.execute("plugin new ../victim")
            self.assertFalse(res_new.ok)
        finally:
            if os.path.exists(victim_path):
                os.remove(victim_path)

    def test_06_plugin_name_sanitization(self):
        """Only [a-z0-9_]+ names are allowed."""
        invalid_names = ["bad name", "bad-name", "bad!name", "bad/name", "bad\\name", "bad.name"]
        for bad in invalid_names:
            with self.subTest(name=bad):
                with self.assertRaises(ValueError):
                    self.pm.validate_plugin_name(bad)
