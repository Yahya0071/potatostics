import sys
import os
import unittest

class BlockedModulesFinder:
    def __init__(self, blocked_modules):
        self.blocked_modules = set(blocked_modules)

    def find_spec(self, fullname, path=None, target=None):
        top_name = fullname.split('.')[0]
        if top_name in self.blocked_modules:
            raise ImportError(f"Test blocked import of forbidden module: {fullname}")
        return None

class TestCleanInstallIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["POTATOSTICS_TESTING"] = "1"
    def test_assets_zero_dependencies(self):
        """assets.py must be importable with tkinter, customtkinter, and PIL strictly blocked."""
        forbidden = {'tkinter', '_tkinter', 'customtkinter', 'PIL'}
        finder = BlockedModulesFinder(forbidden)
        sys.meta_path.insert(0, finder)
        try:
            # Unload assets if present
            if 'assets' in sys.modules:
                del sys.modules['assets']
            import assets
            self.assertTrue(callable(assets.get_asset_path))
            path = assets.get_asset_path("sample_data.csv")
            self.assertIsNotNone(path)
        finally:
            if finder in sys.meta_path:
                sys.meta_path.remove(finder)

    def test_gui_qt_does_not_import_legacy_gui_or_tkinter(self):
        """
        gui_qt must not import anything from the gui (Tk) package or tkinter/customtkinter.
        """
        forbidden = {'tkinter', '_tkinter', 'customtkinter', 'gui'}
        
        # Save old sys.modules and sys.meta_path
        old_meta_path = list(sys.meta_path)
        saved_modules = {k: v for k, v in sys.modules.items() if any(k.startswith(f) for f in forbidden)}
        for k in list(saved_modules.keys()):
            del sys.modules[k]

        finder = BlockedModulesFinder(forbidden)
        sys.meta_path.insert(0, finder)

        try:
            from gui_qt.main_window import launch_qt_app, PotatosticsQtWindow
            self.assertTrue(callable(launch_qt_app))
            self.assertTrue(issubclass(PotatosticsQtWindow, object))
        finally:
            sys.meta_path[:] = old_meta_path
            sys.modules.update(saved_modules)

    def test_main_startup_error_logging(self):
        """main.py must log startup errors to potatostics_startup.log."""
        import main
        log_path = main.STARTUP_LOG_FILE
        if os.path.exists(log_path):
            os.remove(log_path)

        main._log_and_show_error("Test Startup Error", "Simulated failure for verification", exc=RuntimeError("Simulated"))
        self.assertTrue(os.path.exists(log_path))
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Test Startup Error", content)
            self.assertIn("Simulated failure for verification", content)
            self.assertIn("RuntimeError: Simulated", content)

        # Cleanup
        if os.path.exists(log_path):
            os.remove(log_path)

if __name__ == '__main__':
    unittest.main()
