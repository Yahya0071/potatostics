"""
tests/test_stage34_gui_menus.py - Reference 15 Parity GUI Menus Test Suite
Verifies that PotatosticsQtWindow provides comprehensive hierarchical menus
under &Statistics and &Graphics matching Reference 15's econometric workflows.
"""

import unittest
from PySide6 import QtWidgets, QtCore, QtGui
from gui_qt.main_window import PotatosticsQtWindow

app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(["--platform", "offscreen"])


class TestStage34GuiMenus(unittest.TestCase):
    def setUp(self):
        self.win = PotatosticsQtWindow()

    def tearDown(self):
        self.win.close()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            app.processEvents()

    def test_01_graphics_menu_hierarchy_and_coverage(self):
        """Verify Graphics menu contains structured submenus and quick-launch actions."""
        self.assertTrue(hasattr(self.win, "graphics_menu"))
        g_menu = self.win.graphics_menu
        self.assertIsNotNone(g_menu)

        # Check submenus
        submenus = [m.title() for m in g_menu.findChildren(QtWidgets.QMenu)]
        self.assertTrue(any("Two-way" in s for s in submenus), f"Missing Two-way submenu: {submenus}")
        self.assertTrue(any("Distribution" in s for s in submenus), f"Missing Distribution submenu: {submenus}")
        self.assertTrue(any("Causal" in s or "Econometric" in s for s in submenus), f"Missing Causal/Econometric submenu: {submenus}")

        # Check action coverage
        all_actions = [a.text() for a in g_menu.findChildren(QtGui.QAction) if a.text()]
        self.assertTrue(any("scatter" in a.lower() for a in all_actions), "Missing scatter plot action")
        self.assertTrue(any("line" in a.lower() for a in all_actions), "Missing line plot action")
        self.assertTrue(any("binscatter" in a.lower() for a in all_actions), "Missing binscatter action")
        self.assertTrue(any("rdplot" in a.lower() for a in all_actions), "Missing rdplot action")
        self.assertTrue(any("irf" in a.lower() for a in all_actions), "Missing irf action")
        self.assertTrue(any("histogram" in a.lower() for a in all_actions), "Missing histogram action")
        self.assertTrue(any("boxplot" in a.lower() or "box" in a.lower() for a in all_actions), "Missing boxplot action")
        self.assertTrue(any("synth_plot" in a.lower() for a in all_actions), "Missing synth_plot action")
        self.assertTrue(any("coefplot" in a.lower() for a in all_actions), "Missing coefplot action")

    def test_02_statistics_menu_hierarchical_submenus(self):
        """Verify Statistics menu contains Reference 15 structured categories."""
        self.assertTrue(hasattr(self.win, "stat_menu"))
        s_menu = self.win.stat_menu
        self.assertIsNotNone(s_menu)

        submenus = [m.title() for m in s_menu.findChildren(QtWidgets.QMenu)]
        
        # Verify key Reference 15 econometric categories
        expected_categories = [
            "Summaries, tables",
            "Linear models",
            "Binary and ordinal",
            "Count outcomes",
            "Censored and quantile",
            "Endogenous covariates",
            "Longitudinal",
            "Time series",
            "Survival analysis",
            "Survey data",
            "causal inference",
            "Post-estimation",
        ]
        for exp in expected_categories:
            self.assertTrue(
                any(exp.lower() in s.lower() for s in submenus),
                f"Missing Reference 15 category '{exp}' in Statistics menu. Found: {submenus}"
            )

    def test_03_statistics_menu_action_coverage(self):
        """Verify that underlying estimators and diagnostics are exposed in menu actions."""
        s_menu = self.win.stat_menu
        all_actions = [a.text() for a in s_menu.findChildren(QtGui.QAction) if a.text()]
        
        # Core Reference estimators must be reachable
        required_commands = [
            "summarize", "tabulate", "correlate", "esttab",
            "regress", "reghdfe", "cnsreg", "sureg", "bigregress",
            "logit", "probit", "ologit", "mlogit", "brant",
            "poisson", "nbreg",
            "tobit", "qreg",
            "ivregress", "weakiv",
            "xtreg", "hausman",
            "tsset", "tsfilter hp", "var", "dfuller",
            "stset", "sts list", "stcox",
            "svyset", "svy: regress",
            "csdid", "synth", "eventstudy", "bacondecomp", "psmatch",
            "margins", "lincom", "nlcom", "testnl"
        ]
        for cmd in required_commands:
            self.assertTrue(
                any(cmd.lower() in a.lower() for a in all_actions),
                f"Command '{cmd}' not found in any Statistics menu action. Actions: {all_actions[:10]}..."
            )

    def test_04_busy_state_disables_graphics_and_statistics(self):
        """Verify that background command execution serializes and disables graphics & stat menus."""
        self.win._set_busy_state(True)
        self.assertFalse(self.win.stat_menu.isEnabled())
        self.assertFalse(self.win.graphics_menu.isEnabled())
        self.assertFalse(self.win.data_menu.isEnabled())

        self.win._set_busy_state(False)
        self.assertTrue(self.win.stat_menu.isEnabled())
        self.assertTrue(self.win.graphics_menu.isEnabled())
        self.assertTrue(self.win.data_menu.isEnabled())


if __name__ == "__main__":
    unittest.main()
