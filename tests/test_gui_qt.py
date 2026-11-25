"""
tests/test_gui_qt.py - Unit & Integration Test Suite for PyQt6 Modern Desktop Shell
Covers:
1. VirtualPandasModel: Viewport data rendering, in-place cell editing, column header sorting
2. ResultsConsole: Formatting, LaTeX table generation, Markdown table generation
3. CommandBar: Autocomplete updates, history recall
4. HistoryPane: Adding entries, filtering, status coloring
5. VariablesPane: Display, filtering, properties calculation
6. CommandPaletteDialog (Ctrl+K): Fuzzy action & variable search
7. Code-Bridge Dialogs (RegressionDialog, SummarizeDialog): Live syntax preview verification
8. PotatosticsQtWindow: Full workstation initialization, docking, theme switching, command execution
"""

import unittest
import pandas as pd

from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from engine.dataset import Dataset
from gui_qt.data_grid import VirtualPandasModel
from gui_qt.console import ResultsConsole
from gui_qt.command_bar import CommandBar
from gui_qt.history_pane import HistoryPane
from gui_qt.variables_pane import VariablesPane
from gui_qt.command_palette import CommandPaletteDialog
from gui_qt.dialogs import (
    RegressionDialog, SummarizeDialog, ReghdfeDialog, ProbitDialog, MarginsDialog
)
from gui_qt.main_window import PotatosticsQtWindow

# Ensure single QApplication instance across test runner
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(["--platform", "offscreen"])


class TestVirtualDataGrid(unittest.TestCase):
    """Tests for the virtualized table model and spreadsheet widget."""

    def setUp(self):
        self.ds = Dataset()
        df = pd.DataFrame({
            "price": [5000.0, 6000.0, 7000.0],
            "mpg": [20, 25, 30],
            "make": ["Buick", "Cadillac", "Chevrolet"]
        })
        self.ds.load_dataframe(df)
        self.model = VirtualPandasModel(self.ds)

    def test_01_row_and_column_counts(self):
        self.assertEqual(self.model.rowCount(), 3)
        self.assertEqual(self.model.columnCount(), 3)

    def test_02_data_display_and_tooltips(self):
        # Cell (0, 0) is price 5000.0
        idx_00 = self.model.index(0, 0)
        self.assertEqual(self.model.data(idx_00, Qt.ItemDataRole.DisplayRole), "5000")

        # ToolTip contains variable name and obs number
        tooltip = self.model.data(idx_00, Qt.ItemDataRole.ToolTipRole)
        self.assertIn("price", tooltip)
        self.assertIn("Obs #1", tooltip)

    def test_03_in_place_cell_editing(self):
        idx = self.model.index(1, 0)  # Row 1, Price (originally 6000)
        success = self.model.setData(idx, "8500", Qt.ItemDataRole.EditRole)
        self.assertTrue(success)
        self.assertEqual(self.ds.df.iloc[1, 0], 8500.0)

    def test_04_column_sorting(self):
        # Click header to sort ascending (view only)
        self.model.sort_by_column(0)
        self.assertEqual(list(self.ds.df["price"]), [5000.0, 6000.0, 7000.0])
        v_asc = [self.model.data(self.model.index(r, 0), Qt.ItemDataRole.DisplayRole) for r in range(3)]
        self.assertEqual(v_asc, ["5000", "6000", "7000"])

        # Click again to sort descending (view only)
        self.model.sort_by_column(0)
        self.assertEqual(list(self.ds.df["price"]), [5000.0, 6000.0, 7000.0])
        v_desc = [self.model.data(self.model.index(r, 0), Qt.ItemDataRole.DisplayRole) for r in range(3)]
        self.assertEqual(v_desc, ["7000", "6000", "5000"])


class TestResultsConsole(unittest.TestCase):
    """Tests for monospaced console and table export converters."""

    def setUp(self):
        self.console = ResultsConsole()

    def test_05_append_command_and_output(self):
        self.console.append_command("summarize price")
        self.console.append_output("Variable | Obs Mean\nprice | 3 6000")
        text = self.console.toPlainText()
        self.assertIn(". summarize price", text)
        self.assertIn("price | 3 6000", text)

    def test_06_latex_and_markdown_conversion(self):
        table_sample = (
            "Variable       |      Obs         Mean\n"
            "---------------+----------------------\n"
            "price          |        3    6000.0000\n"
        )
        self.console.append_output(table_sample)
        self.console.selectAll()
        self.console._copy_as_latex()
        cb_text = QtWidgets.QApplication.clipboard().text()
        self.assertIn("\\begin{tabular}", cb_text)
        self.assertIn("\\toprule", cb_text)

        self.console._copy_as_markdown()
        md_text = QtWidgets.QApplication.clipboard().text()
        self.assertIn("| Variable", md_text)
        self.assertIn("|---", md_text)


class TestCommandBarAndHistory(unittest.TestCase):
    """Tests for intelligent command input bar and history review pane."""

    def setUp(self):
        self.ds = Dataset()
        self.ds.load_dataframe(pd.DataFrame({"wage": [15, 25], "educ": [12, 16]}))
        self.cmd_bar = CommandBar(self.ds)
        self.history = HistoryPane()

    def test_07_autocomplete_and_history_recall(self):
        self.cmd_bar.update_autocomplete()
        words = self.cmd_bar.completer_model.stringList()
        self.assertIn("wage", words)
        self.assertIn("educ", words)
        self.assertIn("regress", words)

        # History recall
        self.cmd_bar.entry.add_history("summarize wage")
        self.cmd_bar.entry.add_history("regress wage educ")
        self.assertEqual(len(self.cmd_bar.entry.history), 2)

    def test_08_history_pane_entries_and_filtering(self):
        self.history.add_entry("summarize wage", ok=True, return_code=0)
        self.history.add_entry("regress wage nonexistent", ok=False, return_code=111)
        self.assertEqual(len(self.history.history_records), 2)

        # Filter query
        self.history._filter_history("nonexistent")
        self.assertEqual(self.history.list_widget.count(), 1)


class TestVariablesAndPalette(unittest.TestCase):
    """Tests for variables inspector and spotlight command palette."""

    def setUp(self):
        self.ds = Dataset()
        self.ds.load_dataframe(pd.DataFrame({
            "income": [50000, 60000],
            "age": [30, 40]
        }))
        self.ds.var_labels["income"] = "Annual Household Income"
        self.vars_pane = VariablesPane(self.ds)

    def test_09_variables_pane_properties_and_filtering(self):
        self.vars_pane.refresh()
        self.assertEqual(self.vars_pane.lbl_obs.text(), "2")
        self.assertEqual(self.vars_pane.lbl_vars.text(), "2")
        self.assertEqual(self.vars_pane.tree.topLevelItemCount(), 2)

        # Filter by variable label
        self.vars_pane._filter_variables("household")
        visible = [self.vars_pane.tree.topLevelItem(i).text(0) for i in range(2) if not self.vars_pane.tree.topLevelItem(i).isHidden()]
        self.assertEqual(visible, ["income"])

    def test_10_command_palette_search(self):
        palette = CommandPaletteDialog(self.ds)
        palette._filter_list("vif")
        self.assertGreater(palette.list_widget.count(), 0)
        item_text = palette.list_widget.item(0).text()
        self.assertIn("Variance Inflation Factors", item_text)


class TestCodeBridgeDialogs(unittest.TestCase):
    """Tests for point-and-click code-bridge dialogs."""

    def setUp(self):
        self.ds = Dataset()
        self.ds.load_dataframe(pd.DataFrame({
            "price": [5000, 6000],
            "mpg": [20, 25],
            "weight": [3000, 3200],
            "foreign": [0, 1],
            "cluster_id": [1, 2]
        }))

    def test_11_regression_dialog_live_syntax(self):
        dlg = RegressionDialog(self.ds)
        dlg.cmb_dep.setCurrentText("price")
        # Select mpg and weight
        for i in range(dlg.lst_indep.count()):
            item = dlg.lst_indep.item(i)
            if item.text() in ("mpg", "weight"):
                item.setSelected(True)

        dlg.cmb_vce.setCurrentText("Robust (HC1)")
        dlg.update_syntax()
        self.assertEqual(dlg.txt_syntax.text(), "regress price mpg weight, robust")

        # Clustered VCE
        dlg.cmb_vce.setCurrentText("Clustered SE")
        dlg.cmb_cluster.setCurrentText("cluster_id")
        dlg.update_syntax()
        self.assertEqual(dlg.txt_syntax.text(), "regress price mpg weight, vce(cluster cluster_id)")

    def test_12_summarize_dialog_live_syntax(self):
        dlg = SummarizeDialog(self.ds)
        for i in range(dlg.lst_vars.count()):
            item = dlg.lst_vars.item(i)
            if item.text() == "price":
                item.setSelected(True)
        dlg.chk_detail.setChecked(True)
        dlg.update_syntax()
        self.assertEqual(dlg.txt_syntax.text(), "summarize price, detail")

    def test_clustered_models_require_group_before_submit_or_paste(self):
        for dialog_type in (RegressionDialog, ReghdfeDialog, ProbitDialog):
            with self.subTest(dialog=dialog_type.__name__):
                dlg = dialog_type(self.ds)
                dlg.cmb_dep.setCurrentText("foreign")
                predictors = dlg.lst_indep if isinstance(dlg, RegressionDialog) else dlg.lst_indeps
                predictors.findItems("mpg", Qt.MatchFlag.MatchExactly)[0].setSelected(True)
                if isinstance(dlg, ReghdfeDialog):
                    dlg.lst_absorb.findItems("weight", Qt.MatchFlag.MatchExactly)[0].setSelected(True)
                if isinstance(dlg, RegressionDialog):
                    dlg.cmb_vce.setCurrentText("Clustered SE")
                else:
                    dlg.cmb_se.setCurrentText("Clustered")
                emitted = []
                dlg.command_executed.connect(emitted.append)
                dlg.command_pasted.connect(emitted.append)
                self.assertFalse(dlg.btn_submit.isEnabled())
                self.assertFalse(dlg.btn_paste.isEnabled())
                self.assertFalse(dlg.txt_syntax.text())
                dlg._on_submit()
                dlg._on_paste()
                self.assertEqual(emitted, [])
                dlg.cmb_cluster.setCurrentText("cluster_id")
                self.assertTrue(dlg.btn_submit.isEnabled())
                self.assertTrue(dlg.btn_paste.isEnabled())
                dlg.btn_paste.click()
                self.assertIn("vce(cluster cluster_id)", emitted[0])

    def test_hdfe_incomplete_selection_has_no_model_preview(self):
        dlg = ReghdfeDialog(self.ds)
        predictor = dlg.lst_indeps.findItems("mpg", Qt.MatchFlag.MatchExactly)[0]
        effect = dlg.lst_absorb.findItems("foreign", Qt.MatchFlag.MatchExactly)[0]
        for x, fe in ((False, False), (False, True), (True, False), (True, True)):
            predictor.setSelected(x)
            effect.setSelected(fe)
            complete = x and fe
            self.assertEqual(dlg.btn_submit.isEnabled(), complete)
            self.assertEqual(dlg.btn_paste.isEnabled(), complete)
            self.assertEqual(bool(dlg.txt_syntax.text()), complete)
            self.assertEqual("=" in dlg.lbl_equation.text(), complete)

    def test_16_reghdfe_dialog_syntax(self):
        dlg = ReghdfeDialog(self.ds)
        dlg.cmb_dep.setCurrentText("price")
        for i in range(dlg.lst_indeps.count()):
            item = dlg.lst_indeps.item(i)
            if item.text() == "mpg":
                item.setSelected(True)
        for i in range(dlg.lst_absorb.count()):
            item = dlg.lst_absorb.item(i)
            if item.text() == "cluster_id":
                item.setSelected(True)
        dlg.update_syntax()
        self.assertIn("reghdfe price mpg, absorb(cluster_id)", dlg.txt_syntax.text())

    def test_17_probit_dialog_syntax(self):
        dlg = ProbitDialog(self.ds)
        dlg.cmb_dep.setCurrentText("foreign")
        for i in range(dlg.lst_indeps.count()):
            item = dlg.lst_indeps.item(i)
            if item.text() == "mpg":
                item.setSelected(True)
        dlg.update_syntax()
        self.assertIn("probit foreign mpg", dlg.txt_syntax.text())

    def test_18_margins_dialog_syntax(self):
        dlg = MarginsDialog(self.ds)
        dlg.update_syntax()
        self.assertEqual(dlg.txt_syntax.text(), "margins, dydx(*)")


class TestMainWindowIntegration(unittest.TestCase):
    """Tests for full PySide6 window integration, command routing, and theme toggling."""

    def setUp(self):
        self.win = PotatosticsQtWindow()

    def tearDown(self):
        self.win.close()

    def test_13_window_structure_and_docks(self):
        self.assertEqual(self.win.windowTitle(), "Potatostics - Statistical Analysis Software")
        docks = self.win.findChildren(QtWidgets.QDockWidget)
        dock_titles = [d.windowTitle() for d in docks]
        self.assertIn("Command Review", dock_titles)
        self.assertIn("Variables & Properties", dock_titles)

    def test_14_theme_toggling(self):
        self.assertEqual(self.win.current_theme, "Dark")
        self.win.toggle_theme()
        self.assertEqual(self.win.current_theme, "Light")
        self.win.toggle_theme()
        self.assertEqual(self.win.current_theme, "Dark")

    def test_15_command_execution_pipeline(self):
        self.win.run_command("summarize price", sync=True)
        console_text = self.win.console.toPlainText()
        self.assertIn(". summarize price", console_text)
        self.assertIn("price", console_text)
        # History pane recorded the execution
        self.assertGreater(len(self.win.history_pane.history_records), 0)


if __name__ == "__main__":
    unittest.main()
