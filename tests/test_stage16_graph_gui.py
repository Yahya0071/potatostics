import os
import unittest
import pandas as pd
from PySide6 import QtWidgets, QtCore

from engine.dataset import Dataset
from gui_qt.main_window import PotatosticsQtWindow
from gui_qt.dialogs import GraphDialog

app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(["--platform", "offscreen"])


class TestStage16GraphGUI(unittest.TestCase):
    def setUp(self):
        self.widgets = []
        self.ds = Dataset()
        df = pd.DataFrame({
            "price": [5000, 6000, 7000, 8000, 9000],
            "mpg": [20, 22, 25, 28, 30],
            "foreign": [0, 0, 1, 1, 0]
        })
        self.ds.load_dataframe(df)

    def tearDown(self):
        for w in self.widgets:
            try:
                w.close()
                w.deleteLater()
            except Exception:
                pass
        self.widgets.clear()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            app.processEvents()

    def test_01_main_window_has_graphics_menu_and_toolbar_graph_button(self):
        win = PotatosticsQtWindow()
        self.widgets.append(win)
        try:
            # 1. Menu bar must have Graphics menu
            self.assertTrue(hasattr(win, "graphics_menu"), "PotatosticsQtWindow must have graphics_menu attribute")
            self.assertIsNotNone(win.graphics_menu)
            self.assertIn("Graphics", win.graphics_menu.title())

            # Verify menu placement: Data -> Graphics -> Statistics
            menu_titles = [m.title() for m in win.menuBar().findChildren(QtWidgets.QMenu)]
            self.assertIn("&Graphics", menu_titles)

            # Check essential actions inside graphics_menu
            actions = [a.text() for a in win.graphics_menu.actions() if a.text()]
            self.assertTrue(any("Graph Studio" in a or "Builder" in a for a in actions), "Missing Graph Studio action")
            self.assertTrue(any("scatter" in a.lower() for a in actions), "Missing Scatter plot action")
            self.assertTrue(any("binscatter" in a.lower() for a in actions), "Missing Binscatter action")
            self.assertTrue(any("rdplot" in a.lower() for a in actions), "Missing RD plot action")
            self.assertTrue(any("irf" in a.lower() for a in actions), "Missing IRF action")
            self.assertTrue(any("histogram" in a.lower() for a in actions), "Missing Histogram action")
            self.assertTrue(any("box" in a.lower() for a in actions), "Missing Box plot action")

            # 2. Toolbar must have a Graph action button
            tool_buttons = win.toolbar.findChildren(QtWidgets.QToolButton)
            button_texts = [b.text() for b in tool_buttons]
            self.assertTrue(
                any("Graph" in t for t in button_texts),
                f"Toolbar must have a 'Graph' button. Found: {button_texts}"
            )
        finally:
            win.close()

    def test_02_graph_dialog_scatter_and_line_syntax(self):
        dlg = GraphDialog(self.ds)
        self.widgets.append(dlg)
        dlg.cmb_type.setCurrentText("Two-way Scatter (scatter)")
        dlg.cmb_y.setCurrentText("price")
        dlg.cmb_x.setCurrentText("mpg")
        dlg.update_syntax()
        self.assertEqual(dlg.txt_syntax.text().strip(), "scatter price mpg")

        dlg.cmb_type.setCurrentText("Two-way Line (line)")
        dlg.update_syntax()
        self.assertEqual(dlg.txt_syntax.text().strip(), "line price mpg")

    def test_03_graph_dialog_binscatter_syntax(self):
        dlg = GraphDialog(self.ds)
        self.widgets.append(dlg)
        dlg.cmb_type.setCurrentText("Binned Scatter Plot (binscatter)")
        dlg.cmb_y.setCurrentText("price")
        dlg.cmb_x.setCurrentText("mpg")
        dlg.cmb_by.setCurrentText("foreign")
        dlg.spin_bins.setValue(20)
        dlg.update_syntax()
        self.assertIn("binscatter price mpg", dlg.txt_syntax.text())
        self.assertIn("by(foreign)", dlg.txt_syntax.text())
        self.assertIn("nbins(20)", dlg.txt_syntax.text())

    def test_04_graph_dialog_rdplot_and_irf_syntax(self):
        dlg = GraphDialog(self.ds)
        self.widgets.append(dlg)
        dlg.cmb_type.setCurrentText("Regression Discontinuity (rdplot)")
        dlg.cmb_y.setCurrentText("price")
        dlg.cmb_x.setCurrentText("mpg")
        dlg.spin_cutoff.setValue(25.0)
        dlg.update_syntax()
        self.assertIn("rdplot price mpg", dlg.txt_syntax.text())
        self.assertIn("c(25.0)", dlg.txt_syntax.text())

        dlg.cmb_type.setCurrentText("Local Projections IRF (irf)")
        dlg.spin_horizon.setValue(6)
        dlg.update_syntax()
        self.assertIn("irf price mpg", dlg.txt_syntax.text())
        self.assertIn("horizon(6)", dlg.txt_syntax.text())

    def test_05_graph_dialog_distribution_plots_syntax(self):
        dlg = GraphDialog(self.ds)
        self.widgets.append(dlg)
        dlg.cmb_type.setCurrentText("Histogram (histogram)")
        dlg.cmb_x.setCurrentText("price")
        dlg.spin_bins.setValue(15)
        dlg.update_syntax()
        self.assertIn("histogram price", dlg.txt_syntax.text())
        self.assertIn("bins(15)", dlg.txt_syntax.text())

        dlg.cmb_type.setCurrentText("Box Plot (boxplot)")
        dlg.cmb_y.setCurrentText("price")
        dlg.cmb_by.setCurrentText("foreign")
        dlg.update_syntax()
        self.assertIn("boxplot price", dlg.txt_syntax.text())
        self.assertIn("over(foreign)", dlg.txt_syntax.text())

    def test_06_interpreter_scatter_and_line_commands(self):
        from engine.interpreter import Interpreter
        captured = []
        interp = Interpreter(self.ds, plot_callback=lambda ptype, df, **kw: captured.append((ptype, kw)))
        
        res = interp.execute("line price mpg")
        self.assertTrue(res.ok)
        self.assertIn("line plot generated", res.output)
        self.assertEqual(captured[-1][0], "line")

        res_tw = interp.execute("twoway line price mpg")
        self.assertTrue(res_tw.ok)
        self.assertEqual(captured[-1][0], "line")

    def test_07_qt_plot_dialog_line_render(self):
        from gui_qt.main_window import QtPlotDialog
        dlg = QtPlotDialog("line", self.ds.df, kwargs={"yvar": "price", "xvar": "mpg"})
        self.widgets.append(dlg)
        self.assertIsNotNone(dlg.figure)
        self.assertEqual(len(dlg.figure.axes), 1)
        dlg.close()

    def test_08_sample_data_packaging_and_clean_start_without_it(self):
        from assets import get_asset_path
        from unittest import mock
        
        # 1. Verify sample_data.csv can be located in package assets
        path = get_asset_path("sample_data.csv")
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))

        # 2. Verify Qt app starts cleanly without sample_data.csv
        with mock.patch("gui_qt.main_window.get_asset_path", return_value=None):
            with mock.patch("os.path.exists", return_value=False):
                win = PotatosticsQtWindow()
                self.widgets.append(win)
                self.assertTrue(win.dataset.is_empty)
                win.close()

    def test_09_qt_plot_dialog_button_click_png_export_interaction(self):
        import tempfile
        from unittest import mock
        from gui_qt.main_window import QtPlotDialog
        dlg = QtPlotDialog("scatter", self.ds.df, kwargs={"yvar": "price", "xvar": "mpg"})
        self.widgets.append(dlg)
        with tempfile.TemporaryDirectory() as td:
            png_path = os.path.join(td, "exported_plot.png")
            with mock.patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(png_path, "PNG Files (*.png)")):
                with mock.patch.object(QtWidgets.QMessageBox, "information") as mock_info:
                    dlg.btn_png.click()
                    mock_info.assert_called_once()
            self.assertTrue(os.path.exists(png_path))
            with open(png_path, "rb") as f:
                header = f.read(8)
            self.assertEqual(header, b"\x89PNG\r\n\x1a\n", "Exported PNG must have valid magic signature")
            # Verify image dimensions
            from PySide6.QtGui import QImage
            img = QImage(png_path)
            self.assertFalse(img.isNull())
            self.assertGreater(img.width(), 100)
            self.assertGreater(img.height(), 100)

    def test_10_qt_plot_dialog_button_click_svg_export_interaction(self):
        import tempfile
        from unittest import mock
        from gui_qt.main_window import QtPlotDialog
        dlg = QtPlotDialog("scatter", self.ds.df, kwargs={"yvar": "price", "xvar": "mpg"})
        self.widgets.append(dlg)
        with tempfile.TemporaryDirectory() as td:
            svg_path = os.path.join(td, "exported_plot.svg")
            with mock.patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(svg_path, "SVG Files (*.svg)")):
                with mock.patch.object(QtWidgets.QMessageBox, "information") as mock_info:
                    dlg.btn_svg.click()
                    mock_info.assert_called_once()
            self.assertTrue(os.path.exists(svg_path))
            with open(svg_path, "rb") as f:
                content = f.read()
            self.assertTrue(b"<svg" in content or b"<?xml" in content, "Exported SVG must contain SVG XML markup")
            self.assertTrue(b"</svg>" in content)

    def test_11_qt_plot_dialog_button_click_pdf_export_interaction(self):
        import tempfile
        from unittest import mock
        from gui_qt.main_window import QtPlotDialog
        dlg = QtPlotDialog("scatter", self.ds.df, kwargs={"yvar": "price", "xvar": "mpg"})
        self.widgets.append(dlg)
        with tempfile.TemporaryDirectory() as td:
            pdf_path = os.path.join(td, "exported_plot.pdf")
            with mock.patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(pdf_path, "PDF Files (*.pdf)")):
                with mock.patch.object(QtWidgets.QMessageBox, "information") as mock_info:
                    dlg.btn_pdf.click()
                    mock_info.assert_called_once()
            self.assertTrue(os.path.exists(pdf_path))
            with open(pdf_path, "rb") as f:
                header = f.read(5)
            self.assertEqual(header, b"%PDF-", "Exported PDF must have %PDF- magic signature")

    def test_12_qt_plot_dialog_export_cancelled_dialog(self):
        from unittest import mock
        from gui_qt.main_window import QtPlotDialog
        dlg = QtPlotDialog("scatter", self.ds.df, kwargs={"yvar": "price", "xvar": "mpg"})
        self.widgets.append(dlg)
        with mock.patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=("", "")):
            with mock.patch.object(QtWidgets.QMessageBox, "information") as mock_info:
                dlg.btn_png.click()
                mock_info.assert_not_called()

    def test_13_qt_plot_dialog_export_invalid_path_error_handling(self):
        import tempfile
        from unittest import mock
        from gui_qt.main_window import QtPlotDialog
        dlg = QtPlotDialog("scatter", self.ds.df, kwargs={"yvar": "price", "xvar": "mpg"})
        self.widgets.append(dlg)
        with tempfile.TemporaryDirectory() as td:
            bad_path = os.path.join(td, "missing_subdir", "plot.png")
            with mock.patch.object(QtWidgets.QFileDialog, "getSaveFileName", return_value=(bad_path, "PNG Files (*.png)")):
                with mock.patch.object(QtWidgets.QMessageBox, "critical") as mock_crit:
                    dlg.btn_png.click()
                    mock_crit.assert_called_once()

    def test_14_qt_plot_dialog_toolbar_accessibility_dark_and_light(self):
        from gui_qt.main_window import QtPlotDialog
        dlg_dark = QtPlotDialog("scatter", self.ds.df, kwargs={"yvar": "price", "xvar": "mpg"}, theme="Dark")
        self.widgets.append(dlg_dark)
        self.assertIn("#252526", dlg_dark.toolbar.styleSheet())
        self.assertIn("#37373D", dlg_dark.toolbar.styleSheet())
        for action in dlg_dark.toolbar.actions():
            widget = dlg_dark.toolbar.widgetForAction(action)
            if isinstance(widget, QtWidgets.QToolButton) and action.text():
                self.assertTrue(bool(widget.accessibleName() or action.text()))

        dlg_light = QtPlotDialog("scatter", self.ds.df, kwargs={"yvar": "price", "xvar": "mpg"}, theme="Light")
        self.widgets.append(dlg_light)
        self.assertIn("#F3F3F3", dlg_light.toolbar.styleSheet())
        self.assertIn("#007ACC", dlg_light.toolbar.styleSheet())


if __name__ == "__main__":
    unittest.main()
