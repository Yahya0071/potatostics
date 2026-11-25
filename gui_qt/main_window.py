"""
gui_qt/main_window.py - Primary Professional Desktop Shell for Potatostics (PySide6)
Features:
- Native QDockWidget multi-monitor docking system (tear off panes across screens)
- Tabbed workspace: Interactive Results Console + Virtualized Data Grid
- Spotlight Command Palette (Ctrl+K)
- Integrated code-bridge point-and-click dialogs
- Full Dark / Light theme engine
- Responsive command execution
"""

import os
import sys
import re
from threading import Event
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from gui_qt.theme import DARK_THEME_QSS, LIGHT_THEME_QSS
from gui_qt.console import ResultsConsole
from gui_qt.data_grid import DataGridWidget
from gui_qt.command_bar import CommandBar
from gui_qt.history_pane import HistoryPane
from gui_qt.variables_pane import VariablesPane
from gui_qt.command_palette import CommandPaletteDialog
from gui_qt.dialogs import (
    RegressionDialog, SummarizeDialog, TabulateDialog,
    ReghdfeDialog, ProbitDialog, MarginsDialog,
    CsdidDialog, EsttabDialog, GraphDialog,
    CountRegressionDialog, IvregressDialog, PanelRegressionDialog, CorrelateDialog,
    TimeSeriesDialog, PanelSetupDialog, CensoredAndQuantileDialog,
    OrdinalMultinomialDialog, SurvivalAnalysisDialog, CausalInferenceDialog,
    DtableDialog, AdvancedLinearDialog, SurveyDialog, PostEstimationDialog
)
from assets import get_asset_path


import enum


class WorkerState(enum.Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"


class CommandWorker(QtCore.QThread):
    finished_signal = QtCore.Signal(str, object)
    error_signal = QtCore.Signal(str, str)
    cancelled_signal = QtCore.Signal(str)
    plot_requested = QtCore.Signal(str, object, dict)

    def __init__(self, interpreter, cmd, execution_id=0):
        super().__init__()
        self.interpreter = interpreter
        self.cmd = cmd
        self.execution_id = execution_id
        self._is_cancelled = False
        self.state = WorkerState.IDLE
        self.result = None
        self.error_message = None

    def cancel(self):
        if self.state in (WorkerState.TIMED_OUT, WorkerState.FAILED, WorkerState.COMPLETED, WorkerState.CANCELLED):
            return
        self._is_cancelled = True
        self.state = WorkerState.CANCELLING
        self.requestInterruption()

    def run(self):
        self.state = WorkerState.RUNNING
        old_callback = getattr(self.interpreter, 'plot_callback', None)
        old_cancelled = getattr(self.interpreter, 'is_cancelled', None)
        cancellation_observed = False

        def check_cancelled():
            nonlocal cancellation_observed
            cancellation_observed = self._is_cancelled or self.isInterruptionRequested()
            return cancellation_observed

        try:
            self.interpreter.is_cancelled = check_cancelled
            self.interpreter.plot_callback = lambda ptype, data, **kw: self.plot_requested.emit(ptype, data, kw)
            if check_cancelled():
                self.state = WorkerState.CANCELLED
                self.cancelled_signal.emit(self.cmd)
                return
            res = self.interpreter.execute(self.cmd)
            self.result = res
            if self.state == WorkerState.TIMED_OUT:
                # Preserved terminal TIMED_OUT state; do not emit finished/cancelled signals
                pass
            elif cancellation_observed or (
                self._is_cancelled and not res.ok and res.return_code == 1
            ):
                self.state = WorkerState.CANCELLED
                self.cancelled_signal.emit(self.cmd)
            else:
                self.state = WorkerState.COMPLETED
                self.finished_signal.emit(self.cmd, res)
        except Exception as e:
            self.error_message = str(e)
            if self.state == WorkerState.TIMED_OUT:
                pass
            elif isinstance(e, InterruptedError) and cancellation_observed:
                self.state = WorkerState.CANCELLED
                self.cancelled_signal.emit(self.cmd)
            else:
                self.state = WorkerState.FAILED
                self.error_signal.emit(self.cmd, str(e))
        finally:
            self.interpreter.plot_callback = old_callback
            self.interpreter.is_cancelled = old_cancelled


class QtPlotDialog(QtWidgets.QDialog):
    """Interactive Plot Studio Dialog for Potatostics using Matplotlib FigureCanvasQTAgg."""
    def __init__(self, plot_type, data_obj, kwargs=None, theme="Dark", parent=None):
        super().__init__(parent)
        kwargs = kwargs or {}
        self.plot_type = plot_type
        self.data_obj = data_obj
        self.kwargs = kwargs
        self.theme = theme

        title = f"Plot Studio - {plot_type.capitalize()}"
        self.setWindowTitle(title)
        self.resize(850, 620)
        self.setMinimumSize(600, 450)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self._render_plot()
        self._apply_toolbar_styling()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        btn_layout = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(f"Visualizing: {plot_type.upper()}")
        lbl.setStyleSheet("font-weight: bold; color: #888888;")
        btn_layout.addWidget(lbl)
        btn_layout.addStretch()

        self.btn_png = QtWidgets.QPushButton("Export PNG")
        self.btn_pdf = QtWidgets.QPushButton("Export PDF")
        self.btn_svg = QtWidgets.QPushButton("Export SVG")

        self.btn_png.clicked.connect(lambda: self._export_as("png"))
        self.btn_pdf.clicked.connect(lambda: self._export_as("pdf"))
        self.btn_svg.clicked.connect(lambda: self._export_as("svg"))

        btn_layout.addWidget(self.btn_png)
        btn_layout.addWidget(self.btn_pdf)
        btn_layout.addWidget(self.btn_svg)
        layout.addLayout(btn_layout)

    def _render_plot(self):
        import numpy as np
        import pandas as pd
        ax = self.figure.add_subplot(111)
        ptype = self.plot_type.lower()
        data = self.data_obj
        kw = self.kwargs

        if ptype == 'scatter':
            yvar = kw.get('yvar')
            xvar = kw.get('xvar')
            ax.scatter(data[xvar], data[yvar], alpha=0.7, edgecolors='none', color='#0072B2')
            ax.set_xlabel(str(xvar), fontsize=10)
            ax.set_ylabel(str(yvar), fontsize=10)
            ax.set_title(f"Scatter: {yvar} on {xvar}", fontsize=11, fontweight='bold')
        elif ptype == 'line':
            yvar = kw.get('yvar')
            xvar = kw.get('xvar')
            sorted_data = data.sort_values(by=xvar) if xvar in data.columns else data
            ax.plot(sorted_data[xvar], sorted_data[yvar], color='#0072B2', linewidth=2, marker='o', markersize=3)
            ax.set_xlabel(str(xvar), fontsize=10)
            ax.set_ylabel(str(yvar), fontsize=10)
            ax.set_title(f"Line Plot: {yvar} on {xvar}", fontsize=11, fontweight='bold')
        elif ptype == 'histogram':
            var = kw.get('var')
            s = data[var].dropna()
            ax.hist(s, bins=30, edgecolor='white', alpha=0.8, color='#2b83ba')
            ax.set_xlabel(str(var), fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_title(f"Histogram: {var}", fontsize=11, fontweight='bold')
        elif ptype == 'boxplot':
            var = kw.get('var')
            byvar = kw.get('byvar')
            if byvar and byvar in data.columns:
                groups = [grp[var].dropna().values for _, grp in data.groupby(byvar)]
                labels = [str(k) for k, _ in data.groupby(byvar)]
                ax.boxplot(groups, labels=labels)
                ax.set_xlabel(str(byvar), fontsize=10)
            else:
                ax.boxplot(data[var].dropna().values, labels=[str(var)])
            ax.set_ylabel("Value", fontsize=10)
            ax.set_title(f"Box Plot: {var}", fontsize=11, fontweight='bold')
        elif ptype == 'heatmap':
            vars_list = kw.get('vars', [])
            if not vars_list:
                vars_list = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
            corr = data[vars_list].corr()
            cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            self.figure.colorbar(cax, ax=ax)
            ax.set_xticks(range(len(vars_list)))
            ax.set_yticks(range(len(vars_list)))
            ax.set_xticklabels(vars_list, rotation=45, ha='left', fontsize=9)
            ax.set_yticklabels(vars_list, fontsize=9)
            ax.set_title("Correlation Heatmap", pad=20, fontsize=11, fontweight='bold')
        elif ptype == 'coefplot':
            coefs = [c for c in data.get('coefs', []) if c['name'] not in ('_cons', 'Intercept') and not c.get('omitted')]
            names = [c['name'] for c in coefs]
            betas = [c['coef'] for c in coefs]
            ci_lo = [c.get('ci_lower', c['coef'] - 1.96*c['se']) for c in coefs]
            ci_hi = [c.get('ci_upper', c['coef'] + 1.96*c['se']) for c in coefs]
            y_pos = np.arange(len(names))
            err_lo = [b - lo for b, lo in zip(betas, ci_lo)]
            err_hi = [hi - b for b, hi in zip(betas, ci_hi)]
            ax.errorbar(betas, y_pos, xerr=[err_lo, err_hi], fmt='o', capsize=4, color='#2b83ba', elinewidth=1.5)
            ax.axvline(0, color='#d7191c', linestyle='--', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel("Estimate (95% CI)", fontsize=10)
            ax.set_title(f"Coefficients: {data.get('depvar', '')}", fontsize=11, fontweight='bold')
        elif ptype == 'eventstudy':
            dyn = data.get('dynamic_effects', [])
            times = [d['event_time'] for d in dyn]
            atts = [d['att'] for d in dyn]
            ci_lo = [d['ci_lower'] for d in dyn]
            ci_hi = [d['ci_upper'] for d in dyn]
            err_lo = [a - lo for a, lo in zip(atts, ci_lo)]
            err_hi = [hi - a for a, hi in zip(atts, ci_hi)]
            ax.errorbar(times, atts, yerr=[err_lo, err_hi], fmt='-o', capsize=4, color='#2b83ba', elinewidth=1.5)
            ax.axhline(0, color='#888888', linestyle='--')
            ax.axvline(-0.5, color='#d7191c', linestyle=':', label='Treatment')
            ax.set_xlabel("Event Time (t - g)", fontsize=10)
            ax.set_ylabel("Treatment Effect (ATT)", fontsize=10)
            ax.set_title(f"Dynamic Event Study: {data.get('depvar', '')}", fontsize=11, fontweight='bold')
        elif ptype == 'binscatter':
            from engine.chart_studio import plot_binscatter
            plot_binscatter(data, kw['yvar'], kw['xvar'], byvar=kw.get('byvar'), n_bins=kw.get('n_bins', 20), controls=kw.get('controls'), line=kw.get('line', 'linear'), ax=ax)
        elif ptype == 'rdplot':
            from engine.chart_studio import plot_rd
            plot_rd(data, kw['yvar'], kw['xvar'], cutoff=kw.get('cutoff', 0.0), bandwidth=kw.get('bandwidth'), n_bins=kw.get('n_bins', 20), ax=ax)
        elif ptype == 'irf':
            from engine.chart_studio import plot_irf
            plot_irf(data, kw['response_var'], kw['shock_var'], max_horizon=kw.get('max_horizon', 12), ax=ax)
        elif ptype in ('synth_plot', 'synth'):
            from engine.chart_studio import plot_synthetic_control
            plot_synthetic_control(data, ax=ax)

        # Apply theme styling
        if self.theme.lower() == 'dark':
            self.figure.patch.set_facecolor('#202020')
            ax.set_facecolor('#2a2a2a')
            ax.spines['bottom'].set_color('#888888')
            ax.spines['left'].set_color('#888888')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(colors='#dddddd')
            ax.xaxis.label.set_color('#ffffff')
            ax.yaxis.label.set_color('#ffffff')
            ax.title.set_color('#ffffff')
            ax.grid(True, linestyle=':', alpha=0.3, color='#888888')
        else:
            self.figure.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#333333')
            ax.spines['left'].set_color('#333333')
            ax.grid(True, linestyle=':', alpha=0.4, color='#aaaaaa')

        self.figure.tight_layout()

    def _apply_toolbar_styling(self):
        """Apply accessible high-contrast styling to Matplotlib navigation toolbar."""
        from PySide6 import QtGui
        is_dark = self.theme.lower() == 'dark'
        if is_dark:
            self.toolbar.setStyleSheet("""
                QToolBar {
                    background-color: #252526;
                    border: 1px solid #3E3E42;
                    border-radius: 4px;
                    padding: 2px 4px;
                    spacing: 4px;
                }
                QToolButton {
                    background-color: #2D2D30;
                    color: #FFFFFF;
                    border: 1px solid #3E3E42;
                    border-radius: 3px;
                    padding: 4px;
                    margin: 1px;
                }
                QToolButton:hover {
                    background-color: #37373D;
                    border-color: #007ACC;
                    color: #FFFFFF;
                }
                QToolButton:pressed, QToolButton:checked {
                    background-color: #4E4E52;
                    border-color: #007ACC;
                    color: #FFFFFF;
                }
                QLabel {
                    color: #DDDDDD;
                    font-size: 9pt;
                    padding-left: 6px;
                }
                QToolTip {
                    background-color: #1E1E1E;
                    color: #FFFFFF;
                    border: 1px solid #007ACC;
                }
            """)
            for action in self.toolbar.actions():
                icon = action.icon()
                if not icon.isNull():
                    new_icon = QtGui.QIcon()
                    for size in [16, 24, 32, 48]:
                        pix = icon.pixmap(size, size)
                        if not pix.isNull():
                            img = pix.toImage().convertToFormat(QtGui.QImage.Format_ARGB32)
                            img.invertPixels(QtGui.QImage.InvertRgb)
                            new_icon.addPixmap(QtGui.QPixmap.fromImage(img))
                    action.setIcon(new_icon)
                widget = self.toolbar.widgetForAction(action)
                if isinstance(widget, QtWidgets.QToolButton):
                    widget.setAutoRaise(False)
                    if not widget.accessibleName() and action.text():
                        widget.setAccessibleName(action.text())
        else:
            self.toolbar.setStyleSheet("""
                QToolBar {
                    background-color: #F3F3F3;
                    border: 1px solid #CCCCCC;
                    border-radius: 4px;
                    padding: 2px 4px;
                    spacing: 4px;
                }
                QToolButton {
                    background-color: #FFFFFF;
                    color: #222222;
                    border: 1px solid #CCCCCC;
                    border-radius: 3px;
                    padding: 4px;
                    margin: 1px;
                }
                QToolButton:hover {
                    background-color: #E5E5E5;
                    border-color: #007ACC;
                }
                QToolButton:pressed, QToolButton:checked {
                    background-color: #D0D0D0;
                    border-color: #007ACC;
                }
                QLabel {
                    color: #333333;
                    font-size: 9pt;
                    padding-left: 6px;
                }
                QToolTip {
                    background-color: #FFFFFF;
                    color: #000000;
                    border: 1px solid #CCCCCC;
                }
            """)
            for action in self.toolbar.actions():
                widget = self.toolbar.widgetForAction(action)
                if isinstance(widget, QtWidgets.QToolButton):
                    if not widget.accessibleName() and action.text():
                        widget.setAccessibleName(action.text())

    def _export_as(self, fmt, target_path=None) -> bool:
        """
        Exports the plot to the requested format (png, pdf, svg).
        Supports programmatic target_path for headless/automated testing without blocking modal dialogs.
        Returns True on success, False on cancellation or failure.
        """
        fmt = fmt.lower().strip()
        if target_path:
            filepath = target_path
        else:
            filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, f"Export as {fmt.upper()}", f"plot.{fmt}", f"{fmt.upper()} Files (*.{fmt})"
            )
        if not filepath:
            return False
        try:
            target_dir = os.path.dirname(os.path.abspath(filepath))
            if target_dir and not os.path.exists(target_dir):
                raise OSError(f"Target directory does not exist: {target_dir}")
            self.figure.savefig(filepath, format=fmt, bbox_inches='tight', dpi=300)
            if not target_path:
                QtWidgets.QMessageBox.information(self, "Export Successful", f"Plot exported to {filepath}")
            return True
        except Exception as e:
            if not target_path:
                QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export plot: {e}")
            return False

    def closeEvent(self, event):
        try:
            if hasattr(self, "figure") and self.figure is not None:
                self.figure.clf()
            if hasattr(self, "canvas") and self.canvas is not None:
                self.canvas.close()
                self.canvas.deleteLater()
            if hasattr(self, "toolbar") and self.toolbar is not None:
                self.toolbar.close()
                self.toolbar.deleteLater()
        except Exception:
            pass
        event.accept()


class PotatosticsQtWindow(QtWidgets.QMainWindow):
    plot_requested = QtCore.Signal(str, object, dict)
    _snapshot_requested = QtCore.Signal(object, object)
    _session_restored = QtCore.Signal(object)

    def __init__(self, initial_file=None):
        super().__init__()
        self.setWindowTitle("Potatostics - Statistical Analysis Software")
        self.resize(1360, 860)
        self.setMinimumSize(1040, 680)

        self.current_theme = "Dark"
        self.dataset = Dataset()
        self.interpreter = Interpreter(self.dataset, plot_callback=self._emit_plot_request)
        self.interpreter.session_presentation_provider = self._request_session_snapshot
        self.interpreter.session_loaded_observer = self._session_restored.emit
        self._snapshot_requested.connect(self._capture_session_snapshot)
        self._session_restored.connect(self._restore_command_session)
        self.interp = self.interpreter
        self.worker = None
        self.last_command_result = None
        self._current_execution_id = 0
        self._execution_timed_out = False
        self._command_pending = False
        self._session_actions = []

        # Real-time File Monitoring Subsystem
        self.file_watcher = QtCore.QFileSystemWatcher(self)
        self.file_watcher.fileChanged.connect(self._on_monitored_file_changed)
        self._current_monitored_file = None

        self.plot_requested.connect(self._show_plot_dialog)

        self._set_window_icons()
        self._build_ui()
        self._build_menu()
        self.apply_theme("Dark")

        # Load initial file if provided
        self._init_timer = QtCore.QTimer(self)
        self._init_timer.setSingleShot(True)
        self._initial_cmd = None
        if initial_file and os.path.exists(initial_file):
            lower_file = initial_file.lower()
            if lower_file.endswith(".do"):
                cmd_to_run = f'do "{initial_file}"'
            elif lower_file.endswith(".potproj"):
                cmd_to_run = f'session load "{initial_file}"'
            else:
                cmd_to_run = f'use "{initial_file}", clear'
            self._initial_cmd = cmd_to_run
            self._init_timer.timeout.connect(lambda: self.run_command(cmd_to_run))
            self._init_timer.start(150)
        elif initial_file is not False and os.environ.get("POTATOSTICS_TESTING") != "1":
            sample_csv = get_asset_path("sample_data.csv")
            if sample_csv and os.path.exists(sample_csv):
                self._init_timer.timeout.connect(lambda: self.run_command(f'use "{sample_csv}", clear'))
                self._init_timer.start(150)

        self.command_bar.entry.setFocus()

    def _set_window_icons(self):
        ico_path = get_asset_path("potatostics.ico")
        png_path = get_asset_path("potatostics-app-icon-32.png")
        if ico_path and os.path.exists(ico_path):
            self.setWindowIcon(QtGui.QIcon(ico_path))
        elif png_path and os.path.exists(png_path):
            self.setWindowIcon(QtGui.QIcon(png_path))

    def _build_ui(self):
        # 1. Central Tabbed Workspace
        self.central_tabs = QtWidgets.QTabWidget()
        self.central_tabs.setDocumentMode(True)

        # Tab 1: Results Console
        self.console = ResultsConsole(self)
        self.central_tabs.addTab(self.console, "Results")

        # Tab 2: Virtualized Data Spreadsheet
        self.data_grid = DataGridWidget(self.dataset, self)
        self.central_tabs.addTab(self.data_grid, "Data editor")
        self.central_tabs.setTabToolTip(0, "Results (Ctrl+1)")
        self.central_tabs.setTabToolTip(1, "Data editor (Ctrl+2)")
        copy_results = QtWidgets.QToolButton()
        copy_results.setText("Copy results")
        copy_results.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        export_menu = QtWidgets.QMenu(copy_results)
        self.console.add_export_actions(export_menu)
        copy_results.setMenu(export_menu)
        copy_results.setEnabled(False)
        self.console.textChanged.connect(lambda: copy_results.setEnabled(not self.console.document().isEmpty()))
        self.central_tabs.setCornerWidget(copy_results)
        self.central_tabs.currentChanged.connect(lambda index: copy_results.setVisible(index == 0))

        # Main vertical container housing Central Tabs + Bottom Command Bar
        center_container = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_layout.addWidget(self.central_tabs, stretch=1)

        # Bottom Command Input Bar with Threaded Cancel Button
        self.command_bar = CommandBar(self.dataset, self)
        self.command_bar.command_submitted.connect(self.run_command)
        self.command_bar.cancel_requested.connect(self.cancel_command)
        self.btn_cancel = self.command_bar.btn_cancel
        center_layout.addWidget(self.command_bar)

        self.setCentralWidget(center_container)

        # 2. Left Dock: Command Review & History Pane
        self.dock_history = QtWidgets.QDockWidget("Command Review", self)
        self.dock_history.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.history_pane = HistoryPane(self)
        self.history_pane.command_selected.connect(self.command_bar.set_command_text)
        self.history_pane.command_double_clicked.connect(self.run_command)
        self.dock_history.setWidget(self.history_pane)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_history)

        # 3. Right Dock: Variables & Properties Inspector
        self.dock_vars = QtWidgets.QDockWidget("Variables & Properties", self)
        self.dock_vars.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.variables_pane = VariablesPane(self.dataset, self)
        self.variables_pane.variable_double_clicked.connect(lambda v: self.command_bar.set_command_text(f"{self.command_bar.entry.text()} {v}".strip()))
        self.dock_vars.setWidget(self.variables_pane)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_vars)

        # Set dock initial sizes
        self.resizeDocks([self.dock_history, self.dock_vars], [230, 260], Qt.Orientation.Horizontal)

        # 4. Action Toolbar
        self._build_toolbar()

    def _build_toolbar(self):
        self.toolbar = QtWidgets.QToolBar("Quick Actions")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QtCore.QSize(20, 20))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        # Brand Mark Pill
        png_mark = get_asset_path("potatostics-app-icon-32.png")
        if png_mark and os.path.exists(png_mark):
            lbl_ico = QtWidgets.QLabel()
            lbl_ico.setPixmap(QtGui.QPixmap(png_mark).scaled(22, 22, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.toolbar.addWidget(lbl_ico)

        lbl_title = QtWidgets.QLabel("Potatostics")
        lbl_title.setObjectName("workspaceBrand")
        self.toolbar.addWidget(lbl_title)
        self.toolbar.addSeparator()

        # Action Buttons
        self._add_tool_btn("Open data", self._open_file_dialog)
        session = QtWidgets.QToolButton()
        session.setText("Session")
        session.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QtWidgets.QMenu(session)
        menu.addAction("Open session…", self._open_session_dialog)
        menu.addAction("Save session…", self._save_session_dialog)
        session.setMenu(menu)
        self.toolbar.addWidget(session)
        self.toolbar.addSeparator()
        self._add_tool_btn("Summarize", self._open_summarize_dialog)
        self._add_tool_btn("Regression", self._open_regression_dialog)
        self._add_tool_btn("Graph", self._open_graph_dialog)

        # Spacer to push right-hand utilities
        spacer = QtWidgets.QWidget()
        spacer.setObjectName("toolbarSpace")
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.toolbar.addWidget(spacer)

        # Spotlight Ctrl+K button
        self.btn_palette = QtWidgets.QToolButton()
        self.btn_palette.setText("Commands  Ctrl+K")
        self.btn_palette.setToolTip("Find a command or workflow (Ctrl+K)")
        self.btn_palette.clicked.connect(self._open_command_palette)
        self.toolbar.addWidget(self.btn_palette)

        # Theme toggle button
        self.btn_theme = QtWidgets.QToolButton()
        self.btn_theme.setToolTip("Switch theme (F2)")
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.toolbar.addWidget(self.btn_theme)

    def _add_tool_btn(self, text, handler):
        btn = QtWidgets.QToolButton()
        btn.setText(text)
        btn.clicked.connect(handler)
        self.toolbar.addWidget(btn)

    def _build_menu(self):
        menubar = self.menuBar()

        # 1. File Menu
        self.file_menu = menubar.addMenu("&File")

        self.act_open_session = self.file_menu.addAction("&Open Session... (.potproj) (Ctrl+Shift+O)")
        self.act_open_session.setShortcut(QtGui.QKeySequence("Ctrl+Shift+O"))
        self.act_open_session.triggered.connect(self._open_session_dialog)

        self.act_save_session = self.file_menu.addAction("&Save Session... (.potproj) (Ctrl+Shift+S)")
        self.act_save_session.setShortcut(QtGui.QKeySequence("Ctrl+Shift+S"))
        self.act_save_session.triggered.connect(self._save_session_dialog)

        self.file_menu.addSeparator()

        self.act_open_dataset = self.file_menu.addAction("&Open Dataset... (Ctrl+O)")
        self.act_open_dataset.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        self.act_open_dataset.triggered.connect(self._open_file_dialog)

        self.act_save_dataset = self.file_menu.addAction("&Save CSV / Data (Ctrl+S)")
        self.act_save_dataset.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        self.act_save_dataset.triggered.connect(self._save_file_dialog)

        self.file_menu.addSeparator()
        self.act_report = self.file_menu.addAction("Export &Word Report (.docx)...")
        self.act_report.triggered.connect(self._export_word_report)

        self._session_actions = [
            self.act_open_session,
            self.act_save_session,
            self.act_open_dataset,
            self.act_save_dataset,
            self.act_report
        ]

        self.file_menu.addSeparator()
        act_exit = self.file_menu.addAction("E&xit")
        act_exit.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        act_exit.triggered.connect(self.close)

        # 2. View Menu
        self.view_menu = menubar.addMenu("&View")
        act_dark = self.view_menu.addAction("🌙 Dark Theme")
        act_dark.triggered.connect(lambda: self.apply_theme("Dark"))
        act_light = self.view_menu.addAction("☀️ Light Theme")
        act_light.triggered.connect(lambda: self.apply_theme("Light"))
        self.view_menu.addSeparator()
        act_theme_toggle = self.view_menu.addAction("Toggle Dark/Light (F2)")
        act_theme_toggle.setShortcut(QtGui.QKeySequence("F2"))
        act_theme_toggle.triggered.connect(self.toggle_theme)

        self.view_menu.addSeparator()
        self.view_menu.addAction(self.dock_history.toggleViewAction())
        self.view_menu.addAction(self.dock_vars.toggleViewAction())
        self.view_menu.addSeparator()
        self.view_menu.addAction("Results", QtGui.QKeySequence("Ctrl+1"), lambda: self.central_tabs.setCurrentIndex(0))
        self.view_menu.addAction("Data editor", QtGui.QKeySequence("Ctrl+2"), lambda: self.central_tabs.setCurrentIndex(1))
        self.view_menu.addAction("Focus command", QtGui.QKeySequence("Ctrl+L"), self.command_bar.entry.setFocus)

        # 3. Data Menu
        self.data_menu = menubar.addMenu("&Data")
        self.data_menu.addAction("Describe data in memory", lambda: self.run_command("describe"))
        self.data_menu.addAction("Data Editor (Spreadsheet Tab)", lambda: self.central_tabs.setCurrentIndex(1))
        self.data_menu.addSeparator()
        self.data_menu.addAction("Sort data...", lambda: self.command_bar.set_command_text("sort "))
        self.data_menu.addAction("Relational Merge (1:1 / m:1)...", lambda: self.command_bar.set_command_text("merge 1:1 id using "))
        self.data_menu.addAction("Data Reshape (long / wide)...", lambda: self.command_bar.set_command_text("reshape long "))
        self.data_menu.addAction("Collapse / Aggregate Dataset...", lambda: self.command_bar.set_command_text("collapse (mean) "))

        # 4. Graphics Menu
        self.graphics_menu = menubar.addMenu("&Graphics")
        self.graphics_menu.addAction("Graph Studio / Builder...", self._open_graph_dialog)
        self.graphics_menu.addSeparator()

        # Twoway Submenu
        m_twoway = self.graphics_menu.addMenu("&Two-way Graphs (scatter, line...)")
        m_twoway.addAction("Two-way Scatter Plot (scatter)...", lambda: self._open_graph_dialog("scatter"))
        m_twoway.addAction("Two-way Line Plot (line)...", lambda: self._open_graph_dialog("line"))

        # Distribution Submenu
        m_dist = self.graphics_menu.addMenu("&Distributional Plots (histogram, box...)")
        m_dist.addAction("Distribution Histogram (histogram)...", lambda: self._open_graph_dialog("histogram"))
        m_dist.addAction("Distribution Box Plot (boxplot)...", lambda: self._open_graph_dialog("boxplot"))

        # Econometric & Causal Submenu
        m_causal_plots = self.graphics_menu.addMenu("&Econometric & Causal Plots (binscatter, rdplot, irf...)")
        m_causal_plots.addAction("Binned Scatter Plot (binscatter)...", lambda: self._open_graph_dialog("binscatter"))
        m_causal_plots.addAction("Regression Discontinuity Plot (rdplot)...", lambda: self._open_graph_dialog("rdplot"))
        m_causal_plots.addAction("Local Projections IRF (irf)...", lambda: self._open_graph_dialog("irf"))
        m_causal_plots.addAction("Synthetic Control Plot (synth_plot)...", lambda: self._open_graph_dialog("synth_plot"))
        m_causal_plots.addAction("Dynamic Event Study Plot (eventstudy)...", lambda: self._open_graph_dialog("eventstudy"))
        m_causal_plots.addAction("Model Coefficients Plot (coefplot)...", lambda: self._open_graph_dialog("coefplot"))
        m_causal_plots.addAction("Correlation Heatmap (heatmap)...", lambda: self._open_graph_dialog("heatmap"))

        self.graphics_menu.addSeparator()
        # Direct Quick-launch actions
        self.graphics_menu.addAction("Two-way Scatter Plot (scatter)...", lambda: self._open_graph_dialog("scatter"))
        self.graphics_menu.addAction("Two-way Line Plot (line)...", lambda: self._open_graph_dialog("line"))
        self.graphics_menu.addAction("Binned Scatter Plot (binscatter)...", lambda: self._open_graph_dialog("binscatter"))
        self.graphics_menu.addAction("Regression Discontinuity Plot (rdplot)...", lambda: self._open_graph_dialog("rdplot"))
        self.graphics_menu.addAction("Local Projections IRF (irf)...", lambda: self._open_graph_dialog("irf"))
        self.graphics_menu.addAction("Synthetic Control Plot (synth_plot)...", lambda: self._open_graph_dialog("synth_plot"))
        self.graphics_menu.addAction("Distribution Histogram (histogram)...", lambda: self._open_graph_dialog("histogram"))
        self.graphics_menu.addAction("Distribution Box Plot (boxplot)...", lambda: self._open_graph_dialog("boxplot"))
        self.graphics_menu.addAction("Correlation Heatmap (heatmap)...", lambda: self._open_graph_dialog("heatmap"))
        self.graphics_menu.addAction("Model Coefficients Plot (coefplot)...", lambda: self._open_graph_dialog("coefplot"))

        # 5. Statistics Menu (Organized into Hierarchical Submenus)
        self.stat_menu = menubar.addMenu("&Statistics")

        # 5.1 Summaries, tables, and tests
        m_sum = self.stat_menu.addMenu("&Summaries, tables, and tests")
        m_sum.addAction("Summary Statistics (summarize)...", self._open_summarize_dialog)
        m_sum.addAction("Frequency Tables & Cross-Tabulation (tabulate)...", self._open_tabulate_dialog)
        m_sum.addAction("Correlation Matrix (correlate)...", self._open_correlate_dialog)
        m_sum.addAction("Descriptive Statistics Table (dtable)...", self._open_dtable_dialog)
        m_sum.addAction("Export Models Table (esttab)...", self._open_esttab_dialog)

        # 5.2 Linear models and related
        m_lin = self.stat_menu.addMenu("&Linear models and related")
        m_lin.addAction("Linear Regression (OLS / regress)...", self._open_regression_dialog)
        m_lin.addAction("High-Dimensional Fixed Effects (reghdfe)...", self._open_reghdfe_dialog)
        m_lin.addAction("Constrained Linear Regression (cnsreg)...", lambda: self._open_advanced_linear_dialog("cnsreg"))
        m_lin.addAction("Seemingly Unrelated Regression (sureg)...", lambda: self._open_advanced_linear_dialog("sureg"))
        m_lin.addAction("Streaming Big Data OLS (bigregress)...", lambda: self._open_advanced_linear_dialog("bigregress"))

        # 5.3 Binary and ordinal outcomes
        m_bin = self.stat_menu.addMenu("&Binary and ordinal outcomes")
        m_bin.addAction("Logistic Regression (logit)...", lambda: self._open_probit_dialog("logit"))
        m_bin.addAction("Binary Probit Model (probit)...", lambda: self._open_probit_dialog("probit"))
        m_bin.addAction("Ordered Logistic Regression (ologit)...", lambda: self._open_ordinal_multinomial_dialog("ologit"))
        m_bin.addAction("Multinomial Logistic Regression (mlogit)...", lambda: self._open_ordinal_multinomial_dialog("mlogit"))
        m_bin.addAction("Parallel Regression Brant Test (estat brant)...", lambda: self._open_post_estimation_dialog("brant"))

        # 5.4 Count outcomes
        m_count = self.stat_menu.addMenu("&Count outcomes")
        m_count.addAction("Poisson Count Model (poisson)...", lambda: self._open_count_dialog("poisson"))
        m_count.addAction("Negative Binomial Regression (nbreg)...", lambda: self._open_count_dialog("nbreg"))

        # 5.5 Censored and quantile models
        m_cens = self.stat_menu.addMenu("&Censored and quantile models")
        m_cens.addAction("Censored Regression / Tobit (tobit)...", lambda: self._open_censored_quantile_dialog("tobit"))
        m_cens.addAction("Quantile Regression (qreg)...", lambda: self._open_censored_quantile_dialog("qreg"))

        # 5.6 Endogenous covariates / IV
        m_iv = self.stat_menu.addMenu("&Endogenous covariates / IV")
        m_iv.addAction("Two-Stage Least Squares (ivregress 2sls)...", self._open_iv_dialog)
        m_iv.addAction("Montiel Olea-Pflueger Weak IV Test (estat weakiv)...", lambda: self._open_post_estimation_dialog("weakiv"))
        m_iv.addAction("Durbin-Wu-Hausman Endogeneity Test (estat endogenous)...", lambda: self._open_post_estimation_dialog("endogenous"))
        m_iv.addAction("Overidentification Test (estat overid)...", lambda: self._open_post_estimation_dialog("overid"))

        # 5.7 Longitudinal / Panel data
        m_panel = self.stat_menu.addMenu("&Longitudinal / Panel data")
        m_panel.addAction("Declare Panel Dataset (xtset)...", self._open_panel_setup_dialog)
        m_panel.addAction("Panel Fixed Effects (xtreg, fe)...", lambda: self._open_panel_dialog("fe"))
        m_panel.addAction("Panel Random Effects GLS (xtreg, re)...", lambda: self._open_panel_dialog("re"))
        m_panel.addAction("Hausman Specification Test (hausman)...", lambda: self._open_post_estimation_dialog("hausman"))

        # 5.8 Time series
        m_ts = self.stat_menu.addMenu("&Time series")
        m_ts.addAction("Time-Series Studio (All Tools)...", lambda: self._open_timeseries_dialog("tsset"))
        m_ts.addAction("Declare Time-Series Dataset (tsset)...", lambda: self._open_timeseries_dialog("tsset"))
        m_ts.addAction("Augmented Dickey-Fuller Unit Root Test (dfuller)...", lambda: self._open_timeseries_dialog("dfuller"))
        m_ts.addAction("ARIMA Modeling & Forecasting (arima)...", lambda: self._open_timeseries_dialog("arima"))
        m_ts.addAction("Hodrick-Prescott Filter (tsfilter hp)...", lambda: self._open_timeseries_dialog("hp"))
        m_ts.addAction("Vector Autoregression (var)...", lambda: self._open_timeseries_dialog("var"))
        m_ts.addAction("Autocorrelations & Correlogram (corrgram)...", lambda: self._open_timeseries_dialog("corrgram"))

        # 5.9 Survival analysis
        m_surv = self.stat_menu.addMenu("&Survival analysis")
        m_surv.addAction("Declare Survival Data (stset)...", lambda: self._open_survival_dialog("stset"))
        m_surv.addAction("Kaplan-Meier Survivor Table (sts list)...", lambda: self._open_survival_dialog("sts"))
        m_surv.addAction("Cox Proportional Hazards Model (stcox)...", lambda: self._open_survival_dialog("stcox"))

        # 5.10 Survey data analysis
        m_svy = self.stat_menu.addMenu("&Survey data analysis")
        m_svy.addAction("Declare Survey Design (svyset)...", lambda: self._open_survey_dialog("svyset"))
        m_svy.addAction("Survey-Weighted Linear Regression (svy: regress)...", lambda: self._open_survey_dialog("regress"))

        # 5.11 Modern causal inference
        m_causal = self.stat_menu.addMenu("&Modern causal inference")
        m_causal.addAction("Staggered DiD Callaway-Sant'Anna (csdid)...", self._open_csdid_dialog)
        m_causal.addAction("Goodman-Bacon Decomposition (bacondecomp)...", lambda: self._open_causal_dialog("bacondecomp"))
        m_causal.addAction("Synthetic Control Method (synth)...", lambda: self._open_causal_dialog("synth"))
        m_causal.addAction("Dynamic Event Study (eventstudy)...", lambda: self._open_causal_dialog("eventstudy"))
        m_causal.addAction("Propensity Score Matching (psmatch)...", lambda: self._open_causal_dialog("psmatch"))

        # 5.12 Post-estimation and margins
        m_post = self.stat_menu.addMenu("&Post-estimation and margins")
        m_post.addAction("Average Marginal Effects (margins)...", self._open_margins_dialog)
        m_post.addAction("Linear Combinations of Parameters (lincom)...", lambda: self._open_post_estimation_dialog("lincom"))
        m_post.addAction("Nonlinear Combinations of Parameters (nlcom)...", lambda: self._open_post_estimation_dialog("nlcom"))
        m_post.addAction("Nonlinear Hypothesis Wald Test (testnl)...", lambda: self._open_post_estimation_dialog("testnl"))
        m_post.addSeparator()
        m_post.addAction("Variance Inflation Factors (estat vif)...", lambda: self._open_post_estimation_dialog("vif"))
        m_post.addAction("Breusch-Pagan Heteroskedasticity (estat hettest)...", lambda: self._open_post_estimation_dialog("hettest"))
        m_post.addAction("Ramsey RESET Misspecification (estat ovtest)...", lambda: self._open_post_estimation_dialog("ovtest"))

        # 5. Tools Menu
        self.tools_menu = menubar.addMenu("&Tools")
        act_pal = self.tools_menu.addAction("Command Palette (Ctrl+K)...")
        act_pal.setShortcut(QtGui.QKeySequence("Ctrl+K"))
        act_pal.triggered.connect(self._open_command_palette)

        self.tools_menu.addAction("Montiel Olea-Pflueger Weak IV (estat weakiv)...", lambda: self._open_post_estimation_dialog("weakiv"))
        self.tools_menu.addAction("Ramsey RESET Misspecification (estat ovtest)...", lambda: self._open_post_estimation_dialog("ovtest"))
        self.tools_menu.addAction("Breusch-Pagan Heteroskedasticity (estat hettest)...", lambda: self._open_post_estimation_dialog("hettest"))
        self.tools_menu.addAction("Variance Inflation Factors (estat vif)...", lambda: self._open_post_estimation_dialog("vif"))
        self.tools_menu.addSeparator()
        self.tools_menu.addAction("Community Package Hub & Plugins (pkg)...", self._open_package_manager)

        # 6. Help Menu
        self.help_menu = menubar.addMenu("&Help")
        self.help_menu.addAction("Command Reference (help)", lambda: self.run_command("help"))
        self.help_menu.addSeparator()
        self.help_menu.addAction("About Potatostics", self._show_about)

    def apply_theme(self, theme_name):
        self.current_theme = theme_name
        self.console.set_theme(theme_name == "Dark")
        if theme_name == "Dark":
            QtWidgets.QApplication.instance().setStyleSheet(DARK_THEME_QSS)
            self.btn_theme.setText("Light theme")
        else:
            QtWidgets.QApplication.instance().setStyleSheet(LIGHT_THEME_QSS)
            self.btn_theme.setText("Dark theme")

    def toggle_theme(self):
        new_theme = "Light" if self.current_theme == "Dark" else "Dark"
        self.apply_theme(new_theme)

    def is_busy(self):
        """Returns True if a background command worker is currently active."""
        if self.worker is None:
            return False
        return self._command_pending or self.worker.isRunning() or self.worker.state in (WorkerState.RUNNING, WorkerState.CANCELLING)

    def _set_busy_state(self, busy: bool, force: bool = False):
        """
        Enforces strict execution serialization across UI surfaces.
        Disables command-producing menus, toolbar actions, command bar,
        data grid, and history pane during execution.
        """
        if not busy and not force and self.is_busy():
            # Prevent clearing busy state while a worker is still active
            return

        self.command_bar.set_running_state(busy)
        if hasattr(self, 'toolbar') and self.toolbar is not None:
            self.toolbar.setEnabled(not busy)
        if hasattr(self, 'data_menu') and self.data_menu is not None:
            self.data_menu.setEnabled(not busy)
        if hasattr(self, 'graphics_menu') and self.graphics_menu is not None:
            self.graphics_menu.setEnabled(not busy)
        if hasattr(self, 'stat_menu') and self.stat_menu is not None:
            self.stat_menu.setEnabled(not busy)
        if hasattr(self, 'tools_menu') and self.tools_menu is not None:
            self.tools_menu.setEnabled(not busy)
        if hasattr(self, 'history_pane') and self.history_pane is not None:
            self.history_pane.setEnabled(not busy)
        if hasattr(self, 'data_grid') and self.data_grid is not None:
            self.data_grid.setEnabled(not busy)
            if hasattr(self.data_grid, 'set_readonly'):
                self.data_grid.set_readonly(busy)
            elif hasattr(self.data_grid, 'model') and self.data_grid.model is not None:
                self.data_grid.model._is_readonly = busy
        if hasattr(self, 'variables_pane') and self.variables_pane is not None:
            self.variables_pane.setEnabled(not busy)

        for act in getattr(self, '_session_actions', []):
            if act is not None:
                act.setEnabled(not busy)

    def _on_worker_finished(self):
        """Called on GUI thread when worker QThread emits finished."""
        sender = self.sender()
        target_worker = sender if (sender is not None and isinstance(sender, CommandWorker)) else self.worker
        if target_worker:
            target_worker.wait()
            # If sender is obsolete (from a superseded/timed-out command), do not touch current window state
            if target_worker is not self.worker or getattr(target_worker, 'execution_id', 0) != self._current_execution_id:
                self.console.append_output(
                    f"\n[Background worker #{getattr(target_worker, 'execution_id', '?')} for '{target_worker.cmd}' has terminated]\n"
                )
                return

            # For the current active worker, preserve its terminal outcome:
            self._command_pending = False
            if target_worker.state == WorkerState.TIMED_OUT:
                message = f"Timed-out worker for '{target_worker.cmd}' has terminated"
                if target_worker.result is not None and target_worker.result.ok:
                    message += ". The command returned success and may have applied changes"
                self.console.append_output(f"\n[{message}.]\n")
                self.dataset = self.interpreter.dataset
                self.variables_pane.refresh()
                self.data_grid.refresh()
                self.command_bar.update_autocomplete()
                self._set_busy_state(False, force=True)
                return

            if target_worker.state in (WorkerState.RUNNING, WorkerState.CANCELLING):
                if target_worker._is_cancelled:
                    target_worker.state = WorkerState.CANCELLED
                else:
                    target_worker.state = WorkerState.COMPLETED
            # If target_worker.state is FAILED, CANCELLED, or COMPLETED, we preserve it!

        self._set_busy_state(False, force=True)

    def run_command(self, cmd_text, sync=False):
        cmd = cmd_text.strip()
        if not cmd:
            return False

        if self.is_busy():
            self.console.append_error("Cannot execute: another command is currently running.")
            return False

        self._current_execution_id += 1
        self._command_pending = True
        self._execution_timed_out = False
        exec_id = self._current_execution_id

        self._set_busy_state(True)
        self.console.append_command(cmd, execution_id=exec_id)
        self.central_tabs.setCurrentIndex(0)  # Switch to console

        self.worker = CommandWorker(self.interpreter, cmd, execution_id=exec_id)
        self.worker.finished_signal.connect(self._on_command_finished)
        self.worker.error_signal.connect(self._on_command_error)
        self.worker.cancelled_signal.connect(self._on_command_cancelled)
        self.worker.plot_requested.connect(self._show_plot_dialog)
        self.worker.finished.connect(self._on_worker_finished)
        if sync:
            loop = QtCore.QEventLoop()
            self.worker.finished.connect(loop.quit)
        self.worker.start()
        if sync:
            loop.exec()
        return True

    def run_command_sync(self, cmd_text, timeout=15.0):
        """
        Synchronously executes a command in the GUI worker thread and waits
        for full completion, processing Qt events and returning the completed
        CommandResult. Thread-safe and respects Qt thread-affinity.
        """
        import time
        from engine.interpreter import CommandResult
        cmd = cmd_text.strip()
        if not cmd:
            return CommandResult(output="Empty command", ok=False, return_code=198)

        if self.is_busy():
            return CommandResult(output="Cannot execute: another command is currently running", ok=False, return_code=198)

        self.last_command_result = None
        started = self.run_command(cmd, sync=False)
        if not started or self.worker is None:
            return CommandResult(output="Failed to schedule command", ok=False, return_code=198)

        # Wait with timeout for the worker thread to finish while pumping Qt events
        deadline = time.time() + timeout
        while self.worker.isRunning() and time.time() < deadline:
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
            time.sleep(0.01)

        if self.worker.isRunning():
            self.worker.cancel()
            # Wait up to 1 second for cooperative termination
            stopped = self.worker.wait(1000)
            if stopped:
                QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
                self._set_busy_state(False, force=True)
                return self.last_command_result
            else:
                # Non-cooperative worker is still running!
                self.worker.state = WorkerState.TIMED_OUT
                self._execution_timed_out = True
                self.last_command_result = CommandResult(
                    output=f"Command execution timed out after {timeout}s (worker active)",
                    ok=False,
                    error=f"Timeout: worker did not terminate within {timeout}s",
                    return_code=124
                )
                self.console.append_error(
                    str(self.last_command_result),
                    execution_id=self._current_execution_id,
                    command_text=cmd,
                )
                self.history_pane.add_entry(cmd, ok=False, return_code=124)
                # DO NOT clear busy state! Application must remain busy while worker thread is running
                return self.last_command_result

        # Process any remaining pending events (queued signals: finished_signal, error_signal)
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
        self._set_busy_state(False)
        return self.last_command_result

    def cancel_command(self):
        if self.worker and (self.worker.isRunning() or self.worker.state in (WorkerState.RUNNING, WorkerState.CANCELLING)):
            self.worker.cancel()

    def _on_command_cancelled(self, cmd):
        target_worker = self.sender() if (self.sender() is not None and isinstance(self.sender(), CommandWorker)) else self.worker
        if target_worker is not None:
            if target_worker is not self.worker or getattr(target_worker, 'execution_id', 0) != self._current_execution_id:
                return
            if target_worker.state == WorkerState.TIMED_OUT or self._execution_timed_out:
                return

        target_exec_id = getattr(target_worker, 'execution_id', self._current_execution_id) if target_worker else self._current_execution_id
        from engine.interpreter import CommandResult
        result = target_worker.result if target_worker is not None else None
        if result is None or result.ok or result.return_code != 1:
            result = CommandResult(output="(Command cancelled by user)", ok=False, error="Command cancelled by user", return_code=1, command_text=cmd)
        self.last_command_result = result
        self.dataset = self.interpreter.dataset
        self.console.append_error(
            str(result) or "(Command cancelled by user)",
            execution_id=target_exec_id,
            command_text=cmd,
            command_result=result,
        )
        self.history_pane.add_entry(cmd, ok=False, return_code=1)
        self.variables_pane.refresh()
        self.data_grid.refresh()
        self.command_bar.update_autocomplete()

    def _on_command_error(self, cmd, err_msg):
        target_worker = self.sender() if (self.sender() is not None and isinstance(self.sender(), CommandWorker)) else self.worker
        if target_worker is not None:
            if target_worker is not self.worker or getattr(target_worker, 'execution_id', 0) != self._current_execution_id:
                return
            if target_worker.state == WorkerState.TIMED_OUT or self._execution_timed_out:
                return

        target_exec_id = getattr(target_worker, 'execution_id', self._current_execution_id) if target_worker else self._current_execution_id
        from engine.interpreter import CommandResult
        self.last_command_result = CommandResult(output=f"Error: {err_msg}", ok=False, error=str(err_msg), return_code=198, command_text=cmd)
        self.dataset = self.interpreter.dataset
        self.console.append_error(f"Error: {err_msg}", execution_id=target_exec_id, command_text=cmd)
        self.history_pane.add_entry(cmd, ok=False, return_code=198)
        self.variables_pane.refresh()
        self.data_grid.refresh()
        self.command_bar.update_autocomplete()

    def closeEvent(self, event):
        """Cleanly handle window shutdown without unsafe thread termination."""
        if hasattr(self, "_init_timer") and self._init_timer is not None:
            self._init_timer.stop()
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            if not self.worker.wait(1000):
                if self.isVisible():
                    reply = QtWidgets.QMessageBox.question(
                        self,
                        "Command in Progress",
                        "A command is executing and safe cancellation was requested.\n"
                        "Wait for it to reach a safe interruption point?",
                        QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                        QtWidgets.QMessageBox.StandardButton.Yes
                    )
                    if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                        if not self.worker.wait(5000):
                            event.ignore()
                            return
                    else:
                        event.ignore()
                        return
                else:
                    if not self.worker.wait(3000):
                        event.ignore()
                        return
        event.accept()

    def _emit_plot_request(self, plot_type, data_obj, **kwargs):
        self.plot_requested.emit(plot_type, data_obj, kwargs)

    def _show_plot_dialog(self, plot_type, data_obj, kwargs):
        try:
            dlg = QtPlotDialog(plot_type, data_obj, kwargs=kwargs, theme=self.current_theme, parent=self)
            dlg.show()
        except Exception as e:
            self.console.append_error(f"Plot error: {e}")

    def _on_command_finished(self, cmd, res):
        target_worker = self.sender() if (self.sender() is not None and isinstance(self.sender(), CommandWorker)) else self.worker
        if target_worker is not None:
            if target_worker is not self.worker or getattr(target_worker, 'execution_id', 0) != self._current_execution_id:
                self.console.append_output(
                    f"\n[Diagnostic: late finished result for obsolete command '{cmd}' discarded]\n"
                )
                return
            if target_worker.state == WorkerState.TIMED_OUT or self._execution_timed_out:
                self.console.append_output(
                    f"\n[Diagnostic: late command result for timed-out command '{cmd}' discarded]\n"
                )
                return
            if target_worker.state in (WorkerState.CANCELLED, WorkerState.CANCELLING):
                self.console.append_output(
                    f"\n[Diagnostic: late command result for cancelled command '{cmd}' discarded]\n"
                )
                return

        target_exec_id = getattr(target_worker, 'execution_id', self._current_execution_id) if target_worker else self._current_execution_id
        self.last_command_result = res
        self.dataset = self.interpreter.dataset
        if res.ok:
            self.console.append_output(
                str(res),
                execution_id=target_exec_id,
                command_result=res,
                model_snapshot=getattr(res, 'model_snapshot', None),
                command_text=cmd
            )
            if target_worker is not None and target_worker._is_cancelled:
                self.console.append_output("[Command completed after cancellation was requested.]")
            self.history_pane.add_entry(cmd, ok=True, return_code=res.return_code)
            # Track file monitoring if command was use
            cmd_clean = cmd.strip()
            if cmd_clean.lower().startswith("use "):
                m_f = re.search(r'use\s+["\']?(.*?)["\']?(?:,|$)', cmd_clean, re.IGNORECASE)
                if m_f:
                    fpath = m_f.group(1).strip()
                    self._watch_file(fpath)
        else:
            self.console.append_error(
                str(res),
                execution_id=target_exec_id,
                command_result=res,
                command_text=cmd
            )
            self.history_pane.add_entry(cmd, ok=False, return_code=res.return_code)

        # Refresh views on main GUI thread
        self.variables_pane.refresh()
        self.data_grid.refresh()
        self.command_bar.update_autocomplete()

    def _watch_file(self, filepath):
        if not filepath or not os.path.exists(filepath):
            return
        if self._current_monitored_file:
            try:
                self.file_watcher.removePath(self._current_monitored_file)
            except Exception:
                pass
        self._current_monitored_file = os.path.abspath(filepath)
        try:
            self.file_watcher.addPath(self._current_monitored_file)
        except Exception:
            pass

    def _on_monitored_file_changed(self, path):
        if not os.path.exists(path):
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "File Modified Externally",
            f"The open dataset file '{os.path.basename(path)}' was modified on disk outside Potatostics.\n\nDo you want to reload it?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.run_command(f'use "{path}", clear')

    def _session_presentation(self):
        return {
            "history_records": [r.copy() for r in self.history_pane.history_records],
            "console_text": self.console.toPlainText(),
            "console_ranges": self.console.session_ranges(),
        }

    def _request_session_snapshot(self):
        # Script commands can request a save well after submission. Read widgets
        # only on the GUI thread, and only when a save actually needs them.
        snapshot = {}
        ready = Event()
        self._snapshot_requested.emit(snapshot, ready)
        while not ready.wait(0.05):
            if QtCore.QThread.currentThread().isInterruptionRequested():
                raise InterruptedError("Session save cancelled")
        if "error" in snapshot:
            raise ValueError(snapshot["error"])
        return snapshot

    @QtCore.Slot(object, object)
    def _capture_session_snapshot(self, snapshot, ready):
        try:
            snapshot.update(self._session_presentation())
            if self._command_pending:
                # Only completed commands belong in an archived transcript.
                snapshot["console_text"] = snapshot["console_text"].removesuffix(f". {self.worker.cmd}\n")
        except Exception as exc:
            snapshot["error"] = str(exc)
        finally:
            ready.set()

    @QtCore.Slot(object)
    def _restore_command_session(self, restored):
        self._restore_session_presentation(restored)
        if self._command_pending:
            self.console.append_command(self.worker.cmd, execution_id=self.worker.execution_id)

    def _restore_session_presentation(self, restored):
        self.history_pane.history_records.clear()
        self.history_pane.list_widget.clear()
        for record in restored.get("history_records", []):
            self.history_pane.add_entry(
                record.get("cmd") or record.get("command", ""),
                ok=record.get("ok", True),
                return_code=record.get("rc", 0),
            )
        manifest = restored.get("manifest") or {}
        self.console.restore_transcript(restored.get("console_text", ""), manifest.get("console_ranges_v1"))

    def save_session(self, filepath: str) -> bool:
        """Saves complete analytical session (data, history, estimates, console) to .potproj archive."""
        if self.is_busy():
            msg = "Cannot save session: a command is currently running."
            sys.stderr.write(f"{msg}\n")
            if self.isVisible():
                QtWidgets.QMessageBox.warning(self, "Session Operation Blocked", msg)
            return False
        from engine.session_manager import save_session_bundle
        try:
            save_session_bundle(filepath, self.interpreter, **self._session_presentation())
            return True
        except Exception as e:
            sys.stderr.write(f"save_session error: {e}\n")
            if self.isVisible():
                QtWidgets.QMessageBox.critical(self, "Save Session Error", f"Failed to save session:\n{e}")
            return False

    def open_session(self, filepath: str) -> bool:
        """Restores complete analytical session from a .potproj archive."""
        if self.is_busy():
            msg = "Cannot open session: a command is currently running."
            sys.stderr.write(f"{msg}\n")
            if self.isVisible():
                QtWidgets.QMessageBox.warning(self, "Session Operation Blocked", msg)
            return False
        from engine.session_manager import load_session_bundle
        try:
            restored = load_session_bundle(filepath, self.interpreter)
            self.dataset = self.interpreter.dataset
            
            # 1. Update data grid and variables pane
            self.data_grid.refresh()
            self.variables_pane.refresh()
            self.command_bar.update_autocomplete()
            
            self._restore_session_presentation(restored)
            if restored.get("console_text"):
                self.console.append_output(f"\n[Session restored from {os.path.basename(filepath)}]\n")
                
            return True
        except Exception as e:
            if self.isVisible():
                QtWidgets.QMessageBox.critical(self, "Open Session Error", f"Failed to open session:\n{e}")
            return False

    def _save_session_dialog(self):
        if self.is_busy():
            QtWidgets.QMessageBox.warning(self, "Session Operation Blocked", "Cannot save session while a command is executing.")
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Analytical Session", "", "Potatostics Session Archive (*.potproj);;All Files (*)"
        )
        if fname:
            if self.save_session(fname):
                QtWidgets.QMessageBox.information(self, "Session Saved", f"Analytical session successfully saved to:\n{fname}")

    def _open_session_dialog(self):
        if self.is_busy():
            QtWidgets.QMessageBox.warning(self, "Session Operation Blocked", "Cannot open session while a command is executing.")
            return
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Analytical Session", "", "Potatostics Session Archive (*.potproj);;All Files (*)"
        )
        if fname:
            if self.open_session(fname):
                QtWidgets.QMessageBox.information(self, "Session Restored", f"Analytical session successfully restored from:\n{fname}")

    def _open_file_dialog(self):
        if self.is_busy():
            QtWidgets.QMessageBox.warning(self, "Dataset Operation Blocked", "Cannot open dataset while a command is executing.")
            return
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Dataset", "", "Data Files (*.dta *.csv *.parquet *.feather);;Parquet Files (*.parquet);;Binary Datasets (*.dta);;CSV Files (*.csv);;All Files (*)"
        )
        if fname:
            self.run_command(f'use "{fname}", clear')

    def _save_file_dialog(self):
        if self.is_busy():
            QtWidgets.QMessageBox.warning(self, "Dataset Operation Blocked", "Cannot save dataset while a command is executing.")
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Dataset", "", "Binary DTA Datasets (*.dta);;CSV Files (*.csv);;Parquet Files (*.parquet);;All Files (*)"
        )
        if fname:
            self.run_command(f'save "{fname}", replace')

    def _open_graph_dialog(self, graph_type=None):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "Graph Studio", "Please open a dataset first.")
            return
        dlg = GraphDialog(self.dataset, self, initial_type=graph_type)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_regression_dialog(self):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "Regression", "Please open a dataset first.")
            return
        dlg = RegressionDialog(self.dataset, self)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_summarize_dialog(self):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "Summarize", "Please open a dataset first.")
            return
        dlg = SummarizeDialog(self.dataset, self)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_tabulate_dialog(self):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "Tabulate", "Please open a dataset first.")
            return
        dlg = TabulateDialog(self.dataset, self)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_reghdfe_dialog(self):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "reghdfe", "Please open a dataset first.")
            return
        dlg = ReghdfeDialog(self.dataset, self)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_probit_dialog(self, initial_model="probit"):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "Binary Choice", "Please open a dataset first.")
            return
        dlg = ProbitDialog(self.dataset, self, initial_model=initial_model)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_count_dialog(self, initial_model="poisson"):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "Count Models", "Please open a dataset first.")
            return
        dlg = CountRegressionDialog(self.dataset, self, initial_model=initial_model)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_iv_dialog(self):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "ivregress", "Please open a dataset first.")
            return
        dlg = IvregressDialog(self.dataset, self)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_panel_dialog(self, initial_model="fe"):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "xtreg", "Please open a dataset first.")
            return
        dlg = PanelRegressionDialog(self.dataset, self, initial_model=initial_model)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_correlate_dialog(self):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "correlate", "Please open a dataset first.")
            return
        dlg = CorrelateDialog(self.dataset, self)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_margins_dialog(self):
        if not self.interpreter.estimates.last_model:
            QtWidgets.QMessageBox.information(self, "margins", "Please fit a model first before estimating marginal effects.")
            return
        dlg = MarginsDialog(self.dataset, self)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_csdid_dialog(self):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "csdid", "Please open a dataset first.")
            return
        dlg = CsdidDialog(self.dataset, self)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_esttab_dialog(self):
        dlg = EsttabDialog(self.dataset, self)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_timeseries_dialog(self, initial_task="tsset"):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "Time Series", "Please open a dataset first.")
            return
        dlg = TimeSeriesDialog(self.dataset, self, initial_task=initial_task)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_panel_setup_dialog(self):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "xtset", "Please open a dataset first.")
            return
        dlg = PanelSetupDialog(self.dataset, self)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_dtable_dialog(self):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "dtable", "Please open a dataset first.")
            return
        dlg = DtableDialog(self.dataset, self)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_advanced_linear_dialog(self, initial_model="cnsreg"):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "Advanced Linear Regression", "Please open a dataset first.")
            return
        dlg = AdvancedLinearDialog(self.dataset, self, initial_model=initial_model)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_ordinal_multinomial_dialog(self, initial_model="ologit"):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "Ordinal / Multinomial", "Please open a dataset first.")
            return
        dlg = OrdinalMultinomialDialog(self.dataset, self, initial_model=initial_model)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_censored_quantile_dialog(self, initial_model="tobit"):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "Censored / Quantile", "Please open a dataset first.")
            return
        dlg = CensoredAndQuantileDialog(self.dataset, self, initial_model=initial_model)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_survival_dialog(self, initial_task="stset"):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "Survival Analysis", "Please open a dataset first.")
            return
        dlg = SurvivalAnalysisDialog(self.dataset, self, initial_task=initial_task)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_survey_dialog(self, initial_mode="svyset"):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "Survey Data", "Please open a dataset first.")
            return
        dlg = SurveyDialog(self.dataset, self, initial_mode=initial_mode)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_causal_dialog(self, initial_tool="bacondecomp"):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "Causal Inference", "Please open a dataset first.")
            return
        dlg = CausalInferenceDialog(self.dataset, self, initial_tool=initial_tool)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_post_estimation_dialog(self, initial_tool="lincom"):
        dlg = PostEstimationDialog(self.dataset, self, initial_tool=initial_tool)
        dlg.command_executed.connect(self.run_command)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()

    def _open_command_palette(self):
        palette = CommandPaletteDialog(self.dataset, self)
        palette.action_triggered.connect(self._handle_palette_action)
        palette.exec()

    def _handle_palette_action(self, cmd, title):
        c = cmd.strip()
        if c == "open_file":
            self._open_file_dialog()
        elif c == "save_file":
            self._save_file_dialog()
        elif c == "open_graph":
            self._open_graph_dialog()
        elif c == "toggle_theme":
            self.toggle_theme()
        elif c == "regress":
            self._open_regression_dialog()
        elif c in ("reghdfe , absorb()", "reghdfe"):
            self._open_reghdfe_dialog()
        elif c in ("logit", "probit"):
            self._open_probit_dialog(c)
        elif c == "poisson":
            self._open_count_dialog("poisson")
        elif c == "qreg":
            self._open_censored_quantile_dialog("qreg")
        elif c == "tobit , ll(0)":
            self._open_censored_quantile_dialog("tobit")
        elif c == "ivregress 2sls":
            self._open_iv_dialog()
        elif c == "xtreg , fe":
            self._open_panel_dialog("fe")
        elif c == "summarize":
            self._open_summarize_dialog()
        elif c == "tabulate":
            self._open_tabulate_dialog()
        elif c == "correlate":
            self._open_correlate_dialog()
        elif c in ("dfuller", "corrgram", "arima"):
            self._open_timeseries_dialog(c)
        elif c in ("csdid", "synth", "eventstudy", "psmatch"):
            if c == "csdid":
                self._open_csdid_dialog()
            else:
                self._open_causal_dialog(c)
        elif c == "esttab":
            self._open_esttab_dialog()
        elif c in ("scatter", "line", "binscatter", "rdplot", "irf", "synth_plot", "histogram", "boxplot", "heatmap", "coefplot"):
            self._open_graph_dialog(c)
        elif c == "bigregress":
            self._open_advanced_linear_dialog("bigregress")
        elif c == "lincom":
            self._open_post_estimation_dialog("lincom")
        elif c == "ai":
            self.command_bar.set_command_text("ai ")
            self.command_bar.entry.setFocus()
        elif cmd.endswith(" ") or "," in cmd:
            self.command_bar.set_command_text(cmd)
        else:
            self.run_command(cmd)

    def _open_package_manager(self):
        from gui_qt.package_dialog import PackageManagerDialog
        dlg = PackageManagerDialog(self.interpreter, self)
        dlg.command_pasted.connect(self.command_bar.set_command_text)
        dlg.exec()


    def _export_word_report(self):
        if self.dataset.is_empty:
            QtWidgets.QMessageBox.information(self, "Word Report", "Please load a dataset first.")
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Word Report", "Potatostics_Report.docx", "Word Documents (*.docx)")
        if fname:
            self.run_command(f'putdocx "{fname}"')

    def _show_about(self):
        QtWidgets.QMessageBox.about(
            self,
            "About Potatostics",
            "<b>Potatostics v1.0</b><br>"
            "Professional Open-Source Statistical Analysis Software & Econometric Workstation.<br><br>"
            "• Independent Econometric & Statistical Engine<br>"
            "• Modern PySide6 Dockable Workspace with Multi-Monitor Detaching<br>"
            "• Virtualized Data Spreadsheet Grid<br>"
            "• Professional Command Parser & Expression Evaluator<br>"
            "• Robust OLS, Logit, 2SLS IV, Panel FE, ARIMA, DiD, PSMatch<br><br>"
            "Released under the GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later)."
        )

def launch_qt_app(initial_file=None):
    """Initializes and launches the PySide6 Potatostics Desktop Workstation."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    app.setApplicationName("Potatostics")
    app.setOrganizationName("Potatostics")

    window = PotatosticsQtWindow(initial_file=initial_file)
    window.show()
    return app.exec()
