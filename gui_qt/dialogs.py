"""
gui_qt/dialogs.py - Interactive Code-Bridge Dialogs
Each dialog features a live syntax preview strip at the bottom:
users see the exact command line syntax being generated as they click,
with Submit, Paste, and Cancel actions.
"""

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Signal


class BaseBridgeDialog(QtWidgets.QDialog):
    """Base class providing a scrollable body, pinned live syntax preview strip, and Paste/Submit buttons."""
    command_executed = Signal(str)
    command_pasted = Signal(str)

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(480)
        self.setMinimumHeight(260)

    def _fit_to_screen(self):
        """Size dialog comfortably to its content while clamping to screen bounds."""
        screen = self.screen() or QtGui.QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            max_w = min(860, int(avail.width() * 0.90))
            max_h = min(680, int(avail.height() * 0.88))
        else:
            max_w, max_h = 860, 680

        hint = self.sizeHint()
        content_h = self._scroll_content.sizeHint().height() if hasattr(self, "_scroll_content") else 0
        # Add ~175px for the pinned Syntax Preview + Submit/Paste/Cancel footer + margins
        desired_h = max(hint.height(), content_h + 175) if content_h else hint.height() + 16
        target_w = max(self.minimumWidth(), min(hint.width() + 28, max_w))
        target_h = max(self.minimumHeight(), min(desired_h, max_h))
        self.resize(target_w, target_h)

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_to_screen()

    def setup_footer(self, main_layout):
        # Move all body controls into a QScrollArea so tall dialogs never clip off-screen
        scroll_content = QtWidgets.QWidget()
        self._scroll_content = scroll_content
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 6, 0)
        scroll_layout.setSpacing(main_layout.spacing() if main_layout.spacing() >= 0 else 8)

        while main_layout.count():
            item = main_layout.takeAt(0)
            if item.widget() is not None:
                scroll_layout.addWidget(item.widget())
            elif item.layout() is not None:
                scroll_layout.addLayout(item.layout())

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area, stretch=1)

        # Pinned footer: Live syntax preview frame + Action buttons (always visible)
        preview_box = QtWidgets.QGroupBox("Syntax Preview")
        p_layout = QtWidgets.QVBoxLayout(preview_box)
        p_layout.setContentsMargins(8, 8, 8, 8)

        self.txt_syntax = QtWidgets.QLineEdit()
        self.txt_syntax.setReadOnly(True)
        self.txt_syntax.setStyleSheet("background-color: #1E1E1E; color: #4EC9B0; font-family: 'Consolas'; font-weight: bold;")
        p_layout.addWidget(self.txt_syntax)
        main_layout.addWidget(preview_box)

        # Buttons
        btn_box = QtWidgets.QHBoxLayout()
        btn_box.setSpacing(8)

        self.btn_submit = QtWidgets.QPushButton("Submit")
        self.btn_submit.clicked.connect(self._on_submit)
        btn_box.addWidget(self.btn_submit)

        self.btn_paste = QtWidgets.QPushButton("Paste to Command Bar")
        self.btn_paste.setStyleSheet("background-color: #37373D; border-color: #555555;")
        self.btn_paste.clicked.connect(self._on_paste)
        btn_box.addWidget(self.btn_paste)

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.setStyleSheet("background-color: #37373D; border-color: #555555;")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        main_layout.addLayout(btn_box)
        self._fit_to_screen()

    def setup_observation_filter(self, main_layout):
        """Builds an interactive Observation Filter (if / in) box for row subsetting."""
        filter_box = QtWidgets.QGroupBox("Observation Filter (Optional — Restrict Rows / Subsample)")
        f_layout = QtWidgets.QFormLayout(filter_box)
        f_layout.setContentsMargins(10, 8, 10, 8)
        f_layout.setSpacing(6)

        self.txt_if = QtWidgets.QLineEdit()
        self.txt_if.setPlaceholderText("Row condition (e.g., year >= 2015 & cards_issued > 0)")
        self.txt_if.textChanged.connect(self.update_syntax)
        f_layout.addRow("Row Condition (if):", self.txt_if)

        in_row_layout = QtWidgets.QHBoxLayout()
        self.txt_in = QtWidgets.QLineEdit()
        self.txt_in.setPlaceholderText("Row range (e.g., 1/200 or 50/l)")
        self.txt_in.textChanged.connect(self.update_syntax)
        in_row_layout.addWidget(self.txt_in, stretch=2)

        self.btn_grid_sel = QtWidgets.QPushButton("Use Grid Selection")
        self.btn_grid_sel.setToolTip("Auto-fill row range from rows currently highlighted in the Data Grid")
        self.btn_grid_sel.setStyleSheet("background-color: #2D3139; color: #78D7C1; border: 1px solid #4B5263; padding: 3px 8px;")
        self.btn_grid_sel.clicked.connect(self._populate_from_grid_selection)
        in_row_layout.addWidget(self.btn_grid_sel, stretch=1)

        f_layout.addRow("Observation Range (in):", in_row_layout)
        main_layout.addWidget(filter_box)

        # Attempt auto-population if rows are selected in grid
        self._populate_from_grid_selection()

    def _populate_from_grid_selection(self):
        parent = self.parent()
        if parent and hasattr(parent, 'data_grid') and parent.data_grid:
            table_view = getattr(parent.data_grid, 'table_view', None)
            if table_view and table_view.selectionModel():
                sel = table_view.selectionModel()
                indexes = sel.selectedRows() if sel.selectedRows() else sel.selectedIndexes()
                if indexes:
                    rows = sorted(set(idx.row() for idx in indexes))
                    start_1b = rows[0] + 1
                    end_1b = rows[-1] + 1
                    if hasattr(self, 'txt_in'):
                        self.txt_in.setText(f"{start_1b}/{end_1b}")

    def get_if_in_syntax(self) -> str:
        """Returns the formatted ' if <expr>' and/or ' in <range>' string."""
        parts = []
        if hasattr(self, 'txt_if'):
            if_val = self.txt_if.text().strip()
            if if_val:
                parts.append(f"if {if_val}")
        if hasattr(self, 'txt_in'):
            in_val = self.txt_in.text().strip()
            if in_val:
                parts.append(f"in {in_val}")
        return (" " + " ".join(parts)) if parts else ""

    def update_syntax(self):
        # Overridden by child classes
        pass

    def _set_model_preview(self, command="", *, missing=""):
        self.txt_syntax.setText(command)
        self.btn_submit.setEnabled(bool(command))
        self.btn_paste.setEnabled(bool(command))
        if missing:
            self.lbl_equation.setText(missing)

    def _on_submit(self):
        cmd = self.txt_syntax.text().strip()
        if cmd:
            self.accept()
            self.command_executed.emit(cmd)

    def _on_paste(self):
        cmd = self.txt_syntax.text().strip()
        if cmd:
            self.accept()
            self.command_pasted.emit(cmd)


class RegressionDialog(BaseBridgeDialog):
    """Point-and-click OLS dialog with robust/clustered SE options, variable guidance, and live equation bridge."""

    def __init__(self, dataset, parent=None):
        super().__init__("Linear Regression (regress)", parent)
        self.dataset = dataset
        self.setMinimumWidth(560)
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []
        dep_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.dataset.df[c])]

        # Educational Guidance Card
        guide_box = QtWidgets.QFrame()
        guide_box.setObjectName("guideBox")
        guide_box.setStyleSheet(
            "QFrame#guideBox { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }"
        )
        guide_layout = QtWidgets.QVBoxLayout(guide_box)
        guide_layout.setContentsMargins(10, 8, 10, 8)
        guide_layout.setSpacing(4)

        lbl_guide_title = QtWidgets.QLabel("📘 Econometric Variable Guide:")
        lbl_guide_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        guide_layout.addWidget(lbl_guide_title)

        lbl_guide_dep = QtWidgets.QLabel(
            "• <b>Dependent Variable (Outcome Y):</b> The variable you want to explain, predict, or measure "
            "(e.g., <i>Cards Issued</i>, <i>Cash Withdrawn</i>, <i>Wage</i>). Placed on the <b>left-hand side</b> of the equation (must be numeric)."
        )
        lbl_guide_dep.setStyleSheet("font-size: 9.5pt;")
        lbl_guide_dep.setWordWrap(True)
        guide_layout.addWidget(lbl_guide_dep)

        lbl_guide_indep = QtWidgets.QLabel(
            "• <b>Independent Variables (Predictors X):</b> The explanatory factors or drivers hypothesized to cause or explain changes "
            "in Y (e.g., <i>Year</i>, <i>Number of ATMs</i>, <i>Education</i>). Placed on the <b>right-hand side</b> of the equation."
        )
        lbl_guide_indep.setStyleSheet("font-size: 9.5pt;")
        lbl_guide_indep.setWordWrap(True)
        guide_layout.addWidget(lbl_guide_indep)

        layout.addWidget(guide_box)

        # 1. Dependent Variable (strictly numeric for linear regression)
        dep_box = QtWidgets.QGroupBox("1. Dependent Variable (Outcome Y — Left-Hand Side)")
        dep_box_layout = QtWidgets.QVBoxLayout(dep_box)
        dep_box_layout.setContentsMargins(10, 8, 10, 8)
        dep_box_layout.setSpacing(4)

        dep_hint = QtWidgets.QLabel("Select the single numeric outcome variable to be explained / predicted:")
        dep_hint.setStyleSheet("font-size: 9pt;")
        dep_box_layout.addWidget(dep_hint)

        self.cmb_dep = QtWidgets.QComboBox()
        self.cmb_dep.addItems(dep_cols)
        self.cmb_dep.currentIndexChanged.connect(self.update_syntax)
        dep_box_layout.addWidget(self.cmb_dep)
        layout.addWidget(dep_box)

        # 2. Independent Variables
        indep_box = QtWidgets.QGroupBox("2. Independent Variables (Predictors / Covariates X — Right-Hand Side)")
        indep_box_layout = QtWidgets.QVBoxLayout(indep_box)
        indep_box_layout.setContentsMargins(10, 8, 10, 8)
        indep_box_layout.setSpacing(4)

        indep_hint = QtWidgets.QLabel("Select one or more predictor variables (hold Ctrl or Shift to select multiple):")
        indep_hint.setStyleSheet("font-size: 9pt;")
        indep_box_layout.addWidget(indep_hint)

        self.lst_indep = QtWidgets.QListWidget()
        self.lst_indep.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        self.lst_indep.addItems(cols)
        for i in range(self.lst_indep.count()):
            item = self.lst_indep.item(i)
            vname = item.text()
            if not self.dataset.is_empty and vname in self.dataset.df.columns:
                if pd.api.types.is_numeric_dtype(self.dataset.df[vname]):
                    item.setToolTip(f"Numeric predictor: {vname}")
                else:
                    item.setToolTip(f"Categorical / text variable: {vname} (will be estimated as factor i.{vname})")
        self.lst_indep.itemSelectionChanged.connect(self.update_syntax)
        self.lst_indep.setMaximumHeight(110)
        indep_box_layout.addWidget(self.lst_indep)
        layout.addWidget(indep_box)

        # 3. Variance Options
        opt_box = QtWidgets.QGroupBox("3. Standard Errors / Covariance (VCE)")
        opt_layout = QtWidgets.QFormLayout(opt_box)
        opt_layout.setContentsMargins(10, 8, 10, 8)

        self.cmb_vce = QtWidgets.QComboBox()
        self.cmb_vce.addItems(["OLS (Classical)", "Robust (HC1)", "HC2", "HC3", "Clustered SE"])
        self.cmb_vce.currentIndexChanged.connect(self._on_vce_changed)
        opt_layout.addRow("VCE Type:", self.cmb_vce)

        self.cmb_cluster = QtWidgets.QComboBox()
        self.cmb_cluster.addItems(["(None)"] + cols)
        self.cmb_cluster.setEnabled(False)
        self.cmb_cluster.currentIndexChanged.connect(self.update_syntax)
        opt_layout.addRow("Cluster Variable:", self.cmb_cluster)

        layout.addWidget(opt_box)

        # 4. Observation Filter (if / in)
        self.setup_observation_filter(layout)

        # 5. Live Econometric Equation Preview
        eq_box = QtWidgets.QGroupBox("Model Specification (Live Equation Preview)")
        eq_layout = QtWidgets.QVBoxLayout(eq_box)
        eq_layout.setContentsMargins(10, 8, 10, 8)

        self.lbl_equation = QtWidgets.QLabel()
        self.lbl_equation.setStyleSheet(
            "background-color: #1E1E1E; color: #E5C07B; font-family: 'Consolas', monospace; "
            "font-size: 10pt; font-weight: bold; padding: 6px; border: 1px solid #3E4451; border-radius: 4px;"
        )
        self.lbl_equation.setWordWrap(True)
        eq_layout.addWidget(self.lbl_equation)
        layout.addWidget(eq_box)

        # Setup footer
        self.setup_footer(layout)
        self.update_syntax()

    def _on_vce_changed(self):
        is_cluster = (self.cmb_vce.currentText() == "Clustered SE")
        self.cmb_cluster.setEnabled(is_cluster)
        self.update_syntax()

    def update_syntax(self):
        dep = self.cmb_dep.currentText()
        indeps = []
        for item in self.lst_indep.selectedItems():
            var_name = item.text()
            if var_name == dep:
                continue
            if not self.dataset.is_empty and var_name in self.dataset.df.columns:
                if not pd.api.types.is_numeric_dtype(self.dataset.df[var_name]):
                    indeps.append(f"i.{var_name}")
                else:
                    indeps.append(var_name)
            else:
                indeps.append(var_name)
        indep_str = " ".join(indeps)

        if not dep:
            self._set_model_preview(missing="Select a numeric outcome variable.")
            return
        vce_type = self.cmb_vce.currentText()
        if vce_type == "Clustered SE" and self.cmb_cluster.currentIndex() < 1:
            self._set_model_preview(missing="Select a grouping variable for clustered standard errors.")
            return

        subsample_note = ""
        if hasattr(self, 'txt_if') and self.txt_if.text().strip():
            subsample_note = f"  [where {self.txt_if.text().strip()}]"
        elif hasattr(self, 'txt_in') and self.txt_in.text().strip():
            subsample_note = f"  [in {self.txt_in.text().strip()}]"

        # Update live econometric equation
        if indeps:
            rhs_terms = " + ".join([f"β{idx+1}·{v}" for idx, v in enumerate(indeps)])
            self.lbl_equation.setText(f"{dep} = β₀ + {rhs_terms} + ε{subsample_note}")
        else:
            self.lbl_equation.setText(f"{dep} = β₀ + ε  (Constant-only baseline model){subsample_note}")

        opts = []
        if vce_type == "Robust (HC1)":
            opts.append("robust")
        elif vce_type == "HC2":
            opts.append("vce(hc2)")
        elif vce_type == "HC3":
            opts.append("vce(hc3)")
        elif vce_type == "Clustered SE":
            opts.append(f"vce(cluster {self.cmb_cluster.currentText()})")

        opt_str = f", {', '.join(opts)}" if opts else ""
        if_in_str = self.get_if_in_syntax()
        cmd = f"regress {dep} {indep_str}{if_in_str}{opt_str}".strip()
        self._set_model_preview(cmd)


class SummarizeDialog(BaseBridgeDialog):
    """Summary statistics dialog."""

    def __init__(self, dataset, parent=None):
        super().__init__("Descriptive Statistics (summarize)", parent)
        self.dataset = dataset
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []

        lbl_vars = QtWidgets.QLabel("Select variables (leave empty for all):")
        layout.addWidget(lbl_vars)

        self.lst_vars = QtWidgets.QListWidget()
        self.lst_vars.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        self.lst_vars.addItems(cols)
        self.lst_vars.itemSelectionChanged.connect(self.update_syntax)
        layout.addWidget(self.lst_vars)

        self.chk_detail = QtWidgets.QCheckBox("Detailed statistics (percentiles, skewness, kurtosis)")
        self.chk_detail.stateChanged.connect(self.update_syntax)
        layout.addWidget(self.chk_detail)

        self.setup_observation_filter(layout)
        self.setup_footer(layout)
        self.update_syntax()

    def update_syntax(self):
        selected = [item.text() for item in self.lst_vars.selectedItems()]
        var_str = " ".join(selected)
        detail_str = ", detail" if self.chk_detail.isChecked() else ""
        if_in_str = self.get_if_in_syntax()
        cmd = f"summarize {var_str}{if_in_str}{detail_str}".strip()
        self.txt_syntax.setText(cmd)


class TabulateDialog(BaseBridgeDialog):
    """Frequency and cross-tabulation dialog."""

    def __init__(self, dataset, parent=None):
        super().__init__("Tables & Frequencies (tabulate)", parent)
        self.dataset = dataset
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []

        f_layout = QtWidgets.QFormLayout()
        self.cmb_row = QtWidgets.QComboBox()
        self.cmb_row.addItems(cols)
        self.cmb_row.currentIndexChanged.connect(self.update_syntax)
        f_layout.addRow("Row Variable:", self.cmb_row)

        self.cmb_col = QtWidgets.QComboBox()
        self.cmb_col.addItems(["(None - One-Way Table)"] + cols)
        self.cmb_col.currentIndexChanged.connect(self.update_syntax)
        f_layout.addRow("Column Variable (optional):", self.cmb_col)

        layout.addLayout(f_layout)
        self.setup_observation_filter(layout)
        self.setup_footer(layout)
        self.update_syntax()

    def update_syntax(self):
        r = self.cmb_row.currentText()
        c = self.cmb_col.currentText()
        if_in_str = self.get_if_in_syntax()
        if c and c != "(None - One-Way Table)":
            cmd = f"tabulate {r} {c}{if_in_str}".strip()
        else:
            cmd = f"tabulate {r}{if_in_str}".strip()
        self.txt_syntax.setText(cmd)


class ReghdfeDialog(BaseBridgeDialog):
    """High-Dimensional Multi-Way Fixed Effects dialog."""

    def __init__(self, dataset, parent=None):
        super().__init__("High-Dimensional Fixed Effects (reghdfe)", parent)
        self.dataset = dataset
        self.setMinimumWidth(560)
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []
        dep_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.dataset.df[c])]

        # Educational Guidance Card
        guide_box = QtWidgets.QFrame()
        guide_box.setObjectName("guideBox")
        guide_box.setStyleSheet(
            "QFrame#guideBox { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }"
        )
        guide_layout = QtWidgets.QVBoxLayout(guide_box)
        guide_layout.setContentsMargins(10, 8, 10, 8)
        guide_layout.setSpacing(4)

        lbl_guide_title = QtWidgets.QLabel("📘 High-Dimensional Fixed Effects Variable Guide:")
        lbl_guide_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        guide_layout.addWidget(lbl_guide_title)

        lbl_guide_dep = QtWidgets.QLabel(
            "• <b>Dependent Variable (Outcome Y):</b> The outcome you want to explain (e.g., <i>Cards Issued</i>, <i>Cash Withdrawn</i>, <i>Wage</i>). Must be numeric."
        )
        lbl_guide_dep.setStyleSheet("font-size: 9.5pt;")
        lbl_guide_dep.setWordWrap(True)
        guide_layout.addWidget(lbl_guide_dep)

        lbl_guide_indep = QtWidgets.QLabel(
            "• <b>Independent Variables (Predictors X):</b> Explanatory factors of primary interest (e.g., <i>ATMs count</i>, <i>Policy treatment</i>)."
        )
        lbl_guide_indep.setStyleSheet("font-size: 9.5pt;")
        lbl_guide_indep.setWordWrap(True)
        guide_layout.addWidget(lbl_guide_indep)

        lbl_guide_fe = QtWidgets.QLabel(
            "• <b>Absorbed Fixed Effects (α):</b> High-cardinality categorical dimensions (e.g., <i>period</i>, <i>region</i>, <i>entity</i>) to project out."
        )
        lbl_guide_fe.setStyleSheet("font-size: 9.5pt;")
        lbl_guide_fe.setWordWrap(True)
        guide_layout.addWidget(lbl_guide_fe)

        layout.addWidget(guide_box)

        # 1. Dependent Variable
        dep_box = QtWidgets.QGroupBox("1. Dependent Variable (Outcome Y — Left-Hand Side)")
        dep_box_layout = QtWidgets.QVBoxLayout(dep_box)
        dep_box_layout.setContentsMargins(10, 8, 10, 8)
        dep_box_layout.setSpacing(4)

        dep_hint = QtWidgets.QLabel("Select the single numeric outcome variable to be modeled (Y):")
        dep_hint.setStyleSheet("font-size: 9pt;")
        dep_box_layout.addWidget(dep_hint)

        self.cmb_dep = QtWidgets.QComboBox()
        self.cmb_dep.addItems(dep_cols)
        self.cmb_dep.currentIndexChanged.connect(self.update_syntax)
        dep_box_layout.addWidget(self.cmb_dep)
        layout.addWidget(dep_box)

        # 2. Regressors
        indep_box = QtWidgets.QGroupBox("2. Independent Variables (Predictors X — Hold Ctrl/Shift to multi-select)")
        indep_box_layout = QtWidgets.QVBoxLayout(indep_box)
        indep_box_layout.setContentsMargins(10, 8, 10, 8)
        self.lst_indeps = QtWidgets.QListWidget()
        self.lst_indeps.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_indeps.addItems(cols)
        self.lst_indeps.itemSelectionChanged.connect(self.update_syntax)
        self.lst_indeps.setMaximumHeight(85)
        indep_box_layout.addWidget(self.lst_indeps)
        layout.addWidget(indep_box)

        # 3. Absorbed Fixed Effects
        fe_box = QtWidgets.QGroupBox("3. Absorbed Fixed Effects (α — e.g. Entity, Time, Group)")
        fe_box_layout = QtWidgets.QVBoxLayout(fe_box)
        fe_box_layout.setContentsMargins(10, 8, 10, 8)
        self.lst_absorb = QtWidgets.QListWidget()
        self.lst_absorb.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_absorb.addItems(cols)
        self.lst_absorb.itemSelectionChanged.connect(self.update_syntax)
        self.lst_absorb.setMaximumHeight(85)
        fe_box_layout.addWidget(self.lst_absorb)
        layout.addWidget(fe_box)

        # 4. SE options
        se_box = QtWidgets.QHBoxLayout()
        se_box.addWidget(QtWidgets.QLabel("Std. Errors:"))
        self.cmb_se = QtWidgets.QComboBox()
        self.cmb_se.addItems(["Homoskedastic (OLS)", "Robust (HC1)", "Clustered"])
        self.cmb_se.currentIndexChanged.connect(self.update_syntax)
        se_box.addWidget(self.cmb_se)

        self.cmb_cluster = QtWidgets.QComboBox()
        self.cmb_cluster.addItems(["(Cluster Var)"] + cols)
        self.cmb_cluster.currentIndexChanged.connect(self.update_syntax)
        se_box.addWidget(self.cmb_cluster)
        layout.addLayout(se_box)

        # 5. Observation Filter (if / in)
        self.setup_observation_filter(layout)

        # 6. Live Equation Preview
        eq_box = QtWidgets.QGroupBox("Model Specification (Live Equation Preview)")
        eq_layout = QtWidgets.QVBoxLayout(eq_box)
        eq_layout.setContentsMargins(10, 8, 10, 8)

        self.lbl_equation = QtWidgets.QLabel()
        self.lbl_equation.setStyleSheet(
            "background-color: #1E1E1E; color: #E5C07B; font-family: 'Consolas', monospace; "
            "font-size: 10pt; font-weight: bold; padding: 6px; border: 1px solid #3E4451; border-radius: 4px;"
        )
        self.lbl_equation.setWordWrap(True)
        eq_layout.addWidget(self.lbl_equation)
        layout.addWidget(eq_box)

        self.setup_footer(layout)
        self.update_syntax()

    def update_syntax(self):
        dep = self.cmb_dep.currentText()
        indeps = []
        for item in self.lst_indeps.selectedItems():
            var_name = item.text()
            if var_name == dep:
                continue
            if not self.dataset.is_empty and var_name in self.dataset.df.columns:
                if not pd.api.types.is_numeric_dtype(self.dataset.df[var_name]):
                    indeps.append(f"i.{var_name}")
                else:
                    indeps.append(var_name)
            else:
                indeps.append(var_name)
        absorbs = [item.text() for item in self.lst_absorb.selectedItems()]
        se_idx = self.cmb_se.currentIndex()
        self.cmb_cluster.setEnabled(se_idx == 2)
        if not dep:
            self._set_model_preview(missing="Select a numeric outcome variable.")
            return
        if not indeps:
            self._set_model_preview(missing="Select at least one predictor.")
            return
        if not absorbs:
            self._set_model_preview(missing="Select at least one fixed effect to absorb.")
            return
        if se_idx == 2 and self.cmb_cluster.currentIndex() < 1:
            self._set_model_preview(missing="Select a grouping variable for clustered standard errors.")
            return

        indeps_str = " ".join(indeps)
        fe_repr = " + ".join(f"α_{fe}" for fe in absorbs)
        rhs_terms = " + ".join(f"β{idx+1}·{v}" for idx, v in enumerate(indeps))
        subsample_note = ""
        if hasattr(self, 'txt_if') and self.txt_if.text().strip():
            subsample_note = f"  [where {self.txt_if.text().strip()}]"
        elif hasattr(self, 'txt_in') and self.txt_in.text().strip():
            subsample_note = f"  [in {self.txt_in.text().strip()}]"

        self.lbl_equation.setText(f"{dep} = {rhs_terms} + {fe_repr} + ε{subsample_note}")
        opts = [f"absorb({' '.join(absorbs)})"]
        if se_idx == 1:
            opts.append("vce(robust)")
        elif se_idx == 2:
            opts.append(f"vce(cluster {self.cmb_cluster.currentText()})")

        if_in_str = self.get_if_in_syntax()
        cmd = f"reghdfe {dep} {indeps_str}{if_in_str}, {', '.join(opts)}"
        self._set_model_preview(cmd)


class ProbitDialog(BaseBridgeDialog):
    """Binary Choice Regression dialog (supports Logit and Probit)."""

    def __init__(self, dataset, parent=None, initial_model="probit"):
        self.initial_model = (initial_model or "probit").lower()
        title = "Binary Choice Modeling (logit / probit)"
        super().__init__(title, parent)
        self.dataset = dataset
        self.setMinimumWidth(560)
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []
        dep_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.dataset.df[c])]

        # Model Choice
        m_box = QtWidgets.QHBoxLayout()
        m_box.addWidget(QtWidgets.QLabel("Binary Model Link:"))
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(["Logistic Regression (logit)", "Probit Model (probit)"])
        if "logit" in self.initial_model:
            self.cmb_model.setCurrentIndex(0)
        else:
            self.cmb_model.setCurrentIndex(1)
        self.cmb_model.currentIndexChanged.connect(self.update_syntax)
        m_box.addWidget(self.cmb_model, stretch=1)
        layout.addLayout(m_box)

        # Educational Guidance Card
        guide_box = QtWidgets.QFrame()
        guide_box.setObjectName("guideBox")
        guide_box.setStyleSheet(
            "QFrame#guideBox { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }"
        )
        guide_layout = QtWidgets.QVBoxLayout(guide_box)
        guide_layout.setContentsMargins(10, 8, 10, 8)
        guide_layout.setSpacing(4)

        lbl_guide_title = QtWidgets.QLabel("📘 Binary Choice Variable Guide:")
        lbl_guide_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        guide_layout.addWidget(lbl_guide_title)

        lbl_guide_dep = QtWidgets.QLabel(
            "• <b>Binary Outcome (Y ∈ {0, 1}):</b> The indicator representing event occurrence "
            "(e.g., <i>1 = Transaction Approved</i>, <i>0 = Denied</i>). Placed on the <b>left-hand side</b>."
        )
        lbl_guide_dep.setStyleSheet("font-size: 9.5pt;")
        lbl_guide_dep.setWordWrap(True)
        guide_layout.addWidget(lbl_guide_dep)

        lbl_guide_indep = QtWidgets.QLabel(
            "• <b>Independent Variables (Predictors X):</b> Explanatory variables hypothesized to affect the probability "
            "of the event occurring (P(Y = 1 | X)). Placed on the <b>right-hand side</b>."
        )
        lbl_guide_indep.setStyleSheet("font-size: 9.5pt;")
        lbl_guide_indep.setWordWrap(True)
        guide_layout.addWidget(lbl_guide_indep)

        layout.addWidget(guide_box)

        # 1. Binary Outcome Variable
        dep_box = QtWidgets.QGroupBox("1. Binary Outcome Variable (Y ∈ {0, 1})")
        dep_box_layout = QtWidgets.QVBoxLayout(dep_box)
        dep_box_layout.setContentsMargins(10, 8, 10, 8)
        dep_box_layout.setSpacing(4)

        dep_hint = QtWidgets.QLabel("Select the binary indicator variable to model:")
        dep_hint.setStyleSheet("font-size: 9pt;")
        dep_box_layout.addWidget(dep_hint)

        self.cmb_dep = QtWidgets.QComboBox()
        self.cmb_dep.addItems(dep_cols)
        self.cmb_dep.currentIndexChanged.connect(self.update_syntax)
        dep_box_layout.addWidget(self.cmb_dep)
        layout.addWidget(dep_box)

        # 2. Independent Variables
        indep_box = QtWidgets.QGroupBox("2. Independent Variables (Predictors X — Hold Ctrl/Shift to multi-select)")
        indep_box_layout = QtWidgets.QVBoxLayout(indep_box)
        indep_box_layout.setContentsMargins(10, 8, 10, 8)

        self.lst_indeps = QtWidgets.QListWidget()
        self.lst_indeps.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_indeps.addItems(cols)
        self.lst_indeps.itemSelectionChanged.connect(self.update_syntax)
        self.lst_indeps.setMaximumHeight(100)
        indep_box_layout.addWidget(self.lst_indeps)
        layout.addWidget(indep_box)

        # 3. SE options
        se_box = QtWidgets.QHBoxLayout()
        se_box.addWidget(QtWidgets.QLabel("Std. Errors:"))
        self.cmb_se = QtWidgets.QComboBox()
        self.cmb_se.addItems(["Standard MLE", "Robust (HC1)", "Clustered"])
        self.cmb_se.currentIndexChanged.connect(self.update_syntax)
        se_box.addWidget(self.cmb_se)

        self.cmb_cluster = QtWidgets.QComboBox()
        self.cmb_cluster.addItems(["(Cluster Var)"] + cols)
        self.cmb_cluster.currentIndexChanged.connect(self.update_syntax)
        se_box.addWidget(self.cmb_cluster)
        layout.addLayout(se_box)

        # 4. Observation Filter (if / in)
        self.setup_observation_filter(layout)

        # 5. Live Equation Preview
        eq_box = QtWidgets.QGroupBox("Model Specification (Live Equation Preview)")
        eq_layout = QtWidgets.QVBoxLayout(eq_box)
        eq_layout.setContentsMargins(10, 8, 10, 8)

        self.lbl_equation = QtWidgets.QLabel()
        self.lbl_equation.setStyleSheet(
            "background-color: #1E1E1E; color: #E5C07B; font-family: 'Consolas', monospace; "
            "font-size: 10pt; font-weight: bold; padding: 6px; border: 1px solid #3E4451; border-radius: 4px;"
        )
        self.lbl_equation.setWordWrap(True)
        eq_layout.addWidget(self.lbl_equation)
        layout.addWidget(eq_box)

        self.setup_footer(layout)
        self.update_syntax()

    def update_syntax(self):
        dep = self.cmb_dep.currentText()
        is_logit = "logit" in self.cmb_model.currentText().lower()
        cmd_name = "logit" if is_logit else "probit"

        indeps = []
        for item in self.lst_indeps.selectedItems():
            var_name = item.text()
            if var_name == dep:
                continue
            if not self.dataset.is_empty and var_name in self.dataset.df.columns:
                if not pd.api.types.is_numeric_dtype(self.dataset.df[var_name]):
                    indeps.append(f"i.{var_name}")
                else:
                    indeps.append(var_name)
            else:
                indeps.append(var_name)
        indeps_str = " ".join(indeps)

        se_idx = self.cmb_se.currentIndex()
        self.cmb_cluster.setEnabled(se_idx == 2)
        if not dep:
            self._set_model_preview(missing="Select a binary outcome variable.")
            return
        if se_idx == 2 and self.cmb_cluster.currentIndex() < 1:
            self._set_model_preview(missing="Select a grouping variable for clustered standard errors.")
            return

        subsample_note = ""
        if hasattr(self, 'txt_if') and self.txt_if.text().strip():
            subsample_note = f"  [where {self.txt_if.text().strip()}]"
        elif hasattr(self, 'txt_in') and self.txt_in.text().strip():
            subsample_note = f"  [in {self.txt_in.text().strip()}]"

        # Update live equation preview
        if is_logit:
            if indeps:
                rhs_terms = " + ".join([f"β{idx+1}·{v}" for idx, v in enumerate(indeps)])
                self.lbl_equation.setText(f"Pr({dep} = 1 | X) = Λ(β₀ + {rhs_terms}) = 1 / (1 + e^-(β₀ + ...)){subsample_note}")
            else:
                self.lbl_equation.setText(f"Pr({dep} = 1 | X) = Λ(β₀)  (Baseline odds){subsample_note}")
        else:
            if indeps:
                rhs_terms = " + ".join([f"β{idx+1}·{v}" for idx, v in enumerate(indeps)])
                self.lbl_equation.setText(f"Pr({dep} = 1 | X) = Φ(β₀ + {rhs_terms}){subsample_note}")
            else:
                self.lbl_equation.setText(f"Pr({dep} = 1 | X) = Φ(β₀)  (Baseline probability){subsample_note}")

        opts = []
        if se_idx == 1:
            opts.append("robust")
        elif se_idx == 2:
            opts.append(f"vce(cluster {self.cmb_cluster.currentText()})")

        opts_str = f", {' '.join(opts)}" if opts else ""
        if_in_str = self.get_if_in_syntax()
        cmd = f"{cmd_name} {dep} {indeps_str}{if_in_str}{opts_str}"
        self._set_model_preview(cmd)


BinaryChoiceDialog = ProbitDialog


class MarginsDialog(BaseBridgeDialog):
    """Average Marginal Effects dialog."""

    def __init__(self, dataset, parent=None):
        super().__init__("Marginal Effects (margins)", parent)
        self.dataset = dataset
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        layout.addWidget(QtWidgets.QLabel("Computes Average Marginal Effects (dydx) on active fitted model."))
        
        self.chk_all = QtWidgets.QCheckBox("All independent variables: dydx(*)")
        self.chk_all.setChecked(True)
        self.chk_all.stateChanged.connect(self.update_syntax)
        layout.addWidget(self.chk_all)

        self.setup_footer(layout)
        self.update_syntax()

    def update_syntax(self):
        cmd = "margins, dydx(*)" if self.chk_all.isChecked() else "margins"
        self.txt_syntax.setText(cmd)


class CsdidDialog(BaseBridgeDialog):
    """Callaway & Sant'Anna (2021) Staggered DiD dialog."""

    def __init__(self, dataset, parent=None):
        super().__init__("Staggered Difference-in-Differences (csdid)", parent)
        self.dataset = dataset
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.dataset.df[c])]
        dep_cols = num_cols if num_cols else cols

        f_layout = QtWidgets.QFormLayout()
        self.cmb_dep = QtWidgets.QComboBox()
        self.cmb_dep.addItems(dep_cols)
        self.cmb_dep.currentIndexChanged.connect(self.update_syntax)
        f_layout.addRow("Outcome Variable (depvar):", self.cmb_dep)

        self.cmb_ivar = QtWidgets.QComboBox()
        self.cmb_ivar.addItems(cols)
        if getattr(self.dataset, 'panel_id', None) in cols:
            self.cmb_ivar.setCurrentText(self.dataset.panel_id)
        self.cmb_ivar.currentIndexChanged.connect(self.update_syntax)
        f_layout.addRow("Panel Unit ID (ivar):", self.cmb_ivar)

        self.cmb_timevar = QtWidgets.QComboBox()
        self.cmb_timevar.addItems(cols)
        if getattr(self.dataset, 'panel_time', None) in cols:
            self.cmb_timevar.setCurrentText(self.dataset.panel_time)
        self.cmb_timevar.currentIndexChanged.connect(self.update_syntax)
        f_layout.addRow("Time Variable (timevar):", self.cmb_timevar)

        self.cmb_gvar = QtWidgets.QComboBox()
        self.cmb_gvar.addItems(cols)
        self.cmb_gvar.currentIndexChanged.connect(self.update_syntax)
        f_layout.addRow("Treatment Cohort (gvar):", self.cmb_gvar)

        self.cmb_control = QtWidgets.QComboBox()
        self.cmb_control.addItems(["notyettreated", "nevertreated"])
        self.cmb_control.currentIndexChanged.connect(self.update_syntax)
        f_layout.addRow("Control Group:", self.cmb_control)

        layout.addLayout(f_layout)

        self.setup_observation_filter(layout)
        self.setup_footer(layout)
        self.update_syntax()

    def update_syntax(self):
        dep = self.cmb_dep.currentText()
        ivar = self.cmb_ivar.currentText()
        timevar = self.cmb_timevar.currentText()
        gvar = self.cmb_gvar.currentText()
        ctrl = self.cmb_control.currentText()
        if_in_str = self.get_if_in_syntax()

        if dep and ivar and timevar and gvar:
            cmd = f"csdid {dep}{if_in_str}, ivar({ivar}) timevar({timevar}) gvar({gvar}) control({ctrl})"
        else:
            cmd = "csdid"
        self.txt_syntax.setText(cmd)


class EsttabDialog(BaseBridgeDialog):
    """Multi-Model Table Exporter (esttab) dialog."""

    def __init__(self, dataset, parent=None):
        super().__init__("Export Models Table (esttab)", parent)
        self.dataset = dataset
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        f_layout = QtWidgets.QFormLayout()

        self.cmb_format = QtWidgets.QComboBox()
        self.cmb_format.addItems(["Monospace Console", "LaTeX (booktabs)", "Word Document (.docx)", "HTML Table", "CSV Spreadsheet"])
        self.cmb_format.currentIndexChanged.connect(self.update_syntax)
        f_layout.addRow("Export Format:", self.cmb_format)

        self.txt_title = QtWidgets.QLineEdit("Regression Results")
        self.txt_title.textChanged.connect(self.update_syntax)
        f_layout.addRow("Table Title:", self.txt_title)

        file_box = QtWidgets.QHBoxLayout()
        self.txt_file = QtWidgets.QLineEdit()
        self.txt_file.setPlaceholderText("Optional target file (e.g. table.tex, results.docx)")
        self.txt_file.textChanged.connect(self.update_syntax)
        file_box.addWidget(self.txt_file)

        btn_browse = QtWidgets.QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_file)
        file_box.addWidget(btn_browse)
        f_layout.addRow("Output File:", file_box)

        layout.addLayout(f_layout)

        self.setup_footer(layout)
        self.update_syntax()

    def _browse_file(self):
        fmt = self.cmb_format.currentText()
        filt = "All Files (*.*)"
        if "LaTeX" in fmt:
            filt = "LaTeX Files (*.tex);;All Files (*.*)"
        elif "Word" in fmt:
            filt = "Word Documents (*.docx);;All Files (*.*)"
        elif "HTML" in fmt:
            filt = "HTML Files (*.html);;All Files (*.*)"
        elif "CSV" in fmt:
            filt = "CSV Files (*.csv);;All Files (*.*)"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Table As", "", filt)
        if path:
            self.txt_file.setText(path)

    def update_syntax(self):
        fmt = self.cmb_format.currentText()
        title = self.txt_title.text().strip()
        filepath = self.txt_file.text().strip()

        using_str = f" using {filepath}" if filepath else ""
        opts = []
        if title and title != "Regression Results":
            opts.append(f'title("{title}")')

        if "LaTeX" in fmt:
            opts.append("booktabs")
        elif "Word" in fmt:
            opts.append("docx")
        elif "HTML" in fmt:
            opts.append("html")
        elif "CSV" in fmt:
            opts.append("csv")

        opts_str = f", {', '.join(opts)}" if opts else ""
        cmd = f"esttab{using_str}{opts_str}"
        self.txt_syntax.setText(cmd)


class GraphDialog(BaseBridgeDialog):
    """
    Interactive Graph Studio Dialog.
    Supports Twoway Scatter, Line, Binscatter, RD Plot, Local Projections IRF,
    Histogram, Box Plot, Correlation Heatmap, Coefficient Plot, and Synthetic Control.
    """
    def __init__(self, dataset, parent=None, initial_type=None):
        super().__init__("Graph Studio - Potatostics", parent)
        self.dataset = dataset
        self.setMinimumWidth(540)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # 1. Graph Category / Type Selector
        type_box = QtWidgets.QGroupBox("Graph Type")
        t_layout = QtWidgets.QHBoxLayout(type_box)
        self.cmb_type = QtWidgets.QComboBox()
        self.cmb_type.addItems([
            "Two-way Scatter (scatter)",
            "Two-way Line (line)",
            "Binned Scatter Plot (binscatter)",
            "Regression Discontinuity (rdplot)",
            "Local Projections IRF (irf)",
            "Histogram (histogram)",
            "Box Plot (boxplot)",
            "Correlation Heatmap (heatmap)",
            "Coefficient Plot (coefplot)",
            "Synthetic Control (synth_plot)",
            "Dynamic Event Study (eventstudy)"
        ])
        if initial_type:
            for i in range(self.cmb_type.count()):
                if initial_type.lower() in self.cmb_type.itemText(i).lower():
                    self.cmb_type.setCurrentIndex(i)
                    break
        self.cmb_type.currentIndexChanged.connect(self._on_type_changed)
        t_layout.addWidget(QtWidgets.QLabel("Select Graph:"))
        t_layout.addWidget(self.cmb_type, stretch=1)
        layout.addWidget(type_box)

        # 2. Variable Selection
        self.var_box = QtWidgets.QGroupBox("Variables")
        v_layout = QtWidgets.QFormLayout(self.var_box)

        self.cmb_y = QtWidgets.QComboBox()
        self.cmb_x = QtWidgets.QComboBox()
        self.cmb_by = QtWidgets.QComboBox()

        vars_list = list(self.dataset.df.columns) if self.dataset and not self.dataset.is_empty else []
        num_vars = [c for c in vars_list if pd.api.types.is_numeric_dtype(self.dataset.df[c])]
        y_vars = num_vars if num_vars else vars_list
        self.cmb_y.addItems(y_vars)
        self.cmb_x.addItems(vars_list)
        self.cmb_by.addItems(["(None)"] + vars_list)

        if len(vars_list) > 1:
            self.cmb_x.setCurrentIndex(1)

        self.lbl_y = QtWidgets.QLabel("Y Variable (Vertical Axis - Outcome / Measured):")
        self.lbl_x = QtWidgets.QLabel("X Variable (Horizontal Axis - Explanatory / Time):")
        self.lbl_by = QtWidgets.QLabel("Group / By Variable:")

        v_layout.addRow(self.lbl_y, self.cmb_y)
        v_layout.addRow(self.lbl_x, self.cmb_x)
        v_layout.addRow(self.lbl_by, self.cmb_by)

        self.cmb_y.currentIndexChanged.connect(self.update_syntax)
        self.cmb_x.currentIndexChanged.connect(self.update_syntax)
        self.cmb_by.currentIndexChanged.connect(self.update_syntax)
        layout.addWidget(self.var_box)

        # 3. Graph Options Box
        self.opt_box = QtWidgets.QGroupBox("Plot Options")
        o_layout = QtWidgets.QFormLayout(self.opt_box)

        # Bins spinbox
        self.spin_bins = QtWidgets.QSpinBox()
        self.spin_bins.setRange(2, 200)
        self.spin_bins.setValue(20)
        self.spin_bins.valueChanged.connect(self.update_syntax)
        self.lbl_bins = QtWidgets.QLabel("Number of Bins:")
        o_layout.addRow(self.lbl_bins, self.spin_bins)

        # RD Cutoff
        self.spin_cutoff = QtWidgets.QDoubleSpinBox()
        self.spin_cutoff.setRange(-1e9, 1e9)
        self.spin_cutoff.setValue(0.0)
        self.spin_cutoff.setSingleStep(0.5)
        self.spin_cutoff.valueChanged.connect(self.update_syntax)
        self.lbl_cutoff = QtWidgets.QLabel("Cutoff (c):")
        o_layout.addRow(self.lbl_cutoff, self.spin_cutoff)

        # RD Bandwidth
        self.spin_bw = QtWidgets.QDoubleSpinBox()
        self.spin_bw.setRange(0.0, 1e9)
        self.spin_bw.setValue(0.0)
        self.spin_bw.setSingleStep(0.5)
        self.spin_bw.setSpecialValueText("Auto (Silverman)")
        self.spin_bw.valueChanged.connect(self.update_syntax)
        self.lbl_bw = QtWidgets.QLabel("Bandwidth (h):")
        o_layout.addRow(self.lbl_bw, self.spin_bw)

        # IRF Horizon
        self.spin_horizon = QtWidgets.QSpinBox()
        self.spin_horizon.setRange(1, 100)
        self.spin_horizon.setValue(8)
        self.spin_horizon.valueChanged.connect(self.update_syntax)
        self.lbl_horizon = QtWidgets.QLabel("Forecast Horizon (H):")
        o_layout.addRow(self.lbl_horizon, self.spin_horizon)

        # Binscatter Fit Line
        self.cmb_line = QtWidgets.QComboBox()
        self.cmb_line.addItems(["linear", "none"])
        self.cmb_line.currentIndexChanged.connect(self.update_syntax)
        self.lbl_line = QtWidgets.QLabel("Fit Trendline:")
        o_layout.addRow(self.lbl_line, self.cmb_line)

        # Binscatter Controls
        self.txt_controls = QtWidgets.QLineEdit()
        self.txt_controls.setPlaceholderText("Optional control variables (e.g. w1 w2)")
        self.txt_controls.textChanged.connect(self.update_syntax)
        self.lbl_controls = QtWidgets.QLabel("Controls (FWL):")
        o_layout.addRow(self.lbl_controls, self.txt_controls)

        # Export file option
        save_layout = QtWidgets.QHBoxLayout()
        self.txt_save = QtWidgets.QLineEdit()
        self.txt_save.setPlaceholderText("Optional export path (.png, .pdf, .svg)")
        self.txt_save.textChanged.connect(self.update_syntax)
        btn_browse_save = QtWidgets.QPushButton("Browse...")
        btn_browse_save.clicked.connect(self._browse_save)
        save_layout.addWidget(self.txt_save)
        save_layout.addWidget(btn_browse_save)
        self.lbl_save = QtWidgets.QLabel("Save Figure To:")
        o_layout.addRow(self.lbl_save, save_layout)

        layout.addWidget(self.opt_box)

        # 4. Observation Filter (if / in)
        self.setup_observation_filter(layout)

        # 5. Live Syntax Footer
        self.setup_footer(layout)
        self.btn_submit.setText("📈 Generate Plot")

        self._on_type_changed()
        self.update_syntax()

    def _browse_save(self):
        filt = "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg);;All Files (*.*)"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Plot Figure As", "", filt)
        if path:
            self.txt_save.setText(path)

    def _on_type_changed(self):
        t = self.cmb_type.currentText().lower()
        is_binscatter = "binscatter" in t
        is_rd = "rdplot" in t
        is_irf = "irf" in t
        is_hist = "histogram" in t
        is_box = "box" in t
        is_standalone = any(k in t for k in ["heatmap", "coefplot", "synth_plot", "eventstudy"])

        # Variable inputs visibility
        self.var_box.setVisible(not is_standalone)
        if not is_standalone:
            self.lbl_y.setVisible(not is_hist)
            self.cmb_y.setVisible(not is_hist)

            self.lbl_x.setText("Distribution Variable:" if is_hist else ("Shock Variable:" if is_irf else "X Variable (Horizontal):"))
            self.lbl_x.setVisible(not is_box)
            self.cmb_x.setVisible(not is_box)

            self.lbl_by.setText("Over Grouping:" if is_box else "Group / By Variable:")
            self.lbl_by.setVisible(is_binscatter or is_box)
            self.cmb_by.setVisible(is_binscatter or is_box)

        # Options visibility
        self.lbl_bins.setVisible(is_binscatter or is_rd or is_hist)
        self.spin_bins.setVisible(is_binscatter or is_rd or is_hist)

        self.lbl_cutoff.setVisible(is_rd)
        self.spin_cutoff.setVisible(is_rd)

        self.lbl_bw.setVisible(is_rd)
        self.spin_bw.setVisible(is_rd)

        self.lbl_horizon.setVisible(is_irf)
        self.spin_horizon.setVisible(is_irf)

        self.lbl_line.setVisible(is_binscatter)
        self.cmb_line.setVisible(is_binscatter)

        self.lbl_controls.setVisible(is_binscatter)
        self.txt_controls.setVisible(is_binscatter)

        self.update_syntax()

    def update_syntax(self):
        t = self.cmb_type.currentText().lower()
        y = self.cmb_y.currentText().strip()
        x = self.cmb_x.currentText().strip()
        by = self.cmb_by.currentText().strip()
        save_path = self.txt_save.text().strip()
        save_opt = f'save("{save_path}")' if save_path else None
        if_in_str = self.get_if_in_syntax()

        cmd = ""
        if "scatter" in t and "binscatter" not in t:
            opts = [save_opt] if save_opt else []
            opts_str = f", {', '.join(opts)}" if opts else ""
            cmd = f"scatter {y} {x}{if_in_str}{opts_str}" if y and x else "scatter"
        elif "line" in t:
            opts = [save_opt] if save_opt else []
            opts_str = f", {', '.join(opts)}" if opts else ""
            cmd = f"line {y} {x}{if_in_str}{opts_str}" if y and x else "line"
        elif "binscatter" in t:
            opts = []
            if by and by != "(None)":
                opts.append(f"by({by})")
            opts.append(f"nbins({self.spin_bins.value()})")
            line_val = self.cmb_line.currentText()
            if line_val and line_val != "linear":
                opts.append(f"line({line_val})")
            ctrls = self.txt_controls.text().strip()
            if ctrls:
                opts.append(f"controls({ctrls})")
            if save_opt:
                opts.append(save_opt)
            opts_str = f", {', '.join(opts)}" if opts else ""
            cmd = f"binscatter {y} {x}{if_in_str}{opts_str}" if y and x else "binscatter"
        elif "rdplot" in t:
            opts = []
            opts.append(f"c({self.spin_cutoff.value()})")
            if self.spin_bw.value() > 0:
                opts.append(f"bw({self.spin_bw.value()})")
            if self.spin_bins.value() != 20:
                opts.append(f"nbins({self.spin_bins.value()})")
            if save_opt:
                opts.append(save_opt)
            opts_str = f", {', '.join(opts)}" if opts else ""
            cmd = f"rdplot {y} {x}{if_in_str}{opts_str}" if y and x else "rdplot"
        elif "irf" in t:
            opts = []
            opts.append(f"horizon({self.spin_horizon.value()})")
            if save_opt:
                opts.append(save_opt)
            opts_str = f", {', '.join(opts)}" if opts else ""
            cmd = f"irf {y} {x}{if_in_str}{opts_str}" if y and x else "irf"
        elif "histogram" in t:
            opts = []
            opts.append(f"bins({self.spin_bins.value()})")
            if save_opt:
                opts.append(save_opt)
            opts_str = f", {', '.join(opts)}" if opts else ""
            cmd = f"histogram {x}{if_in_str}{opts_str}" if x else "histogram"
        elif "boxplot" in t:
            opts = []
            if by and by != "(None)":
                opts.append(f"over({by})")
            if save_opt:
                opts.append(save_opt)
            opts_str = f", {', '.join(opts)}" if opts else ""
            cmd = f"boxplot {y}{if_in_str}{opts_str}" if y else "boxplot"
        elif "heatmap" in t:
            opts = [save_opt] if save_opt else []
            opts_str = f", {', '.join(opts)}" if opts else ""
            cmd = f"heatmap{if_in_str}{opts_str}"
        elif "coefplot" in t:
            opts = [save_opt] if save_opt else []
            opts_str = f", {', '.join(opts)}" if opts else ""
            cmd = f"coefplot{if_in_str}{opts_str}"
        elif "synth_plot" in t:
            opts = [save_opt] if save_opt else []
            opts_str = f", {', '.join(opts)}" if opts else ""
            cmd = f"synth_plot{if_in_str}{opts_str}"
        elif "eventstudy" in t:
            opts = [save_opt] if save_opt else []
            opts_str = f", {', '.join(opts)}" if opts else ""
            cmd = f"eventstudy{if_in_str}{opts_str}"

        self.txt_syntax.setText(cmd)


class CountRegressionDialog(BaseBridgeDialog):
    """Count Outcome Models dialog (poisson / nbreg)."""

    def __init__(self, dataset, parent=None, initial_model="poisson"):
        self.initial_model = (initial_model or "poisson").lower()
        title = "Count Outcome Models (poisson / nbreg)"
        super().__init__(title, parent)
        self.dataset = dataset
        self.setMinimumWidth(560)
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []
        dep_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.dataset.df[c])]

        # 0. Model Link Selector
        m_box = QtWidgets.QHBoxLayout()
        m_box.addWidget(QtWidgets.QLabel("Count Model Type:"))
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(["Poisson Regression (poisson)", "Negative Binomial (nbreg)"])
        if "nbreg" in self.initial_model:
            self.cmb_model.setCurrentIndex(1)
        else:
            self.cmb_model.setCurrentIndex(0)
        self.cmb_model.currentIndexChanged.connect(self.update_syntax)
        m_box.addWidget(self.cmb_model, stretch=1)
        layout.addLayout(m_box)

        # Guidance Box
        guide_box = QtWidgets.QFrame()
        guide_box.setObjectName("guideBox")
        guide_box.setStyleSheet("QFrame#guideBox { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }")
        guide_layout = QtWidgets.QVBoxLayout(guide_box)
        guide_layout.setContentsMargins(10, 8, 10, 8)
        lbl_g = QtWidgets.QLabel("📘 <b>Count Outcomes:</b> Used when the dependent variable represents event counts (Y ∈ {0, 1, 2, ...}). "
                                 "Poisson assumes Var(Y)=E(Y); Negative Binomial accommodates overdispersion.")
        lbl_g.setStyleSheet("font-size: 9.5pt;")
        lbl_g.setWordWrap(True)
        guide_layout.addWidget(lbl_g)
        layout.addWidget(guide_box)

        # 1. Dependent Variable
        dep_box = QtWidgets.QGroupBox("1. Dependent Variable (Count Outcome Y ≥ 0)")
        dep_layout = QtWidgets.QVBoxLayout(dep_box)
        self.cmb_dep = QtWidgets.QComboBox()
        self.cmb_dep.addItems(dep_cols)
        self.cmb_dep.currentIndexChanged.connect(self.update_syntax)
        dep_layout.addWidget(self.cmb_dep)
        layout.addWidget(dep_box)

        # 2. Independent Variables
        indep_box = QtWidgets.QGroupBox("2. Independent Variables (Predictors X — Multi-select)")
        indep_layout = QtWidgets.QVBoxLayout(indep_box)
        self.lst_indeps = QtWidgets.QListWidget()
        self.lst_indeps.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_indeps.addItems(cols)
        self.lst_indeps.itemSelectionChanged.connect(self.update_syntax)
        self.lst_indeps.setMaximumHeight(100)
        indep_layout.addWidget(self.lst_indeps)
        layout.addWidget(indep_box)

        # 3. Covariance options
        opt_box = QtWidgets.QGroupBox("3. Model & Covariance Options")
        opt_layout = QtWidgets.QFormLayout(opt_box)

        self.cmb_se = QtWidgets.QComboBox()
        self.cmb_se.addItems(["Standard MLE", "Robust (HC1)", "Clustered"])
        self.cmb_se.currentIndexChanged.connect(self.update_syntax)
        opt_layout.addRow("Std. Errors (VCE):", self.cmb_se)

        self.cmb_cluster = QtWidgets.QComboBox()
        self.cmb_cluster.addItems(["(None)"] + cols)
        self.cmb_cluster.currentIndexChanged.connect(self.update_syntax)
        opt_layout.addRow("Cluster Variable:", self.cmb_cluster)

        layout.addWidget(opt_box)

        # 4. Observation Filter (if / in)
        self.setup_observation_filter(layout)

        # 5. Live Equation Preview
        eq_box = QtWidgets.QGroupBox("Model Specification (Live Equation Preview)")
        eq_layout = QtWidgets.QVBoxLayout(eq_box)
        self.lbl_equation = QtWidgets.QLabel()
        self.lbl_equation.setStyleSheet(
            "background-color: #1E1E1E; color: #E5C07B; font-family: 'Consolas', monospace; "
            "font-size: 10pt; font-weight: bold; padding: 6px; border: 1px solid #3E4451; border-radius: 4px;"
        )
        self.lbl_equation.setWordWrap(True)
        eq_layout.addWidget(self.lbl_equation)
        layout.addWidget(eq_box)

        self.setup_footer(layout)
        self.update_syntax()

    def update_syntax(self):
        dep = self.cmb_dep.currentText()
        is_nbreg = "nbreg" in self.cmb_model.currentText().lower()
        cmd_name = "nbreg" if is_nbreg else "poisson"

        indeps = [item.text() for item in self.lst_indeps.selectedItems() if item.text() != dep]
        indeps_str = " ".join(indeps)

        se_idx = self.cmb_se.currentIndex()
        self.cmb_cluster.setEnabled(se_idx == 2)
        if not dep:
            self._set_model_preview(missing="Select a count outcome variable.")
            return
        if se_idx == 2 and self.cmb_cluster.currentIndex() < 1:
            self._set_model_preview(missing="Select a grouping variable for clustered standard errors.")
            return

        subsample_note = ""
        if hasattr(self, 'txt_if') and self.txt_if.text().strip():
            subsample_note = f"  [where {self.txt_if.text().strip()}]"
        elif hasattr(self, 'txt_in') and self.txt_in.text().strip():
            subsample_note = f"  [in {self.txt_in.text().strip()}]"

        if indeps:
            rhs_terms = " + ".join([f"β{idx+1}·{v}" for idx, v in enumerate(indeps)])
            self.lbl_equation.setText(f"E({dep} | X) = exp(β₀ + {rhs_terms}){subsample_note}")
        else:
            self.lbl_equation.setText(f"E({dep} | X) = exp(β₀)  (Baseline count rate){subsample_note}")

        opts = []
        if se_idx == 1:
            opts.append("robust")
        elif se_idx == 2 and self.cmb_cluster.currentIndex() > 0:
            opts.append(f"vce(cluster {self.cmb_cluster.currentText()})")

        opts_str = f", {' '.join(opts)}" if opts else ""
        if_in_str = self.get_if_in_syntax()
        cmd = f"{cmd_name} {dep} {indeps_str}{if_in_str}{opts_str}"
        self._set_model_preview(cmd)


class IvregressDialog(BaseBridgeDialog):
    """Instrumental Variables Regression dialog (ivregress 2sls)."""

    def __init__(self, dataset, parent=None):
        super().__init__("Instrumental Variables / 2SLS (ivregress)", parent)
        self.dataset = dataset
        self.setMinimumWidth(580)
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []
        dep_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.dataset.df[c])]

        # Guidance Card
        guide_box = QtWidgets.QFrame()
        guide_box.setObjectName("guideBox")
        guide_box.setStyleSheet("QFrame#guideBox { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }")
        guide_layout = QtWidgets.QVBoxLayout(guide_box)
        guide_layout.setContentsMargins(10, 8, 10, 8)
        lbl_g = QtWidgets.QLabel(
            "📘 <b>IV / 2SLS Regression:</b> Estimates causal effects when regressors are endogenous (correlated with error term). "
            "Requires excluded instruments (Z) that predict endogenous regressor (Y₂) but have no direct effect on Y."
        )
        lbl_g.setStyleSheet("font-size: 9.5pt;")
        lbl_g.setWordWrap(True)
        guide_layout.addWidget(lbl_g)
        layout.addWidget(guide_box)

        # 1. Outcome
        dep_box = QtWidgets.QGroupBox("1. Dependent Variable (Outcome Y)")
        dep_l = QtWidgets.QVBoxLayout(dep_box)
        self.cmb_dep = QtWidgets.QComboBox()
        self.cmb_dep.addItems(dep_cols)
        self.cmb_dep.currentIndexChanged.connect(self.update_syntax)
        dep_l.addWidget(self.cmb_dep)
        layout.addWidget(dep_box)

        # 2. Variable Lists in Grid
        grid_v = QtWidgets.QGridLayout()

        # Exogenous X1
        box_x1 = QtWidgets.QGroupBox("2. Exogenous Covariates (X₁)")
        l_x1 = QtWidgets.QVBoxLayout(box_x1)
        self.lst_exog = QtWidgets.QListWidget()
        self.lst_exog.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_exog.addItems(cols)
        self.lst_exog.itemSelectionChanged.connect(self.update_syntax)
        self.lst_exog.setMaximumHeight(85)
        l_x1.addWidget(self.lst_exog)
        grid_v.addWidget(box_x1, 0, 0)

        # Endogenous Y2
        box_y2 = QtWidgets.QGroupBox("3. Endogenous Regressors (Y₂)")
        l_y2 = QtWidgets.QVBoxLayout(box_y2)
        self.lst_endog = QtWidgets.QListWidget()
        self.lst_endog.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_endog.addItems(cols)
        self.lst_endog.itemSelectionChanged.connect(self.update_syntax)
        self.lst_endog.setMaximumHeight(85)
        l_y2.addWidget(self.lst_endog)
        grid_v.addWidget(box_y2, 0, 1)

        # Excluded Instruments Z
        box_z = QtWidgets.QGroupBox("4. Excluded Instruments (Z)")
        l_z = QtWidgets.QVBoxLayout(box_z)
        self.lst_inst = QtWidgets.QListWidget()
        self.lst_inst.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_inst.addItems(cols)
        self.lst_inst.itemSelectionChanged.connect(self.update_syntax)
        self.lst_inst.setMaximumHeight(85)
        l_z.addWidget(self.lst_inst)
        grid_v.addWidget(box_z, 1, 0, 1, 2)

        layout.addLayout(grid_v)

        # 3. Options
        opt_box = QtWidgets.QGroupBox("5. Estimation & Variance Options")
        opt_l = QtWidgets.QFormLayout(opt_box)

        self.cmb_est = QtWidgets.QComboBox()
        self.cmb_est.addItems(["2sls (Two-Stage Least Squares)", "liml (Limited Information ML)", "gmm (Generalized Method of Moments)"])
        self.cmb_est.currentIndexChanged.connect(self.update_syntax)
        opt_l.addRow("Estimator:", self.cmb_est)

        self.cmb_se = QtWidgets.QComboBox()
        self.cmb_se.addItems(["Classical", "Robust (HC1)", "Clustered"])
        self.cmb_se.currentIndexChanged.connect(self.update_syntax)
        opt_l.addRow("Std. Errors:", self.cmb_se)

        self.cmb_cluster = QtWidgets.QComboBox()
        self.cmb_cluster.addItems(["(None)"] + cols)
        self.cmb_cluster.currentIndexChanged.connect(self.update_syntax)
        opt_l.addRow("Cluster Var:", self.cmb_cluster)

        layout.addWidget(opt_box)

        # 4. Observation Filter (if / in)
        self.setup_observation_filter(layout)

        # 5. Live Equation Preview
        eq_box = QtWidgets.QGroupBox("Model Specification (Live Equation Preview)")
        eq_l = QtWidgets.QVBoxLayout(eq_box)
        self.lbl_equation = QtWidgets.QLabel()
        self.lbl_equation.setStyleSheet(
            "background-color: #1E1E1E; color: #E5C07B; font-family: 'Consolas', monospace; "
            "font-size: 10pt; font-weight: bold; padding: 6px; border: 1px solid #3E4451; border-radius: 4px;"
        )
        self.lbl_equation.setWordWrap(True)
        eq_l.addWidget(self.lbl_equation)
        layout.addWidget(eq_box)

        self.setup_footer(layout)
        self.update_syntax()

    def update_syntax(self):
        dep = self.cmb_dep.currentText()
        exogs = [item.text() for item in self.lst_exog.selectedItems() if item.text() != dep]
        endogs = [item.text() for item in self.lst_endog.selectedItems() if item.text() != dep]
        insts = [item.text() for item in self.lst_inst.selectedItems() if item.text() != dep]

        se_idx = self.cmb_se.currentIndex()
        self.cmb_cluster.setEnabled(se_idx == 2)

        if not dep:
            self._set_model_preview(missing="Select a dependent outcome variable.")
            return

        est_raw = self.cmb_est.currentText().split()[0].lower()

        subsample_note = ""
        if hasattr(self, 'txt_if') and self.txt_if.text().strip():
            subsample_note = f"  [where {self.txt_if.text().strip()}]"
        elif hasattr(self, 'txt_in') and self.txt_in.text().strip():
            subsample_note = f"  [in {self.txt_in.text().strip()}]"

        rhs_parts = []
        if exogs:
            rhs_parts.append(" + ".join([f"β·{v}" for v in exogs]))
        if endogs:
            rhs_parts.append(" + ".join([f"γ·{v}" for v in endogs]))
        rhs_str = ("β₀ + " + " + ".join(rhs_parts)) if rhs_parts else "β₀"
        inst_note = f"  [instrumented by {', '.join(insts)}]" if insts else ""
        self.lbl_equation.setText(f"{dep} = {rhs_str} + ε{inst_note}{subsample_note}")

        opts = []
        if se_idx == 1:
            opts.append("robust")
        elif se_idx == 2 and self.cmb_cluster.currentIndex() > 0:
            opts.append(f"vce(cluster {self.cmb_cluster.currentText()})")

        opts_str = f", {' '.join(opts)}" if opts else ""
        if_in_str = self.get_if_in_syntax()

        exog_str = f" {' '.join(exogs)}" if exogs else ""
        endog_inst_str = f" ({' '.join(endogs)} = {' '.join(insts)})" if endogs and insts else ""
        cmd = f"ivregress {est_raw} {dep}{exog_str}{endog_inst_str}{if_in_str}{opts_str}"
        self._set_model_preview(cmd)


class PanelRegressionDialog(BaseBridgeDialog):
    """Panel Data Regression dialog (xtreg)."""

    def __init__(self, dataset, parent=None, initial_model="fe"):
        self.initial_model = (initial_model or "fe").lower()
        title = "Panel Data Regression (xtreg)"
        super().__init__(title, parent)
        self.dataset = dataset
        self.setMinimumWidth(560)
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []
        dep_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.dataset.df[c])]

        # Guidance Card
        guide_box = QtWidgets.QFrame()
        guide_box.setObjectName("guideBox")
        guide_box.setStyleSheet("QFrame#guideBox { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }")
        guide_layout = QtWidgets.QVBoxLayout(guide_box)
        guide_layout.setContentsMargins(10, 8, 10, 8)
        lbl_g = QtWidgets.QLabel(
            "📘 <b>Panel Regression (xtreg):</b> Models data indexed by cross-sectional unit (i) across time (t). "
            "Fixed Effects (fe) controls for all time-invariant unobserved heterogeneity; Random Effects (re) models unit effects as random shocks."
        )
        lbl_g.setStyleSheet("font-size: 9.5pt;")
        lbl_g.setWordWrap(True)
        guide_layout.addWidget(lbl_g)
        layout.addWidget(guide_box)

        # 1. Outcome
        dep_box = QtWidgets.QGroupBox("1. Dependent Variable (Outcome Y)")
        dep_l = QtWidgets.QVBoxLayout(dep_box)
        self.cmb_dep = QtWidgets.QComboBox()
        self.cmb_dep.addItems(dep_cols)
        self.cmb_dep.currentIndexChanged.connect(self.update_syntax)
        dep_l.addWidget(self.cmb_dep)
        layout.addWidget(dep_box)

        # 2. Independent Variables
        indep_box = QtWidgets.QGroupBox("2. Independent Variables (Covariates X — Multi-select)")
        indep_l = QtWidgets.QVBoxLayout(indep_box)
        self.lst_indeps = QtWidgets.QListWidget()
        self.lst_indeps.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_indeps.addItems(cols)
        self.lst_indeps.itemSelectionChanged.connect(self.update_syntax)
        self.lst_indeps.setMaximumHeight(100)
        indep_l.addWidget(self.lst_indeps)
        layout.addWidget(indep_box)

        # 3. Model & Panel IDs
        opt_box = QtWidgets.QGroupBox("3. Panel Structure & Estimator")
        opt_l = QtWidgets.QFormLayout(opt_box)

        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(["Fixed Effects (fe)", "Random Effects (re)", "Between Effects (be)"])
        if "re" in self.initial_model:
            self.cmb_model.setCurrentIndex(1)
        else:
            self.cmb_model.setCurrentIndex(0)
        self.cmb_model.currentIndexChanged.connect(self.update_syntax)
        opt_l.addRow("Panel Estimator:", self.cmb_model)

        self.cmb_se = QtWidgets.QComboBox()
        self.cmb_se.addItems(["Classical (Homoskedastic)", "Robust (Cluster on Panel ID)"])
        self.cmb_se.currentIndexChanged.connect(self.update_syntax)
        opt_l.addRow("Std. Errors:", self.cmb_se)

        layout.addWidget(opt_box)

        # 4. Observation Filter (if / in)
        self.setup_observation_filter(layout)

        # 5. Live Equation Preview
        eq_box = QtWidgets.QGroupBox("Model Specification (Live Equation Preview)")
        eq_l = QtWidgets.QVBoxLayout(eq_box)
        self.lbl_equation = QtWidgets.QLabel()
        self.lbl_equation.setStyleSheet(
            "background-color: #1E1E1E; color: #E5C07B; font-family: 'Consolas', monospace; "
            "font-size: 10pt; font-weight: bold; padding: 6px; border: 1px solid #3E4451; border-radius: 4px;"
        )
        self.lbl_equation.setWordWrap(True)
        eq_l.addWidget(self.lbl_equation)
        layout.addWidget(eq_box)

        self.setup_footer(layout)
        self.update_syntax()

    def update_syntax(self):
        dep = self.cmb_dep.currentText()
        indeps = [item.text() for item in self.lst_indeps.selectedItems() if item.text() != dep]
        indeps_str = " ".join(indeps)

        if not dep:
            self._set_model_preview(missing="Select a dependent outcome variable.")
            return

        is_re = "re" in self.cmb_model.currentText().lower()
        est_opt = "re" if is_re else "fe"

        subsample_note = ""
        if hasattr(self, 'txt_if') and self.txt_if.text().strip():
            subsample_note = f"  [where {self.txt_if.text().strip()}]"
        elif hasattr(self, 'txt_in') and self.txt_in.text().strip():
            subsample_note = f"  [in {self.txt_in.text().strip()}]"

        if indeps:
            rhs_terms = " + ".join([f"β·{v}ᵢₜ" for v in indeps])
            if is_re:
                self.lbl_equation.setText(f"{dep}ᵢₜ = μ + αᵢ + {rhs_terms} + εᵢₜ{subsample_note}")
            else:
                self.lbl_equation.setText(f"{dep}ᵢₜ = αᵢ + {rhs_terms} + εᵢₜ  (Within transformation){subsample_note}")
        else:
            self.lbl_equation.setText(f"{dep}ᵢₜ = αᵢ + εᵢₜ{subsample_note}")

        opts = [est_opt]
        if self.cmb_se.currentIndex() == 1:
            opts.append("robust")

        opts_str = f", {' '.join(opts)}"
        if_in_str = self.get_if_in_syntax()
        cmd = f"xtreg {dep} {indeps_str}{if_in_str}{opts_str}"
        self._set_model_preview(cmd)


class CorrelateDialog(BaseBridgeDialog):
    """Correlation and Covariance Matrix dialog."""

    def __init__(self, dataset, parent=None):
        super().__init__("Correlation & Covariance Matrix (correlate)", parent)
        self.dataset = dataset
        self.setMinimumWidth(500)
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.dataset.df[c])]

        # Guidance Card
        guide_box = QtWidgets.QFrame()
        guide_box.setObjectName("guideBox")
        guide_box.setStyleSheet("QFrame#guideBox { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }")
        guide_layout = QtWidgets.QVBoxLayout(guide_box)
        guide_layout.setContentsMargins(10, 8, 10, 8)
        lbl_g = QtWidgets.QLabel(
            "📘 <b>Correlation Matrix:</b> Computes pairwise Pearson correlation coefficients (r ∈ [-1, 1]) "
            "across all selected variables. If no variables are selected, all numeric columns in memory are analyzed."
        )
        lbl_g.setStyleSheet("font-size: 9.5pt;")
        lbl_g.setWordWrap(True)
        guide_layout.addWidget(lbl_g)
        layout.addWidget(guide_box)

        # Variables List
        var_box = QtWidgets.QGroupBox("Select Variables (Leave empty to correlate all numeric columns)")
        var_l = QtWidgets.QVBoxLayout(var_box)
        self.lst_vars = QtWidgets.QListWidget()
        self.lst_vars.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_vars.addItems(num_cols if num_cols else cols)
        self.lst_vars.itemSelectionChanged.connect(self.update_syntax)
        self.lst_vars.setMaximumHeight(130)
        var_l.addWidget(self.lst_vars)
        layout.addWidget(var_box)

        # Options
        opt_box = QtWidgets.QGroupBox("Options")
        opt_l = QtWidgets.QVBoxLayout(opt_box)
        self.chk_cov = QtWidgets.QCheckBox("Compute Covariance Matrix instead of Correlation (covariance)")
        self.chk_cov.stateChanged.connect(self.update_syntax)
        opt_l.addWidget(self.chk_cov)
        layout.addWidget(opt_box)

        # Observation Filter (if / in)
        self.setup_observation_filter(layout)

        self.setup_footer(layout)
        self.btn_submit.setText("Compute Correlation")
        self.update_syntax()

    def update_syntax(self):
        selected = [item.text() for item in self.lst_vars.selectedItems()]
        vars_str = f" {' '.join(selected)}" if selected else ""

        opts = []
        if self.chk_cov.isChecked():
            opts.append("covariance")

        opts_str = f", {', '.join(opts)}" if opts else ""
        if_in_str = self.get_if_in_syntax()
        cmd = f"correlate{vars_str}{if_in_str}{opts_str}"
        self.txt_syntax.setText(cmd)


class TimeSeriesDialog(BaseBridgeDialog):
    """Interactive Time Series Studio Dialog (tsset, dfuller, arima, tsfilter hp, var, corrgram)."""

    def __init__(self, dataset, parent=None, initial_task="tsset"):
        self.initial_task = (initial_task or "tsset").lower()
        title = "Time-Series Analysis & Forecasting Studio"
        super().__init__(title, parent)
        self.dataset = dataset
        self.setMinimumWidth(580)
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.dataset.df[c])]

        # 0. Task selector
        t_box = QtWidgets.QHBoxLayout()
        t_box.addWidget(QtWidgets.QLabel("Time-Series Tool / Model:"))
        self.cmb_task = QtWidgets.QComboBox()
        self.cmb_task.addItems([
            "Declare Time Variable (tsset)",
            "Augmented Dickey-Fuller Unit Root Test (dfuller)",
            "ARIMA Modeling & Forecasting (arima)",
            "Hodrick-Prescott Filter (tsfilter hp)",
            "Vector Autoregression (var)",
            "Autocorrelations & Correlogram (corrgram)"
        ])
        for idx in range(self.cmb_task.count()):
            txt = self.cmb_task.itemText(idx).lower()
            if self.initial_task in txt or (self.initial_task == "hp" and "filter" in txt):
                self.cmb_task.setCurrentIndex(idx)
                break
        self.cmb_task.currentIndexChanged.connect(self._on_task_changed)
        t_box.addWidget(self.cmb_task, stretch=1)
        layout.addLayout(t_box)

        # Guidance Box
        self.guide_box = QtWidgets.QFrame()
        self.guide_box.setObjectName("guideBox")
        self.guide_box.setStyleSheet("QFrame#guideBox { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }")
        g_lay = QtWidgets.QVBoxLayout(self.guide_box)
        g_lay.setContentsMargins(10, 8, 10, 8)
        self.lbl_guide = QtWidgets.QLabel()
        self.lbl_guide.setStyleSheet("font-size: 9.5pt;")
        self.lbl_guide.setWordWrap(True)
        g_lay.addWidget(self.lbl_guide)
        layout.addWidget(self.guide_box)

        # Container for task-specific parameter widgets
        self.param_box = QtWidgets.QGroupBox("Model & Parameter Specification")
        self.param_layout = QtWidgets.QFormLayout(self.param_box)
        self.param_layout.setContentsMargins(10, 8, 10, 8)
        self.param_layout.setSpacing(8)

        # Variable Selectors
        self.cmb_var = QtWidgets.QComboBox()
        self.cmb_var.addItems(num_cols if num_cols else cols)
        for candidate in ["date", "year", "period", "time"]:
            for i in range(self.cmb_var.count()):
                if candidate in self.cmb_var.itemText(i).lower():
                    self.cmb_var.setCurrentIndex(i)
                    break
        self.cmb_var.currentIndexChanged.connect(self.update_syntax)
        self.lbl_var = QtWidgets.QLabel("Time Variable:")
        self.param_layout.addRow(self.lbl_var, self.cmb_var)

        # Multi-select list for VAR
        self.lst_var_depvars = QtWidgets.QListWidget()
        self.lst_var_depvars.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_var_depvars.addItems(num_cols if num_cols else cols)
        self.lst_var_depvars.itemSelectionChanged.connect(self.update_syntax)
        self.lst_var_depvars.setMaximumHeight(85)
        self.lbl_var_depvars = QtWidgets.QLabel("Endogenous Series (≥ 2):")
        self.param_layout.addRow(self.lbl_var_depvars, self.lst_var_depvars)

        # Lags spinbox (for dfuller and var)
        self.spin_lags = QtWidgets.QSpinBox()
        self.spin_lags.setRange(0, 50)
        self.spin_lags.setValue(0)
        self.spin_lags.setSpecialValueText("Auto")
        self.spin_lags.valueChanged.connect(self.update_syntax)
        self.lbl_lags = QtWidgets.QLabel("Lags:")
        self.param_layout.addRow(self.lbl_lags, self.spin_lags)

        # ARIMA parameters (p, d, q, steps)
        self.arima_row = QtWidgets.QHBoxLayout()
        self.spin_p = QtWidgets.QSpinBox()
        self.spin_p.setRange(0, 10)
        self.spin_p.setValue(1)
        self.spin_p.valueChanged.connect(self.update_syntax)
        self.spin_d = QtWidgets.QSpinBox()
        self.spin_d.setRange(0, 3)
        self.spin_d.setValue(0)
        self.spin_d.valueChanged.connect(self.update_syntax)
        self.spin_q = QtWidgets.QSpinBox()
        self.spin_q.setRange(0, 10)
        self.spin_q.setValue(1)
        self.spin_q.valueChanged.connect(self.update_syntax)
        self.spin_steps = QtWidgets.QSpinBox()
        self.spin_steps.setRange(1, 100)
        self.spin_steps.setValue(5)
        self.spin_steps.valueChanged.connect(self.update_syntax)

        self.arima_row.addWidget(QtWidgets.QLabel("AR (p):"))
        self.arima_row.addWidget(self.spin_p)
        self.arima_row.addWidget(QtWidgets.QLabel("Diff (d):"))
        self.arima_row.addWidget(self.spin_d)
        self.arima_row.addWidget(QtWidgets.QLabel("MA (q):"))
        self.arima_row.addWidget(self.spin_q)
        self.arima_row.addWidget(QtWidgets.QLabel("Steps:"))
        self.arima_row.addWidget(self.spin_steps)
        self.lbl_arima = QtWidgets.QLabel("ARIMA Order (p, d, q):")
        self.param_layout.addRow(self.lbl_arima, self.arima_row)

        # HP Filter parameters (cycle, trend, lambda)
        self.hp_row = QtWidgets.QHBoxLayout()
        self.txt_cycle = QtWidgets.QLineEdit("cycle")
        self.txt_cycle.textChanged.connect(self.update_syntax)
        self.txt_trend = QtWidgets.QLineEdit("trend")
        self.txt_trend.textChanged.connect(self.update_syntax)
        self.spin_lambda = QtWidgets.QDoubleSpinBox()
        self.spin_lambda.setRange(0.01, 100000.0)
        self.spin_lambda.setValue(1600.0)
        self.spin_lambda.valueChanged.connect(self.update_syntax)

        self.hp_row.addWidget(QtWidgets.QLabel("Cycle:"))
        self.hp_row.addWidget(self.txt_cycle)
        self.hp_row.addWidget(QtWidgets.QLabel("Trend:"))
        self.hp_row.addWidget(self.txt_trend)
        self.hp_row.addWidget(QtWidgets.QLabel("λ:"))
        self.hp_row.addWidget(self.spin_lambda)
        self.lbl_hp = QtWidgets.QLabel("HP Filter Components:")
        self.param_layout.addRow(self.lbl_hp, self.hp_row)

        layout.addWidget(self.param_box)

        # Observation Filter (if / in)
        self.setup_observation_filter(layout)

        # Live Equation Preview
        eq_box = QtWidgets.QGroupBox("Model Specification (Live Equation Preview)")
        eq_l = QtWidgets.QVBoxLayout(eq_box)
        self.lbl_equation = QtWidgets.QLabel()
        self.lbl_equation.setStyleSheet(
            "background-color: #1E1E1E; color: #E5C07B; font-family: 'Consolas', monospace; "
            "font-size: 10pt; font-weight: bold; padding: 6px; border: 1px solid #3E4451; border-radius: 4px;"
        )
        self.lbl_equation.setWordWrap(True)
        eq_l.addWidget(self.lbl_equation)
        layout.addWidget(eq_box)

        self.setup_footer(layout)
        self.btn_submit.setText("📈 Execute Analysis")

        self._on_task_changed()
        self.update_syntax()

    def _on_task_changed(self):
        t = self.cmb_task.currentText().lower()
        is_tsset = "tsset" in t
        is_dfuller = "dfuller" in t
        is_arima = "arima" in t
        is_hp = "hp" in t or "filter" in t
        is_var = "var" in t and "variable" not in t
        is_corrgram = "corrgram" in t

        # Guidance text
        if is_tsset:
            self.lbl_guide.setText("📘 <b>Time-Series Declaration (tsset):</b> Declares the time dimension column (e.g. <i>year</i>, <i>date</i>, <i>month</i>) and sorts data chronologically. Required before time-filtering, lags, or VAR modeling.")
            self.lbl_var.setText("Time Variable (Date / Period / Year):")
        elif is_dfuller:
            self.lbl_guide.setText("📘 <b>Augmented Dickey-Fuller Test (dfuller):</b> Tests the null hypothesis that a unit root is present in an autoregressive time-series model (H₀: series is non-stationary).")
            self.lbl_var.setText("Target Time Series (Y):")
        elif is_arima:
            self.lbl_guide.setText("📘 <b>ARIMA Modeling (arima):</b> Fits an Autoregressive Integrated Moving Average model for univariate time-series forecasting. (p = AR lags, d = differencing, q = MA errors).")
            self.lbl_var.setText("Dependent Series (Y):")
        elif is_hp:
            self.lbl_guide.setText("📘 <b>Hodrick-Prescott Filter (tsfilter hp):</b> Decomposes a macro series into a smooth non-linear trend component and a cyclical component. (λ=1600 for quarterly, λ=6.25 for annual, λ=14400 for monthly).")
            self.lbl_var.setText("Input Series (Y):")
        elif is_var:
            self.lbl_guide.setText("📘 <b>Vector Autoregression (var):</b> Multi-equation econometric model where each variable is modeled as a linear function of past lags of itself and all other variables.")
        elif is_corrgram:
            self.lbl_guide.setText("📘 <b>Correlogram (corrgram):</b> Computes autocorrelations, partial autocorrelations, and Ljung-Box Q statistics across time lags.")
            self.lbl_var.setText("Target Series (Y):")

        # Visibility toggles
        self.lbl_var.setVisible(not is_var)
        self.cmb_var.setVisible(not is_var)

        self.lbl_var_depvars.setVisible(is_var)
        self.lst_var_depvars.setVisible(is_var)

        self.lbl_lags.setVisible(is_dfuller or is_var)
        self.spin_lags.setVisible(is_dfuller or is_var)
        self.spin_lags.setSpecialValueText("Auto" if is_dfuller else "")
        if is_var and self.spin_lags.value() == 0:
            self.spin_lags.setValue(2)

        self.lbl_arima.setVisible(is_arima)
        for i in range(self.arima_row.count()):
            w = self.arima_row.itemAt(i).widget()
            if w:
                w.setVisible(is_arima)

        self.lbl_hp.setVisible(is_hp)
        for i in range(self.hp_row.count()):
            w = self.hp_row.itemAt(i).widget()
            if w:
                w.setVisible(is_hp)

        self.update_syntax()

    def update_syntax(self):
        t = self.cmb_task.currentText().lower()
        var = self.cmb_var.currentText().strip()
        if_in_str = self.get_if_in_syntax()

        cmd = ""
        if "tsset" in t:
            cmd = f"tsset {var}" if var else "tsset"
            self.lbl_equation.setText(f"Chronological ordering indexed by: {var}")
        elif "dfuller" in t:
            lags = self.spin_lags.value()
            opts = [f"lags({lags})"] if lags > 0 else []
            opts_str = f", {' '.join(opts)}" if opts else ""
            cmd = f"dfuller {var}{if_in_str}{opts_str}" if var else "dfuller"
            self.lbl_equation.setText(f"Δ{var}ₜ = α + β·{var}ₜ₋₁ + ∑ γᵢ·Δ{var}ₜ₋ᵢ + εₜ  [H₀: β = 0 (Unit Root)]")
        elif "arima" in t:
            p = self.spin_p.value()
            d = self.spin_d.value()
            q = self.spin_q.value()
            steps = self.spin_steps.value()
            cmd = f"arima {var}{if_in_str}, ar({p}) d({d}) ma({q}) steps({steps})" if var else "arima"
            diff_str = f"Δ^{d}" if d > 0 else ""
            self.lbl_equation.setText(f"{diff_str}{var}ₜ = μ + ∑_{{i=1}}^{{{p}}} φᵢ·{diff_str}{var}ₜ₋ᵢ + ∑_{{j=1}}^{{{q}}} θⱼ·εₜ₋ⱼ + εₜ")
        elif "hp" in t or "filter" in t:
            cycle = self.txt_cycle.text().strip() or "cycle"
            trend = self.txt_trend.text().strip() or "trend"
            lamb = self.spin_lambda.value()
            cmd = f"tsfilter hp {cycle} = {var}, smooth({lamb}) trend({trend})" if var else "tsfilter hp"
            self.lbl_equation.setText(f"min ∑ [ ({var}ₜ - gₜ)² + {lamb}·((gₜ₊₁ - gₜ) - (gₜ - gₜ₋₁))² ]  (Trend: {trend}, Cycle: {cycle})")
        elif "var" in t and "variable" not in t:
            sel_vars = [item.text() for item in self.lst_var_depvars.selectedItems()]
            lags = max(1, self.spin_lags.value())
            vars_str = " ".join(sel_vars)
            cmd = f"var {vars_str}{if_in_str}, lags(1/{lags})" if len(sel_vars) >= 2 else "var"
            self.lbl_equation.setText(f"Yₜ = c + A₁·Yₜ₋₁ + ... + A_{{{lags}}}·Yₜ₋{{{lags}}} + εₜ  [Y = ({vars_str or 'y₁, y₂'})]")
        elif "corrgram" in t:
            cmd = f"corrgram {var}{if_in_str}" if var else "corrgram"
            self.lbl_equation.setText(f"Autocorrelation & Ljung-Box Q: ρₖ = Cov({var}ₜ, {var}ₜ₋ₖ) / Var({var}ₜ)")

        self._set_model_preview(cmd)


class PanelSetupDialog(BaseBridgeDialog):
    """Declare Panel Dataset (xtset) dialog."""

    def __init__(self, dataset, parent=None):
        super().__init__("Declare Panel Dataset (xtset)", parent)
        self.dataset = dataset
        self.setMinimumWidth(480)
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []

        guide_box = QtWidgets.QFrame()
        guide_box.setObjectName("guideBox")
        guide_box.setStyleSheet("QFrame#guideBox { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }")
        g_l = QtWidgets.QVBoxLayout(guide_box)
        lbl_g = QtWidgets.QLabel(
            "📘 <b>Panel Dataset Declaration (xtset):</b> Informs the system of the cross-sectional unit entity "
            "and time dimension for longitudinal models (xtreg, csdid)."
        )
        lbl_g.setStyleSheet("font-size: 9.5pt;")
        lbl_g.setWordWrap(True)
        g_l.addWidget(lbl_g)
        layout.addWidget(guide_box)

        f_layout = QtWidgets.QFormLayout()
        self.cmb_panelvar = QtWidgets.QComboBox()
        self.cmb_panelvar.addItems(cols)
        self.cmb_panelvar.currentIndexChanged.connect(self.update_syntax)
        f_layout.addRow("Entity / Panel ID (panelvar):", self.cmb_panelvar)

        self.cmb_timevar = QtWidgets.QComboBox()
        self.cmb_timevar.addItems(["(None - purely cross-sectional)"] + cols)
        self.cmb_timevar.currentIndexChanged.connect(self.update_syntax)
        f_layout.addRow("Time Variable (timevar):", self.cmb_timevar)
        layout.addLayout(f_layout)

        self.setup_footer(layout)
        self.btn_submit.setText("Declare Panel Structure")
        self.update_syntax()

    def update_syntax(self):
        pvar = self.cmb_panelvar.currentText()
        tvar = self.cmb_timevar.currentText()
        if tvar and tvar != "(None - purely cross-sectional)":
            cmd = f"xtset {pvar} {tvar}"
        else:
            cmd = f"xtset {pvar}"
        self.txt_syntax.setText(cmd)


class CensoredAndQuantileDialog(BaseBridgeDialog):
    """Censored (Tobit) and Quantile Regression dialog."""

    def __init__(self, dataset, parent=None, initial_model="tobit"):
        self.initial_model = (initial_model or "tobit").lower()
        title = "Censored & Quantile Regression Studio (tobit / qreg)"
        super().__init__(title, parent)
        self.dataset = dataset
        self.setMinimumWidth(560)
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.dataset.df[c])]

        # Model link
        m_box = QtWidgets.QHBoxLayout()
        m_box.addWidget(QtWidgets.QLabel("Model Type:"))
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(["Censored Regression / Tobit (tobit)", "Quantile Regression (qreg)"])
        if "qreg" in self.initial_model or "quantile" in self.initial_model:
            self.cmb_model.setCurrentIndex(1)
        else:
            self.cmb_model.setCurrentIndex(0)
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        m_box.addWidget(self.cmb_model, stretch=1)
        layout.addLayout(m_box)

        # Guidance Card
        self.guide_box = QtWidgets.QFrame()
        self.guide_box.setObjectName("guideBox")
        self.guide_box.setStyleSheet("QFrame#guideBox { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }")
        g_lay = QtWidgets.QVBoxLayout(self.guide_box)
        g_lay.setContentsMargins(10, 8, 10, 8)
        self.lbl_guide = QtWidgets.QLabel()
        self.lbl_guide.setStyleSheet("font-size: 9.5pt;")
        self.lbl_guide.setWordWrap(True)
        g_lay.addWidget(self.lbl_guide)
        layout.addWidget(self.guide_box)

        # 1. Dependent Variable
        dep_box = QtWidgets.QGroupBox("1. Dependent Variable (Outcome Y)")
        dep_l = QtWidgets.QVBoxLayout(dep_box)
        self.cmb_dep = QtWidgets.QComboBox()
        self.cmb_dep.addItems(num_cols if num_cols else cols)
        self.cmb_dep.currentIndexChanged.connect(self.update_syntax)
        dep_l.addWidget(self.cmb_dep)
        layout.addWidget(dep_box)

        # 2. Independent Variables
        indep_box = QtWidgets.QGroupBox("2. Independent Variables (Predictors X — Multi-select)")
        indep_l = QtWidgets.QVBoxLayout(indep_box)
        self.lst_indeps = QtWidgets.QListWidget()
        self.lst_indeps.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_indeps.addItems(cols)
        self.lst_indeps.itemSelectionChanged.connect(self.update_syntax)
        self.lst_indeps.setMaximumHeight(100)
        indep_l.addWidget(self.lst_indeps)
        layout.addWidget(indep_box)

        # 3. Model Specific Options
        self.opt_box = QtWidgets.QGroupBox("3. Model Parameters & Limits")
        self.opt_layout = QtWidgets.QFormLayout(self.opt_box)

        # Tobit Options
        self.chk_ll = QtWidgets.QCheckBox("Left-censoring limit (ll):")
        self.chk_ll.setChecked(True)
        self.chk_ll.stateChanged.connect(self.update_syntax)
        self.spin_ll = QtWidgets.QDoubleSpinBox()
        self.spin_ll.setRange(-1e9, 1e9)
        self.spin_ll.setValue(0.0)
        self.spin_ll.valueChanged.connect(self.update_syntax)
        ll_row = QtWidgets.QHBoxLayout()
        ll_row.addWidget(self.chk_ll)
        ll_row.addWidget(self.spin_ll)
        self.opt_layout.addRow(ll_row)

        self.chk_ul = QtWidgets.QCheckBox("Right-censoring limit (ul):")
        self.chk_ul.setChecked(False)
        self.chk_ul.stateChanged.connect(self.update_syntax)
        self.spin_ul = QtWidgets.QDoubleSpinBox()
        self.spin_ul.setRange(-1e9, 1e9)
        self.spin_ul.setValue(100.0)
        self.spin_ul.valueChanged.connect(self.update_syntax)
        ul_row = QtWidgets.QHBoxLayout()
        ul_row.addWidget(self.chk_ul)
        ul_row.addWidget(self.spin_ul)
        self.opt_layout.addRow(ul_row)

        # Qreg Options
        self.spin_quantile = QtWidgets.QDoubleSpinBox()
        self.spin_quantile.setRange(0.01, 0.99)
        self.spin_quantile.setSingleStep(0.05)
        self.spin_quantile.setValue(0.50)
        self.spin_quantile.valueChanged.connect(self.update_syntax)
        self.lbl_quantile = QtWidgets.QLabel("Quantile (tau ∈ (0, 1)):")
        self.opt_layout.addRow(self.lbl_quantile, self.spin_quantile)

        layout.addWidget(self.opt_box)

        # 4. Observation Filter (if / in)
        self.setup_observation_filter(layout)

        # 5. Live Equation Preview
        eq_box = QtWidgets.QGroupBox("Model Specification (Live Equation Preview)")
        eq_l = QtWidgets.QVBoxLayout(eq_box)
        self.lbl_equation = QtWidgets.QLabel()
        self.lbl_equation.setStyleSheet(
            "background-color: #1E1E1E; color: #E5C07B; font-family: 'Consolas', monospace; "
            "font-size: 10pt; font-weight: bold; padding: 6px; border: 1px solid #3E4451; border-radius: 4px;"
        )
        self.lbl_equation.setWordWrap(True)
        eq_l.addWidget(self.lbl_equation)
        layout.addWidget(eq_box)

        self.setup_footer(layout)
        self.btn_submit.setText("📈 Fit Model")

        self._on_model_changed()
        self.update_syntax()

    def _on_model_changed(self):
        is_qreg = "qreg" in self.cmb_model.currentText().lower()
        if is_qreg:
            self.lbl_guide.setText("📘 <b>Quantile Regression (qreg):</b> Estimates conditional quantile functions (e.g. median regression τ=0.50, 25th percentile τ=0.25, 90th percentile τ=0.90), robust to heavy tails and outliers.")
            self.lbl_quantile.setVisible(True)
            self.spin_quantile.setVisible(True)
            self.chk_ll.setVisible(False)
            self.spin_ll.setVisible(False)
            self.chk_ul.setVisible(False)
            self.spin_ul.setVisible(False)
        else:
            self.lbl_guide.setText("📘 <b>Tobit Censored Regression:</b> Estimates linear models where the dependent variable is top-coded, bottom-coded, or bounded (e.g., non-negative expenditure with zero cluster).")
            self.lbl_quantile.setVisible(False)
            self.spin_quantile.setVisible(False)
            self.chk_ll.setVisible(True)
            self.spin_ll.setVisible(True)
            self.chk_ul.setVisible(True)
            self.spin_ul.setVisible(True)
        self.update_syntax()

    def update_syntax(self):
        dep = self.cmb_dep.currentText()
        is_qreg = "qreg" in self.cmb_model.currentText().lower()
        cmd_name = "qreg" if is_qreg else "tobit"

        indeps = [item.text() for item in self.lst_indeps.selectedItems() if item.text() != dep]
        indeps_str = " ".join(indeps)

        if not dep:
            self._set_model_preview(missing="Select a dependent outcome variable.")
            return

        subsample_note = ""
        if hasattr(self, 'txt_if') and self.txt_if.text().strip():
            subsample_note = f"  [where {self.txt_if.text().strip()}]"
        elif hasattr(self, 'txt_in') and self.txt_in.text().strip():
            subsample_note = f"  [in {self.txt_in.text().strip()}]"

        opts = []
        if is_qreg:
            q_val = self.spin_quantile.value()
            if q_val != 0.5:
                opts.append(f"q({q_val:.2f})")
            rhs = " + ".join([f"β_{{τ}}·{v}" for v in indeps]) if indeps else "β_{{0,τ}}"
            self.lbl_equation.setText(f"Q_{{{q_val:.2f}}}({dep} | X) = {rhs}{subsample_note}")
        else:
            if self.chk_ll.isChecked():
                opts.append(f"ll({self.spin_ll.value()})")
            if self.chk_ul.isChecked():
                opts.append(f"ul({self.spin_ul.value()})")
            rhs = " + ".join([f"β·{v}" for v in indeps]) if indeps else "β₀"
            lim_note = f"  [Censored at {self.spin_ll.value()}]" if self.chk_ll.isChecked() else ""
            self.lbl_equation.setText(f"{dep}* = {rhs} + ε,  {dep} = max(ll, {dep}*){lim_note}{subsample_note}")

        opts_str = f", {', '.join(opts)}" if opts else ""
        if_in_str = self.get_if_in_syntax()
        cmd = f"{cmd_name} {dep} {indeps_str}{if_in_str}{opts_str}"
        self._set_model_preview(cmd)


class OrdinalMultinomialDialog(BaseBridgeDialog):
    """Ordered (ologit) and Multinomial (mlogit) Logistic Regression dialog."""

    def __init__(self, dataset, parent=None, initial_model="ologit"):
        self.initial_model = (initial_model or "ologit").lower()
        title = "Ordinal & Multinomial Discrete Choice (ologit / mlogit)"
        super().__init__(title, parent)
        self.dataset = dataset
        self.setMinimumWidth(560)
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []

        # Model type
        m_box = QtWidgets.QHBoxLayout()
        m_box.addWidget(QtWidgets.QLabel("Discrete Choice Model:"))
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(["Ordered Logistic Regression (ologit)", "Multinomial Logistic Regression (mlogit)"])
        if "mlogit" in self.initial_model or "multi" in self.initial_model:
            self.cmb_model.setCurrentIndex(1)
        else:
            self.cmb_model.setCurrentIndex(0)
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        m_box.addWidget(self.cmb_model, stretch=1)
        layout.addLayout(m_box)

        # Guidance Card
        self.guide_box = QtWidgets.QFrame()
        self.guide_box.setObjectName("guideBox")
        self.guide_box.setStyleSheet("QFrame#guideBox { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }")
        g_lay = QtWidgets.QVBoxLayout(self.guide_box)
        g_lay.setContentsMargins(10, 8, 10, 8)
        self.lbl_guide = QtWidgets.QLabel()
        self.lbl_guide.setStyleSheet("font-size: 9.5pt;")
        self.lbl_guide.setWordWrap(True)
        g_lay.addWidget(self.lbl_guide)
        layout.addWidget(self.guide_box)

        # 1. Dependent Variable
        dep_box = QtWidgets.QGroupBox("1. Categorical / Ordered Outcome (Y)")
        dep_l = QtWidgets.QVBoxLayout(dep_box)
        self.cmb_dep = QtWidgets.QComboBox()
        self.cmb_dep.addItems(cols)
        self.cmb_dep.currentIndexChanged.connect(self.update_syntax)
        dep_l.addWidget(self.cmb_dep)
        layout.addWidget(dep_box)

        # 2. Independent Variables
        indep_box = QtWidgets.QGroupBox("2. Independent Variables (Predictors X — Multi-select)")
        indep_l = QtWidgets.QVBoxLayout(indep_box)
        self.lst_indeps = QtWidgets.QListWidget()
        self.lst_indeps.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_indeps.addItems(cols)
        self.lst_indeps.itemSelectionChanged.connect(self.update_syntax)
        self.lst_indeps.setMaximumHeight(100)
        indep_l.addWidget(self.lst_indeps)
        layout.addWidget(indep_box)

        # 3. Variance Options
        opt_box = QtWidgets.QGroupBox("3. Standard Errors")
        opt_l = QtWidgets.QFormLayout(opt_box)
        self.cmb_se = QtWidgets.QComboBox()
        self.cmb_se.addItems(["Standard MLE", "Robust (HC1)", "Clustered"])
        self.cmb_se.currentIndexChanged.connect(self.update_syntax)
        opt_l.addRow("Std. Errors:", self.cmb_se)

        self.cmb_cluster = QtWidgets.QComboBox()
        self.cmb_cluster.addItems(["(None)"] + cols)
        self.cmb_cluster.currentIndexChanged.connect(self.update_syntax)
        opt_l.addRow("Cluster Var:", self.cmb_cluster)
        layout.addWidget(opt_box)

        # 4. Observation Filter (if / in)
        self.setup_observation_filter(layout)

        # 5. Live Equation Preview
        eq_box = QtWidgets.QGroupBox("Model Specification (Live Equation Preview)")
        eq_l = QtWidgets.QVBoxLayout(eq_box)
        self.lbl_equation = QtWidgets.QLabel()
        self.lbl_equation.setStyleSheet(
            "background-color: #1E1E1E; color: #E5C07B; font-family: 'Consolas', monospace; "
            "font-size: 10pt; font-weight: bold; padding: 6px; border: 1px solid #3E4451; border-radius: 4px;"
        )
        self.lbl_equation.setWordWrap(True)
        eq_l.addWidget(self.lbl_equation)
        layout.addWidget(eq_box)

        self.setup_footer(layout)
        self.btn_submit.setText("📈 Fit Discrete Model")

        self._on_model_changed()
        self.update_syntax()

    def _on_model_changed(self):
        is_mlogit = "mlogit" in self.cmb_model.currentText().lower()
        if is_mlogit:
            self.lbl_guide.setText("📘 <b>Multinomial Logit (mlogit):</b> Used when outcome Y has multiple unordered discrete categories (e.g., choice of transport: Car, Bus, Train). Fits relative-risk odds ratios.")
        else:
            self.lbl_guide.setText("📘 <b>Ordered Logit (ologit):</b> Used when outcome Y has naturally ordered ranks (e.g., Likert scale: Poor, Good, Excellent). Estimates threshold cutpoints under proportional odds.")
        self.update_syntax()

    def update_syntax(self):
        dep = self.cmb_dep.currentText()
        is_mlogit = "mlogit" in self.cmb_model.currentText().lower()
        cmd_name = "mlogit" if is_mlogit else "ologit"

        indeps = [item.text() for item in self.lst_indeps.selectedItems() if item.text() != dep]
        indeps_str = " ".join(indeps)

        se_idx = self.cmb_se.currentIndex()
        self.cmb_cluster.setEnabled(se_idx == 2)
        if not dep:
            self._set_model_preview(missing="Select a categorical outcome variable.")
            return

        subsample_note = ""
        if hasattr(self, 'txt_if') and self.txt_if.text().strip():
            subsample_note = f"  [where {self.txt_if.text().strip()}]"
        elif hasattr(self, 'txt_in') and self.txt_in.text().strip():
            subsample_note = f"  [in {self.txt_in.text().strip()}]"

        if is_mlogit:
            self.lbl_equation.setText(f"Pr({dep} = j | X) = exp(X·βⱼ) / ∑ exp(X·βₖ){subsample_note}")
        else:
            self.lbl_equation.setText(f"Pr({dep} ≤ j | X) = Λ(κⱼ - X·β)  [Ordered Proportional Odds]{subsample_note}")

        opts = []
        if se_idx == 1:
            opts.append("robust")
        elif se_idx == 2 and self.cmb_cluster.currentIndex() > 0:
            opts.append(f"vce(cluster {self.cmb_cluster.currentText()})")

        opts_str = f", {' '.join(opts)}" if opts else ""
        if_in_str = self.get_if_in_syntax()
        cmd = f"{cmd_name} {dep} {indeps_str}{if_in_str}{opts_str}"
        self._set_model_preview(cmd)


class SurvivalAnalysisDialog(BaseBridgeDialog):
    """Survival & Duration Analysis Studio (stset / stcox / sts list)."""

    def __init__(self, dataset, parent=None, initial_task="stset"):
        self.initial_task = (initial_task or "stset").lower()
        title = "Survival Analysis & Duration Models Studio (stset / stcox / sts)"
        super().__init__(title, parent)
        self.dataset = dataset
        self.setMinimumWidth(560)
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []

        # Tool selector
        t_box = QtWidgets.QHBoxLayout()
        t_box.addWidget(QtWidgets.QLabel("Survival Tool / Model:"))
        self.cmb_task = QtWidgets.QComboBox()
        self.cmb_task.addItems([
            "Declare Survival Structure (stset)",
            "Cox Proportional Hazards Model (stcox)",
            "Kaplan-Meier Survivor Table (sts list)"
        ])
        for idx in range(self.cmb_task.count()):
            txt = self.cmb_task.itemText(idx).lower()
            if self.initial_task in txt or (self.initial_task == "cox" and "stcox" in txt):
                self.cmb_task.setCurrentIndex(idx)
                break
        self.cmb_task.currentIndexChanged.connect(self._on_task_changed)
        t_box.addWidget(self.cmb_task, stretch=1)
        layout.addLayout(t_box)

        # Guidance Card
        self.guide_box = QtWidgets.QFrame()
        self.guide_box.setObjectName("guideBox")
        self.guide_box.setStyleSheet("QFrame#guideBox { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }")
        g_lay = QtWidgets.QVBoxLayout(self.guide_box)
        g_lay.setContentsMargins(10, 8, 10, 8)
        self.lbl_guide = QtWidgets.QLabel()
        self.lbl_guide.setStyleSheet("font-size: 9.5pt;")
        self.lbl_guide.setWordWrap(True)
        g_lay.addWidget(self.lbl_guide)
        layout.addWidget(self.guide_box)

        # Parameter Form
        self.param_box = QtWidgets.QGroupBox("Specification & Variables")
        self.param_layout = QtWidgets.QFormLayout(self.param_box)

        self.cmb_time = QtWidgets.QComboBox()
        self.cmb_time.addItems(cols)
        self.cmb_time.currentIndexChanged.connect(self.update_syntax)
        self.lbl_time = QtWidgets.QLabel("Time / Duration Variable:")
        self.param_layout.addRow(self.lbl_time, self.cmb_time)

        self.cmb_fail = QtWidgets.QComboBox()
        self.cmb_fail.addItems(["(None - no censoring)"] + cols)
        self.cmb_fail.currentIndexChanged.connect(self.update_syntax)
        self.lbl_fail = QtWidgets.QLabel("Failure / Event Variable (died):")
        self.param_layout.addRow(self.lbl_fail, self.cmb_fail)

        self.lst_covars = QtWidgets.QListWidget()
        self.lst_covars.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_covars.addItems(cols)
        self.lst_covars.itemSelectionChanged.connect(self.update_syntax)
        self.lst_covars.setMaximumHeight(90)
        self.lbl_covars = QtWidgets.QLabel("Predictors / Covariates:")
        self.param_layout.addRow(self.lbl_covars, self.lst_covars)

        self.cmb_by = QtWidgets.QComboBox()
        self.cmb_by.addItems(["(None)"] + cols)
        self.cmb_by.currentIndexChanged.connect(self.update_syntax)
        self.lbl_by = QtWidgets.QLabel("Group / Stratum Variable (by):")
        self.param_layout.addRow(self.lbl_by, self.cmb_by)

        layout.addWidget(self.param_box)

        # Observation Filter (if / in)
        self.setup_observation_filter(layout)

        # Live Equation Preview
        eq_box = QtWidgets.QGroupBox("Model Specification (Live Equation Preview)")
        eq_l = QtWidgets.QVBoxLayout(eq_box)
        self.lbl_equation = QtWidgets.QLabel()
        self.lbl_equation.setStyleSheet(
            "background-color: #1E1E1E; color: #E5C07B; font-family: 'Consolas', monospace; "
            "font-size: 10pt; font-weight: bold; padding: 6px; border: 1px solid #3E4451; border-radius: 4px;"
        )
        self.lbl_equation.setWordWrap(True)
        eq_l.addWidget(self.lbl_equation)
        layout.addWidget(eq_box)

        self.setup_footer(layout)
        self.btn_submit.setText("📈 Execute Survival Analysis")

        self._on_task_changed()
        self.update_syntax()

    def _on_task_changed(self):
        t = self.cmb_task.currentText().lower()
        is_stset = "stset" in t
        is_stcox = "stcox" in t
        is_sts = "sts" in t

        if is_stset:
            self.lbl_guide.setText("📘 <b>Declare Survival Data (stset):</b> Identifies duration time and failure event status for subsequent hazard rate and survival analyses.")
            self.lbl_time.setVisible(True)
            self.cmb_time.setVisible(True)
            self.lbl_fail.setVisible(True)
            self.cmb_fail.setVisible(True)
            self.lbl_covars.setVisible(False)
            self.lst_covars.setVisible(False)
            self.lbl_by.setVisible(False)
            self.cmb_by.setVisible(False)
        elif is_stcox:
            self.lbl_guide.setText("📘 <b>Cox Proportional Hazards Model (stcox):</b> Semi-parametric model estimating hazard ratios without assuming a specific baseline hazard distribution.")
            self.lbl_time.setVisible(False)
            self.cmb_time.setVisible(False)
            self.lbl_fail.setVisible(False)
            self.cmb_fail.setVisible(False)
            self.lbl_covars.setVisible(True)
            self.lst_covars.setVisible(True)
            self.lbl_by.setVisible(False)
            self.cmb_by.setVisible(False)
        elif is_sts:
            self.lbl_guide.setText("📘 <b>Kaplan-Meier Estimation (sts list):</b> Non-parametric product-limit estimator of the survival function S(t).")
            self.lbl_time.setVisible(False)
            self.cmb_time.setVisible(False)
            self.lbl_fail.setVisible(False)
            self.cmb_fail.setVisible(False)
            self.lbl_covars.setVisible(False)
            self.lst_covars.setVisible(False)
            self.lbl_by.setVisible(True)
            self.cmb_by.setVisible(True)

        self.update_syntax()

    def update_syntax(self):
        t = self.cmb_task.currentText().lower()
        if_in_str = self.get_if_in_syntax()

        cmd = ""
        if "stset" in t:
            timevar = self.cmb_time.currentText()
            failvar = self.cmb_fail.currentText()
            fail_opt = f", fail({failvar})" if failvar and failvar != "(None - no censoring)" else ""
            cmd = f"stset {timevar}{fail_opt}" if timevar else "stset"
            self.lbl_equation.setText(f"Duration: {timevar}, Event: {failvar if failvar != '(None - no censoring)' else 'uncensored'}")
        elif "stcox" in t:
            covs = [item.text() for item in self.lst_covars.selectedItems()]
            covs_str = " ".join(covs)
            cmd = f"stcox {covs_str}{if_in_str}" if covs else "stcox"
            rhs = " + ".join([f"β·{v}" for v in covs]) if covs else "X·β"
            self.lbl_equation.setText(f"h(t | X) = h₀(t) · exp({rhs})")
        elif "sts" in t:
            byvar = self.cmb_by.currentText()
            by_opt = f", by({byvar})" if byvar and byvar != "(None)" else ""
            cmd = f"sts list{if_in_str}{by_opt}"
            self.lbl_equation.setText("S(t) = ∏_{tᵢ ≤ t} (1 - dᵢ / nᵢ)  [Kaplan-Meier Survivor Function]")

        self._set_model_preview(cmd)


class CausalInferenceDialog(BaseBridgeDialog):
    """Modern Causal Inference Studio (bacondecomp / synth / psmatch)."""

    def __init__(self, dataset, parent=None, initial_tool="bacondecomp"):
        self.initial_tool = (initial_tool or "bacondecomp").lower()
        title = "Modern Causal Inference Studio"
        super().__init__(title, parent)
        self.dataset = dataset
        self.setMinimumWidth(560)
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []

        t_box = QtWidgets.QHBoxLayout()
        t_box.addWidget(QtWidgets.QLabel("Causal Method:"))
        self.cmb_tool = QtWidgets.QComboBox()
        self.cmb_tool.addItems([
            "Goodman-Bacon Decomposition (bacondecomp)",
            "Synthetic Control Method (synth)",
            "Propensity Score Matching (psmatch)",
            "Dynamic Event Study (eventstudy)"
        ])
        for idx in range(self.cmb_tool.count()):
            txt = self.cmb_tool.itemText(idx).lower()
            if self.initial_tool in txt or (self.initial_tool == "bacon" and "bacon" in txt) or (self.initial_tool == "event" and "event" in txt):
                self.cmb_tool.setCurrentIndex(idx)
                break
        self.cmb_tool.currentIndexChanged.connect(self._on_tool_changed)
        t_box.addWidget(self.cmb_tool, stretch=1)
        layout.addLayout(t_box)

        # Guidance Box
        self.guide_box = QtWidgets.QFrame()
        self.guide_box.setObjectName("guideBox")
        self.guide_box.setStyleSheet("QFrame#guideBox { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }")
        g_lay = QtWidgets.QVBoxLayout(self.guide_box)
        g_lay.setContentsMargins(10, 8, 10, 8)
        self.lbl_guide = QtWidgets.QLabel()
        self.lbl_guide.setStyleSheet("font-size: 9.5pt;")
        self.lbl_guide.setWordWrap(True)
        g_lay.addWidget(self.lbl_guide)
        layout.addWidget(self.guide_box)

        self.form_box = QtWidgets.QGroupBox("Design & Treatment Variables")
        self.form_layout = QtWidgets.QFormLayout(self.form_box)

        self.cmb_y = QtWidgets.QComboBox()
        self.cmb_y.addItems(cols)
        self.cmb_y.currentIndexChanged.connect(self.update_syntax)
        self.lbl_y = QtWidgets.QLabel("Outcome Variable (Y):")
        self.form_layout.addRow(self.lbl_y, self.cmb_y)

        self.cmb_post = QtWidgets.QComboBox()
        self.cmb_post.addItems(cols)
        self.cmb_post.currentIndexChanged.connect(self.update_syntax)
        self.lbl_post = QtWidgets.QLabel("Post-Treatment Timing Indicator (post):")
        self.form_layout.addRow(self.lbl_post, self.cmb_post)

        self.cmb_treat = QtWidgets.QComboBox()
        self.cmb_treat.addItems(cols)
        self.cmb_treat.currentIndexChanged.connect(self.update_syntax)
        self.lbl_treat = QtWidgets.QLabel("Treatment Group Indicator (treat):")
        self.form_layout.addRow(self.lbl_treat, self.cmb_treat)

        # Event Study Leads & Lags
        ev_box = QtWidgets.QHBoxLayout()
        ev_box.addWidget(QtWidgets.QLabel("Pre-treatment Leads:"))
        self.spn_leads = QtWidgets.QSpinBox()
        self.spn_leads.setRange(1, 20)
        self.spn_leads.setValue(3)
        self.spn_leads.valueChanged.connect(self.update_syntax)
        ev_box.addWidget(self.spn_leads)
        ev_box.addWidget(QtWidgets.QLabel("Post-treatment Lags:"))
        self.spn_lags = QtWidgets.QSpinBox()
        self.spn_lags.setRange(1, 20)
        self.spn_lags.setValue(3)
        self.spn_lags.valueChanged.connect(self.update_syntax)
        ev_box.addWidget(self.spn_lags)
        self.wgt_leads_lags = QtWidgets.QWidget()
        self.wgt_leads_lags.setLayout(ev_box)
        self.lbl_leads_lags = QtWidgets.QLabel("Event Window:")
        self.form_layout.addRow(self.lbl_leads_lags, self.wgt_leads_lags)

        self.lst_covars = QtWidgets.QListWidget()
        self.lst_covars.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_covars.addItems(cols)
        self.lst_covars.itemSelectionChanged.connect(self.update_syntax)
        self.lst_covars.setMaximumHeight(85)
        self.lbl_covars = QtWidgets.QLabel("Matching Covariates (X):")
        self.form_layout.addRow(self.lbl_covars, self.lst_covars)

        layout.addWidget(self.form_box)

        self.setup_observation_filter(layout)

        eq_box = QtWidgets.QGroupBox("Estimand (Live Equation Preview)")
        eq_l = QtWidgets.QVBoxLayout(eq_box)
        self.lbl_equation = QtWidgets.QLabel()
        self.lbl_equation.setStyleSheet(
            "background-color: #1E1E1E; color: #E5C07B; font-family: 'Consolas', monospace; "
            "font-size: 10pt; font-weight: bold; padding: 6px; border: 1px solid #3E4451; border-radius: 4px;"
        )
        self.lbl_equation.setWordWrap(True)
        eq_l.addWidget(self.lbl_equation)
        layout.addWidget(eq_box)

        self.setup_footer(layout)
        self.btn_submit.setText("📈 Estimate Causal Effect")

        self._on_tool_changed()
        self.update_syntax()

    def _on_tool_changed(self):
        t = self.cmb_tool.currentText().lower()
        is_bacon = "bacon" in t
        is_synth = "synth" in t
        is_ps = "match" in t or "psmatch" in t
        is_event = "event" in t or "eventstudy" in t

        if is_bacon:
            self.lbl_guide.setText("📘 <b>Goodman-Bacon Decomposition:</b> Decomposes two-way fixed effects (TWFE) into weighted averages of all 2x2 difference-in-differences comparisons (early vs late, late vs early, treated vs untreated).")
            self.lbl_y.setVisible(True)
            self.cmb_y.setVisible(True)
            self.lbl_post.setVisible(True)
            self.cmb_post.setVisible(True)
            self.lbl_treat.setVisible(True)
            self.cmb_treat.setVisible(True)
            self.lbl_leads_lags.setVisible(False)
            self.wgt_leads_lags.setVisible(False)
            self.lbl_covars.setVisible(False)
            self.lst_covars.setVisible(False)
        elif is_synth:
            self.lbl_guide.setText("📘 <b>Synthetic Control Method:</b> Constructs a weighted synthetic donor pool of untreated units that optimal matches pre-treatment trajectory of the treated unit.")
            self.lbl_y.setVisible(True)
            self.cmb_y.setVisible(True)
            self.lbl_post.setVisible(False)
            self.cmb_post.setVisible(False)
            self.lbl_treat.setVisible(False)
            self.cmb_treat.setVisible(False)
            self.lbl_leads_lags.setVisible(False)
            self.wgt_leads_lags.setVisible(False)
            self.lbl_covars.setVisible(True)
            self.lst_covars.setVisible(True)
        elif is_ps:
            self.lbl_guide.setText("📘 <b>Propensity Score Matching (psmatch):</b> Estimates average treatment effect on the treated (ATT) by matching treated and control units with similar predicted treatment probabilities e(X).")
            self.lbl_y.setVisible(True)
            self.lbl_y.setText("Outcome Variable (outcome):")
            self.cmb_y.setVisible(True)
            self.lbl_post.setVisible(False)
            self.cmb_post.setVisible(False)
            self.lbl_treat.setVisible(True)
            self.lbl_treat.setText("Treatment Variable (treat ∈ {0, 1}):")
            self.cmb_treat.setVisible(True)
            self.lbl_leads_lags.setVisible(False)
            self.wgt_leads_lags.setVisible(False)
            self.lbl_covars.setVisible(True)
            self.lst_covars.setVisible(True)
        elif is_event:
            self.lbl_guide.setText("📘 <b>Dynamic Event Study (eventstudy):</b> Estimates lead and lag treatment effect dynamics relative to adoption date with two-way fixed effects.")
            self.lbl_y.setVisible(True)
            self.lbl_y.setText("Outcome Variable (Y):")
            self.cmb_y.setVisible(True)
            self.lbl_post.setVisible(True)
            self.lbl_post.setText("Post/Period Timing Indicator:")
            self.cmb_post.setVisible(True)
            self.lbl_treat.setVisible(True)
            self.lbl_treat.setText("Treatment Group Indicator (treat):")
            self.cmb_treat.setVisible(True)
            self.lbl_leads_lags.setVisible(True)
            self.wgt_leads_lags.setVisible(True)
            self.lbl_covars.setVisible(False)
            self.lst_covars.setVisible(False)

        self.update_syntax()

    def update_syntax(self):
        t = self.cmb_tool.currentText().lower()
        y = self.cmb_y.currentText()
        post = self.cmb_post.currentText()
        treat = self.cmb_treat.currentText()
        covs = [item.text() for item in self.lst_covars.selectedItems() if item.text() not in (y, treat)]
        covs_str = " ".join(covs)
        if_in_str = self.get_if_in_syntax()

        cmd = ""
        if "bacon" in t:
            cmd = f"bacondecomp {y} {post} {treat}{if_in_str}" if y and post and treat else "bacondecomp"
            self.lbl_equation.setText("β^{TWFE} = ∑ s_{k,U}·β̂_{k,U} + ∑ s_{k,l}·β̂_{k,l}^{early} + ∑ s_{l,k}·β̂_{l,k}^{late}")
        elif "synth" in t:
            cmd = f"synth {y} {covs_str}{if_in_str}" if y and covs else "synth"
            self.lbl_equation.setText("min_W || X₁ - X₀W ||_V  s.t.  w_j ≥ 0,  ∑ w_j = 1")
        elif "match" in t or "psmatch" in t:
            opts = [f"outcome({y})"] if y else []
            opts_str = f", {', '.join(opts)}" if opts else ""
            cmd = f"psmatch {treat} {covs_str}{if_in_str}{opts_str}" if treat and covs else "psmatch"
            self.lbl_equation.setText("ATT = E[ Y(1) - Y(0) | D=1 ]  matched on e(X) = Pr(D=1 | X)")
        elif "event" in t or "eventstudy" in t:
            leads = self.spn_leads.value()
            lags = self.spn_lags.value()
            opts = [f"leads({leads})", f"lags({lags})"]
            opts_str = f", {', '.join(opts)}"
            cmd = f"eventstudy {y} {treat} {post}{if_in_str}{opts_str}" if y and treat and post else "eventstudy"
            self.lbl_equation.setText(f"Y_{{it}} = α_i + λ_t + ∑_{{k=-{leads}, k≠-1}}^{{{lags}}} β_k · D_{{it}}^k + ε_{{it}}")

        self._set_model_preview(cmd)


class DtableDialog(BaseBridgeDialog):
    """Descriptive Statistics & Demographic Table Builder (dtable)."""

    def __init__(self, dataset, parent=None):
        super().__init__("Demographic Summary Table (dtable)", parent)
        self.dataset = dataset
        self.setMinimumWidth(540)
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []

        guide_box = QtWidgets.QFrame()
        guide_box.setObjectName("guideBox")
        guide_box.setStyleSheet("QFrame#guideBox { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }")
        g_lay = QtWidgets.QVBoxLayout(guide_box)
        g_lay.setContentsMargins(10, 8, 10, 8)
        lbl_guide = QtWidgets.QLabel("📘 <b>Descriptive Table (dtable):</b> Formats a publication-ready baseline table of continuous means/SDs and categorical frequencies across groups (Table 1).")
        lbl_guide.setStyleSheet("font-size: 9.5pt;")
        lbl_guide.setWordWrap(True)
        g_lay.addWidget(lbl_guide)
        layout.addWidget(guide_box)

        form_box = QtWidgets.QGroupBox("Table Variables")
        f_lay = QtWidgets.QFormLayout(form_box)

        self.lst_vars = QtWidgets.QListWidget()
        self.lst_vars.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_vars.addItems(cols)
        self.lst_vars.itemSelectionChanged.connect(self.update_syntax)
        self.lst_vars.setMaximumHeight(110)
        f_lay.addRow("Summary Variables:", self.lst_vars)

        self.cmb_by = QtWidgets.QComboBox()
        self.cmb_by.addItem("<None - Overall Sample>")
        self.cmb_by.addItems(cols)
        self.cmb_by.currentIndexChanged.connect(self.update_syntax)
        f_lay.addRow("Group By Variable (by):", self.cmb_by)

        self.edt_export = QtWidgets.QLineEdit()
        self.edt_export.setPlaceholderText("Optional: table1.docx, table1.html, or table1.xlsx")
        self.edt_export.textChanged.connect(self.update_syntax)
        f_lay.addRow("Export Filename:", self.edt_export)

        layout.addWidget(form_box)

        self.setup_observation_filter(layout)

        eq_box = QtWidgets.QGroupBox("Table Output Format")
        eq_l = QtWidgets.QVBoxLayout(eq_box)
        self.lbl_table_spec = QtWidgets.QLabel("Mean (SD) for continuous variables | Frequency (%) for categorical factors")
        self.lbl_table_spec.setStyleSheet("font-size: 9.5pt; color: #98C379;")
        eq_l.addWidget(self.lbl_table_spec)
        layout.addWidget(eq_box)

        self.setup_footer(layout)
        self.btn_submit.setText("📋 Generate Table")
        self.update_syntax()

    def update_syntax(self):
        sel_vars = [item.text() for item in self.lst_vars.selectedItems()]
        vars_str = " ".join(sel_vars)
        by_var = self.cmb_by.currentText()
        exp_file = self.edt_export.text().strip()
        if_in_str = self.get_if_in_syntax()

        opts = []
        if by_var and not by_var.startswith("<None"):
            opts.append(f"by({by_var})")
        if exp_file:
            opts.append(f"export({exp_file})")

        opts_str = f", {', '.join(opts)}" if opts else ""
        cmd = f"dtable {vars_str}{if_in_str}{opts_str}" if vars_str else "dtable"
        self._set_model_preview(cmd)


class AdvancedLinearDialog(BaseBridgeDialog):
    """Advanced Linear Models (cnsreg / sureg / bigregress)."""

    def __init__(self, dataset, parent=None, initial_model="cnsreg"):
        self.initial_model = (initial_model or "cnsreg").lower()
        super().__init__("Advanced Linear Regression Studio", parent)
        self.dataset = dataset
        self.setMinimumWidth(580)
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []

        m_box = QtWidgets.QHBoxLayout()
        m_box.addWidget(QtWidgets.QLabel("Model Architecture:"))
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems([
            "Constrained Linear Regression (cnsreg)",
            "Seemingly Unrelated Regression (sureg)",
            "Streaming Big Data OLS (bigregress)"
        ])
        for idx in range(self.cmb_model.count()):
            txt = self.cmb_model.itemText(idx).lower()
            if self.initial_model in txt or (self.initial_model == "stream" and "bigregress" in txt):
                self.cmb_model.setCurrentIndex(idx)
                break
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        m_box.addWidget(self.cmb_model, stretch=1)
        layout.addLayout(m_box)

        # Guidance Card
        self.guide_box = QtWidgets.QFrame()
        self.guide_box.setObjectName("guideBox")
        self.guide_box.setStyleSheet("QFrame#guideBox { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }")
        g_lay = QtWidgets.QVBoxLayout(self.guide_box)
        g_lay.setContentsMargins(10, 8, 10, 8)
        self.lbl_guide = QtWidgets.QLabel()
        self.lbl_guide.setStyleSheet("font-size: 9.5pt;")
        self.lbl_guide.setWordWrap(True)
        g_lay.addWidget(self.lbl_guide)
        layout.addWidget(self.guide_box)

        # Main Equation Specification Box
        self.form_box = QtWidgets.QGroupBox("Variables & Specification")
        self.form_layout = QtWidgets.QFormLayout(self.form_box)

        self.cmb_dep1 = QtWidgets.QComboBox()
        self.cmb_dep1.addItems(cols)
        self.cmb_dep1.currentIndexChanged.connect(self.update_syntax)
        self.lbl_dep1 = QtWidgets.QLabel("Primary Dependent Variable (Y₁):")
        self.form_layout.addRow(self.lbl_dep1, self.cmb_dep1)

        self.lst_indep1 = QtWidgets.QListWidget()
        self.lst_indep1.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_indep1.addItems(cols)
        self.lst_indep1.itemSelectionChanged.connect(self.update_syntax)
        self.lst_indep1.setMaximumHeight(90)
        self.lbl_indep1 = QtWidgets.QLabel("Primary Independent Variables (X₁):")
        self.form_layout.addRow(self.lbl_indep1, self.lst_indep1)

        # Equation 2 (for sureg)
        self.cmb_dep2 = QtWidgets.QComboBox()
        self.cmb_dep2.addItems(cols)
        self.cmb_dep2.currentIndexChanged.connect(self.update_syntax)
        self.lbl_dep2 = QtWidgets.QLabel("Second Equation Depvar (Y₂):")
        self.form_layout.addRow(self.lbl_dep2, self.cmb_dep2)

        self.lst_indep2 = QtWidgets.QListWidget()
        self.lst_indep2.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_indep2.addItems(cols)
        self.lst_indep2.itemSelectionChanged.connect(self.update_syntax)
        self.lst_indep2.setMaximumHeight(80)
        self.lbl_indep2 = QtWidgets.QLabel("Second Equation Indepvars (X₂):")
        self.form_layout.addRow(self.lbl_indep2, self.lst_indep2)

        # Options for cnsreg
        self.edt_constraints = QtWidgets.QLineEdit()
        self.edt_constraints.setPlaceholderText("e.g. 1 2 (or constraint matrix)")
        self.edt_constraints.setText("1")
        self.edt_constraints.textChanged.connect(self.update_syntax)
        self.lbl_constraints = QtWidgets.QLabel("Constraints List / Spec:")
        self.form_layout.addRow(self.lbl_constraints, self.edt_constraints)

        # Options for bigregress
        self.spn_chunk = QtWidgets.QSpinBox()
        self.spn_chunk.setRange(100, 1000000)
        self.spn_chunk.setSingleStep(5000)
        self.spn_chunk.setValue(10000)
        self.spn_chunk.valueChanged.connect(self.update_syntax)
        self.lbl_chunk = QtWidgets.QLabel("Streaming Chunk Size (# rows):")
        self.form_layout.addRow(self.lbl_chunk, self.spn_chunk)

        self.chk_robust = QtWidgets.QCheckBox("Heteroskedasticity-Robust Standard Errors")
        self.chk_robust.stateChanged.connect(self.update_syntax)
        self.form_layout.addRow("", self.chk_robust)

        layout.addWidget(self.form_box)

        self.setup_observation_filter(layout)

        eq_box = QtWidgets.QGroupBox("Mathematical Formulation (Live Equation Preview)")
        eq_l = QtWidgets.QVBoxLayout(eq_box)
        self.lbl_equation = QtWidgets.QLabel()
        self.lbl_equation.setStyleSheet(
            "background-color: #1E1E1E; color: #E5C07B; font-family: 'Consolas', monospace; "
            "font-size: 10pt; font-weight: bold; padding: 6px; border: 1px solid #3E4451; border-radius: 4px;"
        )
        self.lbl_equation.setWordWrap(True)
        eq_l.addWidget(self.lbl_equation)
        layout.addWidget(eq_box)

        self.setup_footer(layout)
        self.btn_submit.setText("⚡ Fit Model")

        self._on_model_changed()
        self.update_syntax()

    def _on_model_changed(self):
        m = self.cmb_model.currentText().lower()
        is_cns = "cnsreg" in m
        is_su = "sureg" in m
        is_big = "bigregress" in m

        if is_cns:
            self.lbl_guide.setText("📘 <b>Constrained Linear Regression (cnsreg):</b> Implements restricted least squares subject to linear parameter constraints Rβ = r.")
            self.lbl_dep1.setText("Dependent Variable (Y):")
            self.lbl_indep1.setText("Independent Variables (X):")
            self.lbl_dep2.setVisible(False)
            self.cmb_dep2.setVisible(False)
            self.lbl_indep2.setVisible(False)
            self.lst_indep2.setVisible(False)
            self.lbl_constraints.setVisible(True)
            self.edt_constraints.setVisible(True)
            self.lbl_chunk.setVisible(False)
            self.spn_chunk.setVisible(False)
            self.chk_robust.setVisible(False)
        elif is_su:
            self.lbl_guide.setText("📘 <b>Seemingly Unrelated Regression (sureg):</b> Fits a system of equations linked by cross-equation contemporaneous error correlation using Generalized Least Squares (Zellner 1962).")
            self.lbl_dep1.setText("Equation 1 Depvar (Y₁):")
            self.lbl_indep1.setText("Equation 1 Indepvars (X₁):")
            self.lbl_dep2.setVisible(True)
            self.cmb_dep2.setVisible(True)
            self.lbl_indep2.setVisible(True)
            self.lst_indep2.setVisible(True)
            self.lbl_constraints.setVisible(False)
            self.edt_constraints.setVisible(False)
            self.lbl_chunk.setVisible(False)
            self.spn_chunk.setVisible(False)
            self.chk_robust.setVisible(False)
        elif is_big:
            self.lbl_guide.setText("📘 <b>Streaming Out-of-Core Big Data OLS (bigregress):</b> Processes large-scale datasets chunk-by-chunk via online accumulators with bounded memory footprint.")
            self.lbl_dep1.setText("Dependent Variable (Y):")
            self.lbl_indep1.setText("Independent Variables (X):")
            self.lbl_dep2.setVisible(False)
            self.cmb_dep2.setVisible(False)
            self.lbl_indep2.setVisible(False)
            self.lst_indep2.setVisible(False)
            self.lbl_constraints.setVisible(False)
            self.edt_constraints.setVisible(False)
            self.lbl_chunk.setVisible(True)
            self.spn_chunk.setVisible(True)
            self.chk_robust.setVisible(True)

        self.update_syntax()

    def update_syntax(self):
        m = self.cmb_model.currentText().lower()
        y1 = self.cmb_dep1.currentText()
        x1_items = [item.text() for item in self.lst_indep1.selectedItems() if item.text() != y1]
        x1_str = " ".join(x1_items)
        if_in_str = self.get_if_in_syntax()

        cmd = ""
        if "cnsreg" in m:
            cons = self.edt_constraints.text().strip() or "1"
            cmd = f"cnsreg {y1} {x1_str}{if_in_str}, constraints({cons})" if y1 and x1_str else "cnsreg"
            self.lbl_equation.setText(f"min_β || {y1} - X₁β ||²  subject to  R·β = r")
        elif "sureg" in m:
            y2 = self.cmb_dep2.currentText()
            x2_items = [item.text() for item in self.lst_indep2.selectedItems() if item.text() != y2]
            x2_str = " ".join(x2_items)
            eq1 = f"({y1} {x1_str})" if y1 and x1_str else ""
            eq2 = f"({y2} {x2_str})" if y2 and x2_str else ""
            cmd = f"sureg {eq1} {eq2}{if_in_str}".strip() if eq1 and eq2 else "sureg"
            self.lbl_equation.setText("Y₁ = X₁β₁ + ε₁,  Y₂ = X₂β₂ + ε₂,  Cov(ε₁, ε₂) = Σ ⊗ I")
        elif "bigregress" in m:
            opts = []
            if self.spn_chunk.value() != 10000:
                opts.append(f"chunksize({self.spn_chunk.value()})")
            if self.chk_robust.isChecked():
                opts.append("robust")
            opts_str = f", {', '.join(opts)}" if opts else ""
            cmd = f"bigregress {y1} {x1_str}{if_in_str}{opts_str}" if y1 and x1_str else "bigregress"
            self.lbl_equation.setText("β̂ = (∑_b X_b'X_b)⁻¹ (∑_b X_b'y_b)  [Online streaming accumulator]")

        self._set_model_preview(cmd)


class SurveyDialog(BaseBridgeDialog):
    """Survey Data Analysis Studio (svyset / svy: regress)."""

    def __init__(self, dataset, parent=None, initial_mode="svyset"):
        self.initial_mode = (initial_mode or "svyset").lower()
        super().__init__("Complex Survey Data Studio", parent)
        self.dataset = dataset
        self.setMinimumWidth(560)
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        cols = list(self.dataset.df.columns) if not self.dataset.is_empty else []

        m_box = QtWidgets.QHBoxLayout()
        m_box.addWidget(QtWidgets.QLabel("Survey Workflow:"))
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems([
            "Declare Survey Design (svyset)",
            "Survey-Weighted Linear Regression (svy: regress)"
        ])
        for idx in range(self.cmb_mode.count()):
            txt = self.cmb_mode.itemText(idx).lower()
            if self.initial_mode in txt or (self.initial_mode == "regress" and "svy: regress" in txt):
                self.cmb_mode.setCurrentIndex(idx)
                break
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        m_box.addWidget(self.cmb_mode, stretch=1)
        layout.addLayout(m_box)

        # Guidance Card
        self.guide_box = QtWidgets.QFrame()
        self.guide_box.setObjectName("guideBox")
        self.guide_box.setStyleSheet("QFrame#guideBox { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }")
        g_lay = QtWidgets.QVBoxLayout(self.guide_box)
        g_lay.setContentsMargins(10, 8, 10, 8)
        self.lbl_guide = QtWidgets.QLabel()
        self.lbl_guide.setStyleSheet("font-size: 9.5pt;")
        self.lbl_guide.setWordWrap(True)
        g_lay.addWidget(self.lbl_guide)
        layout.addWidget(self.guide_box)

        # svyset form
        self.grp_svyset = QtWidgets.QGroupBox("Survey Sampling Architecture")
        f_set = QtWidgets.QFormLayout(self.grp_svyset)

        self.cmb_psu = QtWidgets.QComboBox()
        self.cmb_psu.addItem("<None / Simple Random Sample>")
        self.cmb_psu.addItems(cols)
        self.cmb_psu.currentIndexChanged.connect(self.update_syntax)
        f_set.addRow("Primary Sampling Unit (psu / cluster):", self.cmb_psu)

        self.cmb_pweight = QtWidgets.QComboBox()
        self.cmb_pweight.addItem("<None>")
        self.cmb_pweight.addItems(cols)
        self.cmb_pweight.currentIndexChanged.connect(self.update_syntax)
        f_set.addRow("Sampling Probability Weight (pweight):", self.cmb_pweight)

        self.cmb_strata = QtWidgets.QComboBox()
        self.cmb_strata.addItem("<None>")
        self.cmb_strata.addItems(cols)
        self.cmb_strata.currentIndexChanged.connect(self.update_syntax)
        f_set.addRow("Stratification Variable (strata):", self.cmb_strata)

        layout.addWidget(self.grp_svyset)

        # svy: regress form
        self.grp_svyreg = QtWidgets.QGroupBox("Survey Regression Specification")
        f_reg = QtWidgets.QFormLayout(self.grp_svyreg)

        self.cmb_dep = QtWidgets.QComboBox()
        self.cmb_dep.addItems(cols)
        self.cmb_dep.currentIndexChanged.connect(self.update_syntax)
        f_reg.addRow("Dependent Variable (Y):", self.cmb_dep)

        self.lst_indep = QtWidgets.QListWidget()
        self.lst_indep.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lst_indep.addItems(cols)
        self.lst_indep.itemSelectionChanged.connect(self.update_syntax)
        self.lst_indep.setMaximumHeight(85)
        f_reg.addRow("Independent Variables (X):", self.lst_indep)

        layout.addWidget(self.grp_svyreg)

        self.setup_observation_filter(layout)

        eq_box = QtWidgets.QGroupBox("Survey Inference Design (Live Equation Preview)")
        eq_l = QtWidgets.QVBoxLayout(eq_box)
        self.lbl_equation = QtWidgets.QLabel()
        self.lbl_equation.setStyleSheet(
            "background-color: #1E1E1E; color: #E5C07B; font-family: 'Consolas', monospace; "
            "font-size: 10pt; font-weight: bold; padding: 6px; border: 1px solid #3E4451; border-radius: 4px;"
        )
        self.lbl_equation.setWordWrap(True)
        eq_l.addWidget(self.lbl_equation)
        layout.addWidget(eq_box)

        self.setup_footer(layout)
        self.btn_submit.setText("🏛️ Declare / Run Survey Analysis")

        self._on_mode_changed()
        self.update_syntax()

    def _on_mode_changed(self):
        is_set = "svyset" in self.cmb_mode.currentText().lower()
        self.grp_svyset.setVisible(is_set)
        self.grp_svyreg.setVisible(not is_set)

        if is_set:
            self.lbl_guide.setText("📘 <b>Survey Design Setup (svyset):</b> Declares the sampling design variables (primary sampling unit clusters, probability weights, and strata) for design-based variance estimation.")
            self.btn_submit.setText("🏛️ Set Survey Design")
        else:
            self.lbl_guide.setText("📘 <b>Survey-Weighted Regression (svy: regress):</b> Fits linear regression adjusting for clustering, stratification, and sampling weights via Taylor series linearization.")
            self.btn_submit.setText("📈 Fit Survey Regression")

        self.update_syntax()

    def update_syntax(self):
        is_set = "svyset" in self.cmb_mode.currentText().lower()
        if is_set:
            psu = self.cmb_psu.currentText()
            pw = self.cmb_pweight.currentText()
            st = self.cmb_strata.currentText()

            psu_str = "" if psu.startswith("<None") else psu
            pw_str = f" [pweight={pw}]" if not pw.startswith("<None") else ""
            st_str = f", strata({st})" if not st.startswith("<None") else ""
            body = (f"{psu_str}{pw_str}").strip()
            cmd = f"svyset {body}{st_str}".strip() if (body or st_str) else "svyset"
            self.lbl_equation.setText("Var(θ̂) = ∑_h (1 - f_h)·(n_h / (n_h - 1)) ∑_i (z_hi - z̄_h)'(z_hi - z̄_h)  [Taylor linearization]")
        else:
            y = self.cmb_dep.currentText()
            xs = [item.text() for item in self.lst_indep.selectedItems() if item.text() != y]
            xs_str = " ".join(xs)
            if_in_str = self.get_if_in_syntax()
            cmd = f"svy: regress {y} {xs_str}{if_in_str}" if y and xs_str else "svy: regress"
            self.lbl_equation.setText("β̂_{svy} = (X' W X)⁻¹ X' W y  with linearized sandwich VCE")

        self._set_model_preview(cmd)


class PostEstimationDialog(BaseBridgeDialog):
    """Post-Estimation Testing, Diagnostics, and Linear/Nonlinear Combinations."""

    def __init__(self, dataset, parent=None, initial_tool="lincom"):
        self.initial_tool = (initial_tool or "lincom").lower()
        super().__init__("Post-Estimation Tests & Linear Combinations", parent)
        self.dataset = dataset
        self.setMinimumWidth(580)
        self._build_controls()

    def _build_controls(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        t_box = QtWidgets.QHBoxLayout()
        t_box.addWidget(QtWidgets.QLabel("Post-Estimation Procedure:"))
        self.cmb_tool = QtWidgets.QComboBox()
        self.cmb_tool.addItems([
            "Linear Combinations of Parameters (lincom)",
            "Nonlinear Combinations of Parameters (nlcom)",
            "Nonlinear Hypothesis Wald Test (testnl)",
            "Variance Inflation Factors (estat vif)",
            "Breusch-Pagan Heteroskedasticity (estat hettest)",
            "Ramsey RESET Misspecification (estat ovtest)",
            "Montiel Olea-Pflueger Weak IV Test (estat weakiv)",
            "Durbin-Wu-Hausman Endogeneity Test (estat endogenous)",
            "Overidentification Test (estat overid)",
            "Brant Parallel Regression Test (estat brant)",
            "Hausman Fixed vs Random Effects Test (hausman fe re)"
        ])
        for idx in range(self.cmb_tool.count()):
            txt = self.cmb_tool.itemText(idx).lower()
            if self.initial_tool in txt or (self.initial_tool == "weak" and "weakiv" in txt) or (self.initial_tool == "hausman" and "hausman" in txt):
                self.cmb_tool.setCurrentIndex(idx)
                break
        self.cmb_tool.currentIndexChanged.connect(self._on_tool_changed)
        t_box.addWidget(self.cmb_tool, stretch=1)
        layout.addLayout(t_box)

        # Guidance Card
        self.guide_box = QtWidgets.QFrame()
        self.guide_box.setObjectName("guideBox")
        self.guide_box.setStyleSheet("QFrame#guideBox { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }")
        g_lay = QtWidgets.QVBoxLayout(self.guide_box)
        g_lay.setContentsMargins(10, 8, 10, 8)
        self.lbl_guide = QtWidgets.QLabel()
        self.lbl_guide.setStyleSheet("font-size: 9.5pt;")
        self.lbl_guide.setWordWrap(True)
        g_lay.addWidget(self.lbl_guide)
        layout.addWidget(self.guide_box)

        # Expression Input (for lincom / nlcom / testnl)
        self.grp_expr = QtWidgets.QGroupBox("Hypothesis / Combination Expression")
        f_exp = QtWidgets.QFormLayout(self.grp_expr)

        self.edt_expr = QtWidgets.QLineEdit()
        self.edt_expr.setPlaceholderText("e.g. x1 - 2*x2 (or x1 / x2 = 1)")
        self.edt_expr.setText("x1 - x2")
        self.edt_expr.textChanged.connect(self.update_syntax)
        self.lbl_expr = QtWidgets.QLabel("Parameter Expression:")
        f_exp.addRow(self.lbl_expr, self.edt_expr)

        layout.addWidget(self.grp_expr)

        eq_box = QtWidgets.QGroupBox("Statistical Hypothesis (Live Equation Preview)")
        eq_l = QtWidgets.QVBoxLayout(eq_box)
        self.lbl_equation = QtWidgets.QLabel()
        self.lbl_equation.setStyleSheet(
            "background-color: #1E1E1E; color: #E5C07B; font-family: 'Consolas', monospace; "
            "font-size: 10pt; font-weight: bold; padding: 6px; border: 1px solid #3E4451; border-radius: 4px;"
        )
        self.lbl_equation.setWordWrap(True)
        eq_l.addWidget(self.lbl_equation)
        layout.addWidget(eq_box)

        self.setup_footer(layout)
        self.btn_submit.setText("🔬 Execute Test")

        self._on_tool_changed()
        self.update_syntax()

    def _on_tool_changed(self):
        t = self.cmb_tool.currentText().lower()
        is_lin = "lincom" in t
        is_nl = "nlcom" in t
        is_testnl = "testnl" in t
        needs_expr = is_lin or is_nl or is_testnl

        self.grp_expr.setVisible(needs_expr)

        if is_lin:
            self.lbl_guide.setText("📘 <b>Linear Combination (lincom):</b> Computes point estimate, standard error, z/t-statistic, and confidence intervals for a linear combination of coefficients c'β.")
            self.lbl_expr.setText("Linear Expression (c'β):")
            self.edt_expr.setPlaceholderText("e.g. price - 2*mpg")
        elif is_nl:
            self.lbl_guide.setText("📘 <b>Nonlinear Combination (nlcom):</b> Computes point estimate and delta-method asymptotic standard errors for a nonlinear function g(β).")
            self.lbl_expr.setText("Nonlinear Expression g(β):")
            self.edt_expr.setPlaceholderText("e.g. exp(price) / (1 + exp(price))")
        elif is_testnl:
            self.lbl_guide.setText("📘 <b>Nonlinear Wald Test (testnl):</b> Tests the nonlinear hypothesis H₀: g(β) = q against H₁: g(β) ≠ q using the Wald test statistic.")
            self.lbl_expr.setText("Equality Hypothesis:")
            self.edt_expr.setPlaceholderText("e.g. price / mpg = 1")
        elif "vif" in t:
            self.lbl_guide.setText("📘 <b>Variance Inflation Factors (estat vif):</b> Measures the severity of multicollinearity among independent variables. VIF > 10 indicates high multicollinearity.")
        elif "hettest" in t:
            self.lbl_guide.setText("📘 <b>Breusch-Pagan / Cook-Weisberg Test (estat hettest):</b> Tests H₀: Constant variance (homoskedasticity) vs H₁: Var(ε | X) = σ² · h(z'α).")
        elif "ovtest" in t:
            self.lbl_guide.setText("📘 <b>Ramsey RESET Test (estat ovtest):</b> Tests for omitted variable bias and incorrect functional form using powers of fitted values (ŷ², ŷ³, ŷ⁴).")
        elif "weakiv" in t:
            self.lbl_guide.setText("📘 <b>Montiel Olea-Pflueger Robust Weak IV Test (estat weakiv):</b> Evaluates instrument strength under heteroskedasticity, autocorrelation, and clustering against critical values.")
        elif "endogenous" in t:
            self.lbl_guide.setText("📘 <b>Durbin-Wu-Hausman Endogeneity Test (estat endogenous):</b> Tests H₀: Specified endogenous regressors are exogenous vs H₁: Regressors are endogenous.")
        elif "overid" in t:
            self.lbl_guide.setText("📘 <b>Sargan-Hansen Overidentification Test (estat overid):</b> Tests H₀: All instruments are valid (orthogonal to error term) in an overidentified IV model.")
        elif "brant" in t:
            self.lbl_guide.setText("📘 <b>Brant Test (estat brant):</b> Tests the parallel regression / proportional odds assumption across ordinal cutoffs in ordered logistic regression.")
        elif "hausman" in t:
            self.lbl_guide.setText("📘 <b>Hausman Specification Test (hausman fe re):</b> Tests H₀: Random effects GLS estimator is consistent and efficient vs H₁: Fixed effects estimator is required.")

        self.update_syntax()

    def update_syntax(self):
        t = self.cmb_tool.currentText().lower()
        expr = self.edt_expr.text().strip()

        cmd = ""
        if "lincom" in t:
            cmd = f"lincom {expr}" if expr else "lincom"
            self.lbl_equation.setText("c'β̂  ~  N( c'β,  c'·V·c )  [Linear Delta Method]")
        elif "nlcom" in t:
            cmd = f"nlcom {expr}" if expr else "nlcom"
            self.lbl_equation.setText("g(β̂)  ~  N( g(β),  ∇g(β̂)'·V·∇g(β̂) )  [First-Order Delta Method]")
        elif "testnl" in t:
            cmd = f"testnl {expr}" if expr else "testnl"
            self.lbl_equation.setText("W = [g(β̂) - q]' · [∇g'·V·∇g]⁻¹ · [g(β̂) - q]  ~  χ²(q)")
        elif "vif" in t:
            cmd = "estat vif"
            self.lbl_equation.setText("VIF_j = 1 / (1 - R_j²)  for each predictor j")
        elif "hettest" in t:
            cmd = "estat hettest"
            self.lbl_equation.setText("H₀: σ_i² = σ²  vs  H₁: σ_i² = σ² · f(α₀ + α'x_i)")
        elif "ovtest" in t:
            cmd = "estat ovtest"
            self.lbl_equation.setText("H₀: Model has no omitted variables  [F-test on ŷ², ŷ³, ŷ⁴]")
        elif "weakiv" in t:
            cmd = "estat weakiv"
            self.lbl_equation.setText("F_{eff} vs Montiel Olea-Pflueger critical values (τ = 5%, 10%)")
        elif "endogenous" in t:
            cmd = "estat endogenous"
            self.lbl_equation.setText("H₀: Regressors are exogenous  [Durbin-Wu-Hausman χ² test]")
        elif "overid" in t:
            cmd = "estat overid"
            self.lbl_equation.setText("H₀: Instruments are orthogonal to error term  [Sargan / Hansen J ~ χ²(L-K)]")
        elif "brant" in t:
            cmd = "estat brant"
            self.lbl_equation.setText("H₀: Parallel regression assumption holds: β_1 = β_2 = ... = β_{J-1}")
        elif "hausman" in t:
            cmd = "hausman fe re"
            self.lbl_equation.setText("H = (β̂_{FE} - β̂_{RE})' · [Var(β̂_{FE}) - Var(β̂_{RE})]⁻¹ · (β̂_{FE} - β̂_{RE})  ~  χ²(k)")

        self._set_model_preview(cmd)




