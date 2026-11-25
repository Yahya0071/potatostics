"""
gui_qt/variables_pane.py - Variables & Dataset Properties Inspector
Provides:
- Searchable variable list showing names, storage types, labels, and value label links
- Double-click to insert variable name into the command line
- Live dataset properties card (observations, variables, memory size, sort status)
"""

from PySide6 import QtWidgets
from PySide6.QtCore import Signal

class VariablesPane(QtWidgets.QWidget):
    variable_double_clicked = Signal(str)

    def __init__(self, dataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 1. Search filter bar
        self.search_box = QtWidgets.QLineEdit()
        self.search_box.setPlaceholderText("Filter variables")
        self.search_box.setAccessibleName("Filter by variable name or label")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.setFixedHeight(28)
        self.search_box.textChanged.connect(self._filter_variables)
        layout.addWidget(self.search_box)

        # 2. Variables Tree View
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Type", "Label", "Val Label"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(False)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        
        # Header column widths
        self.tree.setColumnWidth(0, 85)
        self.tree.setColumnWidth(1, 60)
        self.tree.setColumnWidth(2, 110)
        layout.addWidget(self.tree, stretch=3)

        # 3. Properties Group Box
        prop_box = QtWidgets.QGroupBox("Properties")
        prop_layout = QtWidgets.QFormLayout(prop_box)
        prop_layout.setContentsMargins(8, 8, 8, 8)
        prop_layout.setSpacing(4)

        self.lbl_obs = QtWidgets.QLabel("0")
        self.lbl_vars = QtWidgets.QLabel("0")
        self.lbl_mem = QtWidgets.QLabel("0 KB")
        self.lbl_file = QtWidgets.QLabel("(None)")
        self.lbl_file.setToolTip("Dataset filename")

        prop_layout.addRow("Obs:", self.lbl_obs)
        prop_layout.addRow("Vars:", self.lbl_vars)
        prop_layout.addRow("Size:", self.lbl_mem)
        prop_layout.addRow("File:", self.lbl_file)

        layout.addWidget(prop_box)

    def refresh(self):
        self.tree.clear()
        df = self.dataset.df

        if df is None:
            self.lbl_obs.setText("0")
            self.lbl_vars.setText("0")
            self.lbl_mem.setText("0 KB")
            self.lbl_file.setText("(None)")
            return

        for col in df.columns:
            dtype_str = str(df[col].dtype)
            var_label = self.dataset.var_labels.get(col, "")
            val_label_name = self.dataset.val_label_names.get(col, "")

            item = QtWidgets.QTreeWidgetItem([col, dtype_str, var_label, val_label_name])
            item.setToolTip(0, f"Variable: {col}\nType: {dtype_str}\nLabel: {var_label}")
            self.tree.addTopLevelItem(item)

        # Update properties
        n_obs = len(df)
        n_vars = len(df.columns)
        mem_bytes = df.memory_usage(deep=True).sum()
        mem_str = f"{mem_bytes / 1024:.1f} KB" if mem_bytes < 1024 * 1024 else f"{mem_bytes / (1024 * 1024):.2f} MB"

        self.lbl_obs.setText(f"{n_obs:,}")
        self.lbl_vars.setText(f"{n_vars:,}")
        self.lbl_mem.setText(mem_str)
        fname = self.dataset.filename or "(Memory)"
        import os
        self.lbl_file.setText(os.path.basename(fname))
        self.lbl_file.setToolTip(fname)
        self._filter_variables(self.search_box.text())

    def _filter_variables(self, query):
        query = query.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            name = item.text(0).lower()
            label = item.text(2).lower()
            match = (not query) or (query in name) or (query in label)
            item.setHidden(not match)

    def _on_item_double_clicked(self, item, col):
        varname = item.text(0)
        self.variable_double_clicked.emit(varname)
