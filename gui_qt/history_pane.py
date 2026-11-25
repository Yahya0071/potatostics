"""
gui_qt/history_pane.py - Command Review & History Pane
Provides:
- Searchable history list with error/success status icons
- Single-click to populate command bar
- Double-click to execute immediately
- Right-click export to do-file
"""

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal

class HistoryPane(QtWidgets.QWidget):
    command_selected = Signal(str)
    command_double_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history_records = []
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Search filter bar
        self.search_box = QtWidgets.QLineEdit()
        self.search_box.setPlaceholderText("Filter history")
        self.search_box.setAccessibleName("Filter command history")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.setFixedHeight(28)
        self.search_box.textChanged.connect(self._filter_history)
        layout.addWidget(self.search_box)

        # List Widget
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget)

    def add_entry(self, cmd_str, ok=True, return_code=0):
        idx = len(self.history_records) + 1
        record = {'idx': idx, 'cmd': cmd_str, 'ok': ok, 'rc': return_code}
        self.history_records.append(record)
        if self.search_box.text().strip().lower() in cmd_str.lower():
            self._append_list_item(record)

    def _append_list_item(self, record):
        icon_str = "✔" if record['ok'] else "✖"
        item_text = f"{record['idx']:>3} {icon_str}  {record['cmd']}"
        item = QtWidgets.QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, record['cmd'])
        self.list_widget.addItem(item)
        self.list_widget.scrollToBottom()

    def _filter_history(self, text):
        query = text.strip().lower()
        self.list_widget.clear()
        for rec in self.history_records:
            if not query or query in rec['cmd'].lower():
                self._append_list_item(rec)

    def _on_item_clicked(self, item):
        cmd = item.data(Qt.ItemDataRole.UserRole)
        if cmd:
            self.command_selected.emit(cmd)

    def _on_item_double_clicked(self, item):
        cmd = item.data(Qt.ItemDataRole.UserRole)
        if cmd:
            self.command_double_clicked.emit(cmd)

    def _show_context_menu(self, pos):
        item = self.list_widget.itemAt(pos)
        menu = QtWidgets.QMenu(self)
        if item:
            cmd = item.data(Qt.ItemDataRole.UserRole)
            act_copy = menu.addAction("Copy Command")
            act_copy.triggered.connect(lambda: QtWidgets.QApplication.clipboard().setText(cmd))

            act_run = menu.addAction("Run Command")
            act_run.triggered.connect(lambda: self.command_double_clicked.emit(cmd))
            menu.addSeparator()

        act_copy_all = menu.addAction("Copy All Commands to Clipboard")
        act_copy_all.triggered.connect(self._copy_all_commands)

        menu.exec(self.list_widget.mapToGlobal(pos))

    def _copy_all_commands(self):
        script_text = "\n".join(r['cmd'] for r in self.history_records)
        QtWidgets.QApplication.clipboard().setText(script_text)
