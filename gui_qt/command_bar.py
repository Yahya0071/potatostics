"""
gui_qt/command_bar.py - Intelligent Command Bar with History & Variable Autocomplete
Provides:
- Command prompt prefix (.)
- History recall via Up and Down arrow keys
- Real-time QCompleter autocompleting built-in commands and active dataset variable names
- Instant submit and keyboard shortcut execution
"""

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, Signal

BUILTIN_COMMANDS = [
    "summarize", "regress", "logit", "logistic", "ivregress 2sls", "xtreg", "xtset", "tsset",
    "merge 1:1", "merge m:1", "merge 1:m", "append", "reshape long", "reshape wide",
    "collapse", "egen", "encode", "decode", "label define", "label values", "label variable",
    "recode", "duplicates report", "duplicates tag", "duplicates drop",
    "tabulate", "correlate", "count", "list", "generate", "replace", "drop", "keep", "sort",
    "test", "predict", "estimates store", "esttab", "putdocx",
    "estat vif", "estat hettest", "estat ovtest", "estat dwatson",
    "dfuller", "corrgram", "arima", "did", "psmatch", "help", "clear", "describe"
]

class CommandLineEdit(QtWidgets.QLineEdit):
    """Subclassed QLineEdit to handle Up/Down arrow history recall cleanly."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []
        self.history_idx = -1

    def add_history(self, cmd_text):
        cmd = cmd_text.strip()
        if cmd and (not self.history or self.history[-1] != cmd):
            self.history.append(cmd)
        self.history_idx = len(self.history)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Up:
            if self.history and self.history_idx > 0:
                self.history_idx -= 1
                self.setText(self.history[self.history_idx])
            return
        elif event.key() == Qt.Key.Key_Down:
            if self.history and self.history_idx < len(self.history) - 1:
                self.history_idx += 1
                self.setText(self.history[self.history_idx])
            elif self.history_idx >= len(self.history) - 1:
                self.history_idx = len(self.history)
                self.clear()
            return
        super().keyPressEvent(event)


class CommandBar(QtWidgets.QWidget):
    command_submitted = Signal(str)
    cancel_requested = Signal()

    def __init__(self, dataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 8)
        layout.setSpacing(8)

        self.setObjectName("commandBar")
        lbl_prompt = QtWidgets.QLabel("Command")
        layout.addWidget(lbl_prompt)

        # Line edit with autocomplete
        self.entry = CommandLineEdit()
        lbl_prompt.setBuddy(self.entry)
        self.entry.setAccessibleName("Command")
        self.entry.setPlaceholderText("Enter a command…")
        self.entry.setToolTip("Enter to run · Up and Down to recall commands · Ctrl+L to focus")
        self.entry.setFixedHeight(34)
        self.entry.returnPressed.connect(self._handle_submit)
        layout.addWidget(self.entry, stretch=1)

        # Completer
        self.completer_model = QtCore.QStringListModel(BUILTIN_COMMANDS)
        self.completer = QtWidgets.QCompleter(self.completer_model, self)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.entry.setCompleter(self.completer)

        # Submit button
        self.btn_submit = QtWidgets.QPushButton("Run")
        self.btn_submit.setFixedHeight(34)
        self.btn_submit.setToolTip("Run command (Enter)")
        self.btn_submit.clicked.connect(self._handle_submit)
        layout.addWidget(self.btn_submit)

        # Cancel button
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setFixedHeight(34)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.hide()
        self.btn_cancel.clicked.connect(self.cancel_requested.emit)
        layout.addWidget(self.btn_cancel)

    def set_running_state(self, running: bool):
        self.entry.setEnabled(not running)
        self.btn_submit.setEnabled(not running)
        self.btn_submit.setVisible(not running)
        self.btn_cancel.setEnabled(running)
        self.btn_cancel.setVisible(running)

    def update_autocomplete(self):
        words = list(BUILTIN_COMMANDS)
        if not self.dataset.is_empty:
            words.extend(list(self.dataset.df.columns))
        self.completer_model.setStringList(words)

    def set_command_text(self, text):
        self.entry.setText(text)
        self.entry.setFocus()

    def _handle_submit(self):
        cmd = self.entry.text().strip()
        if not cmd:
            return
        self.entry.add_history(cmd)
        self.entry.clear()
        self.command_submitted.emit(cmd)
