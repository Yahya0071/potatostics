"""
gui_qt/package_dialog.py - Community Package & Plugin Hub Dialog
Provides an approachable, self-explanatory GUI to:
1. Browse, preview, install, and run examples for community econometric packages.
2. Manage, enable, disable, inspect, and uninstall local plugins (including local .py & HTTPS URLs).
3. Author new custom Python commands with starter templates.
4. Learn how plugins and `pkg` commands work via a built-in step-by-step guide.
"""

import ast
import hashlib
import os
from PySide6 import QtCore, QtGui, QtWidgets
from engine.plugins_manager import COMMUNITY_CATALOG


class PackageManagerDialog(QtWidgets.QDialog):
    command_pasted = QtCore.Signal(str)

    def __init__(self, interpreter, parent=None):
        super().__init__(parent)
        self.interpreter = interpreter
        self.pm = interpreter.plugins_mgr
        self.setWindowTitle("Community Package Hub & Plugins (pkg)")
        self.resize(840, 620)
        self._build_ui()
        self.refresh_installed()
        self._populate_catalog()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Beginner-friendly Quick-Start Banner
        banner = QtWidgets.QFrame()
        banner.setObjectName("pkgGuideBanner")
        banner.setStyleSheet(
            "QFrame#pkgGuideBanner { border: 1px solid #3E4451; border-radius: 6px; padding: 6px; }"
        )
        b_lay = QtWidgets.QVBoxLayout(banner)
        b_lay.setContentsMargins(10, 8, 10, 8)
        lbl_banner = QtWidgets.QLabel(
            "<b>What is the Community Package Hub?</b> Add-on statistical commands written in Python that work "
            "just like built-in Potatostics commands.<br>"
            "<b>How to use in 3 steps:</b> "
            "<b>1.</b> Pick a package below and click <b>Install</b> &rarr; "
            "<b>2.</b> <b>Enable</b> it (so it becomes active) &rarr; "
            "<b>3.</b> Run its command in the <b>Command Bar</b> (e.g. <code>winsorize price, cuts(1 99)</code>)."
        )
        lbl_banner.setWordWrap(True)
        b_lay.addWidget(lbl_banner)
        layout.addWidget(banner)

        self.tab_widget = QtWidgets.QTabWidget()
        layout.addWidget(self.tab_widget, stretch=1)

        # Tab 1: Explore Community Library
        self.tab_explore = QtWidgets.QWidget()
        self._build_explore_tab()
        self.tab_widget.addTab(self.tab_explore, "🌐 Explore Packages")

        # Tab 2: Installed Plugins
        self.tab_installed = QtWidgets.QWidget()
        self._build_installed_tab()
        self.tab_widget.addTab(self.tab_installed, "📦 Installed Plugins")

        # Tab 3: Author New Plugin
        self.tab_author = QtWidgets.QWidget()
        self._build_author_tab()
        self.tab_widget.addTab(self.tab_author, "✍️ Create Custom Command")

        # Tab 4: Step-by-Step Help & Guide
        self.tab_help = QtWidgets.QWidget()
        self._build_help_tab()
        self.tab_widget.addTab(self.tab_help, "❓ How It Works")

        # Bottom Close button
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    # ------------------------------------------------------------------
    # Tab 1: Explore Community Packages
    # ------------------------------------------------------------------
    def _build_explore_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_explore)
        layout.setSpacing(8)

        search_layout = QtWidgets.QHBoxLayout()
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Search packages by name or topic (e.g. winsorize, difference-in-differences)...")
        self.search_input.textChanged.connect(self._filter_explore)
        btn_search = QtWidgets.QPushButton("Search")
        btn_search.clicked.connect(self._filter_explore)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(btn_search)
        layout.addLayout(search_layout)

        self.catalog_table = QtWidgets.QTableWidget()
        self.catalog_table.setColumnCount(5)
        self.catalog_table.setHorizontalHeaderLabels(["Package", "Status", "Version", "Author", "Description"])
        self.catalog_table.horizontalHeader().setStretchLastSection(True)
        self.catalog_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.catalog_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.catalog_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.catalog_table.itemSelectionChanged.connect(self._on_catalog_selection_changed)
        layout.addWidget(self.catalog_table, stretch=1)

        # Selected Package Details & Usage Card
        self.details_box = QtWidgets.QGroupBox("Selected Package Details & How to Run It")
        d_lay = QtWidgets.QVBoxLayout(self.details_box)
        d_lay.setSpacing(6)

        self.lbl_pkg_when = QtWidgets.QLabel("Select a package above to see what it does and an example command.")
        self.lbl_pkg_when.setWordWrap(True)
        d_lay.addWidget(self.lbl_pkg_when)

        ex_row = QtWidgets.QHBoxLayout()
        ex_row.addWidget(QtWidgets.QLabel("Example Command:"))
        self.edt_pkg_example = QtWidgets.QLineEdit()
        self.edt_pkg_example.setReadOnly(True)
        self.edt_pkg_example.setStyleSheet("font-family: 'Consolas', monospace; font-weight: bold;")
        ex_row.addWidget(self.edt_pkg_example, stretch=1)

        self.btn_paste_example = QtWidgets.QPushButton("📋 Paste to Command Bar")
        self.btn_paste_example.setToolTip("Copies this example command into the main Command Bar")
        self.btn_paste_example.clicked.connect(self._paste_selected_example)
        ex_row.addWidget(self.btn_paste_example)
        d_lay.addLayout(ex_row)

        layout.addWidget(self.details_box)

        # Action bar
        act_layout = QtWidgets.QHBoxLayout()
        self.btn_preview_cat = QtWidgets.QPushButton("👁️ Preview Source Code")
        self.btn_preview_cat.clicked.connect(self._preview_selected_from_catalog)
        self.btn_install_cat = QtWidgets.QPushButton("⬇️ Install Selected Package")
        self.btn_install_cat.clicked.connect(self._install_selected_from_catalog)
        act_layout.addWidget(self.btn_preview_cat)
        act_layout.addStretch()
        act_layout.addWidget(self.btn_install_cat)
        layout.addLayout(act_layout)

    def _package_status_label(self, pkg_name):
        installed = {p["name"]: p for p in self.pm.list_plugins()}
        if pkg_name not in installed:
            return "Available"
        return installed[pkg_name]["status"]

    def _populate_catalog(self, query=""):
        query = query.strip().lower()
        items = []
        for name, info in COMMUNITY_CATALOG.items():
            haystack = " ".join([
                name,
                info.get("description", ""),
                info.get("when_to_use", ""),
                info.get("author", ""),
            ]).lower()
            if not query or query in haystack:
                items.append(info)

        self.catalog_table.setRowCount(len(items))
        for row, info in enumerate(items):
            name = info["name"]
            self.catalog_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.catalog_table.setItem(row, 1, QtWidgets.QTableWidgetItem(self._package_status_label(name)))
            self.catalog_table.setItem(row, 2, QtWidgets.QTableWidgetItem(info.get("version", "1.0")))
            self.catalog_table.setItem(row, 3, QtWidgets.QTableWidgetItem(info.get("author", "Community")))
            self.catalog_table.setItem(row, 4, QtWidgets.QTableWidgetItem(info.get("description", "")))

        if items and self.catalog_table.currentRow() < 0:
            self.catalog_table.selectRow(0)
        else:
            self._on_catalog_selection_changed()

    def _on_catalog_selection_changed(self):
        row = self.catalog_table.currentRow()
        if row < 0 or self.catalog_table.item(row, 0) is None:
            if not COMMUNITY_CATALOG:
                self.lbl_pkg_when.setText(
                    "No community packages are pre-loaded. Use <b>Install .py File...</b> or "
                    "<b>Install from HTTPS URL...</b> in <b>Installed Plugins</b>, or build your own in <b>Create Custom Command</b>."
                )
            else:
                self.lbl_pkg_when.setText("Select a package above to see what it does and an example command.")
            self.edt_pkg_example.setText("")
            self.btn_paste_example.setEnabled(False)
            self.btn_preview_cat.setEnabled(False)
            self.btn_install_cat.setEnabled(False)
            return

        pkg_name = self.catalog_table.item(row, 0).text()
        info = COMMUNITY_CATALOG.get(pkg_name, {})
        when_txt = info.get("when_to_use") or info.get("description", "")
        syntax_txt = info.get("syntax", pkg_name)
        example_txt = info.get("example", pkg_name)

        self.lbl_pkg_when.setText(f"<b>When to use:</b> {when_txt}<br><b>Syntax:</b> <code>{syntax_txt}</code>")
        self.edt_pkg_example.setText(example_txt)
        self.btn_paste_example.setEnabled(True)
        self.btn_preview_cat.setEnabled(True)
        self.btn_install_cat.setEnabled(True)

    def _paste_selected_example(self):
        cmd = self.edt_pkg_example.text().strip()
        if not cmd:
            return
        self.command_pasted.emit(cmd)
        QtWidgets.QMessageBox.information(
            self,
            "Command Ready",
            f"Pasted into the Command Bar:\n\n{cmd}\n\n"
            "Make sure the plugin is Enabled in 'Installed Plugins', then press Enter in the Command Bar to run it."
        )

    def _preview_selected_from_catalog(self):
        row = self.catalog_table.currentRow()
        if row < 0:
            return
        pkg_name = self.catalog_table.item(row, 0).text()
        info = COMMUNITY_CATALOG.get(pkg_name, {})
        code = info.get("code", "")
        sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
        self._show_code_viewer(f"Source Preview: {pkg_name}", code, f"SHA-256: {sha}")

    def _show_code_viewer(self, title, code, subtitle=""):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(640, 440)
        lay = QtWidgets.QVBoxLayout(dlg)
        if subtitle:
            lbl = QtWidgets.QLabel(subtitle)
            lbl.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            lay.addWidget(lbl)
        txt = QtWidgets.QPlainTextEdit()
        txt.setReadOnly(True)
        txt.setFont(QtGui.QFont("Consolas", 10))
        txt.setPlainText(code)
        lay.addWidget(txt)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        dlg.exec()

    def _filter_explore(self):
        self._populate_catalog(self.search_input.text())

    def _install_selected_from_catalog(self):
        row = self.catalog_table.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.warning(self, "Selection Required", "Please select a package from the table.")
            return

        pkg_name = self.catalog_table.item(row, 0).text()
        info = COMMUNITY_CATALOG.get(pkg_name, {})
        code = info.get("code", "")
        code_bytes = code.encode("utf-8")
        sha = hashlib.sha256(code_bytes).hexdigest()
        size = len(code_bytes)

        confirm_msg = (
            f"Install community package '{pkg_name}'?\n\n"
            f"What it does: {info.get('description', '')}\n"
            f"Example command: {info.get('example', pkg_name)}\n\n"
            f"Safety check:\n"
            f"• Size: {size} bytes\n"
            f"• SHA-256: {sha[:16]}...\n"
            f"• Plugins run with your user permissions and are installed disabled first.\n\n"
            f"Do you want to install '{pkg_name}' now?"
        )
        res = QtWidgets.QMessageBox.question(
            self,
            f"Confirm Install: {pkg_name}",
            confirm_msg,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        if res != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        try:
            self.pm.install_plugin(pkg_name, enable=False)
            enable_choice = QtWidgets.QMessageBox.question(
                self,
                "Package Installed",
                f"'{pkg_name}' was installed safely (currently disabled).\n\n"
                f"Enable and activate it now so you can run '{info.get('example', pkg_name)}'?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            )
            if enable_choice == QtWidgets.QMessageBox.StandardButton.Yes:
                self.pm.enable_plugin(pkg_name)
                self.pm.load_all_plugins()

            self.refresh_installed()
            self._populate_catalog(self.search_input.text())
            self.tab_widget.setCurrentIndex(1)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Installation Failed", str(e))

    # ------------------------------------------------------------------
    # Tab 2: Installed Plugins
    # ------------------------------------------------------------------
    def _build_installed_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_installed)
        layout.setSpacing(8)

        lbl_info = QtWidgets.QLabel(
            "<b>Loaded</b> plugins are active and ready to run in the Command Bar. "
            "<b>Disabled</b> plugins are saved on disk but will not run until you enable them."
        )
        lbl_info.setWordWrap(True)
        layout.addWidget(lbl_info)

        self.installed_table = QtWidgets.QTableWidget()
        self.installed_table.setColumnCount(4)
        self.installed_table.setHorizontalHeaderLabels(["Name", "Status", "Version", "File Location"])
        self.installed_table.horizontalHeader().setStretchLastSection(True)
        self.installed_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.installed_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.installed_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.installed_table, stretch=1)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_refresh = QtWidgets.QPushButton("🔄 Refresh")
        btn_refresh.clicked.connect(self.refresh_installed)
        btn_local = QtWidgets.QPushButton("📂 Install .py File...")
        btn_local.clicked.connect(self._install_from_local_file)
        btn_url = QtWidgets.QPushButton("🌐 Install from HTTPS URL...")
        btn_url.clicked.connect(self._install_from_https_url)
        btn_view = QtWidgets.QPushButton("👁️ View Code")
        btn_view.clicked.connect(self._view_installed_code)
        self.btn_toggle = QtWidgets.QPushButton("✅ Enable / Disable")
        self.btn_toggle.clicked.connect(self._toggle_installed)
        self.btn_uninstall = QtWidgets.QPushButton("🗑️ Uninstall")
        self.btn_uninstall.clicked.connect(self._uninstall_installed)

        btn_layout.addWidget(btn_refresh)
        btn_layout.addWidget(btn_local)
        btn_layout.addWidget(btn_url)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_view)
        btn_layout.addWidget(self.btn_toggle)
        btn_layout.addWidget(self.btn_uninstall)
        layout.addLayout(btn_layout)

    def refresh_installed(self):
        plugins = self.pm.list_plugins()
        self.installed_table.setRowCount(len(plugins))
        for row, p in enumerate(plugins):
            self.installed_table.setItem(row, 0, QtWidgets.QTableWidgetItem(p["name"]))
            self.installed_table.setItem(row, 1, QtWidgets.QTableWidgetItem(p["status"]))
            self.installed_table.setItem(row, 2, QtWidgets.QTableWidgetItem(p.get("version", "1.0")))
            self.installed_table.setItem(row, 3, QtWidgets.QTableWidgetItem(p["path"]))

    def _install_from_local_file(self):
        fpath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Plugin Python File", "", "Python Files (*.py);;All Files (*)"
        )
        if not fpath:
            return
        try:
            res = self.pm.install_plugin(fpath, enable=False)
            self.refresh_installed()
            self._populate_catalog(self.search_input.text())
            QtWidgets.QMessageBox.information(
                self,
                "Plugin Installed (Disabled)",
                f"Installed '{res['name']}' from:\n{fpath}\n\n"
                f"SHA-256: {res['sha256']}\n\n"
                f"Review its code with 'View Code', then click 'Enable / Disable' when ready to use it."
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Install Failed", str(e))

    def _install_from_https_url(self):
        url, ok = QtWidgets.QInputDialog.getText(
            self,
            "Install Plugin from HTTPS URL",
            "Enter the https:// URL to a Potatostics plugin .py file:"
        )
        if not ok or not url.strip():
            return
        try:
            res = self.pm.install_plugin(url.strip(), enable=False)
            self.refresh_installed()
            self._populate_catalog(self.search_input.text())
            QtWidgets.QMessageBox.information(
                self,
                "Plugin Downloaded (Disabled)",
                f"Downloaded '{res['name']}' ({res['size']} bytes).\n"
                f"SHA-256: {res['sha256']}\n\n"
                f"Review its code with 'View Code', then click 'Enable / Disable' when ready to use it."
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Download Failed", str(e))

    def _view_installed_code(self):
        row = self.installed_table.currentRow()
        if row < 0:
            return
        name = self.installed_table.item(row, 0).text()
        path = self.installed_table.item(row, 3).text()
        try:
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
            sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
            self._show_code_viewer(f"Installed Plugin: {name}", code, f"Path: {path}  |  SHA-256: {sha}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Read Error", str(e))

    def _toggle_installed(self):
        row = self.installed_table.currentRow()
        if row < 0:
            return
        name = self.installed_table.item(row, 0).text()
        if self.pm.is_enabled(name):
            self.pm.disable_plugin(name)
        else:
            self.pm.enable_plugin(name)
            self.pm.load_all_plugins()
        self.refresh_installed()
        self._populate_catalog(self.search_input.text())

    def _uninstall_installed(self):
        row = self.installed_table.currentRow()
        if row < 0:
            return
        name = self.installed_table.item(row, 0).text()
        ans = QtWidgets.QMessageBox.question(
            self,
            "Confirm Uninstall",
            f"Are you sure you want to uninstall '{name}'?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        if ans == QtWidgets.QMessageBox.StandardButton.Yes:
            self.pm.uninstall_plugin(name)
            self.refresh_installed()
            self._populate_catalog(self.search_input.text())

    # ------------------------------------------------------------------
    # Tab 3: Create Custom Command
    # ------------------------------------------------------------------
    def _build_author_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_author)
        layout.setSpacing(8)

        lbl_intro = QtWidgets.QLabel(
            "Build your own Potatostics command in Python. Pick a command name (lowercase letters, numbers, "
            "and underscores only), choose a starter template, edit the code if needed, and click <b>Save &amp; Enable</b>."
        )
        lbl_intro.setWordWrap(True)
        layout.addWidget(lbl_intro)

        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addWidget(QtWidgets.QLabel("Command Name:"))
        self.name_edit = QtWidgets.QLineEdit("my_custom_stat")
        top_layout.addWidget(self.name_edit)

        top_layout.addWidget(QtWidgets.QLabel("Starter Template:"))
        self.cmb_template = QtWidgets.QComboBox()
        self.cmb_template.addItems([
            "Summary Statistic (reads a variable)",
            "Transform Column (creates a new variable)",
        ])
        self.cmb_template.currentIndexChanged.connect(self._generate_author_template)
        top_layout.addWidget(self.cmb_template)

        btn_gen = QtWidgets.QPushButton("Reset Template")
        btn_gen.clicked.connect(self._generate_author_template)
        top_layout.addWidget(btn_gen)
        layout.addLayout(top_layout)

        self.code_edit = QtWidgets.QPlainTextEdit()
        font = QtGui.QFont("Consolas", 10)
        self.code_edit.setFont(font)
        layout.addWidget(self.code_edit, stretch=1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_save = QtWidgets.QPushButton("💾 Save, Enable & Register Command")
        btn_save.clicked.connect(self._save_author_plugin)
        btn_paste = QtWidgets.QPushButton("📋 Paste Command Name to Command Bar")
        btn_paste.clicked.connect(lambda: self.command_pasted.emit(f"{self.name_edit.text().strip().lower()} "))
        btn_row.addWidget(btn_save, stretch=1)
        btn_row.addWidget(btn_paste)
        layout.addLayout(btn_row)

        self._generate_author_template()

    def _generate_author_template(self):
        name = self.name_edit.text().strip() or "my_command"
        is_transform = self.cmb_template.currentIndex() == 1
        if is_transform:
            code = f'''"""
Plugin: {name}
Author: User
Description: Creates a standardized z-score column: {name} <varname>
"""
from engine.parser import CommandResult

def {name}_cmd(interp, cmd):
    tokens = cmd.body.split()
    if not tokens:
        return CommandResult("syntax: {name} varname", ok=False, return_code=198)

    var = tokens[0]
    df = interp.dataset.df
    if var not in df.columns:
        return CommandResult(f"variable '{{var}}' not found", ok=False, return_code=111)

    new_col = f"{{var}}_z"
    sd = float(df[var].std())
    if sd == 0:
        return CommandResult(f"variable '{{var}}' has zero standard deviation", ok=False, return_code=198)

    df[new_col] = (df[var] - float(df[var].mean())) / sd
    return CommandResult(f"Generated standardized column '{{new_col}}' via {name}.", ok=True)

def register_plugin(interp):
    interp.register_command("{name}", {name}_cmd)
'''
        else:
            code = f'''"""
Plugin: {name}
Author: User
Description: Custom econometric routine for Potatostics.
"""
import numpy as np
import pandas as pd
from engine.parser import CommandResult

def {name}_cmd(interp, cmd):
    """
    Handler for '{name}'.
    Access dataset via interp.dataset.df.
    Return a CommandResult(text, ok=True) or string.
    """
    tokens = cmd.body.split()
    if not tokens:
        return CommandResult("syntax: {name} varlist [, option(val)]", ok=False, return_code=198)

    var = tokens[0]
    df = interp.dataset.df
    if var not in df.columns:
        return CommandResult(f"variable '{{var}}' not found", ok=False, return_code=111)

    mean_val = float(df[var].mean())
    interp.registers.clear_r()
    interp.registers.set_r_scalar("mean", mean_val)

    return CommandResult(f"{name}: Mean of {{var}} = {{mean_val:.4f}}", ok=True)

def register_plugin(interp):
    interp.register_command("{name}", {name}_cmd)
'''
        self.code_edit.setPlainText(code)

    def _save_author_plugin(self):
        name = self.name_edit.text().strip().lower()
        if not name.isidentifier():
            QtWidgets.QMessageBox.warning(self, "Invalid Name", "Command name must be a valid Python identifier.")
            return

        code = self.code_edit.toPlainText()
        try:
            ast.parse(code)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Python Syntax Error", f"Cannot save: code has syntax errors:\n{e}")
            return

        try:
            target_name = self.pm.validate_plugin_name(name)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Invalid Name", str(e))
            return

        target_file = os.path.join(self.pm.plugins_dir, f"{target_name}.py")
        try:
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(code)
            self.pm.enable_plugin(target_name)
            self.pm.load_all_plugins()
            QtWidgets.QMessageBox.information(
                self,
                "Success",
                f"Command '{target_name}' was saved, enabled, and registered!\n\n"
                f"You can now type '{target_name} <variable>' in the Command Bar."
            )
            self.refresh_installed()
            self._populate_catalog(self.search_input.text())
            self.tab_widget.setCurrentIndex(1)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Failed", str(e))

    # ------------------------------------------------------------------
    # Tab 4: How It Works (Beginner's Guide)
    # ------------------------------------------------------------------
    def _build_help_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_help)
        browser = QtWidgets.QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml("""
        <h3>Community Package Hub — Quick Guide</h3>
        <p><b>What is a plugin?</b><br>
        A plugin is a small Python file (<code>.py</code>) that adds a new command to Potatostics.
        Once installed and enabled, you run it from the Command Bar just like <code>regress</code> or <code>summarize</code>.</p>

        <h4>Step-by-Step Workflow</h4>
        <ol>
          <li><b>Explore &amp; Install:</b> Open the <b>Explore Packages</b> tab, click a package (for example <code>winsorize</code>), read <i>When to use</i>, and click <b>Install Selected Package</b>.</li>
          <li><b>Enable:</b> When asked, click <b>Yes</b> to enable it (or enable/disable it later in <b>Installed Plugins</b>).</li>
          <li><b>Run It:</b> Click <b>Paste to Command Bar</b> (or type the command yourself) and press <b>Enter</b>.</li>
        </ol>

        <h4>Command-Bar Equivalents (Optional)</h4>
        <pre>pkg search winsor          // Find packages in the catalog
pkg install winsorize      // Install safely (disabled by default)
pkg enable winsorize       // Activate the command
winsorize price, cuts(1 99)// Run the installed command
pkg list                   // Show installed plugins and status
pkg disable winsorize      // Deactivate without deleting
pkg uninstall winsorize    // Remove completely
plugin new my_stat         // Create a starter plugin template</pre>

        <h4>Safety &amp; Trust</h4>
        <p>Plugins run Python code with your user permissions. For safety, <b>new packages are installed disabled by default</b>
        and never run during download or installation. You can inspect any plugin's code with <b>View Code</b> before enabling it.</p>
        """)
        layout.addWidget(browser)
