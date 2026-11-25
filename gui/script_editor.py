"""
gui/script_editor.py - Integrated Script Editor Window
Provides a dedicated script environment with line numbers, file management,
Run Selection (Ctrl+D), Run All (Ctrl+R), and Dark/Light theme support.
"""

import os
import tkinter as tk
from tkinter import filedialog, ttk
import customtkinter as ctk
from gui.branding import set_window_icon

class ScriptEditorWindow(ctk.CTkToplevel):
    def __init__(self, parent, run_command_callback):
        super().__init__(parent)
        self.title("Script Editor - Potatostics")
        self.geometry("800x600")
        self.minsize(550, 400)
        set_window_icon(self)
        self.run_command_callback = run_command_callback
        self.current_file = None

        self._build_ui()

    def _build_ui(self):
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        tb_bg = "#2B2D30" if is_dark else "#E4E4E4"
        ed_bg = "#1E1E1E" if is_dark else "#FFFFFF"
        ed_fg = "#D4D4D4" if is_dark else "#1E1E1E"
        ln_bg = "#252526" if is_dark else "#F0F0F0"
        ln_fg = "#858585" if is_dark else "#888888"
        ins_bg = "#FFFFFF" if is_dark else "#000000"
        status_fg = "#888888" if is_dark else "#555555"

        # 1. Toolbar
        toolbar = ctk.CTkFrame(self, height=40, corner_radius=0, fg_color=tb_bg)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        btn_new = ctk.CTkButton(toolbar, text="📄 New", width=70, height=26, command=self._new_file)
        btn_new.pack(side=tk.LEFT, padx=4, pady=6)

        btn_open = ctk.CTkButton(toolbar, text="📂 Open", width=70, height=26, command=self._open_file)
        btn_open.pack(side=tk.LEFT, padx=3, pady=6)

        btn_save = ctk.CTkButton(toolbar, text="💾 Save", width=70, height=26, command=self._save_file)
        btn_save.pack(side=tk.LEFT, padx=3, pady=6)

        # Separator
        sep = ctk.CTkFrame(toolbar, width=2, height=24, fg_color="#555555" if is_dark else "#CCCCCC")
        sep.pack(side=tk.LEFT, padx=6, pady=6)

        btn_run_sel = ctk.CTkButton(toolbar, text="▶ Run Selection (Ctrl+D)", width=160, height=26,
                                    fg_color="#2D5A27", hover_color="#1E3F1A", command=self._run_selection)
        btn_run_sel.pack(side=tk.LEFT, padx=4, pady=6)

        btn_run_all = ctk.CTkButton(toolbar, text="⏩ Run All (Ctrl+R)", width=130, height=26,
                                    fg_color="#0E639C", hover_color="#1177BB", command=self._run_all)
        btn_run_all.pack(side=tk.LEFT, padx=4, pady=6)

        # Status Bar at bottom
        self.status_bar = ctk.CTkLabel(self, text="Ready | Untitled", font=("Segoe UI", 10),
                                       text_color=status_fg, anchor="w")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=2)

        # Editor Area: Line Numbers + Text Editor
        editor_frame = ttk.Frame(self)
        editor_frame.pack(fill=tk.BOTH, expand=True)

        self.line_numbers = tk.Text(editor_frame, width=5, bg=ln_bg, fg=ln_fg,
                                    font=("Consolas", 11), state="disabled", bd=0, padx=4, pady=6)
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)

        self.text_editor = tk.Text(editor_frame, bg=ed_bg, fg=ed_fg, insertbackground=ins_bg,
                                   font=("Consolas", 11), wrap="none", undo=True, bd=0, padx=6, pady=6)
        self.text_editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(editor_frame, orient="vertical", command=self._on_scroll)
        self.text_editor.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Keybindings
        self.text_editor.bind("<KeyRelease>", self._update_line_numbers)
        self.text_editor.bind("<MouseWheel>", self._update_line_numbers)
        self.bind("<Control-d>", lambda e: self._run_selection())
        self.bind("<Control-r>", lambda e: self._run_all())
        self.bind("<Control-s>", lambda e: self._save_file())

        # Sample template
        template = """// Potatostics Script
// Load sample dataset
use sample_data.csv, clear

// Summarize and run OLS regression
summarize price mpg weight
regress price mpg weight, robust

// Post-estimation diagnostics
estat vif
estat hettest
"""
        self.text_editor.insert(tk.END, template)
        self._update_line_numbers()

    def _on_scroll(self, *args):
        self.text_editor.yview(*args)
        self.line_numbers.yview(*args)

    def _update_line_numbers(self, event=None):
        lines = self.text_editor.get("1.0", tk.END).count("\n")
        line_str = "\n".join(str(i) for i in range(1, lines + 1))
        self.line_numbers.configure(state="normal")
        self.line_numbers.delete("1.0", tk.END)
        self.line_numbers.insert("1.0", line_str)
        self.line_numbers.configure(state="disabled")

    def _new_file(self):
        self.text_editor.delete("1.0", tk.END)
        self.current_file = None
        self.status_bar.configure(text="Ready | Untitled")
        self._update_line_numbers()

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open Script File",
            filetypes=[("Script Files (*.do, *.txt, *.py)", "*.do *.txt *.py"), ("All Files", "*.*")]
        )
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            self.text_editor.delete("1.0", tk.END)
            self.text_editor.insert(tk.END, content)
            self.current_file = path
            self.status_bar.configure(text=f"Ready | {os.path.basename(path)}")
            self._update_line_numbers()

    def _save_file(self):
        if not self.current_file:
            path = filedialog.asksaveasfilename(
                title="Save Script File",
                defaultextension=".do",
                filetypes=[("Script Files (*.do)", "*.do"), ("Text Files (*.txt)", "*.txt"), ("All Files", "*.*")]
            )
            if not path:
                return
            self.current_file = path
            
        with open(self.current_file, "w", encoding="utf-8") as f:
            f.write(self.text_editor.get("1.0", tk.END))
        self.status_bar.configure(text=f"Saved | {os.path.basename(self.current_file)}")

    def _run_selection(self):
        try:
            sel = self.text_editor.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            line_start = self.text_editor.index("insert linestart")
            line_end = self.text_editor.index("insert lineend")
            sel = self.text_editor.get(line_start, line_end)
            
        if sel:
            lines = sel.split("\n")
            for line in lines:
                l = line.strip()
                if l and not l.startswith("*") and not l.startswith("//"):
                    self.run_command_callback(l)

    def _run_all(self):
        all_code = self.text_editor.get("1.0", tk.END)
        lines = all_code.split("\n")
        for line in lines:
            l = line.strip()
            if l and not l.startswith("*") and not l.startswith("//"):
                self.run_command_callback(l)
