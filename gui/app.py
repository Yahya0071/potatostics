"""
gui/app.py - Potatostics Primary Desktop Application Window
Multi-pane desktop IDE for statistical analysis:
- Full Top Menu Bar (File, Edit, View/Theme, Data, Graphics, Statistics, Tools & AI, Help)
- Dynamic Dark / Light Theme Switching
- Top Action Toolbar with Quick Actions & Theme Switcher
- Left: Command Review / History
- Center: Output / Results Console with Monospace Colored Formatting
- Right: Variables & Properties Manager
- Bottom: Command Input Bar with History Recall
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from gui.data_editor import DataEditorWindow
from gui.plot_window import PlotWindow
from gui.script_editor import ScriptEditorWindow
from gui.dialogs import (
    RegressionDialog, LogitDialog, IVDialog, PanelDialog,
    DiagnosticsDialog, TimeSeriesDialog, CausalDialog,
    SummarizeDialog, TabulateDialog, ScatterDialog,
    HistogramDialog, BoxplotDialog, GenerateDialog,
    ReportDialog, AICopilotDialog
)
from gui.branding import set_window_icon, get_brand_icon_image, AboutDialog

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class PotatosticsApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Potatostics - Statistical Analysis Software")
        self.geometry("1300x820")
        self.minsize(1040, 660)
        set_window_icon(self)
        
        self.current_theme = "Dark"
        self.dataset = Dataset()
        self.interpreter = Interpreter(self.dataset, plot_callback=self._render_plot)
        self.cmd_history = []
        self.history_index = -1
        
        self._build_menu()
        self._build_ui()
        self._apply_theme("Dark")
        self._print_welcome()

    def _build_menu(self):
        """Top Application Menu Bar (File, Edit, View, Data, Graphics, Statistics, Tools, Help)"""
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # 1. File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Dataset... (Ctrl+O)", command=self._open_file_dialog)
        file_menu.add_command(label="Save CSV (Ctrl+S)", command=self._save_file_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Script Editor... (Ctrl+E)", command=self._open_script_editor)
        file_menu.add_command(label="Generate Word Report... (Ctrl+P)", command=self._open_report_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Clear Dataset", command=self._clear_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)

        # 2. Edit Menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear Results Window", command=self._clear_results)
        edit_menu.add_command(label="Copy", command=lambda: self.focus_get().event_generate("<<Copy>>"))
        edit_menu.add_command(label="Select All", command=lambda: self.output_box.tag_add("sel", "1.0", "end"))

        # 3. View / Appearance Menu (Theme Switcher)
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="🌙 Dark Theme", command=lambda: self._apply_theme("Dark"))
        view_menu.add_command(label="☀️ Light Theme", command=lambda: self._apply_theme("Light"))
        view_menu.add_separator()
        view_menu.add_command(label="Toggle Dark/Light (F2)", command=self.toggle_theme)

        # 4. Data Menu
        data_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Data", menu=data_menu)
        data_menu.add_command(label="Data Editor (Spreadsheet)...", command=self._open_data_editor)
        data_menu.add_separator()
        data_menu.add_command(label="Describe data in memory", command=lambda: self.run_command("describe"))
        
        create_menu = tk.Menu(data_menu, tearoff=0)
        data_menu.add_cascade(label="Create or change data", menu=create_menu)
        create_menu.add_command(label="Create new variable (generate)...", command=self._open_generate_dialog)
        create_menu.add_command(label="Change contents of variable (replace)...", command=lambda: self._prompt_command("replace var = expr if condition"))
        
        data_menu.add_separator()
        data_menu.add_command(label="Sort data...", command=lambda: self._prompt_command("sort varlist"))

        # 5. Graphics Menu
        graphics_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Graphics", menu=graphics_menu)
        graphics_menu.add_command(label="Two-way scatter plot (with OLS fit)...", command=self._open_scatter_dialog)
        graphics_menu.add_command(label="Histogram...", command=self._open_histogram_dialog)
        graphics_menu.add_command(label="Box plot...", command=self._open_boxplot_dialog)
        graphics_menu.add_command(label="Correlation heatmap...", command=lambda: self.run_command("heatmap"))

        # 6. Statistics Menu
        stats_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Statistics", menu=stats_menu)
        
        # 6a. Regression models
        linear_menu = tk.Menu(stats_menu, tearoff=0)
        stats_menu.add_cascade(label="Regression models", menu=linear_menu)
        linear_menu.add_command(label="Linear regression (OLS)...", command=self._open_regression_dialog)
        linear_menu.add_command(label="Logistic regression (Logit)...", command=self._open_logit_dialog)
        linear_menu.add_command(label="Instrumental variables (2SLS)...", command=self._open_iv_dialog)
        linear_menu.add_command(label="Panel fixed effects (xtreg)...", command=self._open_panel_dialog)
        
        # 6b. Post-Estimation Diagnostics
        diag_menu = tk.Menu(stats_menu, tearoff=0)
        stats_menu.add_cascade(label="Post-estimation diagnostics", menu=diag_menu)
        diag_menu.add_command(label="Specification & Diagnostics (VIF, Hettest, RESET, DW)...", command=self._open_diagnostics_dialog)
        diag_menu.add_command(label="Variance Inflation Factors (estat vif)", command=lambda: self.run_command("estat vif"))
        diag_menu.add_command(label="Breusch-Pagan Heteroskedasticity (estat hettest)", command=lambda: self.run_command("estat hettest"))
        diag_menu.add_command(label="Ramsey RESET Test (estat ovtest)", command=lambda: self.run_command("estat ovtest"))
        diag_menu.add_command(label="Durbin-Watson Autocorrelation (estat dwatson)", command=lambda: self.run_command("estat dwatson"))
        diag_menu.add_command(label="Linear Wald test...", command=lambda: self._prompt_command("test var1 = var2"))

        # 6c. Time Series Analysis
        ts_menu = tk.Menu(stats_menu, tearoff=0)
        stats_menu.add_cascade(label="Time series", menu=ts_menu)
        ts_menu.add_command(label="Time series studio (ADF, Correlogram, ARIMA)...", command=self._open_timeseries_dialog)
        ts_menu.add_command(label="Augmented Dickey-Fuller unit root (dfuller)...", command=lambda: self._prompt_command("dfuller varname"))
        ts_menu.add_command(label="Correlogram ACF/PACF (corrgram)...", command=lambda: self._prompt_command("corrgram varname"))
        ts_menu.add_command(label="ARIMA forecasting...", command=lambda: self._prompt_command("arima depvar, ar(1) d(0) ma(0) steps(5)"))

        # 6d. Causal Inference
        causal_menu = tk.Menu(stats_menu, tearoff=0)
        stats_menu.add_cascade(label="Causal inference", menu=causal_menu)
        causal_menu.add_command(label="Causal studio (DiD & PSMatch)...", command=self._open_causal_dialog)
        causal_menu.add_command(label="Difference-in-Differences (did)...", command=lambda: self._prompt_command("did depvar treatvar timevar"))
        causal_menu.add_command(label="Propensity Score Matching (psmatch)...", command=lambda: self._prompt_command("psmatch treatvar cov1 cov2, outcome(depvar)"))

        # 6e. Summaries and tables
        sum_menu = tk.Menu(stats_menu, tearoff=0)
        stats_menu.add_cascade(label="Summaries, tables, tests", menu=sum_menu)
        sum_menu.add_command(label="Summary statistics...", command=self._open_summarize_dialog)
        sum_menu.add_command(label="Frequency tables (tabulate)...", command=self._open_tabulate_dialog)
        sum_menu.add_command(label="Correlations (correlate)...", command=lambda: self._prompt_command("correlate varlist"))
        sum_menu.add_command(label="Compare stored models (esttab)...", command=lambda: self.run_command("esttab"))

        # 7. Tools & AI Menu
        user_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools & AI", menu=user_menu)
        user_menu.add_command(label="AI Econometric Copilot (Ctrl+Shift+A)...", command=self._open_copilot_dialog)
        user_menu.add_command(label="Generate Word Report (.docx)...", command=self._open_report_dialog)
        user_menu.add_separator()
        user_menu.add_command(label="Script Editor... (Ctrl+E)", command=self._open_script_editor)
        user_menu.add_command(label="Execute Script File...", command=self._open_do_file_dialog)

        # 8. Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Command Documentation", command=lambda: self.run_command("help"))
        help_menu.add_separator()
        help_menu.add_command(label="About Potatostics", command=self._show_about)

        # Keyboard shortcuts
        self.bind("<Control-o>", lambda e: self._open_file_dialog())
        self.bind("<Control-s>", lambda e: self._save_file_dialog())
        self.bind("<Control-e>", lambda e: self._open_script_editor())
        self.bind("<Control-p>", lambda e: self._open_report_dialog())
        self.bind("<Control-Shift-A>", lambda e: self._open_copilot_dialog())
        self.bind("<F2>", lambda e: self.toggle_theme())

    def _build_ui(self):
        # 1. Top Quick Action Toolbar
        self.toolbar = ctk.CTkFrame(self, height=44, corner_radius=0, fg_color="#2B2D30")
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        
        # Brand logo / mark on the far left of toolbar
        self.brand_icon = get_brand_icon_image(size=(24, 24))
        if self.brand_icon:
            brand_frame = ctk.CTkFrame(self.toolbar, fg_color="transparent")
            brand_frame.pack(side=tk.LEFT, padx=(10, 8), pady=4)
            lbl_ico = ctk.CTkLabel(brand_frame, image=self.brand_icon, text="")
            lbl_ico.pack(side=tk.LEFT, padx=(0, 6))
            lbl_text = ctk.CTkLabel(brand_frame, text="POTATOSTICS", font=("Consolas", 13, "bold"), text_color=("#005FB8", "#4EC9B0"))
            lbl_text.pack(side=tk.LEFT)

        btn_open = ctk.CTkButton(self.toolbar, text="📂 Open Data", width=95, height=28, command=self._open_file_dialog)
        btn_open.pack(side=tk.LEFT, padx=3, pady=8)
        
        btn_browse = ctk.CTkButton(self.toolbar, text="📊 Data Editor", width=95, height=28, command=self._open_data_editor)
        btn_browse.pack(side=tk.LEFT, padx=3, pady=8)

        btn_script = ctk.CTkButton(self.toolbar, text="📝 Script Editor", width=100, height=28,
                                   fg_color="#0E639C", hover_color="#1177BB", command=self._open_script_editor)
        btn_script.pack(side=tk.LEFT, padx=3, pady=8)

        btn_reg = ctk.CTkButton(self.toolbar, text="📈 Regression", width=95, height=28,
                                fg_color="#2D5A27", hover_color="#1E3F1A", command=self._open_regression_dialog)
        btn_reg.pack(side=tk.LEFT, padx=3, pady=8)

        btn_diag = ctk.CTkButton(self.toolbar, text="🔍 Diagnostics", width=95, height=28,
                                 fg_color="#5D478B", hover_color="#483D8B", command=self._open_diagnostics_dialog)
        btn_diag.pack(side=tk.LEFT, padx=3, pady=8)

        btn_ts = ctk.CTkButton(self.toolbar, text="⏳ Time Series", width=95, height=28,
                               fg_color="#A25900", hover_color="#854800", command=self._open_timeseries_dialog)
        btn_ts.pack(side=tk.LEFT, padx=3, pady=8)

        btn_causal = ctk.CTkButton(self.toolbar, text="🎯 Causal", width=80, height=28,
                                  fg_color="#8B4513", hover_color="#70380E", command=self._open_causal_dialog)
        btn_causal.pack(side=tk.LEFT, padx=3, pady=8)

        btn_su = ctk.CTkButton(self.toolbar, text="🔢 Summarize", width=85, height=28, command=self._open_summarize_dialog)
        btn_su.pack(side=tk.LEFT, padx=3, pady=8)

        btn_plot = ctk.CTkButton(self.toolbar, text="📉 Plot", width=65, height=28, command=self._open_scatter_dialog)
        btn_plot.pack(side=tk.LEFT, padx=3, pady=8)

        btn_est = ctk.CTkButton(self.toolbar, text="📑 Compare", width=80, height=28, command=lambda: self.run_command("esttab"))
        btn_est.pack(side=tk.LEFT, padx=3, pady=8)
        
        btn_docx = ctk.CTkButton(self.toolbar, text="📄 Word Report", width=95, height=28,
                                 fg_color="#2B579A", hover_color="#1E3F66", command=self._open_report_dialog)
        btn_docx.pack(side=tk.LEFT, padx=3, pady=8)

        btn_ai = ctk.CTkButton(self.toolbar, text="🤖 AI Copilot", width=90, height=28,
                               fg_color="#4EC9B0", hover_color="#3BA58F", text_color="#1E1E1E", command=self._open_copilot_dialog)
        btn_ai.pack(side=tk.LEFT, padx=3, pady=8)

        # Theme Switcher Button on Toolbar
        self.btn_theme = ctk.CTkButton(self.toolbar, text="☀️ Light Mode", width=95, height=28,
                                       fg_color="#4A4A4A", hover_color="#606060", command=self.toggle_theme)
        self.btn_theme.pack(side=tk.RIGHT, padx=6, pady=8)

        btn_clear = ctk.CTkButton(self.toolbar, text="🗑 Clear", width=60, height=28, fg_color="#D9534F", hover_color="#C9302C", command=self._clear_data)
        btn_clear.pack(side=tk.RIGHT, padx=3, pady=8)

        # 2. Bottom Command Input Bar
        self.bottom_bar = ctk.CTkFrame(self, height=50, corner_radius=0, fg_color="#2B2D30")
        self.bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.lbl_prompt = ctk.CTkLabel(self.bottom_bar, text=".", font=("Consolas", 18, "bold"), text_color="#007ACC")
        self.lbl_prompt.pack(side=tk.LEFT, padx=(12, 6), pady=8)
        
        self.entry_cmd = ctk.CTkEntry(self.bottom_bar, placeholder_text="Type a command here (e.g. summarize, regress, logit, estat vif, arima, did, putdocx, ai '...')...",
                                       font=("Consolas", 13), height=32)
        self.entry_cmd.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=8)
        self.entry_cmd.bind("<Return>", lambda e: self._submit_command())
        self.entry_cmd.bind("<Up>", self._history_prev)
        self.entry_cmd.bind("<Down>", self._history_next)

        btn_submit = ctk.CTkButton(self.bottom_bar, text="Submit", width=75, height=32, command=self._submit_command)
        btn_submit.pack(side=tk.RIGHT, padx=(4, 12), pady=8)

        # 3. Main Multi-Pane Content Area
        self.paned_main = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#1E1E1E", bd=0, sashwidth=4)
        self.paned_main.pack(fill=tk.BOTH, expand=True)

        # Left Pane: Review / Command History
        self.frame_left = ctk.CTkFrame(self.paned_main, width=220, corner_radius=0, fg_color="#252526")
        self.paned_main.add(self.frame_left, minsize=140)

        self.lbl_review = ctk.CTkLabel(self.frame_left, text="Review History", font=("Segoe UI", 12, "bold"), text_color="#CCCCCC")
        self.lbl_review.pack(anchor="w", padx=10, pady=(8, 4))

        self.listbox_history = tk.Listbox(self.frame_left, bg="#1E1E1E", fg="#D4D4D4",
                                          selectbackground="#094771", selectforeground="#FFFFFF",
                                          font=("Consolas", 10), bd=0, highlightthickness=0)
        self.listbox_history.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self.listbox_history.bind("<Double-Button-1>", self._on_history_double_click)

        # Center Pane: Output / Results Console
        self.frame_center = ctk.CTkFrame(self.paned_main, corner_radius=0, fg_color="#1E1E1E")
        self.paned_main.add(self.frame_center, minsize=420)

        self.output_box = tk.Text(self.frame_center, bg="#1E1E1E", fg="#D4D4D4",
                                  insertbackground="#FFFFFF", font=("Consolas", 11),
                                  wrap=tk.NONE, bd=0, highlightthickness=0)
        
        scroll_y = ttk.Scrollbar(self.frame_center, orient="vertical", command=self.output_box.yview)
        scroll_x = ttk.Scrollbar(self.frame_center, orient="horizontal", command=self.output_box.xview)
        self.output_box.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.output_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Right Pane: Variables & Properties Manager
        self.frame_right = ctk.CTkFrame(self.paned_main, width=280, corner_radius=0, fg_color="#252526")
        self.paned_main.add(self.frame_right, minsize=180)

        self.lbl_vars = ctk.CTkLabel(self.frame_right, text="Variables", font=("Segoe UI", 12, "bold"), text_color="#CCCCCC")
        self.lbl_vars.pack(anchor="w", padx=10, pady=(8, 4))

        # Variables Treeview
        tree_frame = ctk.CTkFrame(self.frame_right, fg_color="#1E1E1E")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        self.tree_vars = ttk.Treeview(tree_frame, columns=("Type", "Format", "Label"), selectmode="browse")
        self.tree_vars.heading("#0", text="Name")
        self.tree_vars.heading("Type", text="Type")
        self.tree_vars.heading("Format", text="Format")
        self.tree_vars.heading("Label", text="Label")
        self.tree_vars.column("#0", width=80)
        self.tree_vars.column("Type", width=50)
        self.tree_vars.column("Format", width=50)
        self.tree_vars.column("Label", width=90)

        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree_vars.yview)
        self.tree_vars.configure(yscrollcommand=tree_scroll.set)
        self.tree_vars.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree_vars.bind("<Double-Button-1>", self._on_variable_double_click)

        # Properties Inspector Pane
        self.lbl_props = ctk.CTkLabel(self.frame_right, text="Properties", font=("Segoe UI", 12, "bold"), text_color="#CCCCCC")
        self.lbl_props.pack(anchor="w", padx=10, pady=(8, 2))

        self.props_text = tk.Text(self.frame_right, bg="#1E1E1E", fg="#A0A0A0",
                                  height=8, font=("Consolas", 10), wrap=tk.WORD, bd=0, highlightthickness=0)
        self.props_text.pack(fill=tk.X, padx=4, pady=(0, 6))
        self.props_text.insert(tk.END, "Variables: 0\nObservations: 0\nMemory: 0 bytes")
        self.props_text.configure(state="disabled")

    def toggle_theme(self):
        new_theme = "Light" if self.current_theme == "Dark" else "Dark"
        self._apply_theme(new_theme)

    def _apply_theme(self, theme_name):
        self.current_theme = theme_name
        ctk.set_appearance_mode(theme_name)
        
        style = ttk.Style()
        
        if theme_name == "Light":
            self.btn_theme.configure(text="🌙 Dark Mode", fg_color="#D0D0D0", hover_color="#B8B8B8", text_color="#1E1E1E")
            
            # Container colors
            self.configure(fg_color="#F3F3F3")
            self.toolbar.configure(fg_color="#E4E4E4")
            self.bottom_bar.configure(fg_color="#E4E4E4")
            self.paned_main.configure(bg="#DCDCDC")
            self.frame_left.configure(fg_color="#EFEFEF")
            self.frame_center.configure(fg_color="#FFFFFF")
            self.frame_right.configure(fg_color="#EFEFEF")
            
            # Labels
            self.lbl_review.configure(text_color="#333333")
            self.lbl_vars.configure(text_color="#333333")
            self.lbl_props.configure(text_color="#333333")
            self.lbl_prompt.configure(text_color="#005FB8")
            
            # Output Box
            self.output_box.configure(bg="#FFFFFF", fg="#1E1E1E", insertbackground="#000000")
            self.output_box.tag_config("cmd", foreground="#005FB8", font=("Consolas", 11, "bold"))
            self.output_box.tag_config("table_header", foreground="#098658", font=("Consolas", 11, "bold"))
            self.output_box.tag_config("stats_accent", foreground="#795E26")
            self.output_box.tag_config("error", foreground="#CD3131")
            self.output_box.tag_config("banner", foreground="#005FB8", font=("Consolas", 12, "bold"))
            self.output_box.tag_config("dim", foreground="#008000")
            
            # Review History & Properties
            self.listbox_history.configure(bg="#FFFFFF", fg="#1E1E1E",
                                           selectbackground="#0078D7", selectforeground="#FFFFFF")
            self.props_text.configure(bg="#FFFFFF", fg="#444444")
            
            # Treeview light style
            style.theme_use("clam")
            style.configure("Treeview", background="#FFFFFF", foreground="#1E1E1E", fieldbackground="#FFFFFF", rowheight=22)
            style.configure("Treeview.Heading", background="#E8E8E8", foreground="#333333", font=("Segoe UI", 9, "bold"))
            style.map("Treeview", background=[("selected", "#0078D7")], foreground=[("selected", "#FFFFFF")])

        else: # Dark Theme
            self.btn_theme.configure(text="☀️ Light Mode", fg_color="#4A4A4A", hover_color="#606060", text_color="#FFFFFF")
            
            # Container colors
            self.configure(fg_color="#252526")
            self.toolbar.configure(fg_color="#2B2D30")
            self.bottom_bar.configure(fg_color="#2B2D30")
            self.paned_main.configure(bg="#1E1E1E")
            self.frame_left.configure(fg_color="#252526")
            self.frame_center.configure(fg_color="#1E1E1E")
            self.frame_right.configure(fg_color="#252526")
            
            # Labels
            self.lbl_review.configure(text_color="#CCCCCC")
            self.lbl_vars.configure(text_color="#CCCCCC")
            self.lbl_props.configure(text_color="#CCCCCC")
            self.lbl_prompt.configure(text_color="#007ACC")
            
            # Output Box
            self.output_box.configure(bg="#1E1E1E", fg="#D4D4D4", insertbackground="#FFFFFF")
            self.output_box.tag_config("cmd", foreground="#9CDCFE", font=("Consolas", 11, "bold"))
            self.output_box.tag_config("table_header", foreground="#4EC9B0", font=("Consolas", 11, "bold"))
            self.output_box.tag_config("stats_accent", foreground="#DCDCAA")
            self.output_box.tag_config("error", foreground="#F44747")
            self.output_box.tag_config("banner", foreground="#007ACC", font=("Consolas", 12, "bold"))
            self.output_box.tag_config("dim", foreground="#6A9955")
            
            # Review History & Properties
            self.listbox_history.configure(bg="#1E1E1E", fg="#D4D4D4",
                                           selectbackground="#094771", selectforeground="#FFFFFF")
            self.props_text.configure(bg="#1E1E1E", fg="#A0A0A0")
            
            # Treeview dark style
            style.theme_use("clam")
            style.configure("Treeview", background="#1E1E1E", foreground="#D4D4D4", fieldbackground="#1E1E1E", rowheight=22)
            style.configure("Treeview.Heading", background="#2B2D30", foreground="#CCCCCC", font=("Segoe UI", 9, "bold"))
            style.map("Treeview", background=[("selected", "#094771")], foreground=[("selected", "#FFFFFF")])

    def _print_welcome(self):
        banner = (
            "========================================================================================\n"
            "                      Potatostics Statistical Analysis Software\n"
            "                              Release 1.0 - Open Source\n"
            "========================================================================================\n"
            "  * Independent statistical computing engine & interactive GUI IDE.\n"
            "  * Theme support: Switch between Dark Mode and Light Mode with 'View -> Theme' or 'F2'.\n"
            "  * Features: OLS (Robust HC1), Logit (Odds Ratios), 2SLS IV, Panel FE, Multi-Model Comparison,\n"
            "    Diagnostics (VIF, Hettest, RESET, DW), Time Series (ADF, ARIMA, Corrgram), Causal (DiD, PSM),\n"
            "    Command assistant (rule-based), Native Word Report Generator (.docx), and User Plugins.\n"
            "  * Open .dta or .csv datasets with 'use <filename>' or 'Open Data'.\n"
            "  * Type 'help' for the complete interactive command reference.\n"
            "----------------------------------------------------------------------------------------\n\n"
        )
        self.output_box.insert(tk.END, banner, "banner")
        self.output_box.see(tk.END)

    def run_command(self, cmd_line):
        cmd = cmd_line.strip()
        if not cmd:
            return
        
        self.cmd_history.append(cmd)
        self.history_index = len(self.cmd_history)
        self.listbox_history.insert(tk.END, cmd)
        self.listbox_history.see(tk.END)
        
        self.output_box.insert(tk.END, f". {cmd}\n", "cmd")
        
        try:
            result = self.interpreter.execute(cmd)
            if result:
                self.output_box.insert(tk.END, f"{result}\n\n")
        except Exception as e:
            self.output_box.insert(tk.END, f"error: {str(e)}\n\n", "error")
            
        self.output_box.see(tk.END)
        self._update_variables_panel()

    def _submit_command(self):
        cmd = self.entry_cmd.get().strip()
        if cmd:
            self.entry_cmd.delete(0, tk.END)
            self.run_command(cmd)

    def _history_prev(self, event):
        if self.cmd_history and self.history_index > 0:
            self.history_index -= 1
            self.entry_cmd.delete(0, tk.END)
            self.entry_cmd.insert(0, self.cmd_history[self.history_index])
        return "break"

    def _history_next(self, event):
        if self.cmd_history and self.history_index < len(self.cmd_history) - 1:
            self.history_index += 1
            self.entry_cmd.delete(0, tk.END)
            self.entry_cmd.insert(0, self.cmd_history[self.history_index])
        elif self.history_index >= len(self.cmd_history) - 1:
            self.history_index = len(self.cmd_history)
            self.entry_cmd.delete(0, tk.END)
        return "break"

    def _on_history_double_click(self, event):
        sel = self.listbox_history.curselection()
        if sel:
            cmd = self.listbox_history.get(sel[0])
            self.entry_cmd.delete(0, tk.END)
            self.entry_cmd.insert(0, cmd)
            self.entry_cmd.focus()

    def _on_variable_double_click(self, event):
        item = self.tree_vars.selection()
        if item:
            var_name = self.tree_vars.item(item, "text")
            curr = self.entry_cmd.get()
            if curr and not curr.endswith(" "):
                self.entry_cmd.insert(tk.END, " " + var_name)
            else:
                self.entry_cmd.insert(tk.END, var_name)
            self.entry_cmd.focus()

    def _prompt_command(self, template):
        self.entry_cmd.delete(0, tk.END)
        self.entry_cmd.insert(0, template)
        self.entry_cmd.focus()

    def _update_variables_panel(self):
        for item in self.tree_vars.get_children():
            self.tree_vars.delete(item)
            
        if self.dataset.is_empty:
            self.props_text.configure(state="normal")
            self.props_text.delete("1.0", tk.END)
            self.props_text.insert(tk.END, "Variables: 0\nObservations: 0\nMemory: 0 bytes")
            self.props_text.configure(state="disabled")
            return
            
        for col in self.dataset.df.columns:
            dtype_str = str(self.dataset.df[col].dtype)
            fmt = self.dataset.var_formats.get(col, "%9.0g" if "int" in dtype_str else ("%9.2f" if "float" in dtype_str else "%9s"))
            lbl = self.dataset.var_labels.get(col, "")
            self.tree_vars.insert("", tk.END, text=col, values=(dtype_str, fmt, lbl))
            
        n_obs = len(self.dataset.df)
        n_vars = len(self.dataset.df.columns)
        mem = self.dataset.df.memory_usage(deep=True).sum()
        
        props_str = f"Variables: {n_vars:,}\n"
        props_str += f"Observations: {n_obs:,}\n"
        props_str += f"Size: {mem:,} bytes\n"
        props_str += f"Label: {self.dataset.dataset_label or 'None'}"
        
        self.props_text.configure(state="normal")
        self.props_text.delete("1.0", tk.END)
        self.props_text.insert(tk.END, props_str)
        self.props_text.configure(state="disabled")

    def _open_file_dialog(self):
        path = filedialog.askopenfilename(
            title="Open Dataset",
            filetypes=[("Data Files (*.dta, *.csv)", "*.dta *.csv"), ("DTA Datasets (*.dta)", "*.dta"), ("CSV Files (*.csv)", "*.csv"), ("All Files", "*.*")]
        )
        if path:
            self.run_command(f'use "{path}", clear')

    def _save_file_dialog(self):
        if self.dataset.is_empty:
            messagebox.showwarning("Warning", "No dataset in memory to save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Dataset As CSV",
            defaultextension=".csv",
            filetypes=[("CSV Files (*.csv)", "*.csv"), ("All Files", "*.*")]
        )
        if path:
            self.run_command(f'save "{path}", replace')

    def _open_data_editor(self):
        if self.dataset.is_empty:
            messagebox.showinfo("Data Editor", "No dataset in memory to view. Open a dataset first.")
            return
        DataEditorWindow(self, self.dataset, on_change_callback=self._update_variables_panel)

    def _open_script_editor(self):
        ScriptEditorWindow(self, self.run_command)

    def _open_regression_dialog(self):
        if self.dataset.is_empty:
            messagebox.showinfo("Linear Regression", "Please load a dataset first.")
            return
        RegressionDialog(self, self.dataset, self.run_command)

    def _open_logit_dialog(self):
        if self.dataset.is_empty:
            messagebox.showinfo("Logistic Regression", "Please load a dataset first.")
            return
        LogitDialog(self, self.dataset, self.run_command)

    def _open_iv_dialog(self):
        if self.dataset.is_empty:
            messagebox.showinfo("Instrumental Variables", "Please load a dataset first.")
            return
        IVDialog(self, self.dataset, self.run_command)

    def _open_panel_dialog(self):
        if self.dataset.is_empty:
            messagebox.showinfo("Panel Data Fixed Effects", "Please load a dataset first.")
            return
        PanelDialog(self, self.dataset, self.run_command)

    def _open_diagnostics_dialog(self):
        DiagnosticsDialog(self, self.run_command)

    def _open_timeseries_dialog(self):
        if self.dataset.is_empty:
            messagebox.showinfo("Time Series", "Please load a dataset first.")
            return
        TimeSeriesDialog(self, self.dataset, self.run_command)

    def _open_causal_dialog(self):
        if self.dataset.is_empty:
            messagebox.showinfo("Causal Inference", "Please load a dataset first.")
            return
        CausalDialog(self, self.dataset, self.run_command)

    def _open_summarize_dialog(self):
        if self.dataset.is_empty:
            messagebox.showinfo("Summarize", "Please load a dataset first.")
            return
        SummarizeDialog(self, self.dataset, self.run_command)

    def _open_tabulate_dialog(self):
        if self.dataset.is_empty:
            messagebox.showinfo("Tabulate", "Please load a dataset first.")
            return
        TabulateDialog(self, self.dataset, self.run_command)

    def _open_scatter_dialog(self):
        if self.dataset.is_empty:
            messagebox.showinfo("Scatter Plot", "Please load a dataset first.")
            return
        ScatterDialog(self, self.dataset, self.run_command)

    def _open_histogram_dialog(self):
        if self.dataset.is_empty:
            messagebox.showinfo("Histogram", "Please load a dataset first.")
            return
        HistogramDialog(self, self.dataset, self.run_command)

    def _open_boxplot_dialog(self):
        if self.dataset.is_empty:
            messagebox.showinfo("Box Plot", "Please load a dataset first.")
            return
        BoxplotDialog(self, self.dataset, self.run_command)

    def _open_generate_dialog(self):
        if self.dataset.is_empty:
            messagebox.showinfo("Generate", "Please load a dataset first.")
            return
        GenerateDialog(self, self.run_command)

    def _open_report_dialog(self):
        if self.dataset.is_empty:
            messagebox.showinfo("Report Generator", "Please load a dataset first.")
            return
        ReportDialog(self, self.run_command)

    def _open_copilot_dialog(self):
        AICopilotDialog(self, self.run_command)

    def _open_do_file_dialog(self):
        path = filedialog.askopenfilename(
            title="Execute Script File",
            filetypes=[("Script Files (*.do, *.txt)", "*.do *.txt"), ("All Files", "*.*")]
        )
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for line in lines:
                l = line.strip()
                if l and not l.startswith("*") and not l.startswith("//"):
                    self.run_command(l)

    def _clear_data(self):
        self.run_command("clear")

    def _clear_results(self):
        self.output_box.delete("1.0", tk.END)
        self._print_welcome()

    def _show_about(self):
        AboutDialog(self)

    def _render_plot(self, plot_type, df, **kwargs):
        PlotWindow(self, plot_type, df, **kwargs)
