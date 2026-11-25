"""
gui/dialogs.py - Interactive Dialog Windows for Potatostics Menu Actions
Provides point-and-click GUI dialogs with automatic Dark & Light theme adaptation:
- Linear Regression (OLS)
- Logistic Regression (Logit / Odds Ratios)
- Instrumental Variables (2SLS)
- Panel Data Fixed Effects (xtreg)
- Post-Estimation Diagnostics (VIF, Breusch-Pagan, Ramsey RESET, Durbin-Watson)
- Time Series Analysis (ADF Unit Root, Correlogram, ARIMA Forecasting)
- Causal Inference (Difference-in-Differences, Propensity Score Matching)
- Summarize & Tabulate
- Generate Variable
- Visualizations (Scatter, Histogram, Boxplot)
- Word Report Generator
- AI Copilot Assistant
"""

import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk

def get_theme_colors():
    is_dark = ctk.get_appearance_mode().lower() == "dark"
    return {
        "dialog_bg": "#252526" if is_dark else "#F3F3F3",
        "frame_bg": "#1E1E1E" if is_dark else "#EAEAEA",
        "lb_bg": "#1E1E1E" if is_dark else "#FFFFFF",
        "lb_fg": "#D4D4D4" if is_dark else "#1E1E1E",
        "select_bg": "#094771" if is_dark else "#0078D7",
        "select_fg": "#FFFFFF",
        "header_color": "#9CDCFE" if is_dark else "#005FB8"
    }

class BaseDialog(ctk.CTkToplevel):
    def __init__(self, parent, title, width=440, height=380):
        super().__init__(parent)
        self.title(title)
        self.geometry(f"{width}x{height}")
        self.minsize(width, height)
        self.transient(parent)
        self.grab_set()
        colors = get_theme_colors()
        self.configure(fg_color=colors["dialog_bg"])

class RegressionDialog(BaseDialog):
    def __init__(self, parent, dataset, on_submit):
        super().__init__(parent, "Linear Regression (OLS) - Potatostics", width=460, height=450)
        self.dataset = dataset
        self.on_submit = on_submit
        colors = get_theme_colors()
        
        numeric_vars = [c for c in dataset.df.columns if dataset.df[c].dtype != "object"]
        
        ctk.CTkLabel(self, text="Dependent variable (Y):", font=("Segoe UI", 12, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(15, 2))
        self.combo_dep = ctk.CTkComboBox(self, values=numeric_vars, width=280)
        if numeric_vars:
            self.combo_dep.set(numeric_vars[0])
        self.combo_dep.pack(anchor="w", padx=20, pady=(0, 10))
        
        ctk.CTkLabel(self, text="Independent variables (X) [Select multiple]:", font=("Segoe UI", 12, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(5, 2))
        
        frame_indep = ctk.CTkFrame(self, fg_color=colors["frame_bg"])
        frame_indep.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        self.listbox_indep = tk.Listbox(frame_indep, selectmode=tk.MULTIPLE, bg=colors["lb_bg"], fg=colors["lb_fg"],
                                        selectbackground=colors["select_bg"], selectforeground=colors["select_fg"],
                                        font=("Consolas", 11), bd=0, highlightthickness=0)
        vsb = ttk.Scrollbar(frame_indep, orient="vertical", command=self.listbox_indep.yview)
        self.listbox_indep.configure(yscrollcommand=vsb.set)
        
        self.listbox_indep.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        for v in numeric_vars:
            self.listbox_indep.insert(tk.END, v)
            
        self.var_robust = tk.BooleanVar(value=False)
        self.chk_robust = ctk.CTkCheckBox(self, text="Robust standard errors (vce(robust) / Huber-White)", variable=self.var_robust)
        self.chk_robust.pack(anchor="w", padx=20, pady=(0, 15))
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        ctk.CTkButton(btn_frame, text="OK / Submit", width=100, command=self._apply).pack(side=tk.RIGHT, padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", width=80, fg_color="#555555", hover_color="#666666", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def _apply(self):
        dep = self.combo_dep.get()
        selected_indices = self.listbox_indep.curselection()
        indeps = [self.listbox_indep.get(i) for i in selected_indices if self.listbox_indep.get(i) != dep]
        
        if not indeps:
            messagebox.showwarning("Incomplete", "Please select at least one independent variable.")
            return
            
        opt = ", robust" if self.var_robust.get() else ""
        cmd = f"regress {dep} {' '.join(indeps)}{opt}"
        self.on_submit(cmd)
        self.destroy()

class LogitDialog(BaseDialog):
    def __init__(self, parent, dataset, on_submit):
        super().__init__(parent, "Logistic Regression (Logit) - Potatostics", width=460, height=450)
        self.dataset = dataset
        self.on_submit = on_submit
        colors = get_theme_colors()
        
        numeric_vars = [c for c in dataset.df.columns if dataset.df[c].dtype != "object"]
        
        ctk.CTkLabel(self, text="Binary Dependent Variable (0/1):", font=("Segoe UI", 12, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(15, 2))
        self.combo_dep = ctk.CTkComboBox(self, values=numeric_vars, width=280)
        if numeric_vars:
            self.combo_dep.set(numeric_vars[0])
        self.combo_dep.pack(anchor="w", padx=20, pady=(0, 10))
        
        ctk.CTkLabel(self, text="Explanatory Predictors [Select multiple]:", font=("Segoe UI", 12, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(5, 2))
        frame_indep = ctk.CTkFrame(self, fg_color=colors["frame_bg"])
        frame_indep.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        self.listbox_indep = tk.Listbox(frame_indep, selectmode=tk.MULTIPLE, bg=colors["lb_bg"], fg=colors["lb_fg"],
                                        selectbackground=colors["select_bg"], selectforeground=colors["select_fg"],
                                        font=("Consolas", 11), bd=0, highlightthickness=0)
        vsb = ttk.Scrollbar(frame_indep, orient="vertical", command=self.listbox_indep.yview)
        self.listbox_indep.configure(yscrollcommand=vsb.set)
        self.listbox_indep.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        for v in numeric_vars:
            self.listbox_indep.insert(tk.END, v)
            
        self.var_or = tk.BooleanVar(value=False)
        self.chk_or = ctk.CTkCheckBox(self, text="Report Odds Ratios (, or)", variable=self.var_or)
        self.chk_or.pack(anchor="w", padx=20, pady=(0, 15))
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        ctk.CTkButton(btn_frame, text="Estimate Logit", width=110, command=self._apply).pack(side=tk.RIGHT, padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", width=80, fg_color="#555555", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def _apply(self):
        dep = self.combo_dep.get()
        indeps = [self.listbox_indep.get(i) for i in self.listbox_indep.curselection() if self.listbox_indep.get(i) != dep]
        if not indeps:
            messagebox.showwarning("Incomplete", "Please select at least one explanatory predictor.")
            return
        opt = ", or" if self.var_or.get() else ""
        self.on_submit(f"logit {dep} {' '.join(indeps)}{opt}")
        self.destroy()

class IVDialog(BaseDialog):
    def __init__(self, parent, dataset, on_submit):
        super().__init__(parent, "Instrumental Variables (2SLS) - Potatostics", width=480, height=480)
        self.dataset = dataset
        self.on_submit = on_submit
        colors = get_theme_colors()
        numeric_vars = [c for c in dataset.df.columns if dataset.df[c].dtype != "object"]
        
        ctk.CTkLabel(self, text="Dependent variable (Y):", font=("Segoe UI", 11, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(10, 2))
        self.combo_dep = ctk.CTkComboBox(self, values=numeric_vars, width=240)
        self.combo_dep.pack(anchor="w", padx=20, pady=(0, 6))
        
        ctk.CTkLabel(self, text="Endogenous Regressor:", font=("Segoe UI", 11, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(4, 2))
        self.combo_endog = ctk.CTkComboBox(self, values=numeric_vars, width=240)
        self.combo_endog.pack(anchor="w", padx=20, pady=(0, 6))
        
        ctk.CTkLabel(self, text="Excluded Instruments (Z) [Select 1 or more]:", font=("Segoe UI", 11, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(4, 2))
        f_inst = ctk.CTkFrame(self, fg_color=colors["frame_bg"], height=80)
        f_inst.pack(fill=tk.X, padx=20, pady=(0, 6))
        self.lb_inst = tk.Listbox(f_inst, selectmode=tk.MULTIPLE, bg=colors["lb_bg"], fg=colors["lb_fg"],
                                  selectbackground=colors["select_bg"], selectforeground=colors["select_fg"], height=4, bd=0, highlightthickness=0)
        self.lb_inst.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        for v in numeric_vars:
            self.lb_inst.insert(tk.END, v)
            
        ctk.CTkLabel(self, text="Exogenous Controls (X) [Optional]:", font=("Segoe UI", 11, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(4, 2))
        f_exog = ctk.CTkFrame(self, fg_color=colors["frame_bg"], height=80)
        f_exog.pack(fill=tk.X, padx=20, pady=(0, 10))
        self.lb_exog = tk.Listbox(f_exog, selectmode=tk.MULTIPLE, bg=colors["lb_bg"], fg=colors["lb_fg"],
                                  selectbackground=colors["select_bg"], selectforeground=colors["select_fg"], height=4, bd=0, highlightthickness=0)
        self.lb_exog.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        for v in numeric_vars:
            self.lb_exog.insert(tk.END, v)
            
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        ctk.CTkButton(btn_frame, text="Estimate 2SLS", width=110, command=self._apply).pack(side=tk.RIGHT, padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", width=80, fg_color="#555555", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def _apply(self):
        dep = self.combo_dep.get()
        endog = self.combo_endog.get()
        insts = [self.lb_inst.get(i) for i in self.lb_inst.curselection()]
        exogs = [self.lb_exog.get(i) for i in self.lb_exog.curselection()]
        
        if not dep or not endog or not insts:
            messagebox.showwarning("Incomplete", "Dependent, endogenous, and instrument variables are required.")
            return
        exog_str = f" {' '.join(exogs)}" if exogs else ""
        cmd = f"ivregress 2sls {dep}{exog_str} ({endog} = {' '.join(insts)})"
        self.on_submit(cmd.strip())
        self.destroy()

class PanelDialog(BaseDialog):
    def __init__(self, parent, dataset, on_submit):
        super().__init__(parent, "Panel Data Fixed Effects (xtreg) - Potatostics", width=460, height=440)
        self.dataset = dataset
        self.on_submit = on_submit
        colors = get_theme_colors()
        cols = list(dataset.df.columns)
        
        ctk.CTkLabel(self, text="Entity / Cross-Section ID (xtset id):", font=("Segoe UI", 11, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(12, 2))
        self.combo_id = ctk.CTkComboBox(self, values=cols, width=260)
        self.combo_id.pack(anchor="w", padx=20, pady=(0, 8))
        
        ctk.CTkLabel(self, text="Dependent Variable (Y):", font=("Segoe UI", 11, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(4, 2))
        self.combo_dep = ctk.CTkComboBox(self, values=cols, width=260)
        self.combo_dep.pack(anchor="w", padx=20, pady=(0, 8))
        
        ctk.CTkLabel(self, text="Independent Variables (X):", font=("Segoe UI", 11, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(4, 2))
        f_vars = ctk.CTkFrame(self, fg_color=colors["frame_bg"])
        f_vars.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        self.lb_indep = tk.Listbox(f_vars, selectmode=tk.MULTIPLE, bg=colors["lb_bg"], fg=colors["lb_fg"],
                                   selectbackground=colors["select_bg"], selectforeground=colors["select_fg"], bd=0, highlightthickness=0)
        self.lb_indep.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        for c in cols:
            self.lb_indep.insert(tk.END, c)
            
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 12))
        ctk.CTkButton(btn_frame, text="Run FE Model", width=110, command=self._apply).pack(side=tk.RIGHT, padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", width=80, fg_color="#555555", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def _apply(self):
        id_var = self.combo_id.get()
        dep = self.combo_dep.get()
        indeps = [self.lb_indep.get(i) for i in self.lb_indep.curselection() if self.lb_indep.get(i) not in (id_var, dep)]
        if not id_var or not dep or not indeps:
            messagebox.showwarning("Incomplete", "Entity ID, Dependent, and at least one Independent variable are required.")
            return
        self.on_submit(f"xtset {id_var}")
        self.on_submit(f"xtreg {dep} {' '.join(indeps)}, fe")
        self.destroy()

class DiagnosticsDialog(BaseDialog):
    def __init__(self, parent, on_submit):
        super().__init__(parent, "Post-Estimation Specification & Diagnostics - Potatostics", width=440, height=320)
        self.on_submit = on_submit
        colors = get_theme_colors()
        
        ctk.CTkLabel(self, text="Select Regression Diagnostic Test:", font=("Segoe UI", 13, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(15, 10))
        
        self.var_test = tk.StringVar(value="vif")
        
        tests = [
            ("Variance Inflation Factor (Multicollinearity - estat vif)", "vif"),
            ("Breusch-Pagan / Cook-Weisberg (Heteroskedasticity - estat hettest)", "hettest"),
            ("Ramsey RESET (Omitted Variables / Misspecification - estat ovtest)", "ovtest"),
            ("Durbin-Watson (First-Order Residual Autocorrelation - estat dwatson)", "dwatson")
        ]
        
        for text, val in tests:
            ctk.CTkRadioButton(self, text=text, variable=self.var_test, value=val).pack(anchor="w", padx=25, pady=6)
            
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill=tk.X, padx=20, pady=(20, 15))
        ctk.CTkButton(btn_frame, text="Run Test", width=100, command=self._apply).pack(side=tk.RIGHT, padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", width=80, fg_color="#555555", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def _apply(self):
        test_type = self.var_test.get()
        self.on_submit(f"estat {test_type}")
        self.destroy()

class TimeSeriesDialog(BaseDialog):
    def __init__(self, parent, dataset, on_submit):
        super().__init__(parent, "Time Series Analysis - Potatostics", width=460, height=430)
        self.dataset = dataset
        self.on_submit = on_submit
        colors = get_theme_colors()
        numeric_vars = [c for c in dataset.df.columns if dataset.df[c].dtype != "object"]
        
        ctk.CTkLabel(self, text="Time Series Target Variable:", font=("Segoe UI", 12, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(12, 2))
        self.combo_var = ctk.CTkComboBox(self, values=numeric_vars, width=280)
        if numeric_vars:
            self.combo_var.set(numeric_vars[0])
        self.combo_var.pack(anchor="w", padx=20, pady=(0, 10))
        
        self.tabview = ctk.CTkTabview(self, width=420, height=250)
        self.tabview.pack(padx=20, pady=(0, 10))
        
        tab_adf = self.tabview.add("Unit Root (ADF)")
        ctk.CTkLabel(tab_adf, text="Augmented Dickey-Fuller Unit Root Test", font=("Segoe UI", 11, "bold")).pack(pady=10)
        ctk.CTkLabel(tab_adf, text="Tests H0: Series contains a unit root (non-stationary)\nvs H1: Series is stationary.", text_color="#A0A0A0").pack(pady=5)
        ctk.CTkButton(tab_adf, text="Run Unit Root Test (dfuller)", width=200, command=lambda: self._run_adf()).pack(pady=15)
        
        tab_corr = self.tabview.add("Correlogram")
        ctk.CTkLabel(tab_corr, text="Autocorrelation & Partial Autocorrelation", font=("Segoe UI", 11, "bold")).pack(pady=10)
        ctk.CTkLabel(tab_corr, text="Computes ACF and PACF across lags with Q-stat tests.", text_color="#A0A0A0").pack(pady=5)
        ctk.CTkButton(tab_corr, text="Compute Correlogram (corrgram)", width=220, command=lambda: self._run_corrgram()).pack(pady=15)
        
        tab_arima = self.tabview.add("ARIMA Model")
        f_orders = ctk.CTkFrame(tab_arima, fg_color="transparent")
        f_orders.pack(pady=10)
        ctk.CTkLabel(f_orders, text="AR (p):").grid(row=0, column=0, padx=5, pady=4)
        self.e_p = ctk.CTkEntry(f_orders, width=50)
        self.e_p.insert(0, "1")
        self.e_p.grid(row=0, column=1, padx=5, pady=4)
        
        ctk.CTkLabel(f_orders, text="Diff (d):").grid(row=0, column=2, padx=5, pady=4)
        self.e_d = ctk.CTkEntry(f_orders, width=50)
        self.e_d.insert(0, "0")
        self.e_d.grid(row=0, column=3, padx=5, pady=4)
        
        ctk.CTkLabel(f_orders, text="MA (q):").grid(row=0, column=4, padx=5, pady=4)
        self.e_q = ctk.CTkEntry(f_orders, width=50)
        self.e_q.insert(0, "0")
        self.e_q.grid(row=0, column=5, padx=5, pady=4)
        
        ctk.CTkLabel(tab_arima, text="Forecast Steps:").pack(pady=(4, 2))
        self.e_steps = ctk.CTkEntry(tab_arima, width=80)
        self.e_steps.insert(0, "5")
        self.e_steps.pack(pady=(0, 10))
        ctk.CTkButton(tab_arima, text="Estimate ARIMA & Forecast", width=200, command=lambda: self._run_arima()).pack()

    def _run_adf(self):
        v = self.combo_var.get()
        if v:
            self.on_submit(f"dfuller {v}")
            self.destroy()

    def _run_corrgram(self):
        v = self.combo_var.get()
        if v:
            self.on_submit(f"corrgram {v}")
            self.destroy()

    def _run_arima(self):
        v = self.combo_var.get()
        p = self.e_p.get().strip() or "1"
        d = self.e_d.get().strip() or "0"
        q = self.e_q.get().strip() or "0"
        steps = self.e_steps.get().strip() or "5"
        self.on_submit(f"arima {v}, ar({p}) d({d}) ma({q}) steps({steps})")
        self.destroy()

class CausalDialog(BaseDialog):
    def __init__(self, parent, dataset, on_submit):
        super().__init__(parent, "Causal Inference Analysis - Potatostics", width=460, height=440)
        self.dataset = dataset
        self.on_submit = on_submit
        cols = list(dataset.df.columns)
        colors = get_theme_colors()
        
        self.tabview = ctk.CTkTabview(self, width=420, height=380)
        self.tabview.pack(padx=20, pady=10)
        
        # DiD Tab
        tab_did = self.tabview.add("Difference-in-Differences")
        ctk.CTkLabel(tab_did, text="Outcome Variable (Y):", font=("Segoe UI", 11, "bold"), text_color=colors["header_color"]).pack(anchor="w", pady=(5, 2))
        self.combo_did_y = ctk.CTkComboBox(tab_did, values=cols, width=260)
        self.combo_did_y.pack(anchor="w", pady=(0, 6))
        
        ctk.CTkLabel(tab_did, text="Treatment Group Indicator (0/1):", font=("Segoe UI", 11, "bold"), text_color=colors["header_color"]).pack(anchor="w", pady=(4, 2))
        self.combo_did_treat = ctk.CTkComboBox(tab_did, values=cols, width=260)
        self.combo_did_treat.pack(anchor="w", pady=(0, 6))
        
        ctk.CTkLabel(tab_did, text="Post-Treatment Period Indicator (0/1):", font=("Segoe UI", 11, "bold"), text_color=colors["header_color"]).pack(anchor="w", pady=(4, 2))
        self.combo_did_time = ctk.CTkComboBox(tab_did, values=cols, width=260)
        self.combo_did_time.pack(anchor="w", pady=(0, 15))
        
        ctk.CTkButton(tab_did, text="Estimate DiD Effect", width=160, command=self._apply_did).pack(anchor="e")
        
        # PSMatch Tab
        tab_ps = self.tabview.add("Propensity Score Match")
        ctk.CTkLabel(tab_ps, text="Binary Treatment Variable (D):", font=("Segoe UI", 11, "bold"), text_color=colors["header_color"]).pack(anchor="w", pady=(4, 2))
        self.combo_ps_d = ctk.CTkComboBox(tab_ps, values=cols, width=240)
        self.combo_ps_d.pack(anchor="w", pady=(0, 4))
        
        ctk.CTkLabel(tab_ps, text="Outcome Variable (Y):", font=("Segoe UI", 11, "bold"), text_color=colors["header_color"]).pack(anchor="w", pady=(4, 2))
        self.combo_ps_y = ctk.CTkComboBox(tab_ps, values=cols, width=240)
        self.combo_ps_y.pack(anchor="w", pady=(0, 4))
        
        ctk.CTkLabel(tab_ps, text="Matching Covariates (X):", font=("Segoe UI", 11, "bold"), text_color=colors["header_color"]).pack(anchor="w", pady=(4, 2))
        f_cov = ctk.CTkFrame(tab_ps, fg_color=colors["frame_bg"], height=70)
        f_cov.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.lb_cov = tk.Listbox(f_cov, selectmode=tk.MULTIPLE, bg=colors["lb_bg"], fg=colors["lb_fg"],
                                 selectbackground=colors["select_bg"], selectforeground=colors["select_fg"], height=3, bd=0, highlightthickness=0)
        self.lb_cov.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        for c in cols:
            self.lb_cov.insert(tk.END, c)
            
        ctk.CTkButton(tab_ps, text="Estimate ATT", width=140, command=self._apply_psmatch).pack(anchor="e")

    def _apply_did(self):
        y = self.combo_did_y.get()
        t = self.combo_did_treat.get()
        post = self.combo_did_time.get()
        if not y or not t or not post:
            messagebox.showwarning("Incomplete", "All three variables are required for DiD.")
            return
        self.on_submit(f"did {y} {t} {post}")
        self.destroy()

    def _apply_psmatch(self):
        d = self.combo_ps_d.get()
        y = self.combo_ps_y.get()
        covs = [self.lb_cov.get(i) for i in self.lb_cov.curselection() if self.lb_cov.get(i) not in (d, y)]
        if not d or not y or not covs:
            messagebox.showwarning("Incomplete", "Treatment, outcome, and at least one covariate are required.")
            return
        self.on_submit(f"psmatch {d} {' '.join(covs)}, outcome({y})")
        self.destroy()

class ReportDialog(BaseDialog):
    def __init__(self, parent, on_submit):
        super().__init__(parent, "Generate Word Report - Potatostics", width=420, height=220)
        self.on_submit = on_submit
        colors = get_theme_colors()
        
        ctk.CTkLabel(self, text="Report Document Filename (.docx):", font=("Segoe UI", 12, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(20, 5))
        self.entry_name = ctk.CTkEntry(self, placeholder_text="Econometric_Report.docx", width=360)
        self.entry_name.insert(0, "Econometric_Report.docx")
        self.entry_name.pack(anchor="w", padx=20, pady=(0, 25))
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        ctk.CTkButton(btn_frame, text="Generate Word Report", width=160, command=self._apply).pack(side=tk.RIGHT, padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", width=80, fg_color="#555555", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def _apply(self):
        name = self.entry_name.get().strip() or "Econometric_Report.docx"
        self.on_submit(f"putdocx {name}")
        self.destroy()

class AICopilotDialog(BaseDialog):
    def __init__(self, parent, on_submit):
        super().__init__(parent, "Potatostics AI Econometric Assistant", width=520, height=280)
        self.on_submit = on_submit
        
        ctk.CTkLabel(self, text="Ask Potatostics Assistant (Natural Language):", font=("Segoe UI", 13, "bold"), text_color="#4EC9B0" if ctk.get_appearance_mode().lower() == "dark" else "#098658").pack(anchor="w", padx=20, pady=(15, 5))
        ctk.CTkLabel(self, text="Ask econometric modeling questions or request analysis in plain English:", text_color="#A0A0A0" if ctk.get_appearance_mode().lower() == "dark" else "#666666").pack(anchor="w", padx=20, pady=(0, 10))
        
        self.entry_prompt = ctk.CTkEntry(self, placeholder_text="e.g. How do I test for heteroskedasticity and fix it?", width=480)
        self.entry_prompt.pack(anchor="w", padx=20, pady=(0, 20))
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        ctk.CTkButton(btn_frame, text="Ask AI Copilot", width=120, fg_color="#4EC9B0", hover_color="#3BA58F", text_color="#1E1E1E", command=self._apply).pack(side=tk.RIGHT, padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", width=80, fg_color="#555555", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def _apply(self):
        prompt = self.entry_prompt.get().strip()
        if prompt:
            self.on_submit(f'ai "{prompt}"')
        self.destroy()

class HistogramDialog(BaseDialog):
    def __init__(self, parent, dataset, on_submit):
        super().__init__(parent, "Histogram Distribution - Potatostics", width=380, height=220)
        self.on_submit = on_submit
        colors = get_theme_colors()
        numeric_vars = [c for c in dataset.df.columns if dataset.df[c].dtype != "object"]
        
        ctk.CTkLabel(self, text="Select Variable for Histogram:", font=("Segoe UI", 12, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(20, 5))
        self.combo_var = ctk.CTkComboBox(self, values=numeric_vars, width=280)
        if numeric_vars:
            self.combo_var.set(numeric_vars[0])
        self.combo_var.pack(anchor="w", padx=20, pady=(0, 25))
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        ctk.CTkButton(btn_frame, text="Plot Histogram", width=120, command=self._apply).pack(side=tk.RIGHT, padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", width=80, fg_color="#555555", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def _apply(self):
        var = self.combo_var.get()
        if var:
            self.on_submit(f"histogram {var}")
        self.destroy()

class BoxplotDialog(BaseDialog):
    def __init__(self, parent, dataset, on_submit):
        super().__init__(parent, "Box Plot - Potatostics", width=380, height=280)
        self.on_submit = on_submit
        colors = get_theme_colors()
        numeric_vars = [c for c in dataset.df.columns if dataset.df[c].dtype != "object"]
        all_vars = ["(None)"] + list(dataset.df.columns)
        
        ctk.CTkLabel(self, text="Target Numeric Variable:", font=("Segoe UI", 12, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(15, 2))
        self.combo_var = ctk.CTkComboBox(self, values=numeric_vars, width=260)
        if numeric_vars:
            self.combo_var.set(numeric_vars[0])
        self.combo_var.pack(anchor="w", padx=20, pady=(0, 10))
        
        ctk.CTkLabel(self, text="Group by Variable (Optional):", font=("Segoe UI", 12, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(5, 2))
        self.combo_over = ctk.CTkComboBox(self, values=all_vars, width=260)
        self.combo_over.set("(None)")
        self.combo_over.pack(anchor="w", padx=20, pady=(0, 20))
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        ctk.CTkButton(btn_frame, text="Plot Boxplot", width=110, command=self._apply).pack(side=tk.RIGHT, padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", width=80, fg_color="#555555", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def _apply(self):
        var = self.combo_var.get()
        over = self.combo_over.get()
        if var:
            if over and over != "(None)":
                self.on_submit(f"boxplot {var}, over({over})")
            else:
                self.on_submit(f"boxplot {var}")
        self.destroy()

class SummarizeDialog(BaseDialog):
    def __init__(self, parent, dataset, on_submit):
        super().__init__(parent, "Summarize - Potatostics", width=380, height=360)
        self.on_submit = on_submit
        colors = get_theme_colors()
        
        vars_list = list(dataset.df.columns)
        ctk.CTkLabel(self, text="Select variables (leave empty for all):", font=("Segoe UI", 12, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(15, 5))
        
        frame_list = ctk.CTkFrame(self, fg_color=colors["frame_bg"])
        frame_list.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        
        self.listbox = tk.Listbox(frame_list, selectmode=tk.MULTIPLE, bg=colors["lb_bg"], fg=colors["lb_fg"],
                                  selectbackground=colors["select_bg"], selectforeground=colors["select_fg"],
                                  font=("Consolas", 11), bd=0, highlightthickness=0)
        vsb = ttk.Scrollbar(frame_list, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=vsb.set)
        
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        for v in vars_list:
            self.listbox.insert(tk.END, v)
            
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        ctk.CTkButton(btn_frame, text="OK", width=90, command=self._apply).pack(side=tk.RIGHT, padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", width=80, fg_color="#555555", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def _apply(self):
        sel = [self.listbox.get(i) for i in self.listbox.curselection()]
        cmd = f"summarize {' '.join(sel)}" if sel else "summarize"
        self.on_submit(cmd)
        self.destroy()

class TabulateDialog(BaseDialog):
    def __init__(self, parent, dataset, on_submit):
        super().__init__(parent, "Tabulate - Potatostics", width=380, height=260)
        self.on_submit = on_submit
        colors = get_theme_colors()
        
        vars_list = ["(None)"] + list(dataset.df.columns)
        
        ctk.CTkLabel(self, text="Row variable:", font=("Segoe UI", 12, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(15, 2))
        self.combo_row = ctk.CTkComboBox(self, values=list(dataset.df.columns), width=240)
        if len(dataset.df.columns) > 0:
            self.combo_row.set(dataset.df.columns[0])
        self.combo_row.pack(anchor="w", padx=20, pady=(0, 10))
        
        ctk.CTkLabel(self, text="Column variable (optional for 2-way):", font=("Segoe UI", 12, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(5, 2))
        self.combo_col = ctk.CTkComboBox(self, values=vars_list, width=240)
        self.combo_col.set("(None)")
        self.combo_col.pack(anchor="w", padx=20, pady=(0, 20))
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        ctk.CTkButton(btn_frame, text="OK", width=90, command=self._apply).pack(side=tk.RIGHT, padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", width=80, fg_color="#555555", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def _apply(self):
        row = self.combo_row.get()
        col = self.combo_col.get()
        if col and col != "(None)":
            cmd = f"tabulate {row} {col}"
        else:
            cmd = f"tabulate {row}"
        self.on_submit(cmd)
        self.destroy()

class ScatterDialog(BaseDialog):
    def __init__(self, parent, dataset, on_submit):
        super().__init__(parent, "Two-way Scatter Plot - Potatostics", width=380, height=260)
        self.on_submit = on_submit
        colors = get_theme_colors()
        
        numeric_vars = [c for c in dataset.df.columns if dataset.df[c].dtype != "object"]
        
        ctk.CTkLabel(self, text="Y variable (Vertical Axis):", font=("Segoe UI", 12, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(15, 2))
        self.combo_y = ctk.CTkComboBox(self, values=numeric_vars, width=240)
        if len(numeric_vars) > 0:
            self.combo_y.set(numeric_vars[0])
        self.combo_y.pack(anchor="w", padx=20, pady=(0, 10))
        
        ctk.CTkLabel(self, text="X variable (Horizontal Axis):", font=("Segoe UI", 12, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(5, 2))
        self.combo_x = ctk.CTkComboBox(self, values=numeric_vars, width=240)
        if len(numeric_vars) > 1:
            self.combo_x.set(numeric_vars[1])
        self.combo_x.pack(anchor="w", padx=20, pady=(0, 20))
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        ctk.CTkButton(btn_frame, text="Plot", width=90, command=self._apply).pack(side=tk.RIGHT, padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", width=80, fg_color="#555555", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def _apply(self):
        y = self.combo_y.get()
        x = self.combo_x.get()
        if y and x:
            self.on_submit(f"scatter {y} {x}")
        self.destroy()

class GenerateDialog(BaseDialog):
    def __init__(self, parent, on_submit):
        super().__init__(parent, "Generate / Create Variable - Potatostics", width=420, height=250)
        self.on_submit = on_submit
        colors = get_theme_colors()
        
        ctk.CTkLabel(self, text="Variable name:", font=("Segoe UI", 12, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(15, 2))
        self.entry_name = ctk.CTkEntry(self, placeholder_text="newvar", width=260)
        self.entry_name.pack(anchor="w", padx=20, pady=(0, 10))
        
        ctk.CTkLabel(self, text="Expression (e.g. weight * 0.45 or price / mpg):", font=("Segoe UI", 12, "bold"), text_color=colors["header_color"]).pack(anchor="w", padx=20, pady=(5, 2))
        self.entry_expr = ctk.CTkEntry(self, placeholder_text="expression", width=360)
        self.entry_expr.pack(anchor="w", padx=20, pady=(0, 20))
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        ctk.CTkButton(btn_frame, text="Generate", width=100, command=self._apply).pack(side=tk.RIGHT, padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", width=80, fg_color="#555555", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def _apply(self):
        name = self.entry_name.get().strip()
        expr = self.entry_expr.get().strip()
        if not name or not expr:
            messagebox.showwarning("Incomplete", "Both name and expression are required.")
            return
        self.on_submit(f"generate {name} = {expr}")
        self.destroy()
