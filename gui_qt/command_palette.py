"""
gui_qt/command_palette.py - Spotlight Command Palette (Ctrl+K)
Provides a keyboard-first fuzzy search launcher across all analytical tools,
statistical models, diagnostics, dataset variables, and IDE actions.
"""

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

ACTIONS = [
    ("📈 Linear Regression (OLS)", "regress", "Estimate ordinary least squares regression"),
    ("📈 Robust Linear Regression (HC1)", "regress , robust", "OLS with heteroskedasticity-robust standard errors"),
    ("🏛️ High-Dimensional Fixed Effects (reghdfe)", "reghdfe , absorb()", "Multi-way fixed effects via alternating projections"),
    ("📊 Logistic Regression (Logit)", "logit", "Binary response maximum likelihood estimation"),
    ("📊 Logistic Odds Ratios", "logistic", "Report odds ratios e^b instead of coefficients"),
    ("📊 Binary Probit Model (probit)", "probit", "Normal cumulative distribution link MLE"),
    ("🔢 Poisson Count Regression (poisson)", "poisson", "Count data regression with optional IRR"),
    ("📉 Censored Regression (tobit)", "tobit , ll(0)", "Tobit model with lower/upper censoring limits"),
    ("📐 Quantile Regression (qreg)", "qreg", "Median and arbitrary quantile regression"),
    ("🎯 Two-Stage Least Squares (2SLS IV)", "ivregress 2sls", "Instrumental variables estimation"),
    ("📑 Panel Fixed Effects (xtreg fe)", "xtreg , fe", "Panel within-group fixed effects estimator"),
    ("🔬 Average Marginal Effects (margins)", "margins, dydx(*)", "Compute dy/dx on active fitted model"),
    ("⚖️ Linear Combinations (lincom)", "lincom ", "Linear combination of estimators with delta method SE"),
    ("🔢 Summary Statistics", "summarize", "Means, standard deviations, min, max"),
    ("📊 One-way Frequency Table", "tabulate", "Categorical frequencies and percentages"),
    ("🔗 Correlation Matrix", "correlate", "Pearson correlation coefficients"),
    ("🔍 Ramsey RESET Test", "estat ovtest", "Test for omitted variables and model misspecification"),
    ("🔍 Breusch-Pagan Heteroskedasticity Test", "estat hettest", "Test for constant residual variance"),
    ("🔍 Variance Inflation Factors (VIF)", "estat vif", "Detect multicollinearity among regressors"),
    ("🔍 Durbin-Watson Autocorrelation", "estat dwatson", "Test for first-order serial correlation"),
    ("⏳ Dickey-Fuller Unit Root Test", "dfuller", "Test for time series stationarity"),
    ("⏳ Autocorrelations (Corrgram)", "corrgram", "ACF and PACF with Q-statistics"),
    ("⏳ ARIMA Time Series", "arima", "Autoregressive integrated moving average"),
    ("🎯 2x2 Difference-in-Differences", "did", "Quasi-experimental causal estimation"),
    ("🎯 Staggered DiD (csdid)", "csdid ", "Callaway & Sant'Anna (2021) staggered DiD with dynamic effects"),
    ("🎯 Synthetic Control (synth)", "synth ", "Convex donor matching with in-space placebo permutation"),
    ("🎯 Dynamic Event Study (eventstudy)", "eventstudy ", "Leads-and-lags regression with pre-trend test"),
    ("📈 Graph Studio / Builder...", "open_graph", "Interactive econometric and statistical graph builder"),
    ("📉 Two-way Scatter Plot (scatter)", "scatter ", "Two-way scatter plot with optional fit line"),
    ("📉 Two-way Line Plot (line)", "line ", "Two-way connected line plot"),
    ("📊 Binned Scatter Plot (binscatter)", "binscatter ", "Binned scatter plot with FWL covariate controls"),
    ("📐 Regression Discontinuity Plot (rdplot)", "rdplot ", "Local linear RD exhibit with 95% confidence bands"),
    ("🌊 Local Projections IRF (irf)", "irf ", "Jordà LP impulse response function with HAC ribbons"),
    ("📊 Synthetic Control Plot (synth_plot)", "synth_plot", "Two-panel treated vs synthetic trajectories and gap exhibit"),
    ("📊 Distribution Histogram", "histogram ", "Univariate distribution histogram"),
    ("📦 Box Plot (boxplot)", "boxplot ", "Tukey box plot with optional grouping"),
    ("🔥 Correlation Heatmap (heatmap)", "heatmap", "Matrix heatmap of numeric variable correlations"),
    ("🎨 Academic Coefficient Plot (coefplot)", "coefplot", "Publication-ready point estimates and confidence intervals"),
    ("🎯 Propensity Score Matching", "psmatch", "Nearest-neighbor covariate matching"),
    ("📑 Compare Stored Models (esttab)", "esttab", "Side-by-side regression comparison table"),
    ("📊 LaTeX Booktabs Exporter (esttab)", "esttab using table.tex, booktabs", "Multi-model publication LaTeX table"),
    ("📄 Export Word Document Report", "putdocx report.docx", "Generate native Word (.docx) report"),
    ("🔍 First-Stage Diagnostics (estat firststage)", "estat firststage", "First-stage regression and classical F diagnostic"),
    ("🚀 Big Data Streaming OLS (bigregress)", "bigregress ", "Out-of-core bounded-memory regression on large datasets or Parquet"),
    ("📂 Open Dataset...", "open_file", "Open .dta, .csv, or .parquet data file"),
    ("💾 Save Dataset...", "save_file", "Export active dataset to CSV or Parquet"),
    ("🤖 Command assistant (rule-based)", "ai", "Ask natural language statistical questions"),
    ("🌙 Toggle Dark / Light Theme", "toggle_theme", "Switch interface appearance mode")
]

class CommandPaletteDialog(QtWidgets.QDialog):
    action_triggered = Signal(str, str)

    def __init__(self, dataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.setWindowTitle("Command Palette")
        self.setFixedWidth(580)
        self.setFixedHeight(380)
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Search Bar
        self.search_box = QtWidgets.QLineEdit()
        self.search_box.setPlaceholderText("Type a command, model, variable or action (e.g. regress, vif, price)...")
        self.search_box.setFixedHeight(36)
        font = QtGui.QFont("Segoe UI", 11)
        self.search_box.setFont(font)
        self.search_box.textChanged.connect(self._filter_list)
        self.search_box.returnPressed.connect(self._execute_selected)
        layout.addWidget(self.search_box)

        # List Widget
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.itemActivated.connect(self._on_item_activated)
        layout.addWidget(self.list_widget)

        self._populate_list("")
        self.search_box.setFocus()

    def _populate_list(self, query):
        self.list_widget.clear()
        q = query.strip().lower()

        # Built-in actions
        for title, cmd, desc in ACTIONS:
            if not q or q in title.lower() or q in cmd.lower() or q in desc.lower():
                item = QtWidgets.QListWidgetItem(f"{title}\n   {desc}")
                item.setData(Qt.ItemDataRole.UserRole, (cmd, title))
                self.list_widget.addItem(item)

        # Variables in active dataset
        if not self.dataset.is_empty:
            for col in self.dataset.df.columns:
                if not q or q in col.lower():
                    vlabel = self.dataset.var_labels.get(col, "")
                    item = QtWidgets.QListWidgetItem(f"📊 Variable: {col}  ({vlabel})")
                    item.setData(Qt.ItemDataRole.UserRole, (f"summarize {col}", f"Variable {col}"))
                    item.setForeground(QtGui.QColor("#9CDCFE"))
                    self.list_widget.addItem(item)

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def _filter_list(self, text):
        self._populate_list(text)

    def _execute_selected(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            item = self.list_widget.item(row)
            self._on_item_activated(item)

    def _on_item_activated(self, item):
        cmd, title = item.data(Qt.ItemDataRole.UserRole)
        self.accept()
        self.action_triggered.emit(cmd, title)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        elif event.key() == Qt.Key.Key_Down:
            curr = self.list_widget.currentRow()
            if curr < self.list_widget.count() - 1:
                self.list_widget.setCurrentRow(curr + 1)
            return
        elif event.key() == Qt.Key.Key_Up:
            curr = self.list_widget.currentRow()
            if curr > 0:
                self.list_widget.setCurrentRow(curr - 1)
            return
        super().keyPressEvent(event)
