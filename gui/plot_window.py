"""
gui/plot_window.py - High-Resolution Chart Studio
Embeds Matplotlib charts (scatter with OLS, histograms, boxplots, correlation heatmaps)
with vector export (SVG, PDF, PNG) and Dark/Light theme support.
"""

import tkinter as tk
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import customtkinter as ctk
import numpy as np
from gui.branding import set_window_icon

class PlotWindow(tk.Toplevel):
    def __init__(self, parent, plot_type, df, **kwargs):
        super().__init__(parent)
        self.title("Graph Studio - Potatostics")
        self.geometry("800x580")
        self.minsize(580, 440)
        set_window_icon(self)
        
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        bg_win = "#1E1E1E" if is_dark else "#F3F3F3"
        bg_fig = "#1E1E1E" if is_dark else "#FFFFFF"
        bg_ax = "#252526" if is_dark else "#FAFAFA"
        fg_text = "#FFFFFF" if is_dark else "#1E1E1E"
        fg_ticks = "#D4D4D4" if is_dark else "#333333"
        fg_spine = "#555555" if is_dark else "#CCCCCC"
        
        self.configure(bg=bg_win)
        
        self.fig = Figure(figsize=(7.5, 5.2), dpi=100, facecolor=bg_fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(bg_ax)
        
        self.ax.tick_params(colors=fg_ticks, which="both")
        for spine in self.ax.spines.values():
            spine.set_color(fg_spine)
        self.ax.xaxis.label.set_color(fg_text)
        self.ax.yaxis.label.set_color(fg_text)
        self.ax.title.set_color(fg_text)
        
        if plot_type == "scatter":
            self._render_scatter(df, kwargs.get("yvar"), kwargs.get("xvar"), is_dark)
        elif plot_type == "histogram":
            self._render_histogram(df, kwargs.get("var"), is_dark)
        elif plot_type == "boxplot":
            self._render_boxplot(df, kwargs.get("var"), kwargs.get("byvar"), is_dark)
        elif plot_type == "heatmap":
            self._render_heatmap(df, kwargs.get("vars"), is_dark)

        self.fig.tight_layout()
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Navigation toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def _render_scatter(self, df, yvar, xvar, is_dark):
        sub = df[[xvar, yvar]].dropna()
        x = sub[xvar].to_numpy(dtype=float)
        y = sub[yvar].to_numpy(dtype=float)
        
        dot_color = "#4EC9B0" if is_dark else "#005FB8"
        line_color = "#F44747" if is_dark else "#D83B01"
        
        self.ax.scatter(x, y, color=dot_color, alpha=0.8, edgecolors="none", s=45, label="Data")
        
        # OLS trend line
        if len(x) > 1:
            poly = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = np.polyval(poly, x_line)
            self.ax.plot(x_line, y_line, color=line_color, linewidth=2, label=f"Fit: y = {poly[0]:.2f}x + {poly[1]:.2f}")
            self.ax.legend(facecolor="#252526" if is_dark else "#FFFFFF", edgecolor="#555555" if is_dark else "#CCCCCC", labelcolor="#FFFFFF" if is_dark else "#1E1E1E")
            
        self.ax.set_title(f"Scatter: {yvar} vs {xvar}", fontsize=13, fontweight="bold")
        self.ax.set_xlabel(xvar, fontsize=11)
        self.ax.set_ylabel(yvar, fontsize=11)
        self.ax.grid(True, linestyle="--", alpha=0.3, color="#888888")

    def _render_histogram(self, df, var, is_dark):
        sub = df[var].dropna().to_numpy(dtype=float)
        bar_color = "#007ACC" if is_dark else "#0078D7"
        edge_color = "#FFFFFF" if is_dark else "#333333"
        
        self.ax.hist(sub, bins="auto", color=bar_color, edgecolor=edge_color, alpha=0.85)
        self.ax.set_title(f"Histogram of {var}", fontsize=13, fontweight="bold")
        self.ax.set_xlabel(var, fontsize=11)
        self.ax.set_ylabel("Frequency", fontsize=11)
        self.ax.grid(True, linestyle="--", alpha=0.3, color="#888888")

    def _render_boxplot(self, df, var, byvar, is_dark):
        if byvar and byvar in df.columns:
            groups = []
            labels = []
            for name, g in df.groupby(byvar):
                vals = g[var].dropna().to_numpy(dtype=float)
                if len(vals) > 0:
                    groups.append(vals)
                    labels.append(str(name))
            bp = self.ax.boxplot(groups, tick_labels=labels, patch_artist=True)
            self.ax.set_title(f"Boxplot of {var} by {byvar}", fontsize=13, fontweight="bold")
            self.ax.set_xlabel(byvar, fontsize=11)
        else:
            vals = df[var].dropna().to_numpy(dtype=float)
            bp = self.ax.boxplot([vals], tick_labels=[var], patch_artist=True)
            self.ax.set_title(f"Boxplot of {var}", fontsize=13, fontweight="bold")
            
        fill_color = "#2D5A27" if is_dark else "#107C41"
        for patch in bp['boxes']:
            patch.set_facecolor(fill_color)
            patch.set_alpha(0.8)
        self.ax.set_ylabel(var, fontsize=11)
        self.ax.grid(True, linestyle="--", alpha=0.3, color="#888888")

    def _render_heatmap(self, df, vars_list, is_dark):
        if vars_list:
            cols = [v for v in vars_list if v in df.columns and df[v].dtype != "object"]
        else:
            cols = [c for c in df.columns if df[c].dtype != "object"]
            
        if len(cols) < 2:
            return
            
        corr_matrix = df[cols].corr().to_numpy()
        cax = self.ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        self.fig.colorbar(cax, ax=self.ax)
        
        self.ax.set_xticks(range(len(cols)))
        self.ax.set_yticks(range(len(cols)))
        self.ax.set_xticklabels(cols, rotation=45, ha="left")
        self.ax.set_yticklabels(cols)
        self.ax.set_title("Correlation Heatmap Matrix", fontsize=13, fontweight="bold", pad=20)
        
        for i in range(len(cols)):
            for j in range(len(cols)):
                val = corr_matrix[i, j]
                self.ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                             color="#000000" if 0.3 < abs(val) < 0.7 else "#FFFFFF",
                             fontsize=9, fontweight="bold")
