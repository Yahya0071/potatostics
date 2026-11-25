"""
gui/branding.py - Visual Identity & Brand Assets Manager for Potatostics
Provides:
- Resolution of brand assets from the logo/ directory (both in dev and frozen builds).
- Cross-platform window icon setter for main and child windows.
- Branded CTkImage generators for toolbars, headers, and dialogs.
- Rich About Dialog showcasing the official Potatostics visual identity.
"""

import os
import tkinter as tk
import customtkinter as ctk
from PIL import Image
from assets import get_asset_path


def set_window_icon(window):
    """
    Applies the official Potatostics icon (.ico and .png) to a Tkinter or CustomTkinter window.
    Handles both Windows title bar icons and high-DPI taskbar icons.
    """
    ico_path = get_asset_path("potatostics.ico")
    png_path = get_asset_path("potatostics-app-icon-32.png") or get_asset_path("potatostics-app-icon.png")

    if ico_path and os.path.exists(ico_path):
        try:
            window.iconbitmap(default=ico_path)
        except Exception:
            try:
                window.iconbitmap(ico_path)
            except Exception:
                pass

    if png_path and os.path.exists(png_path):
        try:
            img = tk.PhotoImage(file=png_path)
            window.iconphoto(False, img)
            # Retain reference to prevent garbage collection
            window._potatostics_icon_ref = img
        except Exception:
            pass


def get_brand_icon_image(size=(26, 26)):
    """Returns a CTkImage of the Potatostics app mark icon for toolbars and buttons."""
    png_path = get_asset_path("potatostics-app-icon-64.png") or get_asset_path("potatostics-app-icon.png")
    if png_path and os.path.exists(png_path):
        try:
            pil_img = Image.open(png_path)
            return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=size)
        except Exception:
            return None
    return None


def get_horizontal_logo_image(size=(260, 56)):
    """Returns a dynamic CTkImage of the horizontal logo banner supporting Dark/Light themes."""
    dark_path = get_asset_path("potatostics-logo-horizontal.png") or get_asset_path("potatostics-logo-horizontal-dark.png")
    light_path = get_asset_path("potatostics-logo-horizontal-dark.png") or get_asset_path("potatostics-logo-horizontal.png")

    if dark_path and os.path.exists(dark_path):
        try:
            pil_dark = Image.open(dark_path)
            pil_light = Image.open(light_path) if light_path and os.path.exists(light_path) else pil_dark
            return ctk.CTkImage(light_image=pil_light, dark_image=pil_dark, size=size)
        except Exception:
            return None
    return None


class AboutDialog(ctk.CTkToplevel):
    """Modern modal dialog displaying Potatostics branding, features, and license."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("About Potatostics")
        self.geometry("540x580")
        self.resizable(False, False)
        set_window_icon(self)

        # Modal behavior
        self.transient(parent)
        self.grab_set()

        self._build_ui()

    def _build_ui(self):
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        bg_card = "#252526" if is_dark else "#FFFFFF"
        text_dim = "#A0A0A0" if is_dark else "#606060"
        border_col = "#3E3E42" if is_dark else "#E0E0E0"

        # Top Banner with Horizontal Logo or Icon
        logo_img = get_horizontal_logo_image(size=(280, 60))
        if logo_img:
            lbl_logo = ctk.CTkLabel(self, image=logo_img, text="")
            lbl_logo.pack(pady=(20, 10))
        else:
            icon_img = get_brand_icon_image(size=(64, 64))
            if icon_img:
                lbl_icon = ctk.CTkLabel(self, image=icon_img, text="")
                lbl_icon.pack(pady=(18, 6))
            lbl_title = ctk.CTkLabel(self, text="POTATOSTICS", font=("Consolas", 22, "bold"))
            lbl_title.pack()

        lbl_subtitle = ctk.CTkLabel(
            self,
            text="Independent Statistical Analysis Software & Econometric IDE",
            font=("Segoe UI", 12, "bold"),
            text_color=("#005FB8", "#4EC9B0")
        )
        lbl_subtitle.pack(pady=(0, 14))

        # Content Card
        card = ctk.CTkFrame(self, fg_color=bg_card, corner_radius=8, border_width=1, border_color=border_col)
        card.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 16))

        info_text = (
            "Release 1.0  |  Open Source  |  LGPL-3.0-or-later License\n\n"
            "Key Capabilities:\n"
            "  • Unified Command Parser & Expression Evaluator\n"
            "  • OLS with Classical, HC1, HC2, HC3 & Cluster-Robust Covariances\n"
            "  • Logistic Regression with Odds Ratios & Clustered Standard Errors\n"
            "  • Two-Stage Least Squares (2SLS IV) with Endogeneity & OverID tests\n"
            "  • Panel Fixed Effects Within Estimator with Unit Effect Tests\n"
            "  • Time Series Analysis: ADF Unit Root, Correlogram, and ARIMA\n"
            "  • Quasi-Experimental Causal Inference: DiD & PSMatch\n"
            "  • Automated Word Document Report Generation (.docx)\n"
            "  • Command assistant (rule-based) & User Extensible Plugin Architecture\n\n"
            "Visual Identity:\n"
            "  Geometric organic mark enclosing a pure mathematical Gaussian\n"
            "  normal distribution: y = exp(-0.5 * t^2)."
        )

        lbl_info = ctk.CTkLabel(
            card,
            text=info_text,
            font=("Consolas", 10),
            justify=tk.LEFT,
            text_color=("#1E1E1E", "#D4D4D4")
        )
        lbl_info.pack(padx=16, pady=14, anchor="w")

        # Bottom Actions
        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.pack(fill=tk.X, padx=24, pady=(0, 16))

        lbl_copyright = ctk.CTkLabel(
            bottom_frame,
            text="Created with Python, NumPy, SciPy, Pandas & CustomTkinter",
            font=("Segoe UI", 9),
            text_color=text_dim
        )
        lbl_copyright.pack(side=tk.LEFT)

        btn_close = ctk.CTkButton(
            bottom_frame,
            text="Close",
            width=80,
            command=self.destroy
        )
        btn_close.pack(side=tk.RIGHT)
