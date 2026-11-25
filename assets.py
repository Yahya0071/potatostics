"""
assets.py - Neutral Brand Asset & Resource Locator for Potatostics
Zero-dependency module: pure standard library (os, sys).
Does NOT import tkinter, customtkinter, PIL, or PySide6.
Usable by gui_qt, legacy gui, tests, and CLI packaging.
"""

import os
import sys

def get_asset_path(filename):
    """
    Locates an asset across package directories, development sources,
    and PyInstaller frozen runtime environments.
    """
    candidates = []
    
    # 1. PyInstaller bundled temp directory
    if hasattr(sys, '_MEIPASS'):
        candidates.append(os.path.join(sys._MEIPASS, "logo", "assets", filename))
        candidates.append(os.path.join(sys._MEIPASS, "assets", filename))
        candidates.append(os.path.join(sys._MEIPASS, "gui", "assets", filename))
        candidates.append(os.path.join(sys._MEIPASS, filename))

    # 2. Package assets directory relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(base_dir, "gui_qt", "assets", filename))
    candidates.append(os.path.join(base_dir, "gui_qt", filename))
    candidates.append(os.path.join(base_dir, "gui", "assets", filename))
    candidates.append(os.path.join(base_dir, "gui", filename))
    candidates.append(os.path.join(base_dir, "logo", "assets", filename))
    candidates.append(os.path.join(base_dir, "logo", filename))
    candidates.append(os.path.join(base_dir, "assets", filename))
    candidates.append(os.path.join(base_dir, filename))

    # 3. Current working directory
    candidates.append(os.path.join(os.getcwd(), "logo", "assets", filename))
    candidates.append(os.path.join(os.getcwd(), "gui", "assets", filename))
    candidates.append(os.path.join(os.getcwd(), "assets", filename))
    candidates.append(os.path.join(os.getcwd(), filename))

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    return None
