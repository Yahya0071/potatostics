"""
gui/data_editor.py - Interactive In-Place Spreadsheet Data Editor
Provides full two-way editing: double-click cell editing, row/column addition,
deletion, and live dataframe synchronization.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import customtkinter as ctk
import pandas as pd
import numpy as np
from gui.branding import set_window_icon

class DataEditorWindow(ctk.CTkToplevel):
    def __init__(self, parent, dataset, on_change_callback=None):
        super().__init__(parent)
        self.title("Data Editor - Potatostics")
        self.geometry("950x620")
        self.minsize(650, 420)
        set_window_icon(self)
        
        self.dataset = dataset
        self.on_change_callback = on_change_callback
        self.entry_popup = None

        self._build_ui()

    def _build_ui(self):
        is_dark = ctk.get_appearance_mode().lower() == "dark"
        tb_bg = "#2B2D30" if is_dark else "#E4E4E4"
        info_color = "#9CDCFE" if is_dark else "#005FB8"
        
        # 1. Action Toolbar
        toolbar = ctk.CTkFrame(self, height=38, corner_radius=0, fg_color=tb_bg)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        btn_add_row = ctk.CTkButton(toolbar, text="➕ Add Row", width=90, height=26, command=self._add_row)
        btn_add_row.pack(side=tk.LEFT, padx=5, pady=6)

        btn_add_col = ctk.CTkButton(toolbar, text="➕ Add Variable", width=110, height=26, command=self._add_col)
        btn_add_col.pack(side=tk.LEFT, padx=3, pady=6)

        btn_del_row = ctk.CTkButton(toolbar, text="➖ Delete Row", width=100, height=26,
                                    fg_color="#D9534F", hover_color="#C9302C", command=self._delete_selected_row)
        btn_del_row.pack(side=tk.LEFT, padx=3, pady=6)

        self.lbl_info = ctk.CTkLabel(toolbar, text=f"Obs: {self.dataset.N:,} | Vars: {self.dataset.K:,}",
                                     font=("Segoe UI", 11, "bold"), text_color=info_color)
        self.lbl_info.pack(side=tk.RIGHT, padx=12, pady=6)

        # 2. Grid with Scrollbars
        grid_frame = ttk.Frame(self)
        grid_frame.pack(fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(grid_frame, orient="vertical")
        hsb = ttk.Scrollbar(grid_frame, orient="horizontal")

        self.cols = ["_obs_"] + list(self.dataset.df.columns)
        self.tree = ttk.Treeview(grid_frame, columns=self.cols, show="headings",
                                 yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tree.heading("_obs_", text="obs")
        self.tree.column("_obs_", width=50, minwidth=40, anchor="center")

        for col in self.dataset.df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, minwidth=60, anchor="center")

        # Double-click cell edit binding
        self.tree.bind("<Double-1>", self._on_cell_double_click)
        self._populate_grid()

    def _populate_grid(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        limit = min(1000, self.dataset.N)
        for idx in range(limit):
            row_vals = [idx + 1]
            for col in self.dataset.df.columns:
                val = self.dataset.df.at[idx, col]
                vl_name = self.dataset.val_label_names.get(col, None)
                if vl_name and vl_name in self.dataset.val_labels and val in self.dataset.val_labels[vl_name]:
                    row_vals.append(self.dataset.val_labels[vl_name][val])
                elif pd.isna(val):
                    row_vals.append(".")
                else:
                    row_vals.append(str(val))
            self.tree.insert("", "end", values=row_vals)

        self.lbl_info.configure(text=f"Obs: {self.dataset.N:,} | Vars: {self.dataset.K:,}")

    def _on_cell_double_click(self, event):
        region = self.tree.identify_region(event.x, event.y)
        if region != "cell":
            return

        column_id = self.tree.identify_column(event.x)
        row_id = self.tree.identify_row(event.y)
        if not row_id or not column_id:
            return

        col_index = int(column_id.replace("#", "")) - 1
        if col_index == 0:
            # Observation column cannot be directly edited
            return

        col_name = self.cols[col_index]
        item_vals = self.tree.item(row_id, "values")
        obs_num = int(item_vals[0]) - 1  # 0-indexed row in DataFrame
        current_val = item_vals[col_index]

        # Geometry of the cell
        x, y, w, h = self.tree.bbox(row_id, column_id)

        # In-place Entry widget
        if self.entry_popup:
            self.entry_popup.destroy()

        self.entry_popup = ttk.Entry(self.tree, font=("Consolas", 10))
        self.entry_popup.insert(0, "" if current_val == "." else current_val)
        self.entry_popup.select_range(0, tk.END)
        self.entry_popup.place(x=x, y=y, width=w, height=h)
        self.entry_popup.focus_set()

        def save_cell_edit(event=None):
            if not self.entry_popup:
                return
            new_text = self.entry_popup.get().strip()
            self.entry_popup.destroy()
            self.entry_popup = None

            # Convert and save to DataFrame
            if new_text == "" or new_text == ".":
                self.dataset.df.at[obs_num, col_name] = np.nan
            else:
                try:
                    if pd.api.types.is_numeric_dtype(self.dataset.df[col_name]):
                        if "." in new_text:
                            new_val = float(new_text)
                        else:
                            new_val = int(new_text)
                    else:
                        new_val = new_text
                    self.dataset.df.at[obs_num, col_name] = new_val
                except ValueError:
                    messagebox.showerror("Type Error", f"Cannot convert '{new_text}' to column type {self.dataset.df[col_name].dtype}.")
                    return

            self._populate_grid()
            if self.on_change_callback:
                self.on_change_callback()

        self.entry_popup.bind("<Return>", save_cell_edit)
        self.entry_popup.bind("<FocusOut>", lambda e: save_cell_edit())
        self.entry_popup.bind("<Escape>", lambda e: self.entry_popup.destroy())

    def _add_row(self):
        new_row = {col: np.nan for col in self.dataset.df.columns}
        self.dataset.df = pd.concat([self.dataset.df, pd.DataFrame([new_row])], ignore_index=True)
        self._populate_grid()
        if self.on_change_callback:
            self.on_change_callback()

    def _add_col(self):
        varname = simpledialog.askstring("Add Variable", "Enter new variable name:", parent=self)
        if varname and varname.strip():
            name = varname.strip()
            if name in self.dataset.df.columns:
                messagebox.showwarning("Exists", f"Variable '{name}' already exists.")
                return
            self.dataset.df[name] = np.nan
            self.dataset.var_labels[name] = ""
            self.dataset.var_formats[name] = "%9.0g"
            
            # Update columns
            self.cols.append(name)
            self.tree["columns"] = self.cols
            self.tree.heading(name, text=name)
            self.tree.column(name, width=100, minwidth=60, anchor="center")
            self._populate_grid()
            if self.on_change_callback:
                self.on_change_callback()

    def _delete_selected_row(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Select Row", "Please click on a row to delete.")
            return
        item_vals = self.tree.item(selected[0], "values")
        obs_idx = int(item_vals[0]) - 1
        self.dataset.df.drop(index=obs_idx, inplace=True)
        self.dataset.df.reset_index(drop=True, inplace=True)
        self._populate_grid()
        if self.on_change_callback:
            self.on_change_callback()
