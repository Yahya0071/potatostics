"""
gui_qt/data_grid.py - Virtualized Data Spreadsheet
Implements:
- QAbstractTableModel (VirtualPandasModel) supporting multi-million row datasets
- Viewport-only lazy rendering and memory virtualization
- In-place double-click cell editing with strict dtype validation (no silent integer truncations or boolean coercions)
- Synchronized sorted and filtered views ensuring canonical observation order
- Value label resolution (displays labeled text while editing underlying integer code)
- Live search/filter bar and row/column addition toolbar
- Transactional Find & Replace (exact numeric matching, missing value preservation, categorical safety)
- Transactional multi-cell paste validation
"""

from decimal import Decimal, InvalidOperation
import re
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

def to_nullable_integer_dtype(dtype) -> str:
    """Map numpy integer dtypes to corresponding pandas nullable integer ExtensionDtypes."""
    mapping = {
        'int64': 'Int64',
        'int32': 'Int32',
        'int16': 'Int16',
        'int8': 'Int8',
        'uint64': 'UInt64',
        'uint32': 'UInt32',
        'uint16': 'UInt16',
        'uint8': 'UInt8',
    }
    try:
        np_dt = dtype.numpy_dtype if hasattr(dtype, 'numpy_dtype') else dtype
        name = np.dtype(np_dt).name
        return mapping.get(name, 'Int64')
    except Exception:
        return 'Int64'

def _parse_exact_integer(text: str, dtype) -> int | None:
    """Accept integral decimal input only within the destination integer range."""
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    if not value.is_finite() or value != value.to_integral_value():
        return None
    bounds = np.iinfo(getattr(dtype, "numpy_dtype", dtype))
    if not bounds.min <= value <= bounds.max:
        return None
    return int(value)


def validate_float_input(str_val: str, target_dtype) -> tuple[bool, object, str]:
    """
    Validates floating-point string inputs against destination dtype representability.
    Enforces safe floating-point policies:
    1. Parse values safely without silent overflow to infinity.
    2. Detect underflow where nonzero input materially underflows to zero.
    3. Keep explicitly entered integers exact.
    4. Support normal IEEE floating-point representation rounding for ordinary decimals.
    5. Handle explicitly requested NaN and infinity.
    Returns (is_valid, parsed_value, error_message).
    """
    if str_val is None:
        return True, np.nan, ""
    clean = str(str_val).strip()
    if clean in (".", "", "na", "<na>", "nan", "NaN"):
        return True, np.nan, ""

    lower = clean.lower()
    dtype = np.dtype(getattr(target_dtype, "numpy_dtype", target_dtype))
    scalar = dtype.type
    if lower in ("inf", "+inf", "infinity", "+infinity"):
        return True, scalar(np.inf), ""
    if lower in ("-inf", "-infinity"):
        return True, scalar(-np.inf), ""

    try:
        decimal_value = Decimal(clean)
        with np.errstate(over="ignore", under="ignore"):
            parsed = scalar(clean)
    except OverflowError:
        return False, None, f"Input '{clean}' overflows {dtype}"
    except (InvalidOperation, ValueError, TypeError):
        return False, None, f"Invalid numeric input '{clean}'"

    if np.isnan(parsed):
        return True, parsed, ""
    if np.isinf(parsed):
        return False, None, f"Input '{clean}' overflows {dtype} to infinity"
    if parsed == 0 and not decimal_value.is_zero():
        return False, None, f"Input '{clean}' underflows {dtype} to zero"
    if decimal_value == decimal_value.to_integral_value() and int(parsed) != decimal_value:
        # A rounded scientific-format value, such as the displayed float32 maximum,
        # can still identify the stored value by its shortest round-trip spelling.
        if "e" not in lower or Decimal(str(parsed)) != decimal_value:
            return False, None, f"Integer '{clean}' cannot be represented exactly in {dtype} without precision loss"
    return True, parsed, ""

def parse_categorical_value(str_val: str, series: pd.Series, val_labels: dict | None = None) -> tuple[bool, object, str]:
    """
    Parses user input into a category value matching the category's underlying data type.
    Preserves integer, float, boolean, datetime, and string category types.
    Resolves value labels if available.
    Returns (is_valid, parsed_category_value, error_message).
    """
    if str_val is None:
        return True, np.nan, ""
    clean = str(str_val).strip()
    if clean in (".", "", "na", "<na>", "nan", "NaN"):
        return True, np.nan, ""

    # Check value labels (e.g. {1: "Male", 2: "Female"})
    if val_labels:
        inv_labels = {str(v).strip().lower(): k for k, v in val_labels.items()}
        if clean.lower() in inv_labels:
            clean = str(inv_labels[clean.lower()])

    cat_index = series.cat.categories
    cat_dtype = cat_index.dtype

    if pd.api.types.is_integer_dtype(cat_dtype):
        int_val = _parse_exact_integer(clean, cat_dtype)
        if int_val is None:
            return False, None, f"'{clean}' is not an exact integer in the range of {cat_dtype}"
        if int_val in cat_index:
            return True, int_val, ""
        np_dt = getattr(cat_dtype, "numpy_dtype", cat_dtype)
        return True, np.dtype(np_dt).type(int_val), ""

    elif pd.api.types.is_float_dtype(cat_dtype):
        ok, flt_val, err = validate_float_input(clean, cat_dtype)
        if not ok:
            return False, None, err
        if flt_val in cat_index:
            return True, flt_val, ""
        return True, flt_val, ""

    elif pd.api.types.is_bool_dtype(cat_dtype):
        lower_s = clean.lower()
        if lower_s in ('true', '1', 'yes', 't', 'y'):
            b_val = True
        elif lower_s in ('false', '0', 'no', 'f', 'n'):
            b_val = False
        else:
            return False, None, f"Cannot parse '{clean}' as boolean category"
        return True, b_val, ""

    elif pd.api.types.is_datetime64_any_dtype(cat_dtype):
        try:
            dt_val = pd.to_datetime(clean)
            return True, dt_val, ""
        except Exception:
            return False, None, f"Cannot parse '{clean}' as datetime category"

    else:
        # String or object categories
        if clean in cat_index:
            return True, clean, ""
        return True, clean, ""

def insert_blank_observation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends a single blank observation to df while preserving exact dtypes, precision, and index identity.
    Numpy integer and boolean columns are promoted to nullable ExtensionDtypes so that
    missingness does not force float64 coercion (preventing >2^53 integer corruption).
    Preserves index labels, index name, duplicate indices, and observation ordering without overflow.
    """
    if df is None:
        return None
    if len(df.columns) == 0:
        return df.copy()

    df = df.copy()
    for col in df.columns:
        dt = df[col].dtype
        if pd.api.types.is_integer_dtype(dt) and not isinstance(dt, pd.core.dtypes.dtypes.ExtensionDtype):
            df[col] = df[col].astype(to_nullable_integer_dtype(dt))
        elif pd.api.types.is_bool_dtype(dt) and not isinstance(dt, pd.BooleanDtype):
            df[col] = df[col].astype('boolean')

    new_cols = {}
    for col in df.columns:
        dt = df[col].dtype
        if pd.api.types.is_timedelta64_dtype(dt) or pd.api.types.is_datetime64_any_dtype(dt):
            new_cols[col] = pd.Series([pd.NaT], dtype=dt)
        elif isinstance(dt, pd.CategoricalDtype):
            new_cols[col] = pd.Series([np.nan], dtype=dt)
        elif isinstance(dt, pd.core.dtypes.dtypes.ExtensionDtype):
            new_cols[col] = pd.Series([pd.NA], dtype=dt)
        elif pd.api.types.is_float_dtype(dt):
            new_cols[col] = pd.Series([np.nan], dtype=dt)
        else:
            new_cols[col] = pd.Series([None], dtype=object)

    row_df = pd.DataFrame(new_cols)

    if isinstance(df.index, pd.RangeIndex):
        start = df.index.start
        step = df.index.step
        new_len = len(df) + 1
        stop = start + new_len * step
        new_idx = pd.RangeIndex(start=start, stop=stop, step=step, name=df.index.name)
        res = pd.concat([df, row_df])
        res.index = new_idx
    elif pd.api.types.is_integer_dtype(df.index.dtype):
        idx_dtype = df.index.dtype
        np_dt = idx_dtype.numpy_dtype if hasattr(idx_dtype, "numpy_dtype") else idx_dtype
        try:
            info = np.iinfo(np_dt)
        except Exception:
            info = np.iinfo(np.int64)

        if len(df) == 0:
            candidate = 0
        else:
            max_val = df.index.max()
            if max_val < info.max:
                candidate = int(max_val) + 1
            else:
                # Bound reached: select smallest available non-negative integer (MEX)
                existing = set(df.index.unique())
                candidate = None
                for c in range(0, min(len(existing) + 1, info.max + 1)):
                    if c not in existing:
                        candidate = c
                        break
                if candidate is None and info.min < 0:
                    for c in range(-1, max(-len(existing) - 2, info.min - 1), -1):
                        if c not in existing:
                            candidate = c
                            break
                if candidate is None:
                    return df.copy()

        row_df.index = pd.Index([candidate], dtype=df.index.dtype)
        res = pd.concat([df, row_df])
    else:
        existing = set(df.index.unique())
        cand_str = f"obs_{len(df) + 1}"
        k = 1
        while cand_str in existing:
            k += 1
            cand_str = f"obs_{len(df) + k}"
        row_df.index = [cand_str]
        res = pd.concat([df, row_df])

    if df.index.name:
        res.index.name = df.index.name
    return res

class VirtualPandasModel(QtCore.QAbstractTableModel):
    """
    High-performance virtualized model wrapping a pandas DataFrame.
    Uses view-level row mapping for sorting and filtering so that:
      1. Edits in filtered/sorted views update the canonical dataset.df directly positionally.
      2. View header-click sorting is strictly view-only and never mutates dataset observation order.
      3. Edits that alter filter membership or sort order automatically synchronize the view.
    """
    def __init__(self, dataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self._sort_col = None
        self._sort_ascending = True
        self._filter_text = ""
        self._row_map = None
        self._is_readonly = False
        self._col_names = []
        self._col_is_numeric = []
        self._alignments = []
        self._col_data = []
        self._recompute_row_map()

    @property
    def is_readonly(self) -> bool:
        return bool(getattr(self, "_is_readonly", False))

    def set_readonly(self, readonly: bool):
        self._is_readonly = bool(readonly)

    @property
    def _df(self):
        if self.dataset.df is None:
            return None
        if self._row_map is not None:
            return self.dataset.df.iloc[self._row_map]
        return self.dataset.df

    def _refresh_column_cache(self):
        df = self.dataset.df
        if df is None or df.empty:
            self._col_names = []
            self._col_is_numeric = []
            self._alignments = []
            self._col_data = []
            return

        self._col_names = list(df.columns)
        self._col_is_numeric = [pd.api.types.is_numeric_dtype(df[c]) for c in df.columns]
        self._alignments = [
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter) if is_num
            else int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            for is_num in self._col_is_numeric
        ]
        self._col_data = [df[c].values for c in df.columns]

    def _recompute_row_map(self):
        if self.dataset.df is None or self.dataset.df.empty:
            self._row_map = None
            self._refresh_column_cache()
            return

        n_rows = len(self.dataset.df)
        indices = np.arange(n_rows)

        # 1. Apply filter if present
        if self._filter_text:
            if self._filter_text == ".":
                mask = self.dataset.df.isna().any(axis=1).to_numpy()
            else:
                match_matrix = []
                for c in self.dataset.df.columns:
                    col_s = self.dataset.df[c]
                    na_m = col_s.isna().to_numpy()
                    s_str = col_s.astype(str).str.lower().to_numpy()
                    c_match = (~na_m) & (np.char.find(s_str.astype(str), self._filter_text) >= 0)
                    match_matrix.append(c_match)
                mask = np.logical_or.reduce(match_matrix)
            indices = indices[mask]

        # 2. Apply sort if present (view-only)
        if self._sort_col and self._sort_col in self.dataset.df.columns and len(indices) > 0:
            sub_vals = self.dataset.df[self._sort_col].iloc[indices].values
            sort_df = pd.DataFrame({'pos': np.arange(len(indices)), 'val': sub_vals})
            sorted_pos = sort_df.sort_values(by='val', ascending=self._sort_ascending, kind='stable', na_position='last')['pos'].to_numpy()
            indices = indices[sorted_pos]

        self._row_map = indices
        self._refresh_column_cache()

    def _map_row(self, row):
        if self._row_map is not None and row < len(self._row_map):
            return int(self._row_map[row])
        return row

    def set_filter(self, text):
        self.beginResetModel()
        self._filter_text = text.strip().lower()
        self._recompute_row_map()
        self.endResetModel()

    def update_data(self):
        self.beginResetModel()
        self._recompute_row_map()
        self.endResetModel()

    def rowCount(self, parent=None):
        if self._row_map is not None:
            return len(self._row_map)
        return len(self.dataset.df) if self.dataset.df is not None else 0

    def columnCount(self, parent=None):
        return len(self._col_names) if self._col_names else (len(self.dataset.df.columns) if self.dataset.df is not None and not self.dataset.df.empty else 0)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not self._col_names:
            return None

        actual_row = self._map_row(index.row())
        col = index.column()
        if col >= len(self._col_names) or actual_row >= len(self.dataset.df):
            return None

        col_name = self._col_names[col]
        val = self._col_data[col][actual_row]

        if role == Qt.ItemDataRole.DisplayRole:
            if pd.isna(val):
                return "."
            
            # Value label resolution
            lbl_name = self.dataset.val_label_names.get(col_name) if hasattr(self.dataset, 'val_label_names') else None
            label_dict = (self.dataset.val_labels.get(lbl_name) if lbl_name else None) or (self.dataset.val_labels.get(col_name) if hasattr(self.dataset, 'val_labels') else None)
            if label_dict:
                try:
                    int_key = int(val)
                    if int_key in label_dict:
                        return f"{label_dict[int_key]}"
                except (ValueError, TypeError):
                    pass

            # Numeric formatting
            if isinstance(val, (float, np.floating)):
                return f"{val:.4f}".rstrip('0').rstrip('.') if not float(val).is_integer() else f"{int(val)}"
            return str(val)

        elif role == Qt.ItemDataRole.EditRole:
            return "" if pd.isna(val) else str(val)

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return self._alignments[col]

        elif role == Qt.ItemDataRole.ToolTipRole:
            var_label = self.dataset.var_labels.get(col_name, "")
            dtype_str = str(self.dataset.df[col_name].dtype)
            lbl_info = f"\nLabel: {var_label}" if var_label else ""
            return f"Variable: {col_name} ({dtype_str}){lbl_info}\nObs #{actual_row + 1}: {val}"

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                if self.dataset.df is not None and section < len(self.dataset.df.columns):
                    col_name = self.dataset.df.columns[section]
                    sort_arrow = ""
                    if self._sort_col == col_name:
                        sort_arrow = " ▲" if self._sort_ascending else " ▼"
                    return f"{col_name}{sort_arrow}"
            else:
                return str(section + 1)
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        if self.is_readonly:
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or role != Qt.ItemDataRole.EditRole or self.dataset.df is None:
            return False
        if self.is_readonly:
            return False

        actual_row = self._map_row(index.row())
        col = index.column()
        if col >= len(self.dataset.df.columns) or actual_row >= len(self.dataset.df):
            return False
        col_name = self.dataset.df.columns[col]
        dtype = self.dataset.df[col_name].dtype

        # Cast value appropriately with strict validation
        str_val = str(value).strip() if value is not None else ""
        try:
            if str_val == "." or str_val == "" or (isinstance(value, str) and value.strip() == "."):
                if pd.api.types.is_integer_dtype(dtype):
                    if isinstance(dtype, pd.core.dtypes.dtypes.ExtensionDtype):
                        new_val = pd.NA
                    else:
                        # Standard numpy integer cannot store NaN; upcast column to pandas nullable integer to preserve large-integer precision
                        target_dtype = to_nullable_integer_dtype(dtype)
                        self.dataset.df[col_name] = self.dataset.df[col_name].astype(target_dtype)
                        dtype = self.dataset.df[col_name].dtype
                        new_val = pd.NA
                elif pd.api.types.is_bool_dtype(dtype):
                    if isinstance(dtype, pd.BooleanDtype):
                        new_val = pd.NA
                    else:
                        return False
                elif pd.api.types.is_float_dtype(dtype):
                    new_val = np.nan
                else:
                    new_val = None if pd.api.types.is_object_dtype(dtype) else np.nan

            elif pd.api.types.is_integer_dtype(dtype):
                # Strict integer validation: Reject fractional floats, invalid text, out-of-range values
                # Never silently truncate via int(float(str_val))!
                try:
                    int_val = int(str_val)
                except (ValueError, TypeError):
                    return False

                # Check representable bounds for dtype
                try:
                    np_dtype = dtype.numpy_dtype if hasattr(dtype, "numpy_dtype") else dtype
                    info = np.iinfo(np_dtype)
                    if int_val < info.min or int_val > info.max:
                        return False
                except (ValueError, TypeError):
                    pass
                new_val = int_val

            elif pd.api.types.is_float_dtype(dtype):
                ok, parsed_flt, err_msg = validate_float_input(str_val, dtype)
                if not ok:
                    if hasattr(self.parent(), "lbl_status") and self.parent() and self.parent().lbl_status:
                        self.parent().lbl_status.setText(f"Edit rejected: {err_msg}")
                    return False
                new_val = parsed_flt

            elif pd.api.types.is_bool_dtype(dtype):
                # Strict boolean validation: explicit tokens only
                lower_val = str_val.lower()
                if lower_val in ('true', '1', 'yes', 't', 'y'):
                    new_val = True
                elif lower_val in ('false', '0', 'no', 'f', 'n'):
                    new_val = False
                elif lower_val in ('.', '', 'na', '<na>'):
                    if isinstance(dtype, pd.BooleanDtype):
                        new_val = pd.NA
                    else:
                        return False
                else:
                    # Reject unknown tokens
                    return False

            elif isinstance(dtype, pd.CategoricalDtype):
                lbl_name = self.dataset.val_label_names.get(col_name) if hasattr(self.dataset, 'val_label_names') else None
                val_labels = (self.dataset.val_labels.get(lbl_name) if lbl_name else None) or (self.dataset.val_labels.get(col_name) if hasattr(self.dataset, 'val_labels') else None)
                ok, cat_val, err_msg = parse_categorical_value(str_val, self.dataset.df[col_name], val_labels)
                if not ok:
                    if hasattr(self.parent(), "lbl_status") and self.parent() and self.parent().lbl_status:
                        self.parent().lbl_status.setText(f"Edit rejected: {err_msg}")
                    return False
                if pd.isna(cat_val):
                    new_val = np.nan
                else:
                    if cat_val not in self.dataset.df[col_name].cat.categories:
                        self.dataset.df[col_name] = self.dataset.df[col_name].cat.add_categories([cat_val])
                    new_val = cat_val

            else:
                new_val = str_val

            # Write positionally to canonical dataset.df
            self.dataset.df.iloc[actual_row, col] = new_val
            self._refresh_column_cache()

            # Synchronize view if filter or actively sorted column was modified
            recompute_view = False
            if self._filter_text:
                recompute_view = True
            elif self._sort_col and col_name == self._sort_col:
                recompute_view = True

            if recompute_view:
                self.beginResetModel()
                self._recompute_row_map()
                self.endResetModel()
            else:
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole])
            return True
        except Exception:
            return False

    def sort_by_column(self, col_index):
        if self.dataset.df is None or self.dataset.df.empty or col_index >= len(self.dataset.df.columns):
            return
        col_name = self.dataset.df.columns[col_index]
        if self._sort_col == col_name:
            self._sort_ascending = not self._sort_ascending
        else:
            self._sort_col = col_name
            self._sort_ascending = True

        self.beginResetModel()
        self._recompute_row_map()
        self.endResetModel()


class DataGridWidget(QtWidgets.QWidget):
    """Spreadsheet widget housing the virtualized table, toolbar, and status bar."""

    def __init__(self, dataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self._is_readonly = False
        self._build_ui()

    @property
    def is_readonly(self) -> bool:
        if getattr(self, "_is_readonly", False):
            return True
        if hasattr(self, "model") and getattr(self.model, "is_readonly", False):
            return True
        if not self.isEnabled():
            return True
        return False

    def set_readonly(self, readonly: bool):
        self._is_readonly = bool(readonly)
        if hasattr(self, "model") and self.model is not None:
            self.model.set_readonly(bool(readonly))

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # 1. Top Mini Action Toolbar
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setContentsMargins(8, 6, 8, 2)
        top_bar.setSpacing(6)

        self.btn_add_obs = QtWidgets.QPushButton("Add row")
        self.btn_add_obs.clicked.connect(self._add_row)
        top_bar.addWidget(self.btn_add_obs)

        self.btn_add_var = QtWidgets.QPushButton("Add variable")
        self.btn_add_var.clicked.connect(self._add_var)
        top_bar.addWidget(self.btn_add_var)

        self.btn_toggle_fr = QtWidgets.QPushButton("Find and replace")
        self.btn_toggle_fr.clicked.connect(self._toggle_find_replace)
        top_bar.addWidget(self.btn_toggle_fr)

        top_bar.addSpacing(12)

        self.txt_filter = QtWidgets.QLineEdit()
        self.txt_filter.setAccessibleName("Filter rows")
        self.txt_filter.setPlaceholderText("Filter rows")
        self.txt_filter.setClearButtonEnabled(True)
        self.txt_filter.setToolTip("Search across all columns (Ctrl+F in the data editor)")
        focus_filter = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Find, self)
        focus_filter.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        focus_filter.activated.connect(self.txt_filter.setFocus)
        self.txt_filter.textChanged.connect(self._apply_filter)
        top_bar.addWidget(self.txt_filter, stretch=1)

        layout.addLayout(top_bar)

        # Find & Replace Bar
        self.fr_container = QtWidgets.QWidget()
        fr_layout = QtWidgets.QHBoxLayout(self.fr_container)
        fr_layout.setContentsMargins(8, 2, 8, 4)
        fr_layout.setSpacing(6)

        fr_layout.addWidget(QtWidgets.QLabel("Find:"))
        self.txt_find = QtWidgets.QLineEdit()
        self.txt_find.setPlaceholderText("Find text or number...")
        self.txt_find.setFixedHeight(26)
        fr_layout.addWidget(self.txt_find)

        fr_layout.addWidget(QtWidgets.QLabel("Replace:"))
        self.txt_replace = QtWidgets.QLineEdit()
        self.txt_replace.setPlaceholderText("Replace with...")
        self.txt_replace.setFixedHeight(26)
        fr_layout.addWidget(self.txt_replace)

        self.chk_exact = QtWidgets.QCheckBox("Exact")
        self.chk_exact.setChecked(True)
        fr_layout.addWidget(self.chk_exact)

        self.chk_case = QtWidgets.QCheckBox("Case")
        self.chk_case.setChecked(False)
        fr_layout.addWidget(self.chk_case)

        self.btn_do_replace = QtWidgets.QPushButton("Replace All")
        self.btn_do_replace.setFixedHeight(26)
        self.btn_do_replace.clicked.connect(self._on_replace_all)
        fr_layout.addWidget(self.btn_do_replace)

        self.btn_close_fr = QtWidgets.QPushButton("×")
        self.btn_close_fr.setAccessibleName("Close find and replace")
        self.btn_close_fr.setToolTip("Close find and replace")
        self.btn_close_fr.setFixedSize(26, 26)
        self.btn_close_fr.clicked.connect(lambda: self.fr_container.setVisible(False))
        fr_layout.addWidget(self.btn_close_fr)

        self.fr_container.setVisible(False)
        layout.addWidget(self.fr_container)

        # 2. Virtualized Table View
        self.table_view = QtWidgets.QTableView()
        self.model = VirtualPandasModel(self.dataset, self)
        self.table_view.setModel(self.model)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.table_view.horizontalHeader().setDefaultSectionSize(110)
        self.table_view.horizontalHeader().sectionClicked.connect(self.model.sort_by_column)
        self.table_view.verticalHeader().setDefaultSectionSize(26)
        layout.addWidget(self.table_view)

        # 3. Status Bar
        self.lbl_status = QtWidgets.QLabel()
        self.lbl_status.setContentsMargins(8, 3, 8, 3)
        layout.addWidget(self.lbl_status)
        self.model.modelReset.connect(self._update_status)
        self.refresh()

    def copy_selection(self) -> str:
        """Copies currently selected table cells to the clipboard in tab-delimited format."""
        selection = self.table_view.selectionModel().selectedIndexes()
        if not selection:
            return ""
        rows = sorted(list(set(idx.row() for idx in selection)))
        cols = sorted(list(set(idx.column() for idx in selection)))
        lines = []
        for r in rows:
            line_vals = []
            for c in cols:
                idx = self.model.index(r, c)
                val_str = str(self.model.data(idx, Qt.ItemDataRole.EditRole) or "")
                line_vals.append(val_str)
            lines.append("\t".join(line_vals))
        text = "\n".join(lines)
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard:
            clipboard.setText(text)
        return text

    def paste_clipboard(self, start_row: int = 0, start_col: int = 0) -> int:
        """
        Pastes tab-delimited text from clipboard starting at (start_row, start_col).
        Uses a transactional validation policy: validates all cells before modifying.
        Returns count of successfully pasted cells (0 if any cell validation fails).
        """
        if self.is_readonly:
            return 0
        clipboard = QtWidgets.QApplication.clipboard()
        if not clipboard:
            return 0
        text = clipboard.text()
        if not text:
            return 0

        lines = [line.split("\t") for line in text.splitlines()]
        if not lines or not lines[0]:
            return 0

        # Stage 1: Validate all target cells and values transactionally
        staged_edits = []
        df = self.dataset.df
        if df is None or df.empty:
            return 0

        for r_offset, row_vals in enumerate(lines):
            r = start_row + r_offset
            if r >= self.model.rowCount():
                break
            actual_row = self.model._map_row(r)
            if actual_row >= len(df):
                break

            for c_offset, val in enumerate(row_vals):
                c = start_col + c_offset
                if c >= self.model.columnCount():
                    break
                col_name = df.columns[c]
                dtype = df[col_name].dtype
                str_val = str(val).strip()

                # Validate value against dtype
                try:
                    if str_val == "." or str_val == "":
                        if pd.api.types.is_integer_dtype(dtype):
                            new_val = pd.NA
                        elif pd.api.types.is_bool_dtype(dtype):
                            if isinstance(dtype, pd.BooleanDtype):
                                new_val = pd.NA
                            else:
                                return 0  # Cannot put NaN in standard bool
                        elif pd.api.types.is_float_dtype(dtype):
                            new_val = np.nan
                        else:
                            new_val = None if pd.api.types.is_object_dtype(dtype) else np.nan

                    elif pd.api.types.is_integer_dtype(dtype):
                        int_val = int(str_val)
                        np_dtype = dtype.numpy_dtype if hasattr(dtype, "numpy_dtype") else dtype
                        info = np.iinfo(np_dtype)
                        if int_val < info.min or int_val > info.max:
                            return 0
                        new_val = int_val

                    elif pd.api.types.is_float_dtype(dtype):
                        ok, parsed_flt, _ = validate_float_input(str_val, dtype)
                        if not ok:
                            return 0
                        new_val = parsed_flt

                    elif pd.api.types.is_bool_dtype(dtype):
                        lower_val = str_val.lower()
                        if lower_val in ('true', '1', 'yes', 't', 'y'):
                            new_val = True
                        elif lower_val in ('false', '0', 'no', 'f', 'n'):
                            new_val = False
                        else:
                            return 0

                    elif isinstance(dtype, pd.CategoricalDtype):
                        lbl_name = self.dataset.val_label_names.get(col_name) if hasattr(self.dataset, 'val_label_names') else None
                        val_labels = (self.dataset.val_labels.get(lbl_name) if lbl_name else None) or (self.dataset.val_labels.get(col_name) if hasattr(self.dataset, 'val_labels') else None)
                        ok, cat_val, _ = parse_categorical_value(str_val, df[col_name], val_labels)
                        if not ok:
                            return 0
                        new_val = np.nan if pd.isna(cat_val) else cat_val
                    else:
                        new_val = str_val

                    staged_edits.append((actual_row, c, col_name, new_val))
                except Exception:
                    # Any validation failure aborts the entire paste transactionally
                    return 0

        if not staged_edits:
            return 0

        # Stage 2: Apply all edits atomically with rollback
        df_backup = self.dataset.df.copy()
        try:
            for actual_row, c, col_name, new_val in staged_edits:
                curr_dt = self.dataset.df[col_name].dtype
                if pd.isna(new_val) and pd.api.types.is_integer_dtype(curr_dt) and not isinstance(curr_dt, pd.core.dtypes.dtypes.ExtensionDtype):
                    self.dataset.df[col_name] = self.dataset.df[col_name].astype(to_nullable_integer_dtype(curr_dt))
                if isinstance(self.dataset.df[col_name].dtype, pd.CategoricalDtype) and not pd.isna(new_val) and new_val not in self.dataset.df[col_name].cat.categories:
                    self.dataset.df[col_name] = self.dataset.df[col_name].cat.add_categories([new_val])
                self.dataset.df.iloc[actual_row, c] = new_val
        except Exception:
            self.dataset.df = df_backup
            return 0

        # Stage 3: Refresh model once
        self.model.update_data()
        return len(staged_edits)

    def refresh(self):
        self.model.update_data()

    def _update_status(self):
        df = self.dataset.df
        n_obs = len(df) if df is not None else 0
        n_vars = len(df.columns) if df is not None else 0
        if self.model._filter_text:
            rows = f"{self.model.rowCount():,} of {n_obs:,} rows matched"
        else:
            rows = f"{n_obs:,} observations"
        self.lbl_status.setText(f"{rows}   ·   {n_vars:,} variables")

    def _add_row(self):
        if self.is_readonly:
            return
        df = self.dataset.df
        if df is None:
            return
        self.dataset.df = insert_blank_observation(df)
        self.refresh()

    def _add_var(self):
        if self.is_readonly:
            return
        text, ok = QtWidgets.QInputDialog.getText(self, "Add Variable", "Enter new variable name:")
        if ok and text.strip():
            varname = text.strip()
            if self.dataset.df is not None and varname in self.dataset.df.columns:
                QtWidgets.QMessageBox.warning(self, "Warning", f"Variable '{varname}' already exists.")
                return
            if self.dataset.df is None:
                self.dataset.df = pd.DataFrame({varname: []})
            else:
                self.dataset.df[varname] = np.nan
            self.refresh()

    def _apply_filter(self, text):
        self.model.set_filter(text)

    def _toggle_find_replace(self):
        vis = not self.fr_container.isVisible()
        self.fr_container.setVisible(vis)
        if vis:
            self.txt_find.setFocus()

    def _on_replace_all(self):
        if self.is_readonly:
            return
        f = self.txt_find.text()
        r = self.txt_replace.text()
        if not f:
            return
        match_case = self.chk_case.isChecked()
        exact_match = self.chk_exact.isChecked()
        n = perform_find_replace(self.dataset, f, r, match_case=match_case, exact_match=exact_match)
        self.refresh()
        self.lbl_status.setText(f"Find & Replace: replaced {n:,} occurrence(s).")


def perform_find_replace(
    dataset,
    find_text: str,
    replace_text: str,
    match_case: bool = False,
    exact_match: bool = False,
    approx_match: bool = False,
    tolerance: float = 1e-8,
    coerce_mixed: bool = False,
) -> int:
    """
    Performs transactional Find & Replace across all cells of a dataset.

    Guarantees:
    - Exact numeric matching by default (prevents 100000.5 being changed when replacing 100000).
    - Preserves large integers (> 2^53) without float64 precision loss.
    - Preserves missing values during string replacement (never converts NaN/None/pd.NA to string literals).
    - Preserves dtypes (integer, float, categorical, nullable). Converts numpy integer to nullable Int64 if replaced with missing.
    - Duplicate index safety: Uses positional row indexing (.iloc / positional masks) never label-based .loc.
    - Genuinely transactional execution: validates all candidate changes across ALL columns before applying;
      if any matched cell in any column cannot represent the replacement, rejects the entire operation leaving dataset untouched.
    - Returns exact count of cells modified (returns 0 for no-op replacements where value is unchanged).
    """
    df = dataset.df
    if df is None or df.empty or find_text is None or find_text == "":
        return 0

    staged_series = {}
    total_changed_cells = 0

    try:
        # Pre-check: if find_text == replace_text and match_case: pure no-op
        if match_case and find_text == replace_text:
            return 0

        for col in df.columns:
            s = df[col]
            s_dtype = s.dtype
            n_rows = len(s)

            # =================================================================
            # Case 1: Missing value search: find_text == "."
            # =================================================================
            if find_text == ".":
                na_mask = s.isna().to_numpy(dtype=bool)
                match_count = int(na_mask.sum())
                if match_count == 0:
                    continue

                if replace_text == "." or replace_text == "":
                    continue

                na_positions = np.where(na_mask)[0]

                if pd.api.types.is_integer_dtype(s_dtype):
                    int_r = _parse_exact_integer(replace_text, s_dtype)
                    if int_r is None:
                        return 0
                    new_s = s.copy()
                    new_s.iloc[na_positions] = int_r

                elif pd.api.types.is_float_dtype(s_dtype):
                    ok_f, flt_r, _ = validate_float_input(replace_text, s_dtype)
                    if not ok_f:
                        return 0
                    new_s = s.copy()
                    new_s.iloc[na_positions] = flt_r

                elif pd.api.types.is_bool_dtype(s_dtype):
                    lower_r = replace_text.strip().lower()
                    if lower_r in ('true', '1', 'yes', 't', 'y'):
                        bool_r = True
                    elif lower_r in ('false', '0', 'no', 'f', 'n'):
                        bool_r = False
                    else:
                        return 0
                    new_s = s.copy()
                    new_s.iloc[na_positions] = bool_r

                elif isinstance(s_dtype, pd.CategoricalDtype):
                    ok_c, cat_r, _ = parse_categorical_value(replace_text, s)
                    if not ok_c:
                        return 0
                    new_s = s.copy()
                    if not pd.isna(cat_r) and cat_r not in new_s.cat.categories:
                        new_s = new_s.cat.add_categories([cat_r])
                    new_s.iloc[na_positions] = cat_r

                else:
                    arr = s.tolist()
                    for p in na_positions:
                        arr[p] = replace_text
                    new_s = pd.Series(arr, index=s.index, dtype=s.dtype)

                staged_series[col] = (new_s, match_count)
                total_changed_cells += match_count
                continue

            # =================================================================
            # Case 2: Integer columns (standard int8..int64, uint8..uint64, Int8..Int64, UInt8..UInt64)
            # =================================================================
            if pd.api.types.is_integer_dtype(s_dtype):
                find_int = _parse_exact_integer(find_text, s_dtype)
                if find_int is None:
                    continue

                na_mask = s.isna().to_numpy(dtype=bool)
                valid_mask = ~na_mask
                if not valid_mask.any():
                    continue

                comp_series = (s == find_int)
                if hasattr(comp_series, "fillna"):
                    comp_series = comp_series.fillna(False)
                match_mask = valid_mask & comp_series.to_numpy(dtype=bool)
                match_count = int(match_mask.sum())
                if match_count == 0:
                    continue

                match_positions = np.where(match_mask)[0]

                new_s = s.copy()
                if replace_text == "." or replace_text == "":
                    # Convert to nullable ExtensionDtype to prevent precision loss!
                    if not isinstance(s_dtype, pd.core.dtypes.dtypes.ExtensionDtype):
                        target_dtype = to_nullable_integer_dtype(s_dtype)
                        new_s = new_s.astype(target_dtype)
                    new_s.iloc[match_positions] = pd.NA
                    staged_series[col] = (new_s, match_count)
                    total_changed_cells += match_count
                    continue

                int_r = _parse_exact_integer(replace_text, s_dtype)
                if int_r is None:
                    return 0
                if int_r == find_int:
                    continue

                new_s.iloc[match_positions] = int_r
                staged_series[col] = (new_s, match_count)
                total_changed_cells += match_count
                continue

            # =================================================================
            # Case 3: Float columns
            # =================================================================
            if pd.api.types.is_float_dtype(s_dtype):
                # Compare exact queries in the stored dtype. Widening float32 to
                # float64 before comparison makes ordinary decimals miss their cells.
                float_dtype = np.dtype(getattr(s_dtype, "numpy_dtype", s_dtype))
                search_dtype = np.promote_types(float_dtype, np.float64) if approx_match else float_dtype
                ok, num_find, _ = validate_float_input(find_text, search_dtype)
                if not ok or np.isnan(num_find):
                    continue

                s_vals = s.to_numpy(dtype=search_dtype, na_value=np.nan)
                if approx_match:
                    match_mask = np.isclose(s_vals, num_find, rtol=tolerance, atol=tolerance, equal_nan=False)
                else:
                    match_mask = s_vals == num_find
                if not match_mask.any():
                    continue

                ok, num_replace, _ = validate_float_input(replace_text, s_dtype)
                if not ok:
                    return 0
                match_mask &= s_vals != num_replace
                match_count = int(match_mask.sum())
                if not match_count:
                    continue

                new_s = s.copy()
                new_s.iloc[np.flatnonzero(match_mask)] = num_replace
                staged_series[col] = (new_s, match_count)
                total_changed_cells += match_count
                continue

            # =================================================================
            # Case 4: Boolean columns
            # =================================================================
            if pd.api.types.is_bool_dtype(s_dtype):
                lower_f = find_text.strip().lower()
                bool_find = None
                if lower_f in ('true', '1', 'yes', 't', 'y'):
                    bool_find = True
                elif lower_f in ('false', '0', 'no', 'f', 'n'):
                    bool_find = False

                if bool_find is None:
                    continue

                na_mask = s.isna().to_numpy(dtype=bool)
                valid_mask = ~na_mask
                if not valid_mask.any():
                    continue

                comp_series = (s == bool_find)
                if hasattr(comp_series, "fillna"):
                    comp_series = comp_series.fillna(False)
                match_mask = valid_mask & comp_series.to_numpy(dtype=bool)
                match_count = int(match_mask.sum())
                if match_count == 0:
                    continue

                match_positions = np.where(match_mask)[0]
                new_s = s.copy()

                lower_r = replace_text.strip().lower()
                if lower_r in ('.', '', 'na', '<na>'):
                    if isinstance(s_dtype, pd.BooleanDtype):
                        new_s.iloc[match_positions] = pd.NA
                    else:
                        return 0
                elif lower_r in ('true', '1', 'yes', 't', 'y'):
                    bool_r = True
                    if bool_r == bool_find:
                        continue
                    new_s.iloc[match_positions] = bool_r
                elif lower_r in ('false', '0', 'no', 'f', 'n'):
                    bool_r = False
                    if bool_r == bool_find:
                        continue
                    new_s.iloc[match_positions] = bool_r
                else:
                    return 0

                staged_series[col] = (new_s, match_count)
                total_changed_cells += match_count
                continue

            # =================================================================
            # Case 5: Categorical columns
            # =================================================================
            if isinstance(s_dtype, pd.CategoricalDtype):
                categories = list(s.cat.categories)
                na_mask = s.isna().to_numpy(dtype=bool)
                valid_mask = ~na_mask
                if not valid_mask.any():
                    continue

                cat_replacements = {}
                for cat in categories:
                    cat_str = str(cat)
                    if exact_match:
                        matched = (cat_str == find_text) if match_case else (cat_str.lower() == find_text.lower())
                        if matched:
                            if replace_text == "." or replace_text == "":
                                cat_replacements[cat] = np.nan
                            else:
                                ok_c, parsed_c, _ = parse_categorical_value(replace_text, s)
                                if not ok_c:
                                    return 0
                                cat_replacements[cat] = parsed_c
                    else:
                        flags = 0 if match_case else re.IGNORECASE
                        pattern = re.escape(find_text)
                        if re.search(pattern, cat_str, flags=flags):
                            new_cat_str = re.sub(pattern, lambda m: replace_text, cat_str, flags=flags)
                            ok_c, parsed_c, _ = parse_categorical_value(new_cat_str, s)
                            if not ok_c:
                                return 0
                            cat_replacements[cat] = parsed_c

                if not cat_replacements:
                    continue

                active_cat_replacements = {k: v for k, v in cat_replacements.items() if (pd.isna(v) or k != v)}
                if not active_cat_replacements:
                    continue

                new_s = s.copy()
                changed_in_col = 0
                for old_cat, new_cat in active_cat_replacements.items():
                    if not pd.isna(new_cat) and new_cat not in new_s.cat.categories:
                        new_s = new_s.cat.add_categories([new_cat])

                for pos in range(n_rows):
                    if not na_mask[pos]:
                        val = s.iloc[pos]
                        if val in active_cat_replacements:
                            new_val = active_cat_replacements[val]
                            new_s.iloc[pos] = new_val
                            changed_in_col += 1

                if changed_in_col > 0:
                    expected_na = int(na_mask.sum()) + sum(
                        1 for pos in range(n_rows)
                        if not na_mask[pos] and (s.iloc[pos] in active_cat_replacements and pd.isna(active_cat_replacements[s.iloc[pos]]))
                    )
                    assert new_s.isna().sum() == expected_na, f"Missing values corrupted in column {col}"
                    staged_series[col] = (new_s, changed_in_col)
                    total_changed_cells += changed_in_col
                continue

            # =================================================================
            # Case 6: String and Object columns
            # =================================================================
            if pd.api.types.is_string_dtype(s_dtype) or pd.api.types.is_object_dtype(s_dtype):
                na_mask = s.isna().to_numpy(dtype=bool)
                valid_mask = ~na_mask
                if not valid_mask.any():
                    continue

                arr = s.tolist()
                changed_in_col = 0
                valid_positions = np.where(valid_mask)[0]
                is_obj = pd.api.types.is_object_dtype(s_dtype)

                if exact_match:
                    for p in valid_positions:
                        raw_val = arr[p]
                        if is_obj and not coerce_mixed and not isinstance(raw_val, str):
                            # Non-string object element in mixed object column: preserve original type without coercion
                            continue
                        val = str(raw_val)
                        matched = (val == find_text) if match_case else (val.lower() == find_text.lower())
                        if matched and val != replace_text:
                            arr[p] = replace_text
                            changed_in_col += 1
                else:
                    flags = 0 if match_case else re.IGNORECASE
                    pattern = re.escape(find_text)
                    for p in valid_positions:
                        raw_val = arr[p]
                        if is_obj and not coerce_mixed and not isinstance(raw_val, str):
                            # Non-string object element in mixed object column: preserve original type without coercion
                            continue
                        val = str(raw_val)
                        new_val, n_subs = re.subn(pattern, lambda m: replace_text, val, flags=flags)
                        if n_subs > 0 and new_val != val:
                            arr[p] = new_val
                            changed_in_col += 1

                if changed_in_col > 0:
                    new_s = pd.Series(arr, index=s.index, dtype=s.dtype)
                    assert new_s.isna().sum() == na_mask.sum(), f"Missing values corrupted in column {col}"
                    staged_series[col] = (new_s, changed_in_col)
                    total_changed_cells += changed_in_col
                continue

        # Transactional commit: apply staged series atomically with rollback protection
        if total_changed_cells > 0:
            backup_cols = {col: dataset.df[col].copy() for col in staged_series.keys()}
            committed_cols = []
            try:
                for col, (new_col_series, _) in staged_series.items():
                    dataset.df[col] = new_col_series
                    committed_cols.append(col)
            except Exception:
                # Rollback every modified column to its exact pre-commit copy
                for col in committed_cols:
                    dataset.df[col] = backup_cols[col]
                return 0

        return total_changed_cells

    except Exception:
        return 0
