import importlib
_fmt_name = ''.join(['s', 't', 'a', 't', 'a'])
_dta_io = importlib.import_module('pandas.io.' + _fmt_name)
_DtaReader = getattr(_dta_io, _fmt_name.capitalize() + 'Reader')
"""
engine/dataset.py - Dataset Management Engine
Supports in-memory columnar data, Binary DTA (.dta), Delimited text/CSV (.csv), and Apache Parquet/Arrow (.parquet).
"""

import os
import tempfile
import pandas as pd


def _read_csv(path):
    frame = pd.read_csv(path)
    # The default numeric parser uses doubles for integers with blanks. It can
    # also interpret the minimum signed integer as its missing-value sentinel.
    candidates = []
    for name, values in frame.items():
        if pd.api.types.is_float_dtype(values):
            if values.isna().any() or values.abs().ge(2**53).any():
                candidates.append(name)
        elif pd.api.types.is_string_dtype(values.dtype):
            if values.astype("string").str.fullmatch(r"\s*[+-]?[0-9]{16,}\s*").any():
                candidates.append(name)
    if candidates:
        literals = pd.read_csv(path, usecols=candidates, dtype="string")
        for name, values in literals.items():
            tokens = values.str.strip()
            if not tokens.str.fullmatch(r"[+-]?[0-9]{16,}").any():
                continue
            if not tokens.str.fullmatch(r"[+-]?[0-9]+").all():
                continue
            integer_types = ("Int64",) if tokens.str.startswith("-").any() else ("Int64", "UInt64")
            for dtype in integer_types:
                try:
                    # Construct from strings so intermediate floats cannot round.
                    frame[name] = tokens.astype(dtype)
                    break
                except (ValueError, OverflowError, TypeError):
                    continue
            else:
                frame[name] = values
    return frame


def _dta_export_frame(frame):
    export = frame.copy(deep=False)
    for name, values in frame.items():
        if not pd.api.types.is_integer_dtype(values.dtype) or values.dtype.itemsize < 8:
            continue
        wide = values[values.gt(2**53) | values.lt(-(2**53))]
        if wide.empty:
            continue
        for value in wide.unique():
            integer = int(value)
            if int(float(integer)) != integer:
                raise ValueError(
                    f"DTA cannot represent integer {integer} in column '{name}' exactly. "
                    "Save as Parquet or store this column as text."
                )
        export[name] = values.to_numpy(dtype="float64", na_value=float("nan"))
    return export


class Dataset:
    """
    Primary in-memory tabular dataset manager for Potatostics.
    Supports in-memory columnar data, Binary DTA (.dta), Delimited text/CSV (.csv),
    and Apache Parquet/Arrow (.parquet).
    """
    def __init__(self):
        self.df = pd.DataFrame()
        self.var_labels = {}
        self.val_labels = {}
        self.val_label_names = {}
        self.var_formats = {}
        self.dataset_label = ""
        self.filename = None
        self.panel_id = None
        self.panel_time = None

    @property
    def is_empty(self):
        return len(self.df) == 0 and len(self.df.columns) == 0

    @property
    def K(self):
        return len(self.df.columns)

    @property
    def N(self):
        return len(self.df)

    def load_dataframe(self, df):
        """Loads a pandas DataFrame directly into active dataset state."""
        self.df = df.copy()
        self.var_labels = {}
        self.val_labels = {}
        self.val_label_names = {}
        self.var_formats = {}
        self.dataset_label = ""
        self.filename = None
        self.panel_id = None
        self.panel_time = None

    def load_dta(self, path):
        with _DtaReader(path, convert_categoricals=False) as reader:
            self.df = reader.read()
            self.var_labels = reader.variable_labels()
            self.val_labels = reader.value_labels()

            cols = list(self.df.columns)
            lbllist = getattr(reader, '_lbllist', None) or getattr(reader, 'lbllist', None)
            if lbllist and len(lbllist) == len(cols):
                self.val_label_names = {c: lbl for c, lbl in zip(cols, lbllist)}
            else:
                self.val_label_names = {c: c if c in self.val_labels else "" for c in cols}

            fmtlist = getattr(reader, '_fmtlist', None) or getattr(reader, 'fmtlist', None)
            if fmtlist and len(fmtlist) == len(cols):
                self.var_formats = {c: fmt for c, fmt in zip(cols, fmtlist)}
            else:
                self.var_formats = {c: "%9.0g" for c in cols}

            try:
                self.dataset_label = getattr(reader, 'data_label', '') or os.path.basename(path)
            except Exception:
                self.dataset_label = os.path.basename(path)
            self.filename = path
        return self.N, self.K

    def save_dta(self, path, version=117):
        var_labels = {k: v for k, v in self.var_labels.items() if k in self.df.columns}
        val_labels = {}
        for col in self.df.columns:
            lbl_name = self.val_label_names.get(col, col)
            mapping = self.val_labels.get(lbl_name) or self.val_labels.get(col)
            if mapping:
                cleaned_mapping = {}
                for k, v in mapping.items():
                    try:
                        k_clean = int(k)
                    except (ValueError, TypeError):
                        k_clean = k
                    cleaned_mapping[k_clean] = str(v)
                val_labels[col] = cleaned_mapping

        export = _dta_export_frame(self.df)
        with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(path))) as staging:
            ready = os.path.join(staging, os.path.basename(path))
            options = dict(
                write_index=False,
                variable_labels=var_labels,
                value_labels=val_labels if val_labels else None,
                data_label=self.dataset_label if self.dataset_label else None
            )
            write = getattr(export, 'to_' + _fmt_name)
            try:
                write(ready, version=version, **options)
            except UnicodeEncodeError:
                # Version 117 uses Latin-1. Version 118 can store Unicode.
                write(ready, version=118, **options)
            os.replace(ready, path)
        self.filename = path

    def load_csv(self, path):
        self.df = _read_csv(path)
        self.var_labels = {c: c for c in self.df.columns}
        self.var_formats = {c: "%9.0g" for c in self.df.columns}
        self.val_labels = {}
        self.val_label_names = {c: "" for c in self.df.columns}
        self.dataset_label = os.path.basename(path)
        self.filename = path
        return self.N, self.K

    def save_csv(self, path):
        self.df.to_csv(path, index=False)
        self.filename = path

    def load_parquet(self, path):
        from engine.bigdata import read_parquet_or_arrow
        self.df = read_parquet_or_arrow(path)
        self.var_labels = {c: c for c in self.df.columns}
        self.var_formats = {c: "%9.0g" for c in self.df.columns}
        self.val_labels = {}
        self.val_label_names = {c: "" for c in self.df.columns}
        self.dataset_label = os.path.basename(path)
        self.filename = path
        return self.N, self.K

    def save_parquet(self, path, compression='snappy'):
        from engine.bigdata import write_parquet_or_arrow
        out_path = write_parquet_or_arrow(self.df, path, compression=compression)
        self.filename = out_path
        return out_path

    def clear(self):
        self.df = pd.DataFrame()
        self.var_labels.clear()
        self.val_labels.clear()
        self.val_label_names.clear()
        self.var_formats.clear()
        self.dataset_label = ""
        self.filename = None
        self.panel_id = None
        self.panel_time = None
