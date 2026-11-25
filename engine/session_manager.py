"""
engine/session_manager.py - Single-File Analytical Session Archive (.potproj)
Provides complete, self-contained, point-and-click session packaging for Potatostics.
Bundles the active in-memory dataset, variable metadata and value labels,
command review history, stored econometric estimates, macros, return registers,
and console transcripts into a portable, shareable ZIP-compressed archive (.potproj).
Hardened against zip bombs, path traversal, unexpected members, and corrupted archives.
"""

import os
import io
import json
import tempfile
import zipfile
import datetime
from contextlib import contextmanager
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

try:
    import pyarrow  # noqa: F401
    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False

# Security whitelist & thresholds
ALLOWED_MEMBERS = {
    "manifest.json",
    "data.parquet",
    "data.csv",
    "dtypes.json",
    "data.json",
    "labels.json",
    "estimates.json",
    "macros.json",
    "registers.json",
    "console.txt",
    "history.json",
}
MAX_MEMBER_SIZE = 500 * 1024 * 1024       # 500 MB per-member limit
MAX_TOTAL_UNCOMPRESSED = 1024 * 1024 * 1024  # 1 GB total uncompressed


class NumpyJSONEncoder(json.JSONEncoder):
    """Encodes NumPy scalar types and arrays into standard JSON primitives."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Index, pd.Series)):
            return list(obj)
        elif hasattr(obj, "tolist"):
            return obj.tolist()
        elif isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def sanitize_for_json(obj: Any) -> Any:
    """Recursively converts numpy keys and objects into standard JSON primitives."""
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, np.integer):
                clean_k = int(k)
            elif isinstance(k, (np.floating, float)):
                clean_k = float(k)
            elif isinstance(k, (bool, type(None))):
                clean_k = k
            else:
                clean_k = str(k)
            new_dict[clean_k] = sanitize_for_json(v)
        return new_dict
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, (pd.Index, pd.Series)):
        return sanitize_for_json(list(obj))
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    return obj


def _reconstruct_model_arrays(model_dict: Any) -> Any:
    """Restores numpy ndarrays and index types inside an estimation model dictionary."""
    if not isinstance(model_dict, dict):
        return model_dict
    m = dict(model_dict)
    array_keys = ["beta", "vcov", "resid", "y_pred", "p_pred", "se", "cov", "grad"]
    for k in array_keys:
        if k in m and m[k] is not None:
            if not isinstance(m[k], np.ndarray):
                try:
                    m[k] = np.asarray(m[k], dtype=float)
                except Exception:
                    m[k] = np.asarray(m[k])
    if "data_index" in m and m["data_index"] is not None:
        if not isinstance(m["data_index"], (pd.Index, pd.Series)):
            m["data_index"] = pd.Index(m["data_index"])
    return m


def _safe_json_load(zf: zipfile.ZipFile, member_name: str) -> Any:
    """Safely decodes and parses JSON from a ZIP archive member."""
    raw = zf.read(member_name).decode("utf-8")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Invalid session archive: invalid JSON in member '{member_name}': {e}")


@contextmanager
def _atomic_session_archive(filepath):
    # Close the temporary file before replacing the destination on Windows.
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(filepath))) as folder:
        staged = os.path.join(folder, "session.potproj")
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            yield archive
        os.replace(staged, filepath)


def save_session_bundle(
    filepath: str,
    interp: Any,
    console_text: str = "",
    history_records: Optional[List[Dict[str, Any]]] = None,
    console_ranges: Optional[List] = None,
) -> str:
    """
    Saves the entire active Potatostics analytical state into a single .potproj archive.
    Uses Parquet (pyarrow) for dataset fidelity, with CSV fallback if pyarrow is absent.
    
    Parameters:
        filepath: Target filesystem path (e.g. "my_session.potproj")
        interp: Potatostics Interpreter instance containing dataset and estimation registers
        console_text: Raw or formatted console output string
        history_records: List of executed command records from the Command Review pane
        
    Returns:
        Canonical absolute path to the saved session archive.
    """
    clean_path = str(filepath).strip().strip('"\'')
    if not clean_path.lower().endswith(".potproj"):
        clean_path += ".potproj"
        
    os.makedirs(os.path.dirname(os.path.abspath(clean_path)), exist_ok=True)
    
    # 1. Build Manifest Metadata
    n_rows, n_cols = interp.dataset.df.shape
    var_names = list(interp.dataset.df.columns)
    
    manifest = {
        "app": "Potatostics",
        "format": "potatostics_session_bundle",
        "version": 2,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_rows": n_rows,
        "dataset_cols": n_cols,
        "variables": var_names,
        "panelvar": interp.panelvar,
        "timevar": interp.timevar,
        "storage_format": "parquet" if HAVE_PYARROW else "csv",
        "models_stored": len(interp.estimates.store) if hasattr(interp, "estimates") else 0
    }
    
    if console_ranges is not None:
        manifest["console_ranges_v1"] = console_ranges

    # 2. Package into single ZIP archive
    with _atomic_session_archive(clean_path) as zf:
        # Write manifest
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        
        # Write dataset
        if n_rows or n_cols:
            dtypes_dict = {col: str(dtype) for col, dtype in interp.dataset.df.dtypes.items()}
            zf.writestr("dtypes.json", json.dumps(dtypes_dict, indent=2))
            if HAVE_PYARROW:
                buf = io.BytesIO()
                # Rows without variables need a physical index for Parquet to count them.
                interp.dataset.df.to_parquet(buf, engine="pyarrow", index=True if not n_cols else None)
                zf.writestr("data.parquet", buf.getvalue())
            else:
                csv_buf = io.StringIO()
                interp.dataset.df.to_csv(csv_buf, index=False, float_format="%.17g")
                zf.writestr("data.csv", csv_buf.getvalue())
            
        # Write variable and value labels
        labels = {
            "var_labels": getattr(interp.dataset, "var_labels", {}),
            "val_labels": getattr(interp.dataset, "val_labels", {}),
            "val_label_names": getattr(interp.dataset, "val_label_names", {}),
            "dataset_label": getattr(interp.dataset, "dataset_label", None),
        }
        zf.writestr("labels.json", json.dumps(sanitize_for_json(labels), cls=NumpyJSONEncoder, indent=2))
        
        # Write stored estimation models and last_model
        models_store = interp.estimates.store if hasattr(interp, "estimates") else {}
        last_model = interp.estimates.last_model if hasattr(interp, "estimates") else None
        estimates_payload = {
            "store": models_store,
            "last_model": last_model,
        }
        zf.writestr("estimates.json", json.dumps(sanitize_for_json(estimates_payload), cls=NumpyJSONEncoder, indent=2))
        
        # Write macros
        macros = {
            "global_macros": getattr(interp.macro_mgr, "globals", {}) if hasattr(interp, "macro_mgr") else {},
            "local_macros": getattr(interp.macro_mgr, "locals", {}) if hasattr(interp, "macro_mgr") else {},
            "r_scalars": getattr(interp.registers, "r_scalars", {}) if hasattr(interp, "registers") else {},
            "e_scalars": getattr(interp.registers, "e_scalars", {}) if hasattr(interp, "registers") else {},
            "r_macros": getattr(interp.registers, "r_macros", {}) if hasattr(interp, "registers") else {},
            "e_macros": getattr(interp.registers, "e_macros", {}) if hasattr(interp, "registers") else {},
        }
        zf.writestr("macros.json", json.dumps(sanitize_for_json(macros), cls=NumpyJSONEncoder, indent=2))
        
        # Write full return registers
        registers_payload = {
            "r_scalars": getattr(interp.registers, "r_scalars", {}) if hasattr(interp, "registers") else {},
            "r_macros": getattr(interp.registers, "r_macros", {}) if hasattr(interp, "registers") else {},
            "r_matrices": getattr(interp.registers, "r_matrices", {}) if hasattr(interp, "registers") else {},
            "e_scalars": getattr(interp.registers, "e_scalars", {}) if hasattr(interp, "registers") else {},
            "e_macros": getattr(interp.registers, "e_macros", {}) if hasattr(interp, "registers") else {},
            "e_matrices": getattr(interp.registers, "e_matrices", {}) if hasattr(interp, "registers") else {},
            "b": getattr(interp.registers, "b", {}) if hasattr(interp, "registers") else {},
            "se": getattr(interp.registers, "se", {}) if hasattr(interp, "registers") else {},
            "rc": getattr(interp.registers, "rc", 0) if hasattr(interp, "registers") else 0,
        }
        zf.writestr("registers.json", json.dumps(sanitize_for_json(registers_payload), cls=NumpyJSONEncoder, indent=2))
        
        # Write console transcript
        zf.writestr("console.txt", console_text or "")
        
        # Write command history records
        zf.writestr("history.json", json.dumps(sanitize_for_json(history_records or []), cls=NumpyJSONEncoder, indent=2))

    return os.path.abspath(clean_path)


def load_session_bundle(filepath: str, interp: Any) -> Dict[str, Any]:
    """
    Restores complete analytical state from a .potproj session archive into the interpreter.
    Enforces archive security checks (member whitelist, path traversal, size limits).
    
    Parameters:
        filepath: Path to the .potproj archive
        interp: Target Potatostics Interpreter instance
        
    Returns:
        Dictionary containing restored 'console_text', 'history_records', and 'manifest'.
    """
    clean_path = str(filepath).strip().strip('"\'')
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Session archive '{clean_path}' not found.")
        
    if not zipfile.is_zipfile(clean_path):
        raise ValueError(f"File '{clean_path}' is not a valid Potatostics session archive.")

    restored = {
        "console_text": "",
        "history_records": [],
        "manifest": {}
    }

    try:
        zf = zipfile.ZipFile(clean_path, mode="r")
    except zipfile.BadZipFile as e:
        raise ValueError(f"File '{clean_path}' is not a valid Potatostics session archive: {e}")

    with zf:
        # --- Security Hardening Audit ---
        infolist = zf.infolist()
        total_uncompressed = 0
        for info in infolist:
            fname = info.filename
            norm_name = fname.replace("\\", "/")
            parts = norm_name.split("/")
            
            # 1. Path traversal & Zip Slip protection
            if os.path.isabs(fname) or ".." in parts or norm_name.startswith("/") or ":" in fname:
                raise ValueError(f"Invalid session archive: path traversal detected in member '{fname}'.")
                
            # 2. Strict member name whitelist
            if norm_name not in ALLOWED_MEMBERS:
                raise ValueError(f"Invalid session archive: unexpected member '{fname}'.")
                
            # 3. Member size limit (zip bomb protection)
            if info.file_size > MAX_MEMBER_SIZE:
                raise ValueError(f"Invalid session archive: member '{fname}' exceeds size limit ({info.file_size} bytes).")
                
            total_uncompressed += info.file_size
            
        # 4. Total uncompressed archive limit
        if total_uncompressed > MAX_TOTAL_UNCOMPRESSED:
            raise ValueError(f"Invalid session archive: total uncompressed size exceeds limit ({total_uncompressed} bytes).")

        namelist = zf.namelist()
        
        # 1. Manifest
        if "manifest.json" in namelist:
            restored["manifest"] = _safe_json_load(zf, "manifest.json")
            
        # 2. Dataset Restoration
        if "data.parquet" in namelist:
            raw_bytes = zf.read("data.parquet")
            df = pd.read_parquet(io.BytesIO(raw_bytes), engine="pyarrow")
            if "dtypes.json" in namelist:
                try:
                    dtypes_map = _safe_json_load(zf, "dtypes.json")
                    for col, dt_str in dtypes_map.items():
                        if col in df.columns:
                            if "category" in dt_str and not isinstance(df[col].dtype, pd.CategoricalDtype):
                                df[col] = df[col].astype("category")
                except Exception:
                    pass
            interp.dataset.load_dataframe(df)
        elif "data.csv" in namelist:
            raw_csv = zf.read("data.csv").decode("utf-8")
            dtypes_map = {}
            if "dtypes.json" in namelist:
                try:
                    dtypes_map = _safe_json_load(zf, "dtypes.json")
                except Exception:
                    dtypes_map = {}
            if restored["manifest"].get("dataset_cols") == 0:
                df = pd.DataFrame(index=pd.RangeIndex(restored["manifest"]["dataset_rows"]))
            else:
                # Read integer columns as strings first so >2^53 integers are never rounded via float64.
                csv_dtypes = {
                    col: "string"
                    for col, dt_str in dtypes_map.items()
                    if "int" in str(dt_str).lower()
                }
                df = pd.read_csv(io.StringIO(raw_csv), dtype=csv_dtypes or None)
            if dtypes_map:
                for col, dt_str in dtypes_map.items():
                    if col not in df.columns:
                        continue
                    try:
                        if "datetime64" in dt_str:
                            df[col] = pd.to_datetime(df[col])
                        elif dt_str in ("Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64",
                                        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
                            df[col] = df[col].astype(dt_str)
                        elif "bool" in dt_str:
                            df[col] = df[col].astype(dt_str if dt_str == "boolean" else "bool")
                        elif "category" in dt_str:
                            df[col] = df[col].astype("category")
                        elif "float" in dt_str:
                            df[col] = df[col].astype(dt_str if dt_str in ("float32", "float64", "Float32", "Float64") else "float64")
                    except Exception:
                        pass
            interp.dataset.load_dataframe(df)
        elif "data.json" in namelist:
            # Backward-compatibility fallback for legacy v1 archives
            data_str = zf.read("data.json").decode("utf-8")
            df = pd.read_json(io.StringIO(data_str), orient="split")
            interp.dataset.load_dataframe(df)
        elif restored["manifest"].get("dataset_rows") == 0 and restored["manifest"].get("dataset_cols") == 0:
            interp.dataset.clear()
        else:
            raise ValueError("Session archive is missing its dataset.")

        # Older bundles did not preserve xtset/tsset and must open undeclared.
        interp.panelvar = restored["manifest"].get("panelvar")
        interp.timevar = restored["manifest"].get("timevar")
            
        # 3. Variable and Value Labels
        if "labels.json" in namelist:
            labels_data = _safe_json_load(zf, "labels.json")
            interp.dataset.var_labels = labels_data.get("var_labels", {})
            raw_val_labels = labels_data.get("val_labels", {})
            restored_val_labels = {}
            for col, mapping in raw_val_labels.items():
                if isinstance(mapping, dict):
                    clean_m = {}
                    for k, v in mapping.items():
                        try:
                            clean_m[int(k)] = v
                        except (ValueError, TypeError):
                            clean_m[k] = v
                    restored_val_labels[col] = clean_m
                else:
                    restored_val_labels[col] = mapping
            interp.dataset.val_labels = restored_val_labels
            interp.dataset.val_label_names = labels_data.get("val_label_names", {})
            interp.dataset.dataset_label = labels_data.get("dataset_label", None)
            
        # Discard results from the previously open project before restoring this one.
        if hasattr(interp, "registers"):
            interp.registers.clear_r()
            interp.registers.clear_e()
            interp.registers.rc = 0

        # 4. Stored Estimates Restoration with NumPy Ndarrays
        if "estimates.json" in namelist and hasattr(interp, "estimates"):
            est_raw = _safe_json_load(zf, "estimates.json")
            if isinstance(est_raw, dict) and ("store" in est_raw or "last_model" in est_raw):
                raw_store = est_raw.get("store", {})
                last_m = est_raw.get("last_model")
            else:
                raw_store = est_raw if isinstance(est_raw, dict) else {}
                last_m = list(raw_store.values())[0] if raw_store else None

            clean_store = {k: _reconstruct_model_arrays(m) for k, m in raw_store.items()}
            if last_m is not None:
                last_m = _reconstruct_model_arrays(last_m)
            elif clean_store:
                last_m = list(clean_store.values())[0]

            interp.estimates.store = clean_store
            interp.estimates.last_model = last_m
            
            # Immediately link _b[x] and _se[x] from last_model into registers
            if last_m and hasattr(interp, "registers"):
                if "coefs" in last_m:
                    b_dict = {item['name']: item['coef'] for item in last_m['coefs'] if 'name' in item and 'coef' in item}
                    se_dict = {item['name']: item['se'] for item in last_m['coefs'] if 'name' in item and 'se' in item}
                    interp.registers.set_estimates(b_dict, se_dict, last_m.get('vcov'))
                
        # 5. Return Registers Restoration
        if "registers.json" in namelist and hasattr(interp, "registers"):
            reg_data = _safe_json_load(zf, "registers.json")
            interp.registers.rc = reg_data.get("rc", 0)
            interp.registers.r_matrices.update(reg_data.get("r_matrices", {}))
            interp.registers.e_matrices.update(reg_data.get("e_matrices", {}))
            for k, v in reg_data.get("r_scalars", {}).items():
                interp.registers.set_r_scalar(k, v)
            for k, v in reg_data.get("r_macros", {}).items():
                interp.registers.set_r_macro(k, v)
            for k, v in reg_data.get("e_scalars", {}).items():
                interp.registers.set_e_scalar(k, v)
            for k, v in reg_data.get("e_macros", {}).items():
                interp.registers.set_e_macro(k, v)
            if "b" in reg_data and "se" in reg_data and reg_data["b"]:
                b_dict = reg_data.get("b", {})
                se_dict = reg_data.get("se", {})
                vcov = None
                if "e_matrices" in reg_data and "V" in reg_data["e_matrices"]:
                    vcov = np.asarray(reg_data["e_matrices"]["V"], dtype=float)
                interp.registers.set_estimates(b_dict, se_dict, vcov)

        # 6. Macros Restoration
        if "macros.json" in namelist:
            macros_data = _safe_json_load(zf, "macros.json")
            if hasattr(interp, "macro_mgr"):
                interp.macro_mgr.globals.clear()
                interp.macro_mgr.globals.update(macros_data.get("global_macros", {}))
                interp.macro_mgr.locals.clear()
                interp.macro_mgr.locals.update(macros_data.get("local_macros", {}))
            if hasattr(interp, "registers"):
                for k, v in macros_data.get("r_scalars", {}).items():
                    interp.registers.set_r_scalar(k, v)
                for k, v in macros_data.get("r_macros", {}).items():
                    interp.registers.set_r_macro(k, v)
                for k, v in macros_data.get("e_scalars", {}).items():
                    interp.registers.set_e_scalar(k, v)
                for k, v in macros_data.get("e_macros", {}).items():
                    interp.registers.set_e_macro(k, v)
                    
        # 7. Console Transcript
        if "console.txt" in namelist:
            restored["console_text"] = zf.read("console.txt").decode("utf-8")
            
        # 8. Command History
        if "history.json" in namelist:
            restored["history_records"] = _safe_json_load(zf, "history.json")

    return restored
