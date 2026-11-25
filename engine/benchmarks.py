"""
engine/benchmarks.py - Independent Scientific Benchmark Validation Framework

Reproducibly cross-validates Potatostics econometric estimators against established
scientific reference implementations (statsmodels, linearmodels, scipy, and analytical benchmarks).
Implements a structured result schema, relative/absolute error comparisons, parameter name matching,
and explicit validation method classifications.
"""

import os
import sys
import json
import datetime
import platform
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import scipy
import scipy.linalg
import scipy.optimize
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.api import VAR as SmVAR
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.duration.hazard_regression import PHReg

try:
    import linearmodels
    from linearmodels.iv import IV2SLS as LmIV2SLS
    from linearmodels.panel import PanelOLS as LmPanelOLS
    HAVE_LINEARMODELS = True
except ImportError:
    HAVE_LINEARMODELS = False

from engine.dataset import Dataset
from engine.estimation import (
    compute_regression, compute_panel_fe, compute_logit, compute_ivregress_2sls
)
from engine.timeseries import compute_hp_filter
from engine.advanced_econometrics import (
    compute_probit, compute_poisson,
    compute_qreg, compute_var, compute_sts_list,
    compute_stcox, compute_ologit
)
from engine.causal_modern import compute_bacondecomp, compute_drdid

POTATOSTICS_VERSION = "1.0.0"


class ValidationMethod(str, Enum):
    INDEPENDENT_EXTERNAL = "INDEPENDENT_EXTERNAL"
    ANALYTICAL_GROUND_TRUTH = "ANALYTICAL_GROUND_TRUTH"
    SIMULATION_GROUND_TRUTH = "SIMULATION_GROUND_TRUTH"
    INTERNAL_CONSISTENCY = "INTERNAL_CONSISTENCY"


class BenchmarkStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_TESTED = "NOT_TESTED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class MetricStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_TESTED = "NOT_TESTED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class MetricType(str, Enum):
    SCALAR_NUMERICAL = "SCALAR_NUMERICAL"
    VECTOR_NUMERICAL = "VECTOR_NUMERICAL"
    MATRIX_NUMERICAL = "MATRIX_NUMERICAL"
    NAMED_PARAMETER = "NAMED_PARAMETER"
    BOOLEAN_ASSERTION = "BOOLEAN_ASSERTION"
    FINITE_DIAGNOSTIC = "FINITE_DIAGNOSTIC"
    NOT_TESTED_RECORD = "NOT_TESTED_RECORD"


def _validate_tolerance(tol_val: Any, tol_name: str) -> Optional[float]:
    if tol_val is None:
        return None
    if isinstance(tol_val, bool) or not isinstance(tol_val, (int, float, np.number)):
        raise ValueError(f"Tolerance '{tol_name}' must be numeric, got {type(tol_val).__name__}")
    float_val = float(tol_val)
    if np.isnan(float_val) or np.isinf(float_val):
        raise ValueError(f"Tolerance '{tol_name}' cannot be NaN or Inf")
    if float_val < 0:
        raise ValueError(f"Tolerance '{tol_name}' cannot be negative, got {float_val}")
    return float_val


def create_benchmark_metric(
    metric_name: str,
    potatostics_value: Any,
    reference_value: Any,
    difference: Optional[float] = None,
    abs_tolerance: Optional[float] = None,
    rel_tolerance: Optional[float] = None,
    evaluation_method: str = "numerical_comparison",
    is_mandatory: bool = True,
    status: Optional[Union[MetricStatus, str]] = None,
    failure_explanation: Optional[str] = None,
    metric_type: Optional[Union[MetricType, str]] = None,
    key_aliases: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Creates and strictly evaluates a benchmark metric conforming to Phase 5 schema.
    Automatic evaluation rule: caller-supplied status=PASS CANNOT override numerical failures,
    tolerance violations, shape mismatches, or non-finite values.
    """
    # Validate tolerances
    abs_tolerance = _validate_tolerance(abs_tolerance, "abs_tolerance")
    rel_tolerance = _validate_tolerance(rel_tolerance, "rel_tolerance")

    # Validate status if provided
    valid_statuses = {s.value for s in MetricStatus}
    if status is not None:
        if isinstance(status, MetricStatus):
            raw_status = status.value
        elif isinstance(status, str):
            if status not in valid_statuses:
                raise ValueError(f"Invalid or unrecognized metric status '{status}'. Must be one of: {sorted(valid_statuses)}")
            raw_status = status
        else:
            raise ValueError(f"Invalid type for status: {type(status).__name__}")
    else:
        raw_status = None

    # Validate or infer metric type
    valid_types = {t.value for t in MetricType}
    if metric_type is not None:
        if isinstance(metric_type, MetricType):
            m_type = metric_type
        elif isinstance(metric_type, str):
            if metric_type not in valid_types:
                raise ValueError(f"unrecognized metric_type '{metric_type}'. Must be one of: {sorted(valid_types)}")
            m_type = MetricType(metric_type)
        else:
            raise ValueError(f"Invalid type for metric_type: {type(metric_type).__name__}")
    else:
        if raw_status in (MetricStatus.NOT_TESTED.value, MetricStatus.SKIPPED.value):
            m_type = MetricType.NOT_TESTED_RECORD
        elif potatostics_value is None and reference_value is None:
            # Missing values without explicit NOT_TESTED/SKIPPED must be evaluated as SCALAR_NUMERICAL
            m_type = MetricType.SCALAR_NUMERICAL
        elif reference_value is None and potatostics_value is not None:
            if isinstance(potatostics_value, dict) and "potatostics_values" in potatostics_value and "reference_values" in potatostics_value:
                m_type = MetricType.NAMED_PARAMETER
            else:
                m_type = MetricType.FINITE_DIAGNOSTIC
        elif isinstance(potatostics_value, bool) or isinstance(reference_value, bool):
            m_type = MetricType.BOOLEAN_ASSERTION
        elif isinstance(potatostics_value, dict) or isinstance(reference_value, dict):
            m_type = MetricType.NAMED_PARAMETER
        else:
            is_arr_pot = isinstance(potatostics_value, (list, tuple, np.ndarray))
            is_arr_ref = isinstance(reference_value, (list, tuple, np.ndarray))
            if is_arr_pot or is_arr_ref:
                arr_p = np.asarray(potatostics_value) if potatostics_value is not None else None
                arr_r = np.asarray(reference_value) if reference_value is not None else None
                if (arr_p is not None and arr_p.ndim >= 2) or (arr_r is not None and arr_r.ndim >= 2):
                    m_type = MetricType.MATRIX_NUMERICAL
                else:
                    m_type = MetricType.VECTOR_NUMERICAL
            elif isinstance(potatostics_value, (int, float, np.number)) or isinstance(reference_value, (int, float, np.number)):
                m_type = MetricType.SCALAR_NUMERICAL
            else:
                m_type = MetricType.SCALAR_NUMERICAL

    # Evaluation
    if raw_status in (MetricStatus.NOT_TESTED.value, MetricStatus.SKIPPED.value):
        eval_status = MetricStatus(raw_status)
        eval_diff = difference
        eval_failure = failure_explanation
    elif raw_status == MetricStatus.ERROR.value:
        eval_status = MetricStatus.ERROR
        eval_diff = difference
        eval_failure = failure_explanation or "Metric evaluation errored."
    elif potatostics_value is None and reference_value is None:
        eval_status = MetricStatus.FAIL
        eval_failure = failure_explanation or "Missing numerical comparison values without explicit NOT_TESTED or SKIPPED status."
        eval_diff = float("nan")
    else:
        eval_status = MetricStatus.PASS
        eval_diff = difference
        eval_failure = failure_explanation

        if m_type == MetricType.BOOLEAN_ASSERTION:
            is_p_bool = isinstance(potatostics_value, (bool, np.bool_))
            is_r_bool = isinstance(reference_value, (bool, np.bool_))
            if not is_p_bool or not is_r_bool:
                eval_status = MetricStatus.FAIL
                eval_failure = eval_failure or f"Strict bool validation failed: potatostics is {type(potatostics_value).__name__}, reference is {type(reference_value).__name__}"
                eval_diff = 1.0
            elif bool(potatostics_value) != bool(reference_value):
                eval_status = MetricStatus.FAIL
                eval_failure = eval_failure or f"Boolean assertion mismatch: potatostics={potatostics_value}, reference={reference_value}"
                eval_diff = 1.0
            else:
                eval_diff = 0.0

        elif m_type == MetricType.FINITE_DIAGNOSTIC:
            arr = np.asarray(potatostics_value)
            if arr.size == 0 and is_mandatory:
                eval_status = MetricStatus.FAIL
                eval_failure = eval_failure or "Empty diagnostic array."
            elif not np.all(np.isfinite(arr)):
                eval_status = MetricStatus.FAIL
                eval_failure = eval_failure or "Non-finite values (NaN or Inf) detected in diagnostic."
            else:
                eval_diff = 0.0

        elif m_type == MetricType.NAMED_PARAMETER:
            default_aliases = {"_cons": "const", "const": "const", "Intercept": "const", "intercept": "const"}
            if key_aliases is not None:
                if not isinstance(key_aliases, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in key_aliases.items()):
                    eval_status = MetricStatus.FAIL
                    eval_failure = eval_failure or "Invalid key_aliases format: must be Dict[str, str]"
                    active_aliases = default_aliases
                else:
                    active_aliases = dict(default_aliases)
                    active_aliases.update(key_aliases)
            else:
                active_aliases = default_aliases

            if eval_status != MetricStatus.FAIL:
                if isinstance(potatostics_value, dict) and "potatostics_values" in potatostics_value and "reference_values" in potatostics_value:
                    # Recompute comparison from raw values regardless of any 'passed' key
                    comp = compare_numerical_dictionaries(
                        potatostics_value["potatostics_values"],
                        potatostics_value["reference_values"],
                        abs_tol=abs_tolerance if abs_tolerance is not None else 1e-7,
                        rel_tol=rel_tolerance if rel_tolerance is not None else 1e-5,
                        key_aliases=active_aliases
                    )
                    if not comp.get("passed", False):
                        eval_status = MetricStatus.FAIL
                        reasons = comp.get("failure_reasons", [])
                        eval_failure = eval_failure or ("; ".join(reasons) if reasons else "Named parameter comparison failed.")
                    eval_diff = comp.get("max_abs_diff", eval_diff)
                elif isinstance(potatostics_value, dict) and isinstance(reference_value, dict):
                    comp = compare_numerical_dictionaries(
                        potatostics_value, reference_value,
                        abs_tol=abs_tolerance if abs_tolerance is not None else 1e-7,
                        rel_tol=rel_tolerance if rel_tolerance is not None else 1e-5,
                        key_aliases=active_aliases
                    )
                    if not comp.get("passed", False):
                        eval_status = MetricStatus.FAIL
                        reasons = comp.get("failure_reasons", [])
                        eval_failure = eval_failure or ("; ".join(reasons) if reasons else "Named parameter comparison failed.")
                    eval_diff = comp.get("max_abs_diff", eval_diff)
                else:
                    eval_status = MetricStatus.FAIL
                    eval_failure = eval_failure or "Invalid dictionary comparison values."

        elif m_type == MetricType.SCALAR_NUMERICAL:
            try:
                p_val = float(potatostics_value)
                r_val = float(reference_value)
                if np.isnan(p_val) or np.isnan(r_val) or np.isinf(p_val) or np.isinf(r_val):
                    eval_status = MetricStatus.FAIL
                    eval_failure = eval_failure or f"Non-finite scalar encountered: potatostics={p_val}, reference={r_val}"
                    eval_diff = float("nan")
                else:
                    d = abs(p_val - r_val)
                    eval_diff = d
                    abs_tol = abs_tolerance if abs_tolerance is not None else 1e-7
                    rel_tol = rel_tolerance if rel_tolerance is not None else 1e-5
                    # Elementwise allclose condition: abs(p - r) <= abs_tol + rel_tol * abs(r)
                    tol_threshold = abs_tol + rel_tol * abs(r_val)
                    if d > tol_threshold:
                        eval_status = MetricStatus.FAIL
                        eval_failure = eval_failure or f"Tolerance exceeded: diff={d:g} > tol_threshold={tol_threshold:g} (abs_tol={abs_tol:g}, rel_tol={rel_tol:g})"
            except Exception as ex:
                eval_status = MetricStatus.FAIL
                eval_failure = eval_failure or f"Failed to evaluate scalar numerical values: {ex}"

        elif m_type == MetricType.VECTOR_NUMERICAL:
            try:
                p_arr = np.asarray(potatostics_value, dtype=float)
                r_arr = np.asarray(reference_value, dtype=float)
                if p_arr.size == 0 or r_arr.size == 0:
                    eval_status = MetricStatus.FAIL
                    eval_failure = eval_failure or "Empty vector comparison set."
                    eval_diff = float("nan")
                elif p_arr.shape != r_arr.shape:
                    eval_status = MetricStatus.FAIL
                    eval_failure = eval_failure or f"Vector shape mismatch: potatostics {p_arr.shape} != reference {r_arr.shape}"
                    eval_diff = float("nan")
                elif not (np.all(np.isfinite(p_arr)) and np.all(np.isfinite(r_arr))):
                    eval_status = MetricStatus.FAIL
                    eval_failure = eval_failure or "Non-finite values (NaN/Inf) detected in vector."
                    eval_diff = float("nan")
                else:
                    abs_err = np.abs(p_arr - r_arr)
                    d = float(np.max(abs_err))
                    eval_diff = d
                    abs_tol = abs_tolerance if abs_tolerance is not None else 1e-7
                    rel_tol = rel_tolerance if rel_tolerance is not None else 1e-5
                    tol_thresh = abs_tol + rel_tol * np.abs(r_arr)
                    violations = np.where(abs_err > tol_thresh)[0]
                    if len(violations) > 0:
                        bad_idx = int(violations[0])
                        eval_status = MetricStatus.FAIL
                        eval_failure = eval_failure or f"Vector element {bad_idx} diff={abs_err[bad_idx]:g} > tolerance {tol_thresh[bad_idx]:g} (abs_tol={abs_tol:g}, rel_tol={rel_tol:g})"
            except Exception as ex:
                eval_status = MetricStatus.FAIL
                eval_failure = eval_failure or f"Failed to evaluate vector numerical values: {ex}"

        elif m_type == MetricType.MATRIX_NUMERICAL:
            try:
                p_mat = np.asarray(potatostics_value, dtype=float)
                r_mat = np.asarray(reference_value, dtype=float)
                if p_mat.size == 0 or r_mat.size == 0:
                    eval_status = MetricStatus.FAIL
                    eval_failure = eval_failure or "Empty matrix comparison set."
                    eval_diff = float("nan")
                elif p_mat.shape != r_mat.shape:
                    eval_status = MetricStatus.FAIL
                    eval_failure = eval_failure or f"Matrix shape mismatch: potatostics {p_mat.shape} != reference {r_mat.shape}"
                    eval_diff = float("nan")
                elif not (np.all(np.isfinite(p_mat)) and np.all(np.isfinite(r_mat))):
                    eval_status = MetricStatus.FAIL
                    eval_failure = eval_failure or "Non-finite values (NaN/Inf) detected in matrix."
                    eval_diff = float("nan")
                else:
                    abs_err = np.abs(p_mat - r_mat)
                    d = float(np.max(abs_err))
                    eval_diff = d
                    abs_tol = abs_tolerance if abs_tolerance is not None else 1e-7
                    rel_tol = rel_tolerance if rel_tolerance is not None else 1e-5
                    tol_thresh = abs_tol + rel_tol * np.abs(r_mat)
                    row_viol, col_viol = np.where(abs_err > tol_thresh)
                    if len(row_viol) > 0:
                        r_idx, c_idx = int(row_viol[0]), int(col_viol[0])
                        eval_status = MetricStatus.FAIL
                        eval_failure = eval_failure or f"Matrix element ({r_idx}, {c_idx}) diff={abs_err[r_idx, c_idx]:g} > tolerance {tol_thresh[r_idx, c_idx]:g} (abs_tol={abs_tol:g}, rel_tol={rel_tol:g})"
            except Exception as ex:
                eval_status = MetricStatus.FAIL
                eval_failure = eval_failure or f"Failed to evaluate matrix numerical values: {ex}"

        # Caller-provided difference is untrusted; internal elementwise evaluation above is authoritative.
        if raw_status == MetricStatus.FAIL.value:
            eval_status = MetricStatus.FAIL

    def _to_json_val(val):
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, (np.floating, float)):
            return float(val)
        if isinstance(val, (np.integer, int)):
            return int(val)
        if isinstance(val, dict):
            return {str(k): _to_json_val(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [_to_json_val(v) for v in val]
        return val

    return {
        "metric_name": metric_name,
        "metric_type": m_type.value if isinstance(m_type, MetricType) else str(m_type),
        "potatostics_value": _to_json_val(potatostics_value),
        "reference_value": _to_json_val(reference_value),
        "difference": float(eval_diff) if eval_diff is not None and np.isfinite(eval_diff) else (eval_diff if eval_diff is not None else difference),
        "abs_tolerance": abs_tolerance,
        "rel_tolerance": rel_tolerance,
        "evaluation_method": evaluation_method,
        "is_mandatory": bool(is_mandatory),
        "status": eval_status.value if isinstance(eval_status, MetricStatus) else str(eval_status),
        "failure_explanation": eval_failure,
        "key_aliases": dict(key_aliases) if isinstance(key_aliases, dict) else None
    }


def aggregate_benchmark_status(metrics: List[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    """
    Central status aggregation function for benchmark evaluation.
    Rules:
    1. If any metric is ERROR: overall status is ERROR.
    2. If any mandatory metric is FAIL: overall status is FAIL.
    3. If any mandatory metric is NOT_TESTED or SKIPPED: overall status is NOT_TESTED / SKIPPED.
    4. If zero mandatory metrics passed: overall status cannot be PASS.
    5. Only if all mandatory metrics pass: overall status is PASS.
    """
    if not metrics:
        return BenchmarkStatus.FAIL.value, "No metrics evaluated."

    valid_statuses = {s.value for s in MetricStatus}
    # Validate schema and status
    for m in metrics:
        if not isinstance(m, dict):
            return BenchmarkStatus.FAIL.value, "Metric item is not a dictionary."
        if not ("metric_name" in m and "status" in m and "is_mandatory" in m):
            return BenchmarkStatus.FAIL.value, f"Metric '{m.get('metric_name', 'unknown')}' missing required schema fields."
        stat = m.get("status")
        if stat not in valid_statuses:
            return BenchmarkStatus.FAIL.value, f"Metric '{m.get('metric_name', 'unknown')}' contains unrecognized status: '{stat}'"
        # Extended schema validation for Phase 1.1
        extended_fields = ("metric_type", "potatostics_value", "reference_value")
        missing_ext = [f for f in extended_fields if f not in m]
        if missing_ext:
            return BenchmarkStatus.FAIL.value, f"Metric '{m.get('metric_name', 'unknown')}' missing required schema fields: {missing_ext}"

    mandatory_metrics = [m for m in metrics if m.get("is_mandatory", True)]
    if not mandatory_metrics:
        return BenchmarkStatus.FAIL.value, "No mandatory metrics specified for benchmark."

    failures = []
    not_tested = []
    skipped = []
    errors = []
    passed_mandatory = 0

    for m in mandatory_metrics:
        stat = m.get("status")
        name = m.get("metric_name", "unknown")
        expl = m.get("failure_explanation") or "Criterion violated"
        m_type = m.get("metric_type")
        if stat == MetricStatus.ERROR.value:
            errors.append(f"{name}: {expl}")
        elif stat == MetricStatus.FAIL.value:
            failures.append(f"{name}: {expl}")
        elif stat == MetricStatus.NOT_TESTED.value:
            not_tested.append(f"{name}: {expl}")
        elif stat == MetricStatus.SKIPPED.value:
            skipped.append(f"{name}: {expl}")
        elif stat == MetricStatus.PASS.value:
            # Strictly verify underlying values - eliminate fabricated PASS records!
            try:
                # Validate key_aliases format if provided
                metric_aliases = m.get("key_aliases")
                if metric_aliases is not None:
                    if not isinstance(metric_aliases, dict):
                        failures.append(f"{name}: Invalid key_aliases type in metric schema: {type(metric_aliases).__name__}")
                        continue
                    if any(not isinstance(ak, str) or not isinstance(av, str) for ak, av in metric_aliases.items()):
                        failures.append(f"{name}: Invalid key_aliases mapping (all keys and values must be strings)")
                        continue

                rechecked = create_benchmark_metric(
                    metric_name=name,
                    potatostics_value=m["potatostics_value"],
                    reference_value=m["reference_value"],
                    abs_tolerance=m.get("abs_tolerance"),
                    rel_tolerance=m.get("rel_tolerance"),
                    evaluation_method=m.get("evaluation_method", "numerical_comparison"),
                    is_mandatory=m.get("is_mandatory", True),
                    status=None,  # Do not trust caller supplied PASS
                    failure_explanation=m.get("failure_explanation"),
                    metric_type=m.get("metric_type"),
                    key_aliases=metric_aliases
                )
                if rechecked["status"] != MetricStatus.PASS.value:
                    failures.append(f"{name}: Claimed PASS but evaluated to {rechecked['status']}: {rechecked.get('failure_explanation')}")
                    continue
            except Exception as ex:
                failures.append(f"{name}: Metric verification failed: {ex}")
                continue

            if m_type != MetricType.NOT_TESTED_RECORD.value:
                passed_mandatory += 1

    if errors:
        return BenchmarkStatus.ERROR.value, "; ".join(errors)
    if failures:
        return BenchmarkStatus.FAIL.value, "; ".join(failures)
    if not_tested:
        return BenchmarkStatus.NOT_TESTED.value, f"Mandatory metric not tested: {'; '.join(not_tested)}"
    if skipped:
        return BenchmarkStatus.SKIPPED.value, f"Mandatory metric skipped: {'; '.join(skipped)}"
    if passed_mandatory == 0:
        return BenchmarkStatus.FAIL.value, "Zero mandatory metrics passed."

    return BenchmarkStatus.PASS.value, None


def compare_numerical_dictionaries(
    pot_dict: Dict[str, float],
    ref_dict: Dict[str, float],
    abs_tol: float = 1e-7,
    rel_tol: float = 1e-5,
    key_aliases: Optional[Dict[str, str]] = None,
    allow_empty: bool = False,
    allow_nan: bool = False,
    allow_inf: bool = False
) -> Dict[str, Any]:
    """
    Compares two dictionaries of parameter names to values under strict scientific criteria:
    - Empty comparison sets fail unless allow_empty=True.
    - Unexpected NaN and Infinities fail unless explicitly permitted.
    - Missing parameter names fail.
    - Duplicate normalized parameter names fail.
    - Parameter ordering does not affect comparisons.
    - Explicit failure reasons returned.
    """
    failure_reasons: List[str] = []

    # Validate key_aliases format if provided
    if key_aliases is not None:
        if not isinstance(key_aliases, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in key_aliases.items()):
            failure_reasons.append("Invalid key_aliases format: must be Dict[str, str]")
            return {
                "matched_keys": [],
                "missing_in_potatostics": [],
                "missing_in_reference": [],
                "potatostics_values": {},
                "reference_values": {},
                "absolute_differences": {},
                "relative_differences": {},
                "max_abs_diff": float("nan"),
                "max_rel_diff": float("nan"),
                "abs_tolerance": float(abs_tol if abs_tol is not None else 1e-7),
                "rel_tolerance": float(rel_tol if rel_tol is not None else 1e-5),
                "failure_reasons": failure_reasons,
                "passed": False
            }

    key_aliases = key_aliases or {}

    # Validate tolerances strictly (reject NaN, Inf, negative)
    abs_tol = _validate_tolerance(abs_tol, "abs_tol")
    rel_tol = _validate_tolerance(rel_tol, "rel_tol")
    if abs_tol is None:
        abs_tol = 1e-7
    if rel_tol is None:
        rel_tol = 1e-5

    # Empty check
    if not pot_dict or not ref_dict:
        if not allow_empty:
            failure_reasons.append(
                f"Empty comparison set: potatostics has {len(pot_dict or {})} keys, "
                f"reference has {len(ref_dict or {})} keys"
            )
            return {
                "matched_keys": [],
                "missing_in_potatostics": list(ref_dict.keys()) if ref_dict else [],
                "missing_in_reference": list(pot_dict.keys()) if pot_dict else [],
                "potatostics_values": {},
                "reference_values": {},
                "absolute_differences": {},
                "relative_differences": {},
                "max_abs_diff": float("nan"),
                "max_rel_diff": float("nan"),
                "abs_tolerance": float(abs_tol),
                "rel_tolerance": float(rel_tol),
                "failure_reasons": failure_reasons,
                "passed": False
            }

    # Normalize reference keys and detect duplicates / non-numeric values
    norm_ref = {}
    for k, v in ref_dict.items():
        alias_k = key_aliases.get(k, k)
        if alias_k in norm_ref:
            failure_reasons.append(f"Duplicate normalized reference parameter name: '{alias_k}'")
        if isinstance(v, bool) or not isinstance(v, (int, float, np.number)):
            failure_reasons.append(f"Non-numeric parameter value for reference '{k}': {type(v).__name__}")
            norm_ref[alias_k] = float("nan")
        else:
            norm_ref[alias_k] = float(v)

    # Normalize potatostics keys and detect duplicates / non-numeric values
    norm_pot = {}
    for k, v in pot_dict.items():
        alias_k = key_aliases.get(k, k)
        if alias_k in norm_pot:
            failure_reasons.append(f"Duplicate normalized potatostics parameter name: '{alias_k}'")
        if isinstance(v, bool) or not isinstance(v, (int, float, np.number)):
            failure_reasons.append(f"Non-numeric parameter value for potatostics '{k}': {type(v).__name__}")
            norm_pot[alias_k] = float("nan")
        else:
            norm_pot[alias_k] = float(v)

    if failure_reasons:
        return {
            "matched_keys": [],
            "missing_in_potatostics": [],
            "missing_in_reference": [],
            "potatostics_values": norm_pot,
            "reference_values": norm_ref,
            "absolute_differences": {},
            "relative_differences": {},
            "max_abs_diff": float("nan"),
            "max_rel_diff": float("nan"),
            "abs_tolerance": float(abs_tol),
            "rel_tolerance": float(rel_tol),
            "failure_reasons": failure_reasons,
            "passed": False
        }

    all_keys = sorted(set(norm_pot.keys()).union(set(norm_ref.keys())))
    matched_keys = []
    missing_pot = []
    missing_ref = []
    abs_diffs = {}
    rel_diffs = {}
    passed = True

    for k in all_keys:
        if k not in norm_pot:
            missing_pot.append(k)
            failure_reasons.append(f"Missing parameter in potatostics: '{k}'")
            passed = False
            continue
        if k not in norm_ref:
            missing_ref.append(k)
            failure_reasons.append(f"Missing parameter in reference: '{k}'")
            passed = False
            continue

        matched_keys.append(k)
        p_val = norm_pot[k]
        r_val = norm_ref[k]

        # Nonfinite checks
        if np.isnan(p_val) or np.isnan(r_val):
            if allow_nan and np.isnan(p_val) and np.isnan(r_val):
                abs_d = 0.0
                rel_d = 0.0
            else:
                failure_reasons.append(f"Unexpected NaN in parameter '{k}': potatostics={p_val}, reference={r_val}")
                passed = False
                abs_d = float("nan")
                rel_d = float("nan")
        elif np.isinf(p_val) or np.isinf(r_val):
            if allow_inf and np.isinf(p_val) and np.isinf(r_val) and (np.sign(p_val) == np.sign(r_val)):
                abs_d = 0.0
                rel_d = 0.0
            else:
                failure_reasons.append(f"Unexpected Infinity in parameter '{k}': potatostics={p_val}, reference={r_val}")
                passed = False
                abs_d = float("nan")
                rel_d = float("nan")
        else:
            abs_d = abs(p_val - r_val)
            rel_d = abs_d / max(abs(r_val), 1e-12)
            allowed_d = abs_tol + rel_tol * abs(r_val)

            if abs_d > allowed_d:
                failure_reasons.append(
                    f"Parameter '{k}' exceeded tolerance: abs_diff={abs_d:.3e} > allowed={allowed_d:.3e} "
                    f"(atol={abs_tol:.3e} + rtol={rel_tol:.3e} * |ref|)"
                )
                passed = False

        abs_diffs[k] = float(abs_d)
        rel_diffs[k] = float(rel_d)

    if not matched_keys and not allow_empty:
        passed = False
        failure_reasons.append("Zero matched parameters between potatostics and reference.")

    valid_abs = [v for v in abs_diffs.values() if not np.isnan(v)]
    valid_rel = [v for v in rel_diffs.values() if not np.isnan(v)]
    max_abs = max(valid_abs) if valid_abs else (float("nan") if not passed else 0.0)
    max_rel = max(valid_rel) if valid_rel else (float("nan") if not passed else 0.0)

    return {
        "matched_keys": matched_keys,
        "missing_in_potatostics": missing_pot,
        "missing_in_reference": missing_ref,
        "potatostics_values": {k: norm_pot[k] for k in matched_keys},
        "reference_values": {k: norm_ref[k] for k in matched_keys},
        "absolute_differences": abs_diffs,
        "relative_differences": rel_diffs,
        "max_abs_diff": float(max_abs),
        "max_rel_diff": float(max_rel),
        "abs_tolerance": float(abs_tol),
        "rel_tolerance": float(rel_tol),
        "failure_reasons": failure_reasons,
        "passed": bool(passed)
    }


def compare_named_covariance_matrices(
    pot_cov: Any,
    ref_cov: Any,
    pot_labels: List[str],
    ref_labels: List[str],
    key_aliases: Optional[Dict[str, str]] = None,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-5
) -> Dict[str, Any]:
    """
    Compares two covariance matrices whose rows and columns correspond to named parameters.
    Normalizes parameter names using key_aliases, detects duplicate or missing parameters,
    reorders/aligns the matrices to identical parameter order, and performs elementwise
    comparison with scientific tolerances.
    """
    failure_reasons: List[str] = []

    # Validate key_aliases format if provided
    if key_aliases is not None:
        if not isinstance(key_aliases, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in key_aliases.items()):
            failure_reasons.append("Invalid key_aliases format: must be Dict[str, str]")
            return {
                "matched_keys": [],
                "missing_in_potatostics": [],
                "missing_in_reference": [],
                "potatostics_values": {},
                "reference_values": {},
                "absolute_differences": {},
                "relative_differences": {},
                "max_abs_diff": float("nan"),
                "max_rel_diff": float("nan"),
                "abs_tolerance": float(abs_tol if abs_tol is not None else 1e-7),
                "rel_tolerance": float(rel_tol if rel_tol is not None else 1e-5),
                "failure_reasons": failure_reasons,
                "passed": False
            }

    key_aliases = key_aliases or {}

    # Validate aliases format
    if not isinstance(key_aliases, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in key_aliases.items()):
        failure_reasons.append("Invalid key_aliases format: must be Dict[str, str]")
        return {
            "passed": False,
            "max_abs_diff": float("nan"),
            "aligned_parameters": [],
            "failure_reasons": failure_reasons
        }

    # Validate tolerances
    abs_tol = _validate_tolerance(abs_tol, "abs_tol")
    rel_tol = _validate_tolerance(rel_tol, "rel_tol")
    if abs_tol is None:
        abs_tol = 1e-6
    if rel_tol is None:
        rel_tol = 1e-5

    p_mat = np.asarray(pot_cov, dtype=float)
    r_mat = np.asarray(ref_cov, dtype=float)

    # Validate dimensions
    if p_mat.ndim != 2 or p_mat.shape[0] != p_mat.shape[1]:
        failure_reasons.append(f"Potatostics covariance matrix must be 2D square: shape={p_mat.shape}")
    if r_mat.ndim != 2 or r_mat.shape[0] != r_mat.shape[1]:
        failure_reasons.append(f"Reference covariance matrix must be 2D square: shape={r_mat.shape}")
    if len(pot_labels) != p_mat.shape[0]:
        failure_reasons.append(f"Potatostics labels count ({len(pot_labels)}) != matrix dim ({p_mat.shape[0]})")
    if len(ref_labels) != r_mat.shape[0]:
        failure_reasons.append(f"Reference labels count ({len(ref_labels)}) != matrix dim ({r_mat.shape[0]})")

    if failure_reasons:
        return {
            "passed": False,
            "max_abs_diff": float("nan"),
            "aligned_parameters": [],
            "failure_reasons": failure_reasons
        }

    # Normalize parameter names
    norm_pot_labels = [key_aliases.get(k, k) for k in pot_labels]
    norm_ref_labels = [key_aliases.get(k, k) for k in ref_labels]

    # Check duplicates
    if len(norm_pot_labels) != len(set(norm_pot_labels)):
        failure_reasons.append("Duplicate normalized parameter name in Potatostics labels")
    if len(norm_ref_labels) != len(set(norm_ref_labels)):
        failure_reasons.append("Duplicate normalized parameter name in Reference labels")

    set_p = set(norm_pot_labels)
    set_r = set(norm_ref_labels)
    missing_in_pot = sorted(set_r - set_p)
    missing_in_ref = sorted(set_p - set_r)
    if missing_in_pot:
        failure_reasons.append(f"Parameters missing in Potatostics: {missing_in_pot}")
    if missing_in_ref:
        failure_reasons.append(f"Parameters missing in Reference: {missing_in_ref}")

    if failure_reasons:
        return {
            "passed": False,
            "max_abs_diff": float("nan"),
            "aligned_parameters": [],
            "failure_reasons": failure_reasons
        }

    # Align matrices to a canonical sorted order of parameters
    aligned_names = sorted(set_p)
    pot_idx = [norm_pot_labels.index(name) for name in aligned_names]
    ref_idx = [norm_ref_labels.index(name) for name in aligned_names]

    aligned_pot = p_mat[np.ix_(pot_idx, pot_idx)]
    aligned_ref = r_mat[np.ix_(ref_idx, ref_idx)]

    # Check non-finites
    if not (np.all(np.isfinite(aligned_pot)) and np.all(np.isfinite(aligned_ref))):
        failure_reasons.append("Non-finite values (NaN/Inf) detected in covariance matrix")
        return {
            "passed": False,
            "max_abs_diff": float("nan"),
            "aligned_parameters": aligned_names,
            "failure_reasons": failure_reasons
        }

    abs_diff = np.abs(aligned_pot - aligned_ref)
    max_diff = float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0
    tol_thresh = abs_tol + rel_tol * np.abs(aligned_ref)
    row_viol, col_viol = np.where(abs_diff > tol_thresh)
    if len(row_viol) > 0:
        r_name = aligned_names[row_viol[0]]
        c_name = aligned_names[col_viol[0]]
        failure_reasons.append(
            f"Covariance element ({r_name}, {c_name}) diff={abs_diff[row_viol[0], col_viol[0]]:.3e} > "
            f"tolerance {tol_thresh[row_viol[0], col_viol[0]]:.3e}"
        )

    passed = len(failure_reasons) == 0
    return {
        "passed": passed,
        "max_abs_diff": max_diff,
        "aligned_parameters": aligned_names,
        "failure_reasons": failure_reasons
    }


def _load_benchmark_data(filename: str) -> pd.DataFrame:
    """Helper to locate and load benchmark datasets from package resources or directory paths."""
    candidates = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidates.extend([
            os.path.join(sys._MEIPASS, "engine", "assets", "data", filename),
            os.path.join(sys._MEIPASS, filename)
        ])
    try:
        import importlib.resources as pkg_resources
        ref = pkg_resources.files("engine.assets.data") / filename
        if ref.is_file():
            candidates.append(str(ref))
    except Exception:
        pass

    candidates.extend([
        os.path.abspath(os.path.join(os.path.dirname(__file__), "assets", "data", filename)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "data", filename)),
        os.path.abspath(os.path.join(os.getcwd(), "engine", "assets", "data", filename)),
        os.path.abspath(os.path.join(os.getcwd(), "tests", "data", filename)),
        os.path.abspath(os.path.join(os.getcwd(), filename))
    ])
    found_path = None
    for p in candidates:
        if os.path.exists(p):
            found_path = p
            break

    if not found_path:
        raise FileNotFoundError(
            f"Benchmark dataset '{filename}' not found. Searched paths: {candidates}. "
            f"Ensure test datasets are bundled in package resources or present in 'engine/assets/data/'."
        )

    if filename.endswith(".dta"):
        ds = Dataset()
        ds.load_dta(found_path)
        return ds.df
    elif filename.endswith(".csv"):
        return pd.read_csv(found_path)
    raise ValueError(f"Unsupported format: {filename}")


class BenchmarkRunner:
    """
    Executes a comprehensive battery of independent econometric benchmarks,
    logging numerical deviations under a structured, scientifically defensible schema.
    """

    def __init__(self, output_dir: str = "tests/data/benchmark_results"):
        self.output_dir = os.path.abspath(output_dir)
        self.results: List[Dict[str, Any]] = []
        self.run_timestamp: Optional[str] = None

    def run_all(self) -> List[Dict[str, Any]]:
        self.results = []
        self.run_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        os.makedirs(self.output_dir, exist_ok=True)

        self.bench_01_ols_classical()
        self.bench_02_ols_robust_hc1()
        self.bench_03_ols_cluster()
        self.bench_04_logit_mle()
        self.bench_05_probit_mle()
        self.bench_06_poisson_mle()
        self.bench_07_quantile_regression()
        if HAVE_LINEARMODELS:
            self.bench_08_ivregress_2sls()
            self.bench_09_panel_fe()
        else:
            self._record_skipped("BENCH_08_IVREGRESS_2SLS", "ivregress", "linearmodels not installed")
            self._record_skipped("BENCH_09_PANEL_FE", "xtreg_fe", "linearmodels not installed")

        self.bench_10_hp_filter()
        self.bench_11_var()
        self.bench_12_kaplan_meier()
        self.bench_13_bacondecomp()
        self.bench_14_drdid()
        self.bench_15_hp_filter_analytical()
        self.bench_16_var_analytical_ols()
        self.bench_17_cox_breslow_analytical()
        self.bench_18_ordered_logit()
        self.bench_19_cox_phreg()

        self._save_results()
        return self.results

    def _record_skipped(self, bench_id: str, estimator: str, reason: str):
        entry = {
            "benchmark_id": bench_id,
            "estimator_name": estimator,
            "dataset_id": "N/A",
            "specification": "N/A",
            "reference_implementation": "N/A",
            "reference_version": "N/A",
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": 0,
            "sample_selection": "N/A",
            "coefficient_comparison": None,
            "standard_error_comparison": None,
            "covariance_comparison": None,
            "metrics": [],
            "additional_diagnostics": {},
            "abs_tolerance": 0.0,
            "rel_tolerance": 0.0,
            "validation_method": ValidationMethod.INDEPENDENT_EXTERNAL.value,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "execution_environment": self._get_env_info(),
            "status": BenchmarkStatus.SKIPPED.value,
            "failure_explanation": reason
        }
        self.results.append(entry)

    def _get_env_info(self) -> Dict[str, str]:
        return {
            "python_version": sys.version.split()[0],
            "os": platform.system(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown"
        }

    # -------------------------------------------------------------------------
    # 1. OLS Classical (Homoskedastic)
    # -------------------------------------------------------------------------
    def bench_01_ols_classical(self):
        bench_id = "BENCH_01_OLS_CLASSICAL"
        df = _load_benchmark_data("auto.dta")
        r = compute_regression(df, "price", ["mpg", "weight"], robust=False)
        X = sm.add_constant(df[["mpg", "weight"]])
        sm_model = sm.OLS(df["price"], X).fit()

        pot_b = {c["name"]: c["coef"] for c in r["coefs"]}
        sm_b = sm_model.params.to_dict()
        aliases = {"_cons": "const"}
        b_comp = compare_numerical_dictionaries(pot_b, sm_b, abs_tol=1e-7, rel_tol=1e-5, key_aliases=aliases)

        pot_se = {c["name"]: c["se"] for c in r["coefs"]}
        sm_se = sm_model.bse.to_dict()
        se_comp = compare_numerical_dictionaries(pot_se, sm_se, abs_tol=1e-7, rel_tol=1e-5, key_aliases=aliases)

        pot_names = [c["name"] for c in r["coefs"]]
        mapped_names = ["const" if n == "_cons" else n for n in pot_names]
        sm_cov = sm_model.cov_params().loc[mapped_names, mapped_names].to_numpy()
        cov_diff = float(np.max(np.abs(r["vcov"] - sm_cov)))
        metrics = [
            create_benchmark_metric(
                "coefficients", pot_b, sm_b,
                difference=b_comp["max_abs_diff"], abs_tolerance=1e-7, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if b_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(b_comp["failure_reasons"]) if not b_comp["passed"] else None
            ),
            create_benchmark_metric(
                "standard_errors", pot_se, sm_se,
                difference=se_comp["max_abs_diff"], abs_tolerance=1e-7, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if se_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(se_comp["failure_reasons"]) if not se_comp["passed"] else None
            ),
            create_benchmark_metric(
                "covariance_matrix", r["vcov"].tolist(), sm_cov.tolist(),
                difference=cov_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if cov_diff < 1e-6 else MetricStatus.FAIL,
                failure_explanation=f"Covariance matrix discrepancy {cov_diff:.2e} exceeds 1e-6" if cov_diff >= 1e-6 else None
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "ols",
            "dataset_id": "auto.dta",
            "specification": "price ~ mpg + weight",
            "reference_implementation": "statsmodels.regression.linear_model.OLS",
            "reference_version": sm.__version__,
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": len(df),
            "sample_selection": "Complete cases (N=74)",
            "coefficient_comparison": b_comp,
            "standard_error_comparison": se_comp,
            "covariance_comparison": {"max_abs_diff": cov_diff, "passed": cov_diff < 1e-6},
            "metrics": metrics,
            "additional_diagnostics": {
                "r2_potatostics": r["r2"],
                "r2_reference": float(sm_model.rsquared),
                "r2_diff": abs(r["r2"] - float(sm_model.rsquared)),
                "df_resid": r["df_r"]
            },
            "abs_tolerance": 1e-7,
            "rel_tolerance": 1e-5,
            "validation_method": ValidationMethod.INDEPENDENT_EXTERNAL.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 2. OLS Robust (HC1)
    # -------------------------------------------------------------------------
    def bench_02_ols_robust_hc1(self):
        bench_id = "BENCH_02_OLS_ROBUST_HC1"
        df = _load_benchmark_data("auto.dta")
        r = compute_regression(df, "price", ["mpg", "weight"], robust=True, vce="hc1")
        X = sm.add_constant(df[["mpg", "weight"]])
        sm_model = sm.OLS(df["price"], X).fit(cov_type="HC1")

        aliases = {"_cons": "const"}
        pot_b = {c["name"]: c["coef"] for c in r["coefs"]}
        sm_b = sm_model.params.to_dict()
        b_comp = compare_numerical_dictionaries(pot_b, sm_b, abs_tol=1e-7, rel_tol=1e-5, key_aliases=aliases)

        pot_se = {c["name"]: c["se"] for c in r["coefs"]}
        sm_se = sm_model.bse.to_dict()
        se_comp = compare_numerical_dictionaries(pot_se, sm_se, abs_tol=1e-7, rel_tol=1e-5, key_aliases=aliases)

        pot_names = [c["name"] for c in r["coefs"]]
        mapped_names = ["const" if n == "_cons" else n for n in pot_names]
        sm_cov = sm_model.cov_params().loc[mapped_names, mapped_names].to_numpy()
        cov_diff = float(np.max(np.abs(r["vcov"] - sm_cov)))
        metrics = [
            create_benchmark_metric(
                "coefficients", pot_b, sm_b,
                difference=b_comp["max_abs_diff"], abs_tolerance=1e-7, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if b_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(b_comp["failure_reasons"]) if not b_comp["passed"] else None
            ),
            create_benchmark_metric(
                "standard_errors", pot_se, sm_se,
                difference=se_comp["max_abs_diff"], abs_tolerance=1e-7, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if se_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(se_comp["failure_reasons"]) if not se_comp["passed"] else None
            ),
            create_benchmark_metric(
                "covariance_matrix", r["vcov"].tolist(), sm_cov.tolist(),
                difference=cov_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if cov_diff < 1e-6 else MetricStatus.FAIL,
                failure_explanation=f"Covariance matrix discrepancy {cov_diff:.2e} exceeds 1e-6" if cov_diff >= 1e-6 else None
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "ols_hc1",
            "dataset_id": "auto.dta",
            "specification": "price ~ mpg + weight [vce=robust / HC1]",
            "reference_implementation": "statsmodels.regression.linear_model.OLS(cov_type='HC1')",
            "reference_version": sm.__version__,
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": len(df),
            "sample_selection": "Complete cases (N=74)",
            "coefficient_comparison": b_comp,
            "standard_error_comparison": se_comp,
            "covariance_comparison": {"max_abs_diff": cov_diff, "passed": cov_diff < 1e-6},
            "metrics": metrics,
            "additional_diagnostics": {
                "df_resid": r["df_r"]
            },
            "abs_tolerance": 1e-7,
            "rel_tolerance": 1e-5,
            "validation_method": ValidationMethod.INDEPENDENT_EXTERNAL.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 3. OLS Cluster-Robust (G / (G-1) * (N-1) / (N-K))
    # -------------------------------------------------------------------------
    def bench_03_ols_cluster(self):
        bench_id = "BENCH_03_OLS_CLUSTER"
        df = _load_benchmark_data("auto.dta").copy()
        df["cluster_id"] = (df.index % 5) + 1
        r = compute_regression(df, "price", ["mpg", "weight"], vce="cluster", cluster_var="cluster_id")
        X = sm.add_constant(df[["mpg", "weight"]])
        sm_model = sm.OLS(df["price"], X).fit(cov_type="cluster", cov_kwds={"groups": df["cluster_id"]})

        aliases = {"_cons": "const"}
        pot_b = {c["name"]: c["coef"] for c in r["coefs"]}
        sm_b = sm_model.params.to_dict()
        b_comp = compare_numerical_dictionaries(pot_b, sm_b, abs_tol=1e-7, rel_tol=1e-5, key_aliases=aliases)

        pot_se = {c["name"]: c["se"] for c in r["coefs"]}
        sm_se = sm_model.bse.to_dict()
        se_comp = compare_numerical_dictionaries(pot_se, sm_se, abs_tol=1e-6, rel_tol=1e-5, key_aliases=aliases)

        pot_names = [c["name"] for c in r["coefs"]]
        mapped_names = ["const" if n == "_cons" else n for n in pot_names]
        sm_cov = sm_model.cov_params().loc[mapped_names, mapped_names].to_numpy()
        cov_diff = float(np.max(np.abs(r["vcov"] - sm_cov)))
        metrics = [
            create_benchmark_metric(
                "coefficients", pot_b, sm_b,
                difference=b_comp["max_abs_diff"], abs_tolerance=1e-7, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if b_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(b_comp["failure_reasons"]) if not b_comp["passed"] else None
            ),
            create_benchmark_metric(
                "standard_errors", pot_se, sm_se,
                difference=se_comp["max_abs_diff"], abs_tolerance=1e-6, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if se_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(se_comp["failure_reasons"]) if not se_comp["passed"] else None
            ),
            create_benchmark_metric(
                "covariance_matrix", r["vcov"].tolist(), sm_cov.tolist(),
                difference=cov_diff, abs_tolerance=1e-5, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if cov_diff < 1e-5 else MetricStatus.FAIL,
                failure_explanation=f"Covariance matrix discrepancy {cov_diff:.2e} exceeds 1e-5" if cov_diff >= 1e-5 else None
            ),
            create_benchmark_metric(
                "cluster_count", r["n_clusters"], 5,
                difference=abs(r["n_clusters"] - 5), abs_tolerance=0,
                is_mandatory=True,
                status=MetricStatus.PASS if r["n_clusters"] == 5 else MetricStatus.FAIL
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "ols_cluster",
            "dataset_id": "auto.dta",
            "specification": "price ~ mpg + weight [cluster=cluster_id]",
            "reference_implementation": "statsmodels.regression.linear_model.OLS(cov_type='cluster')",
            "reference_version": sm.__version__,
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": len(df),
            "sample_selection": "Complete cases with 5 synthetic cluster groups (N=74, G=5)",
            "coefficient_comparison": b_comp,
            "standard_error_comparison": se_comp,
            "covariance_comparison": {"max_abs_diff": cov_diff, "passed": cov_diff < 1e-6},
            "metrics": metrics,
            "additional_diagnostics": {
                "n_clusters_potatostics": r["n_clusters"],
                "n_clusters_reference": 5,
                "df_resid_potatostics": r["df_r"],
                "covariance_correction": "G / (G - 1) * (N - 1) / (N - K)"
            },
            "abs_tolerance": 1e-6,
            "rel_tolerance": 1e-5,
            "validation_method": ValidationMethod.INDEPENDENT_EXTERNAL.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 4. Logit Binary MLE
    # -------------------------------------------------------------------------
    def bench_04_logit_mle(self):
        bench_id = "BENCH_04_LOGIT_MLE"
        df = _load_benchmark_data("auto.dta")
        r = compute_logit(df, "foreign", ["price", "mpg"])
        X = sm.add_constant(df[["price", "mpg"]])
        sm_model = sm.Logit(df["foreign"], X).fit(disp=False)

        aliases = {"_cons": "const"}
        pot_b = {c["name"]: c["coef"] for c in r["coefs"]}
        sm_b = sm_model.params.to_dict()
        b_comp = compare_numerical_dictionaries(pot_b, sm_b, abs_tol=1e-6, rel_tol=1e-5, key_aliases=aliases)

        pot_se = {c["name"]: c["se"] for c in r["coefs"]}
        sm_se = sm_model.bse.to_dict()
        se_comp = compare_numerical_dictionaries(pot_se, sm_se, abs_tol=1e-6, rel_tol=1e-5, key_aliases=aliases)

        ll_diff = abs(r["ll"] - float(sm_model.llf))
        metrics = [
            create_benchmark_metric(
                "coefficients", pot_b, sm_b,
                difference=b_comp["max_abs_diff"], abs_tolerance=1e-6, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if b_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(b_comp["failure_reasons"]) if not b_comp["passed"] else None
            ),
            create_benchmark_metric(
                "standard_errors", pot_se, sm_se,
                difference=se_comp["max_abs_diff"], abs_tolerance=1e-6, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if se_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(se_comp["failure_reasons"]) if not se_comp["passed"] else None
            ),
            create_benchmark_metric(
                "log_likelihood", r["ll"], float(sm_model.llf),
                difference=ll_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if ll_diff < 1e-6 else MetricStatus.FAIL
            ),
            create_benchmark_metric(
                "convergence", r["converged"], True,
                is_mandatory=True,
                status=MetricStatus.PASS if r["converged"] is True else MetricStatus.FAIL
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "logit",
            "dataset_id": "auto.dta",
            "specification": "foreign ~ price + mpg",
            "reference_implementation": "statsmodels.discrete.discrete_model.Logit",
            "reference_version": sm.__version__,
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": len(df),
            "sample_selection": "Complete cases (N=74)",
            "coefficient_comparison": b_comp,
            "standard_error_comparison": se_comp,
            "covariance_comparison": None,
            "metrics": metrics,
            "additional_diagnostics": {
                "log_likelihood_potatostics": r["ll"],
                "log_likelihood_reference": float(sm_model.llf),
                "ll_abs_diff": abs(r["ll"] - float(sm_model.llf)),
                "converged": r["converged"]
            },
            "abs_tolerance": 1e-6,
            "rel_tolerance": 1e-5,
            "validation_method": ValidationMethod.INDEPENDENT_EXTERNAL.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 5. Probit Binary MLE
    # -------------------------------------------------------------------------
    def bench_05_probit_mle(self):
        bench_id = "BENCH_05_PROBIT_MLE"
        df = _load_benchmark_data("auto.dta")
        r = compute_probit(df, "foreign", ["price", "mpg"])
        X = sm.add_constant(df[["price", "mpg"]])
        sm_model = sm.Probit(df["foreign"], X).fit(disp=False)

        aliases = {"_cons": "const"}
        pot_b = {c["name"]: c["coef"] for c in r["coefs"]}
        sm_b = sm_model.params.to_dict()
        b_comp = compare_numerical_dictionaries(pot_b, sm_b, abs_tol=1e-6, rel_tol=1e-5, key_aliases=aliases)

        pot_se = {c["name"]: c["se"] for c in r["coefs"]}
        sm_se = sm_model.bse.to_dict()
        se_comp = compare_numerical_dictionaries(pot_se, sm_se, abs_tol=1e-6, rel_tol=1e-5, key_aliases=aliases)

        ll_diff = abs(r["ll"] - float(sm_model.llf))
        metrics = [
            create_benchmark_metric(
                "coefficients", pot_b, sm_b,
                difference=b_comp["max_abs_diff"], abs_tolerance=1e-6, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if b_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(b_comp["failure_reasons"]) if not b_comp["passed"] else None
            ),
            create_benchmark_metric(
                "standard_errors", pot_se, sm_se,
                difference=se_comp["max_abs_diff"], abs_tolerance=1e-6, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if se_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(se_comp["failure_reasons"]) if not se_comp["passed"] else None
            ),
            create_benchmark_metric(
                "log_likelihood", r["ll"], float(sm_model.llf),
                difference=ll_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if ll_diff < 1e-6 else MetricStatus.FAIL
            ),
            create_benchmark_metric(
                "convergence", r["converged"], True,
                is_mandatory=True,
                status=MetricStatus.PASS if r["converged"] is True else MetricStatus.FAIL
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "probit",
            "dataset_id": "auto.dta",
            "specification": "foreign ~ price + mpg",
            "reference_implementation": "statsmodels.discrete.discrete_model.Probit",
            "reference_version": sm.__version__,
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": len(df),
            "sample_selection": "Complete cases (N=74)",
            "coefficient_comparison": b_comp,
            "standard_error_comparison": se_comp,
            "covariance_comparison": None,
            "metrics": metrics,
            "additional_diagnostics": {
                "log_likelihood_potatostics": r["ll"],
                "log_likelihood_reference": float(sm_model.llf),
                "ll_abs_diff": abs(r["ll"] - float(sm_model.llf)),
                "converged": r["converged"]
            },
            "abs_tolerance": 1e-6,
            "rel_tolerance": 1e-5,
            "validation_method": ValidationMethod.INDEPENDENT_EXTERNAL.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 6. Poisson Count MLE
    # -------------------------------------------------------------------------
    def bench_06_poisson_mle(self):
        bench_id = "BENCH_06_POISSON_MLE"
        df = _load_benchmark_data("auto.dta").dropna(subset=["rep78", "price", "mpg"]).copy()
        r = compute_poisson(df, "rep78", ["price", "mpg"])
        X = sm.add_constant(df[["price", "mpg"]])
        sm_model = sm.Poisson(df["rep78"], X).fit(disp=False)

        aliases = {"_cons": "const"}
        pot_b = {c["name"]: c["coef"] for c in r["coefs"]}
        sm_b = sm_model.params.to_dict()
        b_comp = compare_numerical_dictionaries(pot_b, sm_b, abs_tol=1e-6, rel_tol=1e-5, key_aliases=aliases)

        pot_se = {c["name"]: c["se"] for c in r["coefs"]}
        sm_se = sm_model.bse.to_dict()
        se_comp = compare_numerical_dictionaries(pot_se, sm_se, abs_tol=1e-6, rel_tol=1e-5, key_aliases=aliases)

        ll_diff = abs(r["ll"] - float(sm_model.llf))
        metrics = [
            create_benchmark_metric(
                "coefficients", pot_b, sm_b,
                difference=b_comp["max_abs_diff"], abs_tolerance=1e-6, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if b_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(b_comp["failure_reasons"]) if not b_comp["passed"] else None
            ),
            create_benchmark_metric(
                "standard_errors", pot_se, sm_se,
                difference=se_comp["max_abs_diff"], abs_tolerance=1e-6, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if se_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(se_comp["failure_reasons"]) if not se_comp["passed"] else None
            ),
            create_benchmark_metric(
                "log_likelihood", r["ll"], float(sm_model.llf),
                difference=ll_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if ll_diff < 1e-6 else MetricStatus.FAIL
            ),
            create_benchmark_metric(
                "convergence", r["converged"], True,
                is_mandatory=True,
                status=MetricStatus.PASS if r["converged"] is True else MetricStatus.FAIL
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "poisson",
            "dataset_id": "auto.dta",
            "specification": "rep78 ~ price + mpg",
            "reference_implementation": "statsmodels.discrete.discrete_model.Poisson",
            "reference_version": sm.__version__,
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": len(df),
            "sample_selection": "Listwise complete cases on rep78, price, mpg (N=69)",
            "coefficient_comparison": b_comp,
            "standard_error_comparison": se_comp,
            "covariance_comparison": None,
            "metrics": metrics,
            "additional_diagnostics": {
                "log_likelihood_potatostics": r["ll"],
                "log_likelihood_reference": float(sm_model.llf),
                "ll_abs_diff": abs(r["ll"] - float(sm_model.llf)),
                "converged": r["converged"]
            },
            "abs_tolerance": 1e-6,
            "rel_tolerance": 1e-5,
            "validation_method": ValidationMethod.INDEPENDENT_EXTERNAL.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 7. Quantile Regression (Median with Hall-Sheather Sparsity Standard Errors)
    # -------------------------------------------------------------------------
    def bench_07_quantile_regression(self):
        bench_id = "BENCH_07_QUANTILE_REGRESSION"
        df = _load_benchmark_data("auto.dta")
        r = compute_qreg(df, "price", ["mpg"], quantile=0.5, vce="iid")
        X = sm.add_constant(df[["mpg"]])
        sm_model = QuantReg(df["price"], X).fit(q=0.5)

        aliases = {"_cons": "const"}
        pot_b = {c["name"]: c["coef"] for c in r["coefs"]}
        sm_b = sm_model.params.to_dict()
        b_comp = compare_numerical_dictionaries(pot_b, sm_b, abs_tol=1e-5, rel_tol=1e-4, key_aliases=aliases)

        # Standard error comparison against statsmodels QuantReg Hall-Sheather Epanechnikov SEs
        pot_se = {c["name"]: c["se"] for c in r["coefs"]}
        sm_se = sm_model.bse.to_dict()
        se_comp = compare_numerical_dictionaries(pot_se, sm_se, abs_tol=1e-5, rel_tol=1e-4, key_aliases=aliases)

        # Quantile objective function value: sum rho_tau(y - X*b)
        resid = df["price"] - (r["beta"][0] * df["mpg"] + r["beta"][1])
        pot_obj = float(np.sum(resid * (0.5 - (resid < 0))))

        metrics = [
            create_benchmark_metric(
                "coefficients", pot_b, sm_b,
                difference=b_comp["max_abs_diff"], abs_tolerance=1e-5, rel_tolerance=1e-4,
                is_mandatory=True,
                status=MetricStatus.PASS if b_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(b_comp["failure_reasons"]) if not b_comp["passed"] else None
            ),
            create_benchmark_metric(
                "standard_errors", pot_se, sm_se,
                difference=se_comp["max_abs_diff"], abs_tolerance=1e-5, rel_tolerance=1e-4,
                is_mandatory=True,
                status=MetricStatus.PASS if se_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(se_comp["failure_reasons"]) if not se_comp["passed"] else None
            ),
            create_benchmark_metric(
                "objective_function_positive", pot_obj, None,
                is_mandatory=True,
                status=MetricStatus.PASS if pot_obj > 0 else MetricStatus.FAIL
            ),
            create_benchmark_metric(
                "convergence", r["converged"], True,
                is_mandatory=True,
                status=MetricStatus.PASS if r["converged"] is True else MetricStatus.FAIL
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "qreg",
            "dataset_id": "auto.dta",
            "specification": "price ~ mpg [quantile=0.5]",
            "reference_implementation": "statsmodels.regression.quantile_regression.QuantReg",
            "reference_version": sm.__version__,
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": len(df),
            "sample_selection": "Complete cases (N=74)",
            "coefficient_comparison": b_comp,
            "standard_error_comparison": se_comp,
            "covariance_comparison": None,
            "metrics": metrics,
            "additional_diagnostics": {
                "quantile": 0.5,
                "covariance_estimator": "Hall-Sheather bandwidth with Epanechnikov kernel (IID)",
                "objective_function_val": pot_obj,
                "converged": r["converged"],
                "iterations": r["iterations"]
            },
            "abs_tolerance": 1e-5,
            "rel_tolerance": 1e-4,
            "validation_method": ValidationMethod.INDEPENDENT_EXTERNAL.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 8. Two-Stage Least Squares (2SLS)
    # -------------------------------------------------------------------------
    def bench_08_ivregress_2sls(self):
        bench_id = "BENCH_08_IVREGRESS_2SLS"
        rng = np.random.RandomState(42)
        n = 500
        z = rng.normal(0, 1, n)
        u = rng.normal(0, 1, n)
        x = 1.5 * z + 0.8 * u
        e = 0.8 * u + rng.normal(0, 0.5, n)
        y = 5.0 + 2.0 * x + e
        df = pd.DataFrame({"y": y, "x": x, "z": z})

        r = compute_ivregress_2sls(df, "y", [], ["x"], ["z"])
        lm_model = LmIV2SLS(df["y"], sm.add_constant(pd.DataFrame(index=df.index)), df[["x"]], df[["z"]]).fit(cov_type="unadjusted")

        pot_b = {"x": float(r["coefs"][0]["coef"])}
        lm_b = {"x": float(lm_model.params["x"])}
        b_comp = compare_numerical_dictionaries(pot_b, lm_b, abs_tol=1e-6, rel_tol=1e-5)

        pot_se = {"x": float(r["coefs"][0]["se"])}
        lm_se = {"x": float(lm_model.std_errors["x"])}
        se_comp = compare_numerical_dictionaries(pot_se, lm_se, abs_tol=1e-6, rel_tol=1e-5)

        pot_names = r["var_names"]
        mapped_names = ["const" if n == "_cons" else n for n in pot_names]
        lm_cov = lm_model.cov.loc[mapped_names, mapped_names].to_numpy()
        cov_diff = float(np.max(np.abs(r["vcov"] - lm_cov)))

        metrics = [
            create_benchmark_metric(
                "coefficients", pot_b, lm_b,
                difference=b_comp["max_abs_diff"], abs_tolerance=1e-6, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if b_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(b_comp["failure_reasons"]) if not b_comp["passed"] else None
            ),
            create_benchmark_metric(
                "standard_errors", pot_se, lm_se,
                difference=se_comp["max_abs_diff"], abs_tolerance=1e-6, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if se_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(se_comp["failure_reasons"]) if not se_comp["passed"] else None
            ),
            create_benchmark_metric(
                "covariance_matrix", r["vcov"].tolist(), lm_cov.tolist(),
                difference=cov_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if cov_diff < 1e-6 else MetricStatus.FAIL,
                failure_explanation=f"Covariance matrix discrepancy {cov_diff:.2e} exceeds 1e-6" if cov_diff >= 1e-6 else None
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "ivregress_2sls",
            "dataset_id": "simulated_iv (DGP: y=5+2x+e, x=1.5z+u)",
            "specification": "y ~ (x = z) [2SLS]",
            "reference_implementation": "linearmodels.iv.model.IV2SLS",
            "reference_version": linearmodels.__version__,
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": n,
            "sample_selection": "Synthetic linear IV DGP (N=500)",
            "coefficient_comparison": b_comp,
            "standard_error_comparison": se_comp,
            "covariance_comparison": {"max_abs_diff": cov_diff, "passed": cov_diff < 1e-6},
            "metrics": metrics,
            "additional_diagnostics": {
                "cov_diff": cov_diff
            },
            "abs_tolerance": 1e-6,
            "rel_tolerance": 1e-5,
            "validation_method": ValidationMethod.INDEPENDENT_EXTERNAL.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 9. Panel Fixed Effects (Within)
    # -------------------------------------------------------------------------
    def bench_09_panel_fe(self):
        bench_id = "BENCH_09_PANEL_FE"
        rng = np.random.RandomState(42)
        rows = []
        for i in range(1, 51):
            alpha_i = rng.normal(0, 2)
            for t in range(1, 5):
                x = rng.normal(5, 1) + 0.5 * alpha_i
                y = 10.0 + 3.0 * x + alpha_i + rng.normal(0, 1)
                rows.append({"id": i, "t": t, "x": x, "y": y})
        df = pd.DataFrame(rows)

        r = compute_panel_fe(df, "y", ["x"], "id")
        df_p = df.set_index(["id", "t"])
        lm_fe = LmPanelOLS(df_p["y"], df_p[["x"]], entity_effects=True).fit()

        pot_b = {"x": float(r["coefs"][0]["coef"])}
        lm_b = {"x": float(lm_fe.params["x"])}
        b_comp = compare_numerical_dictionaries(pot_b, lm_b, abs_tol=1e-6, rel_tol=1e-5)

        pot_se = {"x": float(r["coefs"][0]["se"])}
        lm_se = {"x": float(lm_fe.std_errors["x"])}
        se_comp = compare_numerical_dictionaries(pot_se, lm_se, abs_tol=1e-6, rel_tol=1e-5)

        r2_diff = abs(r["r2_within"] - float(lm_fe.rsquared_within))
        metrics = [
            create_benchmark_metric(
                "coefficients", pot_b, lm_b,
                difference=b_comp["max_abs_diff"], abs_tolerance=1e-6, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if b_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(b_comp["failure_reasons"]) if not b_comp["passed"] else None
            ),
            create_benchmark_metric(
                "standard_errors", pot_se, lm_se,
                difference=se_comp["max_abs_diff"], abs_tolerance=1e-6, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if se_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(se_comp["failure_reasons"]) if not se_comp["passed"] else None
            ),
            create_benchmark_metric(
                "within_r2", r["r2_within"], float(lm_fe.rsquared_within),
                difference=r2_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if r2_diff < 1e-6 else MetricStatus.FAIL,
                failure_explanation=f"Within R2 discrepancy {r2_diff:.2e} exceeds 1e-6" if r2_diff >= 1e-6 else None
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "xtreg_fe",
            "dataset_id": "simulated_panel (50 units x 4 periods)",
            "specification": "y ~ x [fe / within]",
            "reference_implementation": "linearmodels.panel.model.PanelOLS(entity_effects=True)",
            "reference_version": linearmodels.__version__,
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": len(df),
            "sample_selection": "Balanced panel (N=200, 50 entities, 4 periods)",
            "coefficient_comparison": b_comp,
            "standard_error_comparison": se_comp,
            "covariance_comparison": None,
            "metrics": metrics,
            "additional_diagnostics": {
                "r2_within_potatostics": r["r2_within"],
                "r2_within_reference": float(lm_fe.rsquared_within),
                "r2_within_diff": r2_diff,
                "df_resid": r["df_r"]
            },
            "abs_tolerance": 1e-6,
            "rel_tolerance": 1e-5,
            "validation_method": ValidationMethod.INDEPENDENT_EXTERNAL.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 10. Hodrick-Prescott Filter (Real Execution Across Multiple Lengths & Lambdas)
    # -------------------------------------------------------------------------
    def bench_10_hp_filter(self):
        bench_id = "BENCH_10_HP_FILTER"
        rng = np.random.RandomState(42)
        test_configs = [
            (50, 100.0),    # Annual
            (100, 1600.0),  # Quarterly
            (250, 14400.0)  # Monthly
        ]

        all_trend_diffs = []
        all_cycle_diffs = []
        diagnostics = {}

        for length, lamb_val in test_configs:
            t = np.linspace(0, 10, length)
            trend_true = 50 + 2 * t
            cycle_true = 5 * np.sin(t)
            series = trend_true + cycle_true + rng.normal(0, 1, length)

            # Execute real Potatostics HP filter routine
            pot_cycle, pot_trend = compute_hp_filter(series, lamb=lamb_val)

            # Statsmodels reference implementation
            sm_cycle, sm_trend = hpfilter(series, lamb=lamb_val)

            t_diff = float(np.max(np.abs(pot_trend - sm_trend)))
            c_diff = float(np.max(np.abs(pot_cycle - sm_cycle)))

            all_trend_diffs.append(t_diff)
            all_cycle_diffs.append(c_diff)
            diagnostics[f"N{length}_lambda{int(lamb_val)}"] = {
                "max_trend_diff": t_diff,
                "max_cycle_diff": c_diff
            }

        max_t_diff = max(all_trend_diffs)
        max_c_diff = max(all_cycle_diffs)
        overall_diff = max(max_t_diff, max_c_diff)
        passed = overall_diff < 1e-8

        metrics = [
            create_benchmark_metric(
                "trend_and_cycle_components",
                {"max_trend_diff": max_t_diff, "max_cycle_diff": max_c_diff},
                {"max_trend_diff": 0.0, "max_cycle_diff": 0.0},
                difference=overall_diff, abs_tolerance=1e-8, rel_tolerance=1e-7,
                is_mandatory=True,
                status=MetricStatus.PASS if passed else MetricStatus.FAIL,
                failure_explanation=f"HP filter component discrepancy {overall_diff:.2e} exceeds 1e-8" if not passed else None
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "tsfilter_hp",
            "dataset_id": "simulated_macro_cycles (lengths 50, 100, 250)",
            "specification": "tsfilter hp cycle = y, smooth(100, 1600, 14400)",
            "reference_implementation": "statsmodels.tsa.filters.hp_filter.hpfilter",
            "reference_version": sm.__version__,
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": 400,
            "sample_selection": "Combined multi-length battery (50 + 100 + 250 = 400 obs)",
            "coefficient_comparison": {
                "matched_keys": ["trend_component", "cyclical_component"],
                "missing_in_potatostics": [],
                "missing_in_reference": [],
                "potatostics_values": {"max_diff": overall_diff},
                "reference_values": {"max_diff": 0.0},
                "absolute_differences": {"trend": max_t_diff, "cycle": max_c_diff},
                "relative_differences": {"trend": max_t_diff / 50.0, "cycle": max_c_diff / 5.0},
                "max_abs_diff": overall_diff,
                "max_rel_diff": overall_diff / 5.0,
                "abs_tolerance": 1e-8,
                "rel_tolerance": 1e-7,
                "passed": passed
            },
            "standard_error_comparison": None,  # Not applicable to deterministic filtering
            "covariance_comparison": None,
            "metrics": metrics,
            "additional_diagnostics": diagnostics,
            "abs_tolerance": 1e-8,
            "rel_tolerance": 1e-7,
            "validation_method": ValidationMethod.INTERNAL_CONSISTENCY.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 11. Vector Autoregression VAR(1) (Full System Validation)
    # -------------------------------------------------------------------------
    def bench_11_var(self):
        bench_id = "BENCH_11_VAR"
        rng = np.random.RandomState(42)
        n = 150
        y1 = np.zeros(n)
        y2 = np.zeros(n)
        for i in range(1, n):
            y1[i] = 0.5 * y1[i - 1] + 0.2 * y2[i - 1] + rng.normal(0, 1)
            y2[i] = 0.1 * y1[i - 1] + 0.4 * y2[i - 1] + rng.normal(0, 1)
        df = pd.DataFrame({"y1": y1, "y2": y2})

        r = compute_var(df, ["y1", "y2"], lags=1)
        sm_var = SmVAR(df).fit(maxlags=1, trend="c")

        # Map all coefficients across all equations
        pot_b = r["b_dict"]
        sm_b = {
            "y1:_cons": float(sm_var.params.loc["const", "y1"]),
            "y1:L1.y1": float(sm_var.params.loc["L1.y1", "y1"]),
            "y1:L1.y2": float(sm_var.params.loc["L1.y2", "y1"]),
            "y2:_cons": float(sm_var.params.loc["const", "y2"]),
            "y2:L1.y1": float(sm_var.params.loc["L1.y1", "y2"]),
            "y2:L1.y2": float(sm_var.params.loc["L1.y2", "y2"])
        }
        b_comp = compare_numerical_dictionaries(pot_b, sm_b, abs_tol=1e-7, rel_tol=1e-5)

        # Standard errors across all equations
        pot_se = r["se_dict"]
        sm_se = {
            "y1:_cons": float(sm_var.stderr.loc["const", "y1"]),
            "y1:L1.y1": float(sm_var.stderr.loc["L1.y1", "y1"]),
            "y1:L1.y2": float(sm_var.stderr.loc["L1.y2", "y1"]),
            "y2:_cons": float(sm_var.stderr.loc["const", "y2"]),
            "y2:L1.y1": float(sm_var.stderr.loc["L1.y1", "y2"]),
            "y2:L1.y2": float(sm_var.stderr.loc["L1.y2", "y2"])
        }
        se_comp = compare_numerical_dictionaries(pot_se, sm_se, abs_tol=1e-7, rel_tol=1e-5)

        aic_diff = abs(r["aic"] - float(sm_var.aic))
        bic_diff = abs(r["bic"] - float(sm_var.bic))
        ll_diff = abs(r["ll"] - float(sm_var.llf))
        cov_diff = float(np.max(np.abs(r["vcov"] - sm_var.cov_params().to_numpy())))
        sigma_diff = float(np.max(np.abs(r["resid_cov"] - sm_var.sigma_u.to_numpy())))

        metrics = [
            create_benchmark_metric(
                "coefficients", pot_b, sm_b,
                difference=b_comp["max_abs_diff"], abs_tolerance=1e-7, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if b_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(b_comp["failure_reasons"]) if not b_comp["passed"] else None
            ),
            create_benchmark_metric(
                "standard_errors", pot_se, sm_se,
                difference=se_comp["max_abs_diff"], abs_tolerance=1e-7, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if se_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(se_comp["failure_reasons"]) if not se_comp["passed"] else None
            ),
            create_benchmark_metric(
                "covariance_matrix", r["vcov"].tolist(), sm_var.cov_params().to_numpy().tolist(),
                difference=cov_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if cov_diff < 1e-6 else MetricStatus.FAIL,
                failure_explanation=f"Covariance matrix discrepancy {cov_diff:.2e} exceeds 1e-6" if cov_diff >= 1e-6 else None
            ),
            create_benchmark_metric(
                "residual_covariance_sigma_u", r["resid_cov"].tolist(), sm_var.sigma_u.to_numpy().tolist(),
                difference=sigma_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if sigma_diff < 1e-6 else MetricStatus.FAIL,
                failure_explanation=f"Residual covariance discrepancy {sigma_diff:.2e} exceeds 1e-6" if sigma_diff >= 1e-6 else None
            ),
            create_benchmark_metric(
                "aic", r["aic"], float(sm_var.aic),
                difference=aic_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if aic_diff < 1e-6 else MetricStatus.FAIL
            ),
            create_benchmark_metric(
                "bic", r["bic"], float(sm_var.bic),
                difference=bic_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if bic_diff < 1e-6 else MetricStatus.FAIL
            ),
            create_benchmark_metric(
                "log_likelihood", r["ll"], float(sm_var.llf),
                difference=ll_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if ll_diff < 1e-6 else MetricStatus.FAIL
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "var",
            "dataset_id": "simulated_var (2 series x 150 obs)",
            "specification": "var y1 y2, lags(1)",
            "reference_implementation": "statsmodels.tsa.api.VAR",
            "reference_version": sm.__version__,
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": n,
            "sample_selection": "Bivariate VAR(1) with constant (N=150, effective N=149)",
            "coefficient_comparison": b_comp,
            "standard_error_comparison": se_comp,
            "covariance_comparison": {"max_abs_diff": cov_diff, "passed": cov_diff < 1e-6},
            "metrics": metrics,
            "additional_diagnostics": {
                "aic_diff": aic_diff,
                "bic_diff": bic_diff,
                "log_likelihood_diff": ll_diff,
                "resid_cov_diff": sigma_diff,
                "lags": 1,
                "n_equations": 2
            },
            "abs_tolerance": 1e-7,
            "rel_tolerance": 1e-5,
            "validation_method": ValidationMethod.INTERNAL_CONSISTENCY.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 12. Kaplan-Meier Survival Function (Analytical Greenwood Parity)
    # -------------------------------------------------------------------------
    def bench_12_kaplan_meier(self):
        bench_id = "BENCH_12_KAPLAN_MEIER"
        # 10 observations with survival times and right-censoring
        t_arr = np.array([2, 3, 5, 5, 6, 8, 9, 10, 12, 15], dtype=float)
        d_arr = np.array([1, 0, 1, 1, 0, 1, 0,  1,  1,  0], dtype=float)
        df = pd.DataFrame({"time": t_arr, "died": d_arr})

        r = compute_sts_list(df, "time", "died")

        # Independent Analytical Ground Truth derived from product-limit definition:
        # Event time 2:  n=10, d=1, c=1 (at t=3). S=9/10=0.900, Greenwood SE = 0.9*sqrt(1/(10*9)) = 0.094868
        # Event time 5:  n=8,  d=2, c=1 (at t=6). S=0.9*(6/8)=0.675, Greenwood SE = 0.675*sqrt(1/90 + 2/(8*6)) = 0.154465
        # Event time 8:  n=5,  d=1, c=1 (at t=9). S=0.675*(4/5)=0.540, Greenwood SE = 0.54*sqrt(1/90 + 2/48 + 1/(5*4)) = 0.170669
        # Event time 10: n=3,  d=1, c=0.         S=0.540*(2/3)=0.360, Greenwood SE = 0.36*sqrt(1/90 + 2/48 + 1/20 + 1/(3*2)) = 0.187315
        # Event time 12: n=2,  d=1, c=1 (at t=15). S=0.360*(1/2)=0.180, Greenwood SE = 0.18*sqrt(1/90 + 2/48 + 1/20 + 1/6 + 1/(2*1)) = 0.160137
        analytical_table = {
            2.0:  {"at_risk": 10, "fail": 1, "lost": 1, "surv": 0.900, "se": 0.900 * np.sqrt(1.0 / (10 * 9))},
            5.0:  {"at_risk": 8,  "fail": 2, "lost": 1, "surv": 0.675, "se": 0.675 * np.sqrt(1.0 / 90 + 2.0 / (8 * 6))},
            8.0:  {"at_risk": 5,  "fail": 1, "lost": 1, "surv": 0.540, "se": 0.540 * np.sqrt(1.0 / 90 + 2.0 / 48 + 1.0 / (5 * 4))},
            10.0: {"at_risk": 3,  "fail": 1, "lost": 0, "surv": 0.360, "se": 0.360 * np.sqrt(1.0 / 90 + 2.0 / 48 + 1.0 / 20 + 1.0 / (3 * 2))},
            12.0: {"at_risk": 2,  "fail": 1, "lost": 1, "surv": 0.180, "se": 0.180 * np.sqrt(1.0 / 90 + 2.0 / 48 + 1.0 / 20 + 1.0 / 6 + 1.0 / (2 * 1))}
        }

        pot_surv_dict = {}
        pot_se_dict = {}
        ref_surv_dict = {}
        ref_se_dict = {}
        risk_diffs = []

        for row in r["table"]:
            t_val = row["time"]
            if t_val in analytical_table:
                pot_surv_dict[f"S(t={int(t_val)})"] = row["survivor"]
                pot_se_dict[f"SE(t={int(t_val)})"] = row["std_err"]
                ref_surv_dict[f"S(t={int(t_val)})"] = analytical_table[t_val]["surv"]
                ref_se_dict[f"SE(t={int(t_val)})"] = analytical_table[t_val]["se"]
                risk_diffs.append(abs(row["beg_total"] - analytical_table[t_val]["at_risk"]))
                risk_diffs.append(abs(row["fail"] - analytical_table[t_val]["fail"]))
                risk_diffs.append(abs(row["net_lost"] - analytical_table[t_val]["lost"]))

        b_comp = compare_numerical_dictionaries(pot_surv_dict, ref_surv_dict, abs_tol=1e-10, rel_tol=1e-8)
        se_comp = compare_numerical_dictionaries(pot_se_dict, ref_se_dict, abs_tol=1e-5, rel_tol=1e-4)

        metrics = [
            create_benchmark_metric(
                "survival_function", pot_surv_dict, ref_surv_dict,
                difference=b_comp["max_abs_diff"], abs_tolerance=1e-10, rel_tolerance=1e-8,
                is_mandatory=True,
                status=MetricStatus.PASS if b_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(b_comp["failure_reasons"]) if not b_comp["passed"] else None
            ),
            create_benchmark_metric(
                "greenwood_standard_errors", pot_se_dict, ref_se_dict,
                difference=se_comp["max_abs_diff"], abs_tolerance=1e-5, rel_tolerance=1e-4,
                is_mandatory=True,
                status=MetricStatus.PASS if se_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(se_comp["failure_reasons"]) if not se_comp["passed"] else None
            ),
            create_benchmark_metric(
                "risk_set_counts", {"max_diff": max(risk_diffs)}, {"max_diff": 0},
                difference=float(max(risk_diffs)), abs_tolerance=0.0,
                is_mandatory=True,
                status=MetricStatus.PASS if max(risk_diffs) == 0 else MetricStatus.FAIL
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "sts_list",
            "dataset_id": "analytical_survival (10 observations, 6 events, 4 censored)",
            "specification": "sts list, time(time) failure(died)",
            "reference_implementation": "Kaplan-Meier Product-Limit & Greenwood Formula",
            "reference_version": "Analytical Identity",
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": 10,
            "sample_selection": "Complete life table across all 5 distinct death event times",
            "coefficient_comparison": b_comp,
            "standard_error_comparison": se_comp,
            "covariance_comparison": None,
            "metrics": metrics,
            "additional_diagnostics": {
                "risk_set_match": bool(max(risk_diffs) == 0),
                "total_events": r["n_events"],
                "total_observations": r["n_obs"]
            },
            "abs_tolerance": 1e-10,
            "rel_tolerance": 1e-8,
            "validation_method": ValidationMethod.ANALYTICAL_GROUND_TRUTH.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 13. Goodman-Bacon (2021) DiD Decomposition (Exact Theorem Parity)
    # -------------------------------------------------------------------------
    def bench_13_bacondecomp(self):
        bench_id = "BENCH_13_BACONDECOMP"
        # Analytical 3-group staggered adoption panel: 12 units x 4 periods
        rows = []
        for uid in range(1, 13):
            k = 2 if 1 <= uid <= 4 else (3 if 5 <= uid <= 8 else None)
            for tid in [1, 2, 3, 4]:
                treated = 1 if (k is not None and tid >= k) else 0
                y = 10.0 + (2.0 if k == 2 else 4.0) * treated + 0.5 * tid + 0.2 * uid
                rows.append({"id": uid, "t": tid, "d": treated, "y": y})
        df = pd.DataFrame(rows)

        r = compute_bacondecomp(df, "y", "d", "id", "t")

        pot_comps = {}
        for c in r["comparisons"]:
            key = f"{c['comparison_type']}_{c['group_k']}_vs_{c['group_control']}"
            pot_comps[key] = {"estimate": float(c["estimate"]), "weight": float(c["weight"])}

        ref_comps = {
            "Treated vs Never Treated_2_vs_Never Treated": {"estimate": 2.0, "weight": 0.3},
            "Treated vs Never Treated_3_vs_Never Treated": {"estimate": 4.0, "weight": 0.4},
            "Earlier vs Later Treated_Early (2)_vs_Late (3)": {"estimate": 2.0, "weight": 0.1},
            "Later vs Earlier Treated_Late (3)_vs_Early (2)": {"estimate": 4.0, "weight": 0.2}
        }

        est_diffs = [abs(pot_comps[k]["estimate"] - ref_comps[k]["estimate"]) for k in ref_comps if k in pot_comps]
        wt_diffs = [abs(pot_comps[k]["weight"] - ref_comps[k]["weight"]) for k in ref_comps if k in pot_comps]
        max_est_diff = max(est_diffs) if est_diffs else 999.0
        max_wt_diff = max(wt_diffs) if wt_diffs else 999.0

        twfe_val = r["twfe"]
        weighted_val = r["weighted_bacon"]
        decomp_diff = abs(twfe_val - weighted_val)
        w_sum = r["weights_sum"]
        w_norm_err = abs(w_sum - 1.0)

        metrics = [
            create_benchmark_metric(
                "individual_2x2_estimates",
                {k: pot_comps[k]["estimate"] for k in pot_comps},
                {k: ref_comps[k]["estimate"] for k in ref_comps},
                difference=max_est_diff, abs_tolerance=1e-8,
                is_mandatory=True,
                status=MetricStatus.PASS if max_est_diff < 1e-8 else MetricStatus.FAIL
            ),
            create_benchmark_metric(
                "decomposition_weights",
                {k: pot_comps[k]["weight"] for k in pot_comps},
                {k: ref_comps[k]["weight"] for k in ref_comps},
                difference=max_wt_diff, abs_tolerance=1e-8,
                is_mandatory=True,
                status=MetricStatus.PASS if max_wt_diff < 1e-8 else MetricStatus.FAIL
            ),
            create_benchmark_metric(
                "weights_sum_to_one", w_sum, 1.0,
                difference=w_norm_err, abs_tolerance=1e-8,
                is_mandatory=True,
                status=MetricStatus.PASS if w_norm_err < 1e-8 else MetricStatus.FAIL
            ),
            create_benchmark_metric(
                "weighted_sum_matches_twfe", weighted_val, twfe_val,
                difference=decomp_diff, abs_tolerance=1e-8,
                is_mandatory=True,
                status=MetricStatus.PASS if decomp_diff < 1e-8 else MetricStatus.FAIL
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "bacondecomp",
            "dataset_id": "analytical_staggered_timing (12 units x 4 periods)",
            "specification": "bacondecomp y d, ivar(id) time(t)",
            "reference_implementation": "Goodman-Bacon (2021) Decomposition Theorem",
            "reference_version": "Analytical Theorem Identity",
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": len(df),
            "sample_selection": "Staggered panel: early cohort (t=2), late cohort (t=3), never-treated",
            "coefficient_comparison": {
                "matched_keys": ["TWFE_vs_Bacon_Weighted_Sum"],
                "missing_in_potatostics": [],
                "missing_in_reference": [],
                "potatostics_values": {"weighted_bacon": weighted_val},
                "reference_values": {"twfe_estimate": twfe_val},
                "absolute_differences": {"TWFE_vs_Bacon_Weighted_Sum": decomp_diff},
                "relative_differences": {"TWFE_vs_Bacon_Weighted_Sum": decomp_diff / max(abs(twfe_val), 1e-12)},
                "max_abs_diff": decomp_diff,
                "max_rel_diff": decomp_diff / max(abs(twfe_val), 1e-12),
                "abs_tolerance": 1e-8,
                "rel_tolerance": 1e-7,
                "passed": decomp_diff < 1e-8
            },
            "standard_error_comparison": None,  # Not applicable (decomposition weights are non-stochastic weights)
            "covariance_comparison": None,
            "metrics": metrics,
            "additional_diagnostics": {
                "twfe_coefficient": twfe_val,
                "weighted_bacon_coefficient": weighted_val,
                "weights_sum": w_sum,
                "weight_normalization_error": w_norm_err,
                "max_est_diff": max_est_diff,
                "max_wt_diff": max_wt_diff,
                "n_comparisons": len(r["comparisons"]),
                "comparison_subgroups": [c["comparison_type"] for c in r["summary"]]
            },
            "abs_tolerance": 1e-8,
            "rel_tolerance": 1e-7,
            "validation_method": ValidationMethod.ANALYTICAL_GROUND_TRUTH.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 14. Doubly Robust Difference-in-Differences (Sant'Anna & Zhao 2020 Multi-Scenario Battery)
    # -------------------------------------------------------------------------
    def bench_14_drdid(self):
        bench_id = "BENCH_14_DRDID"
        # Sant'Anna & Zhao (2020) Data Generating Process with known ATT = 3.0
        # 400 units observed in 2 periods (balanced panel)
        rng = np.random.RandomState(42)
        n_units = 400
        rows = []
        for uid in range(1, n_units + 1):
            x = float(rng.normal(1.0, 0.5))
            # True propensity score model (logistic)
            ps_true = 1.0 / (1.0 + np.exp(-(-0.5 + 0.8 * x)))
            is_treated = 1 if float(rng.uniform(0, 1)) < ps_true else 0
            if uid <= 50:
                is_treated = 1
            elif uid <= 100:
                is_treated = 0

            for tid in [0, 1]:
                d_eff = is_treated if tid == 1 else 0
                # True outcome model (linear in x, true treatment effect = 3.0)
                y = 5.0 + 3.0 * d_eff + 1.5 * x + 0.8 * tid + float(rng.normal(0, 0.2))
                rows.append({"id": uid, "t": tid, "treated": is_treated, "y": y, "x": x})
        df = pd.DataFrame(rows)

        # 1. Primary Panel DRDID
        r = compute_drdid(df, "y", "treated", "t", covars=["x"], unitvar="id")
        att_est = r["att"]
        se_est = r["se"]
        att_diff = abs(att_est - 3.0)

        # 2. Repeated Cross-Sections Variant with genuine independent cross-sections
        rng_rc = np.random.RandomState(123)
        rc_rows = []
        for i in range(250):
            x_i = float(rng_rc.normal(1.0, 0.5))
            ps_i = 1.0 / (1.0 + np.exp(-(-0.5 + 0.8 * x_i)))
            d_i = 1 if rng_rc.uniform(0, 1) < ps_i else 0
            y_i = 5.0 + 1.5 * x_i + float(rng_rc.normal(0, 0.2))
            rc_rows.append({"id": f"t0_{i}", "t": 0, "treated": d_i, "x": x_i, "y": y_i})
        for i in range(250):
            x_i = float(rng_rc.normal(1.0, 0.5))
            ps_i = 1.0 / (1.0 + np.exp(-(-0.5 + 0.8 * x_i)))
            d_i = 1 if rng_rc.uniform(0, 1) < ps_i else 0
            y_i = 5.0 + 3.0 * d_i + 1.5 * x_i + 0.8 + float(rng_rc.normal(0, 0.2))
            rc_rows.append({"id": f"t1_{i}", "t": 1, "treated": d_i, "x": x_i, "y": y_i})
        df_rc = pd.DataFrame(rc_rows)
        r_rc = compute_drdid(df_rc, "y", "treated", "t", covars=["x"], unitvar=None)
        rc_att_diff = abs(r_rc["att"] - 3.0)

        # 3. Double Robustness Property Verification:
        # A. Misspecified outcome model (fit with quadratic term omitted; time-varying non-linear trend)
        df_miss_out = df.copy()
        df_miss_out["y"] = df_miss_out["y"] + 0.8 * df_miss_out["t"] * (df_miss_out["x"] ** 2)
        r_miss_out = compute_drdid(df_miss_out, "y", "treated", "t", covars=["x"], unitvar="id")
        miss_out_diff = abs(r_miss_out["att"] - 3.0)

        # Without covariates, the ATT reduces to a contrast of four independent
        # cell means. The treated-group level shift cancels in the contrast.
        rc_cells = pd.DataFrame([
            (d, t, 1 + t + 3 * d * t + 100 * d + error)
            for d in (0, 1) for t in (0, 1) for error in (-2, -1, 0, 1, 2)
        ], columns=["treated", "t", "y"])
        rc_cell_stats = rc_cells.groupby(["treated", "t"])["y"]
        rc_reference_se = np.sqrt((rc_cell_stats.var(ddof=0) / rc_cell_stats.size()).sum())
        rc_cells_estimate = compute_drdid(rc_cells, "y", "treated", "t")

        metrics = [
            create_benchmark_metric(
                "att_estimate_panel", att_est, 3.0,
                difference=att_diff, abs_tolerance=0.15, rel_tolerance=0.05,
                is_mandatory=True,
                status=MetricStatus.PASS if att_diff < 0.15 else MetricStatus.FAIL,
                failure_explanation=f"Panel DRDID ATT difference {att_diff:.4f} exceeds 0.15" if att_diff >= 0.15 else None
            ),
            create_benchmark_metric(
                "att_estimate_repeated_cross_section", r_rc["att"], 3.0,
                difference=rc_att_diff, abs_tolerance=0.20, rel_tolerance=0.07,
                is_mandatory=True,
                status=MetricStatus.PASS if rc_att_diff < 0.20 else MetricStatus.FAIL,
                failure_explanation=f"RC DRDID ATT difference {rc_att_diff:.4f} exceeds 0.20" if rc_att_diff >= 0.20 else None
            ),
            create_benchmark_metric(
                "double_robustness_outcome_misspecification", r_miss_out["att"], 3.0,
                difference=miss_out_diff, abs_tolerance=0.20, rel_tolerance=0.07,
                is_mandatory=True,
                status=MetricStatus.PASS if miss_out_diff < 0.20 else MetricStatus.FAIL,
                failure_explanation=f"Double robust ATT diff {miss_out_diff:.4f} exceeds 0.20" if miss_out_diff >= 0.20 else None
            ),
            create_benchmark_metric(
                "repeated_cross_section_four_cell_se", rc_cells_estimate["se"], rc_reference_se,
                abs_tolerance=1e-12, rel_tolerance=1e-12,
                evaluation_method="Analytical variance of four independent cell means"
            ),
            create_benchmark_metric(
                "standard_errors", None, None,
                is_mandatory=False,
                status=MetricStatus.NOT_TESTED,
                metric_type=MetricType.NOT_TESTED_RECORD,
                failure_explanation="No external SE comparison for the covariate-adjusted DRDiD specifications"
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "drdid",
            "dataset_id": "santanna_zhao_dgp (400 units x 2 periods, True ATT=3.0)",
            "specification": "drdid y treated t x, ivar(id)",
            "reference_implementation": "Sant'Anna & Zhao (2020) Doubly Robust DiD (Panel & RC)",
            "reference_version": "Mathematical Ground Truth ATT=3.0",
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": len(df),
            "sample_selection": "Simulated panel (N=800 obs, 400 units, 2 periods)",
            "coefficient_comparison": {
                "matched_keys": ["ATT"],
                "missing_in_potatostics": [],
                "missing_in_reference": [],
                "potatostics_values": {"ATT": att_est},
                "reference_values": {"ATT": 3.0},
                "absolute_differences": {"ATT": att_diff},
                "relative_differences": {"ATT": att_diff / 3.0},
                "max_abs_diff": att_diff,
                "max_rel_diff": att_diff / 3.0,
                "abs_tolerance": 0.15,
                "rel_tolerance": 0.05,
                "passed": att_diff < 0.15
            },
            "standard_error_comparison": None,  # Not tested against external standard; reported as None
            "covariance_comparison": None,
            "metrics": metrics,
            "additional_diagnostics": {
                "true_att": 3.0,
                "panel_att_estimate": att_est,
                "panel_std_err": se_est,
                "panel_z_stat": r["z"],
                "repeated_cross_section_att": r_rc["att"],
                "repeated_cross_section_diff": rc_att_diff,
                "misspecified_outcome_att": r_miss_out["att"],
                "misspecified_outcome_diff": miss_out_diff,
                "n_treated_units": r["n_treated"],
                "n_control_units": r["n_control"],
                "data_type": "panel"
            },
            "abs_tolerance": 0.15,
            "rel_tolerance": 0.05,
            "validation_method": ValidationMethod.SIMULATION_GROUND_TRUTH.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 15. Hodrick-Prescott Filter (Analytical Closed-Form Ground Truth)
    # -------------------------------------------------------------------------
    def bench_15_hp_filter_analytical(self):
        bench_id = "BENCH_15_HP_FILTER_ANALYTICAL"
        rng = np.random.RandomState(42)
        n = 120
        t = np.linspace(0, 10, n)
        y = 30 + 1.5 * t + 4 * np.sin(t) + rng.normal(0, 1, n)

        diffs = {}
        for lamb in [100.0, 1600.0, 14400.0]:
            pot_cycle, pot_trend = compute_hp_filter(y, lamb=lamb)

            # Analytical solve: (I + lambda K'K) trend = y
            I = np.eye(n)
            K = np.zeros((n - 2, n))
            for i in range(n - 2):
                K[i, i] = 1.0
                K[i, i + 1] = -2.0
                K[i, i + 2] = 1.0

            A = I + lamb * (K.T @ K)
            analytical_trend = scipy.linalg.solve(A, y, assume_a="pos")
            analytical_cycle = y - analytical_trend

            t_diff = float(np.max(np.abs(pot_trend - analytical_trend)))
            c_diff = float(np.max(np.abs(pot_cycle - analytical_cycle)))
            diffs[f"lambda_{int(lamb)}"] = {"trend_diff": t_diff, "cycle_diff": c_diff}

        max_abs = max(max(d["trend_diff"], d["cycle_diff"]) for d in diffs.values())
        # Scale-aware comparison: condition number kappa(I + lambda K'K) approx 2.3e5 for lambda=14400, N=120
        passed = max_abs < 1e-8

        metrics = [
            create_benchmark_metric(
                "trend_and_cycle_components",
                {"max_trend_diff": max_abs, "max_cycle_diff": max_abs},
                {"max_trend_diff": 0.0, "max_cycle_diff": 0.0},
                difference=max_abs, abs_tolerance=1e-8, rel_tolerance=1e-7,
                is_mandatory=True,
                status=MetricStatus.PASS if passed else MetricStatus.FAIL,
                failure_explanation=f"HP filter analytical solve discrepancy {max_abs:.2e} exceeds 1e-8" if not passed else None
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "tsfilter_hp",
            "dataset_id": "simulated_hp_series (120 periods)",
            "specification": "tsfilter hp cycle = y, smooth(100, 1600, 14400) [Analytical Ground Truth]",
            "reference_implementation": "Closed-Form Central Difference Solve: (I + lambda K'K)^(-1) y",
            "reference_version": "Analytical Ground Truth",
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": n,
            "sample_selection": "120 observations across 3 penalty parameter configurations",
            "coefficient_comparison": {
                "matched_keys": ["trend_component", "cyclical_component"],
                "missing_in_potatostics": [],
                "missing_in_reference": [],
                "potatostics_values": {"max_diff": max_abs},
                "reference_values": {"max_diff": 0.0},
                "absolute_differences": {"trend": max_abs, "cycle": max_abs},
                "relative_differences": {"trend": max_abs / 30.0, "cycle": max_abs / 4.0},
                "max_abs_diff": max_abs,
                "max_rel_diff": max_abs / 4.0,
                "abs_tolerance": 1e-8,
                "rel_tolerance": 1e-7,
                "passed": passed
            },
            "standard_error_comparison": None,
            "covariance_comparison": None,
            "metrics": metrics,
            "additional_diagnostics": diffs,
            "abs_tolerance": 1e-8,
            "rel_tolerance": 1e-7,
            "validation_method": ValidationMethod.ANALYTICAL_GROUND_TRUTH.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 16. Vector Autoregression VAR(1) (Analytical OLS Ground Truth)
    # -------------------------------------------------------------------------
    def bench_16_var_analytical_ols(self):
        bench_id = "BENCH_16_VAR_ANALYTICAL_OLS"
        rng = np.random.RandomState(42)
        n = 150
        y1 = np.zeros(n)
        y2 = np.zeros(n)
        for i in range(1, n):
            y1[i] = 0.5 * y1[i - 1] + 0.2 * y2[i - 1] + rng.normal(0, 1)
            y2[i] = 0.1 * y1[i - 1] + 0.4 * y2[i - 1] + rng.normal(0, 1)
        df = pd.DataFrame({"y1": y1, "y2": y2})

        r = compute_var(df, ["y1", "y2"], lags=1)

        # Direct equation-by-equation OLS: B = (X'X)^(-1) X'Y
        Y_eff = df.iloc[1:][["y1", "y2"]].to_numpy()
        X_lags = df.iloc[:-1][["y1", "y2"]].to_numpy()
        X_mat = np.column_stack([np.ones(len(X_lags)), X_lags])
        B_mat = np.linalg.inv(X_mat.T @ X_mat) @ (X_mat.T @ Y_eff)

        pot_b = r["b_dict"]
        ref_b = {
            "y1:_cons": float(B_mat[0, 0]),
            "y1:L1.y1": float(B_mat[1, 0]),
            "y1:L1.y2": float(B_mat[2, 0]),
            "y2:_cons": float(B_mat[0, 1]),
            "y2:L1.y1": float(B_mat[1, 1]),
            "y2:L1.y2": float(B_mat[2, 1])
        }
        b_comp = compare_numerical_dictionaries(pot_b, ref_b, abs_tol=1e-9, rel_tol=1e-7)

        metrics = [
            create_benchmark_metric(
                "coefficients", pot_b, ref_b,
                difference=b_comp["max_abs_diff"], abs_tolerance=1e-9, rel_tolerance=1e-7,
                is_mandatory=True,
                status=MetricStatus.PASS if b_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(b_comp["failure_reasons"]) if not b_comp["passed"] else None
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "var",
            "dataset_id": "simulated_var (2 series x 150 obs)",
            "specification": "var y1 y2, lags(1) [Analytical Ground Truth]",
            "reference_implementation": "Equation-by-Equation OLS: B = (X'X)^(-1) X'Y",
            "reference_version": "Analytical Ground Truth",
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": n,
            "sample_selection": "Bivariate VAR(1) with constant (effective N=149)",
            "coefficient_comparison": b_comp,
            "standard_error_comparison": None,
            "covariance_comparison": None,
            "metrics": metrics,
            "additional_diagnostics": {
                "lags": 1,
                "n_equations": 2
            },
            "abs_tolerance": 1e-9,
            "rel_tolerance": 1e-7,
            "validation_method": ValidationMethod.ANALYTICAL_GROUND_TRUTH.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 17. Cox Proportional Hazards (Analytical Breslow Optimization Ground Truth)
    # -------------------------------------------------------------------------
    def bench_17_cox_breslow_analytical(self):
        bench_id = "BENCH_17_COX_BRESLOW_ANALYTICAL"
        rng = np.random.RandomState(42)
        n = 60
        x = rng.normal(0, 1, n)
        haz = np.exp(0.8 * x)
        t_event = rng.exponential(1.0 / haz)
        t_censor = rng.uniform(0.5, 3.0, n)
        t_obs = np.minimum(t_event, t_censor)
        failed = (t_event <= t_censor).astype(int)
        df = pd.DataFrame({"time": t_obs, "failed": failed, "x": x})

        r = compute_stcox(df, "time", "failed", ["x"], method="breslow")
        pot_b = {"x": float(r["coefs"][0]["coef"])}

        # Direct numerical optimization of Breslow partial log-likelihood:
        order = np.argsort(t_obs)
        t_s = t_obs[order]
        d_s = failed[order]
        x_s = x[order]
        unq_d = np.sort(np.unique(t_s[d_s == 1]))

        def neg_loglik(beta):
            b = beta[0]
            exp_xb = np.exp(x_s * b)
            ll = 0.0
            for death_t in unq_d:
                mask = (t_s == death_t) & (d_s == 1)
                d_k = np.sum(mask)
                s_k = np.sum(x_s[mask])
                sum_risk = np.sum(exp_xb[t_s >= death_t])
                ll += s_k * b - d_k * np.log(sum_risk)
            return -ll

        opt = scipy.optimize.minimize(neg_loglik, [0.0], method="BFGS")
        ref_b = {"x": float(opt.x[0])}

        b_comp = compare_numerical_dictionaries(pot_b, ref_b, abs_tol=1e-5, rel_tol=1e-4)

        metrics = [
            create_benchmark_metric(
                "coefficients", pot_b, ref_b,
                difference=b_comp["max_abs_diff"], abs_tolerance=1e-5, rel_tolerance=1e-4,
                is_mandatory=True,
                status=MetricStatus.PASS if b_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(b_comp["failure_reasons"]) if not b_comp["passed"] else None
            ),
            create_benchmark_metric(
                "optimizer_convergence", opt.success, True,
                is_mandatory=True,
                status=MetricStatus.PASS if opt.success else MetricStatus.FAIL
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "stcox",
            "dataset_id": "simulated_survival (60 obs, right-censored)",
            "specification": "stcox x, method(breslow) [Analytical Ground Truth]",
            "reference_implementation": "Direct Scipy Numerical Optimization of Breslow Partial Log-Likelihood",
            "reference_version": "Analytical Ground Truth",
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": n,
            "sample_selection": f"60 observations ({int(np.sum(failed))} events, {int(n - np.sum(failed))} censored)",
            "coefficient_comparison": b_comp,
            "standard_error_comparison": None,
            "covariance_comparison": None,
            "metrics": metrics,
            "additional_diagnostics": {
                "events": int(np.sum(failed)),
                "censored": int(n - np.sum(failed)),
                "optimizer_success": bool(opt.success)
            },
            "abs_tolerance": 1e-5,
            "rel_tolerance": 1e-4,
            "validation_method": ValidationMethod.ANALYTICAL_GROUND_TRUTH.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 18. Ordered Logistic Regression (Internal Consistency with OrderedModel)
    # -------------------------------------------------------------------------
    def bench_18_ordered_logit(self):
        bench_id = "BENCH_18_ORDERED_LOGIT"
        rng = np.random.RandomState(42)
        n = 120
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        latent = 0.8 * x1 - 0.5 * x2 + rng.logistic(0, 1, n)
        y = np.select([latent < -0.5, latent < 1.0], [1, 2], default=3)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

        r = compute_ologit(df, "y", ["x1", "x2"])
        mod = OrderedModel(df["y"], df[["x1", "x2"]], distr="logit")
        res = mod.fit(method="bfgs", disp=False)

        pot_b = {c["name"]: float(c["coef"]) for c in r["coefs"] if not c.get("is_cutpoint")}
        sm_b = {"x1": float(res.params["x1"]), "x2": float(res.params["x2"])}
        b_comp = compare_numerical_dictionaries(pot_b, sm_b, abs_tol=1e-7, rel_tol=1e-5)

        pot_se = {c["name"]: float(c["se"]) for c in r["coefs"] if not c.get("is_cutpoint")}
        sm_se = {"x1": float(res.bse["x1"]), "x2": float(res.bse["x2"])}
        se_comp = compare_numerical_dictionaries(pot_se, sm_se, abs_tol=1e-6, rel_tol=1e-5)

        # Cutpoints and cutpoint SE comparison
        pot_cuts = r["cuts"]
        ref_cuts = [float(mod.transform_threshold_params(res.params[-2:])[1]),
                    float(mod.transform_threshold_params(res.params[-2:])[2])]
        cuts_diff = max(abs(pot_cuts[0] - ref_cuts[0]), abs(pot_cuts[1] - ref_cuts[1]))

        # Delta-method threshold covariance propagation for reference cutpoint standard errors
        # theta = [theta_0, theta_1] where c_0 = theta_0, c_1 = theta_0 + exp(theta_1)
        raw_cov = res.cov_params().to_numpy(dtype=float)
        th1 = float(res.params.iloc[-1])
        J_thresh = np.array([
            [1.0, 0.0],
            [1.0, np.exp(np.clip(th1, -20, 20))]
        ])
        cov_thresh = raw_cov[-2:, -2:]
        ref_cut_cov = J_thresh @ cov_thresh @ J_thresh.T
        ref_cut_se = [float(np.sqrt(ref_cut_cov[0, 0])), float(np.sqrt(ref_cut_cov[1, 1]))]
        pot_cut_se = [c["se"] for c in r["coefs"] if c.get("is_cutpoint")]
        cut_se_diff = max(abs(pot_cut_se[0] - ref_cut_se[0]), abs(pot_cut_se[1] - ref_cut_se[1]))

        ll_diff = abs(r["ll"] - float(res.llf))

        # Predicted probabilities across all observations compared to statsmodels predict()
        xb = pot_b["x1"] * df["x1"].to_numpy() + pot_b["x2"] * df["x2"].to_numpy()
        def _logit(v): return 1.0 / (1.0 + np.exp(-v))
        p1 = _logit(pot_cuts[0] - xb)
        p2 = _logit(pot_cuts[1] - xb) - _logit(pot_cuts[0] - xb)
        p3 = 1.0 - _logit(pot_cuts[1] - xb)
        pot_probs = np.column_stack([p1, p2, p3])
        ref_probs = np.asarray(res.predict())
        prob_max_diff = float(np.max(np.abs(pot_probs - ref_probs)))

        probs_row_sums = np.sum(pot_probs, axis=1)
        prob_sum_err = float(np.max(np.abs(probs_row_sums - 1.0)))

        metrics = [
            create_benchmark_metric(
                "coefficients", pot_b, sm_b,
                difference=b_comp["max_abs_diff"], abs_tolerance=1e-7, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if b_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(b_comp["failure_reasons"]) if not b_comp["passed"] else None
            ),
            create_benchmark_metric(
                "standard_errors", pot_se, sm_se,
                difference=se_comp["max_abs_diff"], abs_tolerance=1e-6, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if se_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(se_comp["failure_reasons"]) if not se_comp["passed"] else None
            ),
            create_benchmark_metric(
                "cutpoints", pot_cuts, ref_cuts,
                difference=cuts_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if cuts_diff < 1e-6 else MetricStatus.FAIL
            ),
            create_benchmark_metric(
                "cutpoint_standard_errors", pot_cut_se, ref_cut_se,
                difference=cut_se_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if cut_se_diff < 1e-6 else MetricStatus.FAIL
            ),
            create_benchmark_metric(
                "predicted_probabilities", pot_probs, ref_probs,
                difference=prob_max_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if prob_max_diff < 1e-6 else MetricStatus.FAIL
            ),
            create_benchmark_metric(
                "log_likelihood", r["ll"], float(res.llf),
                difference=ll_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if ll_diff < 1e-6 else MetricStatus.FAIL
            ),
            create_benchmark_metric(
                "predicted_probabilities_sum_to_one", prob_sum_err, 0.0,
                difference=prob_sum_err, abs_tolerance=1e-12,
                is_mandatory=True,
                status=MetricStatus.PASS if prob_sum_err < 1e-12 else MetricStatus.FAIL
            ),
            create_benchmark_metric(
                "convergence", r["converged"], True,
                is_mandatory=True,
                status=MetricStatus.PASS if r["converged"] is True else MetricStatus.FAIL
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "ologit",
            "dataset_id": "simulated_ordinal (120 obs, 3 categories)",
            "specification": "ologit y x1 x2",
            "reference_implementation": "statsmodels.miscmodels.ordinal_model.OrderedModel",
            "reference_version": sm.__version__,
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": n,
            "sample_selection": "120 observations, 3 ordinal outcome levels",
            "coefficient_comparison": b_comp,
            "standard_error_comparison": se_comp,
            "covariance_comparison": None,
            "metrics": metrics,
            "additional_diagnostics": {
                "ll_potatostics": r["ll"],
                "ll_reference": float(res.llf),
                "cuts": r["cuts"],
                "cuts_diff": cuts_diff,
                "cut_se_diff": cut_se_diff,
                "prob_sum_err": prob_sum_err,
                "converged": r["converged"]
            },
            "abs_tolerance": 1e-7,
            "rel_tolerance": 1e-5,
            "validation_method": ValidationMethod.INTERNAL_CONSISTENCY.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    # -------------------------------------------------------------------------
    # 19. Cox Proportional Hazards (Internal Consistency with PHReg)
    # -------------------------------------------------------------------------
    def bench_19_cox_phreg(self):
        bench_id = "BENCH_19_COX_PHREG"
        rng = np.random.RandomState(42)
        n = 80
        x1 = rng.normal(0, 1, n)
        haz = np.exp(0.7 * x1)
        t_event = rng.exponential(1.0 / haz)
        t_censor = rng.uniform(0.5, 3.0, n)
        t_obs = np.minimum(t_event, t_censor)
        failed = (t_event <= t_censor).astype(int)
        df = pd.DataFrame({"time": t_obs, "failed": failed, "x1": x1})

        r = compute_stcox(df, "time", "failed", ["x1"], method="breslow")
        ph = PHReg(endog=df["time"], exog=df[["x1"]], status=df["failed"], ties="breslow")
        res = ph.fit()

        pot_b = {"x1": float(r["coefs"][0]["coef"])}
        sm_b = {"x1": float(res.params[0])}
        b_comp = compare_numerical_dictionaries(pot_b, sm_b, abs_tol=1e-7, rel_tol=1e-5)

        pot_se = {"x1": float(r["coefs"][0]["se"])}
        sm_se = {"x1": float(res.bse[0])}
        se_comp = compare_numerical_dictionaries(pot_se, sm_se, abs_tol=1e-6, rel_tol=1e-5)

        hr_pot = float(r["coefs"][0]["hr"])
        hr_ref = float(np.exp(res.params[0]))
        hr_diff = abs(hr_pot - hr_ref)
        ll_diff = abs(r["ll"] - float(res.llf))

        metrics = [
            create_benchmark_metric(
                "coefficients", pot_b, sm_b,
                difference=b_comp["max_abs_diff"], abs_tolerance=1e-7, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if b_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(b_comp["failure_reasons"]) if not b_comp["passed"] else None
            ),
            create_benchmark_metric(
                "standard_errors", pot_se, sm_se,
                difference=se_comp["max_abs_diff"], abs_tolerance=1e-6, rel_tolerance=1e-5,
                is_mandatory=True,
                status=MetricStatus.PASS if se_comp["passed"] else MetricStatus.FAIL,
                failure_explanation="; ".join(se_comp["failure_reasons"]) if not se_comp["passed"] else None
            ),
            create_benchmark_metric(
                "hazard_ratios", {"x1": hr_pot}, {"x1": hr_ref},
                difference=hr_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if hr_diff < 1e-6 else MetricStatus.FAIL
            ),
            create_benchmark_metric(
                "log_partial_likelihood", r["ll"], float(res.llf),
                difference=ll_diff, abs_tolerance=1e-6,
                is_mandatory=True,
                status=MetricStatus.PASS if ll_diff < 1e-6 else MetricStatus.FAIL
            ),
            create_benchmark_metric(
                "convergence", r["converged"], True,
                is_mandatory=True,
                status=MetricStatus.PASS if r["converged"] is True else MetricStatus.FAIL
            )
        ]
        status, expl = aggregate_benchmark_status(metrics)

        self.results.append({
            "benchmark_id": bench_id,
            "estimator_name": "stcox",
            "dataset_id": "simulated_survival (80 obs, right-censored)",
            "specification": "stcox x1, method(breslow)",
            "reference_implementation": "statsmodels.duration.hazard_regression.PHReg",
            "reference_version": sm.__version__,
            "potatostics_version": POTATOSTICS_VERSION,
            "sample_size": n,
            "sample_selection": f"80 observations ({int(np.sum(failed))} events, {int(n - np.sum(failed))} censored)",
            "coefficient_comparison": b_comp,
            "standard_error_comparison": se_comp,
            "covariance_comparison": None,
            "metrics": metrics,
            "additional_diagnostics": {
                "events": int(np.sum(failed)),
                "censored": int(n - np.sum(failed)),
                "hazard_ratio_diff": hr_diff,
                "log_likelihood_diff": ll_diff,
                "converged": r["converged"]
            },
            "abs_tolerance": 1e-7,
            "rel_tolerance": 1e-5,
            "validation_method": ValidationMethod.INTERNAL_CONSISTENCY.value,
            "timestamp": getattr(self, "run_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "execution_environment": self._get_env_info(),
            "status": status,
            "failure_explanation": expl
        })

    def _save_results(self):
        json_path = os.path.join(self.output_dir, "benchmarks.json")
        def _json_default(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            if isinstance(obj, set):
                return list(obj)
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, default=_json_default)

        md_path = os.path.join(self.output_dir, "benchmark_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown_report())

    def generate_markdown_report(self) -> str:
        report_time_str = (
            datetime.datetime.fromisoformat(self.run_timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')
            if hasattr(self, 'run_timestamp') and self.run_timestamp
            else datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        )
        lines = [
            "# Potatostics — Automated Econometric Benchmark Report",
            f"**Generated at**: {report_time_str}",
            f"**Total Benchmarks**: {len(self.results)} | **Passed**: {sum(1 for r in self.results if r['status'] == BenchmarkStatus.PASS.value)} | **Failed**: {sum(1 for r in self.results if r['status'] == BenchmarkStatus.FAIL.value)} | **Skipped**: {sum(1 for r in self.results if r['status'] == BenchmarkStatus.SKIPPED.value)}",
            "",
            "| Benchmark Identifier | Estimator | Validation Type | Reference Standard | Max Abs Coef Diff | Max Abs SE Diff | Status |",
            "| :--- | :--- | :--- | :--- | :---: | :---: | :---: |"
        ]
        for r in self.results:
            b_diff_str = f"{r['coefficient_comparison']['max_abs_diff']:.2e}" if r.get('coefficient_comparison') else "N/A"
            se_diff_str = f"{r['standard_error_comparison']['max_abs_diff']:.2e}" if r.get('standard_error_comparison') else "N/A"
            lines.append(
                f"| **`{r['benchmark_id']}`** | `{r['estimator_name']}` | `{r['validation_method']}` | {r['reference_implementation']} | "
                f"{b_diff_str} | {se_diff_str} | **{r['status']}** |"
            )
        lines.append("")
        lines.append("All benchmarks compare exact identical datasets, samples, and degrees of freedom conventions.")
        lines.append("Untested metrics are recorded as `None` / `N/A` rather than populated with placeholder zeros.")
        return "\n".join(lines)
