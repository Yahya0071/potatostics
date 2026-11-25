"""
Stage 42 — Benchmark Integrity, Metric Aggregation, and Tolerance Safety Tests
Validates:
1. compare_numerical_dictionaries rejects empty dicts, unexpected NaN, unexpected inf, key collisions.
2. Order invariance of compare_numerical_dictionaries.
3. Central status aggregation logic (mandatory vs optional, fail propagation).
4. Corrupted covariance matrix causes benchmark failure.
5. DRDID checks a four-cell analytical SE and records the unavailable external comparison.
6. Assets bundled loading via importlib.resources.
"""

import math
import numpy as np
import pytest

from engine.benchmarks import (
    compare_numerical_dictionaries,
    aggregate_benchmark_status,
    create_benchmark_metric,
    MetricStatus,
    BenchmarkStatus,
    BenchmarkRunner,
    _load_benchmark_data
)


def test_compare_dictionaries_empty_fails():
    res = compare_numerical_dictionaries({}, {})
    assert not res["passed"]
    assert any("Empty comparison set" in r for r in res["failure_reasons"])


def test_compare_dictionaries_unexpected_nan_fails():
    d1 = {"a": float("nan"), "b": 1.0}
    d2 = {"a": 2.0, "b": 1.0}
    res = compare_numerical_dictionaries(d1, d2)
    assert not res["passed"]
    assert any("NaN" in r for r in res["failure_reasons"])


def test_compare_dictionaries_unexpected_inf_fails():
    d1 = {"a": float("inf"), "b": 1.0}
    d2 = {"a": 2.0, "b": 1.0}
    res = compare_numerical_dictionaries(d1, d2)
    assert not res["passed"]
    assert any("Infinite" in r or "inf" in r.lower() for r in res["failure_reasons"])


def test_compare_dictionaries_duplicate_alias_collision():
    # If key aliases map two different keys in potatostics to the same reference key
    d1 = {"_cons": 1.0, "const": 1.0}
    d2 = {"const": 1.0}
    # Both map to const
    res = compare_numerical_dictionaries(d1, d2, key_aliases={"_cons": "const"})
    assert not res["passed"]
    assert any("Duplicate normalized" in r for r in res["failure_reasons"])


def test_compare_dictionaries_order_invariance():
    d1 = {"z": 10.0, "a": 1.0, "m": 5.0}
    d2 = {"a": 1.0000001, "m": 5.0, "z": 10.0000002}
    res1 = compare_numerical_dictionaries(d1, d2)

    # Reverse insertion order
    d1_rev = {"m": 5.0, "z": 10.0, "a": 1.0}
    d2_rev = {"z": 10.0000002, "a": 1.0000001, "m": 5.0}
    res2 = compare_numerical_dictionaries(d1_rev, d2_rev)

    assert res1["passed"] == res2["passed"]
    assert math.isclose(res1["max_abs_diff"], res2["max_abs_diff"], rel_tol=1e-12)
    assert math.isclose(res1["max_rel_diff"], res2["max_rel_diff"], rel_tol=1e-12)


def test_aggregate_benchmark_status_logic():
    # 1. All mandatory pass -> PASS
    m1 = create_benchmark_metric("b", 1.0, 1.0, is_mandatory=True, status=MetricStatus.PASS)
    m2 = create_benchmark_metric("se", 0.1, 0.1, is_mandatory=True, status=MetricStatus.PASS)
    status, _ = aggregate_benchmark_status([m1, m2])
    assert status == BenchmarkStatus.PASS.value

    # 2. Mandatory fail -> FAIL
    m3 = create_benchmark_metric("cov", 0.5, 0.1, is_mandatory=True, status=MetricStatus.FAIL, failure_explanation="Diff > tol")
    status, expl = aggregate_benchmark_status([m1, m2, m3])
    assert status == BenchmarkStatus.FAIL.value
    assert "cov" in expl

    # 3. Optional metric NOT_TESTED does NOT block PASS
    m_opt = create_benchmark_metric("extra", None, None, is_mandatory=False, status=MetricStatus.NOT_TESTED)
    status, _ = aggregate_benchmark_status([m1, m2, m_opt])
    assert status == BenchmarkStatus.PASS.value

    # 4. Mandatory metric NOT_TESTED blocks PASS
    m_mand_nt = create_benchmark_metric("mand", None, None, is_mandatory=True, status=MetricStatus.NOT_TESTED)
    status, expl = aggregate_benchmark_status([m1, m2, m_mand_nt])
    assert status == BenchmarkStatus.NOT_TESTED.value

    # 5. Empty metrics list fails
    status, _ = aggregate_benchmark_status([])
    assert status == BenchmarkStatus.FAIL.value


def test_corrupted_covariance_causes_benchmark_failure():
    # Corrupting covariance metric forces aggregate_benchmark_status to return FAIL
    b_metric = create_benchmark_metric("coef", {"x": 1.0}, {"x": 1.0}, is_mandatory=True, status=MetricStatus.PASS)
    se_metric = create_benchmark_metric("se", {"x": 0.1}, {"x": 0.1}, is_mandatory=True, status=MetricStatus.PASS)
    corrupted_cov_metric = create_benchmark_metric(
        "covariance_matrix",
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 999.0]],
        difference=998.0, abs_tolerance=1e-6,
        is_mandatory=True,
        status=MetricStatus.FAIL,
        failure_explanation="Covariance discrepancy 998.0 exceeds 1e-6"
    )
    status, expl = aggregate_benchmark_status([b_metric, se_metric, corrupted_cov_metric])
    assert status == BenchmarkStatus.FAIL.value
    assert "covariance_matrix" in expl


def test_drdid_records_analytical_and_external_standard_error_evidence():
    runner = BenchmarkRunner()
    runner.bench_14_drdid()
    drdid_res, = runner.results
    assert drdid_res["standard_error_comparison"] is None
    metrics = {metric["metric_name"]: metric for metric in drdid_res["metrics"]}
    external_se = metrics["standard_errors"]
    assert external_se["status"] == MetricStatus.NOT_TESTED.value
    assert external_se["difference"] is None
    analytical_se = metrics["repeated_cross_section_four_cell_se"]
    assert analytical_se["is_mandatory"]
    assert analytical_se["status"] == MetricStatus.PASS.value
    assert analytical_se["reference_value"] == pytest.approx(np.sqrt(4 * 2.0 / 5))


def test_bundled_assets_load_cleanly():
    # Verify that _load_benchmark_data loads bundled package datasets
    for filename in ["auto.dta", "basque.csv", "mpdta.csv"]:
        df = _load_benchmark_data(filename)
        assert df is not None
        assert len(df) > 0
