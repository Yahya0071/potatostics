"""
Stage 45 — Validation Integrity and Adversarial Benchmark Tests
Validates:
1. Elimination of fabricated PASS records in aggregate_benchmark_status.
2. Complete schema validation of metric dictionaries.
3. Removal of conflicting difference validation in elementwise comparisons.
4. Strict Boolean type validation (rejection of string truthiness).
5. Robust tolerance validation in compare_numerical_dictionaries (rejecting NaN, Inf, negative).
6. Non-numeric parameter rejection and zero-tolerance handling.
"""

import pytest

from engine.benchmarks import (
    create_benchmark_metric,
    aggregate_benchmark_status,
    compare_numerical_dictionaries,
    MetricStatus,
    MetricType,
    BenchmarkStatus
)


class TestStage45ValidationIntegrity:

    def test_01_fabricated_scalar_pass_record_rejected_in_aggregation(self):
        """A metric dict claiming PASS whose underlying scalar numbers disagree must FAIL."""
        fabricated_m = {
            "metric_name": "fabricated_scalar",
            "metric_type": "SCALAR_NUMERICAL",
            "potatostics_value": 1.0,
            "reference_value": 999.0,
            "abs_tolerance": 1e-9,
            "rel_tolerance": 0.0,
            "is_mandatory": True,
            "status": "PASS",
            "difference": 0.0,
            "evaluation_method": "numerical_comparison",
            "failure_explanation": None
        }
        status, expl = aggregate_benchmark_status([fabricated_m])
        assert status == BenchmarkStatus.FAIL.value
        assert expl is not None
        assert "mismatch" in expl.lower() or "tolerance exceeded" in expl.lower() or "fail" in expl.lower()

    def test_02_fabricated_vector_pass_record_rejected_in_aggregation(self):
        """A metric dict claiming PASS whose vector elements disagree must FAIL."""
        fabricated_vector = {
            "metric_name": "fabricated_vector",
            "metric_type": "VECTOR_NUMERICAL",
            "potatostics_value": [1.0, 2.0, 3.0],
            "reference_value": [1.0, 2.0, 99.0],
            "abs_tolerance": 1e-5,
            "rel_tolerance": 1e-5,
            "is_mandatory": True,
            "status": "PASS",
            "difference": 0.0,
            "evaluation_method": "numerical_comparison",
            "failure_explanation": None
        }
        status, expl = aggregate_benchmark_status([fabricated_vector])
        assert status == BenchmarkStatus.FAIL.value

    def test_03_incomplete_metric_schema_rejected_in_aggregation(self):
        """Metrics missing mandatory schema fields must be rejected with FAIL."""
        incomplete_m = {
            "metric_name": "missing_fields",
            "status": "PASS",
            "is_mandatory": True
            # Missing metric_type, potatostics_value, reference_value, tolerances, etc.
        }
        status, expl = aggregate_benchmark_status([incomplete_m])
        assert status == BenchmarkStatus.FAIL.value
        assert "schema" in expl.lower() or "missing" in expl.lower()

    def test_04_caller_difference_does_not_break_elementwise_allclose(self):
        """
        Caller-supplied difference must not override valid elementwise allclose match.
        For actual=[1000.5, 0.00005], ref=[1000.0, 0.0], atol=0.01, rtol=0.001:
        element 0 diff 0.5 <= 0.01 + 0.001 * 1000 = 1.01
        element 1 diff 0.00005 <= 0.01 + 0.001 * 0 = 0.01
        Supplying difference=0.5 must NOT fail the metric.
        """
        actual = [1000.5, 0.00005]
        ref = [1000.0, 0.0]
        m = create_benchmark_metric(
            metric_name="hetero_scale_with_diff",
            potatostics_value=actual,
            reference_value=ref,
            difference=0.5,
            abs_tolerance=0.01,
            rel_tolerance=0.001,
            is_mandatory=True
        )
        assert m["status"] == MetricStatus.PASS.value
        assert m["failure_explanation"] is None

    def test_05_strict_boolean_validation_rejects_string_truthiness(self):
        """String 'False' must NOT compare equal to boolean True via Python truthiness."""
        m1 = create_benchmark_metric(
            metric_name="bool_string_truthiness",
            potatostics_value="False",
            reference_value=True,
            metric_type=MetricType.BOOLEAN_ASSERTION,
            is_mandatory=True
        )
        assert m1["status"] == MetricStatus.FAIL.value
        assert "bool" in m1["failure_explanation"].lower()

        m2 = create_benchmark_metric(
            metric_name="bool_string_both",
            potatostics_value="True",
            reference_value="True",
            metric_type=MetricType.BOOLEAN_ASSERTION,
            is_mandatory=True
        )
        assert m2["status"] == MetricStatus.FAIL.value

        m3 = create_benchmark_metric(
            metric_name="bool_valid",
            potatostics_value=True,
            reference_value=True,
            metric_type=MetricType.BOOLEAN_ASSERTION,
            is_mandatory=True
        )
        assert m3["status"] == MetricStatus.PASS.value

    def test_06_compare_numerical_dictionaries_rejects_nan_tolerance(self):
        """compare_numerical_dictionaries must reject NaN tolerance with ValueError."""
        with pytest.raises(ValueError, match="[Nn]a[Nn]"):
            compare_numerical_dictionaries(
                {"b1": 1.0}, {"b1": 1.0},
                abs_tol=float("nan"), rel_tol=1e-5
            )

    def test_07_compare_numerical_dictionaries_rejects_inf_tolerance(self):
        """compare_numerical_dictionaries must reject Inf tolerance with ValueError."""
        with pytest.raises(ValueError, match="[Ii]nf"):
            compare_numerical_dictionaries(
                {"b1": 1.0}, {"b1": 1.0},
                abs_tol=1e-7, rel_tol=float("inf")
            )

    def test_08_compare_numerical_dictionaries_rejects_negative_tolerance(self):
        """compare_numerical_dictionaries must reject negative tolerance with ValueError."""
        with pytest.raises(ValueError, match="negative"):
            compare_numerical_dictionaries(
                {"b1": 1.0}, {"b1": 1.0},
                abs_tol=-1e-5, rel_tol=1e-5
            )

    def test_09_compare_numerical_dictionaries_rejects_non_numeric_values(self):
        """Non-numeric values in dict comparison must fail gracefully with passed=False."""
        res = compare_numerical_dictionaries(
            {"b1": "not_a_number"}, {"b1": 1.0},
            abs_tol=1e-5, rel_tol=1e-5
        )
        assert not res["passed"]
        assert any("non-numeric" in r.lower() for r in res["failure_reasons"])

    def test_10_zero_tolerance_handling(self):
        """Zero tolerance comparison (0.0, 0.0) must pass on exact match and fail on diff."""
        # Exact match
        res_exact = compare_numerical_dictionaries(
            {"b1": 1.5, "b2": 2.0},
            {"b1": 1.5, "b2": 2.0},
            abs_tol=0.0, rel_tol=0.0
        )
        assert res_exact["passed"]

        # Discrepancy
        res_diff = compare_numerical_dictionaries(
            {"b1": 1.5000000001, "b2": 2.0},
            {"b1": 1.5, "b2": 2.0},
            abs_tol=0.0, rel_tol=0.0
        )
        assert not res_diff["passed"]

    def test_11_aggregator_rechecks_named_parameter_underlying_values(self):
        """Aggregator must detect if a NAMED_PARAMETER metric dictionary was tampered with."""
        tampered_named = {
            "metric_name": "tampered_params",
            "metric_type": "NAMED_PARAMETER",
            "potatostics_value": {"b1": 1.0, "b2": 2.0},
            "reference_value": {"b1": 1.0, "b2": 999.0},
            "abs_tolerance": 1e-5,
            "rel_tolerance": 1e-5,
            "is_mandatory": True,
            "status": "PASS",  # False claim
            "difference": 0.0,
            "evaluation_method": "numerical_comparison",
            "failure_explanation": None
        }
        status, expl = aggregate_benchmark_status([tampered_named])
        assert status == BenchmarkStatus.FAIL.value
