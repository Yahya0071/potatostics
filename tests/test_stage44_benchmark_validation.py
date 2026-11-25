import unittest
import numpy as np

from engine.benchmarks import (
    MetricStatus,
    create_benchmark_metric,
    aggregate_benchmark_status,
    BenchmarkStatus
)


class TestStage44BenchmarkValidation(unittest.TestCase):
    """
    Stage 44: Adversarial tests for strict benchmark evaluation,
    unrecognized statuses, empty comparisons, spoofed precomputed passes,
    tolerance validation, and elementwise tolerance acceptance.
    """

    # -------------------------------------------------------------
    # Defect A: Unknown Status Strings
    # -------------------------------------------------------------
    def test_01_unknown_status_string_rejected_in_factory(self):
        """create_benchmark_metric must reject unknown status strings."""
        with self.assertRaises(ValueError) as ctx:
            create_benchmark_metric(
                "coef", 1.0, 1.0,
                status="SOMETHING_ELSE"
            )
        self.assertIn("Invalid or unrecognized metric status", str(ctx.exception))

    def test_02_unknown_status_in_aggregation_causes_failure(self):
        """aggregate_benchmark_status must fail if ANY metric has an unrecognized status."""
        valid_m = create_benchmark_metric("b", 1.0, 1.0, is_mandatory=True)
        corrupted_m = {
            "metric_name": "extra",
            "potatostics_value": 1.0,
            "reference_value": 1.0,
            "status": "UNOFFICIAL_PASS",
            "is_mandatory": False,
            "failure_explanation": None
        }
        status, expl = aggregate_benchmark_status([valid_m, corrupted_m])
        self.assertEqual(status, BenchmarkStatus.FAIL.value)
        self.assertIn("unrecognized status", expl)

    def test_03_invalid_metric_type_rejected(self):
        """create_benchmark_metric must reject invalid metric types."""
        with self.assertRaises(ValueError) as ctx:
            create_benchmark_metric(
                "b", 1.0, 1.0,
                metric_type="NONEXISTENT_TYPE"
            )
        self.assertIn("unrecognized metric_type", str(ctx.exception))

    # -------------------------------------------------------------
    # Defect B: Empty Comparisons and Missing Values
    # -------------------------------------------------------------
    def test_04_missing_values_without_not_tested_must_fail(self):
        """Calling create_benchmark_metric with None, None without NOT_TESTED must FAIL."""
        m = create_benchmark_metric(
            "uncomputed_metric",
            potatostics_value=None,
            reference_value=None,
            is_mandatory=True
        )
        self.assertEqual(m["status"], MetricStatus.FAIL.value)
        self.assertIn("Missing numerical comparison values", m["failure_explanation"])

    def test_05_empty_comparison_set_fails(self):
        """Empty vector or dictionary comparison sets must FAIL."""
        # Empty vector
        m_vec = create_benchmark_metric(
            "empty_vec",
            potatostics_value=[],
            reference_value=[],
            is_mandatory=True
        )
        self.assertEqual(m_vec["status"], MetricStatus.FAIL.value)
        self.assertIn("Empty", m_vec["failure_explanation"])

        # Empty dict
        m_dict = create_benchmark_metric(
            "empty_dict",
            potatostics_value={},
            reference_value={},
            is_mandatory=True
        )
        self.assertEqual(m_dict["status"], MetricStatus.FAIL.value)
        self.assertIn("Empty", m_dict["failure_explanation"])

    def test_06_zero_genuine_mandatory_metrics_cannot_pass(self):
        """Benchmark with only NOT_TESTED or empty metrics cannot produce PASS."""
        m1 = create_benchmark_metric("m1", None, None, is_mandatory=False, status=MetricStatus.NOT_TESTED)
        status, expl = aggregate_benchmark_status([m1])
        self.assertEqual(status, BenchmarkStatus.FAIL.value)

    # -------------------------------------------------------------
    # Defect C: Trusted Precomputed Comparison
    # -------------------------------------------------------------
    def test_07_spoofed_precomputed_pass_is_recomputed_and_fails(self):
        """An adversarial dict with {'passed': True} but corrupted numbers must FAIL."""
        spoofed = {
            "passed": True,  # Attacking the evaluator
            "potatostics_values": {"b1": 1.0, "b2": 2.0},
            "reference_values": {"b1": 1.0, "b2": 999.0},  # Gross discrepancy
            "max_abs_diff": 0.0
        }
        m = create_benchmark_metric(
            "coefs",
            potatostics_value=spoofed,
            reference_value=None,
            is_mandatory=True
        )
        self.assertEqual(m["status"], MetricStatus.FAIL.value)
        self.assertIn("exceeded tolerance", m["failure_explanation"])

    def test_08_duplicate_normalized_parameter_names_fail(self):
        """Normalized duplicate names (e.g. const and _cons both in pot_dict) must FAIL."""
        pot_dict = {"const": 1.0, "_cons": 1.0}
        ref_dict = {"const": 1.0}
        m = create_benchmark_metric(
            "coefs",
            potatostics_value=pot_dict,
            reference_value=ref_dict,
            is_mandatory=True
        )
        self.assertEqual(m["status"], MetricStatus.FAIL.value)
        self.assertIn("Duplicate", m["failure_explanation"])

    def test_09_missing_and_extra_parameters_fail(self):
        """Missing or extra parameters must cause validation failure."""
        m = create_benchmark_metric(
            "coefs",
            potatostics_value={"x1": 1.0},
            reference_value={"x1": 1.0, "x2": 2.0},
            is_mandatory=True
        )
        self.assertEqual(m["status"], MetricStatus.FAIL.value)
        self.assertIn("Missing parameter", m["failure_explanation"])

    # -------------------------------------------------------------
    # Defect D: Invalid Tolerance Values
    # -------------------------------------------------------------
    def test_10_negative_tolerance_rejected(self):
        """Negative tolerance must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_benchmark_metric("b", 1.0, 1.0, abs_tolerance=-1e-5)
        self.assertIn("cannot be negative", str(ctx.exception))

    def test_11_nan_and_inf_tolerance_rejected(self):
        """NaN and Inf tolerance must raise ValueError."""
        with self.assertRaises(ValueError):
            create_benchmark_metric("b", 1.0, 1.0, abs_tolerance=float("nan"))
        with self.assertRaises(ValueError):
            create_benchmark_metric("b", 1.0, 1.0, rel_tolerance=float("inf"))

    def test_12_non_numeric_tolerance_rejected(self):
        """String or bool tolerance must raise ValueError."""
        with self.assertRaises(ValueError):
            create_benchmark_metric("b", 1.0, 1.0, abs_tolerance="0.001")
        with self.assertRaises(ValueError):
            create_benchmark_metric("b", 1.0, 1.0, rel_tolerance=True)

    # -------------------------------------------------------------
    # Defect E: Elementwise Acceptance Logic
    # -------------------------------------------------------------
    def test_13_heterogeneous_scale_vector_satisfies_elementwise_allclose(self):
        """
        Reproducible example from prompt:
        potatostics = [1000.5, 0.00005]
        reference = [1000.0, 0.0]
        atol = 0.01, rtol = 0.001
        Must PASS under numpy allclose criterion: abs(act - ref) <= atol + rtol * abs(ref)
        Element 0 satisfies rtol: 0.5 <= 0.01 + 0.001 * 1000 = 1.01
        Element 1 satisfies atol: 0.00005 <= 0.01 + 0.001 * 0 = 0.01
        """
        pot = [1000.5, 0.00005]
        ref = [1000.0, 0.0]
        m = create_benchmark_metric(
            "vector_test",
            potatostics_value=pot,
            reference_value=ref,
            abs_tolerance=0.01,
            rel_tolerance=0.001,
            is_mandatory=True
        )
        self.assertEqual(m["status"], MetricStatus.PASS.value)

    def test_14_failing_vector_element_reports_exact_index(self):
        """When an element fails elementwise tolerance, exact index must be reported."""
        pot = [1.0, 2.0, 50.0]
        ref = [1.0, 2.0, 40.0]
        m = create_benchmark_metric(
            "vector_test",
            potatostics_value=pot,
            reference_value=ref,
            abs_tolerance=0.1,
            rel_tolerance=0.01,
            is_mandatory=True
        )
        self.assertEqual(m["status"], MetricStatus.FAIL.value)
        self.assertIn("element 2", m["failure_explanation"])

    def test_15_failing_matrix_reports_row_col_index(self):
        """When a matrix element fails, row and column index must be reported."""
        mat_pot = np.array([[1.0, 0.0], [0.0, 5.0]])
        mat_ref = np.array([[1.0, 0.0], [0.0, 1.0]])
        m = create_benchmark_metric(
            "matrix_test",
            potatostics_value=mat_pot,
            reference_value=mat_ref,
            abs_tolerance=0.01,
            rel_tolerance=0.01,
            is_mandatory=True
        )
        self.assertEqual(m["status"], MetricStatus.FAIL.value)
        self.assertIn("(1, 1)", m["failure_explanation"])

    def test_16_schema_validation_in_aggregation(self):
        """Missing required fields in a metric dict passed to aggregate must FAIL."""
        incomplete_m = {"metric_name": "bad"}
        status, expl = aggregate_benchmark_status([incomplete_m])
        self.assertEqual(status, BenchmarkStatus.FAIL.value)
        self.assertIn("missing required schema fields", expl)


if __name__ == "__main__":
    unittest.main()
