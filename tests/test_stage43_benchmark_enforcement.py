import unittest
import numpy as np

from engine.benchmarks import (
    MetricStatus,
    MetricType,
    create_benchmark_metric
)


class TestStage43BenchmarkEnforcement(unittest.TestCase):
    """
    Stage 43 Negative Regression Tests:
    Verifies that caller-supplied status=PASS CANNOT override numerical failures,
    tolerance violations, NaNs, shape mismatches, or unaligned collections.
    """

    def test_01_caller_pass_overridden_on_scalar_difference(self):
        """Passing status=PASS when difference > tolerance must evaluate to FAIL."""
        m = create_benchmark_metric(
            "corrupted_beta",
            potatostics_value=1.5,
            reference_value=1.0,
            abs_tolerance=1e-5,
            rel_tolerance=1e-4,
            status=MetricStatus.PASS  # Attempting to spoof PASS
        )
        self.assertEqual(m["status"], MetricStatus.FAIL.value)
        self.assertIsNotNone(m["failure_explanation"])
        self.assertIn("Tolerance exceeded", m["failure_explanation"])

    def test_02_caller_pass_overridden_on_nan(self):
        """Passing status=PASS when values contain NaN must evaluate to FAIL."""
        m = create_benchmark_metric(
            "nan_metric",
            potatostics_value=np.nan,
            reference_value=1.0,
            status=MetricStatus.PASS
        )
        self.assertEqual(m["status"], MetricStatus.FAIL.value)
        self.assertIn("Non-finite", m["failure_explanation"])

    def test_03_caller_pass_overridden_on_vector_shape_mismatch(self):
        """Passing status=PASS on mismatched vector shapes must evaluate to FAIL."""
        m = create_benchmark_metric(
            "vector_metric",
            potatostics_value=[1.0, 2.0],
            reference_value=[1.0, 2.0, 3.0],
            status=MetricStatus.PASS
        )
        self.assertEqual(m["status"], MetricStatus.FAIL.value)
        self.assertIn("shape mismatch", m["failure_explanation"])

    def test_04_caller_pass_overridden_on_matrix_difference(self):
        """Passing status=PASS when covariance matrix differs must evaluate to FAIL."""
        cov_pot = np.array([[1.0, 0.2], [0.2, 1.0]])
        cov_ref = np.array([[1.0, 0.5], [0.5, 1.0]])
        m = create_benchmark_metric(
            "vcov",
            potatostics_value=cov_pot,
            reference_value=cov_ref,
            abs_tolerance=1e-4,
            status=MetricStatus.PASS
        )
        self.assertEqual(m["status"], MetricStatus.FAIL.value)
        self.assertIn("diff=", m["failure_explanation"])

    def test_05_caller_pass_overridden_on_failed_dict_comparison(self):
        """Passing status=PASS on corrupted parameter dictionary must evaluate to FAIL."""
        m = create_benchmark_metric(
            "coefficients",
            potatostics_value={"x1": 1.0, "x2": 2.0},
            reference_value={"x1": 1.0, "x2": 5.0},
            abs_tolerance=1e-5,
            status=MetricStatus.PASS
        )
        self.assertEqual(m["status"], MetricStatus.FAIL.value)

    def test_06_not_tested_remains_not_tested(self):
        """Optional un-run metrics designated NOT_TESTED remain NOT_TESTED."""
        m = create_benchmark_metric(
            "external_r",
            potatostics_value=None,
            reference_value=None,
            is_mandatory=False,
            status=MetricStatus.NOT_TESTED
        )
        self.assertEqual(m["status"], MetricStatus.NOT_TESTED.value)
        self.assertEqual(m["metric_type"], MetricType.NOT_TESTED_RECORD.value)


if __name__ == "__main__":
    unittest.main()
