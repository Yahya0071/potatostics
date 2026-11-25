"""
Stage 46 — Benchmark Custom-Alias Regression and Alignment Test Suite.
Verifies custom parameter aliases, schema preservation, conflict detection,
duplicate normalized name detection, parameter ordering, covariance alignment,
and rejection of tampered PASS records.
"""

import unittest
import numpy as np

from engine.benchmarks import (
    create_benchmark_metric,
    aggregate_benchmark_status,
    compare_named_covariance_matrices,
    MetricStatus,
    MetricType,
    BenchmarkStatus
)


class TestStage46BenchmarkAlias(unittest.TestCase):

    def test_01_valid_custom_aliases_factory_and_aggregator_agreement(self):
        """Verifies that valid custom aliases pass in both metric factory and aggregator."""
        pot_b = {"alpha": 1.25, "beta": 3.50}
        ref_b = {"A": 1.25, "B": 3.50}
        aliases = {"A": "alpha", "B": "beta"}

        metric = create_benchmark_metric(
            metric_name="coefficients",
            potatostics_value=pot_b,
            reference_value=ref_b,
            abs_tolerance=1e-7,
            rel_tolerance=1e-5,
            key_aliases=aliases,
            is_mandatory=True
        )

        self.assertEqual(metric["status"], MetricStatus.PASS.value)
        self.assertEqual(metric["key_aliases"], aliases)

        # Aggregation must re-evaluate with the preserved alias mapping and also pass
        status, expl = aggregate_benchmark_status([metric])
        self.assertEqual(status, BenchmarkStatus.PASS.value)
        self.assertIsNone(expl)

    def test_02_invalid_alias_format_rejected(self):
        """Verifies that non-dict or non-string aliases fail cleanly."""
        pot_b = {"alpha": 1.25}
        ref_b = {"A": 1.25}

        # Non-dict aliases
        metric_bad_type = create_benchmark_metric(
            metric_name="coefficients",
            potatostics_value=pot_b,
            reference_value=ref_b,
            key_aliases=["A", "alpha"],  # type: ignore
            is_mandatory=True
        )
        self.assertEqual(metric_bad_type["status"], MetricStatus.FAIL.value)

        # Non-string alias keys
        metric_non_str = create_benchmark_metric(
            metric_name="coefficients",
            potatostics_value=pot_b,
            reference_value=ref_b,
            key_aliases={123: "alpha"},  # type: ignore
            is_mandatory=True
        )
        self.assertEqual(metric_non_str["status"], MetricStatus.FAIL.value)

    def test_03_duplicate_normalized_names_fail(self):
        """Verifies that aliases collapsing two distinct parameters to the same name fail."""
        pot_b = {"alpha": 1.25, "gamma": 2.50}
        ref_b = {"A": 1.25, "B": 2.50}
        # Both A and B map to 'alpha'
        colliding_aliases = {"A": "alpha", "B": "alpha"}

        metric = create_benchmark_metric(
            metric_name="coefficients",
            potatostics_value=pot_b,
            reference_value=ref_b,
            key_aliases=colliding_aliases,
            is_mandatory=True
        )
        self.assertEqual(metric["status"], MetricStatus.FAIL.value)
        self.assertIn("Duplicate normalized reference parameter name", metric["failure_explanation"])

        status, expl = aggregate_benchmark_status([metric])
        self.assertEqual(status, BenchmarkStatus.FAIL.value)

    def test_04_missing_parameters_cannot_be_concealed_by_aliases(self):
        """Verifies that missing parameters fail even if other parameters match through aliases."""
        pot_b = {"alpha": 1.25}
        ref_b = {"A": 1.25, "B": 4.50}
        aliases = {"A": "alpha"}

        metric = create_benchmark_metric(
            metric_name="coefficients",
            potatostics_value=pot_b,
            reference_value=ref_b,
            key_aliases=aliases,
            is_mandatory=True
        )
        self.assertEqual(metric["status"], MetricStatus.FAIL.value)
        self.assertIn("Missing parameter in potatostics: 'B'", metric["failure_explanation"])

        status, expl = aggregate_benchmark_status([metric])
        self.assertEqual(status, BenchmarkStatus.FAIL.value)

    def test_05_incorrect_coefficients_fail_with_aliases(self):
        """Verifies that numerical discrepancies fail even when aliases correctly match names."""
        pot_b = {"alpha": 1.25}
        ref_b = {"A": 9.99}
        aliases = {"A": "alpha"}

        metric = create_benchmark_metric(
            metric_name="coefficients",
            potatostics_value=pot_b,
            reference_value=ref_b,
            abs_tolerance=1e-5,
            key_aliases=aliases,
            is_mandatory=True
        )
        self.assertEqual(metric["status"], MetricStatus.FAIL.value)
        self.assertIn("exceeded tolerance", metric["failure_explanation"])

        status, expl = aggregate_benchmark_status([metric])
        self.assertEqual(status, BenchmarkStatus.FAIL.value)

    def test_06_parameter_ordering_invariance(self):
        """Verifies that dictionary key insertion order does not affect comparison success."""
        pot_b = {"beta": 2.0, "gamma": 3.0, "alpha": 1.0}
        ref_b = {"A": 1.0, "C": 3.0, "B": 2.0}
        aliases = {"A": "alpha", "B": "beta", "C": "gamma"}

        metric = create_benchmark_metric(
            metric_name="coefficients",
            potatostics_value=pot_b,
            reference_value=ref_b,
            abs_tolerance=1e-7,
            key_aliases=aliases,
            is_mandatory=True
        )
        self.assertEqual(metric["status"], MetricStatus.PASS.value)

        status, expl = aggregate_benchmark_status([metric])
        self.assertEqual(status, BenchmarkStatus.PASS.value)

    def test_07_compare_named_covariance_matrices_with_reordered_labels_and_aliases(self):
        """Verifies that compare_named_covariance_matrices reorders/aligns parameters and aliases."""
        # 2x2 covariance matrices
        # Potatostics order: ['alpha', 'beta']
        # Reference order: ['B', 'A']
        cov_pot = np.array([
            [0.04, 0.01],
            [0.01, 0.09]
        ])
        # In reference order [B, A], B corresponds to beta and A to alpha
        cov_ref = np.array([
            [0.09, 0.01],
            [0.01, 0.04]
        ])
        pot_labels = ["alpha", "beta"]
        ref_labels = ["B", "A"]
        aliases = {"A": "alpha", "B": "beta"}

        comp = compare_named_covariance_matrices(
            cov_pot, cov_ref,
            pot_labels=pot_labels,
            ref_labels=ref_labels,
            key_aliases=aliases,
            abs_tol=1e-6
        )
        self.assertTrue(comp["passed"], f"Expected pass, got: {comp['failure_reasons']}")
        self.assertEqual(comp["aligned_parameters"], ["alpha", "beta"])
        self.assertAlmostEqual(comp["max_abs_diff"], 0.0, places=9)

        # Mismatching covariance value should fail
        cov_ref_bad = np.array([
            [0.99, 0.01],
            [0.01, 0.04]
        ])
        comp_bad = compare_named_covariance_matrices(
            cov_pot, cov_ref_bad,
            pot_labels=pot_labels,
            ref_labels=ref_labels,
            key_aliases=aliases,
            abs_tol=1e-6
        )
        self.assertFalse(comp_bad["passed"])
        self.assertIn("diff=", comp_bad["failure_reasons"][0])

    def test_08_tampered_metric_claiming_pass_with_aliases_is_rejected(self):
        """Verifies that a forged PASS metric with numerical mismatch is caught during aggregation."""
        metric = {
            "metric_name": "tampered_coefficients",
            "metric_type": MetricType.NAMED_PARAMETER.value,
            "potatostics_value": {"alpha": 1.0},
            "reference_value": {"A": 500.0},
            "key_aliases": {"A": "alpha"},
            "abs_tolerance": 1e-7,
            "rel_tolerance": 1e-5,
            "difference": 0.0,
            "is_mandatory": True,
            "status": MetricStatus.PASS.value  # Fabricated PASS!
        }
        status, expl = aggregate_benchmark_status([metric])
        self.assertEqual(status, BenchmarkStatus.FAIL.value)
        self.assertIn("Claimed PASS but evaluated to FAIL", expl)


if __name__ == "__main__":
    unittest.main()
