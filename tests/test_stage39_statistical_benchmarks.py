"""Test Stage 39: Automated Independent Scientific Benchmarks

Verifies Potatostics econometric estimators against statsmodels, linearmodels,
and analytical ground truth benchmarks across classical OLS, robust HC1,
cluster-robust SE, Logit, Probit, Poisson, Quantile Regression, IV/2SLS,
Panel FE, HP filter, VAR(1), Kaplan-Meier, Goodman-Bacon DiD, and Doubly Robust DiD.
Validates the structured benchmark result schema, relative/absolute differences,
and explicit validation method classifications.
"""

import json
from pathlib import Path
import pytest
from engine.benchmarks import BenchmarkRunner, ValidationMethod, BenchmarkStatus


class TestStatisticalBenchmarks:
    @classmethod
    def setup_class(cls):
        cls.runner = BenchmarkRunner()
        cls.results = cls.runner.run_all()

    def test_all_14_benchmarks_pass(self):
        assert len(self.results) == 19
        for res in self.results:
            if res["status"] == BenchmarkStatus.SKIPPED.value:
                # Optional dependencies (e.g. linearmodels) may legitimately be skipped
                continue
            assert res["status"] == BenchmarkStatus.PASS.value, (
                f"Benchmark '{res['benchmark_id']}' failed! "
                f"failure_explanation: {res.get('failure_explanation')}"
            )

    def test_benchmark_files_generated(self):
        json_file = Path(self.runner.output_dir) / "benchmarks.json"
        md_file = Path(self.runner.output_dir) / "benchmark_report.md"
        assert json_file.exists()
        assert md_file.exists()

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 19
        assert all(item["status"] in (BenchmarkStatus.PASS.value, BenchmarkStatus.SKIPPED.value) for item in data)

    def test_ols_classical_precision(self):
        ols_res = [r for r in self.results if r["benchmark_id"] == "BENCH_01_OLS_CLASSICAL"][0]
        assert ols_res["coefficient_comparison"]["max_abs_diff"] < 1e-9
        assert ols_res["standard_error_comparison"]["max_abs_diff"] < 1e-9
        assert ols_res["validation_method"] == ValidationMethod.INDEPENDENT_EXTERNAL.value

    def test_cluster_ols_no_zero_placeholder(self):
        clust_res = [r for r in self.results if r["benchmark_id"] == "BENCH_03_OLS_CLUSTER"][0]
        # Must compare actual non-zero differences and match statsmodels
        assert clust_res["coefficient_comparison"]["max_abs_diff"] < 1e-9
        assert clust_res["standard_error_comparison"]["max_abs_diff"] < 1e-6
        assert clust_res["additional_diagnostics"]["n_clusters_potatostics"] == 5

    def test_quantile_regression_actual_se(self):
        qreg_res = [r for r in self.results if r["benchmark_id"] == "BENCH_07_QUANTILE_REGRESSION"][0]
        assert qreg_res["standard_error_comparison"] is not None
        assert qreg_res["standard_error_comparison"]["max_abs_diff"] < 1e-6
        assert qreg_res["additional_diagnostics"]["converged"] is True

    def test_var_all_equations_and_metrics(self):
        var_res = [r for r in self.results if r["benchmark_id"] == "BENCH_11_VAR"][0]
        assert len(var_res["coefficient_comparison"]["matched_keys"]) == 6
        assert len(var_res["standard_error_comparison"]["matched_keys"]) == 6
        assert var_res["additional_diagnostics"]["aic_diff"] < 1e-6
        assert var_res["additional_diagnostics"]["bic_diff"] < 1e-6
        assert var_res["validation_method"] == ValidationMethod.INTERNAL_CONSISTENCY.value

    def test_hp_filter_internal_consistency(self):
        hp_res = [r for r in self.results if r["benchmark_id"] == "BENCH_10_HP_FILTER"][0]
        assert hp_res["validation_method"] == ValidationMethod.INTERNAL_CONSISTENCY.value

    def test_analytical_benchmarks_precision(self):
        hp_ana = [r for r in self.results if r["benchmark_id"] == "BENCH_15_HP_FILTER_ANALYTICAL"][0]
        assert hp_ana["validation_method"] == ValidationMethod.ANALYTICAL_GROUND_TRUTH.value
        assert hp_ana["status"] == BenchmarkStatus.PASS.value

        var_ana = [r for r in self.results if r["benchmark_id"] == "BENCH_16_VAR_ANALYTICAL_OLS"][0]
        assert var_ana["validation_method"] == ValidationMethod.ANALYTICAL_GROUND_TRUTH.value
        assert var_ana["status"] == BenchmarkStatus.PASS.value

        cox_ana = [r for r in self.results if r["benchmark_id"] == "BENCH_17_COX_BRESLOW_ANALYTICAL"][0]
        assert cox_ana["validation_method"] == ValidationMethod.ANALYTICAL_GROUND_TRUTH.value
        assert cox_ana["status"] == BenchmarkStatus.PASS.value

    def test_ordinal_and_cox_wrapper_internal_consistency(self):
        ologit_res = [r for r in self.results if r["benchmark_id"] == "BENCH_18_ORDERED_LOGIT"][0]
        assert ologit_res["validation_method"] == ValidationMethod.INTERNAL_CONSISTENCY.value
        assert ologit_res["status"] == BenchmarkStatus.PASS.value

        cox_ph = [r for r in self.results if r["benchmark_id"] == "BENCH_19_COX_PHREG"][0]
        assert cox_ph["validation_method"] == ValidationMethod.INTERNAL_CONSISTENCY.value
        assert cox_ph["status"] == BenchmarkStatus.PASS.value

    def test_twfe_decomposition_identity(self):
        bacon_res = [r for r in self.results if r["benchmark_id"] == "BENCH_13_BACONDECOMP"][0]
        assert bacon_res["coefficient_comparison"]["max_abs_diff"] < 1e-12
        assert bacon_res["standard_error_comparison"] is None
        assert bacon_res["additional_diagnostics"]["weight_normalization_error"] < 1e-8
        assert bacon_res["validation_method"] == ValidationMethod.ANALYTICAL_GROUND_TRUTH.value

    def test_panel_fe_precision(self):
        fe_res = [r for r in self.results if r["benchmark_id"] == "BENCH_09_PANEL_FE"][0]
        if fe_res["status"] == BenchmarkStatus.SKIPPED.value:
            pytest.skip("linearmodels not available")
        assert fe_res["coefficient_comparison"]["max_abs_diff"] < 1e-12

    def test_kaplan_meier_analytical_precision(self):
        km_res = [r for r in self.results if r["benchmark_id"] == "BENCH_12_KAPLAN_MEIER"][0]
        assert km_res["coefficient_comparison"]["max_abs_diff"] < 1e-12
        assert km_res["standard_error_comparison"]["max_abs_diff"] < 1e-6
        assert km_res["additional_diagnostics"]["risk_set_match"] is True

    def test_drdid_simulation_classification(self):
        drdid_res = [r for r in self.results if r["benchmark_id"] == "BENCH_14_DRDID"][0]
        assert drdid_res["validation_method"] == ValidationMethod.SIMULATION_GROUND_TRUTH.value
        assert drdid_res["coefficient_comparison"]["max_abs_diff"] < 0.15
