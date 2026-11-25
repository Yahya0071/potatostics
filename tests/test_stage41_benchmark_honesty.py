"""
Test Stage 41: Benchmark Honesty, Genuinely Independent Ground Truths, and Packaging Integrity

Verifies:
1. Benchmark honesty: Any estimator that delegates to or imports from a reference library
   (e.g., hpfilter, VAR, PHReg, OrderedModel) must NOT be labeled INDEPENDENT_EXTERNAL.
   It must be labeled INTERNAL_CONSISTENCY.
2. Genuinely independent checks:
   - HP filter vs closed-form solve: (I + lambda K'K) trend = y (ANALYTICAL_GROUND_TRUTH).
   - VAR coefficients vs equation-by-equation OLS computed directly (ANALYTICAL_GROUND_TRUTH).
   - Cox partial likelihood (Breslow) vs direct numerical optimization of partial log-likelihood with scipy (ANALYTICAL_GROUND_TRUTH).
3. Report timestamp generation: ensures timestamp reflects current execution time, not a stored/static value.
"""

import ast
import inspect
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.linalg

from engine.benchmarks import BenchmarkRunner, ValidationMethod
from engine.estimation import (
    compute_regression, compute_logit, compute_panel_fe, compute_ivregress_2sls
)
from engine.advanced_econometrics import (
    compute_probit, compute_poisson, compute_qreg,
    compute_var, compute_sts_list, compute_stcox, compute_ologit
)
from engine.timeseries import compute_hp_filter
from engine.causal_modern import compute_bacondecomp, compute_drdid

ESTIMATOR_FUNC_MAP = {
    "ols": compute_regression,
    "ols_hc1": compute_regression,
    "ols_cluster": compute_regression,
    "logit": compute_logit,
    "probit": compute_probit,
    "poisson": compute_poisson,
    "qreg": compute_qreg,
    "ivregress_2sls": compute_ivregress_2sls,
    "xtreg_fe": compute_panel_fe,
    "tsfilter_hp": compute_hp_filter,
    "var": compute_var,
    "sts_list": compute_sts_list,
    "bacondecomp": compute_bacondecomp,
    "drdid": compute_drdid,
    "stcox": compute_stcox,
    "ologit": compute_ologit,
}


def test_benchmark_honesty_no_wrapped_imports_labeled_independent_external():
    """
    Fails if a benchmark's estimator function or module imports the same function/library
    used as its reference, but is labeled INDEPENDENT_EXTERNAL.
    Such benchmarks must be labeled INTERNAL_CONSISTENCY.
    """
    runner = BenchmarkRunner()
    results = runner.run_all()

    violations = []
    for r in results:
        bench_id = r["benchmark_id"]
        est_name = r["estimator_name"]
        val_method = r.get("validation_method")
        ref_impl = r.get("reference_implementation", "").lower()

        if val_method == ValidationMethod.INDEPENDENT_EXTERNAL.value:
            fn = ESTIMATOR_FUNC_MAP.get(est_name)
            if fn is not None:
                src = inspect.getsource(fn)
                tree = ast.parse(src)
                imported_modules = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imported_modules.append(alias.name.lower())
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imported_modules.append(node.module.lower())

                # Check if estimator imports statsmodels or linearmodels while claiming external independence
                for imp in imported_modules:
                    if ("statsmodels" in imp and "statsmodels" in ref_impl) or \
                       ("linearmodels" in imp and "linearmodels" in ref_impl):
                        violations.append(
                            f"Benchmark '{bench_id}' (estimator '{est_name}') imports '{imp}' "
                            f"which matches reference '{ref_impl}', but is labeled INDEPENDENT_EXTERNAL. "
                            f"Must be reclassified as INTERNAL_CONSISTENCY."
                        )

    assert not violations, "\n".join(violations)


def test_independent_hp_filter_closed_form_ground_truth():
    """
    Validates HP filter directly against the closed-form central difference identity:
      (I + lambda K' K) trend = y
    where K is the (N-2) x N second-difference matrix.
    """
    rng = np.random.RandomState(42)
    n = 120
    t = np.linspace(0, 10, n)
    y = 30 + 1.5 * t + 4 * np.sin(t) + rng.normal(0, 1, n)

    for lamb in [100.0, 1600.0, 14400.0]:
        pot_cycle, pot_trend = compute_hp_filter(y, lamb=lamb)

        # Direct mathematical solve of (I + lambda K' K) trend = y
        I = np.eye(n)
        K = np.zeros((n - 2, n))
        for i in range(n - 2):
            K[i, i] = 1.0
            K[i, i + 1] = -2.0
            K[i, i + 2] = 1.0

        A = I + lamb * (K.T @ K)
        analytical_trend = scipy.linalg.solve(A, y, assume_a="pos")
        analytical_cycle = y - analytical_trend

        # Scale-aware comparison accounting for condition number kappa(I + lambda K'K) approx 2.3e5
        abs_trend_err = float(np.max(np.abs(pot_trend - analytical_trend)))
        rel_trend_err = abs_trend_err / float(np.max(np.abs(y)))
        assert rel_trend_err < 1e-10 or abs_trend_err < 1e-8
        abs_cycle_err = float(np.max(np.abs(pot_cycle - analytical_cycle)))
        assert abs_cycle_err < 1e-8


def test_independent_var_equation_by_equation_ols_ground_truth():
    """
    Validates VAR(1) coefficients directly against self-contained equation-by-equation OLS:
      B = (X' X)^(-1) X' Y
    """
    rng = np.random.RandomState(42)
    n = 150
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    for i in range(1, n):
        y1[i] = 0.5 * y1[i - 1] + 0.2 * y2[i - 1] + rng.normal(0, 1)
        y2[i] = 0.1 * y1[i - 1] + 0.4 * y2[i - 1] + rng.normal(0, 1)
    df = pd.DataFrame({"y1": y1, "y2": y2})

    r = compute_var(df, ["y1", "y2"], lags=1)

    # Self-contained Equation-by-Equation OLS
    Y_eff = df.iloc[1:][["y1", "y2"]].to_numpy()
    X_lags = df.iloc[:-1][["y1", "y2"]].to_numpy()
    X_mat = np.column_stack([np.ones(len(X_lags)), X_lags])  # [const, L1.y1, L1.y2]

    # OLS solve for both equations
    XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    B_mat = XtX_inv @ (X_mat.T @ Y_eff)  # columns are equations y1, y2

    # y1 equation: const, L1.y1, L1.y2
    assert abs(r["b_dict"]["y1:_cons"] - B_mat[0, 0]) < 1e-10
    assert abs(r["b_dict"]["y1:L1.y1"] - B_mat[1, 0]) < 1e-10
    assert abs(r["b_dict"]["y1:L1.y2"] - B_mat[2, 0]) < 1e-10

    # y2 equation: const, L1.y1, L1.y2
    assert abs(r["b_dict"]["y2:_cons"] - B_mat[0, 1]) < 1e-10
    assert abs(r["b_dict"]["y2:L1.y1"] - B_mat[1, 1]) < 1e-10
    assert abs(r["b_dict"]["y2:L1.y2"] - B_mat[2, 1]) < 1e-10


def test_independent_cox_breslow_partial_likelihood_optimization():
    """
    Validates Cox proportional hazards model against direct numerical optimization
    of the Breslow partial log-likelihood using scipy.optimize.minimize:
      ln L(beta) = sum_{k=1}^K [ s_k' beta - d_k * ln( sum_{j in R_k} exp(x_j' beta) ) ]
    """
    # Create survival dataset
    rng = np.random.RandomState(42)
    n = 60
    x = rng.normal(0, 1, n)
    # Exponential hazard model
    haz = np.exp(0.8 * x)
    t_event = rng.exponential(1.0 / haz)
    t_censor = rng.uniform(0.5, 3.0, n)
    t_obs = np.minimum(t_event, t_censor)
    failed = (t_event <= t_censor).astype(int)

    df = pd.DataFrame({"time": t_obs, "failed": failed, "x": x})

    r = compute_stcox(df, "time", "failed", ["x"], method="breslow")
    pot_beta = r["coefs"][0]["coef"]

    # Direct numerical optimization of Breslow partial log-likelihood:
    order = np.argsort(t_obs)
    t_sorted = t_obs[order]
    d_sorted = failed[order]
    x_sorted = x[order]

    unique_death_times = np.sort(np.unique(t_sorted[d_sorted == 1]))

    def neg_breslow_loglik(beta):
        b = beta[0]
        xb = x_sorted * b
        exp_xb = np.exp(xb)
        loglik = 0.0
        for death_t in unique_death_times:
            # Events at death_t
            event_mask = (t_sorted == death_t) & (d_sorted == 1)
            d_k = np.sum(event_mask)
            s_k = np.sum(x_sorted[event_mask])

            # Risk set: all units with t >= death_t
            risk_mask = (t_sorted >= death_t)
            sum_risk = np.sum(exp_xb[risk_mask])

            loglik += s_k * b - d_k * np.log(sum_risk)
        return -loglik

    opt = scipy.optimize.minimize(neg_breslow_loglik, [0.0], method="BFGS")
    analytical_beta = float(opt.x[0])

    # Potatostics stcox must match direct Breslow optimization to high accuracy
    assert abs(pot_beta - analytical_beta) < 1e-5


def test_benchmark_report_timestamp_comes_from_run(tmp_path):
    """
    Verifies that benchmark results and markdown report timestamps are generated
    dynamically from the current execution run, rather than using a static stored timestamp.
    """
    import datetime
    before_run = datetime.datetime.now(datetime.timezone.utc)
    runner = BenchmarkRunner(output_dir=str(tmp_path))
    runner.run_all()
    after_run = datetime.datetime.now(datetime.timezone.utc)

    assert runner.run_timestamp is not None
    run_dt = datetime.datetime.fromisoformat(runner.run_timestamp)
    # The recorded run_dt must fall between before_run and after_run
    assert before_run <= run_dt <= after_run

    md_file = tmp_path / "benchmark_report.md"
    assert md_file.exists()
    content = md_file.read_text(encoding="utf-8")
    formatted_ts = run_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    assert formatted_ts in content
    assert "**Total Benchmarks**: 19" in content
