"""
Stage 42 — Doubly Robust Difference-in-Differences (DRDID) Monte Carlo Validation
Validates:
1. Monte Carlo evaluation (50 replications) of all 4 specification regimes:
   - Scenario A: Both propensity score and outcome models correctly specified.
   - Scenario B: Only propensity score correctly specified (OR misspecified).
   - Scenario C: Only outcome regression correctly specified (PS misspecified).
   - Scenario D: Both models misspecified.
2. Reports and verifies bias, empirical standard error, analytical standard error,
   and 95% nominal confidence interval coverage rate.
3. Validates independent repeated cross-section simulation with distinct units
   and unequal sample sizes across periods.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import root
from scipy.special import expit

from engine.causal_modern import compute_drdid


def test_drdid_monte_carlo_double_robustness_battery():
    """
    Runs a 50-replication Monte Carlo experiment evaluating the double-robustness
    property and asymptotic coverage of the Sant'Anna & Zhao (2020) panel estimator.
    """
    true_att = 3.0
    n_units = 800
    n_reps = 50

    results = {
        "both": {"atts": [], "ses": [], "covered": []},
        "or_only": {"atts": [], "ses": [], "covered": []},
        "ps_only": {"atts": [], "ses": [], "covered": []},
        "both_misspecified": {"atts": [], "ses": [], "covered": []}
    }

    for rep in range(n_reps):
        rng = np.random.RandomState(rep)
        rows = []
        for uid in range(1, n_units + 1):
            x = float(rng.normal(1.0, 0.5))
            x_sq = x ** 2
            # True non-linear propensity model
            z = -1.2 + 0.8 * x + 0.5 * x_sq
            ps_val = 1.0 / (1.0 + np.exp(-z))
            is_treated = 1 if rng.uniform(0, 1) < ps_val else 0

            for tid in [0, 1]:
                d_eff = is_treated if tid == 1 else 0
                # True time-varying non-linear outcome trend
                y = 5.0 + 3.0 * d_eff + 1.5 * x + 0.8 * tid + (2.5 * tid * x_sq) + float(rng.normal(0, 0.2))
                rows.append({"id": uid, "t": tid, "treated": is_treated, "y": y, "x": x, "x_sq": x_sq})
        df = pd.DataFrame(rows)

        # 1. Both correct
        r_both = compute_drdid(df, "y", "treated", "t", ps_covars=["x", "x_sq"], or_covars=["x", "x_sq"], unitvar="id")
        results["both"]["atts"].append(r_both["att"])
        results["both"]["ses"].append(r_both["se"])
        results["both"]["covered"].append(r_both["ci_lower"] <= true_att <= r_both["ci_upper"])

        # 2. Only OR correct (PS misspecified by omitting x_sq)
        r_or = compute_drdid(df, "y", "treated", "t", ps_covars=["x"], or_covars=["x", "x_sq"], unitvar="id")
        results["or_only"]["atts"].append(r_or["att"])
        results["or_only"]["ses"].append(r_or["se"])
        results["or_only"]["covered"].append(r_or["ci_lower"] <= true_att <= r_or["ci_upper"])

        # 3. Only PS correct (OR misspecified by omitting x_sq)
        r_ps = compute_drdid(df, "y", "treated", "t", ps_covars=["x", "x_sq"], or_covars=["x"], unitvar="id")
        results["ps_only"]["atts"].append(r_ps["att"])
        results["ps_only"]["ses"].append(r_ps["se"])
        results["ps_only"]["covered"].append(r_ps["ci_lower"] <= true_att <= r_ps["ci_upper"])

        # 4. Both misspecified (both omit x_sq)
        r_none = compute_drdid(df, "y", "treated", "t", ps_covars=["x"], or_covars=["x"], unitvar="id")
        results["both_misspecified"]["atts"].append(r_none["att"])
        results["both_misspecified"]["ses"].append(r_none["se"])
        results["both_misspecified"]["covered"].append(r_none["ci_lower"] <= true_att <= r_none["ci_upper"])

    # Summary statistics
    summary = {}
    for key, d in results.items():
        bias = float(np.mean(d["atts"]) - true_att)
        emp_se = float(np.std(d["atts"]))
        mean_est_se = float(np.mean(d["ses"]))
        cov_rate = float(np.mean(d["covered"]))
        summary[key] = {
            "bias": bias,
            "emp_se": emp_se,
            "mean_est_se": mean_est_se,
            "coverage": cov_rate
        }

    # Acceptance Criteria Verification:
    # A. Scenarios with at least one model correctly specified have negligible bias (< 0.10)
    assert abs(summary["both"]["bias"]) < 0.10, f"Both correct bias too large: {summary['both']['bias']}"
    assert abs(summary["or_only"]["bias"]) < 0.10, f"OR only bias too large: {summary['or_only']['bias']}"
    assert abs(summary["ps_only"]["bias"]) < 0.10, f"PS only bias too large: {summary['ps_only']['bias']}"

    # B. Coverage near nominal 95% (>= 85% for 50 reps)
    assert summary["both"]["coverage"] >= 0.85, f"Both coverage too low: {summary['both']['coverage']}"
    assert summary["or_only"]["coverage"] >= 0.85, f"OR coverage too low: {summary['or_only']['coverage']}"
    assert summary["ps_only"]["coverage"] >= 0.85, f"PS coverage too low: {summary['ps_only']['coverage']}"

    # C. Both misspecified exhibits clearly elevated bias
    assert abs(summary["both_misspecified"]["bias"]) > 0.12, (
        f"Misspecified model did not exhibit expected bias: {summary['both_misspecified']['bias']}"
    )

    # D. Analytical SE is consistent with empirical SE (within 35%)
    for key in ["both", "or_only", "ps_only"]:
        ratio = summary[key]["mean_est_se"] / summary[key]["emp_se"]
        assert 0.65 <= ratio <= 1.35, f"SE ratio out of range for {key}: {ratio}"


def test_drdid_genuine_repeated_cross_section_simulation():
    """
    Validates repeated cross-sectional DRDID where units in t=0 and t=1 are distinct
    and periods have different sample sizes.
    """
    rng = np.random.RandomState(42)
    n0 = 320
    n1 = 410

    rows = []
    # Pre-period t=0
    for i in range(n0):
        x = float(rng.normal(1.0, 0.5))
        ps = 1.0 / (1.0 + np.exp(-(-0.5 + 0.8 * x)))
        d = 1 if rng.uniform(0, 1) < ps else 0
        y = 5.0 + 1.5 * x + float(rng.normal(0, 0.2))
        rows.append({"id": f"t0_{i}", "t": 0, "treated": d, "x": x, "y": y})

    # Post-period t=1
    for i in range(n1):
        x = float(rng.normal(1.0, 0.5))
        ps = 1.0 / (1.0 + np.exp(-(-0.5 + 0.8 * x)))
        d = 1 if rng.uniform(0, 1) < ps else 0
        y = 5.0 + 3.0 * d + 1.5 * x + 0.8 + float(rng.normal(0, 0.2))
        rows.append({"id": f"t1_{i}", "t": 1, "treated": d, "x": x, "y": y})

    df_rc = pd.DataFrame(rows)
    r = compute_drdid(df_rc, "y", "treated", "t", covars=["x"], unitvar=None)

    assert r["data_type"] == "repeated_cross_section"
    assert r["n_obs"] == n0 + n1
    assert r["n_treated"] > 0
    assert r["n_control"] > 0
    assert abs(r["att"] - 3.0) < 0.15
    assert r["se"] > 0
    assert r["ci_lower"] < r["att"] < r["ci_upper"]


def test_repeated_cross_section_inference_matches_four_cell_contrast():
    rows = []
    for treated, post, count in [(0, 0, 95), (0, 1, 105), (1, 0, 90), (1, 1, 110)]:
        rows.extend((treated, post, 1 + post + 3 * treated * post + error)
                    for error in [-2, -1, 0, 1, 2] * (count // 5))
    data = pd.DataFrame(rows, columns=["treated", "post", "y"])
    # With intercepts alone, the ATT is a difference of four independent cell means.
    cells = data.groupby(["treated", "post"])["y"]
    expected_se = np.sqrt((cells.var(ddof=0) / cells.size()).sum())

    original = compute_drdid(data, "y", "treated", "post")
    shifted = compute_drdid(data.assign(y=data["y"] + 100 * data["treated"]), "y", "treated", "post")
    for result in (original, shifted):
        assert result["att"] == pytest.approx(3.0)
        assert result["se"] == pytest.approx(expected_se, abs=1e-12)
        assert result["ci_lower"] == pytest.approx(3.0 - 1.96 * expected_se)
        assert result["ci_upper"] == pytest.approx(3.0 + 1.96 * expected_se)


@pytest.mark.parametrize("trim_level", [0.995, 0.8])
def test_repeated_cross_section_inference_matches_weighted_refits(trim_level):
    rng = np.random.default_rng(42)
    post = np.repeat([0, 1], [40, 50])
    x = rng.normal(size=post.size)
    treated = rng.binomial(1, expit(0.3 + x))
    y = 2.5 * (treated * post) + (1 + post) * x**2 + rng.normal(size=post.size)
    data = pd.DataFrame({"y": y, "treated": treated, "post": post, "x": x})
    design = np.column_stack([np.ones(len(x)), x])

    def refit(weights):
        def score(params):
            return design.T @ (weights * (treated - expit(design @ params)))

        fitted = root(score, [0.0, 0.0])
        assert fitted.success
        raw_ps = expit(design @ fitted.x)
        propensity = np.clip(raw_ps, 1 - trim_level, trim_level)
        att = 0.0
        for period, direction in [(0, -1), (1, 1)]:
            controls = (post == period) & (treated == 0)
            treated_cell = (post == period) & (treated == 1)
            sqrt_weights = np.sqrt(weights[controls])
            beta = np.linalg.lstsq(design[controls] * sqrt_weights[:, None],
                                   y[controls] * sqrt_weights, rcond=None)[0]
            residuals = y - design @ beta
            control_weights = weights[controls] * propensity[controls] / (1 - propensity[controls])
            att += direction * (np.average(residuals[treated_cell], weights=weights[treated_cell])
                                - np.average(residuals[controls], weights=control_weights))
        return att, raw_ps

    weights = np.ones(len(data))
    reference_att, raw_ps = refit(weights)
    clipped = (raw_ps < 1 - trim_level) | (raw_ps > trim_level)
    assert np.any(clipped[treated == 0]) == (trim_level == 0.8)

    # Perturb each observation's mass and refit both nuisance models independently.
    step = 1e-4
    derivatives = []
    for index in range(len(data)):
        weights[index] = 1.0 + step
        upper_att, _ = refit(weights)
        weights[index] = 1.0 - step
        lower_att, _ = refit(weights)
        weights[index] = 1.0
        derivatives.append((upper_att - lower_att) / (2 * step))

    result = compute_drdid(data, "y", "treated", "post", covars=["x"], trim_level=trim_level)
    assert result["att"] == pytest.approx(reference_att, abs=1e-6)
    assert result["se"] == pytest.approx(np.linalg.norm(derivatives), abs=1e-6)
