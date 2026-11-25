import os
import numpy as np
import pandas as pd
import pytest
from engine.interpreter import Interpreter
from engine.dataset import Dataset
from engine.plugins_manager import PluginsManager
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.api import VAR
from statsmodels.duration.survfunc import SurvfuncRight


@pytest.fixture
def clean_interp(tmp_path):
    """Fixture providing an interpreter with a clean plugins directory in tmp_path."""
    plugins_dir = str(tmp_path / "plugins")
    os.makedirs(plugins_dir, exist_ok=True)
    dataset = Dataset()
    interp = Interpreter(dataset)
    interp.custom_commands.clear()
    pm = PluginsManager(interp, plugins_dir=plugins_dir, enabled=True)
    interp.plugins_mgr = pm
    return interp, plugins_dir


def test_dofile_cannot_enable_script_plugins_setting(clean_interp, tmp_path):
    """
    Test Item 1: 'set allow_script_plugins on' must be refused when run from a
    do-file or script. It must leave the setting OFF and block plugin execution.
    """
    interp, plugins_dir = clean_interp

    # Create a plugin named 'evil'
    evil_file = os.path.join(plugins_dir, "evil.py")
    with open(evil_file, "w", encoding="utf-8") as f:
        f.write('def command_evil(interp, line):\n    return "PWNED"\n')

    # Verify initially disabled
    assert not interp.plugins_mgr.get_allow_script_plugins()
    plugin_names = [p["name"] for p in interp.plugins_mgr.list_plugins()]
    assert "evil" in plugin_names
    assert not interp.plugins_mgr.is_enabled("evil")

    # Create a do-file that attempts to weaken safety and enable the plugin
    dofile = tmp_path / "attack.do"
    dofile.write_text("set allow_script_plugins on\npkg enable evil\n", encoding="utf-8")

    # Run do-file
    res = interp.execute(f'do "{dofile}"')

    # It must fail or report the refusal
    assert not interp.plugins_mgr.get_allow_script_plugins()
    assert not interp.plugins_mgr.is_enabled("evil")
    assert "cannot be executed from a do-file" in res.output or "Security restriction" in res.output

    # The evil command must not be registered or runnable
    res_cmd = interp.execute("evil")
    assert not res_cmd.ok


def test_tsfilter_vs_statsmodels_hpfilter(clean_interp):
    """
    Test Item 3: Validate tsfilter hp against statsmodels.tsa.filters.hp_filter.hpfilter.
    """
    interp, _ = clean_interp

    # Synthetic GDP-like trend + cycle series
    rng = np.random.default_rng(42)
    n = 80
    time = np.arange(1, n + 1)
    true_trend = 100.0 + 1.5 * time
    true_cycle = 5.0 * np.sin(2 * np.pi * time / 16.0) + rng.normal(0, 1, n)
    y = true_trend + true_cycle

    df = pd.DataFrame({"time": time, "gdp": y})
    interp.dataset.df = df

    res_ts = interp.execute("tsset time")
    assert res_ts.ok

    res_filt = interp.execute("tsfilter hp cycle_pot = gdp, smooth(1600) trend(trend_pot)")
    assert res_filt.ok

    # Benchmark directly with statsmodels
    cycle_sm, trend_sm = hpfilter(y, lamb=1600.0)

    np.testing.assert_allclose(interp.dataset.df["cycle_pot"].to_numpy(dtype=float), cycle_sm, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(interp.dataset.df["trend_pot"].to_numpy(dtype=float), trend_sm, rtol=1e-7, atol=1e-7)


def test_var_vs_statsmodels_var(clean_interp):
    """
    Test Item 3: Validate VAR against statsmodels.tsa.api.VAR.
    """
    interp, _ = clean_interp

    rng = np.random.default_rng(123)
    n = 120
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    e = rng.normal(0, 1, size=(n, 2))
    for t in range(1, n):
        y1[t] = 0.5 * y1[t-1] + 0.2 * y2[t-1] + e[t, 0]
        y2[t] = 0.1 * y1[t-1] + 0.6 * y2[t-1] + e[t, 1]

    df = pd.DataFrame({"time": np.arange(1, n + 1), "y1": y1, "y2": y2})
    interp.dataset.df = df

    interp.execute("tsset time")
    res_var = interp.execute("var y1 y2, lags(1)")
    assert res_var.ok

    # Statsmodels reference
    sm_data = df[["y1", "y2"]]
    sm_var = VAR(sm_data)
    sm_res = sm_var.fit(maxlags=1)

    pot_aic = interp.registers.get_e("aic")
    pot_bic = interp.registers.get_e("bic")
    pot_b = interp.registers.b

    # Compare AIC and BIC with statsmodels
    assert abs(pot_aic - sm_res.aic) < 1e-4
    assert abs(pot_bic - sm_res.bic) < 1e-4

    # Check coefficients
    sm_params_y1 = sm_res.params["y1"].to_dict()
    assert abs(pot_b.get("y1:L1.y1") - sm_params_y1["L1.y1"]) < 1e-4
    assert abs(pot_b.get("y1:L1.y2") - sm_params_y1["L1.y2"]) < 1e-4

    sm_params_y2 = sm_res.params["y2"].to_dict()
    assert abs(pot_b.get("y2:L1.y1") - sm_params_y2["L1.y1"]) < 1e-4
    assert abs(pot_b.get("y2:L1.y2") - sm_params_y2["L1.y2"]) < 1e-4


def test_svy_regress_vs_cluster_stratified_linearization(clean_interp):
    """
    Test Item 3: Validate svy: regress against exact cluster-stratified Taylor series
    linearization formula (Lumley 2004 / Cochran 1977 / R survey::svyglm).
    """
    interp, _ = clean_interp

    # Setup stratified cluster dataset
    data = [
        # strata, psu, weight, x1, x2, y
        [1, 101, 2.5, 1.2, 3.4, 10.5],
        [1, 101, 2.5, 1.5, 3.1, 11.2],
        [1, 102, 3.0, 2.1, 4.0, 14.1],
        [1, 102, 3.0, 2.4, 3.8, 14.9],
        [2, 201, 1.5, 0.8, 2.2, 7.8],
        [2, 201, 1.5, 0.9, 2.5, 8.4],
        [2, 202, 2.0, 1.8, 2.9, 12.0],
        [2, 202, 2.0, 1.7, 3.1, 11.8],
    ]
    cols = ["strata", "psu", "weight", "x1", "x2", "y"]
    df = pd.DataFrame(data, columns=cols)
    interp.dataset.df = df

    res_set = interp.execute("svyset psu [pw=weight], strata(strata)")
    assert res_set.ok

    res_reg = interp.execute("svy: regress y x1 x2")
    assert res_reg.ok

    # Independent computation of WLS point estimates and Taylor linearization SEs:
    X = np.column_stack([df[["x1", "x2"]].to_numpy(), np.ones(len(df))])
    y = df["y"].to_numpy()
    w = df["weight"].to_numpy()
    W = np.diag(w)

    XtWX = X.T @ W @ X
    inv_XtWX = np.linalg.inv(XtWX)
    beta_ref = inv_XtWX @ (X.T @ (w * y))

    residuals = y - X @ beta_ref
    u = (w * residuals)[:, None] * X  # n x k score contributions

    # Between-cluster deviance within strata:
    V_u = np.zeros((3, 3))
    for h in [1, 2]:
        idx_h = df["strata"] == h
        psus_in_h = df.loc[idx_h, "psu"].unique()
        n_h = len(psus_in_h)
        z_clusters = []
        for p in psus_in_h:
            idx_p = idx_h & (df["psu"] == p)
            z_clusters.append(np.sum(u[idx_p], axis=0))
        z_clusters = np.array(z_clusters)
        z_mean = np.mean(z_clusters, axis=0)
        dev = z_clusters - z_mean
        V_u += (n_h / (n_h - 1)) * (dev.T @ dev)

    vcov_ref = inv_XtWX @ V_u @ inv_XtWX
    se_ref = np.sqrt(np.diag(vcov_ref))

    b_pot = interp.registers.b
    se_pot = interp.registers.se

    np.testing.assert_allclose([b_pot["x1"], b_pot["x2"], b_pot["_cons"]], beta_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose([se_pot["x1"], se_pot["x2"], se_pot["_cons"]], se_ref, rtol=1e-6, atol=1e-6)


def test_sts_list_vs_kaplan_meier_benchmark(clean_interp):
    """
    Test Item 3: Validate sts list against Kaplan-Meier product-limit formula
    and statsmodels.duration.survfunc.SurvfuncRight.
    """
    interp, _ = clean_interp

    # Standard survival dataset with right-censoring
    times = [1, 2, 3, 4, 5, 6]
    failed = [1, 0, 1, 1, 0, 1]
    df = pd.DataFrame({"study_time": times, "event": failed})
    interp.dataset.df = df

    res_stset = interp.execute("stset study_time, failure(event)")
    assert res_stset.ok

    res_sts = interp.execute("sts list")
    assert res_sts.ok
    assert "Beg." in res_sts.output
    assert "Survivor" in res_sts.output

    # Compare with statsmodels SurvfuncRight
    sf = SurvfuncRight(times, failed)
    for probability in sf.surv_prob:
        assert f"{probability:.4f}" in res_sts.output

    # S(1) = 5/6 = 0.8333
    # At t=2, 1 censored -> at risk at t=3 is 4.
    # S(3) = 5/6 * (1 - 1/4) = 5/6 * 3/4 = 5/8 = 0.6250
    # At t=4, at risk is 3. Fail = 1 -> S(4) = 5/8 * (1 - 1/3) = 5/8 * 2/3 = 5/12 = 0.4167
    # At t=5, 1 censored -> at risk at t=6 is 1. Fail = 1 -> S(6) = 0.0
    assert "0.8333" in res_sts.output
    assert "0.6250" in res_sts.output
    assert "0.4167" in res_sts.output
