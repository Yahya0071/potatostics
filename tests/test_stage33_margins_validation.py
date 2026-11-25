import os
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from engine.interpreter import Interpreter
from engine.dataset import Dataset


@pytest.fixture
def interp():
    dataset = Dataset()
    return Interpreter(dataset)


def test_margins_vs_statsmodels_and_williams2012(interp):
    """
    Test Item: Validate margins, dydx(*) against Williams (2012) /
    statsmodels.discrete.discrete_model.LogitResults.get_margeff.
    """
    auto_path = os.path.join(os.path.dirname(__file__), "data", "auto.dta")
    res_use = interp.execute(f'use "{auto_path}", clear')
    assert res_use.ok

    # Fit logit foreign price mpg
    res_log = interp.execute("logit foreign price mpg")
    assert res_log.ok

    # Run margins, dydx(*)
    res_mar = interp.execute("margins, dydx(*)")
    assert res_mar.ok

    # Benchmark directly with statsmodels Logit get_margeff(at='overall', method='dydx')
    df = interp.dataset.df
    y = df["foreign"].to_numpy(dtype=float)
    X = sm.add_constant(df[["price", "mpg"]].to_numpy(dtype=float))
    sm_mod = sm.Logit(y, X).fit(disp=False)
    sm_me = sm_mod.get_margeff(at="overall", method="dydx")

    ame_sm = sm_me.margeff
    se_sm = sm_me.margeff_se

    # In Potatostics margins output
    assert "price" in res_mar.output
    assert "mpg" in res_mar.output

    # Compare numeric AME values and Delta-method standard errors
    # mpg is at index 1
    assert "0.0376" in res_mar.output
    assert "0.0074" in res_mar.output

    # Also verify stored return values in r()
    b_mpg = interp.registers.get_r("b_mpg")
    se_mpg = interp.registers.get_r("se_mpg")
    assert abs(b_mpg - ame_sm[1]) < 1e-5
    assert abs(se_mpg - se_sm[1]) < 1e-5


def test_cite_margins_williams2012(interp):
    """
    Test that 'cite margins' outputs Richard Williams (2012) citation and DOI.
    """
    res = interp.execute("cite margins")
    assert res.ok
    assert "Williams" in res.output
    assert "2012" in res.output
    assert "10.1177/1536867X1201200209" in res.output
    assert "@article{williams2012using" in res.output


def test_multilevel_margins_use_base_category(interp):
    counts = [3, 10, 17]
    y = np.concatenate([np.arange(20) < k for k in counts]).astype(float)
    interp.dataset.load_dataframe(pd.DataFrame(dict(y=y, g=np.repeat([1, 2, 3], 20))))
    assert interp.execute("logit y i.g").ok
    assert interp.execute("margins, dydx(i.g)").ok
    p0 = counts[0] / 20
    for level, k in enumerate(counts[1:], start=2):
        p = k / 20
        assert interp.registers.get_r(f"b_{level}.g") == pytest.approx(p - p0)
        assert interp.registers.get_r(f"se_{level}.g") == pytest.approx(
            np.sqrt((p * (1-p) + p0 * (1-p0)) / 20))


def test_interactions_do_not_report_design_column_coefficients_as_effects(interp):
    x = np.tile(np.arange(5.0), 2)
    group = np.repeat([0, 1], 5)
    noise = np.tile([0.2, -0.4, 0.4, -0.4, 0.2], 2)
    y = 1 + 2*x + 5*group + 4*x*group + noise
    interp.dataset.load_dataframe(pd.DataFrame(dict(x=x, g=group, y=y)))
    assert interp.execute("regress y c.x##i.g").ok
    for variable in ("c.x", "x", "i.g", "*"):
        result = interp.execute(f"margins, dydx({variable})")
        assert not result.ok
        assert "interaction" in result.output.lower()


def test_logit_margins_use_estimation_design_after_sort_and_reopen(interp, tmp_path):
    from engine.session_manager import save_session_bundle, load_session_bundle
    from scipy.special import expit

    x = np.r_[np.tile([-2., -1., 0., 1., 2.], 5), np.arange(-100., -75.)]
    y = [int(repeat < k) for repeat in range(5) for k in (1, 2, 2, 3, 4)]
    y.extend([0, 1, 0, 1, 0] * 5)
    interp.dataset.load_dataframe(pd.DataFrame(dict(x=x, y=y, use=np.arange(50) < 25)))
    assert interp.execute("logit y x if use == 1").ok
    slope, intercept = interp.estimates.last_model["beta"]
    p = expit(slope * x[:25] + intercept)
    expected = np.mean(slope * p * (1-p))
    assert interp.execute("margins, dydx(x)").ok
    original_se = interp.registers.get_r("se_x")
    assert interp.execute("sort x").ok
    session_path = str(tmp_path / "sorted.potproj")
    save_session_bundle(session_path, interp=interp)
    reopened = Interpreter(Dataset())
    load_session_bundle(session_path, reopened)
    for fitted in (interp, reopened):
        assert fitted.execute("margins, dydx(x)").ok
        assert fitted.registers.get_r("b_x") == pytest.approx(expected)
        assert fitted.registers.get_r("se_x") == pytest.approx(original_se)


def test_standardized_residuals_and_stored_estimate_registers(interp, tmp_path):
    from engine.session_manager import save_session_bundle, load_session_bundle

    x = np.arange(8.0)
    y = np.array([1., 4., 2., 6., 9., 8., 12., 11.])
    interp.dataset.load_dataframe(pd.DataFrame(dict(x=x, y=y)))
    assert interp.execute("regress y x in 2/7, robust").ok
    before = interp.registers.e_scalars.copy()
    slope = interp.registers.b["x"]
    assert interp.execute("estimates store first").ok
    assert interp.execute("regress x y").ok
    session_path = str(tmp_path / "models.potproj")
    save_session_bundle(session_path, interp=interp)
    reopened = Interpreter(Dataset())
    load_session_bundle(session_path, reopened)
    assert reopened.execute("estimates restore first").ok
    assert reopened.registers.e_scalars == before
    assert reopened.registers.b["x"] == slope
    assert reopened.registers.get_e("depvar") == "y"
    assert reopened.registers.get_e("vce") == "robust"
    np.testing.assert_allclose(reopened.registers.get_e("V"), reopened.estimates.last_model["vcov"])
    assert not reopened.execute("estimates restore missing").ok
    assert reopened.execute("predict z, rstandard").ok
    X = np.c_[x[1:7], np.ones(6)]
    hat = X @ np.linalg.pinv(X)
    residual = (np.eye(6)-hat) @ y[1:7]
    expected = residual / np.sqrt((residual @ residual / 4) * (1-np.diag(hat)))
    np.testing.assert_allclose(reopened.dataset.df.z[1:7], expected)
    assert reopened.dataset.df.z.iloc[[0, 7]].isna().all()
    assert reopened.execute("replace x = 99 in 4").ok
    assert not reopened.execute("predict stale, rstandard").ok
    assert "stale" not in reopened.dataset.df
    assert reopened.execute("generate binary = mod(x,2)").ok
    assert reopened.execute("logit binary x").ok
    assert not reopened.execute("predict bad, rstandard").ok
    assert "bad" not in reopened.dataset.df


def test_all_missing_summary_returns_undefined_statistics(interp):
    interp.dataset.load_dataframe(pd.DataFrame({"x": [np.nan, np.nan]}))
    result = interp.execute("summarize x, detail")
    assert result.ok
    assert interp.registers.get_r("N") == 0
    assert interp.registers.get_r("sum") == 0
    for stat in ("mean", "sd", "Var", "min", "max", "p50", "kurtosis"):
        assert np.isnan(interp.registers.get_r(stat))
    assert "0.0000" not in result.output


def _audit_sample(interp):
    x = np.repeat(np.arange(-2., 3.), 10)
    y = np.concatenate([np.arange(10) < k for k in (1, 3, 5, 7, 9)]).astype(float)
    interp.dataset.load_dataframe(pd.DataFrame(dict(x=x, y=y, group=x)))


def test_audit_redundant_wald(interp):
    _audit_sample(interp)
    assert interp.execute("regress y x").ok
    assert interp.execute("test x = 0").ok
    p_single = interp.registers.get_r("p")
    assert interp.execute("test (x = 0) (2*x = 0)").ok
    assert interp.registers.get_r("df") == 1
    assert interp.registers.get_r("p") == pytest.approx(p_single)
    assert not interp.execute("test (x = 0) (x = 1)").ok


@pytest.mark.parametrize("verb", ["regress", "logit"])
def test_audit_cluster_predictor(interp, verb):
    _audit_sample(interp)
    assert interp.execute(f"{verb} y x, vce(cluster group)").ok
    expected = interp.estimates.last_model["vcov"].copy()
    result = interp.execute(f"{verb} y x, vce(cluster x)")
    assert result.ok, result.output
    np.testing.assert_allclose(interp.estimates.last_model["vcov"], expected)


@pytest.mark.parametrize(("verb", "family", "mean_opt"), [
    ("logit", sm.families.Binomial(), "pr"),
    ("probit", sm.families.Binomial(link=sm.families.links.Probit()), "pr"),
    ("poisson", sm.families.Poisson(), "n"),
])
def test_audit_response_residuals(interp, verb, family, mean_opt):
    _audit_sample(interp)
    df = interp.dataset.df
    mu = sm.GLM(df.y, sm.add_constant(df.x), family=family).fit().fittedvalues
    assert interp.execute(f"{verb} y x").ok
    assert interp.execute(f"predict fitted, {mean_opt}").ok
    np.testing.assert_allclose(df.fitted, mu, rtol=1e-5)
    assert interp.execute("predict err, resid").ok
    np.testing.assert_allclose(df.err, df.y - mu, rtol=1e-5)


@pytest.mark.parametrize("opts", ["stdp", "resid xb", "n", "xb(foo)"])
def test_audit_predict_invalid_preserves_target(interp, opts):
    _audit_sample(interp)
    assert interp.execute("regress y x").ok
    expected = interp.dataset.df.y.copy()
    assert not interp.execute(f"predict y, {opts} replace").ok
    pd.testing.assert_series_equal(interp.dataset.df.y, expected)


@pytest.mark.parametrize("verb", ["poisson", "nbreg"])
@pytest.mark.parametrize("opt", ["exposure(group)", "offset(x)"])
def test_audit_count_offset_rejected(interp, verb, opt):
    _audit_sample(interp)
    assert interp.execute("regress y x").ok
    old_model = interp.estimates.last_model
    result = interp.execute(f"{verb} y x, {opt}")
    assert not result.ok and opt.split("(")[0] in result.output
    assert interp.estimates.last_model is old_model


@pytest.mark.parametrize("verb", ["use", "import delimited"])
def test_audit_panel_balance_and_replacement(interp, tmp_path, verb):
    df = pd.DataFrame({"id": [1, 1, 2, 2], "t": [1, 2, 1, 3]})
    interp.dataset.load_dataframe(df)
    assert "(unbalanced)" in interp.execute("xtset id t").output
    df.loc[3, "t"] = 2
    interp.dataset.load_dataframe(df)
    assert "(balanced)" in interp.execute("xtset id t").output
    assert not interp.execute(f'{verb} "{tmp_path / "missing.csv"}"').ok
    assert interp.execute("xtset").ok
    target = tmp_path / "other.csv"
    pd.DataFrame({"v": [1, 2]}).to_csv(target, index=False)
    assert interp.execute(f'{verb} "{target}", clear').ok
    assert not interp.execute("xtset").ok
