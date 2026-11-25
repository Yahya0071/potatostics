import numpy as np
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm
from engine.interpreter import Interpreter
from engine.dataset import Dataset


@pytest.fixture
def interp():
    dataset = Dataset()
    return Interpreter(dataset)


def test_brant_exact_formula_vs_paper(interp):
    """
    Validate estat brant against the exact algebraic specification in Brant (1990)
    Section 3 & Appendix:
      X^2 = (D beta)^T [D V(beta) D^T]^-1 (D beta)
      with degrees of freedom (k - 2) * p.
    """
    rng = np.random.default_rng(2026)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.uniform(-2, 2, n)

    # Ordinal outcome with 3 categories: 0, 1, 2
    latent = 1.0 * x1 - 0.5 * x2 + rng.logistic(0, 1, n)
    y = np.zeros(n, dtype=int)
    y[latent > -0.5] = 1
    y[latent > 1.2] = 2

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    interp.dataset.df = df

    # Fit ologit
    res_ol = interp.execute("ologit y x1 x2")
    assert res_ol.ok

    # Run estat brant
    res_brant = interp.execute("estat brant")
    assert res_brant.ok

    chi2_val = interp.registers.get_r("chi2")
    p_val = interp.registers.get_r("p")
    df_val = interp.registers.get_r("df")

    # (3 categories - 2) * 2 covariates = 2 degrees of freedom
    assert df_val == 2

    # Step-by-step independent calculation directly from Brant (1990) equations (3.1) & (3.2):
    X = np.column_stack([np.ones(n), x1, x2])
    z1 = (y > 0).astype(float)
    z2 = (y > 1).astype(float)

    logit1 = sm.Logit(z1, X).fit(disp=False)
    logit2 = sm.Logit(z2, X).fit(disp=False)

    b1 = logit1.params[1:]  # drop intercept
    b2 = logit2.params[1:]

    p1 = logit1.predict()
    p2 = logit2.predict()

    # W_jl diagonal matrices
    w11 = p1 * (1.0 - p1)
    w22 = p2 * (1.0 - p2)
    w12 = p2 * (1.0 - p1)  # z2=1 implies z1=1

    inv_xwx1 = np.linalg.pinv(X.T @ (w11[:, None] * X))
    inv_xwx2 = np.linalg.pinv(X.T @ (w22[:, None] * X))

    v11 = inv_xwx1
    v22 = inv_xwx2
    v12 = inv_xwx1 @ (X.T @ (w12[:, None] * X)) @ inv_xwx2
    v21 = v12.T

    # Drop intercepts to get slope covariance
    v_slopes = np.zeros((4, 4))
    v_slopes[0:2, 0:2] = v11[1:3, 1:3]
    v_slopes[2:4, 2:4] = v22[1:3, 1:3]
    v_slopes[0:2, 2:4] = v12[1:3, 1:3]
    v_slopes[2:4, 0:2] = v21[1:3, 1:3]

    # Contrast D = [I_2, -I_2] -> testing b1 - b2 = 0
    D = np.array([
        [-1.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0]
    ])
    beta_concat = np.concatenate([b1, b2])
    diff = D @ beta_concat
    v_diff = D @ v_slopes @ D.T

    chi2_ref = float(diff.T @ np.linalg.pinv(v_diff) @ diff)
    p_ref = float(1.0 - stats.chi2.cdf(chi2_ref, 2))

    assert abs(chi2_val - chi2_ref) < 1e-4
    assert abs(p_val - p_ref) < 1e-4


def test_brant_detail_option(interp):
    """
    Test that estat brant, detail displays the separate binary logit coefficients
    for each threshold cutpoint (similar to Standard brant, detail).
    """
    rng = np.random.default_rng(42)
    n = 150
    x = rng.normal(0, 1, n)
    latent = 0.8 * x + rng.logistic(0, 1, n)
    y = np.zeros(n, dtype=int)
    y[latent > -0.5] = 1
    y[latent > 1.0] = 2

    df = pd.DataFrame({"y": y, "x": x})
    interp.dataset.df = df
    interp.execute("ologit y x")

    # Call with detail
    res_det = interp.execute("estat brant, detail")
    assert res_det.ok
    # Must contain binary logit model breakdown
    assert "detail" in res_det.output.lower() or "y >" in res_det.output or "binary" in res_det.output.lower()
