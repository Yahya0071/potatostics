"""
tests/test_property_econometrics.py

Property-Based Testing for Potatostics Econometric Invariants using Hypothesis:
- Verifies mathematical invariance properties across thousands of generated inputs.
- Property 1: OLS Intercept Shift Invariance (y' = y + C => beta_0' = beta_0 + C, slopes unchanged).
- Property 2: OLS Scale Homogeneity (y' = S*y => beta' = S*beta, se' = |S|*se, t-stats unchanged).
- Property 3: Residual Orthogonality (sum(e) == 0, X^T * e == 0 for models with constant).
- Property 4: Classical R-squared bounds (0 <= R^2 <= 1.0).
- Property 5: Missingness isolation (dropping unobserved rows preserves identical estimators).
- Property 6: Safe AST mathematical expression evaluator bounded robustness.
"""

import operator
import unittest
import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st

from engine.estimation import (
    compute_regression,
    compute_ivregress_2sls,
    compute_panel_fe,
    compute_logit,
)
from engine.parser import evaluate_dta_expression


class TestPropertyEconometrics(unittest.TestCase):

    # =========================================================================
    # Property 1: OLS Shift-Invariance (Intercept Shift)
    # =========================================================================
    @settings(max_examples=50, deadline=None)
    @given(
        shift=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=1, max_value=10000)
    )
    def test_property_01_ols_intercept_shift_invariance(self, shift, seed):
        """
        Property: Shifting the outcome variable y by constant C:
        - Increases intercept estimate by exactly C.
        - Leaves all slope coefficients and their standard errors identical.
        - Leaves R-squared, F-statistic, and residuals identical.
        """
        rng = np.random.default_rng(seed)
        n = 50
        x1 = rng.normal(10.0, 3.0, size=n)
        x2 = rng.uniform(1.0, 5.0, size=n)
        y = 5.0 + 2.0 * x1 - 1.5 * x2 + rng.normal(0.0, 1.0, size=n)

        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        df_shifted = pd.DataFrame({"y": y + shift, "x1": x1, "x2": x2})

        res_orig = compute_regression(df, "y", ["x1", "x2"])
        res_shift = compute_regression(df_shifted, "y", ["x1", "x2"])

        # Compare slopes
        orig_coefs = {c["name"]: c["coef"] for c in res_orig["coefs"]}
        shift_coefs = {c["name"]: c["coef"] for c in res_shift["coefs"]}

        self.assertAlmostEqual(orig_coefs["x1"], shift_coefs["x1"], places=5)
        self.assertAlmostEqual(orig_coefs["x2"], shift_coefs["x2"], places=5)
        self.assertAlmostEqual(shift_coefs["_cons"], orig_coefs["_cons"] + shift, places=5)

        # R-squared invariance
        self.assertAlmostEqual(res_orig["r2"], res_shift["r2"], places=6)

    # =========================================================================
    # Property 2: OLS Scale-Homogeneity
    # =========================================================================
    @settings(max_examples=50, deadline=None)
    @given(
        scale=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=1, max_value=10000)
    )
    def test_property_02_ols_scale_homogeneity(self, scale, seed):
        """
        Property: Multiplying outcome variable y by positive scalar S:
        - Scales all coefficients by exactly S.
        - Scales all standard errors by exactly S.
        - Leaves t-statistics and p-values invariant.
        - Leaves R-squared invariant.
        """
        rng = np.random.default_rng(seed)
        n = 40
        x = rng.normal(0.0, 1.0, size=n)
        y = 3.0 + 1.8 * x + rng.normal(0.0, 0.5, size=n)

        df = pd.DataFrame({"y": y, "x": x})
        df_scaled = pd.DataFrame({"y": y * scale, "x": x})

        res_orig = compute_regression(df, "y", ["x"])
        res_scaled = compute_regression(df_scaled, "y", ["x"])

        for c_orig, c_scaled in zip(res_orig["coefs"], res_scaled["coefs"], strict=True):
            self.assertAlmostEqual(c_scaled["coef"], c_orig["coef"] * scale, places=4)
            self.assertAlmostEqual(c_scaled["se"], c_orig["se"] * scale, places=4)
            self.assertAlmostEqual(c_scaled["t"], c_orig["t"], places=4)
            self.assertAlmostEqual(c_scaled["p"], c_orig["p"], places=4)

        self.assertAlmostEqual(res_orig["r2"], res_scaled["r2"], places=6)

    # =========================================================================
    # Property 3: Residual Orthogonality and Zero-Mean
    # =========================================================================
    @settings(max_examples=50, deadline=None)
    @given(
        n=st.integers(min_value=20, max_value=100),
        seed=st.integers(min_value=1, max_value=10000)
    )
    def test_property_03_residual_orthogonality(self, n, seed):
        """
        Property: For classical OLS with an intercept:
        - Sum of residuals equals 0 (within machine precision).
        - Residuals are orthogonal to each regressor: X^T * e = 0.
        """
        rng = np.random.default_rng(seed)
        x1 = rng.normal(5.0, 2.0, size=n)
        x2 = rng.exponential(3.0, size=n)
        y = 2.0 + 0.5 * x1 - 1.2 * x2 + rng.normal(0.0, 1.0, size=n)

        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        res = compute_regression(df, "y", ["x1", "x2"])

        residuals = res["resid"]
        # 1. Sum of residuals is 0
        self.assertAlmostEqual(float(np.sum(residuals)), 0.0, places=5)

        # 2. Orthogonality to regressors
        self.assertAlmostEqual(float(np.dot(x1, residuals)), 0.0, places=4)
        self.assertAlmostEqual(float(np.dot(x2, residuals)), 0.0, places=4)

    # =========================================================================
    # Property 4: Classical R-Squared Bounds
    # =========================================================================
    @settings(max_examples=50, deadline=None)
    @given(
        noise_std=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=1, max_value=10000)
    )
    def test_property_04_r_squared_bounds(self, noise_std, seed):
        """
        Property: Classical linear regression R-squared is strictly bounded in [0.0, 1.0].
        """
        rng = np.random.default_rng(seed)
        n = 30
        x = rng.normal(0.0, 1.0, size=n)
        y = 1.0 + 2.0 * x + rng.normal(0.0, noise_std, size=n)

        df = pd.DataFrame({"y": y, "x": x})
        res = compute_regression(df, "y", ["x"])

        r2 = res["r2"]
        self.assertGreaterEqual(r2, -1e-9, f"R^2 must be >= 0: {r2}")
        self.assertLessEqual(r2, 1.0 + 1e-9, f"R^2 must be <= 1: {r2}")

    # =========================================================================
    # Property 5: Missingness Isolation
    # =========================================================================
    @settings(max_examples=40, deadline=None)
    @given(
        n_missing=st.integers(min_value=1, max_value=15),
        seed=st.integers(min_value=1, max_value=10000)
    )
    def test_property_05_missingness_isolation(self, n_missing, seed):
        """
        Property: Appending rows containing NaN values for unused rows must not alter
        the regression estimates on the complete observed sample.
        """
        rng = np.random.default_rng(seed)
        n_obs = 35
        x = rng.normal(10.0, 2.0, size=n_obs)
        y = 4.0 + 1.2 * x + rng.normal(0.0, 0.8, size=n_obs)

        df_complete = pd.DataFrame({"y": y, "x": x})
        res_complete = compute_regression(df_complete, "y", ["x"])

        # Append missing rows
        missing_rows = pd.DataFrame({
            "y": [np.nan] * n_missing,
            "x": [np.nan] * n_missing
        })
        df_with_nans = pd.concat([df_complete, missing_rows], ignore_index=True)

        res_nans = compute_regression(df_with_nans, "y", ["x"])

        self.assertEqual(res_complete["n"], res_nans["n"])
        for c1, c2 in zip(res_complete["coefs"], res_nans["coefs"], strict=True):
            self.assertAlmostEqual(c1["coef"], c2["coef"], places=6)
            self.assertAlmostEqual(c1["se"], c2["se"], places=6)

    # =========================================================================
    # Property 6: Safe AST Expression Evaluator Robustness
    # =========================================================================
    @settings(max_examples=60, deadline=None)
    @given(
        val_a=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        val_b=st.floats(min_value=0.01, max_value=1e4, allow_nan=False, allow_infinity=False),
        op=st.sampled_from(["+", "-", "*", "/"])
    )
    def test_property_06_safe_ast_expression_evaluator(self, val_a, val_b, op):
        """Valid bounded arithmetic agrees with Python for every generated operand pair."""
        expr = f"{val_a} {op} {val_b}"
        df = pd.DataFrame({"x": [1.0]})
        operations = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
        expected = operations[op](val_a, val_b)
        result = evaluate_dta_expression(expr, df)
        actual = float(result.iloc[0]) if hasattr(result, "iloc") else float(result)
        self.assertEqual(actual, expected, expr)

    # =========================================================================
    # Property 7: IV/2SLS Residual Orthogonality to Projected Regressors
    # =========================================================================
    @settings(max_examples=40, deadline=None)
    @given(
        n=st.integers(min_value=40, max_value=120),
        seed=st.integers(min_value=1, max_value=10000)
    )
    def test_property_07_iv_2sls_orthogonality(self, n, seed):
        """
        Property: In 2SLS, the structural residuals e = y - X*beta must be orthogonal
        to the instrument projection space X_hat = P_Z * X: X_hat^T * e == 0.
        """
        rng = np.random.default_rng(seed)
        z1 = rng.normal(size=n)
        z2 = rng.normal(size=n)
        w = rng.normal(size=n)
        u = rng.normal(size=n)
        # Endogenous x correlated with structural error u
        x = 0.6 * z1 + 0.4 * z2 + 0.2 * w + 0.5 * u
        y = 2.0 + 1.5 * x - 0.8 * w + u

        df = pd.DataFrame({"y": y, "x": x, "w": w, "z1": z1, "z2": z2})
        res = compute_ivregress_2sls(df, "y", ["w"], ["x"], ["z1", "z2"])
        e = res["resid"]

        # Z includes exogenous regressor w, excluded instruments z1, z2, and constant
        Z = np.column_stack([w, z1, z2, np.ones(n)])
        X = np.column_stack([w, x, np.ones(n)])
        # X_hat = Z * (Z^T Z)^(-1) Z^T X
        X_hat = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]

        ortho_check = X_hat.T @ e
        self.assertTrue(
            np.allclose(ortho_check, 0.0, atol=1e-4),
            f"2SLS residuals must be orthogonal to projected regressors: {ortho_check}"
        )

    # =========================================================================
    # Property 8: IV/2SLS Underidentification Order Condition Enforcement
    # =========================================================================
    @settings(max_examples=30, deadline=None)
    @given(
        n_endog=st.integers(min_value=2, max_value=4),
        seed=st.integers(min_value=1, max_value=10000)
    )
    def test_property_08_iv_underidentification_order_condition(self, n_endog, seed):
        """
        Property: When the number of excluded instruments is strictly less than
        the number of endogenous variables (L < K_endog), compute_ivregress_2sls
        must raise a ValueError identifying underidentification.
        """
        rng = np.random.default_rng(seed)
        n = 50
        data = {"y": rng.normal(size=n), "w": rng.normal(size=n)}
        endog_vars = []
        for i in range(n_endog):
            name = f"x{i+1}"
            data[name] = rng.normal(size=n)
            endog_vars.append(name)

        # Provide strictly fewer instruments than endogenous variables (e.g., 1 instrument)
        data["z1"] = rng.normal(size=n)
        df = pd.DataFrame(data)

        with self.assertRaises(ValueError) as ctx:
            compute_ivregress_2sls(df, "y", ["w"], endog_vars, ["z1"])

        self.assertIn("underidentified", str(ctx.exception).lower())

    # =========================================================================
    # Property 9: Panel Fixed Effects Demeaned Residual Zero-Mean per Entity
    # =========================================================================
    @settings(max_examples=35, deadline=None)
    @given(
        n_entities=st.integers(min_value=4, max_value=12),
        t_periods=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=1, max_value=10000)
    )
    def test_property_09_panel_fe_demeaned_residuals(self, n_entities, t_periods, seed):
        """
        Property: In within-entity fixed effects estimation, the within-group
        residual mean for every entity i must equal zero: sum_{t} e_{it} == 0.
        """
        rng = np.random.default_rng(seed)
        total_obs = n_entities * t_periods
        entities = np.repeat(np.arange(n_entities), t_periods)
        alpha_i = np.repeat(rng.normal(scale=2.0, size=n_entities), t_periods)
        x = rng.normal(scale=1.5, size=total_obs)
        y = alpha_i + 2.5 * x + rng.normal(scale=0.5, size=total_obs)

        df = pd.DataFrame({"entity": entities, "y": y, "x": x})
        res = compute_panel_fe(df, "y", ["x"], "entity")

        df["resid"] = res["resid"]
        entity_means = df.groupby("entity")["resid"].mean().to_numpy()

        self.assertTrue(
            np.allclose(entity_means, 0.0, atol=1e-5),
            f"Within-entity residual means must be zero: {entity_means}"
        )

    # =========================================================================
    # Property 10: Binary Logit Predicted Probability Strict Bounds
    # =========================================================================
    @settings(max_examples=35, deadline=None)
    @given(
        n=st.integers(min_value=50, max_value=120),
        seed=st.integers(min_value=1, max_value=10000)
    )
    def test_property_10_logit_probability_bounds(self, n, seed):
        """
        Property: In logistic regression, predicted probabilities p_pred must be
        strictly bounded within (0.0, 1.0) for every observation.
        """
        rng = np.random.default_rng(seed)
        x = rng.normal(scale=1.0, size=n)
        # Latent index
        true_p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x)))
        y = (rng.uniform(size=n) < true_p).astype(int)

        # Ensure both classes exist
        if np.sum(y == 1) == 0:
            y[0] = 1
        elif np.sum(y == 0) == 0:
            y[0] = 0

        df = pd.DataFrame({"y": y, "x": x})
        res = compute_logit(df, "y", ["x"])

        p_pred = np.asarray(res["p_pred"], dtype=float)
        self.assertTrue(np.all(p_pred > 0.0), f"p_pred must be strictly > 0: min={np.min(p_pred)}")
        self.assertTrue(np.all(p_pred < 1.0), f"p_pred must be strictly < 1: max={np.max(p_pred)}")


if __name__ == "__main__":
    unittest.main()
