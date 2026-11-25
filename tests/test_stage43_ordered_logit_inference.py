import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch

from engine.advanced_econometrics import compute_ologit


class TestStage43OrderedLogitInference(unittest.TestCase):
    """
    Stage 43: Verification of Ordered Logit Delta-Method Cutpoint Standard Errors,
    Observed Fisher Information Fallback, and Non-Degeneracy Checks.
    """

    def setUp(self):
        np.random.seed(42)

    def test_01_analytical_vs_finite_difference_jacobian(self):
        """Verify the Delta-method cutpoint Jacobian matches finite differences for J=3, 4, 5."""
        for J in [3, 4, 5]:
            n_cuts = J - 1
            theta = np.random.uniform(-1.0, 1.0, size=n_cuts)

            # Analytical Jacobian
            # c[0] = theta[0]
            # c[j] = theta[0] + sum_{m=1}^j exp(theta[m])
            J_thresh = np.zeros((n_cuts, n_cuts))
            J_thresh[:, 0] = 1.0
            for j in range(1, n_cuts):
                for m in range(1, j + 1):
                    J_thresh[j, m] = np.exp(theta[m])

            # Numerical Jacobian via central differences
            eps = 1e-6
            J_num = np.zeros((n_cuts, n_cuts))
            for m in range(n_cuts):
                step = np.zeros(n_cuts)
                step[m] = eps
                th_plus = theta + step
                th_minus = theta - step

                c_plus = np.zeros(n_cuts)
                c_plus[0] = th_plus[0]
                if n_cuts > 1:
                    c_plus[1:] = th_plus[0] + np.cumsum(np.exp(th_plus[1:]))

                c_minus = np.zeros(n_cuts)
                c_minus[0] = th_minus[0]
                if n_cuts > 1:
                    c_minus[1:] = th_minus[0] + np.cumsum(np.exp(th_minus[1:]))

                J_num[:, m] = (c_plus - c_minus) / (2.0 * eps)

            diff = np.max(np.abs(J_thresh - J_num))
            self.assertLess(diff, 1e-6, f"Jacobian mismatch for J={J}: max diff {diff}")

    def test_02_cutpoint_se_uses_delta_method(self):
        """Ensure compute_ologit cutpoint standard errors come from Delta method, not raw bse."""
        n = 200
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        latent = 0.7 * x1 - 0.4 * x2 + np.random.logistic(0, 1, n)
        y = np.select([latent < -0.8, latent < 0.8], [1, 2], default=3)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

        r = compute_ologit(df, "y", ["x1", "x2"])

        # Check cutpoints
        cut_rows = [c for c in r["coefs"] if c.get("is_cutpoint")]
        self.assertEqual(len(cut_rows), 2)

        from statsmodels.miscmodels.ordinal_model import OrderedModel
        mod = OrderedModel(df["y"], df[["x1", "x2"]], distr="logit")
        res = mod.fit(method="bfgs", disp=False)

        raw_cov = res.cov_params().to_numpy()
        k = 2
        # Cutpoint 2 SE in raw params is res.bse["2/3"], but transformed cutpoint SE must be Delta SE
        th1 = float(res.params.iloc[k + 1])
        # Delta method for cutpoint 2:
        # Var(c2) = Var(theta0) + exp(2*theta1)*Var(theta1) + 2*exp(theta1)*Cov(theta0, theta1)
        var_c1 = raw_cov[k, k]
        var_c2 = (
            raw_cov[k, k]
            + np.exp(2.0 * th1) * raw_cov[k + 1, k + 1]
            + 2.0 * np.exp(th1) * raw_cov[k, k + 1]
        )
        expected_se_c1 = np.sqrt(var_c1)
        expected_se_c2 = np.sqrt(var_c2)

        self.assertAlmostEqual(cut_rows[0]["se"], expected_se_c1, places=5)
        self.assertAlmostEqual(cut_rows[1]["se"], expected_se_c2, places=5)

        # Delta SE for cutpoint 2 must differ from raw res.bse
        raw_cut2_se = float(res.bse.iloc[k + 1])
        self.assertNotAlmostEqual(cut_rows[1]["se"], raw_cut2_se, places=3)

        # Check confidence intervals
        z_crit = 1.959963984540054
        self.assertAlmostEqual(cut_rows[1]["ci_lower"], cut_rows[1]["coef"] - z_crit * cut_rows[1]["se"], places=5)
        self.assertAlmostEqual(cut_rows[1]["ci_upper"], cut_rows[1]["coef"] + z_crit * cut_rows[1]["se"], places=5)

    def test_03_fallback_information_matrix_no_fabricated_numbers(self):
        """When statsmodels OrderedModel is unavailable, fallback must invert observed Hessian."""
        n = 150
        x = np.random.normal(0, 1, n)
        latent = 0.9 * x + np.random.logistic(0, 1, n)
        y = np.select([latent < -0.5, latent < 0.7], [1, 2], default=3)
        df = pd.DataFrame({"y": y, "x": x})

        # Run reference with statsmodels
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        mod = OrderedModel(df["y"], df[["x"]], distr="logit")
        res = mod.fit(method="bfgs", disp=False)

        # Force fallback in compute_ologit by raising ImportError in OrderedModel
        with patch.dict("sys.modules", {"statsmodels.miscmodels.ordinal_model": None}):
            r_fallback = compute_ologit(df, "y", ["x"])

        self.assertEqual(r_fallback.get("estimation_method"), "scipy_bfgs_fallback")
        beta_fallback = r_fallback["coefs"][0]["coef"]
        se_fallback = r_fallback["coefs"][0]["se"]

        # Must NOT be fabricated 0.1
        self.assertNotEqual(se_fallback, 0.1)
        self.assertAlmostEqual(beta_fallback, float(res.params.iloc[0]), delta=0.05)
        self.assertAlmostEqual(se_fallback, float(res.bse.iloc[0]), delta=0.05)

    def test_04_fallback_singular_hessian_reports_failure(self):
        """Singular or non-PD Hessian in fallback must set SE to NaN and report failure, not fabricated SE."""
        # Collinear design
        df = pd.DataFrame({
            "y": [1, 2, 3, 1, 2, 3],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],  # perfectly collinear
        })

        with patch.dict("sys.modules", {"statsmodels.miscmodels.ordinal_model": None}):
            r = compute_ologit(df, "y", ["x1", "x2"])

        self.assertEqual(r.get("inference_status"), "FAILED")
        for c in r["coefs"]:
            self.assertTrue(np.isnan(c["se"]))
            self.assertNotEqual(c["se"], 0.1)


if __name__ == "__main__":
    unittest.main()
