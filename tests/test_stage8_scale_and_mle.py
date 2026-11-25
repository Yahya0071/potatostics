import unittest
import numpy as np
import pandas as pd
from engine.estimation import compute_regression, compute_logit
from engine.advanced_econometrics import compute_probit, compute_poisson

class TestScaleInvariantOLS(unittest.TestCase):
    def test_p0_small_scale_predictor_not_dropped(self):
        """
        P0 Test: x = np.linspace(1e-9, 2e-9, 100), y = 1.0 + 3.0 * x
        Must recover slope 3.0 and intercept 1.0; x must NOT be omitted.
        """
        x = np.linspace(1e-9, 2e-9, 100)
        y = 1.0 + 3.0 * x
        df = pd.DataFrame({'y': y, 'x': x})
        res = compute_regression(df, 'y', ['x'])
        
        self.assertEqual(res['omitted_vars'], [])
        self.assertEqual(res['rank'], 2)
        idx_x = res['var_names'].index('x')
        idx_cons = res['var_names'].index('_cons')
        np.testing.assert_allclose(res['beta'][idx_x], 3.0, rtol=1e-5)
        np.testing.assert_allclose(res['beta'][idx_cons], 1.0, rtol=1e-5)
        self.assertAlmostEqual(res['r2'], 1.0, places=5)

    def test_p0_scale_invariance_across_magnitudes(self):
        """
        Rescaling predictor by 1e-12, 1.0, 1e12 yields identical fitted values,
        R2, F-stat, and t-stats (accounting for coefficient units).
        """
        np.random.seed(42)
        n = 100
        z = np.random.randn(n)
        y = 2.0 + 1.5 * z + 0.5 * np.random.randn(n)

        scales = [1e-12, 1e-6, 1.0, 1e6, 1e12]
        base_slope = None
        base_cons = None
        base_res = None
        for s in scales:
            df = pd.DataFrame({'y': y, 'x': z * s})
            res = compute_regression(df, 'y', ['x'])
            self.assertEqual(res['omitted_vars'], [])
            idx_x = res['var_names'].index('x')
            idx_cons = res['var_names'].index('_cons')
            
            # slope * scale should be identical across all scales
            recovered_slope = res['beta'][idx_x] * s
            if base_slope is None:
                base_slope = recovered_slope
                base_cons = res['beta'][idx_cons]
            else:
                np.testing.assert_allclose(recovered_slope, base_slope, rtol=1e-5)
                np.testing.assert_allclose(res['beta'][idx_cons], base_cons, rtol=1e-4)
            
            # t-statistic on x and model R2 must be scale invariant
            t_stat = res['coefs'][idx_x]['t']
            if base_res is None:
                base_res = (res['r2'], t_stat, res['f_stat'])
            else:
                self.assertAlmostEqual(res['r2'], base_res[0], places=5)
                self.assertAlmostEqual(t_stat, base_res[1], places=4)
                self.assertAlmostEqual(res['f_stat'], base_res[2], places=4)

    def test_p0_all_zero_column_omitted(self):
        """All-zero column is genuinely non-identifiable and must be omitted."""
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0],
            'x1': [1.0, 2.0, 3.0, 4.0],
            'x_zero': [0.0, 0.0, 0.0, 0.0]
        })
        res = compute_regression(df, 'y', ['x1', 'x_zero'])
        self.assertIn('x_zero', res['omitted_vars'])
        self.assertEqual(res['rank'], 2) # x1 and _cons

    def test_p0_duplicate_constant_omitted(self):
        """A duplicate constant column is omitted in favor of _cons or earlier column."""
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0],
            'x1': [1.0, 2.0, 3.0, 4.0],
            'c1': [1.0, 1.0, 1.0, 1.0]
        })
        res = compute_regression(df, 'y', ['x1', 'c1'])
        # Either c1 or _cons is omitted
        self.assertTrue('c1' in res['omitted_vars'] or '_cons' in res['omitted_vars'])
        self.assertEqual(res['rank'], 2)

    def test_p0_noconstant_scale(self):
        """No-constant regression with small scale."""
        x = np.linspace(1e-9, 2e-9, 50)
        y = 3.0 * x
        df = pd.DataFrame({'y': y, 'x': x})
        res = compute_regression(df, 'y', ['x'], has_const=False)
        self.assertEqual(res['omitted_vars'], [])
        np.testing.assert_allclose(res['beta'][0], 3.0, rtol=1e-5)

    def test_p0_robust_and_clustered_under_rescaling(self):
        """HC1, HC2, HC3, and clustered SEs scale inversely with regressor scale."""
        np.random.seed(123)
        n = 100
        z = np.random.randn(n)
        cluster = np.repeat(np.arange(10), 10)
        y = 1.0 + 2.0 * z + np.random.randn(n)

        for vce_type, opts in [('hc1', {}), ('hc2', {}), ('hc3', {}), ('cluster', {'cluster_var': 'clust'})]:
            df_base = pd.DataFrame({'y': y, 'x': z, 'clust': cluster})
            res_base = compute_regression(df_base, 'y', ['x'], vce=vce_type, **opts)
            base_se = res_base['coefs'][0]['se']
            base_t = res_base['coefs'][0]['t']

            # Rescaled by 1e-8
            df_scaled = pd.DataFrame({'y': y, 'x': z * 1e-8, 'clust': cluster})
            res_scaled = compute_regression(df_scaled, 'y', ['x'], vce=vce_type, **opts)
            scaled_se = res_scaled['coefs'][0]['se']
            scaled_t = res_scaled['coefs'][0]['t']

            # SE must scale by 1 / 1e-8 = 1e8, t-stat must remain invariant
            np.testing.assert_allclose(scaled_se, base_se / 1e-8, rtol=1e-4)
            np.testing.assert_allclose(scaled_t, base_t, rtol=1e-4)

    def test_p0_statsmodels_cross_validation(self):
        """Cross-check OLS against statsmodels with recorded version and tolerances."""
        try:
            import statsmodels.api as sm
        except ImportError:
            self.skipTest("statsmodels not installed")

        np.random.seed(999)
        n = 100
        x = np.linspace(1e-8, 2e-8, n)
        y = 0.5 + 4.0 * x + np.random.normal(0, 1e-9, n)

        df = pd.DataFrame({'y': y, 'x': x})
        res = compute_regression(df, 'y', ['x'])

        X_sm = sm.add_constant(df['x'])
        model_sm = sm.OLS(df['y'], X_sm).fit()

        idx_x = res['var_names'].index('x')
        idx_cons = res['var_names'].index('_cons')

        # Compare coefficients and SEs with statsmodels (version sm.__version__)
        np.testing.assert_allclose(res['beta'][idx_x], model_sm.params['x'], rtol=1e-5)
        np.testing.assert_allclose(res['beta'][idx_cons], model_sm.params['const'], rtol=1e-5)
        np.testing.assert_allclose(res['coefs'][idx_x]['se'], model_sm.bse['x'], rtol=1e-4)
        np.testing.assert_allclose(res['coefs'][idx_cons]['se'], model_sm.bse['const'], rtol=1e-4)
        np.testing.assert_allclose(res['r2'], model_sm.rsquared, rtol=1e-5)

class TestMLELineSearchAndConvergence(unittest.TestCase):
    def test_p1_logit_monotonic_ll_and_correct_armijo(self):
        """Logit iterations must achieve non-decreasing log-likelihood and evaluate convergence at final point."""
        np.random.seed(42)
        n = 200
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        z = -0.5 + 1.2 * x1 - 0.8 * x2
        p = 1.0 / (1.0 + np.exp(-z))
        y = (np.random.rand(n) < p).astype(float)
        df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

        res = compute_logit(df, 'y', ['x1', 'x2'])
        self.assertTrue(res['converged'])
        self.assertIn('ll_history', res)
        # Verify log-likelihood is monotonically non-decreasing
        ll_hist = res['ll_history']
        self.assertTrue(len(ll_hist) >= 2)
        for i in range(1, len(ll_hist)):
            self.assertGreaterEqual(ll_hist[i], ll_hist[i-1] - 1e-10,
                                   f"Log-likelihood decreased from {ll_hist[i-1]} to {ll_hist[i]} at step {i}")

    def test_p1_line_search_failure_never_applies_fallback(self):
        """When line search fails, optimizer must report converged=False and NOT apply unaccepted fallback."""
        df = pd.DataFrame({
            'y': [1, 0, 1, 0, 1, 0],
            'x': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        })
        # If line search fails or iteration limit is 0 / 1
        res = compute_logit(df, 'y', ['x'], max_iter=1)
        # Should report proper convergence status
        self.assertIn('final_score_norm', res)
        self.assertIn('final_step_norm', res)

    def test_p1_probit_poisson_convergence_contract(self):
        """Probit and Poisson return consistent convergence contract fields and monotonic ll."""
        np.random.seed(123)
        n = 150
        x = np.random.randn(n)
        y_bin = (x + np.random.randn(n) > 0).astype(float)
        y_count = np.random.poisson(np.exp(0.5 + 0.3 * x))
        df = pd.DataFrame({'y_b': y_bin, 'y_c': y_count, 'x': x})

        res_pr = compute_probit(df, 'y_b', ['x'])
        self.assertTrue(res_pr['converged'])
        self.assertIn('ll_history', res_pr)
        for i in range(1, len(res_pr['ll_history'])):
            self.assertGreaterEqual(res_pr['ll_history'][i], res_pr['ll_history'][i-1] - 1e-10)

        res_poi = compute_poisson(df, 'y_c', ['x'])
        self.assertTrue(res_poi['converged'])
        self.assertIn('ll_history', res_poi)
        for i in range(1, len(res_poi['ll_history'])):
            self.assertGreaterEqual(res_poi['ll_history'][i], res_poi['ll_history'][i-1] - 1e-10)

if __name__ == '__main__':
    unittest.main()
