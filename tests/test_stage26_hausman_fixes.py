import unittest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from engine.dataset import Dataset
from engine.interpreter import Interpreter
try:
    from linearmodels.panel import PanelOLS, RandomEffects
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False

class TestHausmanFixes(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_01_hausman_non_pd_message_and_nonzero_rc(self):
        """When V_b - V_B is not positive definite, print failure message, no p-value, and nonzero return code."""
        np.random.seed(4)
        n_entities = 15
        n_periods = 4
        entity = np.repeat(np.arange(n_entities), n_periods)
        time = np.tile(np.arange(n_periods), n_entities)
        
        u = np.repeat(np.random.randn(n_entities) * 2.0, n_periods)
        x1 = u + np.random.randn(len(entity)) * 0.5
        x2 = np.random.randn(len(entity))
        y = 1.0 + 1.5 * x1 - 0.5 * x2 + u + np.random.randn(len(entity))

        df = pd.DataFrame({'entity': entity, 'time': time, 'y': y, 'x1': x1, 'x2': x2})
        ds = Dataset()
        ds.load_dataframe(df)
        interp = Interpreter(ds)
        interp.execute("xtset entity time")
        interp.execute("xtreg y x1 x2, fe")
        interp.execute("estimates store fe_mod")
        interp.execute("xtreg y x1 x2, re")
        interp.execute("estimates store re_mod")

        res = interp.execute("hausman fe_mod re_mod")
        # Under non-PD, must return nonzero return code and specific error message, no p-value
        self.assertFalse(res.ok, "Expected hausman with non-PD V_diff to return ok=False")
        self.assertNotEqual(res.return_code, 0, "Return code must be nonzero on non-PD")
        self.assertIn("chi2 < 0: the model fails the asymptotic assumptions of the Hausman test", res.output)
        self.assertNotIn("Prob>chi2", res.output)
        self.assertNotIn("Prob > chi2", res.output)

    def test_02_hausman_sigmamore_and_sigmaless(self):
        """hausman must support sigmamore and sigmaless options, and reject both together."""
        n_entities = 50
        n_periods = 8
        entity = np.repeat(np.arange(n_entities), n_periods)
        time = np.tile(np.arange(n_periods), n_entities)
        x1 = np.random.randn(len(entity))
        x2 = np.random.randn(len(entity))
        y = 1.0 + 2.0 * x1 - 3.0 * x2 + np.random.randn(len(entity))

        df = pd.DataFrame({'entity': entity, 'time': time, 'y': y, 'x1': x1, 'x2': x2})
        ds = Dataset()
        ds.load_dataframe(df)
        interp = Interpreter(ds)
        interp.execute("xtset entity time")
        interp.execute("xtreg y x1 x2, fe")
        interp.execute("estimates store fe_mod")
        interp.execute("xtreg y x1 x2, re")
        interp.execute("estimates store re_mod")

        res_more = interp.execute("hausman fe_mod re_mod, sigmamore")
        self.assertTrue(res_more.ok, f"hausman sigmamore failed: {res_more.error}")
        chi2_more = interp.registers.get_r("chi2")
        self.assertIsNotNone(chi2_more)

        res_less = interp.execute("hausman fe_mod re_mod, sigmaless")
        self.assertTrue(res_less.ok, f"hausman sigmaless failed: {res_less.error}")
        chi2_less = interp.registers.get_r("chi2")
        self.assertIsNotNone(chi2_less)

        # Mutually exclusive: cannot specify both
        res_both = interp.execute("hausman fe_mod re_mod, sigmamore sigmaless")
        self.assertFalse(res_both.ok)
        self.assertEqual(res_both.return_code, 198)

    def test_03_hausman_correlated_effect_not_p_one(self):
        """When x is correlated with panel effect, Hausman test rejects H0 and must not report p = 1.0000."""
        n_entities = 100
        n_periods = 10
        entity = np.repeat(np.arange(n_entities), n_periods)
        time = np.tile(np.arange(n_periods), n_entities)
        u_i = np.random.randn(n_entities) * 2.0
        u = np.repeat(u_i, n_periods)
        
        # Endogenous regressor x1 correlated with individual effect u_i
        x1 = 0.8 * u + np.random.randn(len(entity))
        x2 = np.random.randn(len(entity))
        y = 1.0 + 2.0 * x1 + 3.0 * x2 + u + np.random.randn(len(entity))

        df = pd.DataFrame({'entity': entity, 'time': time, 'y': y, 'x1': x1, 'x2': x2})
        ds = Dataset()
        ds.load_dataframe(df)
        interp = Interpreter(ds)
        interp.execute("xtset entity time")
        interp.execute("xtreg y x1 x2, fe")
        interp.execute("estimates store fe_mod")
        interp.execute("xtreg y x1 x2, re")
        interp.execute("estimates store re_mod")

        res = interp.execute("hausman fe_mod re_mod, sigmamore")
        self.assertTrue(res.ok, f"hausman failed: {res.error}")
        p_val = interp.registers.get_r("p")
        chi2_stat = interp.registers.get_r("chi2")

        self.assertIsNotNone(p_val)
        self.assertNotEqual(p_val, 1.0000, "Hausman on correlated effect must not report p = 1.0000")
        self.assertLess(p_val, 0.05, f"Hausman should reject H0 when x is correlated with u_i (got p={p_val})")
        self.assertGreater(chi2_stat, 0)

    def test_04_hausman_exogenous_vs_linearmodels(self):
        """Cross-validate Hausman statistic against independent computation from linearmodels."""
        if not HAS_LINEARMODELS:
            self.skipTest("linearmodels not available")
        n_entities = 40
        n_periods = 6
        entity = np.repeat(np.arange(n_entities), n_periods)
        time = np.tile(np.arange(n_periods), n_entities)
        u = np.repeat(np.random.randn(n_entities) * 1.5, n_periods)
        x1 = np.random.randn(len(entity))
        x2 = np.random.randn(len(entity))
        y = 2.0 + 1.2 * x1 - 0.8 * x2 + u + np.random.randn(len(entity))

        df = pd.DataFrame({'entity': entity, 'time': time, 'y': y, 'x1': x1, 'x2': x2})
        
        # Independent computation with linearmodels
        df_lm = df.set_index(['entity', 'time'])
        fe_fit = PanelOLS(df_lm['y'], df_lm[['x1', 'x2']], entity_effects=True).fit()
        re_fit = RandomEffects(df_lm['y'], sm.add_constant(df_lm[['x1', 'x2']])).fit()

        b_fe = fe_fit.params[['x1', 'x2']].to_numpy()
        b_re = re_fit.params[['x1', 'x2']].to_numpy()
        v_fe = fe_fit.cov.loc[['x1', 'x2'], ['x1', 'x2']].to_numpy()
        v_re = re_fit.cov.loc[['x1', 'x2'], ['x1', 'x2']].to_numpy()

        diff_b = b_fe - b_re
        v_diff = v_fe - v_re
        expected_chi2 = float(diff_b.T @ np.linalg.inv(v_diff) @ diff_b)

        # Potatostics execution
        ds = Dataset()
        ds.load_dataframe(df)
        interp = Interpreter(ds)
        interp.execute("xtset entity time")
        interp.execute("xtreg y x1 x2, fe")
        interp.execute("estimates store fe_mod")
        interp.execute("xtreg y x1 x2, re")
        interp.execute("estimates store re_mod")

        res = interp.execute("hausman fe_mod re_mod")
        self.assertTrue(res.ok, f"hausman failed: {res.error}")
        pot_chi2 = interp.registers.get_r("chi2")

        self.assertAlmostEqual(pot_chi2, expected_chi2, places=3)
