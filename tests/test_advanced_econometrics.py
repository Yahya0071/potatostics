"""
tests/test_advanced_econometrics.py - Unit Tests for Phase 2D:
1. reghdfe / absorb (High-Dimensional Multi-Way Fixed Effects via MAP)
2. probit (Binary Probit Model Newton-Raphson MLE)
3. poisson (Poisson Count Regression MLE with IRR)
4. tobit (Censored Normal Regression)
5. qreg (Quantile Regression via HiGHS LP)
6. margins (Average Marginal Effects for Linear, Probit, Logit, Poisson)
7. lincom (Linear Combinations of Estimators)
"""

import unittest
import numpy as np
import pandas as pd

from engine.dataset import Dataset
from engine.interpreter import Interpreter


class TestAdvancedEconometrics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 300
        firm_ids = np.random.choice(np.arange(1, 21), size=n)
        year_ids = np.random.choice([2018, 2019, 2020, 2021, 2022], size=n)
        industry_ids = np.random.choice([1, 2, 3, 4], size=n)

        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(2, 0.5, n)
        
        # Linear DGP with two fixed effects
        firm_fe = {f: np.random.normal(0, 2) for f in range(1, 21)}
        year_fe = {y: np.random.normal(0, 1) for y in [2018, 2019, 2020, 2021, 2022]}
        y_lin = 3.0 * x1 - 1.5 * x2 + np.array([firm_fe[f] for f in firm_ids]) + np.array([year_fe[y] for y in year_ids]) + np.random.normal(0, 0.5, n)

        # Probit DGP
        z_probit = 0.8 * x1 - 0.5 * x2 + np.random.normal(0, 1, n)
        y_probit = (z_probit > 0).astype(int)

        # Poisson DGP (counts)
        log_mu = 0.4 * x1 - 0.2 * x2 + 1.0
        y_poisson = np.random.poisson(np.exp(np.clip(log_mu, -2, 3)))

        # Tobit DGP (censored at 0)
        y_latent = 2.0 * x1 + 1.0 * x2 + np.random.normal(0, 1, n)
        y_tobit = np.maximum(0.0, y_latent)

        self.df = pd.DataFrame({
            'y_lin': y_lin,
            'y_probit': y_probit,
            'y_poisson': y_poisson,
            'y_tobit': y_tobit,
            'x1': x1,
            'x2': x2,
            'firm': firm_ids,
            'year': year_ids,
            'ind': industry_ids
        })

        self.dataset = Dataset()
        self.dataset.load_dataframe(self.df)
        self.interpreter = Interpreter(self.dataset)

    def test_01_reghdfe_two_way_fixed_effects(self):
        res = self.interpreter.execute("reghdfe y_lin x1 x2, absorb(firm year)")
        self.assertTrue(res.ok)
        self.assertIn("HDFE Linear regression", res.output)
        self.assertIn("Within R2", res.output)

        regs = self.interpreter.registers
        self.assertEqual(regs.e_macros.get('cmd'), 'reghdfe')
        self.assertEqual(regs.e_scalars.get('N'), 300)
        self.assertAlmostEqual(regs.b.get('x1'), 3.0, delta=0.25)
        self.assertAlmostEqual(regs.b.get('x2'), -1.5, delta=0.25)

    def test_02_reghdfe_with_clustered_se(self):
        res = self.interpreter.execute("reghdfe y_lin x1 x2, absorb(firm) vce(cluster firm)")
        self.assertTrue(res.ok)
        regs = self.interpreter.registers
        self.assertGreater(regs.se.get('x1'), 0.0)

    def test_03_probit_estimation(self):
        res = self.interpreter.execute("probit y_probit x1 x2")
        self.assertTrue(res.ok)
        self.assertIn("Probit regression", res.output)
        self.assertIn("Pseudo R2", res.output)

        regs = self.interpreter.registers
        self.assertEqual(regs.e_macros.get('cmd'), 'probit')
        self.assertGreater(regs.e_scalars.get('chi2'), 0.0)
        self.assertIn('x1', regs.b)
        self.assertIn('x2', regs.b)
        self.assertIn('_cons', regs.b)

    def test_04_poisson_estimation_and_irr(self):
        res = self.interpreter.execute("poisson y_poisson x1 x2")
        self.assertTrue(res.ok)
        self.assertIn("Poisson regression", res.output)

        regs = self.interpreter.registers
        self.assertEqual(regs.e_macros.get('cmd'), 'poisson')
        self.assertIn('x1', regs.b)

        # Test with IRR option
        res_irr = self.interpreter.execute("poisson y_poisson x1 x2, irr")
        self.assertTrue(res_irr.ok)
        self.assertIn("IRR", res_irr.output)

    def test_05_tobit_censored_regression(self):
        res = self.interpreter.execute("tobit y_tobit x1 x2, ll(0)")
        self.assertTrue(res.ok)
        self.assertIn("Tobit regression", res.output)
        self.assertIn("Left-censored", res.output)

        regs = self.interpreter.registers
        self.assertEqual(regs.e_macros.get('cmd'), 'tobit')
        self.assertIn('x1', regs.b)
        self.assertIn('/sigma', regs.b)

    def test_06_qreg_median_and_quartiles(self):
        # Median regression (q=0.5)
        res_med = self.interpreter.execute("qreg y_lin x1 x2")
        self.assertTrue(res_med.ok)
        self.assertIn("Quantile regression (.q = 0.50)", res_med.output)

        regs = self.interpreter.registers
        self.assertAlmostEqual(regs.b.get('x1'), 3.0, delta=0.5)

        # 75th percentile regression
        res_q75 = self.interpreter.execute("qreg y_lin x1 x2, quantile(0.75)")
        self.assertTrue(res_q75.ok)
        self.assertIn(".q = 0.75", res_q75.output)

    def test_07_margins_on_linear_and_probit(self):
        # Fit probit then compute average marginal effects
        self.interpreter.execute("probit y_probit x1 x2")
        res_m = self.interpreter.execute("margins, dydx(*)")
        self.assertTrue(res_m.ok)
        self.assertIn("Average marginal effects (AME)", res_m.output)
        self.assertIn("dy/dx", res_m.output)
        self.assertIn("x1", res_m.output)
        self.assertIn("x2", res_m.output)

    def test_08_lincom_hypothesis_testing(self):
        # Fit OLS
        self.interpreter.execute("regress y_lin x1 x2")
        # Test linear combination: x1 + 2*x2
        res_lc = self.interpreter.execute("lincom x1 + 2*x2")
        self.assertTrue(res_lc.ok)
        self.assertIn("Linear comb.", res_lc.output)
        self.assertIn("(1)", res_lc.output)

        regs = self.interpreter.registers
        self.assertIn('estimate', regs.r_scalars)
        self.assertIn('se', regs.r_scalars)
        self.assertIn('t', regs.r_scalars)
        self.assertIn('p', regs.r_scalars)
        expected_est = regs.b['x1'] + 2.0 * regs.b['x2']
        self.assertAlmostEqual(regs.r_scalars['estimate'], expected_est, places=4)

    def test_09_reghdfe_with_factor_variables(self):
        res = self.interpreter.execute("reghdfe y_lin x1 i.ind, absorb(firm)")
        self.assertTrue(res.ok)
        self.assertIn("HDFE Linear regression", res.output)

    def test_10_poisson_robust_and_cluster_vce(self):
        res_rob = self.interpreter.execute("poisson y_poisson x1 x2, robust")
        self.assertTrue(res_rob.ok)
        res_clust = self.interpreter.execute("poisson y_poisson x1 x2, vce(cluster firm)")
        self.assertTrue(res_clust.ok)

    def test_11_tobit_two_sided_censoring(self):
        res = self.interpreter.execute("tobit y_tobit x1 x2, ll(0) ul(4)")
        self.assertTrue(res.ok)
        self.assertIn("Right-censored", res.output)

    def test_12_margins_on_poisson(self):
        self.interpreter.execute("poisson y_poisson x1 x2")
        res_m = self.interpreter.execute("margins, dydx(x1)")
        self.assertTrue(res_m.ok)
        self.assertIn("x1", res_m.output)

    def test_13_probit_error_non_binary(self):
        res = self.interpreter.execute("probit y_lin x1 x2")
        self.assertFalse(res.ok)
        self.assertIn("must be binary", res.error)

    def test_14_lincom_difference_and_constant(self):
        self.interpreter.execute("regress y_lin x1 x2")
        res = self.interpreter.execute("lincom x1 - x2 + 10")
        self.assertTrue(res.ok)
        regs = self.interpreter.registers
        expected = (regs.b['x1'] - regs.b['x2']) + 10.0
        self.assertAlmostEqual(regs.r_scalars['estimate'], expected, places=4)


if __name__ == '__main__':
    unittest.main()
