import unittest
import numpy as np
import pandas as pd
from engine.dataset import Dataset
from engine.interpreter import Interpreter

class TestNbreg(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate the exact benchmark count dataset verified against Reference 15
        np.random.seed(42)
        n = 200
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        alpha_true = 0.5
        mu = np.exp(0.5 + 0.8 * x1 - 0.4 * x2)
        lam = np.random.gamma(shape=1.0 / alpha_true, scale=alpha_true * mu)
        y = np.random.poisson(lam)

        cls.df = pd.DataFrame({
            'y': y,
            'x1': x1,
            'x2': x2,
            'cluster_id': np.repeat(np.arange(20), 10)
        })

    def setUp(self):
        self.dataset = Dataset()
        self.dataset.load_dataframe(self.df.copy())
        self.interp = Interpreter(self.dataset)

    def test_01_nbreg_golden_reference(self):
        """Verify nbreg estimates, alpha dispersion, and LR test against Reference 15."""
        res = self.interp.execute("nbreg y x1 x2")
        self.assertTrue(res.ok, f"nbreg failed: {res.error}")

        # Reference 15 Golden Benchmark:
        # x1: Coef = 0.8129916, Std. Err. = 0.0821662
        # x2: Coef = -0.2779643, Std. Err. = 0.0784306
        # _cons: Coef = 0.4382707, Std. Err. = 0.0796505
        # /lnalpha = -0.7432777, Std. Err. = 0.2520115
        # alpha = 0.4755527, Std. Err. = 0.1198448
        # Log likelihood = -337.29807, LR chi2 = 92.23, LR test alpha=0 chi2 = 48.73
        b_x1 = self.interp.registers.b.get("x1")
        b_x2 = self.interp.registers.b.get("x2")
        b_cons = self.interp.registers.b.get("_cons")

        se_x1 = self.interp.registers.se.get("x1")
        se_x2 = self.interp.registers.se.get("x2")
        se_cons = self.interp.registers.se.get("_cons")

        self.assertIsNotNone(b_x1, "x1 coefficient missing")
        self.assertAlmostEqual(b_x1, 0.8129916, places=3)
        self.assertAlmostEqual(b_x2, -0.2779643, places=3)
        self.assertAlmostEqual(b_cons, 0.4382707, places=3)

        self.assertAlmostEqual(se_x1, 0.0821662, places=3)
        self.assertAlmostEqual(se_x2, 0.0784306, places=3)
        self.assertAlmostEqual(se_cons, 0.0796505, places=3)

        # Alpha and log-likelihood
        alpha = self.interp.registers.e_scalars.get("alpha")
        lnalpha = self.interp.registers.e_scalars.get("lnalpha")
        ll = self.interp.registers.e_scalars.get("ll")
        chi2_c = self.interp.registers.e_scalars.get("chi2_c")

        self.assertIsNotNone(alpha, "e(alpha) missing")
        self.assertAlmostEqual(alpha, 0.4755527, places=2)
        self.assertAlmostEqual(lnalpha, -0.7432777, places=2)
        self.assertAlmostEqual(ll, -337.29807, places=1)
        self.assertAlmostEqual(chi2_c, 48.73, places=1)

    def test_02_nbreg_irr_option(self):
        """Verify nbreg with irr option calculates incidence rate ratios exp(beta)."""
        res = self.interp.execute("nbreg y x1 x2, irr")
        self.assertTrue(res.ok, f"nbreg irr failed: {res.error}")
        self.assertIn("IRR", res.output)
        # IRR for x1 = exp(0.81299) approx 2.2546
        self.assertIn("2.254", res.output)

    def test_03_nbreg_robust_and_cluster(self):
        """Verify nbreg with robust and clustered standard errors."""
        res_rob = self.interp.execute("nbreg y x1 x2, vce(robust)")
        self.assertTrue(res_rob.ok, f"nbreg robust failed: {res_rob.error}")
        self.assertIn("Robust std. err.", res_rob.output)

        res_cl = self.interp.execute("nbreg y x1 x2, vce(cluster cluster_id)")
        self.assertTrue(res_cl.ok, f"nbreg cluster failed: {res_cl.error}")
        self.assertIn("clusters in cluster_id", res_cl.output)

if __name__ == "__main__":
    unittest.main()
