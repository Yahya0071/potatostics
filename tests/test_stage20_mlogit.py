import unittest
import numpy as np
import pandas as pd
from engine.dataset import Dataset
from engine.interpreter import Interpreter

class TestMlogit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate the exact benchmark multinomial dataset verified against Reference 15
        np.random.seed(42)
        n = 300
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        u1 = 0.5 + 1.2 * x1 - 0.7 * x2
        u2 = -0.3 - 0.8 * x1 + 1.1 * x2
        exp_u = np.column_stack([np.ones(n), np.exp(u1), np.exp(u2)])
        probs = exp_u / exp_u.sum(axis=1, keepdims=True)
        y = np.array([np.random.choice([0, 1, 2], p=p) for p in probs])

        cls.df = pd.DataFrame({
            'y': y,
            'x1': x1,
            'x2': x2,
            'group_id': np.repeat(np.arange(30), 10)
        })

    def setUp(self):
        self.dataset = Dataset()
        self.dataset.load_dataframe(self.df.copy())
        self.interp = Interpreter(self.dataset)

    def test_01_mlogit_golden_reference(self):
        """Verify mlogit coefficients, standard errors, and log-likelihood against Reference 15."""
        res = self.interp.execute("mlogit y x1 x2, baseoutcome(0)")
        self.assertTrue(res.ok, f"mlogit failed: {res.error}")

        # Reference 15 Golden Benchmark:
        # Eq 1: x1: 1.25668 (se 0.21877), x2: -0.64235 (se 0.19925), _cons: 0.37284 (se 0.16539)
        # Eq 2: x1: -1.05791 (se 0.25858), x2: 1.20154 (se 0.23529), _cons: -0.69076 (se 0.23617)
        # Log likelihood = -227.21739, LR chi2(4) = 175.68
        ll = self.interp.registers.e_scalars.get("ll")
        chi2 = self.interp.registers.e_scalars.get("chi2")
        df_m = self.interp.registers.e_scalars.get("df_m")

        self.assertIsNotNone(ll, "e(ll) missing")
        self.assertAlmostEqual(ll, -227.21739, places=2)
        self.assertAlmostEqual(chi2, 175.68, places=1)
        self.assertEqual(int(df_m), 4)

        # Coefficients for equation 1:
        b_1_x1 = self.interp.registers.b.get("1:x1")
        b_1_x2 = self.interp.registers.b.get("1:x2")
        b_1_cons = self.interp.registers.b.get("1:_cons")

        se_1_x1 = self.interp.registers.se.get("1:x1")
        se_1_x2 = self.interp.registers.se.get("1:x2")
        se_1_cons = self.interp.registers.se.get("1:_cons")

        self.assertIsNotNone(b_1_x1, "1:x1 missing")
        self.assertAlmostEqual(b_1_x1, 1.25668, places=3)
        self.assertAlmostEqual(b_1_x2, -0.6423543, places=3)
        self.assertAlmostEqual(b_1_cons, 0.3728385, places=3)

        self.assertAlmostEqual(se_1_x1, 0.2187697, places=3)
        self.assertAlmostEqual(se_1_x2, 0.199246, places=3)
        self.assertAlmostEqual(se_1_cons, 0.1653946, places=3)

        # Coefficients for equation 2:
        b_2_x1 = self.interp.registers.b.get("2:x1")
        b_2_x2 = self.interp.registers.b.get("2:x2")
        b_2_cons = self.interp.registers.b.get("2:_cons")

        se_2_x1 = self.interp.registers.se.get("2:x1")
        se_2_x2 = self.interp.registers.se.get("2:x2")
        se_2_cons = self.interp.registers.se.get("2:_cons")

        self.assertIsNotNone(b_2_x1, "2:x1 missing")
        self.assertAlmostEqual(b_2_x1, -1.057911, places=3)
        self.assertAlmostEqual(b_2_x2, 1.201544, places=3)
        self.assertAlmostEqual(b_2_cons, -0.6907585, places=3)

        self.assertAlmostEqual(se_2_x1, 0.2585807, places=3)
        self.assertAlmostEqual(se_2_x2, 0.2352868, places=3)
        self.assertAlmostEqual(se_2_cons, 0.2361673, places=3)

    def test_02_mlogit_rrr_option(self):
        """Verify mlogit relative risk ratios exp(beta) when rrr option is specified."""
        res = self.interp.execute("mlogit y x1 x2, baseoutcome(0) rrr")
        self.assertTrue(res.ok, f"mlogit rrr failed: {res.error}")
        self.assertIn("RRR", res.output)
        # exp(1.25668) approx 3.5137
        self.assertIn("3.51", res.output)

    def test_03_mlogit_robust_and_cluster(self):
        """Verify mlogit with robust and cluster options."""
        res_rob = self.interp.execute("mlogit y x1 x2, vce(robust)")
        self.assertTrue(res_rob.ok, f"mlogit robust failed: {res_rob.error}")
        self.assertIn("Robust std. err.", res_rob.output)

        res_cl = self.interp.execute("mlogit y x1 x2, vce(cluster group_id)")
        self.assertTrue(res_cl.ok, f"mlogit cluster failed: {res_cl.error}")
        self.assertIn("clusters in group_id", res_cl.output)

if __name__ == "__main__":
    unittest.main()
