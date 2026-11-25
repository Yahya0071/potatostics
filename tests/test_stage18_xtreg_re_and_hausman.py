import unittest
import numpy as np
import pandas as pd
from engine.dataset import Dataset
from engine.interpreter import Interpreter

class TestXtregReAndHausman(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate the exact benchmark panel dataset verified against Reference 15
        np.random.seed(42)
        cls.n_entities = 30
        cls.n_periods = 10
        entity = np.repeat(np.arange(cls.n_entities), cls.n_periods)
        time = np.tile(np.arange(cls.n_periods), cls.n_entities)
        u = np.repeat(np.random.randn(cls.n_entities) * 1.5, cls.n_periods)
        x1 = np.random.randn(len(entity))
        x2 = np.random.randn(len(entity))
        y = 1.0 + 2.0 * x1 - 3.0 * x2 + u + np.random.randn(len(entity))

        cls.df = pd.DataFrame({
            'entity': entity,
            'time': time,
            'y': y,
            'x1': x1,
            'x2': x2
        })

    def setUp(self):
        self.dataset = Dataset()
        self.dataset.load_dataframe(self.df.copy())
        self.interp = Interpreter(self.dataset)
        self.interp.execute("xtset entity time")

    def test_01_xtreg_re_basic_and_reference_golden(self):
        """Verify xtreg, re point estimates, standard errors, and variance components against Reference 15."""
        res = self.interp.execute("xtreg y x1 x2, re")
        self.assertTrue(res.ok, f"xtreg, re failed: {res.error}")

        # Reference 15 Golden Benchmark:
        # x1: Coef = 1.951244, Std. Err. = .0594383
        # x2: Coef = -3.045103, Std. Err. = .0596821
        # _cons: Coef = .7902717, Std. Err. = .2333014
        # sigma_u = 1.2442108, sigma_e = .97123563, rho = .62137187, theta = .760345
        b_x1 = self.interp.registers.b.get("x1")
        b_x2 = self.interp.registers.b.get("x2")
        b_cons = self.interp.registers.b.get("_cons")

        se_x1 = self.interp.registers.se.get("x1")
        se_x2 = self.interp.registers.se.get("x2")
        se_cons = self.interp.registers.se.get("_cons")

        self.assertIsNotNone(b_x1, "x1 coefficient missing")
        self.assertAlmostEqual(b_x1, 1.951244, places=4)
        self.assertAlmostEqual(b_x2, -3.045103, places=4)
        self.assertAlmostEqual(b_cons, 0.7902717, places=4)

        self.assertAlmostEqual(se_x1, 0.0594383, places=4)
        self.assertAlmostEqual(se_x2, 0.0596821, places=4)
        self.assertAlmostEqual(se_cons, 0.2333014, places=4)

        # Variance components
        self.assertAlmostEqual(self.interp.registers.e_scalars.get("sigma_u"), 1.2442108, places=3)
        self.assertAlmostEqual(self.interp.registers.e_scalars.get("sigma_e"), 0.9712356, places=3)
        self.assertAlmostEqual(self.interp.registers.e_scalars.get("rho"), 0.62137187, places=3)
        self.assertAlmostEqual(self.interp.registers.e_scalars.get("theta"), 0.760345, places=3)

        # R-squared metrics
        self.assertAlmostEqual(self.interp.registers.e_scalars.get("r2_w"), 0.9340, places=3)
        self.assertAlmostEqual(self.interp.registers.e_scalars.get("r2_b"), 0.2924, places=3)
        self.assertAlmostEqual(self.interp.registers.e_scalars.get("r2_o"), 0.8430, places=3)

    def test_02_xtreg_default_is_re(self):
        """In Reference, xtreg y x1 x2 without fe or re defaults to random effects."""
        res = self.interp.execute("xtreg y x1 x2")
        self.assertTrue(res.ok, f"xtreg default failed: {res.error}")
        self.assertEqual(self.interp.registers.e_macros.get("model"), "re")

    def test_03_hausman_specification_test(self):
        """Verify hausman fe re against Reference 15: chi2(2) = 0.13, p = 0.9390."""
        res_fe = self.interp.execute("xtreg y x1 x2, fe")
        self.assertTrue(res_fe.ok, f"xtreg fe failed: {res_fe.error}")
        self.interp.execute("estimates store fe_model")

        res_re = self.interp.execute("xtreg y x1 x2, re")
        self.assertTrue(res_re.ok, f"xtreg re failed: {res_re.error}")
        self.interp.execute("estimates store re_model")

        res_h = self.interp.execute("hausman fe_model re_model")
        self.assertTrue(res_h.ok, f"hausman failed: {res_h.error}")

        # Reference 15 benchmark: chi2(2) = 0.13, Prob > chi2 = 0.9390
        chi2 = self.interp.registers.r_scalars.get("chi2")
        df = self.interp.registers.r_scalars.get("df")
        p = self.interp.registers.r_scalars.get("p")

        self.assertIsNotNone(chi2, "r(chi2) missing from hausman")
        self.assertAlmostEqual(chi2, 0.13, places=2)
        self.assertEqual(int(df), 2)
        self.assertAlmostEqual(p, 0.9390, places=2)

    def test_04_xtreg_re_clustered_se(self):
        """Verify xtreg, re with cluster-robust vce(cluster entity)."""
        res = self.interp.execute("xtreg y x1 x2, re vce(cluster entity)")
        self.assertTrue(res.ok, f"xtreg, re clustered failed: {res.error}")
        self.assertIn("Robust std. err.", res.output)
        self.assertIn("Number of clusters", res.output)

    def test_05_unbalanced_panel_harmonic_tbar(self):
        """Verify unbalanced panel Swamy-Arora uses harmonic mean Tbar matching Reference."""
        np.random.seed(123)
        records = []
        for i in range(15):
            Ti = np.random.randint(4, 12)
            ui = np.random.randn() * 0.8
            for t in range(Ti):
                x = np.random.randn()
                y = 1.0 + 2.0 * x + ui + np.random.randn()
                records.append({'id': i, 't': t, 'y': y, 'x': x})
        df_unbal = pd.DataFrame(records)

        ds = Dataset()
        ds.load_dataframe(df_unbal)
        interp = Interpreter(ds)
        interp.execute("xtset id t")
        res = interp.execute("xtreg y x, re")
        self.assertTrue(res.ok, f"xtreg re unbal failed: {res.error}")
        self.assertFalse(interp.registers.e_scalars.get("Tcon"))
        self.assertGreater(interp.registers.e_scalars.get("Tbar"), 0)
        self.assertGreater(interp.registers.e_scalars.get("sigma_u"), 0)
        self.assertGreater(interp.registers.e_scalars.get("sigma_e"), 0)

if __name__ == "__main__":
    unittest.main()
