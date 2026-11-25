import unittest
import numpy as np
import pandas as pd
from engine.dataset import Dataset
from engine.interpreter import Interpreter

class TestSprint4Subsystems(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        # Time series data
        t = np.arange(1, n + 1)
        trend = 0.5 * t
        cycle = 2.0 * np.sin(2 * np.pi * t / 20.0)
        y = trend + cycle + np.random.randn(n) * 0.2
        y1 = 0.6 * np.roll(y, 1) + np.random.randn(n) * 0.5
        y2 = 0.4 * np.roll(y, 1) - 0.3 * np.roll(y1, 1) + np.random.randn(n) * 0.5

        # Survival data
        fail_time = np.random.exponential(scale=10.0, size=n) + 1.0
        died = (np.random.rand(n) > 0.3).astype(int)
        age = np.random.normal(50, 10, n)
        drug = np.random.binomial(1, 0.5, n)

        self.df = pd.DataFrame({
            'time': t,
            'y': y,
            'y1': y1,
            'y2': y2,
            'fail_time': fail_time,
            'died': died,
            'age': age,
            'drug': drug
        })
        self.dataset = Dataset()
        self.dataset.load_dataframe(self.df.copy())
        self.interp = Interpreter(self.dataset)

    # --- 1. tsfilter hp ---
    def test_01_tsfilter_hp(self):
        """Verify tsfilter hp decomposes series into trend and cyclical components."""
        self.interp.execute("tsset time")
        res = self.interp.execute("tsfilter hp cycle_y = y, smooth(1600) trend(trend_y)")
        self.assertTrue(res.ok, f"tsfilter hp failed: {res.error}")

        df = self.interp.dataset.df
        self.assertIn("cycle_y", df.columns)
        self.assertIn("trend_y", df.columns)
        # Cycle + Trend must equal original y
        recon = df['cycle_y'] + df['trend_y']
        np.testing.assert_allclose(recon.to_numpy(), df['y'].to_numpy(), rtol=1e-5, atol=1e-5)

    # --- 2. var (Vector Autoregression) ---
    def test_02_var(self):
        """Verify var multi-equation autoregression and AIC/SBIC."""
        self.interp.execute("tsset time")
        res = self.interp.execute("var y1 y2, lags(1 2)")
        self.assertTrue(res.ok, f"var failed: {res.error}")
        self.assertIn("Vector autoregression", res.output)
        self.assertIn("AIC", res.output)
        self.assertIn("SBIC", res.output)
        self.assertIsNotNone(self.interp.registers.b.get("y1:L1.y1"))
        self.assertIsNotNone(self.interp.registers.b.get("y2:L1.y2"))

    # --- 3. stset & stcox (Survival Analysis) ---
    def test_03_survival_stset_and_stcox(self):
        """Verify stset survival setup and stcox Cox proportional hazards regression."""
        res_set = self.interp.execute("stset fail_time, failure(died)")
        self.assertTrue(res_set.ok, f"stset failed: {res_set.error}")

        df = self.interp.dataset.df
        self.assertIn("_t", df.columns)
        self.assertIn("_d", df.columns)
        self.assertIn("_st", df.columns)

        res_cox = self.interp.execute("stcox age drug")
        self.assertTrue(res_cox.ok, f"stcox failed: {res_cox.error}")
        self.assertIn("Cox proportional hazards regression", res_cox.output)
        self.assertIn("Haz. ratio", res_cox.output)
        self.assertIsNotNone(self.interp.registers.b.get("age"))
        self.assertIsNotNone(self.interp.registers.b.get("drug"))

if __name__ == "__main__":
    unittest.main()

