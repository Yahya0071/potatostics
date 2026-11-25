import unittest
import numpy as np
import pandas as pd
from engine.causal_modern import compute_csdid


class TestCSDiDBootstrap(unittest.TestCase):
    def test_csdid_pretrend_multiplier_bootstrap_metadata(self):
        """Test that compute_csdid uses multiplier bootstrap for the pre-trend test."""
        # Simple panel with 2 cohorts, 4 periods
        rng = np.random.default_rng(42)
        N = 60
        units = np.repeat(np.arange(N), 4)
        times = np.tile(np.arange(4), N)
        g = np.repeat(np.where(np.arange(N) < 30, 2, 0), 4)
        y = rng.normal(0, 1, len(units))
        df = pd.DataFrame({'id': units, 't': times, 'g': g, 'y': y})

        res = compute_csdid(df, 'y', 'id', 't', 'g', seed=123)
        self.assertIn('pretrend_method', res)
        self.assertIn('multiplier_bootstrap', res['pretrend_method'])
        self.assertTrue(0.0 <= res['pretrend_p'] <= 1.0)
        self.assertGreater(res['pretrend_chi2'], 0.0)

    def test_csdid_pretrend_monte_carlo_rejection_rate(self):
        """
        Monte Carlo test with >= 200 datasets under the true parallel trends null.
        Assert that the empirical rejection rate at alpha=0.05 is <= 7% (matching Callaway & Sant'Anna).
        """
        rng = np.random.default_rng(42)
        n_sims = 200
        rejections = 0
        N = 80
        T = 4

        for sim in range(n_sims):
            units = np.repeat(np.arange(N), T)
            times = np.tile(np.arange(T), N)
            # Cohort 2 treats at t=2 (so t=0, 1 are pre-treatment periods, baseline t=1)
            g = np.repeat(np.where(np.arange(N) < 40, 2, 0), T)
            # True null: parallel trends, y = unit_fe + time_fe + noise
            u_fe = np.repeat(rng.normal(0, 1, N), T)
            t_fe = times * 0.4
            eps = rng.normal(0, 1, len(units))
            y = u_fe + t_fe + eps
            df = pd.DataFrame({'id': units, 't': times, 'g': g, 'y': y})

            # Run csdid with multiplier bootstrap
            res = compute_csdid(df, 'y', 'id', 't', 'g', n_boot=499, seed=sim)
            if res['pretrend_p'] < 0.05:
                rejections += 1

        rej_rate = rejections / n_sims
        print(f"\n[Monte Carlo Short Panel] 200 datasets, rejection rate at 5%: {rej_rate:.3f} ({rejections}/{n_sims})")
        self.assertLessEqual(rej_rate, 0.070, f"csdid pre-trend rejection rate {rej_rate:.3f} exceeds 7%")

    def test_csdid_pretrend_long_panel_monte_carlo_rejection_rate(self):
        """
        Monte Carlo test on longer staggered panel:
        90 units, years 2000-2010 (11 years), cohorts 2005 and 2007 plus never-treated.
        Unit random effect, linear trend, N(0, 1) noise. True parallel trends null.
        200 datasets. Assert empirical rejection rate at alpha=0.05 is <= 7%.
        """
        rng = np.random.default_rng(123)
        n_sims = 200
        rejections = 0
        N = 90
        years = np.arange(2000, 2011)
        T = len(years)

        for sim in range(n_sims):
            units = np.repeat(np.arange(N), T)
            times = np.tile(years, N)
            g = np.repeat(np.where(np.arange(N) < 30, 2005, np.where(np.arange(N) < 60, 2007, 0)), T)
            u_fe = np.repeat(rng.normal(0, 1, N), T)
            trend = times * 0.05
            eps = rng.normal(0, 1, len(units))
            y = u_fe + trend + eps
            df = pd.DataFrame({'id': units, 't': times, 'g': g, 'y': y})

            res = compute_csdid(df, 'y', 'id', 't', 'g', n_boot=499, seed=sim)
            if res['pretrend_p'] < 0.05:
                rejections += 1

        rej_rate = rejections / n_sims
        print(f"\n[Monte Carlo Long Panel] 200 datasets, rejection rate at 5%: {rej_rate:.3f} ({rejections}/{n_sims})")
        self.assertLessEqual(rej_rate, 0.070, f"csdid pre-trend rejection rate {rej_rate:.3f} exceeds 7%")


if __name__ == '__main__':
    unittest.main()
