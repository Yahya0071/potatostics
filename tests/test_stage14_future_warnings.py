import unittest
import warnings
import numpy as np
import pandas as pd
from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.timeseries import compute_dfuller, compute_corrgram


class TestFutureWarningsCompatibility(unittest.TestCase):
    def setUp(self):
        # Enforce that any FutureWarning raises an error as in CI
        warnings.simplefilter("error", FutureWarning)

    def tearDown(self):
        warnings.simplefilter("default", FutureWarning)

    def test_data_ops_no_reliance_on_groupby_groups_dict(self):
        """Assert that engine/data_ops.py does not rely on grouped.groups dictionary keys."""
        import inspect
        import engine.data_ops as data_ops
        src = inspect.getsource(data_ops)
        self.assertNotIn("grouped.groups", src, "engine/data_ops.py still relies on grouped.groups")
        self.assertNotIn(".groups.items()", src, "engine/data_ops.py still iterates over .groups.items()")

    def test_data_ops_by_group_no_future_warnings(self):
        """Test by-groups execution and egen/replace without FutureWarning or groupby().groups reliance."""
        ds = Dataset()
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C', 'C'],
            'x': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        })
        ds.load_dataframe(df)
        interp = Interpreter(ds)

        # 1. by group: egen
        res_egen = interp.execute("by group: egen mean_x = mean(x)")
        self.assertTrue(res_egen.ok, res_egen.output)
        self.assertIn("mean_x", ds.df.columns)

        # 2. by group: gen with lag
        res_lag = interp.execute("by group: gen lag_x = x[_n-1]")
        self.assertTrue(res_lag.ok, res_lag.output)

        # 3. by group: summarize
        res_sum = interp.execute("by group: summarize x")
        self.assertTrue(res_sum.ok, res_sum.output)
        self.assertIn("group = A", res_sum.output)
        self.assertIn("group = B", res_sum.output)

    def test_timeseries_adfuller_and_acf_result_object(self):
        """Test ADF and corrgram with result_object support under strict FutureWarning filter."""
        rng = np.random.default_rng(42)
        series = np.cumsum(rng.normal(0, 1, 100))

        res_adf = compute_dfuller(series, regression='c', lags=2)
        self.assertIn('stat', res_adf)
        self.assertIn('p_value', res_adf)
        self.assertIn('lags_used', res_adf)
        self.assertIn('critical_values', res_adf)

        res_acf = compute_corrgram(series, max_lags=6)
        self.assertIn('rows', res_acf)
        self.assertEqual(len(res_acf['rows']), 6)
        for r in res_acf['rows']:
            self.assertIn('acf', r)
            self.assertIn('pacf', r)
            self.assertIn('q_stat', r)
            self.assertIn('p_value', r)


if __name__ == '__main__':
    unittest.main()
