import unittest
import numpy as np
import pandas as pd
from engine.dataset import Dataset
from engine.interpreter import Interpreter

class TestDataOpsSprint2(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'id': np.arange(1, 11),
            'group': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            'val': [10.0, 20.0, 20.0, 40.0, 50.0, 100.0, 100.0, 100.0, 200.0, 300.0],
            'str_col': ["  hello world  ", "analytics", "econometrics", "python", "potatostics",
                        "apple", "banana", "orange", "grape", "fruit salad"]
        })
        self.dataset = Dataset()
        self.dataset.load_dataframe(self.df.copy())
        self.interp = Interpreter(self.dataset)

    # --- 1. Extended egen functions ---
    def test_01_egen_pctile(self):
        """Verify egen pctile(exp), p(#) overall and by groups."""
        res = self.interp.execute("egen p50 = pctile(val), p(50)")
        self.assertTrue(res.ok, f"egen pctile failed: {res.error}")
        self.assertIn("p50", self.interp.dataset.df.columns)
        # Median of 10..300 is (45.0 in linear interpolation or 40/50 depending on method)
        p50_val = self.interp.dataset.df['p50'].iloc[0]
        self.assertAlmostEqual(p50_val, np.percentile(self.df['val'], 50), places=3)

        # By group
        res_by = self.interp.execute("bysort group: egen p50_grp = pctile(val), p(50)")
        self.assertTrue(res_by.ok, f"bysort egen pctile failed: {res_by.error}")
        grp1_p50 = self.interp.dataset.df[self.interp.dataset.df['group'] == 1]['p50_grp'].iloc[0]
        self.assertAlmostEqual(grp1_p50, 20.0, places=3)

    def test_02_egen_std(self):
        """Verify egen std(exp) computes standardized z-scores (mean 0, sd 1)."""
        res = self.interp.execute("egen z_val = std(val)")
        self.assertTrue(res.ok, f"egen std failed: {res.error}")
        z = self.interp.dataset.df['z_val']
        self.assertAlmostEqual(float(z.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(z.std(ddof=1)), 1.0, places=4)

    def test_03_egen_mode(self):
        """Verify egen mode(exp) calculates statistical mode."""
        res = self.interp.execute("bysort group: egen mode_val = mode(val)")
        self.assertTrue(res.ok, f"egen mode failed: {res.error}")
        # group 1 has mode 20, group 2 has mode 100
        grp1_mode = self.interp.dataset.df[self.interp.dataset.df['group'] == 1]['mode_val'].iloc[0]
        grp2_mode = self.interp.dataset.df[self.interp.dataset.df['group'] == 2]['mode_val'].iloc[0]
        self.assertEqual(grp1_mode, 20.0)
        self.assertEqual(grp2_mode, 100.0)

    # --- 2. Date and Time Functions ---
    def test_04_date_time_builtins(self):
        """Verify mdy(), year(), month(), day(), quarter(), date() matching Reference baseline."""
        # Jan 1, 1960 is DTA epoch day 0
        res1 = self.interp.execute("gen d0 = mdy(1, 1, 1960)")
        self.assertTrue(res1.ok, f"gen mdy failed: {res1.error}")
        self.assertEqual(self.interp.dataset.df['d0'].iloc[0], 0)

        # May 12, 1980 is 7437 days
        res2 = self.interp.execute("gen d1 = mdy(5, 12, 1980)")
        self.assertTrue(res2.ok, f"gen mdy 1980 failed: {res2.error}")
        self.assertEqual(self.interp.dataset.df['d1'].iloc[0], 7437)

        # Extraction functions
        self.interp.execute("gen y = year(d1)")
        self.interp.execute("gen m = month(d1)")
        self.interp.execute("gen d = day(d1)")
        self.interp.execute("gen q = quarter(d1)")

        self.assertEqual(self.interp.dataset.df['y'].iloc[0], 1980)
        self.assertEqual(self.interp.dataset.df['m'].iloc[0], 5)
        self.assertEqual(self.interp.dataset.df['d'].iloc[0], 12)
        self.assertEqual(self.interp.dataset.df['q'].iloc[0], 2)

    # --- 3. String Functions ---
    def test_05_string_builtins(self):
        """Verify substr(), strpos(), trim(), lower(), upper(), regexm()."""
        self.interp.execute("gen s_raw = \"  econometrics  \"")
        res_trim = self.interp.execute("gen s_trim = trim(s_raw)")
        self.assertTrue(res_trim.ok, f"trim failed: {res_trim.error}")
        self.assertEqual(self.interp.dataset.df['s_trim'].iloc[0], "econometrics")

        self.interp.execute("gen s_up = upper(s_trim)")
        self.assertEqual(self.interp.dataset.df['s_up'].iloc[0], "ECONOMETRICS")

        self.interp.execute("gen s_sub = substr(s_trim, 1, 7)")
        self.assertEqual(self.interp.dataset.df['s_sub'].iloc[0], "商econome" if False else "econome")

        self.interp.execute("gen pos = strpos(s_trim, \"metric\")")
        self.assertEqual(self.interp.dataset.df['pos'].iloc[0], 6)

        self.interp.execute("gen has_eco = regexm(s_trim, \"^econ.*\")")
        self.assertEqual(self.interp.dataset.df['has_eco'].iloc[0], 1)

if __name__ == "__main__":
    unittest.main()
