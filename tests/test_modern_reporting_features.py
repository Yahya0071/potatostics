"""
tests/test_modern_reporting_features.py - Modern 18 & ModernReporting Feature Implementations
Covers:
1. dtable: Demographic / Balance Table Exporter (Table 1)
2. ologit & estat brant: Ordered Logit & Brant test of parallel regression
3. Conley (1999) Spatial HAC standard errors
4. did_multiplegt: de Chaisemartin & D'Haultfoeuille (2020) DiD with switching treatments
5. svyset & svy: Taylor linearization survey regression
"""

import os
import shutil
import unittest
import numpy as np
import pandas as pd
from engine.dataset import Dataset
from engine.interpreter import Interpreter

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
AUTO_PATH = os.path.join(DATA_DIR, "auto.dta").replace("\\", "/")


class TestModernReportingFeatures(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset()
        self.interpreter = Interpreter(self.dataset)
        os.makedirs("test_outputs", exist_ok=True)

    def tearDown(self):
        if os.path.exists("test_outputs"):
            shutil.rmtree("test_outputs", ignore_errors=True)

    def test_01_dtable_balance_table(self):
        """Modern 18 dtable: Demographic / Balance Table 1 generation and export."""
        res_load = self.interpreter.execute(f'use "{AUTO_PATH}", clear')
        self.assertTrue(res_load.ok)
        # Run dtable with continuous and categorical variables split by foreign
        cmd = "dtable price weight mpg i.foreign, by(foreign) export(test_outputs/dtable_test.csv)"
        res = self.interpreter.execute(cmd)
        self.assertTrue(res.ok, f"dtable failed: {res.error}")

        # Verify output text structure
        output = res.output
        self.assertIn("Domestic", output)
        self.assertIn("Foreign", output)
        self.assertIn("Total", output)
        self.assertIn("p-value", output)
        self.assertIn("price", output)
        self.assertIn("weight", output)
        self.assertIn("mpg", output)

        # Verify return registers
        self.assertEqual(self.interpreter.registers.get_r("N"), 74)
        self.assertEqual(self.interpreter.registers.get_r("k_vars"), 4)
        self.assertEqual(self.interpreter.registers.get_r("byvar"), "foreign")

        # Verify exported CSV
        self.assertTrue(os.path.exists("test_outputs/dtable_test.csv"))
        df_csv = pd.read_csv("test_outputs/dtable_test.csv")
        self.assertFalse(df_csv.empty)

    def test_02_ologit_and_brant_test(self):
        """ModernReporting: Ordered Logit (ologit) and Brant test of parallel regression (estat brant)."""
        self.interpreter.execute(f'use "{AUTO_PATH}", clear')
        # rep78 is ordinal 1 to 5. Fit ologit
        res_fit = self.interpreter.execute("ologit rep78 price mpg foreign")
        self.assertTrue(res_fit.ok, f"ologit failed: {res_fit.error}")
        output = res_fit.output
        self.assertIn("Ordered logistic regression", output)
        self.assertIn("/cut1", output)
        self.assertIn("/cut2", output)
        self.assertIn("/cut3", output)
        self.assertIn("/cut4", output)

        # Run estat brant
        res_brant = self.interpreter.execute("estat brant")
        self.assertTrue(res_brant.ok, f"estat brant failed: {res_brant.error}")
        out_b = res_brant.output
        self.assertIn("Brant test of parallel regression", out_b)
        self.assertIn("All", out_b)
        self.assertIn("price", out_b)
        self.assertIn("mpg", out_b)
        self.assertIn("foreign", out_b)

        # Check return registers
        self.assertIsNotNone(self.interpreter.registers.get_r("chi2"))
        self.assertIsNotNone(self.interpreter.registers.get_r("p"))
        self.assertIsNotNone(self.interpreter.registers.get_r("df"))

    def test_03_spatial_hac_conley(self):
        """ModernReporting: Conley (1999) Spatial HAC Standard Errors via vce(spatial, ...)."""
        np.random.seed(42)
        n = 100
        # Simulated spatial coordinates (e.g. US state centroids)
        lats = np.random.uniform(30.0, 45.0, size=n)
        lons = np.random.uniform(-100.0, -75.0, size=n)
        x = np.random.normal(0, 1, size=n)
        # Spatially correlated error
        dist_mat = np.sqrt((lats[:, None] - lats[None, :])**2 + (lons[:, None] - lons[None, :])**2)
        cov_err = np.exp(-dist_mat / 5.0)
        err = np.random.multivariate_normal(np.zeros(n), cov_err)
        y = 1.5 + 2.0 * x + err

        df_spatial = pd.DataFrame({'y': y, 'x': x, 'latitude': lats, 'longitude': lons})
        csv_path = "test_outputs/spatial_test.csv"
        df_spatial.to_csv(csv_path, index=False)

        self.interpreter.execute(f"use {csv_path}, clear")
        # Run OLS first to get baseline SE
        res_ols = self.interpreter.execute("regress y x")
        self.assertTrue(res_ols.ok)
        self.assertGreater(self.interpreter.registers.se["x"], 0.0)

        # Run Conley Spatial HAC with 500km cutoff
        cmd_spatial = "regress y x, vce(spatial, lat(latitude) lon(longitude) cutoff(500))"
        res_spatial = self.interpreter.execute(cmd_spatial)
        self.assertTrue(res_spatial.ok, f"spatial regression failed: {res_spatial.error}")
        out_spatial = res_spatial.output
        self.assertIn("Spatial HAC", out_spatial)
        self.assertIn("Conley", out_spatial)

        se_spatial = self.interpreter.registers.se["x"]
        self.assertGreater(se_spatial, 0.0)
        self.assertEqual(self.interpreter.registers.get_e("vce"), "spatial")

    def test_04_did_multiplegt_switching(self):
        """ModernReporting: de Chaisemartin & D'Haultfoeuille (2020) did_multiplegt for reversible/switching treatments."""
        np.random.seed(123)
        n_units = 60
        n_times = 5
        records = []
        for i in range(1, n_units + 1):
            unit_fe = np.random.normal(0, 1)
            # Reversible treatment pattern:
            # Group 1 (1..20): never treated
            # Group 2 (21..40): treated at t=3,4, untreated at t=5 (switching back)
            # Group 3 (41..60): treated at t=4,5
            for t in range(1, n_times + 1):
                time_fe = 0.5 * t
                if 21 <= i <= 40 and t in (3, 4):
                    d = 1
                elif 41 <= i <= 60 and t in (4, 5):
                    d = 1
                else:
                    d = 0
                y = 10.0 + unit_fe + time_fe + 3.5 * d + np.random.normal(0, 0.2)
                records.append({'unit': i, 'year': t, 'd': d, 'y': y})

        df_panel = pd.DataFrame(records)
        csv_path = "test_outputs/did_multiplegt_test.csv"
        df_panel.to_csv(csv_path, index=False)

        self.interpreter.execute(f"use {csv_path}, clear")
        cmd = "did_multiplegt y unit year d, dynamic(1) placebo(1)"
        res = self.interpreter.execute(cmd)
        self.assertTrue(res.ok, f"did_multiplegt failed: {res.error}")
        output = res.output
        self.assertIn("DID_M", output)
        self.assertIn("Switchers", output)
        self.assertIn("Average instantaneous effect", output)

        # Check estimate is close to 3.5
        est = self.interpreter.registers.get_e("b_did_m")
        self.assertAlmostEqual(est, 3.5, delta=0.5)
        self.assertEqual(self.interpreter.registers.get_e("cmd"), "did_multiplegt")

    def test_05_svyset_and_svy_regress(self):
        """Modern 18: svyset and svy: prefix for complex survey design and Taylor linearization."""
        self.interpreter.execute(f'use "{AUTO_PATH}", clear')
        # Configure survey design: weights=trunk, strata=foreign, psu=turn
        res_svyset = self.interpreter.execute("svyset [pweight=trunk], strata(foreign) psu(turn)")
        self.assertTrue(res_svyset.ok, f"svyset failed: {res_svyset.error}")
        out_set = res_svyset.output
        self.assertIn("pweight", out_set)
        self.assertIn("trunk", out_set)
        self.assertIn("foreign", out_set)
        self.assertIn("turn", out_set)

        # Run svy: regress
        res_svy_reg = self.interpreter.execute("svy: regress price mpg weight")
        self.assertTrue(res_svy_reg.ok, f"svy: regress failed: {res_svy_reg.error}")
        out_reg = res_svy_reg.output
        self.assertIn("Survey: Linear regression", out_reg)
        self.assertIn("Number of strata", out_reg)
        self.assertIn("Number of PSUs", out_reg)
        self.assertIn("Design df", out_reg)
        self.assertIn("Linearized", out_reg)

        # Registers
        self.assertEqual(self.interpreter.registers.get_e("cmd"), "svy: regress")
        self.assertGreater(self.interpreter.registers.get_e("df_r"), 0)
        self.assertEqual(self.interpreter.registers.get_e("N"), 74)

    def test_06_forest_plot_meta_analysis(self):
        """ModernReporting: Forest plot enhancements for meta-analysis with study weights & pooled diamond."""
        from engine.chart_studio import plot_forest
        studies = [
            {'study': 'Smith (2018)', 'theta': 0.45, 'se': 0.12},
            {'study': 'Johnson (2019)', 'theta': 0.62, 'se': 0.15},
            {'study': 'Williams (2020)', 'theta': 0.38, 'se': 0.10},
            {'study': 'Brown (2021)', 'theta': 0.55, 'se': 0.14},
            {'study': 'Jones (2022)', 'theta': 0.50, 'se': 0.11},
        ]
        save_file = "test_outputs/meta_forest.png"
        plot_forest(studies, save_path=save_file)
        self.assertTrue(os.path.exists(save_file))
        self.assertGreater(os.path.getsize(save_file), 1000)

    def test_07_grid_find_and_replace_logic(self):
        """Modern 18: Data Editor Find & Replace across cells."""
        self.interpreter.execute(f'use "{AUTO_PATH}", clear')
        df = self.interpreter.dataset.df
        orig_buick_count = (df['make'].str.contains("Buick", na=False)).sum()
        self.assertGreater(orig_buick_count, 0)

        # Execute replace logic across dataset
        from gui_qt.data_grid import perform_find_replace
        count = perform_find_replace(self.interpreter.dataset, "Buick", "PotatoBuick")
        self.assertEqual(count, orig_buick_count)

        new_count = (self.interpreter.dataset.df['make'].str.contains("PotatoBuick", na=False)).sum()
        self.assertEqual(new_count, orig_buick_count)


if __name__ == "__main__":
    unittest.main()
