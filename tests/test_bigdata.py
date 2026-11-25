"""
tests/test_bigdata.py - Unit Tests for Phase 4: Big Data & High-Performance Core
1. StreamingRegressionAccumulator (Out-of-Core X'X Gram Matrix Accumulation)
2. Equivalence to full-memory OLS (numerical precision to 10^-10)
3. 2-pass streaming heteroskedasticity-robust (HC1) standard errors
4. Apache Parquet / Feather zero-copy serialization and loading
5. Out-of-core estimation directly on .parquet disk files
6. bigregress command in Interpreter and return registers
7. Polars-accelerated multi-threaded summary statistics and group collapse
"""

import os
import unittest
import numpy as np
import pandas as pd

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.bigdata import (
    StreamingRegressionAccumulator,
    compute_streaming_ols,
    read_parquet_or_arrow,
    write_parquet_or_arrow,
    polars_fast_summary,
    polars_fast_collapse,
    HAS_POLARS
)


class TestBigDataCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(os.path.dirname(__file__), "test_bigdata_outputs")
        os.makedirs(cls.test_dir, exist_ok=True)

        # Generate synthetic benchmark dataset (10,000 rows across 5 chunks)
        np.random.seed(42)
        n = 10000
        cls.x1 = np.random.normal(2.0, 1.5, n)
        cls.x2 = np.random.uniform(10.0, 50.0, n)
        cls.x3 = np.random.normal(-1.0, 0.5, n)
        cls.cat = np.random.choice(["A", "B", "C", "D"], size=n)
        
        # True beta: cons = 5.0, b1 = 2.5, b2 = -0.4, b3 = 1.2
        noise = np.random.normal(0, 1.0, n)
        cls.y = 5.0 + 2.5 * cls.x1 - 0.4 * cls.x2 + 1.2 * cls.x3 + noise

        cls.df = pd.DataFrame({
            'y': cls.y,
            'x1': cls.x1,
            'x2': cls.x2,
            'x3': cls.x3,
            'group': cls.cat
        })

        # Save to Parquet
        cls.parquet_path = os.path.join(cls.test_dir, "benchmark.parquet")
        write_parquet_or_arrow(cls.df, cls.parquet_path)

    @classmethod
    def tearDownClass(cls):
        import shutil
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        self.dataset = Dataset()
        self.interp = Interpreter(self.dataset)

    def test_01_accumulator_exact_equivalence_to_ols(self):
        """Verify that streaming accumulator matches analytical OLS down to 10^-10."""
        # Standard analytical OLS
        X_full = np.column_stack([self.x1, self.x2, self.x3, np.ones(len(self.y))])
        inv_XtX_true = np.linalg.inv(X_full.T @ X_full)
        beta_true = inv_XtX_true @ (X_full.T @ self.y)
        ssr_true = float(np.sum((self.y - X_full @ beta_true) ** 2))
        s2_true = ssr_true / (len(self.y) - 4)
        se_true = np.sqrt(np.diag(s2_true * inv_XtX_true))

        # Streaming accumulator with 10 small chunks
        acc = StreamingRegressionAccumulator(n_features=3, add_constant=True)
        chunk_size = 1000
        for start in range(0, len(self.y), chunk_size):
            X_chunk = np.column_stack([
                self.x1[start:start + chunk_size],
                self.x2[start:start + chunk_size],
                self.x3[start:start + chunk_size]
            ])
            y_chunk = self.y[start:start + chunk_size]
            acc.update(X_chunk, y_chunk)

        res = acc.finalize('y', ['x1', 'x2', 'x3'], vce='classical')

        # Check equivalence
        np.testing.assert_allclose(res['beta'], beta_true, rtol=1e-9, atol=1e-9)
        coef_se = [c['se'] for c in res['coefs']]
        np.testing.assert_allclose(coef_se, se_true, rtol=1e-9, atol=1e-9)
        self.assertAlmostEqual(res['ss_r'], ssr_true, places=5)
        self.assertEqual(res['n'], 10000)

    def test_02_compute_streaming_ols_with_robust_se(self):
        """Test streaming OLS with 2-pass heteroskedasticity-robust standard errors."""
        res_homo = compute_streaming_ols(self.df, 'y', ['x1', 'x2', 'x3'], vce='classical', chunk_size=2000)
        res_robust = compute_streaming_ols(self.df, 'y', ['x1', 'x2', 'x3'], vce='robust', chunk_size=2000)

        # Coefficients must be identical
        np.testing.assert_allclose(res_homo['beta'], res_robust['beta'])
        # Standard errors should be positive and finite
        for c in res_robust['coefs']:
            self.assertGreater(c['se'], 0.0)
            self.assertTrue(np.isfinite(c['t']))

    def test_03_parquet_serialization_and_dataset_integration(self):
        """Test reading and writing Apache Parquet via dataset and standalone helpers."""
        test_pq = os.path.join(self.test_dir, "test_io.parquet")
        write_parquet_or_arrow(self.df, test_pq)
        self.assertTrue(os.path.exists(test_pq))

        df_read = read_parquet_or_arrow(test_pq)
        self.assertEqual(len(df_read), len(self.df))
        self.assertListEqual(list(df_read.columns), list(self.df.columns))

        # Test through Dataset.load_parquet
        d = Dataset()
        n, k = d.load_parquet(test_pq)
        self.assertEqual(n, 10000)
        self.assertEqual(k, 5)

    def test_04_streaming_ols_directly_from_parquet_disk(self):
        """Run out-of-core streaming regression directly on a .parquet file."""
        res = compute_streaming_ols(self.parquet_path, 'y', ['x1', 'x2', 'x3'], vce='robust', chunk_size=2500)
        self.assertEqual(res['n'], 10000)
        self.assertEqual(res['depvar'], 'y')
        self.assertEqual(len(res['coefs']), 4)
        
        # Verify estimated parameters match true DGP
        b_dict = {c['name']: c['coef'] for c in res['coefs']}
        self.assertAlmostEqual(b_dict['x1'], 2.5, delta=0.1)
        self.assertAlmostEqual(b_dict['x2'], -0.4, delta=0.1)
        self.assertAlmostEqual(b_dict['x3'], 1.2, delta=0.1)
        self.assertAlmostEqual(b_dict['_cons'], 5.0, delta=0.3)

    def test_05_bigregress_interpreter_command(self):
        """Test bigregress command in interpreter with disk streaming and return registers."""
        cmd = f"bigregress y x1 x2 x3 using \"{self.parquet_path}\", robust chunksize(2000)"
        res = self.interp.execute(cmd)

        self.assertTrue(res.ok)
        self.assertIn("Linear regression", res.output)
        self.assertIn("Number of obs", res.output)
        self.assertIn("R-squared", res.output)

        # Check return registers
        self.assertEqual(self.interp.registers.e_scalars['N'], 10000)
        self.assertIn('x1', self.interp.registers.b)
        self.assertIn('x2', self.interp.registers.b)
        self.assertIn('x3', self.interp.registers.b)
        self.assertIn('_cons', self.interp.registers.b)
        self.assertGreater(self.interp.registers.e_scalars['r2'], 0.8)

    def test_06_regress_using_parquet_file(self):
        """Test standard 'regress dep indeps using file.parquet' auto-delegation."""
        cmd = f"regress y x1 x2 using \"{self.parquet_path}\""
        res = self.interp.execute(cmd)
        self.assertTrue(res.ok)
        self.assertIn("Number of obs", res.output)
        self.assertIn("10,000", res.output)

    def test_07_use_and_save_parquet_commands(self):
        """Test 'use file.parquet, clear' and 'save newfile.parquet, replace'."""
        use_cmd = f"use \"{self.parquet_path}\", clear"
        res_use = self.interp.execute(use_cmd)
        self.assertTrue(res_use.ok)
        self.assertIn("obs: 10,000", res_use.output)

        save_target = os.path.join(self.test_dir, "saved_via_cmd.parquet")
        save_cmd = f"save \"{save_target}\", replace"
        res_save = self.interp.execute(save_cmd)
        self.assertTrue(res_save.ok)
        self.assertTrue(os.path.exists(save_target))

    @unittest.skipUnless(HAS_POLARS, "Polars not installed")
    def test_08_polars_fast_summary_and_collapse(self):
        """Test Polars-accelerated multi-threaded summary and group collapse."""
        summary = polars_fast_summary(self.df, ['x1', 'x2'])
        self.assertEqual(len(summary), 2)
        s_x1 = summary[0]
        self.assertEqual(s_x1['var'], 'x1')
        self.assertEqual(s_x1['n'], 10000)
        self.assertAlmostEqual(s_x1['mean'], float(self.df['x1'].mean()), places=4)
        self.assertAlmostEqual(s_x1['sd'], float(self.df['x1'].std()), places=4)

        # Fast collapse by group
        collapsed = polars_fast_collapse(self.df, {'mean': ['y', 'x1'], 'count': ['x2']}, by_vars=['group'])
        self.assertEqual(len(collapsed), 4)  # 4 categories: A, B, C, D
        self.assertIn('group', collapsed.columns)
        self.assertIn('y', collapsed.columns)


if __name__ == '__main__':
    unittest.main()
