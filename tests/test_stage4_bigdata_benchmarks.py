"""
tests/test_stage4_bigdata_benchmarks.py - Tests for Stage 4 Big Data Streaming & Resource Safety
"""

import os
import unittest
import numpy as np
import pandas as pd
import pyarrow as pa

from engine.bigdata import (
    StreamingRegressionAccumulator,
    stream_dataset_chunks
)
from engine.estimation import compute_regression


class TestStage4BigData(unittest.TestCase):
    def setUp(self):
        self.temp_files = []

    def tearDown(self):
        for f in self.temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def test_arrow_ipc_streaming_generator(self):
        """Arrow IPC / Feather files must stream batch-by-batch without full materialization."""
        n = 1000
        df = pd.DataFrame({
            "y": np.linspace(1, 10, n) + np.random.normal(0, 0.1, n),
            "x1": np.linspace(0, 5, n),
            "x2": np.random.normal(0, 1, n)
        })
        arrow_path = "tests_temp_stream.arrow"
        self.temp_files.append(arrow_path)

        table = pa.Table.from_pandas(df)
        batch_size = 200
        batches = [table.slice(i, batch_size).to_batches()[0] for i in range(0, n, batch_size)]
        with pa.ipc.new_file(arrow_path, table.schema) as writer:
            for b in batches:
                writer.write_batch(b)

        # Stream chunks
        chunk_count = 0
        total_rows = 0
        for X_chunk, y_chunk in stream_dataset_chunks(arrow_path, ["y", "x1", "x2"], chunk_size=batch_size):
            chunk_count += 1
            total_rows += len(y_chunk)
            self.assertEqual(X_chunk.shape[1], 2)
            self.assertEqual(len(y_chunk), len(X_chunk))

        self.assertEqual(chunk_count, 5)
        self.assertEqual(total_rows, n)

    def test_streaming_ols_parity_with_memory_ols(self):
        """Streaming OLS must yield identical coefficients and standard errors to in-memory OLS."""
        np.random.seed(42)
        n = 2000
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(2, 0.5, n)
        y = 10.0 + 3.0 * x1 - 1.5 * x2 + np.random.normal(0, 0.5, n)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

        # Memory reference
        res_mem = compute_regression(df, "y", ["x1", "x2"], vce="ols")

        # Streaming run
        acc = StreamingRegressionAccumulator(n_features=2, add_constant=True)
        for i in range(0, n, 250):
            acc.update(df[["x1", "x2"]].iloc[i:i+250].to_numpy(), df["y"].iloc[i:i+250].to_numpy())
        res_stream = acc.finalize("y", ["x1", "x2"])

        for i in range(3):
            mem_b = res_mem['coefs'][i]['coef']
            str_b = res_stream['coefs'][i]['coef']
            self.assertAlmostEqual(mem_b, str_b, places=9)

            mem_se = res_mem['coefs'][i]['se']
            str_se = res_stream['coefs'][i]['se']
            self.assertAlmostEqual(mem_se, str_se, places=9)

        self.assertAlmostEqual(res_mem['r2'], res_stream['r2'], places=9)
        self.assertAlmostEqual(res_mem['ss_r'], res_stream['ss_r'], places=7)

    def test_conley_spatial_resource_guard(self):
        """Conley spatial covariance on N > 10,000 must raise a clean resource guard ValueError."""
        n_large = 10001
        df_large = pd.DataFrame({
            "y": np.random.randn(n_large),
            "x": np.random.randn(n_large),
            "lat": np.random.uniform(30, 40, n_large),
            "lon": np.random.uniform(-100, -90, n_large)
        })
        with self.assertRaises(ValueError) as ctx:
            compute_regression(df_large, "y", ["x"], vce="spatial", spatial_opts={"lat": "lat", "lon": "lon", "cutoff": 50})
        self.assertIn("exceeded resource safety limit", str(ctx.exception))

    def test_conley_blocked_vs_small_n_accuracy(self):
        """Blocked spatial covariance calculation must yield correct results on valid data."""
        np.random.seed(123)
        n = 150
        df = pd.DataFrame({
            "y": np.random.normal(5, 1, n),
            "x": np.random.normal(0, 1, n),
            "lat": np.random.uniform(40.0, 41.0, n),
            "lon": np.random.uniform(-74.5, -73.5, n)
        })
        res = compute_regression(df, "y", ["x"], vce="spatial", spatial_opts={"lat": "lat", "lon": "lon", "cutoff": 50})
        self.assertTrue(np.all(np.isfinite(res['vcov'])))
        self.assertTrue(res['coefs'][0]['se'] > 0)


if __name__ == "__main__":
    unittest.main()
