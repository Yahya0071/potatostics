import unittest
import numpy as np
import pandas as pd
from engine.bigdata import StreamingRegressionAccumulator
from engine.estimation import compute_regression

class TestStreamingOLSStability(unittest.TestCase):
    def test_p1_streaming_near_collinear_30000_rows(self):
        """
        P1 Test: 30,000 observations with near-collinear x2 = x1 + 1e-8 * noise.
        Streaming OLS must NOT return wild coefficients (+/- 1000s).
        It must omit x2 as collinear and estimate beta_x1 ~ 1.5 and beta_cons ~ 1.0,
        matching in-memory QR.
        """
        np.random.seed(42)
        n = 30000
        x1 = np.random.randn(n)
        noise = np.random.randn(n)
        x2 = x1 + 1e-8 * noise
        y = 1.0 + 2.0 * x1 - 0.5 * x2 + np.random.randn(n)

        # In-memory reference
        df_mem = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})
        res_mem = compute_regression(df_mem, 'y', ['x1', 'x2'])
        self.assertIn('x2', res_mem['omitted_vars'])
        x1_mem = [c['coef'] for c in res_mem['coefs'] if c['name'] == 'x1'][0]
        self.assertAlmostEqual(x1_mem, 1.5, delta=0.05)

        # Streaming accumulator
        acc = StreamingRegressionAccumulator(k=3, add_constant=True)
        chunk_size = 5000
        for i in range(0, n, chunk_size):
            X_b = np.column_stack([x1[i:i+chunk_size], x2[i:i+chunk_size]])
            y_b = y[i:i+chunk_size]
            acc.update(X_b, y_b)

        res_stream = acc.finalize('y', ['x1', 'x2'])
        
        # Must detect collinearity and omit x2, keeping rank at 2
        self.assertEqual(res_stream['rank'], 2)
        self.assertIn('x2', res_stream['omitted_vars'])
        
        x1_stream = [c['coef'] for c in res_stream['coefs'] if c['name'] == 'x1'][0]
        x2_stream = [c['coef'] for c in res_stream['coefs'] if c['name'] == 'x2'][0]
        
        # Must be close to 1.5 and NOT wild +/- 10,000
        self.assertAlmostEqual(x1_stream, 1.5, delta=0.05)
        self.assertEqual(x2_stream, 0.0) # omitted
        self.assertTrue(any(c['name'] == 'x2' and c.get('omitted') for c in res_stream['coefs']))

    def test_p1_streaming_scale_invariance(self):
        """Streaming OLS handles predictor scales from 1e-12 to 1e12 without failure."""
        np.random.seed(123)
        n = 2000
        x_base = np.random.randn(n)
        y = 2.0 + 3.0 * x_base + np.random.randn(n)

        base_coef = None
        for scale in [1e-12, 1e-6, 1.0, 1e6, 1e12]:
            acc = StreamingRegressionAccumulator(k=2, add_constant=True)
            for i in range(0, n, 500):
                acc.update((x_base[i:i+500] * scale)[:, None], y[i:i+500])
            res = acc.finalize('y', ['x'])
            self.assertEqual(res['omitted_vars'], [])
            x_coef = [c['coef'] for c in res['coefs'] if c['name'] == 'x'][0]
            recovered_slope = x_coef * scale
            if base_coef is None:
                base_coef = recovered_slope
            else:
                np.testing.assert_allclose(recovered_slope, base_coef, rtol=1e-5)

    def test_p1_streaming_chunk_concatenation_collinearity(self):
        """Columns that become collinear only after chunk concatenation are properly detected and omitted."""
        # Chunk 1: x1 and x2 look distinct
        # Chunk 2: x2 is exactly 2 * x1
        x1_part1 = np.array([1.0, 2.0, 3.0, 4.0])
        x2_part1 = np.array([2.0, 4.0, 6.0, 8.0])
        y_part1 = np.array([5.0, 7.0, 9.0, 11.0])

        x1_part2 = np.array([5.0, 6.0, 7.0, 8.0])
        x2_part2 = np.array([10.0, 12.0, 14.0, 16.0])
        y_part2 = np.array([13.0, 15.0, 17.0, 19.0])

        acc = StreamingRegressionAccumulator(k=3, add_constant=True)
        acc.update(np.column_stack([x1_part1, x2_part1]), y_part1)
        acc.update(np.column_stack([x1_part2, x2_part2]), y_part2)

        res = acc.finalize('y', ['x1', 'x2'])
        self.assertEqual(res['rank'], 2)
        self.assertIn('x2', res['omitted_vars'])

    def test_p1_streaming_hc1_robust_covariance(self):
        """Two-pass streaming HC1 matches in-memory robust regression."""
        np.random.seed(42)
        n = 1000
        x = np.random.randn(n)
        y = 1.0 + 2.5 * x + np.random.randn(n) * (1.0 + 0.5 * np.abs(x))

        df = pd.DataFrame({'y': y, 'x': x})
        res_mem = compute_regression(df, 'y', ['x'], vce='hc1')

        acc = StreamingRegressionAccumulator(k=2, add_constant=True)
        chunk = 200
        for i in range(0, n, chunk):
            acc.update(x[i:i+chunk, None], y[i:i+chunk])

        def second_pass():
            for i in range(0, n, chunk):
                yield x[i:i+chunk, None], y[i:i+chunk]

        res_stream = acc.finalize('y', ['x'], vce='hc1', second_pass_generator=second_pass())
        x_se_mem = [c['se'] for c in res_mem['coefs'] if c['name'] == 'x'][0]
        x_se_stream = [c['se'] for c in res_stream['coefs'] if c['name'] == 'x'][0]
        np.testing.assert_allclose(x_se_stream, x_se_mem, rtol=1e-3)

from engine.dataset import Dataset
from gui_qt.data_grid import VirtualPandasModel
from PySide6.QtCore import Qt

class TestDataGridFilteredEditsAndViewSorting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create headless QCoreApplication / QApplication if not exists
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_p1_filtered_grid_edits_update_canonical_dataset(self):
        """Editing a visible row in a filtered grid must mutate canonical dataset.df."""
        ds = Dataset()
        df = pd.DataFrame({
            'make': ['Toyota', 'Honda', 'Ford', 'Tesla'],
            'price': [20000, 22000, 25000, 45000]
        })
        ds.load_dataframe(df)

        model = VirtualPandasModel(ds)
        # Filter for 'Tesla'
        model.set_filter('Tesla')
        self.assertEqual(model.rowCount(), 1)

        # Visible row 0 corresponds to Tesla (canonical row 3)
        # Edit price to 42000
        idx = model.index(0, 1) # row 0, col 1 ('price')
        model.setData(idx, 42000, Qt.ItemDataRole.EditRole)

        # Clear filter
        model.set_filter('')
        self.assertEqual(model.rowCount(), 4)

        # Canonical dataset row for Tesla must have price 42000
        self.assertEqual(ds.df.loc[3, 'price'], 42000)

    def test_p1_grid_sorting_is_view_only(self):
        """Header click sorting must NOT mutate underlying dataset.df row order."""
        ds = Dataset()
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'val': [300, 100, 200]
        })
        ds.load_dataframe(df)

        original_order = ds.df['id'].tolist() # [1, 2, 3]

        model = VirtualPandasModel(ds)
        # Sort ascending by 'val' (col 1)
        model.sort_by_column(1)

        # Underlying dataset order must be completely unchanged
        self.assertEqual(ds.df['id'].tolist(), original_order)

        # Visible top row in model should be row with val 100 (id 2)
        idx_top_val = model.index(0, 1)
        self.assertEqual(model.data(idx_top_val, Qt.ItemDataRole.EditRole), "100")

        # Edit top row 'val' to 150
        model.setData(idx_top_val, 150, Qt.ItemDataRole.EditRole)

        # The edit must update canonical row for id 2, not row 0 (id 1)
        row_id2 = ds.df[ds.df['id'] == 2].iloc[0]
        self.assertEqual(row_id2['val'], 150)
        row_id1 = ds.df[ds.df['id'] == 1].iloc[0]
        self.assertEqual(row_id1['val'], 300)

    def test_p1_filter_edit_refresh_and_types(self):
        """Filter, edit, call update_data(), and test int/float/string/missing/duplicate index."""
        ds = Dataset()
        # Duplicate index: [0, 0, 1]
        df = pd.DataFrame({
            'cat': ['A', 'B', 'B'],
            'int_val': [10, 20, 30],
            'flt_val': [1.5, 2.5, 3.5],
            'str_val': ['foo', 'bar', 'baz']
        }, index=[0, 0, 1])
        ds.load_dataframe(df)

        model = VirtualPandasModel(ds)
        # Filter for 'B' -> matches rows 1 and 2
        model.set_filter('B')
        self.assertEqual(model.rowCount(), 2)

        # Edit float cell in visible row 1 (which is canonical row 2)
        idx_flt = model.index(1, 2) # flt_val
        model.setData(idx_flt, 99.5, Qt.ItemDataRole.EditRole)

        # Edit int cell to missing '.' in visible row 0 (which is canonical row 1)
        idx_int = model.index(0, 1) # int_val
        model.setData(idx_int, ".", Qt.ItemDataRole.EditRole)

        # Refresh
        model.update_data()

        # Check canonical dataset
        self.assertEqual(ds.df.iloc[2]['flt_val'], 99.5)
        self.assertTrue(pd.isna(ds.df.iloc[1]['int_val']))

        # Clear filter
        model.set_filter('')
        self.assertEqual(model.rowCount(), 3)
        self.assertEqual(ds.df.iloc[2]['flt_val'], 99.5)
        self.assertTrue(pd.isna(ds.df.iloc[1]['int_val']))

    def test_p1_sort_and_filter_composition(self):
        """Filtering and sorting compose correctly on view mappings."""
        ds = Dataset()
        df = pd.DataFrame({
            'group': ['G1', 'G2', 'G1', 'G2', 'G1'],
            'val': [50, 10, 30, 20, 40]
        })
        ds.load_dataframe(df)

        model = VirtualPandasModel(ds)
        # Filter for G1 -> values 50 (row 0), 30 (row 2), 40 (row 4)
        model.set_filter('G1')
        self.assertEqual(model.rowCount(), 3)

        # Sort ascending by val (col 1)
        model.sort_by_column(1)
        # Visible rows should be: 30, 40, 50
        v0 = model.data(model.index(0, 1), Qt.ItemDataRole.DisplayRole)
        v1 = model.data(model.index(1, 1), Qt.ItemDataRole.DisplayRole)
        v2 = model.data(model.index(2, 1), Qt.ItemDataRole.DisplayRole)
        self.assertEqual(v0, "30")
        self.assertEqual(v1, "40")
        self.assertEqual(v2, "50")

        # Edit visible row 0 (which is val 30, canonical row 2) to 35
        model.setData(model.index(0, 1), 35, Qt.ItemDataRole.EditRole)
        self.assertEqual(ds.df.iloc[2]['val'], 35)

        # Underlying dataframe order should NOT have been mutated
        self.assertEqual(ds.df['group'].tolist(), ['G1', 'G2', 'G1', 'G2', 'G1'])

class TestThreadSafetyAndCancellation(unittest.TestCase):
    def test_p1_no_qthread_terminate_in_source(self):
        """Assert no production code path anywhere in gui_qt invokes QThread.terminate()."""
        import os
        gui_dir = os.path.join(os.path.dirname(__file__), "..", "gui_qt")
        for root, _, files in os.walk(gui_dir):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        self.assertNotIn(".terminate()", content,
                                         f"Unsafe QThread.terminate() found in {filepath}")

    def test_p1_command_worker_cooperative_cancel(self):
        """CommandWorker sets cancellation token and signals cooperative cancellation."""
        from gui_qt.main_window import CommandWorker
        from engine.interpreter import Interpreter
        from engine.dataset import Dataset

        ds = Dataset()
        interp = Interpreter(ds)
        worker = CommandWorker(interp, "summarize")
        self.assertFalse(worker._is_cancelled)
        worker.cancel()
        self.assertTrue(worker._is_cancelled)

if __name__ == '__main__':
    unittest.main()
