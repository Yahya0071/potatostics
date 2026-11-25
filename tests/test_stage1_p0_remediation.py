"""
tests/test_stage1_p0_remediation.py - TDD tests for Stage 1 (P0 Statistical Integrity,
Numerical Stability, and DTA Round-Trip Metadata).
"""

import os
import unittest
import tempfile
import numpy as np
import pandas as pd

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.estimation import compute_logit, compute_regression, compute_ivregress_2sls
from engine.advanced_econometrics import compute_probit, compute_poisson, compute_tobit, compute_qreg
from engine.formatter import format_logit, format_probit, format_poisson, format_tobit, format_qreg
from engine.bigdata import StreamingRegressionAccumulator

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
AUTO_PATH = os.path.join(DATA_DIR, "auto.dta").replace("\\", "/")


class TestStage1P0Remediation(unittest.TestCase):
    def setUp(self):
        self.ds = Dataset()
        self.interp = Interpreter(self.ds)

    # =========================================================================
    # 1.1 Shared Convergence Contract & Optimizer Robustness
    # =========================================================================

    def test_logit_complete_separation_detected(self):
        """Logit with complete separation must set converged=False and report separation."""
        df_sep = pd.DataFrame({
            'y': [0, 0, 0, 0, 1, 1, 1, 1],
            'x': [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]
        })
        res = compute_logit(df_sep, 'y', ['x'], max_iter=25)
        self.assertFalse(res['converged'], "Logit should not claim convergence on complete separation")
        self.assertIn("separation", res['convergence_status'].lower())
        self.assertIn("warnings", res)
        self.assertTrue(len(res['warnings']) > 0 or not res['converged'])

        # Formatter must visibly flag non-convergence
        formatted = format_logit(res)
        self.assertIn("WARNING", formatted)
        self.assertIn("DID NOT CONVERGE", formatted)

        # Interpreter command must prominently display non-convergence warning banner
        self.ds.load_dataframe(df_sep)
        cmd_res = self.interp.execute("logit y x")
        self.assertIn("WARNING: ESTIMATES DID NOT CONVERGE", cmd_res.output)
        self.assertIn("Complete or quasi-complete separation detected", cmd_res.output)
        self.assertEqual(self.interp.registers.get_e("converged"), 0)

    def test_shared_convergence_contract_fields(self):
        """Verify that logit, probit, poisson, tobit, and qreg all provide the unified convergence contract."""
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n)
        df_mle = pd.DataFrame({
            'y_bin': np.random.binomial(1, 0.5, n),
            'y_cnt': np.random.poisson(3, n),
            'y_cont': 2.0 * x + np.random.normal(0, 1, n),
            'x': x
        })

        estimators = [
            ("logit", compute_logit(df_mle, 'y_bin', ['x'])),
            ("probit", compute_probit(df_mle, 'y_bin', ['x'])),
            ("poisson", compute_poisson(df_mle, 'y_cnt', ['x'])),
            ("tobit", compute_tobit(df_mle, 'y_cont', ['x'])),
            ("qreg", compute_qreg(df_mle, 'y_cont', ['x']))
        ]

        required_keys = ['converged', 'convergence_status', 'iterations', 'max_iter', 'gradient_norm', 'warnings']
        for name, res in estimators:
            for k in required_keys:
                self.assertIn(k, res, f"Estimator '{name}' is missing convergence contract field '{k}'")
            self.assertIsInstance(res['converged'], bool, f"'{name}' converged must be boolean")
            self.assertIsInstance(res['convergence_status'], str, f"'{name}' convergence_status must be string")
            self.assertIsInstance(res['iterations'], int, f"'{name}' iterations must be integer")
            self.assertIsInstance(res['max_iter'], int, f"'{name}' max_iter must be integer")
            self.assertIsInstance(res['gradient_norm'], float, f"'{name}' gradient_norm must be float")
            self.assertIsInstance(res['warnings'], list, f"'{name}' warnings must be list")

    def test_probit_poisson_non_convergence_formatting(self):
        """Formatters for probit, poisson, tobit must visibly flag non-convergence."""
        dummy_res = {
            'depvar': 'y', 'n': 50, 'df_m': 1, 'lr_chi2': 0.0, 'prob_chi2': 1.0,
            'pseudo_r2': 0.0, 'll': -100.0, 'converged': False,
            'convergence_status': 'Maximum iterations reached without convergence',
            'coefs': [{'name': 'x', 'coef': 1.0, 'se': 1.0, 'z': 1.0, 't': 1.0, 'p': 0.3, 'ci_lower': -1.0, 'ci_upper': 3.0}],
            'n_uncensored': 30, 'n_left': 10, 'n_right': 10, 'quantile': 0.5, 'df_r': 48
        }
        for fmt, name in [(format_probit, "probit"), (format_poisson, "poisson"), (format_tobit, "tobit"), (format_qreg, "qreg")]:
            out = fmt(dummy_res)
            self.assertIn("WARNING", out, f"{name} formatter did not show non-convergence warning")
            self.assertIn("DID NOT CONVERGE", out, f"{name} formatter did not show DID NOT CONVERGE")

    # =========================================================================
    # 1.2 Numerically Stable Least Squares
    # =========================================================================

    def test_ols_collinear_and_ill_conditioned_stability(self):
        """Pivoted QR / stable solver must detect collinear regressors and omit them rather than exploding."""
        np.random.seed(123)
        n = 100
        x1 = np.linspace(1, 10, n)
        # Perfectly collinear x2 = 2 * x1
        x2 = 2.0 * x1
        # Near-collinear x3 = x1 + 1e-13 * noise
        x3 = x1 + 1e-13 * np.random.randn(n)
        y = 3.0 * x1 + 5.0 + np.random.randn(n)
        df_col = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'x3': x3})

        res = compute_regression(df_col, 'y', ['x1', 'x2', 'x3'])
        # Must detect rank deficiency: rank should be 2 (x1 + constant), not 4
        self.assertEqual(res['rank'], 2)
        self.assertTrue(len(res['omitted_vars']) >= 2)
        self.assertIn('x2', res['omitted_vars'])

        # Non-omitted coefficient on x1 should be close to 3.0 (not wild +/- 1000)
        x1_coef = [c for c in res['coefs'] if c['name'] == 'x1'][0]
        self.assertAlmostEqual(x1_coef['coef'], 3.0, delta=0.5)

        # Omitted variables must be clearly marked
        x2_coef = [c for c in res['coefs'] if c['name'] == 'x2'][0]
        self.assertTrue(x2_coef.get('omitted', False))

    def test_ivregress_2sls_stable_projection(self):
        """2SLS projection onto Z must use stable QR projection without singular matrix crash."""
        np.random.seed(42)
        n = 100
        z1 = np.random.randn(n)
        z2 = z1.copy()  # Collinear instrument
        x = 0.8 * z1 + 0.2 * np.random.randn(n)
        y = 2.0 * x + np.random.randn(n)
        df_iv = pd.DataFrame({'y': y, 'x': x, 'z1': z1, 'z2': z2})

        # Should handle or cleanly identify collinear instrument rank deficiency
        res = compute_ivregress_2sls(df_iv, 'y', ['x'], [], ['z1', 'z2'])
        self.assertTrue(np.isfinite(res['coefs'][0]['coef']))
        self.assertAlmostEqual(res['coefs'][0]['coef'], 2.0, delta=0.6)

    def test_streaming_ols_cancellation_protection(self):
        """Streaming OLS must not produce negative SSR or NaN when outcome has huge offset."""
        np.random.seed(42)
        n = 1000
        x = np.random.uniform(10, 20, n)
        # Huge baseline offset with high R2 (typical cancellation trigger in y'y - b'X'y)
        y = 1e8 + 3.5 * x + np.random.normal(0, 0.01, n)

        acc = StreamingRegressionAccumulator(k=2, add_constant=True)
        for i in range(0, n, 100):
            batch_x = x[i:i+100, None]
            batch_y = y[i:i+100]
            acc.update(batch_x, batch_y)

        res = acc.finalize(['x'])
        self.assertTrue(np.isfinite(res['r2']))
        self.assertTrue(0.0 <= res['r2'] <= 1.0)
        self.assertTrue(res['ssr'] >= 0.0)
        self.assertAlmostEqual(res['coefs'][0]['coef'], 3.5, delta=0.1)

    # =========================================================================
    # 1.3 Standard DTA Metadata Round-Trip
    # =========================================================================

    def test_dta_value_labels_survive_round_trip(self):
        """DTA value labels and mappings must survive save and reload end-to-end."""
        self.ds.load_dta(AUTO_PATH)
        # Verify original loaded labels
        self.assertIn('foreign', self.ds.val_labels)
        orig_mapping = self.ds.val_labels['foreign']
        self.assertEqual(orig_mapping.get(0), 'Domestic')
        self.assertEqual(orig_mapping.get(1), 'Foreign')
        self.assertEqual(self.ds.val_label_names.get('foreign'), 'foreign')

        # Save to temp DTA and reload into fresh Dataset instance
        temp_dta = tempfile.mktemp(suffix=".dta")
        try:
            self.ds.save_dta(temp_dta, version=117)
            ds_reload = Dataset()
            ds_reload.load_dta(temp_dta)

            # Assert value labels survived
            self.assertIn('foreign', ds_reload.val_labels)
            reloaded_mapping = ds_reload.val_labels['foreign']
            self.assertEqual(reloaded_mapping.get(0), 'Domestic')
            self.assertEqual(reloaded_mapping.get(1), 'Foreign')
            self.assertEqual(ds_reload.val_label_names.get('foreign'), 'foreign')

            # Ensure data grid model displays resolved labels "Domestic" / "Foreign"
            from gui_qt.data_grid import VirtualPandasModel
            from PySide6.QtCore import Qt
            from PySide6 import QtWidgets
            _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            model = VirtualPandasModel(ds_reload)
            col_idx = list(ds_reload.df.columns).index('foreign')
            # Check row 0 (Domestic)
            first_val = model.data(model.index(0, col_idx), Qt.ItemDataRole.DisplayRole)
            self.assertIn(first_val, ['Domestic', 'Foreign'])
        finally:
            if os.path.exists(temp_dta):
                os.remove(temp_dta)


if __name__ == '__main__':
    unittest.main()
