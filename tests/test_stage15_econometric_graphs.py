import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.chart_studio import (
    plot_binscatter,
    plot_rd,
    plot_irf,
    plot_synthetic_control
)


class TestEconometricGraphs(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        # 1. Micro panel / cross-sectional data for binscatter
        n = 500
        x = rng.normal(10.0, 2.0, n)
        group = rng.choice([0, 1], n)
        y = 2.0 + 1.5 * x + 0.8 * group + rng.normal(0, 1.0, n)
        self.df_binscatter = pd.DataFrame({'x': x, 'y': y, 'group': group})

        # 2. Sharp RD data
        n_rd = 600
        x_rd = rng.normal(0.0, 2.0, n_rd)
        d_rd = (x_rd >= 0.0).astype(int)
        y_rd = 1.0 + 2.0 * x_rd + 3.5 * d_rd + rng.normal(0, 0.8, n_rd)
        self.df_rd = pd.DataFrame({'x': x_rd, 'd': d_rd, 'y': y_rd})

        # 3. Macro time-series for IRF
        T = 200
        shock = rng.normal(0, 1.0, T)
        y_ts = np.zeros(T)
        for t in range(1, T):
            y_ts[t] = 0.6 * y_ts[t - 1] + 1.2 * shock[t] + rng.normal(0, 0.5)
        self.df_ts = pd.DataFrame({'time': np.arange(1, T + 1), 'shock': shock, 'y': y_ts})

    def test_binscatter_quantile_computation_and_plot(self):
        """Test binscatter computes 20 quantile bins and linear fit with robust SEs."""
        res = plot_binscatter(self.df_binscatter, 'y', 'x', n_bins=20)
        self.assertIn('bins', res)
        self.assertIn('slope', res)
        self.assertIn('se', res)
        self.assertIn('r2', res)
        self.assertIn('figure', res)
        self.assertEqual(len(res['bins']), 20)
        self.assertAlmostEqual(res['slope'], 1.5, delta=0.25)
        plt.close(res['figure'])

        # Multi-group binscatter
        res_by = plot_binscatter(self.df_binscatter, 'y', 'x', byvar='group', n_bins=10)
        self.assertIn('groups', res_by)
        self.assertEqual(len(res_by['groups']), 2)
        plt.close(res_by['figure'])

    def test_rdplot_local_linear_and_jump(self):
        """Test rdplot estimates local linear jump at cutoff with 95% confidence bands."""
        res = plot_rd(self.df_rd, 'y', 'x', cutoff=0.0, n_bins=20)
        self.assertIn('tau', res)
        self.assertIn('se', res)
        self.assertIn('p_value', res)
        self.assertIn('figure', res)
        self.assertIn('left_bins', res)
        self.assertIn('right_bins', res)
        # Check estimated jump recovers true jump ~3.5
        self.assertAlmostEqual(res['tau'], 3.5, delta=0.8)
        self.assertLess(res['p_value'], 0.001)
        plt.close(res['figure'])

    def test_irf_local_projections_dual_ribbons(self):
        """Test Jordà (2005) local projections IRF with Newey-West HAC errors and dual bands."""
        res = plot_irf(self.df_ts, 'y', 'shock', max_horizon=8)
        self.assertIn('horizons', res)
        self.assertIn('figure', res)
        self.assertEqual(len(res['horizons']), 9)  # 0 to 8
        h0 = res['horizons'][0]
        self.assertAlmostEqual(h0['estimate'], 1.2, delta=0.3)
        self.assertTrue(h0['ci68_lo'] < h0['estimate'] < h0['ci68_hi'])
        self.assertTrue(h0['ci95_lo'] < h0['ci68_lo'])
        self.assertTrue(h0['ci95_hi'] > h0['ci68_hi'])
        plt.close(res['figure'])

    def test_synth_plot_trajectories_and_gap(self):
        """Test synthetic control trajectory and gap plot rendering."""
        synth_data = {
            'time': np.arange(2000, 2011),
            'treated': np.linspace(10, 25, 11),
            'synthetic': np.linspace(10, 20, 11),
            'gap': np.linspace(10, 25, 11) - np.linspace(10, 20, 11),
            'treat_time': 2005
        }
        res = plot_synthetic_control(synth_data)
        self.assertIn('figure', res)
        plt.close(res['figure'])

    def test_interpreter_binscatter_command(self):
        """Test interpreter executes binscatter command and triggers plot callback."""
        ds = Dataset()
        ds.load_dataframe(self.df_binscatter)
        plots_requested = []

        def callback(ptype, data_obj, **kwargs):
            plots_requested.append((ptype, kwargs))

        interp = Interpreter(ds, plot_callback=callback)
        cmd_res = interp.execute("binscatter y x, nbins(15)")
        self.assertTrue(cmd_res.ok, cmd_res.output)
        self.assertEqual(len(plots_requested), 1)
        self.assertEqual(plots_requested[0][0], 'binscatter')

    def test_interpreter_rdplot_command(self):
        """Test interpreter executes rdplot command and triggers plot callback."""
        ds = Dataset()
        ds.load_dataframe(self.df_rd)
        plots_requested = []

        def callback(ptype, data_obj, **kwargs):
            plots_requested.append((ptype, kwargs))

        interp = Interpreter(ds, plot_callback=callback)
        cmd_res = interp.execute("rdplot y x, c(0.0)")
        self.assertTrue(cmd_res.ok, cmd_res.output)
        self.assertEqual(len(plots_requested), 1)
        self.assertEqual(plots_requested[0][0], 'rdplot')

    def test_interpreter_irf_command(self):
        """Test interpreter executes irf command and triggers plot callback."""
        ds = Dataset()
        ds.load_dataframe(self.df_ts)
        plots_requested = []

        def callback(ptype, data_obj, **kwargs):
            plots_requested.append((ptype, kwargs))

        interp = Interpreter(ds, plot_callback=callback)
        cmd_res = interp.execute("irf y shock, horizon(6)")
        self.assertTrue(cmd_res.ok, cmd_res.output)
        self.assertEqual(len(plots_requested), 1)
        self.assertEqual(plots_requested[0][0], 'irf')

    def test_qt_plot_dialog_renders_all_types(self):
        """Test QtPlotDialog renders binscatter, rdplot, irf, and synth_plot."""
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PySide6 import QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from gui_qt.main_window import QtPlotDialog

        # 1. binscatter dialog
        dlg_bin = QtPlotDialog('binscatter', self.df_binscatter, {'yvar': 'y', 'xvar': 'x', 'n_bins': 10})
        self.assertIsNotNone(dlg_bin.figure)
        dlg_bin.close()
        dlg_bin.deleteLater()

        # 2. rdplot dialog
        dlg_rd = QtPlotDialog('rdplot', self.df_rd, {'yvar': 'y', 'xvar': 'x', 'cutoff': 0.0, 'n_bins': 10})
        self.assertIsNotNone(dlg_rd.figure)
        dlg_rd.close()
        dlg_rd.deleteLater()

        # 3. irf dialog
        dlg_irf = QtPlotDialog('irf', self.df_ts, {'response_var': 'y', 'shock_var': 'shock', 'max_horizon': 6})
        self.assertIsNotNone(dlg_irf.figure)
        dlg_irf.close()
        dlg_irf.deleteLater()

        # 4. synth_plot dialog
        synth_data = {
            'time': np.arange(2000, 2011),
            'treated': np.linspace(10, 25, 11),
            'synthetic': np.linspace(10, 20, 11),
            'gap': np.linspace(0, 5, 11),
            'treat_time': 2005
        }
        dlg_synth = QtPlotDialog('synth_plot', synth_data)
        self.assertIsNotNone(dlg_synth.figure)
        dlg_synth.close()
        dlg_synth.deleteLater()

        app.processEvents()


if __name__ == '__main__':
    unittest.main()
