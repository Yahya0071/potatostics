"""
tests/test_causal_and_export.py - Comprehensive Unit Tests for Phase 3:
1. Callaway & Sant'Anna (2021) Staggered DiD (csdid)
2. Synthetic Control Method (synth) with convex optimization and placebo permutation
3. Dynamic Event-Study regression (eventstudy) with pre-trend Wald test
4. Multi-Model Exporter (esttab): ASCII, LaTeX booktabs, Word docx, HTML, CSV
5. Academic Chart Studio (chart_studio / coefplot) with AER, Nature, Economist themes
6. Montiel Olea-Pflueger Weak Instrument Diagnostic (estat weakiv)
7. PyQt6 UI dialogs (CsdidDialog, EsttabDialog)
"""

import os
import unittest
import numpy as np
import pandas as pd

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.causal_modern import (
    compute_csdid, compute_synth, compute_event_study
)
from engine.chart_studio import plot_event_study


class TestCausalModernAndExporters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create temp output directory for file export tests
        cls.test_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
        os.makedirs(cls.test_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        import shutil
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        np.random.seed(123)
        self.dataset = Dataset()
        self.interp = Interpreter(self.dataset)

        # Build balanced panel dataset for staggered DiD & Event Study
        # 12 units observed across 6 periods: 2010 to 2015
        # Cohorts: units 1-4 treated in 2012, units 5-8 treated in 2014, units 9-12 never treated (0)
        units = []
        times = []
        cohorts = []
        outcomes = []
        covars = []

        for u in range(1, 13):
            if u <= 4:
                g = 2012
            elif u <= 8:
                g = 2014
            else:
                g = 0  # never treated

            u_fe = np.random.normal(0, 1.0)
            for t in range(2010, 2016):
                t_fe = 0.5 * (t - 2010)
                is_treated = (g > 0) and (t >= g)
                true_att = 3.5 if is_treated else 0.0
                covar_val = np.random.normal(1.0, 0.3)
                noise = np.random.normal(0, 0.4)
                y = 10.0 + u_fe + t_fe + 0.8 * covar_val + true_att + noise

                units.append(u)
                times.append(t)
                cohorts.append(g)
                outcomes.append(y)
                covars.append(covar_val)

        self.panel_df = pd.DataFrame({
            'unit': units,
            'time': times,
            'cohort': cohorts,
            'y': outcomes,
            'x': covars
        })

    def test_01_csdid_algorithmic(self):
        """Test compute_csdid mathematical engine."""
        res = compute_csdid(self.panel_df, 'y', 'unit', 'time', 'cohort', control_type='notyettreated')
        
        self.assertEqual(res['model_type'], 'csdid')
        self.assertEqual(res['n_units'], 12)
        self.assertEqual(len(res['cohorts']), 2)
        self.assertIn(2012, res['cohorts'])
        self.assertIn(2014, res['cohorts'])

        # Overall ATT should be positive and close to true effect of 3.5
        self.assertGreater(res['overall_att'], 2.0)
        self.assertLess(res['overall_att'], 5.0)
        self.assertGreater(res['overall_se'], 0.0)
        self.assertLess(res['overall_p'], 0.05)

        # Dynamic effects should cover relative event times
        dyn = res['dynamic_effects']
        self.assertGreater(len(dyn), 0)
        event_times = [d['event_time'] for d in dyn]
        self.assertTrue(any(e < 0 for e in event_times))
        self.assertTrue(any(e >= 0 for e in event_times))

        # Pre-trend test should be present
        self.assertIn('pretrend_p', res)
        self.assertGreaterEqual(res['pretrend_p'], 0.0)

    def test_02_csdid_interpreter_and_registers(self):
        """Test csdid command execution and return registers."""
        self.dataset.load_dataframe(self.panel_df)
        self.interp.dataset = self.dataset

        cmd = "csdid y, ivar(unit) timevar(time) gvar(cohort) control(notyettreated)"
        res = self.interp.execute(cmd)

        self.assertTrue(res.ok)
        self.assertIn("Callaway & Sant'Anna (2021) Staggered Difference-in-Differences", res.output)
        self.assertIn("Overall ATT", res.output)
        self.assertIn("Pre-trend test", res.output)

        # Check return registers
        self.assertIn("r(att)", self.interp.macro_manager.return_scalars)
        self.assertIn("r(se)", self.interp.macro_manager.return_scalars)
        self.assertIn("r(p_pretrend)", self.interp.macro_manager.return_scalars)
        self.assertGreater(self.interp.macro_manager.return_scalars["r(att)"], 2.0)

    def test_03_synth_algorithmic(self):
        """Test compute_synth Synthetic Control Method convex donor matching."""
        # Unit 1 is treated at time 2013, units 2..12 are donor pool
        res = compute_synth(
            self.panel_df,
            depvar='y',
            idvar='unit',
            timevar='time',
            treated_id='1',
            treat_time=2013,
            predictors=['x'],
            run_placebos=True
        )

        self.assertEqual(res['model_type'], 'synth')
        self.assertEqual(res['treated_id'], '1')
        self.assertEqual(res['treat_time'], 2013)
        self.assertGreater(res['pre_mspe'], 0.0)
        self.assertGreater(res['post_mspe'], 0.0)
        self.assertGreater(res['mspe_ratio'], 0.0)

        # Donor weights sum to approximately 1.0
        total_w = sum(w['weight'] for w in res['weights'])
        self.assertAlmostEqual(total_w, 1.0, places=2)
        for w in res['weights']:
            self.assertGreaterEqual(w['weight'], 0.0)
            self.assertLessEqual(w['weight'], 1.0)

        # Trajectory has all time periods
        self.assertEqual(len(res['trajectory']), len(sorted(self.panel_df['time'].unique())))

    def test_04_synth_interpreter_and_registers(self):
        """Test synth command execution and return registers."""
        self.dataset.load_dataframe(self.panel_df)
        self.interp.dataset = self.dataset

        cmd = "synth y x, trunit(1) trperiod(2013) unit(unit) time(time)"
        res = self.interp.execute(cmd)

        self.assertTrue(res.ok)
        self.assertIn("Synthetic Control Method Estimation", res.output)
        self.assertIn("Post/Pre MSPE Ratio", res.output)
        self.assertIn("Synthetic Weight", res.output)

        self.assertIn("r(pre_mspe)", self.interp.macro_manager.return_scalars)
        self.assertIn("r(post_mspe)", self.interp.macro_manager.return_scalars)
        self.assertIn("r(mspe_ratio)", self.interp.macro_manager.return_scalars)

    def test_05_event_study_algorithmic_and_graph(self):
        """Test compute_event_study regression and vector graph export."""
        res = compute_event_study(
            self.panel_df,
            depvar='y',
            unitvar='unit',
            timevar='time',
            treat_time_var='cohort',
            pre_window=2,
            post_window=2
        )

        self.assertEqual(res['model_type'], 'eventstudy')
        self.assertIn('event_table', res)
        # Verify k = -1 is reference period with 0 coef
        ref = [row for row in res['event_table'] if row['k'] == -1]
        self.assertEqual(len(ref), 1)
        self.assertEqual(ref[0]['coef'], 0.0)
        self.assertEqual(ref[0]['se'], 0.0)

        # Test graph generation
        graph_path = os.path.join(self.test_dir, "test_eventstudy.png")
        plot_event_study(res, theme='aer', save_path=graph_path)
        self.assertTrue(os.path.exists(graph_path))
        self.assertGreater(os.path.getsize(graph_path), 1000)

    def test_06_event_study_interpreter(self):
        """Test eventstudy command via interpreter."""
        self.dataset.load_dataframe(self.panel_df)
        self.interp.dataset = self.dataset

        graph_path = os.path.join(self.test_dir, "test_event_cmd.png")
        cmd = f"eventstudy y, unit(unit) time(time) treattime(cohort) pre(2) post(2) savegraph({graph_path})"
        res = self.interp.execute(cmd)

        self.assertTrue(res.ok)
        self.assertIn("Dynamic Event-Study Regression", res.output)
        self.assertIn("Pre-trend test", res.output)
        self.assertIn("(ref)", res.output)
        self.assertTrue(os.path.exists(graph_path))

        self.assertIn("r(chi2_pretrend)", self.interp.macro_manager.return_scalars)
        self.assertIn("r(p_pretrend)", self.interp.macro_manager.return_scalars)

    def _populate_regression_models(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 2.0 + 1.5 * x1 - 0.8 * x2 + np.random.normal(0, 0.5, n)
        df_reg = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})
        self.dataset.load_dataframe(df_reg)
        self.interp.dataset = self.dataset
        self.interp.execute("regress y x1")
        self.interp.execute("estimates store Model1")
        self.interp.execute("regress y x1 x2")
        self.interp.execute("estimates store Model2")

    def test_07_esttab_multi_model_comparison(self):
        """Test esttab command storing multiple models and displaying comparison table."""
        self._populate_regression_models()

        # Run esttab console table
        tab_res = self.interp.execute("esttab Model1 Model2")
        self.assertTrue(tab_res.ok)
        self.assertIn("Model1", tab_res.output)
        self.assertIn("Model2", tab_res.output)
        self.assertIn("x1", tab_res.output)
        self.assertIn("x2", tab_res.output)
        self.assertIn("Observations", tab_res.output)
        self.assertIn("R-squared", tab_res.output)

    def test_08_esttab_latex_booktabs_export(self):
        """Test esttab LaTeX booktabs file export."""
        self._populate_regression_models()
        tex_path = os.path.join(self.test_dir, "test_table.tex")
        res = self.interp.execute(f"esttab using {tex_path}, booktabs title(\"Empirical Results\")")
        self.assertTrue(res.ok)
        self.assertTrue(os.path.exists(tex_path))

        with open(tex_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn(r"\begin{table}", content)
        self.assertIn(r"\toprule", content)
        self.assertIn(r"\midrule", content)
        self.assertIn(r"\bottomrule", content)
        self.assertIn("Empirical Results", content)

    def test_09_esttab_word_docx_export(self):
        """Test esttab Word (.docx) file export."""
        self._populate_regression_models()
        docx_path = os.path.join(self.test_dir, "test_table.docx")
        res = self.interp.execute(f"esttab using {docx_path}, docx")
        self.assertTrue(res.ok)
        self.assertTrue(os.path.exists(docx_path))
        self.assertGreater(os.path.getsize(docx_path), 5000)

    def test_10_esttab_html_and_csv_export(self):
        """Test esttab HTML and CSV export."""
        self._populate_regression_models()
        html_path = os.path.join(self.test_dir, "test_table.html")
        csv_path = os.path.join(self.test_dir, "test_table.csv")

        r_html = self.interp.execute(f"esttab using {html_path}, html")
        self.assertTrue(r_html.ok)
        self.assertTrue(os.path.exists(html_path))
        with open(html_path, "r", encoding="utf-8") as f:
            h_text = f.read()
        self.assertIn("<table class='potatostics-table'>", h_text)

        r_csv = self.interp.execute(f"esttab using {csv_path}, csv")
        self.assertTrue(r_csv.ok)
        self.assertTrue(os.path.exists(csv_path))
        df_csv = pd.read_csv(csv_path)
        self.assertIn("Variable", df_csv.columns)

    def test_11_coefplot_generation(self):
        """Test coefplot command with academic themes."""
        self._populate_regression_models()
        png_path = os.path.join(self.test_dir, "test_coefplot.png")
        res = self.interp.execute(f"coefplot, save({png_path}) theme(nature)")
        self.assertTrue(res.ok)
        self.assertTrue(os.path.exists(png_path))
        self.assertGreater(os.path.getsize(png_path), 1000)

    def test_12_weak_iv_diagnostics(self):
        """Test Montiel Olea-Pflueger effective F statistic on IV regression."""
        np.random.seed(42)
        n = 150
        z = np.random.normal(0, 1, n)
        u = np.random.normal(0, 1, n)
        x = 0.8 * z + 0.5 * u + np.random.normal(0, 0.5, n)
        y = 2.0 * x + u + np.random.normal(0, 0.5, n)
        w = np.random.normal(0, 1, n)

        df_iv = pd.DataFrame({'y': y, 'x': x, 'z': z, 'w': w})
        self.dataset.load_dataframe(df_iv)
        self.interp.dataset = self.dataset

        # Estimate 2SLS
        r_iv = self.interp.execute("ivregress 2sls y (x = z) w")
        self.assertTrue(r_iv.ok)

        # Run estat weakiv
        r_diag = self.interp.execute("estat weakiv")
        self.assertTrue(r_diag.ok)
        self.assertIn("First-stage F (classical)", r_diag.output)
        self.assertIn("First-stage F statistic", r_diag.output)
        self.assertIn("Stock-Yogo rule-of-thumb threshold", r_diag.output)

        # Check return register
        self.assertIn("r(effective_F)", self.interp.macro_manager.return_scalars)
        self.assertGreater(self.interp.macro_manager.return_scalars["r(effective_F)"], 10.0)

    def test_13_gui_dialogs_csdid_esttab(self):
        """Test GUI bridge dialogs for CsdidDialog and EsttabDialog."""
        from PySide6 import QtWidgets
        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        from gui_qt.dialogs import CsdidDialog, EsttabDialog

        self.dataset.load_dataframe(self.panel_df)
        dlg_csdid = CsdidDialog(self.dataset)
        dlg_csdid.cmb_dep.setCurrentText("y")
        dlg_csdid.cmb_ivar.setCurrentText("unit")
        dlg_csdid.cmb_timevar.setCurrentText("time")
        dlg_csdid.cmb_gvar.setCurrentText("cohort")
        dlg_csdid.update_syntax()

        syntax_csdid = dlg_csdid.txt_syntax.text()
        self.assertIn("csdid y", syntax_csdid)
        self.assertIn("ivar(unit)", syntax_csdid)
        self.assertIn("timevar(time)", syntax_csdid)
        self.assertIn("gvar(cohort)", syntax_csdid)

        dlg_esttab = EsttabDialog(self.dataset)
        dlg_esttab.cmb_format.setCurrentText("LaTeX (booktabs)")
        dlg_esttab.txt_file.setText("results.tex")
        dlg_esttab.update_syntax()

        syntax_esttab = dlg_esttab.txt_syntax.text()
        self.assertIn("esttab using results.tex", syntax_esttab)
        self.assertIn("booktabs", syntax_esttab)


if __name__ == '__main__':
    unittest.main()
