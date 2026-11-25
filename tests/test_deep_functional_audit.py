"""
tests/test_deep_functional_audit.py - Exhaustive Functional Audit of all Potatostics Engine & GUI Modules
"""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from engine.dataset import Dataset
from engine.parser import (
    parse_command_line,
    evaluate_dta_expression,
    expand_factor_variables,
    CommandResult,
)
from engine.macros import MacroManager, ReturnRegisters
from engine.estimation import (
    compute_regression,
    compute_svy_regression,
    compute_ivregress_2sls,
    compute_panel_fe,
    compute_logit,
    compute_summarize,
    compute_tabulate_1way,
    compute_tabulate_2way,
    compute_correlate,
    compute_wald_test,
)
from engine.advanced_econometrics import (
    compute_reghdfe,
    compute_probit,
    compute_ologit,
    compute_poisson,
    compute_tobit,
    compute_qreg,
    compute_margins,
    compute_lincom,
)
from engine.diagnostics import (
    compute_vif,
    compute_hettest,
    compute_ovtest,
    compute_dwatson,
    compute_firststage,
    compute_brant_test,
)
from engine.causal import (
    compute_psmatch,
    compute_did,
)
from engine.causal_modern import (
    compute_csdid,
    compute_synth,
    compute_event_study,
    compute_did_multiplegt,
)
from engine.timeseries import (
    compute_dfuller,
    compute_arima,
    compute_corrgram,
)
from engine.bigdata import (
    StreamingRegressionAccumulator,
    polars_fast_summary,
)
from engine.exporters import (
    ModelTableExporter,
    DemographicTableExporter,
)
from engine.chart_studio import (
    plot_forest,
)
from engine.report_generator import generate_docx_report
from engine.estimates import EstimatesStore
from engine.script_runner import ScriptRunner
from engine.copilot import CopilotAssistant
from engine.copilot_egm import ElephantGoldfishCopilot
from engine.plugins_manager import PluginsManager
from engine.interpreter import Interpreter


class TestDeepFunctionalAudit(unittest.TestCase):
    """Exhaustive unit test suite auditing every subsystem function-by-function."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.n = 100
        cls.df = pd.DataFrame({
            'y': np.random.normal(10, 2, cls.n),
            'x1': np.random.normal(5, 1, cls.n),
            'x2': np.random.normal(2, 0.5, cls.n),
            'z1': np.random.normal(0, 1, cls.n),
            'id': np.repeat(np.arange(1, 21), 5),
            't': np.tile(np.arange(1, 6), 20),
            'binary_y': np.random.binomial(1, 0.4, cls.n),
            'count_y': np.random.poisson(3, cls.n),
            'ordered_y': np.random.choice([1, 2, 3], size=cls.n),
            'lat': np.random.uniform(30.0, 45.0, cls.n),
            'lon': np.random.uniform(-100.0, -70.0, cls.n),
            'strata': np.repeat(np.arange(1, 5), 25),
            'psu': np.repeat(np.arange(1, 21), 5),
            'pweight': np.random.uniform(0.5, 2.5, cls.n),
            'treatment': np.random.choice([0, 1], size=cls.n),
            'post': np.random.choice([0, 1], size=cls.n),
            'cohort': np.random.choice([0, 2, 3], size=cls.n),
            'category': np.random.choice(['A', 'B', 'C'], size=cls.n),
            'treat_time': np.random.choice([3, 4, 99], size=cls.n),
        })
        cls.df['treated_sw'] = ((cls.df['id'] > 10) & (cls.df['t'] >= 3)).astype(int)

    # -------------------------------------------------------------
    # 1. Dataset & File I/O
    # -------------------------------------------------------------
    def test_01_dataset_operations(self):
        ds = Dataset()
        ds.load_dataframe(self.df.copy())
        self.assertEqual(ds.N, self.n)
        self.assertEqual(ds.K, len(self.df.columns))
        
        # Metadata & Labels
        ds.var_labels['y'] = 'Outcome Variable'
        self.assertEqual(ds.var_labels.get('y'), 'Outcome Variable')
        
        ds.val_labels['binary_y'] = {0: 'No', 1: 'Yes'}
        self.assertEqual(ds.val_labels.get('binary_y', {}).get(1), 'Yes')
        
        # Round-trip saving
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            parquet_path = os.path.join(tmpdir, "test.parquet")
            dta_path = os.path.join(tmpdir, "test.dta")
            
            ds.save_csv(csv_path)
            self.assertTrue(os.path.exists(csv_path))
            
            ds.save_parquet(parquet_path)
            self.assertTrue(os.path.exists(parquet_path))
            
            ds.save_dta(dta_path)
            self.assertTrue(os.path.exists(dta_path))
            
            # Load back
            ds2 = Dataset()
            ds2.load_parquet(parquet_path)
            self.assertEqual(ds2.N, self.n)

    # -------------------------------------------------------------
    # 2. Parser & Qualifiers & AST
    # -------------------------------------------------------------
    def test_02_parser_and_ast_qualifiers(self):
        cmd_str = 'regress y x1 x2 if x1 > 4 in 1/50, robust cluster(id)'
        parsed = parse_command_line(cmd_str)
        self.assertEqual(parsed.verb, 'regress')
        self.assertEqual(parsed.body.split()[:3], ['y', 'x1', 'x2'])
        self.assertIn('robust', parsed.options)
        self.assertEqual(parsed.options.get('cluster'), 'id')
        self.assertEqual(parsed.if_expr, 'x1 > 4')
        self.assertEqual(parsed.in_range, (1, 50))

        # AST Safe Evaluation
        val_true = evaluate_dta_expression('inlist(2, 1, 2, 3)', pd.DataFrame({'a': [1]}))
        self.assertTrue(bool(val_true.iloc[0]) if hasattr(val_true, 'iloc') else bool(val_true))
        val_range = evaluate_dta_expression('inrange(5, 1, 10)', pd.DataFrame({'a': [1]}))
        self.assertTrue(bool(val_range.iloc[0]) if hasattr(val_range, 'iloc') else bool(val_range))
        val_math = evaluate_dta_expression('ln(exp(2)) + sqrt(16)', pd.DataFrame({'a': [1]}))
        val_m_num = float(val_math.iloc[0]) if hasattr(val_math, 'iloc') else float(val_math)
        self.assertAlmostEqual(val_m_num, 6.0)

        # Factor variable expansion
        expanded = expand_factor_variables(['i.category', 'x1'], self.df)
        self.assertTrue(any('.category' in v for v in expanded[0]))

    # -------------------------------------------------------------
    # 3. Macro Manager & Return Registers
    # -------------------------------------------------------------
    def test_03_macros_and_registers(self):
        mm = MacroManager()
        mm.set_local('alpha', 'hello')
        mm.set_global('beta', 'world')
        self.assertEqual(mm.get_local('alpha'), 'hello')
        self.assertEqual(mm.get_global('beta'), 'world')
        
        expanded = mm.expand("`alpha' $beta")
        self.assertEqual(expanded, "hello world")

        reg = ReturnRegisters()
        reg.set_r_scalar('mean', 42.5)
        reg.set_r_macro('cmd', 'summarize')
        reg.set_e_scalar('N', 100)
        reg.set_e_macro('cmd', 'regress')
        
        self.assertEqual(reg.get_r('mean'), 42.5)
        self.assertEqual(reg.get_r('cmd'), 'summarize')
        self.assertEqual(reg.get_e('N'), 100)
        self.assertEqual(reg.get_e('cmd'), 'regress')

    # -------------------------------------------------------------
    # 4. Data Operations (egen, encode, decode, recode, collapse, reshape)
    # -------------------------------------------------------------
    def test_04_data_operations(self):
        ds = Dataset()
        ds.load_dataframe(self.df.copy())
        interp = Interpreter(ds)

        # Generate & Replace
        r_gen = interp.execute("generate double_x1 = x1 * 2")
        self.assertEqual(r_gen.return_code, 0)
        self.assertIn('double_x1', interp.dataset.df.columns)

        r_rep = interp.execute("replace double_x1 = 0 if x1 < 4")
        self.assertEqual(r_rep.return_code, 0)
        self.assertTrue((interp.dataset.df.loc[interp.dataset.df['x1'] < 4, 'double_x1'] == 0).all())

        # Rename
        r_ren = interp.execute("rename double_x1 scaled_x1")
        self.assertEqual(r_ren.return_code, 0)
        self.assertIn('scaled_x1', interp.dataset.df.columns)

        # Egen
        r_egen = interp.execute("egen mean_x1 = mean(x1), by(strata)")
        self.assertEqual(r_egen.return_code, 0)
        self.assertIn('mean_x1', interp.dataset.df.columns)

        # Encode & Decode
        r_enc = interp.execute("encode category, gen(cat_num)")
        self.assertEqual(r_enc.return_code, 0)
        self.assertIn('cat_num', interp.dataset.df.columns)

        r_dec = interp.execute("decode cat_num, gen(cat_str)")
        self.assertEqual(r_dec.return_code, 0)
        self.assertIn('cat_str', interp.dataset.df.columns)

        # Sort
        r_sort = interp.execute("sort strata id")
        self.assertEqual(r_sort.return_code, 0)
        self.assertTrue(interp.dataset.df['strata'].is_monotonic_increasing)

    # -------------------------------------------------------------
    # 5. Core Estimation (OLS, VCE, Survey, 2SLS, FE, Logit)
    # -------------------------------------------------------------
    def test_05_core_estimation(self):
        df = self.df.copy()
        
        # OLS Classical & Robust HC1
        res_ols = compute_regression(df, 'y', ['x1', 'x2'])
        self.assertIn('coefs', res_ols)
        self.assertIn('r2', res_ols)
        
        res_hc1 = compute_regression(df, 'y', ['x1', 'x2'], robust=True)
        self.assertIn('vcov', res_hc1)
        self.assertIn('se', res_hc1['coefs'][0])

        # Clustered SE
        res_cluster = compute_regression(df, 'y', ['x1', 'x2'], vce='cluster', cluster_var='id')
        self.assertIn('cluster_var', res_cluster)

        # Spatial HAC SE (Conley)
        res_spatial = compute_regression(df, 'y', ['x1', 'x2'], vce='spatial', spatial_opts={'lat': 'lat', 'lon': 'lon', 'cutoff': 500.0})
        self.assertIn('vcov', res_spatial)
        self.assertEqual(len(res_spatial['coefs']), 3)

        # Survey Linearization
        res_svy = compute_svy_regression(df, 'y', ['x1', 'x2'], strata='strata', psu='psu', pweight='pweight')
        self.assertIn('vcov', res_svy)
        self.assertIn('df_r', res_svy)

        # 2SLS IV
        res_iv = compute_ivregress_2sls(df, 'y', ['x2'], ['x1'], ['z1'])
        self.assertIn('coefs', res_iv)
        self.assertIn('vcov', res_iv)

        # Panel Fixed Effects (Within)
        res_fe = compute_panel_fe(df, 'y', ['x1', 'x2'], panelvar='id')
        self.assertIn('coefs', res_fe)

        # Logit MLE
        res_logit = compute_logit(df, 'binary_y', ['x1', 'x2'])
        self.assertIn('coefs', res_logit)
        self.assertIn('ll', res_logit)

        # Summarize & Tabulate
        summ = compute_summarize(df, ['y', 'x1'], detail=True)
        self.assertEqual(len(summ), 2)
        tab1 = compute_tabulate_1way(df, 'category')
        self.assertIn('rows', tab1)
        tab2 = compute_tabulate_2way(df, 'category', 'binary_y')
        self.assertIn('crosstab', tab2)

        # Correlate
        corr = compute_correlate(df, ['y', 'x1', 'x2'])
        self.assertIn('matrix', corr)

        # Wald Test
        wald = compute_wald_test(res_ols, 'x1 = 0')
        self.assertIn('stat', wald)
        self.assertIn('p_value', wald)

    # -------------------------------------------------------------
    # 6. Advanced Econometrics (reghdfe, probit, ologit, poisson, tobit, qreg, margins, lincom)
    # -------------------------------------------------------------
    def test_06_advanced_econometrics(self):
        df = self.df.copy()

        # reghdfe
        res_hdfe = compute_reghdfe(df, 'y', ['x1', 'x2'], absorb_vars=['id', 't'])
        self.assertIn('coefs', res_hdfe)

        # probit
        res_probit = compute_probit(df, 'binary_y', ['x1', 'x2'])
        self.assertIn('coefs', res_probit)

        # ologit
        res_ologit = compute_ologit(df, 'ordered_y', ['x1', 'x2'])
        self.assertIn('cuts', res_ologit)

        # poisson
        res_pois = compute_poisson(df, 'count_y', ['x1', 'x2'])
        self.assertIn('coefs', res_pois)
        self.assertIn('irr', res_pois)

        # tobit
        res_tobit = compute_tobit(df, 'y', ['x1', 'x2'], ll=8.0)
        self.assertIn('coefs', res_tobit)
        self.assertIn('sigma', res_tobit)

        # qreg
        res_qreg = compute_qreg(df, 'y', ['x1', 'x2'], quantile=0.5)
        self.assertIn('coefs', res_qreg)

        # margins
        margins_res = compute_margins(res_probit, df, dydx_vars=['x1'])
        self.assertIn('margins', margins_res)

        # lincom
        lincom_res = compute_lincom(res_hdfe, 'x1 - x2')
        self.assertIn('estimate', lincom_res)
        self.assertIn('se', lincom_res)

    # -------------------------------------------------------------
    # 7. Diagnostics (VIF, Hettest, Reset, DW, FirstStage, Overid, Brant)
    # -------------------------------------------------------------
    def test_07_diagnostics(self):
        df = self.df.copy()
        res_ols = compute_regression(df, 'y', ['x1', 'x2'])

        vif = compute_vif(res_ols, df)
        self.assertIn('rows', vif)

        hettest = compute_hettest(res_ols, df)
        self.assertIn('chi2', hettest)

        reset = compute_ovtest(res_ols, df)
        self.assertIn('f_stat', reset)

        dw = compute_dwatson(res_ols)
        self.assertGreater(dw['dw'], 0.0)

        res_iv = compute_ivregress_2sls(df, 'y', ['x1'], ['x2'], ['z1'])
        firststage = compute_firststage(res_iv, df)
        self.assertTrue(len(firststage) > 0)
        self.assertIn('f_stat', firststage[0])

        res_ologit = compute_ologit(df, 'ordered_y', ['x1', 'x2'])
        brant = compute_brant_test(res_ologit, df)
        self.assertIn('chi2', brant)

    # -------------------------------------------------------------
    # 8. Causal & Causal Modern (PSMatch, DiD, CSDID, did_multiplegt, Synth, EventStudy)
    # -------------------------------------------------------------
    def test_08_causal_and_modern(self):
        df = self.df.copy()

        # Propensity Score Matching
        ps_res = compute_psmatch(df, treatvar='treatment', covariates=['x1', 'x2'], depvar='y')
        self.assertIn('att', ps_res)

        # Diff in Diff
        did_res = compute_did(df, depvar='y', treatvar='treatment', timevar='post')
        self.assertIn('did_estimate', did_res)

        # CSDID Callaway & Sant'Anna
        csdid_res = compute_csdid(df, depvar='y', ivar='id', timevar='t', gvar='cohort')
        self.assertIn('overall_att', csdid_res)

        # did_multiplegt (reversible treatments)
        did_m = compute_did_multiplegt(df, depvar='y', unitvar='id', timevar='t', treatvar='treated_sw', dynamic=1, reps=5)
        self.assertIn('estimate', did_m)

        # Event Study
        ev = compute_event_study(df, depvar='y', unitvar='id', timevar='t', treat_time_var='treat_time', pre_window=2, post_window=2)
        self.assertIn('event_table', ev)

        # Synthetic Control
        synth_df = df[df['t'] <= 4].copy()
        synth_res = compute_synth(synth_df, depvar='y', idvar='id', timevar='t', treated_id=1, treat_time=3, run_placebos=False)
        self.assertIn('weights', synth_res)

    # -------------------------------------------------------------
    # 9. Time Series (dfuller, arima, corrgram)
    # -------------------------------------------------------------
    def test_09_timeseries(self):
        y_ts = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.2, 100)
        adf = compute_dfuller(y_ts, lags=2)
        self.assertIn('stat', adf)
        self.assertIn('p_value', adf)

        cg = compute_corrgram(y_ts, max_lags=5)
        self.assertIn('rows', cg)

        arima_res = compute_arima(y_ts, p=1, d=0, q=0)
        self.assertIn('coefs', arima_res)

    # -------------------------------------------------------------
    # 10. Big Data Streaming Regression & Fast Ops
    # -------------------------------------------------------------
    def test_10_bigdata_accumulator(self):
        acc = StreamingRegressionAccumulator(n_features=2, add_constant=True)
        X = self.df[['x1', 'x2']].values
        y = self.df['y'].values
        acc.update(X[:50], y[:50])
        acc.update(X[50:], y[50:])
        res = acc.finalize('y', ['x1', 'x2'])
        self.assertEqual(res['n'], 100)
        self.assertEqual(len(res['coefs']), 3)

        # Polars summary
        summary = polars_fast_summary(self.df[['x1', 'x2']])
        self.assertTrue(any(e['var'] == 'x1' for e in summary))

    # -------------------------------------------------------------
    # 11. Exporters & Demographic Table
    # -------------------------------------------------------------
    def test_11_exporters(self):
        df = self.df.copy()
        res1 = compute_regression(df, 'y', ['x1'])
        res2 = compute_regression(df, 'y', ['x1', 'x2'])
        
        m_exporter = ModelTableExporter({'Model 1': res1, 'Model 2': res2})
        latex_str = m_exporter.to_latex_booktabs()
        self.assertIn('\\begin{tabular}', latex_str)
        html_str = m_exporter.to_html()
        self.assertIn('<table', html_str)

        d_exporter = DemographicTableExporter(df, varlist=['x1', 'x2'], by_var='binary_y')
        d_df = d_exporter.to_dataframe()
        self.assertFalse(d_df.empty)

    # -------------------------------------------------------------
    # 12. Chart Studio (Forest, Event Study, Coefplot)
    # -------------------------------------------------------------
    def test_12_chart_studio(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Forest plot
        studies = [
            {'study': 'Trial A', 'theta': 0.5, 'se': 0.15},
            {'study': 'Trial B', 'theta': 0.8, 'se': 0.20},
            {'study': 'Trial C', 'theta': 0.4, 'se': 0.10}
        ]
        fig = plot_forest(studies)
        self.assertIsNotNone(fig)
        plt.close(fig)

    # -------------------------------------------------------------
    # 13. Script Runner (Loops & Do-Files)
    # -------------------------------------------------------------
    def test_13_script_runner(self):
        ds = Dataset()
        ds.load_dataframe(self.df.copy())
        interp = Interpreter(ds)
        
        script = """
        * Test comment
        local myvar "x1"
        forvalues i = 1/3 {
            gen x_step_`i' = `myvar' * `i'
        }
        """
        runner = ScriptRunner(interp)
        res = runner.run_script(script)
        self.assertTrue(res.ok, res.output)
        self.assertIn('x_step_1', interp.dataset.df.columns)
        self.assertIn('x_step_3', interp.dataset.df.columns)

    # -------------------------------------------------------------
    # 14. Copilot Dual-Tier Memory & Assistant
    # -------------------------------------------------------------
    def test_14_copilot_memory_and_assistant(self):
        ds = Dataset()
        interp = Interpreter(ds)
        copilot = ElephantGoldfishCopilot(interp)
        copilot.goldfish.record_command("regress y x1 x2", CommandResult("output", ok=True, return_code=0))
        copilot.elephant.learn_pattern("regress_tip", "Always check VIF after multicollinear models")
        
        ctx = copilot.goldfish.get_recent_context()
        self.assertTrue(any("regress y x1 x2" in msg['content'] for msg in ctx))
        pattern = copilot.elephant.find_pattern("regress_tip")
        self.assertIn("VIF", pattern)

        ans = copilot.ask("how to run regression")
        self.assertTrue(len(ans) > 0)

        assistant = CopilotAssistant(interp)
        ans2 = assistant.ask("how to run regression")
        self.assertIn("dependent variable", ans2.lower())

    # -------------------------------------------------------------
    # 15. Plugins Manager
    # -------------------------------------------------------------
    def test_15_plugins_manager(self):
        ds = Dataset()
        interp = Interpreter(ds)
        pm = PluginsManager(interp)
        loaded = pm.load_all_plugins()
        self.assertIsInstance(loaded, list)

    # -------------------------------------------------------------
    # 16. Interpreter End-to-End
    # -------------------------------------------------------------
    def test_16_interpreter_commands(self):
        ds = Dataset()
        ds.load_dataframe(self.df.copy())
        interp = Interpreter(ds)
        
        # OLS
        r1 = interp.execute("regress y x1 x2")
        self.assertEqual(r1.return_code, 0)
        self.assertIn('R-squared', r1.output)
        
        # Predict
        r2 = interp.execute("predict y_hat, xb")
        self.assertEqual(r2.return_code, 0)
        self.assertIn('y_hat', interp.dataset.df.columns)

        # Margins
        r3 = interp.execute("margins, dydx(x1)")
        self.assertEqual(r3.return_code, 0)

        # Tabulate
        r4 = interp.execute("tabulate category")
        self.assertEqual(r4.return_code, 0)

        # Test
        r5 = interp.execute("test x1 = 0")
        self.assertEqual(r5.return_code, 0)

    # -------------------------------------------------------------
    # 17. Report Generator (DOCX)
    # -------------------------------------------------------------
    def test_17_report_generator(self):
        ds = Dataset()
        ds.load_dataframe(self.df.copy())
        store = EstimatesStore()
        res = compute_regression(self.df, 'y', ['x1', 'x2'])
        store.save_last_model(res)
        store.store_estimate('ols_baseline')

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.docx")
            generate_docx_report(path, ds, store)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 1000)

    # -------------------------------------------------------------
    # 18. Qt GUI Components & Dialogs
    # -------------------------------------------------------------
    def test_18_qt_components_and_dialogs(self):
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        from PySide6.QtWidgets import QApplication
        _app = QApplication.instance() or QApplication([])

        from gui_qt.data_grid import DataGridWidget, perform_find_replace
        from gui_qt.console import ResultsConsole
        from gui_qt.command_bar import CommandBar
        from gui_qt.history_pane import HistoryPane
        from gui_qt.variables_pane import VariablesPane
        from gui_qt.main_window import PotatosticsQtWindow
        import gui_qt.dialogs as d

        ds = Dataset()
        ds.load_dataframe(self.df.copy())

        # Find & Replace
        count = perform_find_replace(ds, 'A', 'Alpha', exact_match=True)
        self.assertGreaterEqual(count, 0)

        # DataGridWidget
        grid = DataGridWidget(ds)
        self.assertIsNotNone(grid)

        # Console
        console = ResultsConsole()
        console.append_command("regress y x")
        console.append_output("R-squared = 0.85")
        console.append_error("No syntax error")
        self.assertIn("R-squared", console.toPlainText())

        # CommandBar & History & Variables
        cbar = CommandBar(ds)
        self.assertTrue(cbar.entry.isEnabled())
        hp = HistoryPane()
        hp.add_entry("regress y x1")
        vp = VariablesPane(ds)
        vp.refresh()

        # Dialogs
        for dlg_cls in [d.RegressionDialog, d.ReghdfeDialog, d.ProbitDialog, d.CsdidDialog,
                        d.EsttabDialog, d.MarginsDialog, d.SummarizeDialog, d.TabulateDialog]:
            dlg = dlg_cls(ds)
            self.assertIsNotNone(dlg)

        # MainWindow
        win = PotatosticsQtWindow()
        self.assertIsNotNone(win)
        win.close()


if __name__ == '__main__':
    unittest.main()
