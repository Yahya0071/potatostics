"""
tests/test_audit_section4_5.py - TDD Verification Suite for Section 4 (PySide6 Licensing) & Section 5 (Threading)
"""

import os
import re
import unittest


class TestSection4And5PySide6AndThreading(unittest.TestCase):
    def test_4a_no_pyqt6_imports_remain(self):
        """No files in gui_qt/ or main.py may import PyQt6. All must use PySide6."""
        gui_files = [
            'main.py',
            os.path.join('gui_qt', 'command_bar.py'),
            os.path.join('gui_qt', 'command_palette.py'),
            os.path.join('gui_qt', 'console.py'),
            os.path.join('gui_qt', 'data_grid.py'),
            os.path.join('gui_qt', 'dialogs.py'),
            os.path.join('gui_qt', 'history_pane.py'),
            os.path.join('gui_qt', 'main_window.py'),
            os.path.join('gui_qt', 'theme.py'),
            os.path.join('gui_qt', 'variables_pane.py')
        ]

        violations = []
        for fpath in gui_files:
            if not os.path.exists(fpath):
                continue
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
                for idx, line in enumerate(fp, 1):
                    if re.search(r'\b(PyQt6|pyqtSignal|pyqtSlot)\b', line):
                        violations.append(f"{fpath}:{idx}: {line.strip()}")

        self.assertEqual(violations, [], msg="Found PyQt6 imports or symbols:\n" + "\n".join(violations))

    def test_4b_pyside6_imports_functional(self):
        """Verify PySide6 core imports work cleanly in gui_qt components."""
        from PySide6 import QtCore
        self.assertTrue(hasattr(QtCore, 'Signal'))
        self.assertTrue(hasattr(QtCore, 'Slot'))

    def test_5a_worker_thread_and_cancel_button_exist(self):
        """Window must run interpreter in a worker thread, disable command bar while running, and have Cancel button."""
        from PySide6 import QtWidgets
        from gui_qt.main_window import PotatosticsQtWindow
        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["--platform", "offscreen"])
        win = PotatosticsQtWindow()
        self.assertTrue(hasattr(win, 'btn_cancel') or hasattr(win.command_bar, 'btn_cancel'),
                        msg="GUI must have a Cancel button for long-running commands")
        self.assertTrue(hasattr(win, 'worker') or hasattr(win, '_execute_in_thread'),
                        msg="GUI must have threaded worker execution support")

    def test_regression_dialog_numeric_depvar_protection(self):
        """RegressionDialog must filter non-numeric columns from dependent variable dropdown and provide live equation preview."""
        import pandas as pd
        from PySide6 import QtWidgets
        from engine.dataset import Dataset
        from gui_qt.dialogs import RegressionDialog, ReghdfeDialog, ProbitDialog

        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["--platform", "offscreen"])
        ds = Dataset()
        ds.df = pd.DataFrame({
            'period': ['1996-07', '1996-08', '1996-09'],
            'cards_issued': [100, 150, 200],
            'year': [1996, 1996, 1996],
            'approved': [1, 0, 1]
        })
        dlg = RegressionDialog(ds)
        # period must NOT be in cmb_dep
        dep_items = [dlg.cmb_dep.itemText(i) for i in range(dlg.cmb_dep.count())]
        self.assertNotIn('period', dep_items)
        self.assertIn('cards_issued', dep_items)
        self.assertIn('year', dep_items)

        # Selecting period as independent variable must format as factor variable i.period
        dlg.cmb_dep.setCurrentText('cards_issued')
        for i in range(dlg.lst_indep.count()):
            item = dlg.lst_indep.item(i)
            if item.text() == 'period':
                item.setSelected(True)
        dlg.update_syntax()
        self.assertIn('i.period', dlg.txt_syntax.text())
        self.assertTrue(hasattr(dlg, 'lbl_equation'))
        self.assertIn('cards_issued', dlg.lbl_equation.text())
        self.assertIn('i.period', dlg.lbl_equation.text())

        # Test ReghdfeDialog live equation
        reghdfe_dlg = ReghdfeDialog(ds)
        reghdfe_dlg.cmb_dep.setCurrentText('cards_issued')
        for i in range(reghdfe_dlg.lst_indeps.count()):
            item = reghdfe_dlg.lst_indeps.item(i)
            if item.text() == 'year':
                item.setSelected(True)
        for i in range(reghdfe_dlg.lst_absorb.count()):
            item = reghdfe_dlg.lst_absorb.item(i)
            if item.text() == 'period':
                item.setSelected(True)
        reghdfe_dlg.update_syntax()
        self.assertIn('year', reghdfe_dlg.lbl_equation.text())
        self.assertIn('α_period', reghdfe_dlg.lbl_equation.text())

        # Test ProbitDialog live equation
        probit_dlg = ProbitDialog(ds)
        probit_dlg.cmb_dep.setCurrentText('approved')
        for i in range(probit_dlg.lst_indeps.count()):
            item = probit_dlg.lst_indeps.item(i)
            if item.text() == 'year':
                item.setSelected(True)
        probit_dlg.update_syntax()
        self.assertIn('Pr(approved = 1 | X)', probit_dlg.lbl_equation.text())
        self.assertIn('year', probit_dlg.lbl_equation.text())

        # Test Observation Filter (if / in) on RegressionDialog and SummarizeDialog
        dlg.txt_if.setText('year >= 1996')
        dlg.txt_in.setText('1/2')
        dlg.update_syntax()
        self.assertIn('if year >= 1996 in 1/2', dlg.txt_syntax.text())
        self.assertIn('[where year >= 1996]', dlg.lbl_equation.text())

        from gui_qt.dialogs import SummarizeDialog, TabulateDialog, CsdidDialog, GraphDialog
        sum_dlg = SummarizeDialog(ds)
        sum_dlg.txt_if.setText('cards_issued > 100')
        sum_dlg.update_syntax()
        self.assertIn('if cards_issued > 100', sum_dlg.txt_syntax.text())

        # Test TabulateDialog
        tab_dlg = TabulateDialog(ds)
        tab_dlg.txt_if.setText('year == 1996')
        tab_dlg.update_syntax()
        self.assertIn('if year == 1996', tab_dlg.txt_syntax.text())

        # Test CsdidDialog
        csdid_dlg = CsdidDialog(ds)
        csdid_dlg.cmb_dep.setCurrentText('cards_issued')
        csdid_dlg.cmb_ivar.setCurrentText('year')
        csdid_dlg.cmb_timevar.setCurrentText('year')
        csdid_dlg.cmb_gvar.setCurrentText('year')
        csdid_dlg.txt_if.setText('cards_issued > 50')
        csdid_dlg.update_syntax()
        self.assertIn('if cards_issued > 50', csdid_dlg.txt_syntax.text())

        # Test GraphDialog
        graph_dlg = GraphDialog(ds)
        graph_dlg.cmb_y.setCurrentText('cards_issued')
        graph_dlg.cmb_x.setCurrentText('year')
        graph_dlg.txt_if.setText('year >= 1996')
        graph_dlg.txt_in.setText('1/10')
        graph_dlg.update_syntax()
        self.assertIn('if year >= 1996 in 1/10', graph_dlg.txt_syntax.text())

    def test_new_dialog_options_pages(self):
        """Verify that every test and model has dedicated interactive options dialog with observation filter and equation preview."""
        import pandas as pd
        from PySide6 import QtWidgets
        from engine.dataset import Dataset
        from gui_qt.dialogs import (
            ProbitDialog, CountRegressionDialog, IvregressDialog,
            PanelRegressionDialog, CorrelateDialog, GraphDialog
        )

        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["--platform", "offscreen"])
        ds = Dataset()
        ds.df = pd.DataFrame({
            'wage': [10.5, 15.0, 22.0, 18.5, 30.0],
            'educ': [12, 14, 16, 12, 18],
            'experience': [5, 10, 15, 8, 20],
            'female': [1, 0, 1, 0, 1],
            'region': [1, 2, 1, 2, 1],
            'id': [101, 102, 103, 104, 105],
            'year': [2020, 2020, 2021, 2021, 2022],
            'nearc4': [1, 0, 1, 0, 1],
            'treated': [0, 1, 0, 1, 1]
        })

        # 1. Binary Choice Dialog (Logit / Probit)
        dlg_logit = ProbitDialog(ds, initial_model="logit")
        dlg_logit.cmb_dep.setCurrentText("female")
        for i in range(dlg_logit.lst_indeps.count()):
            item = dlg_logit.lst_indeps.item(i)
            if item.text() in ("wage", "educ"):
                item.setSelected(True)
        dlg_logit.txt_if.setText("year >= 2020")
        dlg_logit.update_syntax()
        self.assertIn("logit female", dlg_logit.txt_syntax.text())
        self.assertIn("wage educ", dlg_logit.txt_syntax.text())
        self.assertIn("if year >= 2020", dlg_logit.txt_syntax.text())
        self.assertIn("Pr(female = 1 | X)", dlg_logit.lbl_equation.text())

        # Switch to Probit
        dlg_logit.cmb_model.setCurrentText("Probit Model (probit)")
        dlg_logit.update_syntax()
        self.assertIn("probit female", dlg_logit.txt_syntax.text())

        # 2. Count Regression Dialog (Poisson / Negative Binomial)
        dlg_count = CountRegressionDialog(ds, initial_model="poisson")
        dlg_count.cmb_dep.setCurrentText("educ")
        for i in range(dlg_count.lst_indeps.count()):
            item = dlg_count.lst_indeps.item(i)
            if item.text() == "experience":
                item.setSelected(True)
        dlg_count.cmb_se.setCurrentText("Robust (HC1)")
        dlg_count.update_syntax()
        self.assertIn("poisson educ experience", dlg_count.txt_syntax.text())
        self.assertIn("robust", dlg_count.txt_syntax.text())
        self.assertIn("E(educ | X) = exp", dlg_count.lbl_equation.text())

        # Switch to nbreg
        dlg_count.cmb_model.setCurrentText("Negative Binomial (nbreg)")
        dlg_count.update_syntax()
        self.assertIn("nbreg educ experience", dlg_count.txt_syntax.text())
        for model_index in (0, 1):
            dlg_count.cmb_model.setCurrentIndex(model_index)
            dlg_count.cmb_cluster.setCurrentIndex(0)
            dlg_count.cmb_se.setCurrentText("Clustered")
            self.assertFalse(dlg_count.btn_submit.isEnabled())
            self.assertFalse(dlg_count.btn_paste.isEnabled())
            self.assertEqual(dlg_count.txt_syntax.text(), "")
            dlg_count.cmb_cluster.setCurrentText("year")
            self.assertTrue(dlg_count.btn_submit.isEnabled())
            self.assertIn("vce(cluster year)", dlg_count.txt_syntax.text())

        # 3. IV Regress Dialog (2SLS)
        dlg_iv = IvregressDialog(ds)
        dlg_iv.cmb_dep.setCurrentText("wage")
        for i in range(dlg_iv.lst_exog.count()):
            if dlg_iv.lst_exog.item(i).text() == "experience":
                dlg_iv.lst_exog.item(i).setSelected(True)
        for i in range(dlg_iv.lst_endog.count()):
            if dlg_iv.lst_endog.item(i).text() == "educ":
                dlg_iv.lst_endog.item(i).setSelected(True)
        for i in range(dlg_iv.lst_inst.count()):
            if dlg_iv.lst_inst.item(i).text() == "nearc4":
                dlg_iv.lst_inst.item(i).setSelected(True)
        dlg_iv.cmb_se.setCurrentText("Robust (HC1)")
        dlg_iv.update_syntax()
        self.assertIn("ivregress 2sls wage experience (educ = nearc4)", dlg_iv.txt_syntax.text())
        self.assertIn("robust", dlg_iv.txt_syntax.text())
        self.assertIn("wage = β₀ + β·experience + γ·educ + ε", dlg_iv.lbl_equation.text())

        # 4. Panel Regression Dialog (xtreg)
        dlg_panel = PanelRegressionDialog(ds, initial_model="fe")
        dlg_panel.cmb_dep.setCurrentText("wage")
        for i in range(dlg_panel.lst_indeps.count()):
            if dlg_panel.lst_indeps.item(i).text() == "experience":
                dlg_panel.lst_indeps.item(i).setSelected(True)
        dlg_panel.update_syntax()
        self.assertIn("xtreg wage experience, fe", dlg_panel.txt_syntax.text())
        self.assertIn("wageᵢₜ = αᵢ", dlg_panel.lbl_equation.text())

        # Switch to Random Effects (re)
        dlg_panel.cmb_model.setCurrentText("Random Effects (re)")
        dlg_panel.update_syntax()
        self.assertIn("xtreg wage experience, re", dlg_panel.txt_syntax.text())
        self.assertIn("wageᵢₜ = μ + αᵢ", dlg_panel.lbl_equation.text())

        # 5. Correlate Dialog
        dlg_corr = CorrelateDialog(ds)
        dlg_corr.chk_cov.setChecked(True)
        dlg_corr.txt_if.setText("year == 2020")
        dlg_corr.update_syntax()
        self.assertIn("correlate", dlg_corr.txt_syntax.text())
        self.assertIn("covariance", dlg_corr.txt_syntax.text())
        self.assertIn("if year == 2020", dlg_corr.txt_syntax.text())

        # 6. Graph Studio Event Study
        dlg_graph = GraphDialog(ds, initial_type="eventstudy")
        dlg_graph.update_syntax()
        self.assertIn("eventstudy", dlg_graph.txt_syntax.text())

        # 7. Time-Series Dialog (tsset, dfuller, arima, hp, var, corrgram)
        from gui_qt.dialogs import TimeSeriesDialog, PanelSetupDialog
        # tsset
        dlg_ts = TimeSeriesDialog(ds, initial_task="tsset")
        dlg_ts.cmb_var.setCurrentText("year")
        dlg_ts.update_syntax()
        self.assertEqual(dlg_ts.txt_syntax.text().strip(), "tsset year")

        # dfuller
        dlg_ts.cmb_task.setCurrentText("Augmented Dickey-Fuller Unit Root Test (dfuller)")
        dlg_ts.cmb_var.setCurrentText("wage")
        dlg_ts.spin_lags.setValue(2)
        dlg_ts.update_syntax()
        self.assertIn("dfuller wage", dlg_ts.txt_syntax.text())
        self.assertIn("lags(2)", dlg_ts.txt_syntax.text())
        self.assertIn("H₀: β = 0 (Unit Root)", dlg_ts.lbl_equation.text())

        # arima
        dlg_ts.cmb_task.setCurrentText("ARIMA Modeling & Forecasting (arima)")
        dlg_ts.cmb_var.setCurrentText("wage")
        dlg_ts.spin_p.setValue(2)
        dlg_ts.spin_d.setValue(1)
        dlg_ts.spin_q.setValue(1)
        dlg_ts.spin_steps.setValue(10)
        dlg_ts.update_syntax()
        self.assertIn("arima wage", dlg_ts.txt_syntax.text())
        self.assertIn("ar(2) d(1) ma(1) steps(10)", dlg_ts.txt_syntax.text())

        # hp filter
        dlg_ts.cmb_task.setCurrentText("Hodrick-Prescott Filter (tsfilter hp)")
        dlg_ts.cmb_var.setCurrentText("wage")
        dlg_ts.txt_cycle.setText("w_cycle")
        dlg_ts.txt_trend.setText("w_trend")
        dlg_ts.spin_lambda.setValue(1600.0)
        dlg_ts.update_syntax()
        self.assertIn("tsfilter hp w_cycle = wage, smooth(1600.0) trend(w_trend)", dlg_ts.txt_syntax.text())

        # var
        dlg_ts.cmb_task.setCurrentText("Vector Autoregression (var)")
        for i in range(dlg_ts.lst_var_depvars.count()):
            if dlg_ts.lst_var_depvars.item(i).text() in ("wage", "educ"):
                dlg_ts.lst_var_depvars.item(i).setSelected(True)
        dlg_ts.spin_lags.setValue(2)
        dlg_ts.update_syntax()
        self.assertIn("var wage educ, lags(1/2)", dlg_ts.txt_syntax.text())

        # corrgram
        dlg_ts.cmb_task.setCurrentText("Autocorrelations & Correlogram (corrgram)")
        dlg_ts.cmb_var.setCurrentText("wage")
        dlg_ts.update_syntax()
        self.assertIn("corrgram wage", dlg_ts.txt_syntax.text())

        # 8. Panel Setup Dialog (xtset)
        dlg_xtset = PanelSetupDialog(ds)
        dlg_xtset.cmb_panelvar.setCurrentText("id")
        dlg_xtset.cmb_timevar.setCurrentText("year")
        dlg_xtset.update_syntax()
        self.assertEqual(dlg_xtset.txt_syntax.text().strip(), "xtset id year")

        # 9. Censored and Quantile Dialog (tobit / qreg)
        from gui_qt.dialogs import (
            CensoredAndQuantileDialog, OrdinalMultinomialDialog,
            SurvivalAnalysisDialog, CausalInferenceDialog,
            DtableDialog, AdvancedLinearDialog, SurveyDialog, PostEstimationDialog
        )
        dlg_tobit = CensoredAndQuantileDialog(ds, initial_model="tobit")
        dlg_tobit.cmb_dep.setCurrentText("wage")
        dlg_tobit.spin_ll.setValue(0.0)
        dlg_tobit.chk_ll.setChecked(True)
        for i in range(dlg_tobit.lst_indeps.count()):
            if dlg_tobit.lst_indeps.item(i).text() == "educ":
                dlg_tobit.lst_indeps.item(i).setSelected(True)
        dlg_tobit.update_syntax()
        self.assertIn("tobit wage educ, ll(0.0)", dlg_tobit.txt_syntax.text())

        dlg_qreg = CensoredAndQuantileDialog(ds, initial_model="qreg")
        dlg_qreg.cmb_dep.setCurrentText("wage")
        dlg_qreg.spin_quantile.setValue(0.75)
        for i in range(dlg_qreg.lst_indeps.count()):
            if dlg_qreg.lst_indeps.item(i).text() == "educ":
                dlg_qreg.lst_indeps.item(i).setSelected(True)
        dlg_qreg.update_syntax()
        self.assertIn("qreg wage educ, q(0.75)", dlg_qreg.txt_syntax.text())

        # 10. Ordinal & Multinomial Dialog (ologit / mlogit)
        dlg_ologit = OrdinalMultinomialDialog(ds, initial_model="ologit")
        dlg_ologit.cmb_dep.setCurrentText("educ")
        for i in range(dlg_ologit.lst_indeps.count()):
            if dlg_ologit.lst_indeps.item(i).text() == "wage":
                dlg_ologit.lst_indeps.item(i).setSelected(True)
        dlg_ologit.update_syntax()
        self.assertIn("ologit educ wage", dlg_ologit.txt_syntax.text())

        # 11. Survival Analysis Dialog (stset, stcox, sts)
        dlg_surv = SurvivalAnalysisDialog(ds, initial_task="stset")
        dlg_surv.cmb_time.setCurrentText("year")
        dlg_surv.cmb_fail.setCurrentText("treated")
        dlg_surv.update_syntax()
        self.assertEqual(dlg_surv.txt_syntax.text().strip(), "stset year, fail(treated)")

        dlg_stcox = SurvivalAnalysisDialog(ds, initial_task="stcox")
        dlg_stcox.cmb_task.setCurrentText("Cox Proportional Hazards Model (stcox)")
        for i in range(dlg_stcox.lst_covars.count()):
            if dlg_stcox.lst_covars.item(i).text() == "educ":
                dlg_stcox.lst_covars.item(i).setSelected(True)
        dlg_stcox.update_syntax()
        self.assertEqual(dlg_stcox.txt_syntax.text().strip(), "stcox educ")

        # 12. Causal Inference Dialog (bacondecomp, synth, psmatch, eventstudy)
        dlg_bacon = CausalInferenceDialog(ds, initial_tool="bacondecomp")
        dlg_bacon.cmb_y.setCurrentText("wage")
        dlg_bacon.cmb_post.setCurrentText("year")
        dlg_bacon.cmb_treat.setCurrentText("treated")
        dlg_bacon.update_syntax()
        self.assertIn("bacondecomp wage year treated", dlg_bacon.txt_syntax.text())

        dlg_evs = CausalInferenceDialog(ds, initial_tool="eventstudy")
        dlg_evs.cmb_tool.setCurrentText("Dynamic Event Study (eventstudy)")
        dlg_evs.cmb_y.setCurrentText("wage")
        dlg_evs.cmb_post.setCurrentText("year")
        dlg_evs.cmb_treat.setCurrentText("treated")
        dlg_evs.spn_leads.setValue(2)
        dlg_evs.spn_lags.setValue(4)
        dlg_evs.update_syntax()
        self.assertIn("eventstudy wage treated year, leads(2), lags(4)", dlg_evs.txt_syntax.text())

        # 13. Dtable Dialog
        dlg_dt = DtableDialog(ds)
        for i in range(dlg_dt.lst_vars.count()):
            if dlg_dt.lst_vars.item(i).text() in ("wage", "educ"):
                dlg_dt.lst_vars.item(i).setSelected(True)
        dlg_dt.cmb_by.setCurrentText("treated")
        dlg_dt.edt_export.setText("table1.docx")
        dlg_dt.update_syntax()
        self.assertIn("dtable wage educ, by(treated), export(table1.docx)", dlg_dt.txt_syntax.text())

        # 14. Advanced Linear Dialog (cnsreg, sureg, bigregress)
        dlg_cns = AdvancedLinearDialog(ds, initial_model="cnsreg")
        dlg_cns.cmb_dep1.setCurrentText("wage")
        for i in range(dlg_cns.lst_indep1.count()):
            if dlg_cns.lst_indep1.item(i).text() == "educ":
                dlg_cns.lst_indep1.item(i).setSelected(True)
        dlg_cns.edt_constraints.setText("1")
        dlg_cns.update_syntax()
        self.assertIn("cnsreg wage educ, constraints(1)", dlg_cns.txt_syntax.text())

        # 15. Survey Dialog (svyset, svy: regress)
        dlg_svy = SurveyDialog(ds, initial_mode="svyset")
        dlg_svy.cmb_psu.setCurrentText("id")
        dlg_svy.cmb_pweight.setCurrentText("wage")
        dlg_svy.cmb_strata.setCurrentText("treated")
        dlg_svy.update_syntax()
        self.assertIn("svyset id [pweight=wage], strata(treated)", dlg_svy.txt_syntax.text())

        # 16. Post-Estimation Dialog (lincom, vif, etc.)
        dlg_post = PostEstimationDialog(ds, initial_tool="lincom")
        dlg_post.edt_expr.setText("educ - 2*year")
        dlg_post.update_syntax()
        self.assertEqual(dlg_post.txt_syntax.text().strip(), "lincom educ - 2*year")

        dlg_vif = PostEstimationDialog(ds, initial_tool="vif")
        dlg_vif.update_syntax()
        self.assertEqual(dlg_vif.txt_syntax.text().strip(), "estat vif")


if __name__ == '__main__':
    unittest.main()
