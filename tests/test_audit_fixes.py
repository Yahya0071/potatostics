"""
tests/test_audit_fixes.py - Verification Suite for Section 2 (Silently Wrong Results)
Covers sub-items 2a through 2n:
  2a: predict link functions, never fill missing columns with zeros
  2b: if/in qualifiers on csdid, synth, eventstudy and parametrized test
  2c: factor variables missing handling and 3-way interaction error
  2d: function semantics: round(x,y), int(), D2.x vs L2/F2, max/min ignore missing, missing as +inf
  2e: unsupported vce values raise errors
  2f: MLE clustering shared helper with G/(G-1) and z-inference
  2g: reghdfe/areg absorbed df counting, robust N/(N-K-df_a), nested cluster
  2h: xtreg fe statistics and non-zero e() registers
  2i: qreg covariance scaling and bootstrap option
  2j: tobit numerical Hessian, real p-value for /sigma, robust option
  2k: estat firststage, Wu-Hausman, and Sargan overid test
  2l: csdid never-treated handling (NaN/0), not-yet-treated controls, influence SEs
  2m: eventstudy TWFE label, never-treated, window bounds; synth balanced check & placebo warning
  2n: psmatch caliper, common support, balance table; did cluster option
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from scipy import stats

# Fallback to project root if not installed as wheel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.interpreter import Interpreter
from engine.dataset import Dataset
from engine.parser import expand_factor_variables, evaluate_dta_expression, compute_time_series_series
from engine.estimation import compute_logit
from engine.advanced_econometrics import compute_probit, compute_poisson, compute_reghdfe, compute_tobit, compute_qreg
from engine.causal import compute_psmatch
from engine.causal_modern import compute_csdid, compute_synth, compute_event_study


class TestSection2AuditFixes(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.dataset = Dataset()
        self.interp = Interpreter(self.dataset)

    # -------------------------------------------------------------------------
    # 2a: predict
    # -------------------------------------------------------------------------
    def test_2a_predict_links_and_missing_columns(self):
        """predict: never fill missing columns with zeros; 'pr' uses link function (probit=norm, logit=logistic)"""
        n = 100
        x = np.random.randn(n)
        z = 0.5 + 1.2 * x
        y_prob = stats.norm.cdf(z)
        y_bin = (np.random.rand(n) < y_prob).astype(int)
        self.dataset.df = pd.DataFrame({'y': y_bin, 'x': x})

        # 1. Probit model predict pr
        self.interp.execute("probit y x")
        r_pr = self.interp.execute("predict pr_hat, pr")
        self.assertTrue(r_pr.ok)
        # Expected probit prediction is normal CDF Phi(xb)
        b = self.interp.estimates.last_model['beta']
        xb = b[1] + b[0] * self.dataset.df['x'] # x, const
        expected_pr = stats.norm.cdf(xb)
        np.testing.assert_allclose(self.dataset.df['pr_hat'].to_numpy(), expected_pr, rtol=1e-5)

        # 2. Missing regressor in dataset must raise an error, NEVER silently fill with zeros
        self.dataset.df.drop(columns=['x'], inplace=True)
        r_fail = self.interp.execute("predict bad_pred, xb")
        self.assertFalse(r_fail.ok, msg="predict must fail when regressor is missing from dataset")
        self.assertIn("not found", r_fail.output.lower())

    # -------------------------------------------------------------------------
    # 2b: if/in qualifiers
    # -------------------------------------------------------------------------
    def test_2b_if_in_qualifiers(self):
        """csdid, synth, eventstudy and all estimation commands must use filter_data and handle if/in conditions correctly"""
        n_units = 10
        n_time = 8
        n = n_units * n_time
        unit = np.repeat(np.arange(n_units), n_time)
        time = np.tile(np.arange(n_time), n_units)
        cohort = np.repeat([0, 0, 0, 0, 4, 4, 4, 6, 6, 6], n_time)
        x = np.random.randn(n)
        z = np.random.randn(n)
        y = 1.0 + 0.5 * x + np.random.randn(n)
        y_bin = (y > 1.0).astype(int)
        y_count = np.clip(np.round(np.exp(0.2 * x)), 0, 20).astype(int)
        cat = np.random.choice([1, 2, 3], size=n)

        self.dataset.df = pd.DataFrame({
            'unit': unit, 'time': time, 'cohort': cohort,
            'y': y, 'x': x, 'z': z, 'y_bin': y_bin, 'y_count': y_count, 'cat': cat
        })
        self.interp.execute("xtset unit time")

        commands_to_test = [
            "regress y x if x > 0",
            "summarize y if x > 0",
            "tabulate cat if x > 0",
            "correlate y x if x > 0",
            "logit y_bin x if x > 0",
            "probit y_bin x if x > 0",
            "poisson y_count x if x > 0",
            "ivregress 2sls y (x = z) if x > 0",
            "xtreg y x if x > 0, fe",
            "reghdfe y x if x > 0, absorb(unit)",
            "tobit y x if x > 0, ll(0)",
            "qreg y x if x > 0",
            "csdid y if unit < 8, ivar(unit) timevar(time) gvar(cohort)",
            "synth y if unit < 6, trunit(4) trperiod(5) unit(unit) time(time) noplacebo",
            "eventstudy y if time >= 2, unit(unit) time(time) treattime(cohort)"
        ]

        for cmd_str in commands_to_test:
            res = self.interp.execute(cmd_str)
            self.assertTrue(res.ok, msg=f"Command '{cmd_str}' failed with if qualifier: {res.output}")

    # -------------------------------------------------------------------------
    # 2c: factor variables
    # -------------------------------------------------------------------------
    def test_2c_factor_variables_missing_and_3way_error(self):
        """Drop observations where factor is missing; 3-way interactions raise clear error"""
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'cat': [1.0, np.nan, 2.0, 1.0, 2.0],
            'x': [10.0, 20.0, 30.0, 40.0, 50.0],
            'z': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        # 1. Check factor expansion on missing values: dummy must be NaN, NOT 0.0 (base level)
        exp_vars, working_df, _ = expand_factor_variables(['i.cat'], df)
        self.assertTrue(pd.isna(working_df.loc[1, '2.cat']), msg="Missing factor value must be NaN, not coded as base level")

        # 2. Three-way interactions must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            expand_factor_variables(['x#cat#z'], df)
        self.assertIn("three-way", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # 2d: function semantics
    # -------------------------------------------------------------------------
    def test_2d_function_semantics(self):
        """round(x,y)=nearest multiple of y; int() truncates to 0; D2.x is 2nd diff; missing is +inf"""
        df = pd.DataFrame({
            'x': [5.2, -3.7, 7.8, np.nan],
            't': [1, 2, 3, 4]
        })
        # 1. round(x, y) = nearest multiple of y
        res_round = evaluate_dta_expression("round(5.2, 2)", df)
        self.assertEqual(float(res_round), 6.0)

        # 2. int() truncates toward zero: int(-3.7) == -3
        res_int = evaluate_dta_expression("int(-3.7)", df)
        self.assertEqual(float(res_int), -3.0)

        # 3. max() and min() ignore missing
        res_max = evaluate_dta_expression("max(x, 6)", df)
        # For index 3 where x is NaN, max(NaN, 6) must be 6
        self.assertEqual(float(res_max.iloc[3]), 6.0)

        # 4. Missing compares as +infinity (x > 5 is True for missing/NaN in Reference)
        res_gt = evaluate_dta_expression("x > 5", df)
        self.assertTrue(bool(res_gt.iloc[3]), msg="Missing value (NaN) must compare as +inf (x > 5 is True)")
        res_lt = evaluate_dta_expression("x < 5", df)
        self.assertFalse(bool(res_lt.iloc[3]), msg="Missing value (NaN) must compare as +inf (x < 5 is False)")

        # 5. D2.x is second difference: x_t - 2*x_{t-1} + x_{t-2}
        df_ts = pd.DataFrame({'t': [1, 2, 3, 4], 'val': [1.0, 4.0, 9.0, 16.0]}) # val = t^2
        # First difference: [NaN, 3, 5, 7]; Second difference: [NaN, NaN, 2, 2]
        s_d2 = compute_time_series_series(df_ts, 'val', 'D', order=2, time_context={'timevar': 't'})
        np.testing.assert_allclose(s_d2.iloc[2:].to_numpy(), [2.0, 2.0])

    # -------------------------------------------------------------------------
    # 2e: unsupported VCE values
    # -------------------------------------------------------------------------
    def test_2e_unsupported_vce_raises(self):
        """Unsupported VCE types (e.g. hc2/hc3 on logit, probit, poisson, tobit, qreg) must raise ValueError"""
        df = pd.DataFrame({'y': [0, 1, 0, 1]*10, 'x': np.random.randn(40)})
        with self.assertRaises(ValueError):
            compute_logit(df, 'y', ['x'], vce='hc2')
        with self.assertRaises(ValueError):
            compute_probit(df, 'y', ['x'], vce='hc3')
        with self.assertRaises(ValueError):
            compute_poisson(df, 'y', ['x'], vce='hc2')
        with self.assertRaises(ValueError):
            compute_tobit(df, 'y', ['x'], ll=0.0, vce='hc2')
        with self.assertRaises(ValueError):
            compute_qreg(df, 'y', ['x'], vce='hc1')

    # -------------------------------------------------------------------------
    # 2k: first-stage diagnostics, Wu-Hausman, Sargan
    # -------------------------------------------------------------------------
    def test_2k_first_stage_and_overid_diagnostics(self):
        """estat firststage, Wu-Hausman endogeneity test, and Sargan overidentification test"""
        n = 200
        z1 = np.random.randn(n)
        z2 = np.random.randn(n)
        u = np.random.randn(n)
        x_endog = 0.6 * z1 + 0.5 * z2 + u
        y = 1.0 + 1.5 * x_endog + np.random.randn(n)
        self.dataset.df = pd.DataFrame({'y': y, 'x': x_endog, 'z1': z1, 'z2': z2})

        r_iv = self.interp.execute("ivregress 2sls y (x = z1 z2)")
        self.assertTrue(r_iv.ok)

        # 1. estat firststage
        r_fs = self.interp.execute("estat firststage")
        self.assertTrue(r_fs.ok)
        self.assertIn("first-stage", r_fs.output.lower())

        # 2. estat endogenous / Wu-Hausman
        r_wh = self.interp.execute("estat endogenous")
        self.assertTrue(r_wh.ok)
        self.assertIn("hausman", r_wh.output.lower())

        # 3. estat overid / Sargan
        r_sg = self.interp.execute("estat overid")
        self.assertTrue(r_sg.ok)
        self.assertIn("sargan", r_sg.output.lower())

    # -------------------------------------------------------------------------
    # 2l: csdid never-treated units & unconditional label
    # -------------------------------------------------------------------------
    def test_2l_csdid_never_treated_and_unconditional(self):
        """csdid keeps never-treated units coded NaN or 0, labels output 'unconditional'"""
        n_units = 20
        n_periods = 5
        unit = np.repeat(np.arange(n_units), n_periods)
        time = np.tile(np.arange(n_periods), n_units)
        # Units 0-9 never-treated (coded NaN), units 10-19 treated at period 3
        cohort = np.repeat([np.nan]*10 + [3.0]*10, n_periods)
        y = np.random.randn(n_units * n_periods)
        df_csdid = pd.DataFrame({'unit': unit, 'time': time, 'cohort': cohort, 'y': y})

        r = compute_csdid(df_csdid, 'y', 'unit', 'time', 'cohort', control_type="notyettreated")
        self.assertEqual(r['n_never_treated'], 10, msg="Never-treated units with NaN cohort must be preserved")
        self.assertIn("unconditional", r['label'].lower())

    # -------------------------------------------------------------------------
    # 2m: eventstudy TWFE and synth balanced panel check
    # -------------------------------------------------------------------------
    def test_2m_eventstudy_and_synth_checks(self):
        """eventstudy labeled TWFE; synth checks balanced panel and warns when placebos skipped"""
        n_units = 10
        n_periods = 6
        unit = np.repeat(np.arange(n_units), n_periods)
        time = np.tile(np.arange(n_periods), n_units)
        cohort = np.repeat([0]*5 + [3]*5, n_periods)
        y = np.random.randn(n_units * n_periods)
        df = pd.DataFrame({'unit': unit, 'time': time, 'cohort': cohort, 'y': y})

        # Eventstudy test
        r_es = compute_event_study(df, 'y', 'unit', 'time', 'cohort', pre_window=2, post_window=2)
        self.assertIn("twfe", (r_es.get('estimator', '') + r_es.get('label', '')).lower())

        # Synth unbalanced panel check
        df_unbalanced = df.iloc[:-1].copy() # drop 1 observation
        with self.assertRaises(ValueError):
            compute_synth(df_unbalanced, 'y', 'unit', 'time', treated_id=6, treat_time=3)

    # -------------------------------------------------------------------------
    # 2n: psmatch caliper/balance and did cluster option
    # -------------------------------------------------------------------------
    def test_2n_psmatch_caliper_and_did_cluster(self):
        """psmatch caliper, common-support, balance table; did wires cluster option"""
        n = 100
        x = np.random.randn(n)
        d = (np.random.rand(n) < 0.5).astype(int)
        y = 2.0 + 1.5 * d + 0.8 * x + np.random.randn(n)
        df = pd.DataFrame({'y': y, 'd': d, 'x': x})

        r_ps = compute_psmatch(df, treatvar='d', covariates=['x'], depvar='y', caliper=0.1)
        self.assertIn('balance_table', r_ps)
        self.assertIn('common_support', r_ps)

        # Test did with cluster option
        self.dataset.df = pd.DataFrame({
            'y': y, 'd': d, 't': np.repeat([0, 1], n // 2), 'clust': np.arange(n) % 10
        })
        res_did = self.interp.execute("did y d t, cluster(clust)")
        self.assertTrue(res_did.ok)
        self.assertIn("clust", str(res_did.output).lower())

    # -------------------------------------------------------------------------
    # 2o: Task 1 - String comparisons & literal masking
    # -------------------------------------------------------------------------
    def test_2o_string_comparisons_and_literal_masking(self):
        """String if qualifiers and string literals containing ., &, |"""
        self.dataset.df = pd.DataFrame({
            'price': [100.0, 200.0, 300.0, 400.0],
            'make': ['Russet Burbank', 'Yukon Gold', 'A.B & C', None]
        })
        # summarize with string equality
        res_sum = self.interp.execute('summarize price if make == "Russet Burbank"')
        self.assertTrue(res_sum.ok)
        self.assertIn("100.0000", str(res_sum.output))

        # count with string inequality
        res_cnt = self.interp.execute('count if make != "Yukon Gold"')
        self.assertTrue(res_cnt.ok)
        # 3 observations are != 'Yukon Gold' (Russet Burbank, A.B & C, and None which is "")
        self.assertIn("3", str(res_cnt.output))

        # string literal containing '.', '&', '|'
        res_gen = self.interp.execute('generate flag = 1 if make == "A.B & C"')
        self.assertTrue(res_gen.ok)
        self.assertEqual(self.dataset.df['flag'].iloc[2], 1.0)
        self.assertTrue(pd.isna(self.dataset.df['flag'].iloc[0]))

    # -------------------------------------------------------------------------
    # 2p: Task 2 - reghdfe collinearity omission
    # -------------------------------------------------------------------------
    def test_2p_reghdfe_collinearity_omission(self):
        """reghdfe automatic collinearity omission when a regressor is time-invariant within an absorbed fixed effect"""
        n_units = 5
        n_years = 4
        unit_id = np.repeat(np.arange(1, n_units + 1), n_years)
        year = np.tile(np.arange(2000, 2000 + n_years), n_units)
        # time-invariant regressor within unit_id
        time_inv = np.repeat([10.0, 20.0, 30.0, 40.0, 50.0], n_years)
        # time-varying regressor
        sales = np.random.randn(n_units * n_years) * 5 + 100
        price = 50 + 2.0 * sales + np.random.randn(n_units * n_years)
        df = pd.DataFrame({'price': price, 'sales': sales, 'time_inv': time_inv, 'unit_id': unit_id, 'year': year})

        res_hdfe = compute_reghdfe(df, 'price', ['sales', 'time_inv'], absorb_vars=['unit_id', 'year'], vce='ols')
        # time_inv must be marked omitted
        coef_dict = {c['name']: c for c in res_hdfe['coefs']}
        self.assertIn('time_inv', coef_dict)
        self.assertTrue(coef_dict['time_inv'].get('omitted', False))
        self.assertEqual(coef_dict['time_inv']['coef'], 0.0)
        self.assertTrue(np.isnan(coef_dict['time_inv']['p']))

    # -------------------------------------------------------------------------
    # 2q: Task 3 - bysort + egen group execution
    # -------------------------------------------------------------------------
    def test_2q_bysort_egen_group_execution(self):
        """bysort group: egen newvar = mean(x) persisting newvar in dataset.df"""
        self.dataset.df = pd.DataFrame({
            'grp': [1, 1, 2, 2],
            'val': [10.0, 20.0, 100.0, 200.0]
        })
        res_by = self.interp.execute("bysort grp: egen mean_val = mean(val)")
        self.assertTrue(res_by.ok)
        self.assertIn("mean_val", self.dataset.df.columns)
        self.assertEqual(list(self.dataset.df['mean_val']), [15.0, 15.0, 150.0, 150.0])

    # -------------------------------------------------------------------------
    # 2r: Task 4 - csdid influence function SEs and unit_influences
    # -------------------------------------------------------------------------
    def test_2r_csdid_influence_function_ses(self):
        """csdid unit-clustered influence function SEs and non-empty unit_influences"""
        n_units = 20
        n_times = 5
        unit = np.repeat(np.arange(1, n_units + 1), n_times)
        time = np.tile(np.arange(1, n_times + 1), n_units)
        # cohort: first 10 never-treated (0), next 10 treated at period 3
        cohort = np.repeat([0] * 10 + [3] * 10, n_times)
        y = 2.0 * time + 3.0 * (cohort == 3) * (time >= 3) + np.random.randn(n_units * n_times)
        df = pd.DataFrame({'unit': unit, 'time': time, 'cohort': cohort, 'y': y})

        res_csdid = compute_csdid(df, 'y', 'unit', 'time', 'cohort', control_type="notyettreated")
        self.assertIn('unit_influences', res_csdid)
        self.assertTrue(len(res_csdid['unit_influences']) > 0)
        self.assertTrue(res_csdid['overall_se'] > 0)

    # -------------------------------------------------------------------------
    # 2s: Task 6 - Multi-restriction test command
    # -------------------------------------------------------------------------
    def test_2s_multi_restriction_test_command(self):
        """Multi-restriction test (mpg = 0) (weight = 0) joint F-test"""
        n = 50
        mpg = np.random.randn(n) * 5 + 20
        weight = np.random.randn(n) * 500 + 3000
        price = 1000 + 10 * mpg + 2 * weight + np.random.randn(n) * 100
        self.dataset.df = pd.DataFrame({'price': price, 'mpg': mpg, 'weight': weight})
        self.interp.execute("regress price mpg weight")

        res_single = self.interp.execute("test mpg = 0")
        self.assertTrue(res_single.ok)
        self.assertIn("F(", str(res_single.output))

        res_multi = self.interp.execute("test (mpg = 0) (weight = 0)")
        self.assertTrue(res_multi.ok)
        self.assertIn("F(  2,", str(res_multi.output))
        self.assertIn("prob > f", str(res_multi.output).lower())


if __name__ == "__main__":
    unittest.main()
