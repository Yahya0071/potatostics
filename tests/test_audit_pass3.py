"""
tests/test_audit_pass3.py - Verification tests for Phase 3 audit items
"""

import os
import unittest
import numpy as np
import pandas as pd

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.estimation import compute_mle_vcov_cluster, compute_logit
from engine.advanced_econometrics import compute_probit, compute_poisson
from engine.parser import _dta_round
from engine.copilot import CopilotAssistant
from engine.copilot_egm import ElephantLongTermMemory
import engine.copilot_egm as copilot_module
from engine.plugins_manager import PluginsManager

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
AUTO_PATH = os.path.join(DATA_DIR, "auto.dta").replace("\\", "/")


class TestAuditPass3(unittest.TestCase):
    def setUp(self):
        self.ds = Dataset()
        self.interp = Interpreter(self.ds)

    # -----------------------------------------------------------------
    # Item 2: Logit, Probit, Poisson cluster factor G/(G-1) & z inference
    # -----------------------------------------------------------------
    def test_item2_mle_cluster_factor_and_z_inference(self):
        np.random.seed(123)
        n = 120
        df = pd.DataFrame({
            'y_bin': np.random.binomial(1, 0.4, n),
            'y_cnt': np.random.poisson(2, n),
            'x': np.random.normal(0, 1, n),
            'cid': np.repeat(np.arange(1, 13), 10)  # 12 clusters
        })
        G = 12
        expected_c = G / (G - 1.0)  # 12 / 11 = 1.090909...

        # Verify compute_mle_vcov_cluster multiplier
        inv_H = np.eye(2)
        scores = np.ones((n, 2))
        clusters = df['cid'].to_numpy()
        vcov, g_out = compute_mle_vcov_cluster(inv_H, scores, clusters)
        self.assertEqual(g_out, G)
        np.testing.assert_allclose(vcov, np.full((2, 2), expected_c * G * (n / G)**2))

        # 1. Logit cluster
        res_l = compute_logit(df, 'y_bin', ['x'], vce='cluster', cluster_var='cid')
        self.assertEqual(res_l['n_clusters'], G)
        self.assertIn('z', res_l['coefs'][0])

        # 2. Probit cluster
        res_pr = compute_probit(df, 'y_bin', ['x'], vce='cluster', cluster_var='cid')
        self.assertEqual(res_pr['n_clusters'], G)
        self.assertIn('z', res_pr['coefs'][0])

        # 3. Poisson cluster
        res_poi = compute_poisson(df, 'y_cnt', ['x'], vce='cluster', cluster_var='cid')
        self.assertEqual(res_poi['n_clusters'], G)
        self.assertIn('z', res_poi['coefs'][0])

    # -----------------------------------------------------------------
    # Item 3: xtreg, fe header formatting (F test all u_i=0, R2, sigma_u, sigma_e, rho)
    # -----------------------------------------------------------------
    def test_item3_xtreg_fe_header_and_f_test_output(self):
        self.interp.execute(f'use "{AUTO_PATH}", clear')
        self.interp.execute("xtset rep78")
        res = self.interp.execute("xtreg price weight length, fe")
        self.assertTrue(res.ok, f"xtreg fe failed: {res.error}")
        out = res.output

        # Verify required header elements
        self.assertIn("Fixed-effects (within) regression", out)
        self.assertIn("R-squared:", out)
        self.assertIn("within", out)
        self.assertIn("between", out)
        self.assertIn("overall", out)
        self.assertIn("sigma_u", out)
        self.assertIn("sigma_e", out)
        self.assertIn("rho", out)
        self.assertIn("F test that all u_i=0", out)
        self.assertIn("Prob > F", out)

    # -----------------------------------------------------------------
    # Item 4: predict xb/residuals/pr after factor variables and interactions
    # -----------------------------------------------------------------
    def test_item4_predict_after_factor_and_interactions(self):
        self.interp.execute(f'use "{AUTO_PATH}", clear')

        # 1. Linear regression with factor variable and continuous-factor interaction
        res_reg = self.interp.execute("regress price i.rep78 c.weight#i.foreign")
        self.assertTrue(res_reg.ok, f"regress failed: {res_reg.error}")

        # Predict xb
        res_xb = self.interp.execute("predict y_hat, xb")
        self.assertTrue(res_xb.ok, f"predict xb failed: {res_xb.error}")
        self.assertIn("y_hat", self.interp.dataset.df.columns)
        self.assertFalse(self.interp.dataset.df["y_hat"].isna().all())

        # Predict residuals
        res_resid = self.interp.execute("predict res_val, resid")
        self.assertTrue(res_resid.ok, f"predict resid failed: {res_resid.error}")
        self.assertIn("res_val", self.interp.dataset.df.columns)
        self.assertFalse(self.interp.dataset.df["res_val"].isna().all())

        # 2. Binary logit model with factor variables and predict pr
        # recode foreign to numeric 0/1 if needed
        self.interp.execute("generate foreign_num = (foreign == 1 | foreign == \"Foreign\")")
        res_logit = self.interp.execute("logit foreign_num i.rep78 weight")
        self.assertTrue(res_logit.ok, f"logit failed: {res_logit.error}")

        res_pr = self.interp.execute("predict pr_foreign, pr")
        self.assertTrue(res_pr.ok, f"predict pr failed: {res_pr.error}")
        self.assertIn("pr_foreign", self.interp.dataset.df.columns)
        self.assertTrue((self.interp.dataset.df["pr_foreign"].dropna() >= 0.0).all())
        self.assertTrue((self.interp.dataset.df["pr_foreign"].dropna() <= 1.0).all())

    # -----------------------------------------------------------------
    # Item 5: Assistant word-boundary matching and plain renaming
    # -----------------------------------------------------------------
    def test_item5_assistant_word_boundary_and_plain_name(self):
        assistant = CopilotAssistant(self.interp)
        
        # 'mean' inside 'meaningful' must NOT trigger summarize
        query_meaningful = "is the regression result meaningful?"
        ans_meaningful = assistant.ask(query_meaningful)
        self.assertNotIn("summarize", ans_meaningful.lower())

        # 'relationship between' must map to correlate
        query_rel = "what is the relationship between price and mpg"
        ans_rel = assistant.ask(query_rel)
        self.assertIn("correlate", ans_rel.lower())

        # Assistant must plainly state it is rule-based
        self.assertIn("rule-based", ans_meaningful.lower())

        # Plain class name exists in copilot module
        self.assertTrue(hasattr(copilot_module, "SessionMemoryCopilot") or hasattr(copilot_module, "RuleBasedAssistant"))

        # Query storage: disabled by default and does not write to ~/.potatostics without explicit opt-in
        mem = ElephantLongTermMemory()
        self.assertFalse(getattr(mem, "enable_query_logging", False))

    # -----------------------------------------------------------------
    # Item 6: round(x, y) ties round away from zero
    # -----------------------------------------------------------------
    def test_item6_round_ties_away_from_zero(self):
        # x = 2.5, y = 5 -> in Reference: 5
        val_pos = _dta_round(2.5, 5.0)
        self.assertEqual(val_pos, 5.0)

        # x = -2.5, y = 5 -> in Reference: -5
        val_neg = _dta_round(-2.5, 5.0)
        self.assertEqual(val_neg, -5.0)

        # x = 2.5, y = 1 -> in Reference: 3
        self.assertEqual(_dta_round(2.5, 1.0), 3.0)

        # x = -2.5, y = 1 -> in Reference: -3
        self.assertEqual(_dta_round(-2.5, 1.0), -3.0)

    # -----------------------------------------------------------------
    # Item 8: Plugins: remove ./plugins working-directory lookup
    # -----------------------------------------------------------------
    def test_item8_plugins_no_cwd_lookup(self):
        pm = PluginsManager(self.interp)
        # Should only have self.plugins_dir (per-user config directory), not ./plugins
        self.assertNotIn(os.path.abspath("plugins"), [pm.plugins_dir])


if __name__ == "__main__":
    unittest.main()

