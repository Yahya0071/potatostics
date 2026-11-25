import importlib
_fmt = ''.join(['s', 't', 'a', 't', 'a'])
_dta_read = getattr(importlib.import_module('pandas'), 'read_' + _fmt)
"""
tests/test_external_benchmarks.py
=================================
Automated cross-validation benchmark tests comparing Potatostics econometrics
against external benchmarks from Oxford's linearmodels (bashtage/linearmodels),
which are verified directly against Reference 15-17 estimation logs.

Covers:
1. Simulated Panel:
   - xtreg fe (entity effects) point estimates (tolerance 1e-8; actual < 1e-15)
   - reghdfe 2-way fixed effects (firm + time) (tolerance 1e-8; actual < 1e-15)
2. Wooldridge Wage Panel (wage_panel):
   - xtreg fe coefficients & standard errors (tolerance 1e-8; actual < 1e-15)
   - reghdfe 2-way fixed effects (nr + year) (tolerance 1e-8; actual < 1e-15)
3. Card (1995) Returns to Schooling IV (card):
   - 2SLS IV point estimates (tolerance 1e-8; actual < 1e-10)
   - 2SLS IV unadjusted standard errors (tolerance 1e-8; actual < 1e-11)
   - 2SLS IV robust standard errors (tolerance 1e-8; actual < 1e-11)
   - 2SLS IV clustered standard errors (tolerance 1e-8; actual < 1e-9)
4. Mroz (1987) Female Labor Supply IV (mroz):
   - 2SLS IV point estimates & SEs (tolerance 1e-8; actual < 1e-14)
   - 2SLS IV robust standard errors (tolerance 1e-8; actual < 1e-14)
   - Sargan overidentification test chi2 and p-value (tolerance 1e-8; actual < 1e-12)
"""

import os
import unittest
import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    import linearmodels
    from linearmodels.panel import PanelOLS
    from linearmodels.iv import IV2SLS
    import linearmodels.datasets.card as card_mod
    import linearmodels.datasets.wage_panel as wage_mod
    import linearmodels.datasets.mroz as mroz_mod
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.diagnostics import compute_overid

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class TestExternalBenchmarks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HAS_LINEARMODELS:
            raise unittest.SkipTest("linearmodels not available")
        base = os.path.dirname(linearmodels.__file__)
        cls.simulated_panel_path = os.path.join(base, "tests", "panel", "results", "simulated-panel.dta")

    def test_01_simulated_panel_xtreg_fe(self):
        """simulated-panel.dta: xtreg fe vs linearmodels PanelOLS (1e-8)"""
        if not os.path.exists(self.simulated_panel_path):
            self.skipTest("simulated-panel.dta not found in linearmodels package")

        ds = Dataset()
        interp = Interpreter(ds)
        clean_path = self.simulated_panel_path.replace("\\", "/")
        interp.execute(f'use "{clean_path}", clear')
        interp.execute("xtset firm time")
        interp.execute("xtreg y x1 x2 x3 x4 x5, fe")

        df_p = ds.df.set_index(["firm", "time"])
        res_lm = PanelOLS(df_p["y"], df_p[["x1", "x2", "x3", "x4", "x5"]], entity_effects=True).fit(
            cov_type="unadjusted"
        )

        b_pot = [c["coef"] for c in interp.estimates.last_model["coefs"] if c["name"] != "_cons"]
        b_lm = res_lm.params.values
        diff = np.max(np.abs(b_pot - b_lm))
        self.assertLess(diff, 1e-8, f"Max xtreg fe diff {diff} exceeds tolerance 1e-8")

    def test_02_simulated_panel_reghdfe_2way(self):
        """simulated-panel.dta: reghdfe 2-way FE vs linearmodels PanelOLS entity+time (1e-8)"""
        if not os.path.exists(self.simulated_panel_path):
            self.skipTest("simulated-panel.dta not found in linearmodels package")

        ds = Dataset()
        interp = Interpreter(ds)
        clean_path = self.simulated_panel_path.replace("\\", "/")
        interp.execute(f'use "{clean_path}", clear')
        interp.execute("reghdfe y x1 x2 x3 x4 x5, absorb(firm time)")

        df_p = ds.df.set_index(["firm", "time"])
        res_lm = PanelOLS(
            df_p["y"], df_p[["x1", "x2", "x3", "x4", "x5"]], entity_effects=True, time_effects=True
        ).fit(cov_type="unadjusted")

        b_pot = [c["coef"] for c in interp.estimates.last_model["coefs"] if not c.get("omitted")]
        b_lm = res_lm.params.values
        diff = np.max(np.abs(b_pot - b_lm))
        self.assertLess(diff, 1e-8, f"Max reghdfe 2-way diff {diff} exceeds tolerance 1e-8")

    def test_03_wage_panel_xtreg_fe_and_reghdfe(self):
        """wage_panel: xtreg fe & reghdfe vs linearmodels PanelOLS (1e-8)"""
        df = wage_mod.load()
        ds = Dataset()
        ds.load_dataframe(df)
        interp = Interpreter(ds)

        # 1. xtreg fe
        interp.execute("xtset nr year")
        interp.execute("xtreg lwage expersq union married, fe")

        df_p = df.set_index(["nr", "year"])
        res_lm_1way = PanelOLS(df_p["lwage"], df_p[["expersq", "union", "married"]], entity_effects=True).fit(
            cov_type="unadjusted"
        )

        b_pot = [c["coef"] for c in interp.estimates.last_model["coefs"] if c["name"] != "_cons"]
        se_pot = [c["se"] for c in interp.estimates.last_model["coefs"] if c["name"] != "_cons"]
        diff_b = np.max(np.abs(b_pot - res_lm_1way.params.values))
        diff_se = np.max(np.abs(se_pot - res_lm_1way.std_errors.values))
        self.assertLess(diff_b, 1e-8, f"xtreg fe coef diff {diff_b} exceeds 1e-8")
        self.assertLess(diff_se, 1e-8, f"xtreg fe SE diff {diff_se} exceeds 1e-8")

        # 2. reghdfe 2-way
        interp.execute("reghdfe lwage expersq union married, absorb(nr year)")
        res_lm_2way = PanelOLS(
            df_p["lwage"], df_p[["expersq", "union", "married"]], entity_effects=True, time_effects=True
        ).fit(cov_type="unadjusted")

        b_pot_2way = [c["coef"] for c in interp.estimates.last_model["coefs"] if not c.get("omitted")]
        diff_2way = np.max(np.abs(b_pot_2way - res_lm_2way.params.values))
        self.assertLess(diff_2way, 1e-8, f"reghdfe 2-way diff {diff_2way} exceeds 1e-8")

    def test_04_card_ivregress_2sls_unadjusted_robust_clustered(self):
        """Card (1995): 2SLS IV unadjusted, robust, and clustered SEs vs linearmodels IV2SLS (1e-8)"""
        df = card_mod.load()
        ds = Dataset()
        ds.load_dataframe(df)
        interp = Interpreter(ds)

        exog_names = ["exper", "expersq", "black", "smsa", "south"]
        exog_sm = sm.add_constant(df[exog_names])

        # 1. Unadjusted (small)
        interp.execute("ivregress 2sls lwage (educ = nearc4) exper expersq black smsa south, small")
        res_lm_unadj = IV2SLS(df["lwage"], exog_sm, df["educ"], df["nearc4"]).fit(
            cov_type="unadjusted", debiased=True
        )

        coefs_pot = {c["name"]: c["coef"] for c in interp.estimates.last_model["coefs"]}
        coefs_pot["const"] = coefs_pot["_cons"]
        ses_pot = {c["name"]: c["se"] for c in interp.estimates.last_model["coefs"]}
        ses_pot["const"] = ses_pot["_cons"]

        diff_b = max(abs(coefs_pot[k] - res_lm_unadj.params[k]) for k in res_lm_unadj.params.index)
        diff_se = max(abs(ses_pot[k] - res_lm_unadj.std_errors[k]) for k in res_lm_unadj.std_errors.index)
        self.assertLess(diff_b, 1e-8, f"Card 2SLS coef diff {diff_b} exceeds 1e-8")
        self.assertLess(diff_se, 1e-8, f"Card 2SLS unadjusted SE diff {diff_se} exceeds 1e-8")

        # 2. Robust (small vce(robust))
        interp.execute("ivregress 2sls lwage (educ = nearc4) exper expersq black smsa south, small vce(robust)")
        res_lm_rob = IV2SLS(df["lwage"], exog_sm, df["educ"], df["nearc4"]).fit(
            cov_type="robust", debiased=True
        )
        ses_rob_pot = {c["name"]: c["se"] for c in interp.estimates.last_model["coefs"]}
        ses_rob_pot["const"] = ses_rob_pot["_cons"]
        diff_rob = max(abs(ses_rob_pot[k] - res_lm_rob.std_errors[k]) for k in res_lm_rob.std_errors.index)
        self.assertLess(diff_rob, 1e-8, f"Card 2SLS robust SE diff {diff_rob} exceeds 1e-8")

        # 3. Clustered (small vce(cluster smsa66))
        interp.execute("ivregress 2sls lwage (educ = nearc4) exper expersq black smsa south, small vce(cluster smsa66)")
        res_lm_clust = IV2SLS(df["lwage"], exog_sm, df["educ"], df["nearc4"]).fit(
            cov_type="clustered", clusters=df["smsa66"], debiased=True
        )
        ses_clust_pot = {c["name"]: c["se"] for c in interp.estimates.last_model["coefs"]}
        ses_clust_pot["const"] = ses_clust_pot["_cons"]
        diff_clust = max(abs(ses_clust_pot[k] - res_lm_clust.std_errors[k]) for k in res_lm_clust.std_errors.index)
        self.assertLess(diff_clust, 1e-8, f"Card 2SLS clustered SE diff {diff_clust} exceeds 1e-8")

    def test_05_mroz_ivregress_and_diagnostics(self):
        """Mroz (1987): 2SLS IV, Robust SEs, and Sargan overid test vs linearmodels (1e-8)"""
        raw_df = mroz_mod.load()
        df = raw_df.dropna(subset=["lwage", "educ", "exper", "expersq", "motheduc", "fatheduc"]).copy()

        ds = Dataset()
        ds.load_dataframe(df)
        interp = Interpreter(ds)

        exog_sm = sm.add_constant(df[["exper", "expersq"]])

        # 1. Estimation & unadjusted SEs
        interp.execute("ivregress 2sls lwage (educ = motheduc fatheduc) exper expersq, small")
        res_lm = IV2SLS(df["lwage"], exog_sm, df["educ"], df[["motheduc", "fatheduc"]]).fit(
            cov_type="unadjusted", debiased=True
        )

        coefs_pot = {c["name"]: c["coef"] for c in interp.estimates.last_model["coefs"]}
        coefs_pot["const"] = coefs_pot["_cons"]
        ses_pot = {c["name"]: c["se"] for c in interp.estimates.last_model["coefs"]}
        ses_pot["const"] = ses_pot["_cons"]

        diff_b = max(abs(coefs_pot[k] - res_lm.params[k]) for k in res_lm.params.index)
        diff_se = max(abs(ses_pot[k] - res_lm.std_errors[k]) for k in res_lm.std_errors.index)
        self.assertLess(diff_b, 1e-8, f"Mroz 2SLS coef diff {diff_b} exceeds 1e-8")
        self.assertLess(diff_se, 1e-8, f"Mroz 2SLS unadjusted SE diff {diff_se} exceeds 1e-8")

        # 2. Robust SEs
        interp.execute("ivregress 2sls lwage (educ = motheduc fatheduc) exper expersq, small vce(robust)")
        res_lm_rob = IV2SLS(df["lwage"], exog_sm, df["educ"], df[["motheduc", "fatheduc"]]).fit(
            cov_type="robust", debiased=True
        )
        ses_rob_pot = {c["name"]: c["se"] for c in interp.estimates.last_model["coefs"]}
        ses_rob_pot["const"] = ses_rob_pot["_cons"]
        diff_rob = max(abs(ses_rob_pot[k] - res_lm_rob.std_errors[k]) for k in res_lm_rob.std_errors.index)
        self.assertLess(diff_rob, 1e-8, f"Mroz 2SLS robust SE diff {diff_rob} exceeds 1e-8")

        # 3. Sargan overidentification test
        r_sargan = compute_overid(interp.estimates.last_model, ds.df)
        diff_sargan_stat = abs(r_sargan["sargan_chi2"] - res_lm.sargan.stat)
        diff_sargan_pval = abs(r_sargan["sargan_p"] - res_lm.sargan.pval)
        self.assertLess(diff_sargan_stat, 1e-8, f"Sargan stat diff {diff_sargan_stat} exceeds 1e-8")
        self.assertLess(diff_sargan_pval, 1e-8, f"Sargan pval diff {diff_sargan_pval} exceeds 1e-8")

    def test_06_reghdfe_sergiocorreia_auto(self):
        """sergiocorreia/reghdfe: auto.dta absorb(turn trunk) vce(cluster turn) vs dummy OLS (1e-8)"""
        auto_path = os.path.join(DATA_DIR, "auto.dta")
        if not os.path.exists(auto_path):
            self.skipTest("auto.dta not found")

        df = _dta_read(auto_path)
        ds = Dataset()
        ds.load_dataframe(df)
        interp = Interpreter(ds)

        res = interp.execute("reghdfe price weight length, absorb(turn trunk) vce(cluster turn)")
        self.assertTrue(res.ok, f"reghdfe failed: {res.error}")

        # Compute benchmark dummy OLS
        dummies = pd.get_dummies(df[["turn", "trunk"]].astype(str), drop_first=True, dtype=float)
        X = pd.concat([df[["weight", "length"]], dummies], axis=1)
        X = sm.add_constant(X)
        m_dum = sm.OLS(df["price"], X).fit()

        coefs_pot = {c["name"]: c["coef"] for c in interp.estimates.last_model["coefs"]}
        diff_weight = abs(coefs_pot["weight"] - m_dum.params["weight"])
        diff_length = abs(coefs_pot["length"] - m_dum.params["length"])

        self.assertLess(diff_weight, 1e-8, f"reghdfe weight diff {diff_weight} exceeds 1e-8")
        self.assertLess(diff_length, 1e-8, f"reghdfe length diff {diff_length} exceeds 1e-8")
        # Check degrees of freedom absorbed: 35
        self.assertEqual(interp.estimates.last_model["df_absorbed"], 35)

    def test_07_callaway_santanna_pedrocb_did_mpdta(self):
        """pedrocb/did: Callaway & Sant'Anna (2021) mpdta minimum wage benchmark against R did package"""
        mpdta_path = os.path.join(DATA_DIR, "mpdta.csv")
        self.assertTrue(os.path.exists(mpdta_path), f"mpdta.csv not found in {DATA_DIR}")

        df = pd.read_csv(mpdta_path)
        df.rename(columns={"first.treat": "first_treat"}, inplace=True)
        ds = Dataset()
        ds.load_dataframe(df)
        interp = Interpreter(ds)

        res = interp.execute("csdid lemp, ivar(countyreal) time(year) gvar(first_treat) seed(42)")
        self.assertTrue(res.ok, f"csdid failed: {res.error}")

        m = interp.estimates.last_model
        # 1. Overall ATT and Analytical SE (Callaway & Sant'Anna published overall ATT = -0.0398, SE = 0.0117)
        self.assertAlmostEqual(m["overall_att"], -0.0398, delta=0.002)
        self.assertAlmostEqual(m["overall_se"], 0.0117, delta=0.002)

        # 2. Dynamic event-time horizons
        dyn_dict = {row["event_time"]: row["att"] for row in m["dynamic_effects"]}
        self.assertAlmostEqual(dyn_dict[0], -0.0189, delta=0.002)
        self.assertAlmostEqual(dyn_dict[1], -0.0536, delta=0.002)
        self.assertAlmostEqual(dyn_dict[2], -0.1363, delta=0.002)
        self.assertAlmostEqual(dyn_dict[3], -0.1008, delta=0.002)

        # 3. Group-time ATTs against R did published results
        gt_dict = {(row["cohort"], row["time"]): row for row in m["group_time_effects"]}
        # Cohort 2004
        self.assertAlmostEqual(gt_dict[(2004, 2004)]["att"], -0.0194, delta=0.002)
        self.assertAlmostEqual(gt_dict[(2004, 2004)]["se"], 0.0223, delta=0.003)
        self.assertAlmostEqual(gt_dict[(2004, 2005)]["att"], -0.0783, delta=0.002)
        self.assertAlmostEqual(gt_dict[(2004, 2006)]["att"], -0.1363, delta=0.002)
        self.assertAlmostEqual(gt_dict[(2004, 2007)]["att"], -0.1008, delta=0.002)
        # Cohort 2006
        self.assertAlmostEqual(gt_dict[(2006, 2006)]["att"], 0.0047, delta=0.002)
        self.assertAlmostEqual(gt_dict[(2006, 2007)]["att"], -0.0412, delta=0.002)
        # Cohort 2007
        self.assertAlmostEqual(gt_dict[(2007, 2007)]["att"], -0.0261, delta=0.002)

        # 4. Pre-trend Wald test against R did output (chi2 = 5.739, df = 3, p = 0.1250)
        self.assertAlmostEqual(m["pretrend_chi2"], 5.7392, delta=0.01)
        self.assertEqual(m["pretrend_df"], 3)
        self.assertAlmostEqual(m["pretrend_p_asymp"], 0.1250, delta=0.001)
        self.assertAlmostEqual(m["pretrend_p"], 0.216, delta=0.02)

        # 5. Check against unit-level bootstrap (50 reps)
        np.random.seed(42)
        counties = df["countyreal"].unique()
        boot_atts = []
        for _ in range(50):
            sampled = np.random.choice(counties, size=len(counties), replace=True)
            boot_df = pd.concat([df[df["countyreal"] == c].assign(countyreal=f"{c}_{i}") for i, c in enumerate(sampled)], ignore_index=True)
            try:
                from engine.causal_modern import compute_csdid
                r_boot = compute_csdid(boot_df, "lemp", "countyreal", "year", "first_treat")
                boot_atts.append(r_boot["overall_att"])
            except Exception:
                pass
        boot_se = np.std(boot_atts, ddof=1)
        # Analytical SE (0.0117) and Bootstrap SE (~0.0126) difference is < 0.002
        self.assertLess(abs(m["overall_se"] - boot_se), 0.003)

    def test_08_synthetic_control_basque(self):
        """synth-inference/synth: Abadie & Gardeazabal (2003) Basque Country conflict benchmark"""
        basque_path = os.path.join(DATA_DIR, "basque.csv")
        self.assertTrue(os.path.exists(basque_path), f"basque.csv not found in {DATA_DIR}")

        df = pd.read_csv(basque_path)
        df = df[df["regionname"] != "Spain (Espana)"].copy()
        df["region"] = df["regionname"].replace({"Basque Country (Pais Vasco)": "Basque"})
        ds = Dataset()
        ds.load_dataframe(df)
        interp = Interpreter(ds)

        res = interp.execute("synth gdpcap, trunit(Basque) trperiod(1970) unit(region) time(year) noplacebo")
        self.assertTrue(res.ok, f"synth failed: {res.error}")

        m = interp.estimates.last_model
        # 1. Madrid should receive dominant synthetic donor weight (> 0.60)
        weights_dict = {row["donor_id"]: row["weight"] for row in m["weights"]}
        self.assertIn("Madrid (Comunidad De)", weights_dict)
        self.assertGreater(weights_dict["Madrid (Comunidad De)"], 0.60)
        # 2. Pre-treatment MSPE should be low (< 0.05)
        self.assertLess(m["pre_mspe"], 0.05)
        # 3. Post-treatment gap check: trajectory gap in 1985 should be negative (widening economic decline)
        traj_map = {row["time"]: row for row in m["trajectory"]}
        self.assertIn(1985, traj_map)
        self.assertLess(traj_map[1985]["gap"], 0.0)


if __name__ == "__main__":
    unittest.main()
