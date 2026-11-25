"""
tests/test_reference.py - Reference Validation Suite (Defensible Econometric Gate)
Compares Potatostics estimation outputs to statsmodels, linearmodels,
and documented Benchmark formulas stored in tests/golden_reference.json.
Uses real assertions with strict tolerances:
  - 1e-8 where algebra is identical (OLS, 2SLS, Panel FE, reghdfe, ADF)
  - 1e-6 for numerical optimizers (Logit, Probit, Poisson, qreg, tobit)
"""

import os
import sys
import json
import unittest
import numpy as np
import pandas as pd

# Fallback to project root if not installed as wheel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.estimation import (
    compute_regression,
    compute_logit,
    compute_ivregress_2sls,
    compute_panel_fe
)
from engine.advanced_econometrics import (
    compute_probit,
    compute_poisson,
    compute_reghdfe,
    compute_tobit,
    compute_qreg
)
from engine.timeseries import compute_dfuller

TOL_ALGEBRA = 1e-8
TOL_OPTIM = 1e-6

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_reference.json")


class TestReferenceSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GOLDEN_PATH):
            from tests.make_golden import generate_golden
            generate_golden()
        with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
            cls.golden = json.load(f)

        cls.df_ols = pd.DataFrame(cls.golden["datasets"]["df_ols"])
        cls.df_logit = pd.DataFrame(cls.golden["datasets"]["df_logit"])
        cls.df_probit = pd.DataFrame(cls.golden["datasets"]["df_probit"])
        cls.df_poisson = pd.DataFrame(cls.golden["datasets"]["df_poisson"])
        cls.df_iv = pd.DataFrame(cls.golden["datasets"]["df_iv"])
        cls.df_panel = pd.DataFrame(cls.golden["datasets"]["df_panel"])
        cls.df_tobit = pd.DataFrame(cls.golden["datasets"]["df_tobit"])
        cls.ts_series = np.array(cls.golden["datasets"]["ts_series"])

    # =========================================================================
    # 1. OLS Reference Tests (Algebraic exactness: 1e-8)
    # =========================================================================
    def test_01_ols_classical(self):
        """OLS Classical: exact algebraic equivalence with statsmodels OLS (1e-8)"""
        ref = self.golden["reference"]["ols"]["classical"]
        res = compute_regression(self.df_ols, "y", ["x1", "x2"], robust=False, vce="ols")
        
        # Coefs & SEs (order: x1, x2, _cons)
        for i, name in enumerate(["x1", "x2", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_ALGEBRA,
                                   msg=f"OLS classical coef mismatch on {name}")
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["bse"][i], delta=TOL_ALGEBRA,
                                   msg=f"OLS classical SE mismatch on {name}")
            self.assertAlmostEqual(res["coefs"][i]["t"], ref["tvalues"][i], delta=TOL_ALGEBRA,
                                   msg=f"OLS classical t mismatch on {name}")
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["pvalues"][i], delta=TOL_ALGEBRA,
                                   msg=f"OLS classical p mismatch on {name}")
        self.assertAlmostEqual(res["r2"], ref["r2"], delta=TOL_ALGEBRA)
        self.assertAlmostEqual(res["adj_r2"], ref["adj_r2"], delta=TOL_ALGEBRA)
        self.assertAlmostEqual(res["f_stat"], ref["fvalue"], delta=TOL_ALGEBRA)
        self.assertAlmostEqual(res["prob_f"], ref["f_pvalue"], delta=TOL_ALGEBRA)

    def test_02_ols_hc1(self):
        """OLS HC1: Robust sandwich formula N/(N-K) sandwich (1e-8)"""
        ref = self.golden["reference"]["ols"]["hc1"]
        res = compute_regression(self.df_ols, "y", ["x1", "x2"], robust=True, vce="hc1")
        for i, name in enumerate(["x1", "x2", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["bse"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["t"], ref["tvalues"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["pvalues"][i], delta=TOL_ALGEBRA)

    def test_03_ols_hc2(self):
        """OLS HC2: 1/(1-h_ii) leverage-adjusted sandwich (1e-8)"""
        ref = self.golden["reference"]["ols"]["hc2"]
        res = compute_regression(self.df_ols, "y", ["x1", "x2"], vce="hc2")
        for i, name in enumerate(["x1", "x2", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["bse"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["t"], ref["tvalues"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["pvalues"][i], delta=TOL_ALGEBRA)

    def test_04_ols_hc3(self):
        """OLS HC3: 1/(1-h_ii)^2 leverage-adjusted sandwich (1e-8)"""
        ref = self.golden["reference"]["ols"]["hc3"]
        res = compute_regression(self.df_ols, "y", ["x1", "x2"], vce="hc3")
        for i, name in enumerate(["x1", "x2", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["bse"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["t"], ref["tvalues"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["pvalues"][i], delta=TOL_ALGEBRA)

    def test_05_ols_cluster(self):
        """OLS Cluster: Benchmark formula G/(G-1) * (N-1)/(N-K) cluster adjustment (1e-8)"""
        ref = self.golden["reference"]["ols"]["cluster"]
        res = compute_regression(self.df_ols, "y", ["x1", "x2"], vce="cluster", cluster_var="cluster")
        for i, name in enumerate(["x1", "x2", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["bse"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["t"], ref["tvalues"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["pvalues"][i], delta=TOL_ALGEBRA)

    # =========================================================================
    # 2. Logit Reference Tests (Optimizer tolerance: 1e-6)
    # =========================================================================
    def test_06_logit_mle(self):
        """Logit MLE Classical: Newton-Raphson vs statsmodels Logit (1e-6)"""
        ref = self.golden["reference"]["logit"]
        res = compute_logit(self.df_logit, "y", ["x1", "x2"], vce="mle")
        for i, name in enumerate(["x1", "x2", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["mle"]["bse"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["z"], ref["mle"]["zvalues"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["mle"]["pvalues"][i], delta=TOL_OPTIM)
        self.assertAlmostEqual(res["ll"], ref["ll"], delta=TOL_OPTIM)
        self.assertAlmostEqual(res["lr_chi2"], ref["lr_chi2"], delta=TOL_OPTIM)
        self.assertAlmostEqual(res["pseudo_r2"], ref["pseudo_r2"], delta=TOL_OPTIM)

    def test_07_logit_robust(self):
        """Logit Robust: Reference N/(N-1) sandwich multiplier (1e-6)"""
        ref = self.golden["reference"]["logit"]
        res = compute_logit(self.df_logit, "y", ["x1", "x2"], vce="robust")
        for i, name in enumerate(["x1", "x2", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["robust"]["bse"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["z"], ref["robust"]["zvalues"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["robust"]["pvalues"][i], delta=TOL_OPTIM)

    def test_08_logit_cluster(self):
        """Logit Cluster: Reference G/(G-1) multiplier and z-inference per convention decision (1e-6)"""
        ref = self.golden["reference"]["logit"]
        res = compute_logit(self.df_logit, "y", ["x1", "x2"], vce="cluster", cluster_var="cluster")
        for i, name in enumerate(["x1", "x2", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["cluster"]["bse"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["z"], ref["cluster"]["zvalues"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["cluster"]["pvalues"][i], delta=TOL_OPTIM)

    # =========================================================================
    # 3. Probit Reference Tests (Optimizer tolerance: 1e-6)
    # =========================================================================
    def test_09_probit_mle(self):
        """Probit MLE Classical: Newton-Raphson vs statsmodels Probit (1e-6)"""
        ref = self.golden["reference"]["probit"]
        res = compute_probit(self.df_probit, "y", ["x1", "x2"], vce="mle")
        for i, name in enumerate(["x1", "x2", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["mle"]["bse"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["z"], ref["mle"]["zvalues"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["mle"]["pvalues"][i], delta=TOL_OPTIM)
        self.assertAlmostEqual(res["ll"], ref["ll"], delta=TOL_OPTIM)
        self.assertAlmostEqual(res["lr_chi2"], ref["lr_chi2"], delta=TOL_OPTIM)
        self.assertAlmostEqual(res["pseudo_r2"], ref["pseudo_r2"], delta=TOL_OPTIM)

    def test_10_probit_robust(self):
        """Probit Robust: Reference N/(N-1) sandwich multiplier (1e-6)"""
        ref = self.golden["reference"]["probit"]
        res = compute_probit(self.df_probit, "y", ["x1", "x2"], vce="robust")
        for i, name in enumerate(["x1", "x2", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["robust"]["bse"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["z"], ref["robust"]["zvalues"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["robust"]["pvalues"][i], delta=TOL_OPTIM)

    def test_11_probit_cluster(self):
        """Probit Cluster: Reference G/(G-1) multiplier and z-inference per convention decision (1e-6)"""
        ref = self.golden["reference"]["probit"]
        res = compute_probit(self.df_probit, "y", ["x1", "x2"], vce="cluster", cluster_var="cluster")
        for i, name in enumerate(["x1", "x2", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["cluster"]["bse"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["z"], ref["cluster"]["zvalues"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["cluster"]["pvalues"][i], delta=TOL_OPTIM)

    # =========================================================================
    # 4. Poisson Reference Tests (Optimizer tolerance: 1e-6)
    # =========================================================================
    def test_12_poisson_mle(self):
        """Poisson MLE Classical: Newton-Raphson vs statsmodels Poisson (1e-6)"""
        ref = self.golden["reference"]["poisson"]
        res = compute_poisson(self.df_poisson, "y", ["x1", "x2"], vce="mle")
        for i, name in enumerate(["x1", "x2", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["mle"]["bse"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["z"], ref["mle"]["zvalues"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["mle"]["pvalues"][i], delta=TOL_OPTIM)
        self.assertAlmostEqual(res["ll"], ref["ll"], delta=TOL_OPTIM)
        self.assertAlmostEqual(res["lr_chi2"], ref["lr_chi2"], delta=TOL_OPTIM)
        self.assertAlmostEqual(res["pseudo_r2"], ref["pseudo_r2"], delta=TOL_OPTIM)

    def test_13_poisson_robust(self):
        """Poisson Robust: Reference N/(N-1) sandwich multiplier (1e-6)"""
        ref = self.golden["reference"]["poisson"]
        res = compute_poisson(self.df_poisson, "y", ["x1", "x2"], vce="robust")
        for i, name in enumerate(["x1", "x2", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["robust"]["bse"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["z"], ref["robust"]["zvalues"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["robust"]["pvalues"][i], delta=TOL_OPTIM)

    def test_14_poisson_cluster(self):
        """Poisson Cluster: Reference G/(G-1) multiplier and z-inference per convention decision (1e-6)"""
        ref = self.golden["reference"]["poisson"]
        res = compute_poisson(self.df_poisson, "y", ["x1", "x2"], vce="cluster", cluster_var="cluster")
        for i, name in enumerate(["x1", "x2", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["cluster"]["bse"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["z"], ref["cluster"]["zvalues"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["cluster"]["pvalues"][i], delta=TOL_OPTIM)

    # =========================================================================
    # 5. 2SLS Reference Tests (Algebraic exactness: 1e-8)
    # =========================================================================
    def test_15_2sls_small_sample(self):
        """2SLS small-sample: s^2=e'e/(N-K) and t-inference (1e-8)"""
        ref = self.golden["reference"]["iv"]
        res = compute_ivregress_2sls(self.df_iv, "y", exog_vars=["x1"], endog_vars=["y_endog"],
                                     instruments=["z1", "z2"], small=True, vce="unadjusted")
        for i, name in enumerate(["y_endog", "x1", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["small"]["bse"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["t"], ref["small"]["tvalues"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["small"]["pvalues"][i], delta=TOL_ALGEBRA)

    def test_16_2sls_large_sample(self):
        """2SLS large-sample: s^2=e'e/N and z-inference (Reference default) (1e-8)"""
        ref = self.golden["reference"]["iv"]
        res = compute_ivregress_2sls(self.df_iv, "y", exog_vars=["x1"], endog_vars=["y_endog"],
                                     instruments=["z1", "z2"], small=False, vce="unadjusted")
        for i, name in enumerate(["y_endog", "x1", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["large"]["bse"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["z"], ref["large"]["zvalues"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["large"]["pvalues"][i], delta=TOL_ALGEBRA)

    def test_17_2sls_robust(self):
        """2SLS Robust: sandwich covariance without small-sample factor (1e-8)"""
        ref = self.golden["reference"]["iv"]
        res = compute_ivregress_2sls(self.df_iv, "y", exog_vars=["x1"], endog_vars=["y_endog"],
                                     instruments=["z1", "z2"], small=False, vce="robust")
        for i, name in enumerate(["y_endog", "x1", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["robust"]["bse"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["z"], ref["robust"]["zvalues"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["robust"]["pvalues"][i], delta=TOL_ALGEBRA)

    def test_18_2sls_cluster(self):
        """2SLS Cluster: Reference G/(G-1) cluster sandwich and z-inference (1e-8)"""
        ref = self.golden["reference"]["iv"]
        res = compute_ivregress_2sls(self.df_iv, "y", exog_vars=["x1"], endog_vars=["y_endog"],
                                     instruments=["z1", "z2"], small=False, vce="cluster", cluster_var="cluster")
        for i, name in enumerate(["y_endog", "x1", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["cluster"]["bse"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["z"], ref["cluster"]["zvalues"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["cluster"]["pvalues"][i], delta=TOL_ALGEBRA)

    # =========================================================================
    # 6. xtreg fe and reghdfe Reference Tests (Algebraic exactness: 1e-8)
    # =========================================================================
    def test_19_xtreg_fe_classical(self):
        """xtreg fe Classical: within R2, between R2, overall R2, F for u_i=0 (1e-8)"""
        ref = self.golden["reference"]["panel"]
        res = compute_panel_fe(self.df_panel, "y", ["x1", "x2"], panelvar="entity", vce="classical")
        for i, name in enumerate(["x1", "x2"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["classical"]["bse"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["t"], ref["classical"]["tvalues"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["classical"]["pvalues"][i], delta=TOL_ALGEBRA)
        self.assertAlmostEqual(res["r2_within"], ref["r2_within"], delta=TOL_ALGEBRA)
        self.assertAlmostEqual(res["r2_between"], ref["r2_between"], delta=TOL_ALGEBRA)
        self.assertAlmostEqual(res["r2_overall"], ref["r2_overall"], delta=TOL_ALGEBRA)
        self.assertAlmostEqual(res["f_effects"], ref["f_pooled"], delta=TOL_ALGEBRA)
        self.assertAlmostEqual(res["prob_f_effects"], ref["prob_f_pooled"], delta=TOL_ALGEBRA)

    def test_20_xtreg_fe_cluster(self):
        """xtreg fe Clustered: clustered on entity SEs (1e-8)"""
        ref = self.golden["reference"]["panel"]
        res = compute_panel_fe(self.df_panel, "y", ["x1", "x2"], panelvar="entity", vce="cluster", cluster_var="entity")
        for i, name in enumerate(["x1", "x2"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["cluster"]["bse"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["t"], ref["cluster"]["tvalues"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["cluster"]["pvalues"][i], delta=TOL_ALGEBRA)

    def test_21_reghdfe_1fe_equals_xtreg_fe(self):
        """reghdfe with 1 FE: must algebraically equal xtreg fe for coefs and SEs (1e-8)"""
        res_fe = compute_panel_fe(self.df_panel, "y", ["x1", "x2"], panelvar="entity", vce="classical")
        res_hdfe = compute_reghdfe(self.df_panel, "y", ["x1", "x2"], absorb_vars=["entity"], vce="ols")
        for i in range(2):
            self.assertAlmostEqual(res_hdfe["coefs"][i]["coef"], res_fe["coefs"][i]["coef"], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res_hdfe["coefs"][i]["se"], res_fe["coefs"][i]["se"], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res_hdfe["coefs"][i]["t"], res_fe["coefs"][i]["t"], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res_hdfe["coefs"][i]["p"], res_fe["coefs"][i]["p"], delta=TOL_ALGEBRA)
        self.assertAlmostEqual(res_hdfe["r2_within"], res_fe["r2_within"], delta=TOL_ALGEBRA)

    def test_22_reghdfe_2fe_vs_dummy_ols(self):
        """reghdfe with 2 FEs (entity + time): compare with dummy-variable OLS (1e-8)"""
        ref = self.golden["reference"]["panel"]
        res = compute_reghdfe(self.df_panel, "y", ["x1", "x2"], absorb_vars=["entity", "time"], vce="ols")
        for i in range(2):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["reghdfe_2fe_params"][i], delta=TOL_ALGEBRA)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["reghdfe_2fe_bse"][i], delta=TOL_ALGEBRA)

    # =========================================================================
    # 7. Quantile Regression Reference Tests (Optimizer tolerance: 1e-6)
    # =========================================================================
    def test_23_qreg_vs_quantreg(self):
        """qreg vs statsmodels QuantReg (vcov='iid'): coefs AND standard errors (1e-6)"""
        ref = self.golden["reference"]["qreg"]
        res = compute_qreg(self.df_ols, "y", ["x1", "x2"], quantile=0.5)
        for i, name in enumerate(["x1", "x2", "_cons"]):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["bse"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["t"], ref["tvalues"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["p"], ref["pvalues"][i], delta=TOL_OPTIM)

    # =========================================================================
    # 8. Tobit Reference Tests (Optimizer tolerance: 1e-6)
    # =========================================================================
    def test_24_tobit_vs_numerical_hessian(self):
        """tobit vs numerical-Hessian reference: coefs, SEs, /sigma and its SE (1e-6)"""
        ref = self.golden["reference"]["tobit"]
        res = compute_tobit(self.df_tobit, "y", ["x1", "x2"], ll=0.0)
        # Check x1, x2, _cons
        for i in range(3):
            self.assertAlmostEqual(res["coefs"][i]["coef"], ref["params"][i], delta=TOL_OPTIM)
            self.assertAlmostEqual(res["coefs"][i]["se"], ref["bse"][i], delta=TOL_OPTIM)
        # Check /sigma
        sig_row = [c for c in res["coefs"] if c["name"] == "/sigma"][0]
        self.assertAlmostEqual(sig_row["coef"], ref["sigma"], delta=TOL_OPTIM)
        self.assertAlmostEqual(sig_row["se"], ref["se_sigma"], delta=TOL_OPTIM)
        self.assertGreater(sig_row["p"], 0.0, msg="/sigma p-value must be real, not hardcoded 0.0")

    # =========================================================================
    # 9. ADF Reference Tests (Explicit lags: 1e-8)
    # =========================================================================
    def test_25_adf_explicit_lags(self):
        """ADF with explicit lags (lag=1 and lag=4) vs statsmodels adfuller (1e-8)"""
        ref = self.golden["reference"]["adf"]
        res1 = compute_dfuller(self.ts_series, lags=1)
        self.assertAlmostEqual(res1["stat"], ref["lag_1"]["stat"], delta=TOL_ALGEBRA)
        self.assertAlmostEqual(res1["p_value"], ref["lag_1"]["p_value"], delta=TOL_ALGEBRA)
        self.assertEqual(res1["lags_used"], ref["lag_1"]["lags_used"])

        res4 = compute_dfuller(self.ts_series, lags=4)
        self.assertAlmostEqual(res4["stat"], ref["lag_4"]["stat"], delta=TOL_ALGEBRA)
        self.assertAlmostEqual(res4["p_value"], ref["lag_4"]["p_value"], delta=TOL_ALGEBRA)
        self.assertEqual(res4["lags_used"], ref["lag_4"]["lags_used"])


if __name__ == "__main__":
    unittest.main()
