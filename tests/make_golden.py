"""
tests/make_golden.py - Golden Reference Generator for Potatostics
Generates synthetic datasets and exact econometric reference values
using statsmodels, linearmodels, and documented Benchmark formulas.
Commits output to tests/golden_reference.json so runtime tests do not
require linearmodels or slow optimizer re-computations.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_reference.json")

def generate_golden():
    np.random.seed(42)
    n = 600

    # -------------------------------------------------------------
    # 1. Dataset for OLS (N=600, K=2 regressors, G=20 clusters)
    # -------------------------------------------------------------
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    g_ols = (np.arange(n) % 20).astype(int)
    u_g = np.random.randn(20) * 0.8
    e = u_g[g_ols] + np.random.randn(n) * 0.7
    y_ols = 2.0 + 1.5 * x1 - 0.8 * x2 + e
    df_ols = pd.DataFrame({"y": y_ols, "x1": x1, "x2": x2, "cluster": g_ols})

    X_ols = sm.add_constant(df_ols[["x1", "x2"]])
    m_ols_class = sm.OLS(df_ols["y"], X_ols).fit()
    m_ols_hc1 = sm.OLS(df_ols["y"], X_ols).fit(cov_type="HC1")
    m_ols_hc2 = sm.OLS(df_ols["y"], X_ols).fit(cov_type="HC2")
    m_ols_hc3 = sm.OLS(df_ols["y"], X_ols).fit(cov_type="HC3")
    m_ols_clust = sm.OLS(df_ols["y"], X_ols).fit(
        cov_type="cluster", cov_kwds={"groups": df_ols["cluster"], "use_correction": True}
    )

    def extract_ols_res(m):
        names = ["x1", "x2", "const"]
        return {
            "params": [float(m.params[k]) for k in names],
            "bse": [float(m.bse[k]) for k in names],
            "tvalues": [float(m.tvalues[k]) for k in names],
            "pvalues": [float(m.pvalues[k]) for k in names],
            "r2": float(m.rsquared),
            "adj_r2": float(m.rsquared_adj),
            "fvalue": float(m.fvalue) if m.fvalue is not None else 0.0,
            "f_pvalue": float(m.f_pvalue) if m.f_pvalue is not None else 1.0,
        }

    ols_ref = {
        "classical": extract_ols_res(m_ols_class),
        "hc1": extract_ols_res(m_ols_hc1),
        "hc2": extract_ols_res(m_ols_hc2),
        "hc3": extract_ols_res(m_ols_hc3),
        "cluster": extract_ols_res(m_ols_clust),
    }

    # -------------------------------------------------------------
    # 2. Dataset for Binary Choice (Logit / Probit)
    # -------------------------------------------------------------
    g_bin = (np.arange(n) % 20).astype(int)
    z_bin = 0.3 + 1.2 * x1 - 0.7 * x2
    p_logit = 1.0 / (1.0 + np.exp(-z_bin))
    y_logit = (np.random.rand(n) < p_logit).astype(int)

    p_probit = stats.norm.cdf(z_bin)
    y_probit = (np.random.rand(n) < p_probit).astype(int)

    df_logit = pd.DataFrame({"y": y_logit, "x1": x1, "x2": x2, "cluster": g_bin})
    df_probit = pd.DataFrame({"y": y_probit, "x1": x1, "x2": x2, "cluster": g_bin})

    # Fit Logit and Probit using Newton-Raphson MLE in statsmodels
    sm_logit = sm.Logit(df_logit["y"], X_ols).fit(disp=False, method="newton", tol=1e-12)
    sm_probit = sm.Probit(df_probit["y"], X_ols).fit(disp=False, method="newton", tol=1e-12)

    def extract_mle_res(sm_m, df_data, model_type="logit"):
        names = ["x1", "x2", "const"]
        beta = np.array([sm_m.params[k] for k in names])
        # Column stack [x1, x2, 1] so columns strictly match names: x1, x2, _cons
        X = np.column_stack([df_data[["x1", "x2"]].to_numpy(dtype=float), np.ones(len(df_data))])
        y = df_data["y"].to_numpy(dtype=float)
        n_obs = len(y)
        k_params = len(names)

        if model_type == "logit":
            eta = X @ beta
            p = 1.0 / (1.0 + np.exp(-eta))
            scores = X * (y - p)[:, None]
            w = p * (1.0 - p)
            H = -(X.T * w) @ X
        elif model_type == "probit":
            eta = X @ beta
            phi = stats.norm.pdf(eta)
            Phi = np.clip(stats.norm.cdf(eta), 1e-15, 1.0 - 1e-15)
            lambda_i = np.where(y == 1, phi / Phi, -phi / (1.0 - Phi))
            scores = X * lambda_i[:, None]
            w = lambda_i * (lambda_i + eta)
            H = -(X.T * w) @ X
        elif model_type == "poisson":
            eta = X @ beta
            mu = np.exp(eta)
            scores = X * (y - mu)[:, None]
            H = -(X.T * mu) @ X

        inv_H = np.linalg.inv(-H)

        # 1. Classical MLE
        vcov_mle = inv_H
        se_mle = np.sqrt(np.diag(vcov_mle))
        z_mle = beta / se_mle
        p_mle = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_mle)))

        # 2. Robust (Benchmark formula: N/(N-1) * inv_H * (scores.T @ scores) * inv_H)
        omega = scores.T @ scores
        c_rob = n_obs / (n_obs - 1.0)
        vcov_rob = c_rob * (inv_H @ omega @ inv_H)
        se_rob = np.sqrt(np.diag(vcov_rob))
        z_rob = beta / se_rob
        p_rob = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_rob)))

        # 3. Clustered (Benchmark formula resolved in decision A1: G/(G-1) * inv_H * (sum s_g s_g') * inv_H, z-inference)
        clusters = df_data["cluster"].to_numpy()
        unq = np.unique(clusters)
        G = len(unq)
        M = np.zeros((k_params, k_params))
        for cl in unq:
            s_g = np.sum(scores[clusters == cl], axis=0)
            M += np.outer(s_g, s_g)
        c_clust = G / (G - 1.0)
        vcov_clust = c_clust * (inv_H @ M @ inv_H)
        se_clust = np.sqrt(np.diag(vcov_clust))
        z_clust = beta / se_clust
        p_clust = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_clust)))

        return {
            "params": [float(b) for b in beta],
            "ll": float(sm_m.llf),
            "lr_chi2": float(sm_m.llr),
            "pseudo_r2": float(sm_m.prsquared),
            "mle": {
                "bse": [float(s) for s in se_mle],
                "zvalues": [float(z) for z in z_mle],
                "pvalues": [float(p) for p in p_mle],
            },
            "robust": {
                "bse": [float(s) for s in se_rob],
                "zvalues": [float(z) for z in z_rob],
                "pvalues": [float(p) for p in p_rob],
            },
            "cluster": {
                "bse": [float(s) for s in se_clust],
                "zvalues": [float(z) for z in z_clust],
                "pvalues": [float(p) for p in p_clust],
            },
        }

    logit_ref = extract_mle_res(sm_logit, df_logit, "logit")
    probit_ref = extract_mle_res(sm_probit, df_probit, "probit")

    # -------------------------------------------------------------
    # 3. Dataset for Poisson Count Regression
    # -------------------------------------------------------------
    z_pois = 0.5 + 0.4 * x1 - 0.3 * x2
    mu_pois = np.exp(z_pois)
    y_pois = np.random.poisson(mu_pois)
    df_poisson = pd.DataFrame({"y": y_pois, "x1": x1, "x2": x2, "cluster": g_bin})
    sm_poisson = sm.Poisson(df_poisson["y"], X_ols).fit(disp=False, method="newton", tol=1e-12)
    poisson_ref = extract_mle_res(sm_poisson, df_poisson, "poisson")

    # -------------------------------------------------------------
    # 4. Dataset for 2SLS (IV)
    # -------------------------------------------------------------
    z1 = np.random.randn(n)
    z2 = np.random.randn(n)
    u_iv = np.random.randn(n)
    v_iv = 0.6 * u_iv + np.random.randn(n) * 0.4
    y_endog = 0.5 * z1 + 0.8 * z2 + 0.4 * x1 + u_iv
    y_iv = 1.0 + 1.8 * y_endog - 0.9 * x1 + v_iv
    df_iv = pd.DataFrame({
        "y": y_iv, "y_endog": y_endog, "x1": x1, "z1": z1, "z2": z2, "cluster": g_ols, "const": 1.0
    })

    names_iv = ["y_endog", "x1", "const"]
    # Linearmodels IV2SLS reference
    # Small-sample (Reference 'small'): s^2 = e'e / (n-k), t-inference
    lm_iv_small = IV2SLS(df_iv["y"], df_iv[["const", "x1"]], df_iv["y_endog"], df_iv[["z1", "z2"]]).fit(
        cov_type="unadjusted", debiased=True
    )
    # Large-sample (Reference default): s^2 = e'e / n, z-inference
    lm_iv_large = IV2SLS(df_iv["y"], df_iv[["const", "x1"]], df_iv["y_endog"], df_iv[["z1", "z2"]]).fit(
        cov_type="unadjusted", debiased=False
    )
    # Robust (Reference default robust without small: debiased=False)
    lm_iv_rob = IV2SLS(df_iv["y"], df_iv[["const", "x1"]], df_iv["y_endog"], df_iv[["z1", "z2"]]).fit(
        cov_type="robust", debiased=False
    )

    # Cluster: Benchmark formula G/(G-1) * inv_XtX * (sum s_g s_g') * inv_XtX, z-inference
    X_iv_mat = df_iv[["y_endog", "x1", "const"]].to_numpy(dtype=float)
    Z_iv_mat = df_iv[["z1", "z2", "x1", "const"]].to_numpy(dtype=float)
    y_iv_vec = df_iv["y"].to_numpy(dtype=float)
    P_Z = Z_iv_mat @ np.linalg.inv(Z_iv_mat.T @ Z_iv_mat) @ Z_iv_mat.T
    X_hat = P_Z @ X_iv_mat
    bread_iv = np.linalg.inv(X_hat.T @ X_hat)
    beta_iv = bread_iv @ (X_hat.T @ y_iv_vec)
    resid_iv = y_iv_vec - X_iv_mat @ beta_iv

    unq_iv_clust = np.unique(g_ols)
    G_iv = len(unq_iv_clust)
    meat_iv = np.zeros((3, 3))
    for cl in unq_iv_clust:
        idx = (g_ols == cl)
        s_g = X_hat[idx].T @ resid_iv[idx]
        meat_iv += np.outer(s_g, s_g)
    vcov_iv_clust = (G_iv / (G_iv - 1.0)) * (bread_iv @ meat_iv @ bread_iv)
    se_iv_clust = np.sqrt(np.diag(vcov_iv_clust))
    z_iv_clust = beta_iv / se_iv_clust
    p_iv_clust = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_iv_clust)))

    iv_ref = {
        "params": [float(lm_iv_large.params[k]) for k in names_iv],
        "small": {
            "bse": [float(lm_iv_small.std_errors[k]) for k in names_iv],
            "tvalues": [float(lm_iv_small.tstats[k]) for k in names_iv],
            "pvalues": [float(lm_iv_small.pvalues[k]) for k in names_iv],
        },
        "large": {
            "bse": [float(lm_iv_large.std_errors[k]) for k in names_iv],
            "zvalues": [float(lm_iv_large.tstats[k]) for k in names_iv],
            "pvalues": [float(lm_iv_large.pvalues[k]) for k in names_iv],
        },
        "robust": {
            "bse": [float(lm_iv_rob.std_errors[k]) for k in names_iv],
            "zvalues": [float(lm_iv_rob.tstats[k]) for k in names_iv],
            "pvalues": [float(lm_iv_rob.pvalues[k]) for k in names_iv],
        },
        "cluster": {
            "bse": [float(s) for s in se_iv_clust],
            "zvalues": [float(z) for z in z_iv_clust],
            "pvalues": [float(p) for p in p_iv_clust],
        },
    }

    # -------------------------------------------------------------
    # 5. Dataset for Panel FE & reghdfe (N=600, entities=40, T=15)
    # -------------------------------------------------------------
    n_entities = 40
    n_periods = 15
    total_panel = n_entities * n_periods
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(n_periods), n_entities)
    u_entity = np.repeat(np.random.randn(n_entities) * 1.5, n_periods)
    px1 = np.random.randn(total_panel)
    px2 = np.random.randn(total_panel)
    py = 2.5 + 1.2 * px1 - 0.5 * px2 + u_entity + np.random.randn(total_panel)
    df_panel = pd.DataFrame({"entity": entity, "time": time, "y": py, "x1": px1, "x2": px2})

    df_p = df_panel.set_index(["entity", "time"])
    lm_fe_class = PanelOLS(df_p["y"], df_p[["x1", "x2"]], entity_effects=True).fit(cov_type="unadjusted")

    # 2 FE Dummy OLS reference for reghdfe with 2 FEs (entity + time)
    df_dummies = pd.get_dummies(df_panel[["entity", "time"]].astype(str), drop_first=True, dtype=float)
    X_2fe = pd.concat([df_panel[["x1", "x2"]], df_dummies], axis=1)
    X_2fe_sm = sm.add_constant(X_2fe)
    m_2fe_sm = sm.OLS(df_panel["y"], X_2fe_sm).fit()

    # Reference [XT] xtreg fe definitions:
    # 1. R2 between = Corr(y_means, X_means @ beta)^2
    # 2. R2 overall = Corr(y, X @ beta)^2
    # 3. Clustered SE on panelvar: c = (G/(G-1)) * ((N-1)/(N - n_entities - K)), t-inference with G-1 df
    grouped_p = df_panel.groupby("entity")
    y_means = grouped_p["y"].mean().to_numpy()
    X_means = np.column_stack([grouped_p["x1"].mean().to_numpy(), grouped_p["x2"].mean().to_numpy()])
    beta_panel = np.array([lm_fe_class.params["x1"], lm_fe_class.params["x2"]])
    r2_between_ref = float(np.corrcoef(y_means, X_means @ beta_panel)[0, 1] ** 2)
    r2_overall_ref = float(np.corrcoef(df_panel["y"].to_numpy(), df_panel[["x1", "x2"]].to_numpy() @ beta_panel)[0, 1] ** 2)

    ydm = (df_panel["y"] - grouped_p["y"].transform("mean")).to_numpy()
    Xdm = np.column_stack([(df_panel["x1"] - grouped_p["x1"].transform("mean")).to_numpy(),
                           (df_panel["x2"] - grouped_p["x2"].transform("mean")).to_numpy()])
    res_fe = ydm - Xdm @ beta_panel
    bread_fe = np.linalg.inv(Xdm.T @ Xdm)
    meat_fe = np.zeros((2, 2))
    for g in range(n_entities):
        idx = (df_panel["entity"] == g)
        sg = Xdm[idx].T @ res_fe[idx]
        meat_fe += np.outer(sg, sg)
    c_fe_clust = (n_entities / (n_entities - 1.0)) * ((total_panel - 1.0) / (total_panel - n_entities - 2.0))
    vcov_fe_clust = c_fe_clust * (bread_fe @ meat_fe @ bread_fe)
    se_fe_clust = np.sqrt(np.diag(vcov_fe_clust))
    t_fe_clust = beta_panel / se_fe_clust
    p_fe_clust = 2.0 * (1.0 - stats.t.cdf(np.abs(t_fe_clust), n_entities - 1))

    panel_ref = {
        "params": [float(lm_fe_class.params[k]) for k in ["x1", "x2"]],
        "classical": {
            "bse": [float(lm_fe_class.std_errors[k]) for k in ["x1", "x2"]],
            "tvalues": [float(lm_fe_class.tstats[k]) for k in ["x1", "x2"]],
            "pvalues": [float(lm_fe_class.pvalues[k]) for k in ["x1", "x2"]],
        },
        "cluster": {
            "bse": [float(s) for s in se_fe_clust],
            "tvalues": [float(t) for t in t_fe_clust],
            "pvalues": [float(p) for p in p_fe_clust],
        },
        "r2_within": float(lm_fe_class.rsquared_within),
        "r2_between": r2_between_ref,
        "r2_overall": r2_overall_ref,
        "f_pooled": float(lm_fe_class.f_pooled.stat),
        "prob_f_pooled": float(lm_fe_class.f_pooled.pval),
        "reghdfe_2fe_params": [float(m_2fe_sm.params[k]) for k in ["x1", "x2"]],
        "reghdfe_2fe_bse": [float(m_2fe_sm.bse[k]) for k in ["x1", "x2"]],
    }

    # -------------------------------------------------------------
    # 6. Quantile Regression (qreg q=0.5) vs statsmodels QuantReg
    # -------------------------------------------------------------
    # Use high-precision IRLS fit (p_tol=1e-12, max_iter=2000)
    sm_qreg = sm.QuantReg(df_ols["y"], sm.add_constant(df_ols[["x1", "x2"]])).fit(
        q=0.5, vcov="iid", p_tol=1e-12, max_iter=2000
    )
    names_qreg = ["x1", "x2", "const"]
    qreg_ref = {
        "params": [float(sm_qreg.params[k]) for k in names_qreg],
        "bse": [float(sm_qreg.bse[k]) for k in names_qreg],
        "tvalues": [float(sm_qreg.tvalues[k]) for k in names_qreg],
        "pvalues": [float(sm_qreg.pvalues[k]) for k in names_qreg],
    }

    # -------------------------------------------------------------
    # 7. Tobit Censored Regression vs Numerical Hessian Reference
    # -------------------------------------------------------------
    y_latent = 0.8 + 1.2 * x1 - 0.6 * x2 + np.random.randn(n) * 1.2
    y_tobit = np.maximum(0.0, y_latent)
    df_tobit = pd.DataFrame({"y": y_tobit, "x1": x1, "x2": x2})

    X_tob_mat = np.column_stack([df_tobit[["x1", "x2"]].to_numpy(dtype=float), np.ones(n)])
    y_tob_vec = df_tobit["y"].to_numpy(dtype=float)
    is_left = (y_tob_vec <= 0.0)
    is_unc = ~is_left

    def tobit_nll(theta):
        b = theta[:-1]
        sigma = np.exp(theta[-1])
        xb = X_tob_mat @ b
        val = 0.0
        z_u = (y_tob_vec[is_unc] - xb[is_unc]) / sigma
        val += np.sum(np.log(sigma) + 0.5 * np.log(2.0 * np.pi) + 0.5 * (z_u ** 2))
        z_l = (0.0 - xb[is_left]) / sigma
        Phi_l = np.clip(stats.norm.cdf(z_l), 1e-15, 1.0 - 1e-15)
        val -= np.sum(np.log(Phi_l))
        return val

    init_b = np.linalg.pinv(X_tob_mat.T @ X_tob_mat) @ (X_tob_mat.T @ y_tob_vec)
    init_sig = np.log(max(0.1, np.std(y_tob_vec)))
    theta_0 = np.append(init_b, init_sig)
    opt = minimize(tobit_nll, theta_0, method="L-BFGS-B", tol=1e-12)
    theta_opt = opt.x
    b_opt = theta_opt[:-1]
    sig_opt = np.exp(theta_opt[-1])

    # Central difference Hessian for theta = [b(x1), b(x2), b(_cons), log_sigma]
    p_len = len(theta_opt)
    hess = np.zeros((p_len, p_len))
    eps = 1e-5
    for i in range(p_len):
        for j in range(p_len):
            ei = np.zeros(p_len)
            ej = np.zeros(p_len)
            ei[i] = eps
            ej[j] = eps
            fpp = tobit_nll(theta_opt + ei + ej)
            fpm = tobit_nll(theta_opt + ei - ej)
            fmp = tobit_nll(theta_opt - ei + ej)
            fmm = tobit_nll(theta_opt - ei - ej)
            hess[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)

    vcov_theta = np.linalg.inv(hess)
    vcov_beta = vcov_theta[:-1, :-1]
    se_beta = np.sqrt(np.diag(vcov_beta))
    var_log_sig = vcov_theta[-1, -1]
    se_sigma = float(np.sqrt(max(0.0, var_log_sig)) * sig_opt)

    # Coefs order: x1 (col 0), x2 (col 1), _cons (col 2)
    tobit_params = [float(b_opt[0]), float(b_opt[1]), float(b_opt[2])]
    tobit_se = [float(se_beta[0]), float(se_beta[1]), float(se_beta[2])]

    tobit_ref = {
        "params": tobit_params,
        "bse": tobit_se,
        "sigma": float(sig_opt),
        "se_sigma": se_sigma,
        "ll": float(-opt.fun),
    }

    # -------------------------------------------------------------
    # 8. Time Series ADF (Augmented Dickey-Fuller)
    # -------------------------------------------------------------
    ts_walk = np.cumsum(np.random.randn(n))
    res_adf_1 = adfuller(ts_walk, maxlag=1, autolag=None, regression="c")
    res_adf_4 = adfuller(ts_walk, maxlag=4, autolag=None, regression="c")

    adf_ref = {
        "lag_1": {
            "stat": float(res_adf_1[0]),
            "p_value": float(res_adf_1[1]),
            "lags_used": int(res_adf_1[2]),
            "n_obs": int(res_adf_1[3]),
        },
        "lag_4": {
            "stat": float(res_adf_4[0]),
            "p_value": float(res_adf_4[1]),
            "lags_used": int(res_adf_4[2]),
            "n_obs": int(res_adf_4[3]),
        },
    }

    # -------------------------------------------------------------
    # Bundle into full golden dictionary
    # -------------------------------------------------------------
    golden = {
        "datasets": {
            "df_ols": df_ols.to_dict(orient="list"),
            "df_logit": df_logit.to_dict(orient="list"),
            "df_probit": df_probit.to_dict(orient="list"),
            "df_poisson": df_poisson.to_dict(orient="list"),
            "df_iv": df_iv.to_dict(orient="list"),
            "df_panel": df_panel.to_dict(orient="list"),
            "df_tobit": df_tobit.to_dict(orient="list"),
            "ts_series": ts_walk.tolist(),
        },
        "reference": {
            "ols": ols_ref,
            "logit": logit_ref,
            "probit": probit_ref,
            "poisson": poisson_ref,
            "iv": iv_ref,
            "panel": panel_ref,
            "qreg": qreg_ref,
            "tobit": tobit_ref,
            "adf": adf_ref,
        },
    }

    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2)

    print(f"Golden reference regenerated successfully at {GOLDEN_PATH}")

if __name__ == "__main__":
    generate_golden()
