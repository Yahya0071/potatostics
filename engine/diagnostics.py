"""
engine/diagnostics.py - Regression Diagnostics Suite
Implements Variance Inflation Factors (VIF), Breusch-Pagan / Cook-Weisberg heteroskedasticity test,
Ramsey RESET omitted variable test, and Durbin-Watson autocorrelation test.
"""

import numpy as np
from scipy import stats

def compute_vif(model_result, df):
    indeps = [v for v in model_result['var_names'] if v != '_cons']
    if len(indeps) < 2:
        raise ValueError("VIF requires at least two independent variables.")
        
    data_idx = model_result.get('data_index')
    if data_idx is not None and not df.loc[data_idx].empty:
        data = df.loc[data_idx, indeps].dropna()
    else:
        data = df[indeps].dropna()
    X = data.to_numpy(dtype=float)
    
    vif_rows = []
    # For each regressor j, regress on all other regressors
    for j, var_j in enumerate(indeps):
        y_j = X[:, j]
        other_indices = [idx for idx in range(X.shape[1]) if idx != j]
        X_others = X[:, other_indices]
        X_others_with_cons = np.column_stack([X_others, np.ones(len(y_j))])
        
        try:
            XtX = X_others_with_cons.T @ X_others_with_cons
            beta_j = np.linalg.inv(XtX) @ (X_others_with_cons.T @ y_j)
            y_pred_j = X_others_with_cons @ beta_j
            ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
            ss_res = np.sum((y_j - y_pred_j) ** 2)
            r2_j = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            r2_j = max(0.0, min(0.999999, r2_j))
            vif_j = 1.0 / (1.0 - r2_j)
            tol_j = 1.0 - r2_j
        except np.linalg.LinAlgError:
            vif_j = np.inf
            tol_j = 0.0
            
        vif_rows.append({
            'variable': var_j,
            'vif': float(vif_j),
            'tolerance': float(tol_j)
        })
        
    finite_vifs = [r['vif'] for r in vif_rows if np.isfinite(r['vif'])]
    mean_vif = float(np.mean(finite_vifs)) if finite_vifs else float('inf')
    return {
        'rows': vif_rows,
        'mean_vif': mean_vif
    }

def compute_hettest(model_result, df):
    # Breusch-Pagan / Cook-Weisberg test for heteroskedasticity
    # Regress e_i^2 on y_hat
    resid = model_result.get('resid')
    y_pred = model_result.get('y_pred')
    if resid is None or y_pred is None:
        raise ValueError("Residuals and fitted values are required for hettest.")
        
    n = len(resid)
    sigma2 = np.mean(resid ** 2)
    u = (resid ** 2) / (sigma2 + 1e-15)
    
    # Regress u on y_pred and constant
    Z = np.column_stack([y_pred, np.ones(n)])
    try:
        ZtZ = Z.T @ Z
        gamma = np.linalg.inv(ZtZ) @ (Z.T @ u)
        u_pred = Z @ gamma
        ss_reg = np.sum((u_pred - np.mean(u)) ** 2)
        lm_stat = float(0.5 * ss_reg)
        p_val = float(1.0 - stats.chi2.cdf(lm_stat, 1))
    except Exception as e:
        raise ValueError(f"Hettest computation error: {e}")
        
    return {
        'chi2': lm_stat,
        'df': 1,
        'p_value': p_val,
        'null_hypothesis': "Constant variance (homoskedasticity)"
    }

def compute_ovtest(model_result, df):
    # Ramsey RESET test for omitted variables using y_hat^2, y_hat^3, y_hat^4
    y_pred = model_result.get('y_pred')
    depvar = model_result['depvar']
    indeps = [v for v in model_result['var_names'] if v != '_cons']
    data_idx = model_result['data_index']
    
    data = df.loc[data_idx]
    y = data[depvar].to_numpy(dtype=float)
    X_orig = data[indeps].to_numpy(dtype=float)
    n = len(y)
    k_orig = len(indeps)
    
    # Scale powers of fitted values to prevent overflow
    y_std = (y_pred - np.mean(y_pred)) / (np.std(y_pred) + 1e-15)
    y2 = y_std ** 2
    y3 = y_std ** 3
    y4 = y_std ** 4
    
    # Unrestricted regression: y on X_orig, constant, y2, y3, y4
    X_unrestricted = np.column_stack([X_orig, np.ones(n), y2, y3, y4])
    try:
        XtX_ur = X_unrestricted.T @ X_unrestricted
        beta_ur = np.linalg.inv(XtX_ur) @ (X_unrestricted.T @ y)
        resid_ur = y - X_unrestricted @ beta_ur
        ss_ur = float(np.sum(resid_ur ** 2))
        
        # Restricted regression: original model
        ss_r = model_result['ss_r']
        
        df_num = 3  # three powers added
        df_den = n - k_orig - 1 - 3
        
        if df_den <= 0:
            raise ValueError("Insufficient degrees of freedom for RESET test.")
            
        f_stat = float(((ss_r - ss_ur) / df_num) / (ss_ur / df_den))
        p_val = float(1.0 - stats.f.cdf(f_stat, df_num, df_den))
    except Exception as e:
        raise ValueError(f"OVtest computation error: {e}")
        
    return {
        'f_stat': f_stat,
        'df_num': df_num,
        'df_den': df_den,
        'p_value': p_val,
        'null_hypothesis': "Model has no omitted variables"
    }

def compute_dwatson(model_result):
    resid = model_result.get('resid')
    if resid is None:
        raise ValueError("Residuals are required to compute Durbin-Watson statistic.")
    diff = np.diff(resid)
    d = float(np.sum(diff ** 2) / np.sum(resid ** 2))
    return {
        'dw': d,
        'interpretation': (
            "No autocorrelation" if 1.8 <= d <= 2.2 else
            ("Positive serial correlation" if d < 1.8 else "Negative serial correlation")
        )
    }


def compute_firststage(model_result, df):
    """
    Computes first-stage regressions for 2SLS instrumental variables models.
    Returns tables of first-stage coefficients, partial R-squared, and classical first-stage F.
    """
    if model_result.get('model_type') not in ('ivregress', '2sls'):
        raise ValueError("First-stage diagnostics require a 2SLS IV model.")

    endog_vars = model_result['endog_vars']
    exog_vars = model_result['exog_vars']
    instruments = model_result['instruments']
    data_idx = model_result.get('data_index')

    cols = list(dict.fromkeys(endog_vars + exog_vars + instruments))
    data = df.loc[data_idx, cols].dropna() if data_idx is not None else df[cols].dropna()
    n = len(data)
    q = len(instruments)

    results = []
    vce = model_result.get('vce', 'unadjusted').lower()
    cluster_var = model_result.get('cluster_var')

    for endog in endog_vars:
        y_fs = data[endog].to_numpy(dtype=float)
        Z_excl = data[instruments].to_numpy(dtype=float)
        Z_incl = data[exog_vars].to_numpy(dtype=float) if exog_vars else np.empty((n, 0))

        X_unres = np.column_stack([Z_excl, Z_incl, np.ones(n)])
        X_res = np.column_stack([Z_incl, np.ones(n)])

        beta_unres, _, _, _ = np.linalg.lstsq(X_unres, y_fs, rcond=None)
        beta_res, _, _, _ = np.linalg.lstsq(X_res, y_fs, rcond=None)

        resid_unres = y_fs - X_unres @ beta_unres
        resid_res = y_fs - X_res @ beta_res

        ssr_unres = float(np.sum(resid_unres ** 2))
        ssr_res = float(np.sum(resid_res ** 2))
        tss = float(np.sum((y_fs - np.mean(y_fs)) ** 2))

        df_r = n - X_unres.shape[1]
        inv_XtX = np.linalg.pinv(X_unres.T @ X_unres)

        if vce in ('robust', 'hc1'):
            scores = X_unres * resid_unres[:, None]
            c = n / max(1, df_r)
            vcov_fs = c * (inv_XtX @ (scores.T @ scores) @ inv_XtX)
            se = np.sqrt(np.maximum(0, np.diag(vcov_fs)))
            t_stat = np.where(se > 0, beta_unres / se, 0.0)
            p_vals = 2.0 * (1.0 - stats.norm.cdf(np.abs(t_stat)))
            try:
                beta_excl = beta_unres[:q]
                vcov_excl = vcov_fs[:q, :q]
                f_stat = float((beta_excl.T @ np.linalg.inv(vcov_excl) @ beta_excl) / q)
                p_val = float(1.0 - stats.f.cdf(f_stat, q, df_r))
            except Exception:
                f_stat = float(((ssr_res - ssr_unres) / q) / (ssr_unres / df_r)) if df_r > 0 and ssr_unres > 0 else 0.0
                p_val = float(1.0 - stats.f.cdf(f_stat, q, df_r)) if df_r > 0 else 1.0

        elif vce == 'cluster' and cluster_var and cluster_var in data.columns:
            clusters = data[cluster_var].to_numpy()
            unq_clust = np.unique(clusters)
            G = len(unq_clust)
            meat = np.zeros((X_unres.shape[1], X_unres.shape[1]), dtype=float)
            for cl in unq_clust:
                idx_c = (clusters == cl)
                X_c = X_unres[idx_c, :]
                e_c = resid_unres[idx_c]
                s_c = X_c.T @ e_c
                meat += np.outer(s_c, s_c)
            q_factor = (G / max(1, G - 1)) * ((n - 1) / max(1, df_r))
            vcov_fs = q_factor * (inv_XtX @ meat @ inv_XtX)
            se = np.sqrt(np.maximum(0, np.diag(vcov_fs)))
            t_stat = np.where(se > 0, beta_unres / se, 0.0)
            p_vals = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), max(1, G - 1)))
            try:
                beta_excl = beta_unres[:q]
                vcov_excl = vcov_fs[:q, :q]
                f_stat = float((beta_excl.T @ np.linalg.inv(vcov_excl) @ beta_excl) / q)
                p_val = float(1.0 - stats.f.cdf(f_stat, q, max(1, G - 1)))
            except Exception:
                f_stat = float(((ssr_res - ssr_unres) / q) / (ssr_unres / df_r)) if df_r > 0 and ssr_unres > 0 else 0.0
                p_val = float(1.0 - stats.f.cdf(f_stat, q, max(1, G - 1))) if df_r > 0 else 1.0

        else:
            s2 = ssr_unres / max(1, df_r)
            vcov_fs = s2 * inv_XtX
            se = np.sqrt(np.maximum(0, np.diag(vcov_fs)))
            t_stat = np.where(se > 0, beta_unres / se, 0.0)
            p_vals = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df_r))
            f_stat = float(((ssr_res - ssr_unres) / q) / (ssr_unres / df_r)) if df_r > 0 and ssr_unres > 0 else 0.0
            p_val = float(1.0 - stats.f.cdf(f_stat, q, df_r)) if df_r > 0 else 1.0

        r2 = 1.0 - (ssr_unres / tss) if tss > 0 else 0.0
        partial_r2 = (ssr_res - ssr_unres) / ssr_res if ssr_res > 0 else 0.0

        names = instruments + exog_vars + ['_cons']
        coef_rows = []
        for i, name in enumerate(names):
            coef_rows.append({
                'name': name,
                'coef': float(beta_unres[i]),
                'se': float(se[i]),
                't': float(t_stat[i]),
                'p': float(p_vals[i])
            })

        results.append({
            'target': endog,
            'f_stat': f_stat,
            'df_num': q,
            'df_den': df_r,
            'p_value': p_val,
            'r2': r2,
            'partial_r2': partial_r2,
            'coefs': coef_rows
        })

    return results


def compute_endogenous(model_result, df):
    """
    Wu-Hausman and Durbin tests for endogeneity of regressors in 2SLS.
    """
    if model_result.get('model_type') not in ('ivregress', '2sls'):
        raise ValueError("Endogeneity test requires a 2SLS IV model.")

    endog_vars = model_result['endog_vars']
    exog_vars = model_result['exog_vars']
    instruments = model_result['instruments']
    depvar = model_result['depvar']
    data_idx = model_result.get('data_index')

    cols = list(dict.fromkeys([depvar] + endog_vars + exog_vars + instruments))
    data = df.loc[data_idx, cols].dropna() if data_idx is not None else df[cols].dropna()
    n = len(data)
    m = len(endog_vars)

    y = data[depvar].to_numpy(dtype=float)
    X_endog = data[endog_vars].to_numpy(dtype=float)
    X_exog = data[exog_vars].to_numpy(dtype=float) if exog_vars else np.empty((n, 0))
    Z_inst = data[instruments].to_numpy(dtype=float)

    # First stage residuals for each endogenous variable
    Z_all = np.column_stack([Z_inst, X_exog, np.ones(n)])
    inv_ZtZ = np.linalg.inv(Z_all.T @ Z_all)
    P_Z = Z_all @ inv_ZtZ @ Z_all.T
    
    V_hat = X_endog - P_Z @ X_endog

    # Second stage with V_hat included
    X_unres = np.column_stack([X_endog, X_exog, np.ones(n), V_hat])
    X_res = np.column_stack([X_endog, X_exog, np.ones(n)])

    b_unres, _, _, _ = np.linalg.lstsq(X_unres, y, rcond=None)
    b_res, _, _, _ = np.linalg.lstsq(X_res, y, rcond=None)

    ssr_unres = float(np.sum((y - X_unres @ b_unres) ** 2))
    ssr_res = float(np.sum((y - X_res @ b_res) ** 2))

    df_den = n - X_unres.shape[1]
    wu_hausman_f = float(((ssr_res - ssr_unres) / m) / (ssr_unres / df_den))
    wu_p = float(1.0 - stats.f.cdf(wu_hausman_f, m, df_den))

    durbin_chi2 = float(m * wu_hausman_f)
    durbin_p = float(1.0 - stats.chi2.cdf(durbin_chi2, m))

    return {
        'wu_hausman_f': wu_hausman_f,
        'df_num': m,
        'df_den': df_den,
        'wu_p': wu_p,
        'durbin_chi2': durbin_chi2,
        'durbin_df': m,
        'durbin_p': durbin_p,
        'null_hypothesis': "Variables are exogenous"
    }


def compute_overid(model_result, df):
    """
    Sargan and Basmann tests of overidentifying restrictions in 2SLS.
    """
    if model_result.get('model_type') not in ('ivregress', '2sls'):
        raise ValueError("Overidentification test requires a 2SLS IV model.")

    endog_vars = model_result['endog_vars']
    exog_vars = model_result['exog_vars']
    instruments = model_result['instruments']
    data_idx = model_result.get('data_index')
    resid = model_result['resid']

    L = len(instruments)
    m = len(endog_vars)
    df_overid = L - m

    if df_overid <= 0:
        return {
            'applicable': False,
            'df': 0,
            'msg': "Model is exactly identified (or underidentified); no overidentifying restrictions."
        }

    cols = list(dict.fromkeys(instruments + exog_vars))
    data = df.loc[data_idx, cols].dropna() if data_idx is not None else df[cols].dropna()
    n = len(data)

    Z_inst = data[instruments].to_numpy(dtype=float)
    X_exog = data[exog_vars].to_numpy(dtype=float) if exog_vars else np.empty((n, 0))
    Z_all = np.column_stack([Z_inst, X_exog, np.ones(n)])

    # Regress 2SLS residuals on all instruments
    b_z, _, _, _ = np.linalg.lstsq(Z_all, resid, rcond=None)
    resid_fit = Z_all @ b_z
    ssr_fit = float(np.sum((resid - resid_fit) ** 2))
    tss_resid = float(np.sum(resid ** 2))

    r2_e = 1.0 - (ssr_fit / tss_resid) if tss_resid > 0 else 0.0
    sargan_chi2 = float(n * r2_e)
    sargan_p = float(1.0 - stats.chi2.cdf(sargan_chi2, df_overid))

    df_den = n - Z_all.shape[1]
    basmann_f = float(((tss_resid - ssr_fit) / df_overid) / (ssr_fit / df_den)) if df_den > 0 and ssr_fit > 0 else 0.0
    basmann_p = float(1.0 - stats.f.cdf(basmann_f, df_overid, df_den)) if df_den > 0 else 1.0

    return {
        'applicable': True,
        'sargan_chi2': sargan_chi2,
        'df': df_overid,
        'sargan_p': sargan_p,
        'basmann_f': basmann_f,
        'df_den': df_den,
        'basmann_p': basmann_p,
        'null_hypothesis': "Overidentifying restrictions are valid (instruments are uncorrelated with error)"
    }


def compute_brant_test(model_result, df):
    """
    Brant (1990) Wald test for the parallel regression / proportional odds assumption
    after ordered logistic regression (ologit).
    """
    if model_result.get('model_type') != 'ologit':
        raise ValueError("estat brant can only be calculated after ologit")

    depvar = model_result['depvar']
    indepvars = model_result['indepvars']
    k = len(indepvars)

    cols = [depvar] + indepvars
    data = df[cols].dropna().copy()
    n = len(data)
    if n == 0:
        raise ValueError("no observations in estimation sample")

    y = data[depvar].to_numpy()
    unq = np.sort(np.unique(y))
    J = len(unq)
    if J <= 2:
        raise ValueError("Brant test requires an ordered outcome with at least 3 categories")

    import statsmodels.api as sm
    X_raw = data[indepvars].to_numpy(dtype=float)
    X_mat = sm.add_constant(X_raw)

    betas = []
    W_list = []
    inv_XWX = []
    binary_models = []

    for j in range(J - 1):
        z = (y > unq[j]).astype(float)
        if len(np.unique(z)) < 2:
            continue
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logit_mod = sm.Logit(z, X_mat).fit(disp=False, maxiter=100)
                b_j = logit_mod.params[1:]
                p_j = np.clip(logit_mod.predict(), 1e-12, 1.0 - 1e-12)
                b_all = logit_mod.params
                se_all = logit_mod.bse
        except Exception:
            b_ols = np.linalg.pinv(X_mat) @ z
            b_j = b_ols[1:]
            p_j = np.clip(X_mat @ b_ols, 1e-12, 1.0 - 1e-12)
            b_all = b_ols
            se_all = np.zeros_like(b_ols)

        betas.append(b_j)
        W_list.append(p_j)
        w_j = p_j * (1.0 - p_j)
        xw_x = X_mat.T @ (w_j[:, None] * X_mat)
        inv_XWX.append(np.linalg.pinv(xw_x))

        cut_label = f"{depvar} > {unq[j]}"
        binary_models.append({
            'cutoff': cut_label,
            'b': {vname: float(b_j[i]) for i, vname in enumerate(indepvars)},
            'b_cons': float(b_all[0]),
            'se': {vname: float(se_all[i + 1]) for i, vname in enumerate(indepvars)},
            'se_cons': float(se_all[0])
        })

    num_cuts = len(betas)
    if num_cuts < 2:
        raise ValueError("Insufficient non-separated categories for Brant test.")

    V_full = np.zeros((num_cuts * (k + 1), num_cuts * (k + 1)))
    for j in range(num_cuts):
        for l in range(num_cuts):
            j_min, l_max = min(j, l), max(j, l)
            w_jl = W_list[l_max] * (1.0 - W_list[j_min])
            middle = X_mat.T @ (w_jl[:, None] * X_mat)
            v_block = inv_XWX[j] @ middle @ inv_XWX[l]
            V_full[j * (k + 1):(j + 1) * (k + 1), l * (k + 1):(l + 1) * (k + 1)] = v_block

    idx_slopes = []
    for j in range(num_cuts):
        idx_slopes.extend([j * (k + 1) + m for m in range(1, k + 1)])

    beta_star = np.concatenate(betas)
    V_star = V_full[np.ix_(idx_slopes, idx_slopes)]

    D = np.zeros(((num_cuts - 1) * k, num_cuts * k))
    for m in range(num_cuts - 1):
        D[m * k:(m + 1) * k, :k] = -np.eye(k)
        D[m * k:(m + 1) * k, (m + 1) * k:(m + 2) * k] = np.eye(k)

    d = D @ beta_star
    V_d = D @ V_star @ D.T
    chi2_all = float(max(0.0, d.T @ np.linalg.pinv(V_d) @ d))
    df_all = (num_cuts - 1) * k
    p_all = float(1.0 - stats.chi2.cdf(chi2_all, df_all))

    rows = [{
        'variable': 'All',
        'chi2': chi2_all,
        'p': p_all,
        'df': df_all
    }]

    for var_idx, vname in enumerate(indepvars):
        b_var = np.array([betas[j][var_idx] for j in range(num_cuts)])
        v_var = np.zeros((num_cuts, num_cuts))
        for j in range(num_cuts):
            for l in range(num_cuts):
                v_var[j, l] = V_star[j * k + var_idx, l * k + var_idx]
        D_var = np.zeros((num_cuts - 1, num_cuts))
        for m in range(num_cuts - 1):
            D_var[m, 0] = -1.0
            D_var[m, m + 1] = 1.0
        d_v = D_var @ b_var
        v_dv = D_var @ v_var @ D_var.T
        chi2_v = float(max(0.0, d_v.T @ np.linalg.pinv(v_dv) @ d_v))
        df_v = num_cuts - 1
        p_v = float(1.0 - stats.chi2.cdf(chi2_v, df_v))
        rows.append({
            'variable': vname,
            'chi2': chi2_v,
            'p': p_v,
            'df': df_v
        })

    return {
        'chi2': chi2_all,
        'p': p_all,
        'df': df_all,
        'rows': rows,
        'depvar': depvar,
        'indepvars': indepvars,
        'binary_models': binary_models
    }

