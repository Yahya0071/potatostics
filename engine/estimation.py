"""
engine/statistics.py - Numerical and Econometric Computation Engine
Implements OLS, Robust HC1/HC2/HC3, Clustered Standard Errors,
Logistic Regression (Newton-Raphson MLE with robust & cluster VCE),
Two-Stage Least Squares (2SLS with large/small sample and cluster VCE),
Panel Fixed Effects (with cluster VCE), Wald tests, and predictions.
"""

import re
import numpy as np
import pandas as pd
from scipy import linalg, stats
import scipy.linalg

def compute_summarize(df, varnames=None, detail=False):
    if not varnames:
        varnames = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    results = []
    for var in varnames:
        if var not in df.columns:
            continue
        s = df[var].dropna()
        n = len(s)
        if n == 0:
            entry = {'var': var, 'obs': 0, 'sum': 0.0}
            entry.update(dict.fromkeys(('mean', 'std', 'min', 'max', 'Var'), np.nan))
            if detail:
                undefined = ('p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'skewness', 'kurtosis')
                entry.update(dict.fromkeys(undefined, np.nan))
                entry.update(sum_w=0.0, smallest=[np.nan]*4, largest=[np.nan]*4)
            results.append(entry)
        else:
            mean_val = float(s.mean())
            std_val = float(s.std(ddof=1)) if n > 1 else 0.0
            var_val = float(s.var(ddof=1)) if n > 1 else 0.0
            sum_val = float(s.sum())
            entry = {
                'var': var,
                'obs': n,
                'mean': mean_val,
                'std': std_val,
                'min': float(s.min()),
                'max': float(s.max()),
                'sum': sum_val,
                'Var': var_val
            }
            if detail:
                pcts = np.percentile(s, [1, 5, 10, 25, 50, 75, 90, 95, 99])
                skew_val = float(stats.skew(s, bias=False)) if n >= 3 else 0.0
                kurt_val = float(stats.kurtosis(s, fisher=False, bias=False)) if n >= 4 else 3.0
                sorted_vals = np.sort(s.to_numpy())
                smallest_4 = sorted_vals[:min(4, n)].tolist()
                largest_4 = sorted_vals[-min(4, n):].tolist()
                entry.update({
                    'p1': float(pcts[0]),
                    'p5': float(pcts[1]),
                    'p10': float(pcts[2]),
                    'p25': float(pcts[3]),
                    'p50': float(pcts[4]),
                    'p75': float(pcts[5]),
                    'p90': float(pcts[6]),
                    'p95': float(pcts[7]),
                    'p99': float(pcts[8]),
                    'skewness': skew_val,
                    'kurtosis': kurt_val,
                    'sum_w': float(n),
                    'smallest': smallest_4,
                    'largest': largest_4
                })
            results.append(entry)
    return results

def _order_preserving_qr_selection(X, tol=1e-7):
    """
    Select linearly independent columns of X in original left-to-right order (standard econometric convention),
    omitting collinear columns while retaining numerical stability and scale-invariance.
    """
    n, p = X.shape
    if p == 0:
        return [], [], np.zeros((n, 0)), np.zeros((0, 0))

    scales = np.zeros(p, dtype=float)
    for j in range(p):
        scales[j] = float(np.linalg.norm(X[:, j]))
    
    max_scale = float(np.max(scales)) if p > 0 else 0.0
    zero_thresh = float(np.finfo(float).eps * max(n, p) * max_scale)

    kept = []
    omitted = []
    Q_cols = []
    for j in range(p):
        if scales[j] <= zero_thresh or scales[j] == 0.0:
            omitted.append(j)
            continue
        # Work with normalized column so unit of measurement does not affect rank detection
        v = (X[:, j] / scales[j]).astype(float, copy=True)
        # Modified Gram-Schmidt with 2-pass re-orthogonalization for numerical stability
        for _ in range(2):
            for q in Q_cols:
                v -= np.dot(q, v) * q
        norm_res = float(np.linalg.norm(v))
        if norm_res < tol:
            omitted.append(j)
        else:
            q_new = v / norm_res
            Q_cols.append(q_new)
            kept.append(j)
    
    if len(kept) > 0:
        X_kept = X[:, kept]
        Q_kept, R_kept = scipy.linalg.qr(X_kept, mode='economic')
    else:
        Q_kept = np.zeros((n, 0))
        R_kept = np.zeros((0, 0))
    return kept, omitted, Q_kept, R_kept


def compute_regression(df, depvar, indepvars, robust=False, vce='ols', cluster_var=None, has_const=True, spatial_opts=None):
    """
    Fits Ordinary Least Squares (OLS) regression.
    Supports vce: 'ols', 'robust'/'hc1', 'hc2', 'hc3', 'cluster', 'spatial'/'conley'.
    Supports has_const=True (default with intercept) and has_const=False (noconstant).
    """
    if robust and vce == 'ols':
        vce = 'robust'
    vce = vce.lower() if vce else 'ols'
    spatial_opts = spatial_opts or {}

    needed_cols = [depvar] + indepvars
    if vce == 'cluster' and cluster_var and cluster_var not in needed_cols:
        needed_cols.append(cluster_var)
    if vce in ('spatial', 'conley'):
        lat_col = spatial_opts.get('lat')
        lon_col = spatial_opts.get('lon')
        if lat_col and lat_col not in needed_cols:
            needed_cols.append(lat_col)
        if lon_col and lon_col not in needed_cols:
            needed_cols.append(lon_col)

    data = df[needed_cols].dropna()
    n = len(data)
    k = len(indepvars)
    min_n = k + 1 if has_const else k
    
    if n <= min_n:
        raise ValueError(f"Insufficient observations ({n}) for {k} regressors.")

    y = data[depvar].to_numpy(dtype=float)
    X_raw = data[indepvars].to_numpy(dtype=float) if k > 0 else np.empty((n, 0))
    
    if has_const:
        X = np.column_stack([X_raw, np.ones(n)])
        names = indepvars + ['_cons']
        df_m = k
        df_r = n - k - 1
        df_t = n - 1
        y_mean = np.mean(y)
        ss_t = float(np.sum((y - y_mean) ** 2))
    else:
        X = X_raw
        names = list(indepvars)
        df_m = k
        df_r = n - k
        df_t = n
        ss_t = float(np.sum(y ** 2))
    
    p = X.shape[1]
    kept, omitted_cols, Q_1, R_11 = _order_preserving_qr_selection(X)
    rank = len(kept)
    omitted_vars = [names[j] for j in omitted_cols]

    beta = np.zeros(p, dtype=float)
    if rank > 0:
        S_kept = np.linalg.norm(X[:, kept], axis=0)
        S_kept = np.where(S_kept > 0, S_kept, 1.0)
        Xs_kept = X[:, kept] / S_kept
        Qs, Rs = scipy.linalg.qr(Xs_kept, mode='economic')
        beta_scaled = scipy.linalg.solve_triangular(Rs, Qs.T @ y)
        beta[kept] = beta_scaled / S_kept

        Rs_inv = scipy.linalg.solve_triangular(Rs, np.eye(rank))
        inv_XsXs = Rs_inv @ Rs_inv.T
        inv_R_R = inv_XsXs / np.outer(S_kept, S_kept)
        inv_XtX = np.zeros((p, p), dtype=float)
        inv_XtX[np.ix_(kept, kept)] = inv_R_R

        diag_Rs = np.abs(np.diag(Rs))
        cond_number = float(np.max(diag_Rs) / np.min(diag_Rs)) if np.min(diag_Rs) > 0 else 1.0
    else:
        inv_XtX = np.zeros((p, p), dtype=float)
        cond_number = 1.0

    y_pred = X @ beta
    resid = y - y_pred

    ss_r = float(np.sum(resid ** 2))
    df_m = max(0, rank - (1 if has_const else 0))
    df_r = max(1, n - rank)
    df_t = n - (1 if has_const else 0)
    ss_m = max(0.0, ss_t - ss_r)

    ms_m = ss_m / df_m if df_m > 0 else 0.0
    ms_r = ss_r / df_r if df_r > 0 else 0.0
    ms_t = ss_t / df_t if df_t > 0 else 0.0

    # Classical ANOVA F
    f_stat = ms_m / ms_r if ms_r > 0 and df_m > 0 else 0.0
    prob_f = float(1.0 - stats.f.cdf(f_stat, df_m, df_r)) if df_m > 0 and df_r > 0 else 1.0

    r2 = max(0.0, min(1.0, ss_m / ss_t)) if ss_t > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (df_t / df_r) if df_r > 0 else 0.0
    root_mse = float(np.sqrt(ms_r))

    warnings_list = []
    if rank < p:
        warnings_list.append(f"Rank deficiency detected: {len(omitted_vars)} collinear regressors omitted ({', '.join(omitted_vars)}).")
    if cond_number > 1e10:
        warnings_list.append(f"High regressor condition number ({cond_number:.2e}). Results may be sensitive to collinearity.")

    n_clusters = None
    df_vce = df_r

    if vce == 'ols':
        vcov = ms_r * inv_XtX
    elif vce in ('robust', 'hc1'):
        # HC1 without allocating N x N matrix: (X * resid^2)' X
        scale = n / df_r
        omega = (X * (resid[:, None] ** 2)).T @ X
        vcov = scale * (inv_XtX @ omega @ inv_XtX)
    elif vce == 'hc2':
        # Leverage h_ii
        h_diag = np.sum((X @ inv_XtX) * X, axis=1)
        w2 = (resid ** 2) / np.maximum(1e-12, (1.0 - h_diag))
        omega = (X * w2[:, None]).T @ X
        vcov = inv_XtX @ omega @ inv_XtX
    elif vce == 'hc3':
        h_diag = np.sum((X @ inv_XtX) * X, axis=1)
        w3 = (resid ** 2) / np.maximum(1e-12, (1.0 - h_diag) ** 2)
        omega = (X * w3[:, None]).T @ X
        vcov = inv_XtX @ omega @ inv_XtX
    elif vce == 'cluster':
        if not cluster_var or cluster_var not in data.columns:
            raise ValueError(f"Cluster variable '{cluster_var}' not found.")
        clusters = data[cluster_var].to_numpy()
        unique_clusters = np.unique(clusters)
        G = len(unique_clusters)
        if G <= 1:
            raise ValueError(f"Cluster variable '{cluster_var}' must have at least 2 distinct clusters (found {G}).")
        
        c = (G / (G - 1.0)) * ((n - 1.0) / (n - rank))
        M = np.zeros((p, p))
        scores = X * resid[:, None]
        for g in unique_clusters:
            s_g = np.sum(scores[clusters == g], axis=0)
            M += np.outer(s_g, s_g)
        vcov = c * (inv_XtX @ M @ inv_XtX)
        n_clusters = G
        df_vce = G - 1
    elif vce in ('spatial', 'conley'):
        lat_col = spatial_opts.get('lat')
        lon_col = spatial_opts.get('lon')
        cutoff = float(spatial_opts.get('cutoff', 100.0))
        if not lat_col or lat_col not in data.columns:
            raise ValueError(f"Spatial latitude variable '{lat_col}' not found.")
        if not lon_col or lon_col not in data.columns:
            raise ValueError(f"Spatial longitude variable '{lon_col}' not found.")

        lats = data[lat_col].to_numpy(dtype=float)
        lons = data[lon_col].to_numpy(dtype=float)

        # Resource safety guard for quadratic spatial distance matrix
        MAX_SPATIAL_N = 10000
        if n > MAX_SPATIAL_N:
            raise ValueError(
                f"Conley spatial covariance exceeded resource safety limit (N={n} > {MAX_SPATIAL_N}). "
                f"Full spatial cross-distance requires O(N^2) memory ({n*n*8 / 1e9:.2f} GB). "
                f"Use spatial clustering (vce(cluster ...)) or regional aggregation for large spatial datasets."
            )

        rad_lat = np.radians(lats)
        rad_lon = np.radians(lons)
        E = resid[:, None] * X
        omega = np.zeros((p, p), dtype=float)

        # Compute meat matrix in row blocks B <= 1000 to keep memory O(B * N) rather than O(N^2)
        block_size = min(1000, n)
        for i_start in range(0, n, block_size):
            i_end = min(i_start + block_size, n)
            b_lat = rad_lat[i_start:i_end]
            b_lon = rad_lon[i_start:i_end]

            dlat = b_lat[:, None] - rad_lat[None, :]
            dlon = b_lon[:, None] - rad_lon[None, :]
            a = np.sin(dlat / 2.0)**2 + np.cos(b_lat)[:, None] * np.cos(rad_lat)[None, :] * np.sin(dlon / 2.0)**2
            a = np.clip(a, 0.0, 1.0)
            d_km = 2.0 * 6371.0088 * np.arcsin(np.sqrt(a))

            # Bartlett triangular kernel
            u = d_km / cutoff
            W_block = np.maximum(0.0, 1.0 - u)

            # Accumulate block product into meat matrix: E_block.T @ (W_block @ E)
            omega += E[i_start:i_end].T @ (W_block @ E)

        scale = n / df_r if df_r > 0 else 1.0
        vcov = scale * (inv_XtX @ omega @ inv_XtX)
        df_vce = df_r
    else:
        raise ValueError(f"Unknown vce type: '{vce}'")

    # If robust / cluster / hc2 / hc3: compute robust Wald F test on non-omitted regressors
    if vce != 'ols' and df_m > 0:
        kept_indeps = [j for j in kept if j < k]
        if kept_indeps:
            beta_indep = beta[kept_indeps]
            vcov_indep = vcov[np.ix_(kept_indeps, kept_indeps)]
            try:
                inv_vindep = np.linalg.pinv(vcov_indep)
                f_stat = float((beta_indep.T @ inv_vindep @ beta_indep) / len(kept_indeps))
                prob_f = float(1.0 - stats.f.cdf(f_stat, len(kept_indeps), df_vce))
            except Exception:
                f_stat = 0.0
                prob_f = 1.0

    se = np.sqrt(np.maximum(0, np.diag(vcov)))
    t_stat = np.where(se > 0, np.divide(beta, se, out=np.zeros_like(beta), where=(se > 0)), 0.0)
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df_vce))

    t_crit = stats.t.ppf(0.975, df_vce)
    ci_lower = beta - t_crit * se
    ci_upper = beta + t_crit * se

    coef_table = []
    for i, name in enumerate(names):
        is_omitted = name in omitted_vars
        coef_table.append({
            'name': name,
            'coef': float(beta[i]),
            'se': 0.0 if is_omitted else float(se[i]),
            't': 0.0 if is_omitted else float(t_stat[i]),
            'p': 1.0 if is_omitted else float(p_val[i]),
            'ci_lower': 0.0 if is_omitted else float(ci_lower[i]),
            'ci_upper': 0.0 if is_omitted else float(ci_upper[i]),
            'omitted': is_omitted
        })

    return {
        'model_type': 'ols',
        'depvar': depvar,
        'indepvars': indepvars,
        'n': n,
        'rank': rank,
        'omitted_vars': omitted_vars,
        'cond_number': cond_number,
        'warnings': warnings_list,
        'df_m': df_m,
        'df_r': df_vce,
        'df_t': df_t,
        'ss_m': ss_m,
        'ss_r': ss_r,
        'ss_t': ss_t,
        'ms_m': ms_m,
        'ms_r': ms_r,
        'ms_t': ms_t,
        'f_stat': f_stat,
        'prob_f': prob_f,
        'r2': r2,
        'adj_r2': adj_r2,
        'root_mse': root_mse,
        'coefs': coef_table,
        'beta': beta,
        'vcov': vcov,
        'var_names': names,
        'data_index': data.index,
        'fitted_design': X,
        'leverage': np.sum(Qs**2, axis=1) if rank else np.zeros(n),
        'y_pred': y_pred,
        'resid': resid,
        'robust': (vce != 'ols'),
        'vce': vce,
        'cluster_var': cluster_var,
        'n_clusters': n_clusters,
        'spatial_opts': spatial_opts
    }

def standardized_ols_residuals(model, index, design, y):
    """Return internal studentized residuals at the original estimation observations."""
    if model.get('model_type') != 'ols':
        raise ValueError("rstandard requires an OLS regression")
    if 'leverage' not in model or 'fitted_design' not in model:
        raise ValueError("refit the OLS model to compute standardized residuals")
    sample = pd.Index(model['data_index'])
    if index.equals(sample):
        positions = np.arange(len(index))
    elif index.is_unique and sample.is_unique:
        positions = index.get_indexer(sample)
    else:
        raise ValueError("cannot identify the fitted observations with duplicate index labels")
    residuals = np.asarray(model['resid'])
    if (np.any(positions < 0)
            or not np.array_equal(design[positions], model['fitted_design'])
            or not np.allclose(y[positions] - model['y_pred'], residuals, rtol=1e-12, atol=1e-12)):
        raise ValueError("the estimation observations have changed; refit before requesting rstandard")
    one_minus_hat = 1.0 - np.asarray(model['leverage'])
    usable = (one_minus_hat > np.finfo(float).eps * max(1, design.shape[1])) & (model['root_mse'] > 0)
    result = np.full(len(index), np.nan)
    result[positions[usable]] = residuals[usable] / (model['root_mse'] * np.sqrt(one_minus_hat[usable]))
    return result


def compute_mle_vcov_cluster(inv_H, scores, clusters):
    """
    Computes standard econometric cluster-robust variance covariance matrix for MLE models:
    VCOV = (G / (G - 1)) * inv_H @ (sum_{g=1}^G s_g s_g') @ inv_H
    where s_g = sum_{i in g} score_i.
    Inference is standard normal z.
    """
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    if G <= 1:
        raise ValueError(f"Cluster variable must have at least 2 distinct clusters (found {G}).")
    k = inv_H.shape[0]
    meat = np.zeros((k, k), dtype=float)
    for g in unique_clusters:
        s_g = np.sum(scores[clusters == g], axis=0)
        meat += np.outer(s_g, s_g)
    c = G / (G - 1.0)
    vcov = c * (inv_H @ meat @ inv_H)
    return vcov, G

def compute_logit(df, depvar, indepvars, odds_ratio=False, max_iter=35, tol=1e-6, vce='mle', cluster_var=None):
    """
    Fits Logistic regression via Newton-Raphson MLE.
    Supports vce: 'mle', 'robust', 'cluster'.
    """
    vce = vce.lower() if vce else 'mle'
    needed_cols = [depvar] + indepvars
    if vce == 'cluster' and cluster_var and cluster_var not in needed_cols:
        needed_cols.append(cluster_var)

    data = df[needed_cols].dropna()
    n = len(data)
    k = len(indepvars)
    
    if n <= k + 1:
        raise ValueError(f"Insufficient observations ({n}) for {k} regressors.")
        
    if isinstance(data[depvar].dtype, pd.CategoricalDtype):
        y = data[depvar].cat.codes.to_numpy(dtype=float)
    else:
        try:
            y = data[depvar].to_numpy(dtype=float)
        except (ValueError, TypeError):
            unq = sorted([v for v in data[depvar].unique() if pd.notna(v)])
            if len(unq) == 2:
                y = (data[depvar] == unq[1]).astype(float).to_numpy()
            else:
                y = data[depvar].to_numpy(dtype=float)
    unique_vals = np.unique(y)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f"Dependent variable '{depvar}' must be binary (0 or 1).")
    if len(unique_vals) < 2:
        raise ValueError(f"Dependent variable '{depvar}' has no variation (all values are {unique_vals[0]}).")
        
    X_raw = data[indepvars].to_numpy(dtype=float) if k > 0 else np.empty((n, 0))
    X = np.column_stack([X_raw, np.ones(n)])
    names = indepvars + ['_cons']
    
    # Null model log-likelihood
    y_bar = np.mean(y)
    ll_0 = n * (y_bar * np.log(y_bar + 1e-15) + (1.0 - y_bar) * np.log(1.0 - y_bar + 1e-15))
    
    def _compute_logit_ll(b):
        eta_val = np.clip(X @ b, -35.0, 35.0)
        ll_i = np.where(y == 1, -np.log1p(np.exp(-eta_val)), -np.log1p(np.exp(eta_val)))
        return float(np.sum(ll_i))

    # Newton-Raphson MLE solver with backtracking line search and separation detection
    beta = np.zeros(k + 1)
    converged = False
    status = "Iteration limit reached without convergence"
    warnings_list = []
    current_ll = _compute_logit_ll(beta)
    ll_history = [float(current_ll)]
    eta = np.clip(X @ beta, -35.0, 35.0)
    p = 1.0 / (1.0 + np.exp(-eta))
    grad = X.T @ (y - p)
    grad_norm = float(np.linalg.norm(grad))
    step_norm = 0.0
    iteration = 0

    if grad_norm < tol:
        converged = True
        status = "Convergence achieved (initial gradient within tolerance)"
    else:
        for it in range(max_iter):
            iteration = it + 1

            # Separation detection: complete separation occurs when all y=1 have p > 0.999 and all y=0 have p < 0.001
            is_sep = False
            if n >= 4:
                p_pos = p[y == 1]
                p_neg = p[y == 0]
                if len(p_pos) > 0 and len(p_neg) > 0:
                    if np.all(p_pos > 0.999) and np.all(p_neg < 0.001):
                        is_sep = True

            if is_sep or np.max(np.abs(beta)) > 1e4:
                status = "Complete or quasi-complete separation detected (outcome perfectly predicted)"
                warnings_list.append("Complete or quasi-complete separation detected; standard errors and inference are invalid.")
                converged = False
                break

            w = p * (1.0 - p)
            w = np.clip(w, 1e-15, 0.25)
            H = -(X.T * w) @ X

            try:
                step = np.linalg.solve(-H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(-H) @ grad
                warnings_list.append("Hessian matrix is singular.")

            dir_deriv = float(np.dot(grad, step))
            if dir_deriv <= 0:
                step = grad / (grad_norm + 1e-12)
                dir_deriv = float(np.dot(grad, step))

            # Backtracking Armijo line search for log-likelihood maximization
            alpha = 1.0
            step_accepted = False
            for _ in range(10):
                beta_trial = beta + alpha * step
                trial_ll = _compute_logit_ll(beta_trial)
                if np.isfinite(trial_ll) and trial_ll >= current_ll + 1e-4 * alpha * dir_deriv:
                    beta = beta_trial
                    current_ll = trial_ll
                    step_accepted = True
                    break
                alpha *= 0.5

            if not step_accepted:
                converged = False
                status = "Line search failed to find sufficient increase in log-likelihood"
                warnings_list.append("Line search failed to find sufficient increase; optimization stopped.")
                break

            ll_history.append(float(current_ll))
            step_norm = float(np.max(np.abs(alpha * step)))

            # Re-evaluate probabilities and gradient at accepted parameters
            eta = np.clip(X @ beta, -35.0, 35.0)
            p = 1.0 / (1.0 + np.exp(-eta))
            grad = X.T @ (y - p)
            grad_norm = float(np.linalg.norm(grad))

            if step_norm < tol or grad_norm < tol:
                converged = True
                status = "Convergence achieved (gradient and step within tolerance)"
                break

    if not converged and not any("separation" in w.lower() for w in warnings_list):
        warnings_list.append("Model did not achieve convergence. Standard errors and inference should be treated as exploratory.")

    # Final probabilities and log-likelihood at final beta
    eta = np.clip(X @ beta, -35.0, 35.0)
    p = 1.0 / (1.0 + np.exp(-eta))
    ll = float(np.sum(y * np.log(p + 1e-15) + (1.0 - y) * np.log(1.0 - p + 1e-15)))

    # Final Hessian at final beta
    w = p * (1.0 - p)
    w = np.clip(w, 1e-15, 0.25)
    H = -(X.T * w) @ X
    try:
        inv_H = np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        inv_H = np.linalg.pinv(-H)

    n_clusters = None
    df_vce = n - k - 1

    if vce == 'mle':
        vcov = inv_H
    elif vce in ('robust', 'hc1'):
        scores = X * (y - p)[:, None]
        omega = scores.T @ scores
        c = n / (n - 1.0)
        vcov = c * (inv_H @ omega @ inv_H)
    elif vce == 'cluster':
        if not cluster_var or cluster_var not in data.columns:
            raise ValueError(f"Cluster variable '{cluster_var}' not found.")
        clusters = data[cluster_var].to_numpy()
        scores = X * (y - p)[:, None]
        vcov, n_clusters = compute_mle_vcov_cluster(inv_H, scores, clusters)
        df_vce = n_clusters - 1
    else:
        raise ValueError(f"Unsupported vce '{vce}' for logit. Allowed: 'mle', 'robust', 'cluster'.")

    se = np.sqrt(np.maximum(0, np.diag(vcov)))
    z_stat = np.where(se > 0, beta / se, 0.0)
    p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat)))
    z_crit = stats.norm.ppf(0.975)
    
    lr_chi2 = float(2.0 * (ll - ll_0))
    prob_chi2 = float(1.0 - stats.chi2.cdf(lr_chi2, k)) if k > 0 else 1.0
    pseudo_r2 = float(1.0 - (ll / ll_0)) if ll_0 != 0 else 0.0
    
    coef_table = []
    for i, name in enumerate(names):
        b = float(beta[i])
        s = float(se[i])
        z = float(z_stat[i])
        pv = float(p_val[i])
        ci_l = b - z_crit * s
        ci_u = b + z_crit * s
        
        if odds_ratio and name != '_cons':
            or_val = float(np.exp(np.clip(b, -100, 100)))
            or_se = float(or_val * s)
            ci_l_or = float(np.exp(np.clip(ci_l, -100, 100)))
            ci_u_or = float(np.exp(ci_u)) if ci_u < 700 else float('inf')
            coef_table.append({
                'name': name,
                'coef': or_val,
                'se': or_se,
                'z': z,
                'p': pv,
                'ci_lower': ci_l_or,
                'ci_upper': ci_u_or,
                'is_odds_ratio': True
            })
        else:
            coef_table.append({
                'name': name,
                'coef': b,
                'se': s,
                'z': z,
                'p': pv,
                'ci_lower': ci_l,
                'ci_upper': ci_u,
                'is_odds_ratio': False
            })
            
    return {
        'model_type': 'logit',
        'depvar': depvar,
        'indepvars': indepvars,
        'n': n,
        'df_m': k,
        'df_r': df_vce,
        'll': ll,
        'll_0': ll_0,
        'lr_chi2': lr_chi2,
        'prob_chi2': prob_chi2,
        'pseudo_r2': pseudo_r2,
        'coefs': coef_table,
        'beta': beta,
        'vcov': vcov,
        'var_names': names,
        'data_index': data.index,
        'fitted_design': X,
        'p_pred': p,
        'odds_ratio': odds_ratio,
        'converged': converged,
        'convergence_status': status,
        'iterations': iteration,
        'max_iter': max_iter,
        'gradient_norm': grad_norm,
        'final_score_norm': grad_norm,
        'final_step_norm': step_norm,
        'll_history': ll_history,
        'warnings': warnings_list,
        'vce': vce,
        'cluster_var': cluster_var,
        'n_clusters': n_clusters
    }

def compute_ivregress_2sls(df, depvar, exog_vars, endog_vars, instruments, small=False, vce='unadjusted', cluster_var=None):
    """
    Fits Two-Stage Least Squares (2SLS) instrumental variables regression.
    Supports small-sample convention (small=True) and large-sample convention (small=False).
    Supports vce: 'unadjusted', 'robust', 'cluster'.
    """
    vce = vce.lower() if vce else 'unadjusted'
    all_cols = list(set([depvar] + exog_vars + endog_vars + instruments))
    if vce == 'cluster' and cluster_var:
        all_cols.append(cluster_var)

    data = df[all_cols].dropna()
    n = len(data)
    k = len(endog_vars) + len(exog_vars)
    
    if len(instruments) < len(endog_vars):
        raise ValueError(f"Equation underidentified: {len(instruments)} instruments for {len(endog_vars)} endogenous variables.")
        
    if n <= k + 1:
        raise ValueError(f"Insufficient observations ({n}) for {k} regressors.")
    
    y = data[depvar].to_numpy(dtype=float)
    X_endog = data[endog_vars].to_numpy(dtype=float)
    X_exog = data[exog_vars].to_numpy(dtype=float) if exog_vars else np.empty((n, 0))
    Z_inst = data[instruments].to_numpy(dtype=float)
    
    Z = np.column_stack([Z_inst, X_exog, np.ones(n)])
    X_actual = np.column_stack([X_endog, X_exog, np.ones(n)])
    
    # Projected regressors using pivoted QR on Z to handle collinear instruments stably
    Q_Z, R_Z, P_Z = scipy.linalg.qr(Z, mode='economic', pivoting=True)
    diag_Z = np.abs(np.diag(R_Z))
    tol_Z = max(n, Z.shape[1]) * np.finfo(float).eps * diag_Z[0] if len(diag_Z) > 0 and diag_Z[0] > 0 else 1e-12
    rank_Z = int(np.sum(diag_Z > tol_Z))
    if rank_Z < X_actual.shape[1]:
        raise ValueError("Model is underidentified: effective instrument rank is less than number of regressors.")
    Q_Z_kept = Q_Z[:, :rank_Z]
    X_hat = Q_Z_kept @ (Q_Z_kept.T @ X_actual)

    # Second stage using QR on X_hat
    Q_X, R_X, P_X = scipy.linalg.qr(X_hat, mode='economic', pivoting=True)
    diag_X = np.abs(np.diag(R_X))
    tol_X = max(n, X_hat.shape[1]) * np.finfo(float).eps * diag_X[0] if len(diag_X) > 0 and diag_X[0] > 0 else 1e-12
    rank_X = int(np.sum(diag_X > tol_X))
    if rank_X < X_hat.shape[1]:
        raise ValueError("Projected regressor matrix X_hat lacks full rank. Model is underidentified or instruments lack rank.")

    R_11 = R_X[:rank_X, :rank_X]
    beta_piv = scipy.linalg.solve_triangular(R_11, Q_X[:, :rank_X].T @ y)
    beta = np.zeros(X_hat.shape[1])
    beta[P_X[:rank_X]] = beta_piv

    R_inv = scipy.linalg.solve_triangular(R_11, np.eye(rank_X))
    inv_XhXh = np.zeros((X_hat.shape[1], X_hat.shape[1]))
    inv_XhXh[np.ix_(P_X[:rank_X], P_X[:rank_X])] = R_inv @ R_inv.T

    resid = y - X_actual @ beta
    
    df_r = n - k - 1
    # Large-sample divides by n; small-sample divides by n - k - 1
    sigma2 = float(np.sum(resid ** 2) / (df_r if small else n))
    
    n_clusters = None
    df_vce = df_r if small else n

    if vce == 'unadjusted':
        vcov = sigma2 * inv_XhXh
    elif vce in ('robust', 'hc1'):
        scale = (n / df_r) if small else 1.0
        scores = X_hat * resid[:, None]
        omega = scores.T @ scores
        vcov = scale * (inv_XhXh @ omega @ inv_XhXh)
    elif vce == 'cluster':
        if not cluster_var or cluster_var not in data.columns:
            raise ValueError(f"Cluster variable '{cluster_var}' not found.")
        clusters = data[cluster_var].to_numpy()
        unique_clusters = np.unique(clusters)
        G = len(unique_clusters)
        if G <= 1:
            raise ValueError(f"Cluster variable '{cluster_var}' must have at least 2 distinct clusters.")
        c = (G / (G - 1.0)) * (((n - 1.0) / (n - (k + 1.0))) if small else 1.0)
        M = np.zeros((k + 1, k + 1))
        scores = X_hat * resid[:, None]
        for g in unique_clusters:
            s_g = np.sum(scores[clusters == g], axis=0)
            M += np.outer(s_g, s_g)
        vcov = c * (inv_XhXh @ M @ inv_XhXh)
        n_clusters = G
        df_vce = G - 1 if small else n
    else:
        raise ValueError(f"Unsupported vce '{vce}' for ivregress. Allowed: 'unadjusted', 'robust', 'cluster'.")

    se = np.sqrt(np.maximum(0, np.diag(vcov)))
    stat_val = np.where(se > 0, beta / se, 0.0)
    
    if small:
        p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(stat_val), df_vce))
        crit_val = stats.t.ppf(0.975, df_vce)
    else:
        p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(stat_val)))
        crit_val = stats.norm.ppf(0.975)

    names = endog_vars + exog_vars + ['_cons']
    coef_table = []
    for i, name in enumerate(names):
        b = float(beta[i])
        s = float(se[i])
        st = float(stat_val[i])
        pv = float(p_val[i])
        ci_l = b - crit_val * s
        ci_u = b + crit_val * s
        entry = {
            'name': name,
            'coef': b,
            'se': s,
            'stat': st,
            't': st,
            'z': st,
            'p': pv,
            'ci_lower': ci_l,
            'ci_upper': ci_u
        }
        coef_table.append(entry)
        
    y_mean = np.mean(y)
    ss_t = float(np.sum((y - y_mean) ** 2))
    ss_r = float(np.sum(resid ** 2))
    r2 = 1.0 - (ss_r / ss_t) if ss_t > 0 else 0.0
    root_mse = float(np.sqrt(sigma2))
    
    return {
        'model_type': '2sls',
        'depvar': depvar,
        'endog_vars': endog_vars,
        'exog_vars': exog_vars,
        'instruments': instruments,
        'n': n,
        'df_r': df_vce,
        'r2': r2,
        'root_mse': root_mse,
        'coefs': coef_table,
        'beta': beta,
        'vcov': vcov,
        'var_names': names,
        'data_index': data.index,
        'resid': resid,
        'small': small,
        'vce': vce,
        'cluster_var': cluster_var,
        'n_clusters': n_clusters
    }

def compute_panel_fe(df, depvar, indepvars, panelvar, timevar=None, vce='classical', cluster_var=None):
    """
    Fits Panel Fixed Effects (within estimator).
    Supports vce: 'classical', 'robust', 'cluster'.
    """
    vce = vce.lower() if vce else 'classical'
    raw_cols = [depvar, panelvar] + indepvars
    if timevar and timevar in df.columns:
        raw_cols.append(timevar)
    if vce == 'cluster' and cluster_var and cluster_var in df.columns:
        raw_cols.append(cluster_var)

    needed_cols = list(dict.fromkeys(raw_cols))
    data = df[needed_cols].dropna().copy()
    n = len(data)
    num_entities = int(data[panelvar].nunique())
    k = len(indepvars)
    df_r = n - num_entities - k
    
    if df_r <= 0:
        raise ValueError(f"Degrees of freedom ({df_r}) <= 0 for {num_entities} panels.")
        
    grouped = data.groupby(panelvar)
    y_raw = data[depvar].to_numpy(dtype=float)
    X_raw = data[indepvars].to_numpy(dtype=float) if k > 0 else np.empty((n, 0))

    # Pooled OLS for F-test of individual effects u_i = 0
    X_pooled = np.column_stack([X_raw, np.ones(n)])
    try:
        inv_XpXp = np.linalg.pinv(X_pooled.T @ X_pooled)
        beta_pooled = inv_XpXp @ (X_pooled.T @ y_raw)
        ss_pooled = float(np.sum((y_raw - X_pooled @ beta_pooled) ** 2))
    except Exception:
        ss_pooled = float(np.sum((y_raw - np.mean(y_raw)) ** 2))

    # Within demeaning
    y_dm = (data[depvar] - grouped[depvar].transform('mean')).to_numpy(dtype=float)
    X_dm = np.zeros((n, k), dtype=float)
    for col_idx, col in enumerate(indepvars):
        X_dm[:, col_idx] = (data[col] - grouped[col].transform('mean')).to_numpy(dtype=float)
        
    # Numerically stable QR on within-demeaned regressors
    Q_dm, R_dm, P_dm = scipy.linalg.qr(X_dm, mode='economic', pivoting=True)
    diag_dm = np.abs(np.diag(R_dm))
    tol_dm = max(n, k) * np.finfo(float).eps * diag_dm[0] if len(diag_dm) > 0 and diag_dm[0] > 0 else 1e-12
    rank_dm = int(np.sum(diag_dm > tol_dm))

    omitted_vars = []
    if rank_dm < k:
        omitted_vars = [indepvars[j] for j in sorted(P_dm[rank_dm:])]

    beta = np.zeros(k)
    inv_XtX = np.zeros((k, k))
    if rank_dm > 0:
        R_11 = R_dm[:rank_dm, :rank_dm]
        Q_1 = Q_dm[:, :rank_dm]
        beta_piv = scipy.linalg.solve_triangular(R_11, Q_1.T @ y_dm)
        beta[P_dm[:rank_dm]] = beta_piv

        R_inv = scipy.linalg.solve_triangular(R_11, np.eye(rank_dm))
        inv_XtX[np.ix_(P_dm[:rank_dm], P_dm[:rank_dm])] = R_inv @ R_inv.T
    
    resid = y_dm - X_dm @ beta
    ss_r = float(np.sum(resid ** 2))
    sigma2 = ss_r / df_r
    sigma_e = float(np.sqrt(sigma2))
    
    n_clusters = None
    df_vce = df_r

    if vce == 'classical':
        vcov = sigma2 * inv_XtX
    elif vce in ('robust', 'hc1'):
        omega = (X_dm * (resid[:, None] ** 2)).T @ X_dm
        scale = n / df_r
        vcov = scale * (inv_XtX @ omega @ inv_XtX)
    elif vce == 'cluster':
        c_var = cluster_var if cluster_var else panelvar
        if c_var not in data.columns:
            raise ValueError(f"Cluster variable '{c_var}' not found.")
        clusters = data[c_var].to_numpy()
        unique_clusters = np.unique(clusters)
        G = len(unique_clusters)
        if G <= 1:
            raise ValueError(f"Cluster variable '{c_var}' must have at least 2 distinct clusters.")
        c = (G / (G - 1.0)) * ((n - 1.0) / df_r)
        M = np.zeros((k, k), dtype=float)
        scores = X_dm * resid[:, None]
        for g in unique_clusters:
            s_g = np.sum(scores[clusters == g], axis=0)
            M += np.outer(s_g, s_g)
        vcov = c * (inv_XtX @ M @ inv_XtX)
        n_clusters = G
        df_vce = G - 1
    else:
        raise ValueError(f"Unsupported vce '{vce}' for panel_fe. Allowed: 'classical', 'robust', 'cluster'.")

    se = np.sqrt(np.maximum(0, np.diag(vcov)))
    t_stat = np.where(se > 0, beta / se, 0.0)
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df_vce))
    t_crit = stats.t.ppf(0.975, df_vce)
    
    coef_table = []
    for i, name in enumerate(indepvars):
        is_omitted = name in omitted_vars
        b = float(beta[i])
        s = 0.0 if is_omitted else float(se[i])
        t = 0.0 if is_omitted else float(t_stat[i])
        pv = 1.0 if is_omitted else float(p_val[i])
        ci_l = 0.0 if is_omitted else b - t_crit * s
        ci_u = 0.0 if is_omitted else b + t_crit * s
        coef_table.append({
            'name': name,
            'coef': b,
            'se': s,
            't': t,
            'p': pv,
            'ci_lower': ci_l,
            'ci_upper': ci_u,
            'omitted': is_omitted
        })
        
    ss_t_within = float(np.sum(y_dm ** 2))
    r2_within = max(0.0, 1.0 - (ss_r / ss_t_within)) if ss_t_within > 0 else 0.0
    
    # Grand mean constant
    x_means = np.array([data[col].mean() for col in indepvars]) if k > 0 else np.array([])
    cons_est = float(data[depvar].mean() - (np.dot(beta, x_means) if k > 0 else 0.0))
    cons_var = float((sigma2 / n) + (x_means @ vcov @ x_means if k > 0 else 0.0)) if n > 0 else 0.0
    cons_se = float(np.sqrt(np.maximum(0, cons_var)))
    cons_t = float(cons_est / cons_se) if cons_se > 0 else 0.0
    cons_p = float(2.0 * (1.0 - stats.t.cdf(np.abs(cons_t), df_vce))) if df_vce > 0 else 0.0
    
    coef_table.append({
        'name': '_cons',
        'coef': cons_est,
        'se': cons_se,
        't': cons_t,
        'p': cons_p,
        'ci_lower': cons_est - t_crit * cons_se,
        'ci_upper': cons_est + t_crit * cons_se
    })
    
    vcov_full = np.zeros((k + 1, k + 1))
    vcov_full[:k, :k] = vcov
    vcov_full[k, k] = cons_var

    # Between and Overall R2: Standard theory defines r2_between = Corr(y_means, X_means @ beta)^2
    # and r2_overall = Corr(y_raw, X_raw @ beta)^2 (documented in [XT] xtreg fe)
    y_means = grouped[depvar].mean().to_numpy(dtype=float)
    T_bar = float(n) / float(num_entities)
    X_means = np.zeros((num_entities, k), dtype=float)
    for col_idx, col in enumerate(indepvars):
        X_means[:, col_idx] = grouped[col].mean().to_numpy(dtype=float)

    if k > 0:
        corr_b = np.corrcoef(y_means, X_means @ beta)[0, 1]
        r2_between = float(corr_b ** 2) if not np.isnan(corr_b) else 0.0
        corr_o = np.corrcoef(y_raw, X_raw @ beta)[0, 1]
        r2_overall = float(corr_o ** 2) if not np.isnan(corr_o) else 0.0
    else:
        r2_between = 0.0
        r2_overall = 0.0

    # sigma_u and rho
    u_i = y_means - cons_est - (X_means @ beta if k > 0 else 0.0)
    var_u_sample = float(np.sum((u_i - np.mean(u_i)) ** 2) / max(1, num_entities - 1))
    sigma_u = float(np.sqrt(max(0.0, var_u_sample - (sigma2 / T_bar))))
    rho = float((sigma_u ** 2) / (sigma_u ** 2 + sigma_e ** 2)) if (sigma_u ** 2 + sigma_e ** 2) > 0 else 0.0

    # F-test for all u_i = 0
    df_effects = num_entities - 1
    f_effects = float(((ss_pooled - ss_r) / df_effects) / (ss_r / df_r)) if df_effects > 0 and ss_r > 0 else 0.0
    prob_f_effects = float(1.0 - stats.f.cdf(f_effects, df_effects, df_r)) if df_effects > 0 else 1.0

    # Overall model F-stat
    if k > 0 and df_vce > 0 and (1.0 - r2_within) > 0:
        f_stat = float((r2_within / k) / ((1.0 - r2_within) / df_vce))
        prob_f = float(1.0 - stats.f.cdf(f_stat, k, df_vce))
    else:
        f_stat = 0.0
        prob_f = 1.0

    grp_sizes = grouped.size()
    min_obs = int(grp_sizes.min()) if len(grp_sizes) > 0 else 0
    avg_obs = float(grp_sizes.mean()) if len(grp_sizes) > 0 else 0.0
    max_obs = int(grp_sizes.max()) if len(grp_sizes) > 0 else 0
    corr_u_xb = float(np.corrcoef(u_i, X_means @ beta)[0, 1]) if k > 0 and len(u_i) > 1 and not np.isnan(np.corrcoef(u_i, X_means @ beta)[0, 1]) else 0.0

    return {
        'model_type': 'panel_fe',
        'depvar': depvar,
        'indepvars': indepvars,
        'panelvar': panelvar,
        'timevar': timevar,
        'n': n,
        'rank': rank_dm,
        'omitted_vars': omitted_vars,
        'num_panels': num_entities,
        'n_groups': num_entities,
        'df_m': k,
        'df_r': df_vce,
        'f_stat': f_stat,
        'prob_f': prob_f,
        'min_obs': min_obs,
        'avg_obs': avg_obs,
        'max_obs': max_obs,
        'corr_u_xb': corr_u_xb,
        'r2_within': r2_within,
        'r2_between': r2_between,
        'r2_overall': r2_overall,
        'sigma_e': sigma_e,
        'rmse': sigma_e,
        'sigma_u': sigma_u,
        'rho': rho,
        'df_effects': df_effects,
        'f_effects': f_effects,
        'prob_f_effects': prob_f_effects,
        'coefs': coef_table,
        'beta': np.append(beta, cons_est),
        'vcov': vcov_full,
        'var_names': indepvars + ['_cons'],
        'data_index': data.index,
        'resid': resid,
        'vce': vce,
        'n_clusters': n_clusters,
        'cluster_var': cluster_var
    }


def compute_panel_re(data, depvar, indepvars, panelvar, timevar=None, vce='classical', cluster_var=None):
    """
    Computes Random-Effects GLS regression using the Swamy-Arora variance component decomposition.
    Matches Standard random effects conventions:
      - sigma_e from fixed effects within residuals (sse / (N - n - K))
      - sigma_u from between residuals with harmonic mean Tbar = n / sum(1/T_i)
      - theta_i = 1 - sqrt(sigma_e^2 / (T_i * sigma_u^2 + sigma_e^2))
      - GLS quasi-demeaning with transformed constant (1 - theta_i)
      - Standard normal z-inference (z, P>|z|, 95% normal CI)
      - Model Wald chi2 test of slope coefficients
    """
    all_vars = [depvar] + list(indepvars) + [panelvar]
    if timevar and timevar not in all_vars:
        all_vars.append(timevar)
    if cluster_var and cluster_var not in all_vars:
        all_vars.append(cluster_var)

    data = data.dropna(subset=all_vars).copy()
    n = len(data)
    if n == 0:
        raise ValueError("No observations after dropping missing values.")

    grouped = data.groupby(panelvar, sort=True)
    num_entities = data[panelvar].nunique()
    if num_entities < 2:
        raise ValueError("Panel variable must have at least 2 distinct entities.")

    k = len(indepvars)

    # 1. Fixed Effects (Within) for sigma_e
    y_raw = data[depvar].to_numpy(dtype=float)
    y_mean_trans = grouped[depvar].transform('mean').to_numpy(dtype=float)
    y_dm = y_raw - y_mean_trans

    if k > 0:
        X_raw = data[indepvars].to_numpy(dtype=float)
        X_mean_trans = np.zeros_like(X_raw)
        for col_idx, col in enumerate(indepvars):
            X_mean_trans[:, col_idx] = grouped[col].transform('mean').to_numpy(dtype=float)
        X_dm = X_raw - X_mean_trans

        # Check collinearity in within space
        rank_dm = np.linalg.matrix_rank(X_dm)
        omitted_vars = set()
        if rank_dm < k:
            kept_cols = []
            for col_idx in range(k):
                test_cols = kept_cols + [col_idx]
                if np.linalg.matrix_rank(X_dm[:, test_cols]) == len(test_cols):
                    kept_cols.append(col_idx)
                else:
                    omitted_vars.add(indepvars[col_idx])
            X_dm_clean = X_dm[:, kept_cols] if kept_cols else np.zeros((n, 0))
        else:
            X_dm_clean = X_dm

        if X_dm_clean.shape[1] > 0:
            beta_fe = np.linalg.lstsq(X_dm_clean, y_dm, rcond=None)[0]
            resid_fe = y_dm - X_dm_clean @ beta_fe
        else:
            beta_fe = np.array([])
            resid_fe = y_dm
    else:
        X_dm = np.zeros((n, 0))
        X_dm_clean = X_dm
        omitted_vars = set()
        beta_fe = np.array([])
        resid_fe = y_dm

    df_fe = max(1, n - num_entities - (len(beta_fe) if k > 0 else 0))
    s2_e = float(np.sum(resid_fe ** 2) / df_fe)
    sigma_e = float(np.sqrt(s2_e))

    # 2. Between Effects for sigma_u
    y_means = grouped[depvar].mean().to_numpy(dtype=float)
    if k > 0:
        X_means = np.zeros((num_entities, k), dtype=float)
        for col_idx, col in enumerate(indepvars):
            X_means[:, col_idx] = grouped[col].mean().to_numpy(dtype=float)
        X_between = np.column_stack([X_means, np.ones(num_entities)])
    else:
        X_between = np.ones((num_entities, 1))

    beta_between = np.linalg.lstsq(X_between, y_means, rcond=None)[0]
    resid_between = y_means - X_between @ beta_between
    df_between = max(1, num_entities - (k + 1))
    s2_between = float(np.sum(resid_between ** 2) / df_between)

    # Harmonic mean of group sizes Tbar = n_groups / sum(1/T_i)
    grp_sizes = grouped.size()
    T_i = grp_sizes.to_numpy(dtype=float)
    T_bar = float(num_entities / np.sum(1.0 / T_i)) if np.all(T_i > 0) else float(grp_sizes.mean())

    s2_u = max(0.0, s2_between - (s2_e / T_bar))
    sigma_u = float(np.sqrt(s2_u))
    rho = float(s2_u / (s2_u + s2_e)) if (s2_u + s2_e) > 0 else 0.0

    # 3. Theta calculation and Quasi-Demeaning
    theta_per_entity = 1.0 - np.sqrt(s2_e / (T_i * s2_u + s2_e))
    theta_map = dict(zip(grp_sizes.index, theta_per_entity))
    theta_obs = data[panelvar].map(theta_map).to_numpy(dtype=float)

    y_star = y_raw - theta_obs * y_mean_trans
    if k > 0:
        X_star_cols = [X_raw[:, c] - theta_obs * X_mean_trans[:, c] for c in range(k)]
        X_star = np.column_stack(X_star_cols + [1.0 - theta_obs])
    else:
        X_star = (1.0 - theta_obs)[:, None]

    # 4. OLS on Quasi-Demeaned variables
    beta_re = np.linalg.lstsq(X_star, y_star, rcond=None)[0]
    resid_star = y_star - X_star @ beta_re
    df_r = max(1, n - (k + 1))
    s2_star = float(np.sum(resid_star ** 2) / df_r)

    inv_XtX = np.linalg.inv(X_star.T @ X_star)

    # 5. Variance-Covariance Matrix
    n_clusters = None
    if vce == 'classical':
        vcov = s2_star * inv_XtX
    elif vce in ('robust', 'hc1'):
        omega = (X_star * (resid_star[:, None] ** 2)).T @ X_star
        scale = n / float(df_r)
        vcov = scale * (inv_XtX @ omega @ inv_XtX)
    elif vce == 'cluster':
        c_var = cluster_var if cluster_var else panelvar
        if c_var not in data.columns:
            raise ValueError(f"Cluster variable '{c_var}' not found.")
        clusters = data[c_var].to_numpy()
        unique_clusters = np.unique(clusters)
        G = len(unique_clusters)
        if G <= 1:
            raise ValueError(f"Cluster variable '{c_var}' must have at least 2 distinct clusters.")
        c = (G / (G - 1.0)) * ((n - 1.0) / df_r)
        M = np.zeros((k + 1, k + 1), dtype=float)
        scores = X_star * resid_star[:, None]
        for g in unique_clusters:
            s_g = np.sum(scores[clusters == g], axis=0)
            M += np.outer(s_g, s_g)
        vcov = c * (inv_XtX @ M @ inv_XtX)
        n_clusters = G
    else:
        raise ValueError(f"Unsupported vce '{vce}' for xtreg, re.")

    se = np.sqrt(np.maximum(0, np.diag(vcov)))
    z_stat = np.where(se > 0, beta_re / se, 0.0)
    p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat)))
    z_crit = 1.959963984540054

    coef_names = list(indepvars) + ['_cons']
    coef_table = []
    for i, name in enumerate(coef_names):
        b = float(beta_re[i])
        s = float(se[i])
        z = float(z_stat[i])
        pv = float(p_val[i])
        ci_l = b - z_crit * s
        ci_u = b + z_crit * s
        coef_table.append({
            'name': name,
            'coef': b,
            'se': s,
            't': z,
            'z': z,
            'p': pv,
            'ci_lower': ci_l,
            'ci_upper': ci_u,
            'omitted': False
        })

    # Overall Wald Chi2 Test on non-constant coefficients
    if k > 0:
        inv_V_k = np.linalg.pinv(vcov[:k, :k])
        wald_chi2 = float(beta_re[:k].T @ inv_V_k @ beta_re[:k])
        prob_chi2 = float(1.0 - stats.chi2.cdf(wald_chi2, k))
    else:
        wald_chi2 = 0.0
        prob_chi2 = 1.0

    # R2 metrics: within, between, overall
    if k > 0:
        ss_within_tot = float(np.sum(y_dm ** 2))
        e_within = y_dm - X_dm @ beta_re[:k]
        r2_within = float(1.0 - (np.sum(e_within ** 2) / ss_within_tot)) if ss_within_tot > 0 else 0.0
        r2_within = max(0.0, min(1.0, r2_within))

        corr_b = np.corrcoef(y_means, X_means @ beta_re[:k])[0, 1]
        r2_between = float(corr_b ** 2) if not np.isnan(corr_b) else 0.0

        corr_o = np.corrcoef(y_raw, X_raw @ beta_re[:k])[0, 1]
        r2_overall = float(corr_o ** 2) if not np.isnan(corr_o) else 0.0
    else:
        r2_within = 0.0
        r2_between = 0.0
        r2_overall = 0.0

    min_obs = int(grp_sizes.min()) if len(grp_sizes) > 0 else 0
    avg_obs = float(grp_sizes.mean()) if len(grp_sizes) > 0 else 0.0
    max_obs = int(grp_sizes.max()) if len(grp_sizes) > 0 else 0

    return {
        'model_type': 'panel_re',
        'depvar': depvar,
        'indepvars': indepvars,
        'panelvar': panelvar,
        'timevar': timevar,
        'n': n,
        'num_panels': num_entities,
        'n_groups': num_entities,
        'df_m': k,
        'df_r': df_r,
        'chi2': wald_chi2,
        'prob_chi2': prob_chi2,
        'min_obs': min_obs,
        'avg_obs': avg_obs,
        'max_obs': max_obs,
        'r2_within': r2_within,
        'r2_between': r2_between,
        'r2_overall': r2_overall,
        'sigma_e': sigma_e,
        'rmse': float(np.sqrt(s2_star)),
        's2_star': s2_star,
        'sigma_u': sigma_u,
        'sigma': float(np.sqrt(s2_u + s2_e)),
        'rho': rho,
        'theta': float(np.mean(theta_per_entity)),
        'Tbar': T_bar,
        'Tcon': bool(min_obs == max_obs),
        'coefs': coef_table,
        'beta': beta_re,
        'vcov': vcov,
        'var_names': coef_names,
        'data_index': data.index,
        'resid': resid_star,
        'vce': vce,
        'n_clusters': n_clusters,
        'cluster_var': cluster_var
    }

def parse_linear_expr(expr_str, var_names):
    c = np.zeros(len(var_names), dtype=float)
    const = 0.0
    tokens = re.findall(r'([+-]?\s*[^+-]+)', expr_str)
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        sign = -1.0 if tok.startswith('-') else 1.0
        tok_clean = tok.lstrip('+-').strip()
        if '*' in tok_clean:
            mult_str, vname = tok_clean.split('*', 1)
            try:
                mult = float(mult_str.strip()) * sign
                vname = vname.strip()
            except ValueError:
                mult = float(vname.strip()) * sign
                vname = mult_str.strip()
        else:
            try:
                val = float(tok_clean) * sign
                const += val
                continue
            except ValueError:
                mult = sign
                vname = tok_clean
        if vname in var_names:
            c[var_names.index(vname)] += mult
        elif vname in ('_cons', 'Intercept') and any(cn in var_names for cn in ('_cons', 'Intercept')):
            cons_name = '_cons' if '_cons' in var_names else 'Intercept'
            c[var_names.index(cons_name)] += mult
        else:
            raise ValueError(f"Variable '{vname}' not in fitted model ({', '.join(var_names)}).")
    return c, const


def compute_wald_test(model_result, hypothesis):
    var_names = model_result['var_names']
    beta = np.asarray(model_result['beta'], dtype=float)
    vcov = np.asarray(model_result['vcov'], dtype=float)
    k_total = len(var_names)
    
    h = hypothesis.strip()
    parenthesized = re.findall(r'\((.*?)\)', h)
    if parenthesized:
        restrictions = [p.strip() for p in parenthesized if p.strip()]
    else:
        tokens = h.split()
        if "=" not in h:
            expanded_tokens = []
            for t in tokens:
                if t.startswith("i."):
                    base = t[2:]
                    matching = [col for col in var_names if col.endswith(f".{base}") or f".{base}#" in col or f"#{base}" in col]
                    if matching:
                        expanded_tokens.extend(matching)
                    else:
                        expanded_tokens.append(t)
                else:
                    expanded_tokens.append(t)
            if len(expanded_tokens) > 1 and all(t in var_names for t in expanded_tokens):
                restrictions = [f"{t} = 0" for t in expanded_tokens]
            elif len(expanded_tokens) == 1 and expanded_tokens[0] in var_names:
                restrictions = [f"{expanded_tokens[0]} = 0"]
            else:
                restrictions = [h]
        else:
            restrictions = [h]

    q = len(restrictions)
    R = np.zeros((q, k_total), dtype=float)
    r_vec = np.zeros(q, dtype=float)

    for j, rest in enumerate(restrictions):
        if "=" in rest:
            parts = rest.split("=", 1)
            lhs_str = parts[0].strip()
            rhs_str = parts[1].strip()
        else:
            lhs_str = rest.strip()
            rhs_str = "0"

        c_lhs, const_lhs = parse_linear_expr(lhs_str, var_names)
        c_rhs, const_rhs = parse_linear_expr(rhs_str, var_names)

        R[j, :] = c_lhs - c_rhs
        r_vec[j] = const_rhs - const_lhs

    # Equivalent constraints may be expressed in different units. Normalize
    # before checking consistency and selecting independent equations.
    scale = np.max(np.abs(R), axis=1)
    scale[scale == 0] = 1.0
    R /= scale[:, None]
    r_vec /= scale
    rank = int(np.linalg.matrix_rank(R))
    if rank < q:
        rhs_scale = np.max(np.abs(r_vec)) or 1.0
        if np.linalg.matrix_rank(np.column_stack((R, r_vec / rhs_scale))) > rank:
            raise ValueError("contradictory restrictions")
        if rank == 0:
            raise ValueError("hypothesis contains no independent restrictions")
        _, _, pivots = linalg.qr(R.T, mode='economic', pivoting=True)
        R = R[pivots[:rank]]
        r_vec = r_vec[pivots[:rank]]
        q = rank

    diff = R @ beta - r_vec
    R_V_Rt = R @ vcov @ R.T
    if np.linalg.matrix_rank(R_V_Rt) < q:
        raise ValueError("restriction covariance is singular")
    wald_stat = float(diff.T @ np.linalg.solve(R_V_Rt, diff))

    mtype = model_result.get('model_type', 'ols')
    is_small_sample_f = mtype in ('ols', 'panel_fe', 'reghdfe') or (mtype in ('ivregress', '2sls') and model_result.get('small', False))

    if is_small_sample_f:
        df_r = model_result.get('df_r', max(1, len(beta)))
        f_stat = float(wald_stat / q)
        p_val = float(stats.f.sf(f_stat, q, df_r))
        return {
            'test_type': 'F',
            'stat': f_stat,
            'df_num': q,
            'df_denom': df_r,
            'p_value': p_val,
            'hypothesis': hypothesis,
            'restrictions': restrictions
        }
    else:
        chi2_stat = float(wald_stat)
        p_val = float(stats.chi2.sf(chi2_stat, q))
        return {
            'test_type': 'chi2',
            'stat': chi2_stat,
            'df': q,
            'p_value': p_val,
            'hypothesis': hypothesis,
            'restrictions': restrictions
        }

def compute_tabulate_1way(df, varname, val_labels=None):
    s = df[varname].dropna()
    total = len(s)
    if total == 0:
        return {'var': varname, 'total': 0, 'rows': []}
        
    counts = s.value_counts(sort=False).sort_index()
    rows = []
    cum = 0.0
    for val, freq in counts.items():
        pct = (freq / total) * 100.0
        cum += pct
        lbl = val_labels.get(val, str(val)) if val_labels else str(val)
        rows.append({
            'value': val,
            'label': lbl,
            'freq': int(freq),
            'percent': float(pct),
            'cum_percent': float(cum)
        })
    return {'var': varname, 'total': total, 'rows': rows}

def compute_tabulate_2way(df, row_var, col_var, row_labels=None, col_labels=None):
    sub = df[[row_var, col_var]].dropna()
    ct = pd.crosstab(sub[row_var], sub[col_var], margins=True, margins_name="Total")
    return {
        'row_var': row_var,
        'col_var': col_var,
        'crosstab': ct
    }

def compute_correlate(df, varnames, covariance=False):
    sub = df[varnames].dropna()
    mat = sub.cov() if covariance else sub.corr()
    return {
        'vars': varnames,
        'n': len(sub),
        'corr': mat,
        'matrix': mat,
        'is_covariance': covariance
    }


def compute_svy_regression(df, depvar, indepvars, pweight=None, strata=None, psu=None, has_const=True):
    """
    Fits Linear Regression with Complex Survey Design via Taylor Series Linearization.
    Supports sampling weights (pweight), stratification (strata), and primary sampling units (psu).
    """
    needed_cols = [depvar] + indepvars
    if pweight and pweight not in needed_cols:
        needed_cols.append(pweight)
    if strata and strata not in needed_cols:
        needed_cols.append(strata)
    if psu and psu not in needed_cols:
        needed_cols.append(psu)

    data = df[needed_cols].dropna().copy()
    n = len(data)
    if n == 0:
        raise ValueError("no observations in estimation sample")

    y = data[depvar].to_numpy(dtype=float)
    k = len(indepvars)
    X_raw = data[indepvars].to_numpy(dtype=float) if k > 0 else np.empty((n, 0))

    if has_const:
        X = np.column_stack([X_raw, np.ones(n)])
        names = indepvars + ['_cons']
    else:
        X = X_raw
        names = list(indepvars)

    w = data[pweight].to_numpy(dtype=float) if pweight and pweight in data.columns else np.ones(n)
    pop_size = float(np.sum(w))

    XtWX = X.T @ (w[:, None] * X)
    try:
        inv_XtWX = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        inv_XtWX = np.linalg.pinv(XtWX)

    beta = inv_XtWX @ (X.T @ (w * y))
    y_pred = X @ beta
    resid = y - y_pred

    u = (w * resid)[:, None] * X

    strata_vec = data[strata].to_numpy() if strata and strata in data.columns else np.zeros(n, dtype=int)
    psu_vec = data[psu].to_numpy() if psu and psu in data.columns else np.arange(n)

    unique_strata = np.unique(strata_vec)
    total_strata = len(unique_strata)
    total_psus = 0
    V_u = np.zeros((X.shape[1], X.shape[1]), dtype=float)

    for h in unique_strata:
        idx_h = (strata_vec == h)
        psus_h = np.unique(psu_vec[idx_h])
        n_h = len(psus_h)
        total_psus += n_h
        if n_h <= 1:
            continue
        z_hj = []
        for j in psus_h:
            idx_hj = idx_h & (psu_vec == j)
            z_hj.append(np.sum(u[idx_hj], axis=0))
        z_hj = np.array(z_hj)
        z_bar = np.mean(z_hj, axis=0)
        dev = z_hj - z_bar
        S_h = (n_h / (n_h - 1.0)) * (dev.T @ dev)
        V_u += S_h

    df_survey = max(1, total_psus - total_strata)
    vcov = inv_XtWX @ V_u @ inv_XtWX

    y_wmean = np.sum(w * y) / pop_size
    ss_tot = np.sum(w * (y - y_wmean) ** 2)
    ss_res = np.sum(w * (resid ** 2))
    r2 = max(0.0, 1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    if k > 0:
        b_sub = beta[:k]
        V_sub = vcov[:k, :k]
        try:
            inv_Vsub = np.linalg.inv(V_sub)
            f_stat = float((b_sub.T @ inv_Vsub @ b_sub) / k)
            prob_f = float(1.0 - stats.f.cdf(f_stat, k, df_survey))
        except np.linalg.LinAlgError:
            f_stat = 0.0
            prob_f = 1.0
    else:
        f_stat = 0.0
        prob_f = 1.0

    se = np.sqrt(np.maximum(0.0, np.diag(vcov)))
    t_stat = np.where(se > 0, beta / se, 0.0)
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df_survey))
    t_crit = stats.t.ppf(0.975, df_survey)
    ci_lower = beta - t_crit * se
    ci_upper = beta + t_crit * se

    coef_table = []
    for i, name in enumerate(names):
        coef_table.append({
            'name': name,
            'coef': float(beta[i]),
            'se': float(se[i]),
            't': float(t_stat[i]),
            'p': float(p_val[i]),
            'ci_lower': float(ci_lower[i]),
            'ci_upper': float(ci_upper[i])
        })

    return {
        'model_type': 'svy_regress',
        'depvar': depvar,
        'indepvars': indepvars,
        'n': n,
        'pop_size': pop_size,
        'n_strata': total_strata,
        'n_psus': total_psus,
        'df_r': df_survey,
        'df_m': k,
        'f_stat': f_stat,
        'prob_f': prob_f,
        'r2': r2,
        'coefs': coef_table,
        'vcov': vcov,
        'pweight': pweight,
        'strata': strata,
        'psu': psu
    }

