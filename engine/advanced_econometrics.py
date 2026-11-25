"""
engine/advanced_econometrics.py - High-Dimensional Fixed Effects, Nonlinear MLE,
Quantile Regression, Marginal Effects, and Linear Combinations.

Implements:
1. reghdfe / absorb: High-Dimensional Multi-Way Fixed Effects via Method of Alternating Projections (MAP).
2. probit: Binary Probit Model (Newton-Raphson MLE with Robust & Cluster VCE).
3. poisson: Poisson Count Regression MLE (IRLS / Newton-Raphson with IRR option).
4. tobit: Censored Normal Regression (MLE with lower and upper limits).
5. qreg: Quantile Regression via HiGHS Linear Programming with Koenker-Bassett / Powell SE.
6. margins: Average Marginal Effects (AME) for Linear, Probit, Logit, and Poisson with Delta Method SE.
7. lincom: Linear Combinations of Estimators with Delta Method Variance and CIs.
"""

import re
import ast
import numpy as np
import pandas as pd
from scipy import stats, optimize, special
import scipy.linalg
from scipy.special import gammaln
from scipy.optimize import minimize
from engine.parser import _eval_ast_node, validate_ast, MAX_EXPR_LENGTH


# =====================================================================
# 1. High-Dimensional Fixed Effects (reghdfe / absorb)
# =====================================================================

def alternating_projections_demean(vec, factor_codes, factor_counts, max_iter=1000, tol=1e-8):
    """
    Demeans a 1D vector across pre-factored categorical factor codes
    using the Method of Alternating Projections (MAP).
    Guarantees convergence for any number of nested or non-nested fixed effects.
    """
    res = vec.astype(float).copy()
    n_factors = len(factor_codes)
    if n_factors == 0:
        return res

    for _ in range(max_iter):
        max_diff = 0.0
        for f_idx, inv in enumerate(factor_codes):
            counts = factor_counts[f_idx]
            sums = np.bincount(inv, weights=res, minlength=len(counts))
            means = sums / np.maximum(counts, 1)
            group_means = means[inv]
            
            res -= group_means
            diff = float(np.max(np.abs(group_means)))
            if diff > max_diff:
                max_diff = diff
        if max_diff < tol:
            break
    return res


def compute_reghdfe(df, depvar, indepvars, absorb_vars, vce='ols', cluster_var=None):
    """
    Fits High-Dimensional Multi-Way Fixed Effects OLS Regression.
    Projects out high-dimensional fixed effects via alternating projections (Frisch-Waugh-Lovell).
    """
    all_needed = [depvar] + indepvars + absorb_vars
    if cluster_var and cluster_var not in all_needed:
        all_needed.append(cluster_var)

    data = df[all_needed].dropna().copy()
    n = len(data)
    if n == 0:
        raise ValueError("no observations remaining after dropping missing values")

    k = len(indepvars)
    if k == 0:
        raise ValueError("reghdfe requires at least one independent regressor")

    # Factorize once outside the alternating-projections loop
    factor_codes = []
    factor_counts = []
    for f in absorb_vars:
        codes = pd.Categorical(data[f]).codes
        factor_codes.append(codes)
        factor_counts.append(np.bincount(codes))
    
    y_raw = data[depvar].to_numpy(dtype=float)
    y_tilde = alternating_projections_demean(y_raw, factor_codes, factor_counts)

    X_tilde = np.zeros((n, k), dtype=float)
    col_raws = []
    for col_idx, var in enumerate(indepvars):
        col_raw = data[var].to_numpy(dtype=float)
        col_raws.append(col_raw)
        X_tilde[:, col_idx] = alternating_projections_demean(col_raw, factor_codes, factor_counts)

    # 1. Collinearity check against absorbed fixed effects
    cand_indices = []
    for col_idx in range(k):
        col_norm = float(np.linalg.norm(X_tilde[:, col_idx]))
        col_raw = col_raws[col_idx]
        raw_scale = max(float(np.linalg.norm(col_raw - np.mean(col_raw))), 1.0)
        if (col_norm / raw_scale) >= 1e-7 and col_norm >= 1e-10:
            cand_indices.append(col_idx)

    if not cand_indices:
        raise ValueError("All independent variables are collinear with the absorbed fixed effects.")

    # 2. QR pivoting among non-zero demeaned regressors
    if len(cand_indices) == 1:
        kept_indices = cand_indices
    else:
        X_cand = X_tilde[:, cand_indices]
        _, R, P = scipy.linalg.qr(X_cand, mode='economic', pivoting=True)
        diag_R = np.abs(np.diag(R))
        tol = 1e-7 * (diag_R[0] if len(diag_R) > 0 else 1.0)
        rank = int(np.sum(diag_R > tol))
        if rank == 0:
            raise ValueError("All independent variables are collinear with the absorbed fixed effects.")
        kept_in_cand = set(P[:rank])
        kept_indices = [cand_indices[i] for i in range(len(cand_indices)) if i in kept_in_cand]

    k_kept = len(kept_indices)
    X_tilde_kept = X_tilde[:, kept_indices]

    # Solve OLS on kept demeaned regressors using QR decomposition
    if k_kept > 0:
        Q_k, R_k = scipy.linalg.qr(X_tilde_kept, mode='economic')
        beta_kept = scipy.linalg.solve_triangular(R_k, Q_k.T @ y_tilde)
        R_inv = scipy.linalg.solve_triangular(R_k, np.eye(k_kept))
        inv_XtX_kept = R_inv @ R_inv.T
    else:
        beta_kept = np.array([])
        inv_XtX_kept = np.zeros((0, 0))

    y_hat = X_tilde_kept @ beta_kept if k_kept > 0 else np.zeros(n)
    resid = y_tilde - y_hat

    # Calculate degrees of freedom absorbed:
    df_absorbed = 1
    for f in factor_codes:
        df_absorbed += (len(np.unique(f)) - 1)

    df_r = max(1, n - k_kept - df_absorbed)
    ss_r = float(np.sum(resid ** 2))
    
    y_raw_mean = np.mean(y_raw)
    ss_t_raw = float(np.sum((y_raw - y_raw_mean) ** 2))
    r2_overall = max(0.0, 1.0 - (ss_r / ss_t_raw)) if ss_t_raw > 0 else 0.0

    ss_t_within = float(np.sum(y_tilde ** 2))
    r2_within = max(0.0, 1.0 - (ss_r / ss_t_within)) if ss_t_within > 0 else 0.0

    # Variance-covariance matrix
    vce = vce.lower() if vce else 'ols'
    n_clusters = None

    if vce in ('robust', 'hc1'):
        scores = X_tilde_kept * resid[:, None]
        omega = scores.T @ scores
        c = (n / max(1, n - k_kept - df_absorbed))
        vcov_kept = c * (inv_XtX_kept @ omega @ inv_XtX_kept)
    elif vce == 'cluster':
        if not cluster_var or cluster_var not in data.columns:
            raise ValueError(f"cluster variable '{cluster_var}' not found")
        clusters = data[cluster_var].to_numpy()
        unq_clust = np.unique(clusters)
        G = len(unq_clust)
        if G <= 1:
            raise ValueError(f"cluster variable '{cluster_var}' must have at least 2 distinct clusters")
        n_clusters = G
        df_r = max(1, G - 1)
        
        meat = np.zeros((k_kept, k_kept), dtype=float)
        for cl in unq_clust:
            idx = (clusters == cl)
            u_g = resid[idx]
            X_g = X_tilde_kept[idx, :]
            s_g = X_g.T @ u_g
            meat += np.outer(s_g, s_g)
        
        q = (G / max(1, G - 1)) * ((n - 1) / max(1, n - k_kept - df_absorbed))
        vcov_kept = q * (inv_XtX_kept @ meat @ inv_XtX_kept)
    elif vce == 'ols':
        sigma2 = ss_r / df_r
        vcov_kept = sigma2 * inv_XtX_kept
    else:
        raise ValueError(f"Unsupported vce '{vce}' for reghdfe. Allowed: 'ols', 'robust', 'cluster'.")

    se_kept = np.sqrt(np.maximum(0, np.diag(vcov_kept)))
    t_kept = np.where(se_kept > 0, beta_kept / se_kept, 0.0)
    p_kept = 2.0 * (1.0 - stats.t.cdf(np.abs(t_kept), df_r))

    t_crit = stats.t.ppf(0.975, df_r)
    ci_lower_kept = beta_kept - t_crit * se_kept
    ci_upper_kept = beta_kept + t_crit * se_kept

    # F-statistic for the kept regressors
    try:
        f_stat = float((beta_kept.T @ np.linalg.inv(vcov_kept) @ beta_kept) / k_kept)
        prob_f = float(1.0 - stats.f.cdf(f_stat, k_kept, df_r))
    except Exception:
        f_stat = 0.0
        prob_f = 1.0

    # Reconstruct full beta, vcov, and coef_table
    beta_full = np.zeros(k, dtype=float)
    vcov_full = np.zeros((k, k), dtype=float)
    for idx_k, orig_idx in enumerate(kept_indices):
        beta_full[orig_idx] = beta_kept[idx_k]
        for idx_k2, orig_idx2 in enumerate(kept_indices):
            vcov_full[orig_idx, orig_idx2] = vcov_kept[idx_k, idx_k2]

    coef_table = []
    for i, name in enumerate(indepvars):
        if i in kept_indices:
            k_idx = kept_indices.index(i)
            coef_table.append({
                'name': name,
                'coef': float(beta_kept[k_idx]),
                'se': float(se_kept[k_idx]),
                't': float(t_kept[k_idx]),
                'p': float(p_kept[k_idx]),
                'ci_lower': float(ci_lower_kept[k_idx]),
                'ci_upper': float(ci_upper_kept[k_idx]),
                'omitted': False
            })
        else:
            coef_table.append({
                'name': name,
                'coef': 0.0,
                'se': 0.0,
                't': 0.0,
                'p': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'omitted': True
            })

    return {
        'model_type': 'reghdfe',
        'depvar': depvar,
        'indepvars': indepvars,
        'absorb_vars': absorb_vars,
        'n': n,
        'k': k_kept,
        'df_r': df_r,
        'df_absorbed': df_absorbed,
        'r2_within': r2_within,
        'r2_overall': r2_overall,
        'r2': r2_within,
        'f_stat': f_stat,
        'prob_f': prob_f,
        'coefs': coef_table,
        'beta': beta_full,
        'vcov': vcov_full,
        'var_names': indepvars,
        'data_index': data.index,
        'resid': resid,
        'vce': vce,
        'cluster_var': cluster_var,
        'n_clusters': n_clusters
    }


# =====================================================================
# 2. Binary Probit Model (Newton-Raphson MLE)
# =====================================================================

def compute_probit(df, depvar, indepvars, vce='mle', cluster_var=None, max_iter=100, tol=1e-7):
    """
    Fits Binary Probit Regression via Newton-Raphson Maximum Likelihood Estimation.
    P(y = 1 | X) = Phi(X beta).
    """
    all_needed = [depvar] + indepvars
    if cluster_var and cluster_var not in all_needed:
        all_needed.append(cluster_var)

    data = df[all_needed].dropna().copy()
    n = len(data)
    if n == 0:
        raise ValueError("no observations remaining after dropping missing values")

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
    unq_y = np.unique(y)
    if not np.all(np.isin(unq_y, [0, 1])):
        raise ValueError(f"dependent variable '{depvar}' must be binary 0/1 for probit")

    k = len(indepvars)
    X = np.column_stack([data[indepvars].to_numpy(dtype=float), np.ones(n)])
    names = indepvars + ['_cons']
    n_params = len(names)

    # Initial estimates via OLS scaled
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ (y - 0.5) * 2.5)

    # Constant-only model log-likelihood
    p_bar = np.clip(np.mean(y), 1e-6, 1.0 - 1e-6)
    ll_0 = float(np.sum(y * np.log(p_bar) + (1.0 - y) * np.log(1.0 - p_bar)))

    def _compute_probit_ll(b):
        eta_val = np.clip(X @ b, -8.0, 8.0)
        Phi_val = np.clip(stats.norm.cdf(eta_val), 1e-15, 1.0 - 1e-15)
        return float(np.sum(y * np.log(Phi_val) + (1.0 - y) * np.log(1.0 - Phi_val)))

    converged = False
    status = "Iteration limit reached without convergence"
    warnings_list = []
    current_ll = _compute_probit_ll(beta)
    ll_history = [float(current_ll)]
    iteration = 0
    step_norm = 0.0

    eta = np.clip(X @ beta, -8.0, 8.0)
    phi = stats.norm.pdf(eta)
    Phi = np.clip(stats.norm.cdf(eta), 1e-15, 1.0 - 1e-15)
    lambda_i = np.where(y == 1, phi / Phi, -phi / (1.0 - Phi))
    grad = X.T @ lambda_i
    grad_norm = float(np.linalg.norm(grad))

    if grad_norm < tol:
        converged = True
        status = "Convergence achieved (initial gradient within tolerance)"
    else:
        for it in range(max_iter):
            iteration = it + 1

            # Check separation
            if n >= 4:
                p_pos = Phi[y == 1]
                p_neg = Phi[y == 0]
                if len(p_pos) > 0 and len(p_neg) > 0 and np.all(p_pos > 0.999) and np.all(p_neg < 0.001):
                    status = "Complete or quasi-complete separation detected"
                    warnings_list.append("Complete or quasi-complete separation detected; inference is invalid.")
                    converged = False
                    break

            # Hessian: H = - X' W X where W = lambda_i * (lambda_i + eta)
            w = lambda_i * (lambda_i + eta)
            w = np.clip(w, 1e-12, 1e12)
            H = - (X.T * w) @ X

            try:
                step = np.linalg.solve(-H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(-H) @ grad
                warnings_list.append("Ill-conditioned Hessian.")

            dir_deriv = float(np.dot(grad, step))
            if dir_deriv <= 0:
                step = grad / (grad_norm + 1e-12)
                dir_deriv = float(np.dot(grad, step))

            # Backtracking Armijo line search for log-likelihood maximization
            alpha = 1.0
            step_accepted = False
            for _ in range(10):
                beta_trial = beta + alpha * step
                trial_ll = _compute_probit_ll(beta_trial)
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

            # Re-evaluate gradient at accepted parameters
            eta = np.clip(X @ beta, -8.0, 8.0)
            phi = stats.norm.pdf(eta)
            Phi = np.clip(stats.norm.cdf(eta), 1e-15, 1.0 - 1e-15)
            lambda_i = np.where(y == 1, phi / Phi, -phi / (1.0 - Phi))
            grad = X.T @ lambda_i
            grad_norm = float(np.linalg.norm(grad))

            if step_norm < tol or grad_norm < tol:
                converged = True
                status = "Convergence achieved (gradient and step within tolerance)"
                break

    if not converged and not any("separation" in w.lower() for w in warnings_list):
        warnings_list.append("Model did not achieve convergence. Standard errors and inference should be treated as exploratory.")

    # Final log-likelihood and Hessian
    eta = np.clip(X @ beta, -8.0, 8.0)
    Phi = np.clip(stats.norm.cdf(eta), 1e-15, 1.0 - 1e-15)
    ll = float(np.sum(y * np.log(Phi) + (1.0 - y) * np.log(1.0 - Phi)))

    phi = stats.norm.pdf(eta)
    lambda_i = np.where(y == 1, phi / Phi, -phi / (1.0 - Phi))
    w = np.clip(lambda_i * (lambda_i + eta), 1e-12, 1e12)
    H = - (X.T * w) @ X
    inv_H = np.linalg.inv(-H) if np.linalg.cond(H) < 1e12 else np.linalg.pinv(-H)

    vce = vce.lower() if vce else 'mle'
    n_clusters = None
    df_vce = n - n_params

    if vce == 'mle':
        vcov = inv_H
    elif vce in ('robust', 'hc1'):
        scores = X * lambda_i[:, None]
        omega = scores.T @ scores
        c = n / max(1, n - 1)
        vcov = c * (inv_H @ omega @ inv_H)
    elif vce == 'cluster':
        if not cluster_var or cluster_var not in data.columns:
            raise ValueError(f"cluster variable '{cluster_var}' not found")
        clusters = data[cluster_var].to_numpy()
        scores = X * lambda_i[:, None]
        from engine.estimation import compute_mle_vcov_cluster
        vcov, n_clusters = compute_mle_vcov_cluster(inv_H, scores, clusters)
        df_vce = n_clusters - 1
    else:
        raise ValueError(f"Unsupported vce '{vce}' for probit. Allowed: 'mle', 'robust', 'cluster'.")

    se = np.sqrt(np.maximum(0, np.diag(vcov)))
    z_stat = np.where(se > 0, beta / se, 0.0)
    p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat)))
    z_crit = stats.norm.ppf(0.975)
    ci_lower = beta - z_crit * se
    ci_upper = beta + z_crit * se

    lr_chi2 = max(0.0, 2.0 * (ll - ll_0))
    prob_chi2 = float(1.0 - stats.chi2.cdf(lr_chi2, k))
    pseudo_r2 = max(0.0, 1.0 - (ll / ll_0)) if ll_0 != 0 else 0.0

    coef_table = []
    for i, name in enumerate(names):
        coef_table.append({
            'name': name,
            'coef': float(beta[i]),
            'se': float(se[i]),
            'z': float(z_stat[i]),
            'p': float(p_val[i]),
            'ci_lower': float(ci_lower[i]),
            'ci_upper': float(ci_upper[i]),
            'is_odds_ratio': False
        })

    return {
        'model_type': 'probit',
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
        'p_pred': Phi,
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


# =====================================================================
# 3. Poisson Count Regression MLE
# =====================================================================

def compute_poisson(df, depvar, indepvars, irr=False, vce='mle', cluster_var=None, max_iter=100, tol=1e-7):
    """
    Fits Poisson Count Model via Iteratively Reweighted Least Squares / Newton-Raphson.
    E[y | X] = exp(X beta).
    """
    all_needed = [depvar] + indepvars
    if cluster_var and cluster_var not in all_needed:
        all_needed.append(cluster_var)

    data = df[all_needed].dropna().copy()
    n = len(data)
    if n == 0:
        raise ValueError("no observations remaining after dropping missing values")

    y = data[depvar].to_numpy(dtype=float)
    if np.any(y < 0):
        raise ValueError(f"dependent variable '{depvar}' must be non-negative count data")

    k = len(indepvars)
    X = np.column_stack([data[indepvars].to_numpy(dtype=float), np.ones(n)])
    names = indepvars + ['_cons']
    n_params = len(names)

    # Initial beta from log(y + 0.5)
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ np.log(y + 0.5))

    # Constant-only log-likelihood
    y_mean = max(np.mean(y), 1e-6)
    ll_0 = float(np.sum(y * np.log(y_mean) - y_mean - gammaln(y + 1.0)))

    def _compute_poisson_ll(b):
        eta_val = np.clip(X @ b, -30.0, 30.0)
        mu_val = np.exp(eta_val)
        return float(np.sum(y * eta_val - mu_val - gammaln(y + 1.0)))

    converged = False
    status = "Iteration limit reached without convergence"
    warnings_list = []
    current_ll = _compute_poisson_ll(beta)
    ll_history = [float(current_ll)]
    iteration = 0
    step_norm = 0.0

    eta = np.clip(X @ beta, -30.0, 30.0)
    mu = np.exp(eta)
    grad = X.T @ (y - mu)
    grad_norm = float(np.linalg.norm(grad))

    if grad_norm < tol:
        converged = True
        status = "Convergence achieved (initial gradient within tolerance)"
    else:
        for it in range(max_iter):
            iteration = it + 1

            # Hessian: - X' diag(mu) X
            H = - (X.T * mu) @ X

            try:
                step = np.linalg.solve(-H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(-H) @ grad
                warnings_list.append("Ill-conditioned Hessian in Poisson.")

            dir_deriv = float(np.dot(grad, step))
            if dir_deriv <= 0:
                step = grad / (grad_norm + 1e-12)
                dir_deriv = float(np.dot(grad, step))

            # Backtracking Armijo line search for log-likelihood maximization
            alpha = 1.0
            step_accepted = False
            for _ in range(10):
                beta_trial = beta + alpha * step
                trial_ll = _compute_poisson_ll(beta_trial)
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

            # Re-evaluate gradient at accepted parameters
            eta = np.clip(X @ beta, -30.0, 30.0)
            mu = np.exp(eta)
            grad = X.T @ (y - mu)
            grad_norm = float(np.linalg.norm(grad))

            if step_norm < tol or grad_norm < tol:
                converged = True
                status = "Convergence achieved (gradient and step within tolerance)"
                break

    if not converged and not any("separation" in w.lower() for w in warnings_list):
        warnings_list.append("Model did not achieve convergence. Standard errors and inference should be treated as exploratory.")

    # Final mu and log-likelihood
    eta = np.clip(X @ beta, -30.0, 30.0)
    mu = np.exp(eta)
    ll = float(np.sum(y * eta - mu - gammaln(y + 1.0)))

    H = - (X.T * mu) @ X
    inv_H = np.linalg.inv(-H) if np.linalg.cond(H) < 1e12 else np.linalg.pinv(-H)

    vce = vce.lower() if vce else 'mle'
    n_clusters = None
    df_vce = n - n_params

    if vce == 'mle':
        vcov = inv_H
    elif vce in ('robust', 'hc1'):
        scores = X * (y - mu)[:, None]
        omega = scores.T @ scores
        c = n / max(1, n - 1)
        vcov = c * (inv_H @ omega @ inv_H)
    elif vce == 'cluster':
        if not cluster_var or cluster_var not in data.columns:
            raise ValueError(f"cluster variable '{cluster_var}' not found")
        clusters = data[cluster_var].to_numpy()
        scores = X * (y - mu)[:, None]
        from engine.estimation import compute_mle_vcov_cluster
        vcov, n_clusters = compute_mle_vcov_cluster(inv_H, scores, clusters)
        df_vce = n_clusters - 1
    else:
        raise ValueError(f"Unsupported vce '{vce}' for poisson. Allowed: 'mle', 'robust', 'cluster'.")

    se = np.sqrt(np.maximum(0, np.diag(vcov)))
    z_stat = np.where(se > 0, beta / se, 0.0)
    p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat)))
    z_crit = stats.norm.ppf(0.975)
    ci_lower = beta - z_crit * se
    ci_upper = beta + z_crit * se

    lr_chi2 = max(0.0, 2.0 * (ll - ll_0))
    prob_chi2 = float(1.0 - stats.chi2.cdf(lr_chi2, k))
    pseudo_r2 = max(0.0, 1.0 - (ll / ll_0)) if ll_0 != 0 else 0.0

    coef_table = []
    for i, name in enumerate(names):
        b = float(beta[i])
        s = float(se[i])
        z = float(z_stat[i])
        pv = float(p_val[i])
        ci_l = float(ci_lower[i])
        ci_u = float(ci_upper[i])

        if irr and name != '_cons':
            irr_val = float(np.exp(np.clip(b, -100, 100)))
            irr_se = float(irr_val * s)
            ci_l_irr = float(np.exp(np.clip(ci_l, -100, 100)))
            ci_u_irr = float(np.exp(np.clip(ci_u, -100, 100)))
            coef_table.append({
                'name': name,
                'coef': irr_val,
                'se': irr_se,
                'z': z,
                'p': pv,
                'ci_lower': ci_l_irr,
                'ci_upper': ci_u_irr,
                'is_irr': True
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
                'is_irr': False
            })

    return {
        'model_type': 'poisson',
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
        'mu_pred': mu,
        'irr': irr,
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


# =====================================================================
# 4. Censored Normal Regression (Tobit Model)
# =====================================================================

def compute_tobit(df, depvar, indepvars, ll=None, ul=None, vce='mle'):
    """
    Fits Censored Normal Tobit Regression with lower limit ll and/or upper limit ul.
    Estimates beta and sigma simultaneously via Maximum Likelihood.
    """
    all_needed = [depvar] + indepvars
    data = df[all_needed].dropna().copy()
    n = len(data)
    if n == 0:
        raise ValueError("no observations remaining after dropping missing values")

    y = data[depvar].to_numpy(dtype=float)
    k = len(indepvars)
    X = np.column_stack([data[indepvars].to_numpy(dtype=float), np.ones(n)])
    names = indepvars + ['_cons']
    n_params = len(names)

    # Determine censoring indicator masks
    is_left = (y <= ll) if ll is not None else np.zeros(n, dtype=bool)
    is_right = (y >= ul) if ul is not None else np.zeros(n, dtype=bool)
    is_uncensored = ~is_left & ~is_right

    n_left = int(np.sum(is_left))
    n_right = int(np.sum(is_right))
    n_uncensored = int(np.sum(is_uncensored))

    # Starting values via uncensored OLS
    beta_init = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    res_init = y - X @ beta_init
    sigma_init = max(float(np.std(res_init)), 0.1)

    # Initial parameter vector: [beta_0, ..., beta_k, log_sigma]
    theta_0 = np.append(beta_init, np.log(sigma_init))

    def tobit_neg_loglik(theta):
        b = theta[:-1]
        sigma = np.exp(theta[-1])
        xb = X @ b

        val = 0.0
        # 1. Uncensored
        if n_uncensored > 0:
            z_u = (y[is_uncensored] - xb[is_uncensored]) / sigma
            val += np.sum(np.log(sigma) + 0.5 * np.log(2.0 * np.pi) + 0.5 * (z_u ** 2))

        # 2. Left-censored
        if n_left > 0:
            z_l = (ll - xb[is_left]) / sigma
            Phi_l = np.clip(stats.norm.cdf(z_l), 1e-15, 1.0 - 1e-15)
            val -= np.sum(np.log(Phi_l))

        # 3. Right-censored
        if n_right > 0:
            z_r = (ul - xb[is_right]) / sigma
            Phi_r = np.clip(stats.norm.sf(z_r), 1e-15, 1.0 - 1e-15)
            val -= np.sum(np.log(Phi_r))

        return val

    opt_res = minimize(tobit_neg_loglik, theta_0, method='L-BFGS-B', tol=1e-12)
    theta_hat = opt_res.x
    beta_hat = theta_hat[:-1]
    sigma_hat = float(np.exp(theta_hat[-1]))
    ll_val = float(-opt_res.fun)

    # Variance-covariance matrix from numerical Hessian at the optimum
    p_len = len(theta_hat)
    hess = np.zeros((p_len, p_len))
    eps = 1e-5
    for i in range(p_len):
        for j in range(p_len):
            ei = np.zeros(p_len)
            ej = np.zeros(p_len)
            ei[i] = eps
            ej[j] = eps
            fpp = tobit_neg_loglik(theta_hat + ei + ej)
            fpm = tobit_neg_loglik(theta_hat + ei - ej)
            fmp = tobit_neg_loglik(theta_hat - ei + ej)
            fmm = tobit_neg_loglik(theta_hat - ei - ej)
            hess[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)

    inv_H = np.linalg.inv(hess) if np.linalg.cond(hess) < 1e12 else np.linalg.pinv(hess)

    vce = vce.lower() if vce else 'mle'
    if vce in ('robust', 'hc1'):
        grad_obs = np.zeros((n, p_len))
        for obs_i in range(n):
            def obs_nll(th, idx=obs_i):
                b = th[:-1]
                sig = np.exp(th[-1])
                xbi = float(np.dot(X[idx], b))
                yi = float(y[idx])
                if is_left[idx]:
                    z = (ll - xbi) / sig
                    return -float(np.log(np.clip(stats.norm.cdf(z), 1e-15, 1.0)))
                elif is_right[idx]:
                    z = (ul - xbi) / sig
                    return -float(np.log(np.clip(stats.norm.sf(z), 1e-15, 1.0)))
                else:
                    z = (yi - xbi) / sig
                    return float(np.log(sig) + 0.5 * np.log(2.0 * np.pi) + 0.5 * (z ** 2))
            for pj in range(p_len):
                ej = np.zeros(p_len)
                ej[pj] = eps
                grad_obs[obs_i, pj] = (obs_nll(theta_hat + ej) - obs_nll(theta_hat - ej)) / (2.0 * eps)
        omega = grad_obs.T @ grad_obs
        c = n / max(1, n - 1)
        vcov_theta = c * (inv_H @ omega @ inv_H)
    elif vce == 'mle':
        vcov_theta = inv_H
    else:
        raise ValueError(f"Unsupported vce '{vce}' for tobit. Allowed: 'mle', 'robust'.")

    # Delta method for sigma: Var(sigma) = Var(log_sigma) * sigma^2
    vcov_beta = vcov_theta[:n_params, :n_params]
    var_log_sigma = vcov_theta[-1, -1]
    se_sigma = float(np.sqrt(max(0.0, var_log_sigma)) * sigma_hat)

    se_beta = np.sqrt(np.maximum(0, np.diag(vcov_beta)))
    t_stat = np.where(se_beta > 0, beta_hat / se_beta, 0.0)
    p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(t_stat)))
    z_crit = stats.norm.ppf(0.975)
    ci_lower = beta_hat - z_crit * se_beta
    ci_upper = beta_hat + z_crit * se_beta

    coef_table = []
    for i, name in enumerate(names):
        coef_table.append({
            'name': name,
            'coef': float(beta_hat[i]),
            'se': float(se_beta[i]),
            'z': float(t_stat[i]),
            'p': float(p_val[i]),
            'ci_lower': float(ci_lower[i]),
            'ci_upper': float(ci_upper[i])
        })

    # Append /sigma with real p-value
    z_sigma = sigma_hat / se_sigma if se_sigma > 0 else 0.0
    p_sigma = float(2.0 * stats.norm.sf(np.abs(z_sigma)))
    coef_table.append({
        'name': '/sigma',
        'coef': sigma_hat,
        'se': se_sigma,
        'z': z_sigma,
        'p': p_sigma,
        'ci_lower': max(0.0, sigma_hat - z_crit * se_sigma),
        'ci_upper': sigma_hat + z_crit * se_sigma
    })

    return {
        'model_type': 'tobit',
        'depvar': depvar,
        'indepvars': indepvars,
        'n': n,
        'k': k,
        'll_val': ll_val,
        'll': ll_val,
        'n_uncensored': n_uncensored,
        'n_left': n_left,
        'n_right': n_right,
        'll_limit': ll,
        'ul_limit': ul,
        'coefs': coef_table,
        'beta': beta_hat,
        'sigma': sigma_hat,
        'converged': bool(opt_res.success),
        'convergence_status': str(opt_res.message),
        'iterations': int(opt_res.nit),
        'max_iter': 1000,
        'gradient_norm': float(np.linalg.norm(opt_res.jac)) if hasattr(opt_res, 'jac') and opt_res.jac is not None else 0.0,
        'warnings': [] if opt_res.success else [str(opt_res.message)],
        'vcov': vcov_beta,
        'var_names': names,
        'data_index': data.index
    }


# =====================================================================
# 5. Quantile Regression (qreg via HiGHS Linear Programming)
# =====================================================================

def compute_qreg(df, depvar, indepvars, quantile=0.5, vce='iid', reps=200):
    """
    Fits Quantile Regression for any quantile q in (0, 1) using Iteratively
    Reweighted Least Squares (IRLS) scaling to 100k+ rows without dense matrices.
    Computes standard errors via Hall-Sheather Epanechnikov sparsity estimation (vcov='iid')
    or pairs bootstrap (vcov='bootstrap').
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be strictly between 0 and 1")

    all_needed = [depvar] + indepvars
    data = df[all_needed].dropna().copy()
    n = len(data)
    if n == 0:
        raise ValueError("no observations remaining after dropping missing values")

    y = data[depvar].to_numpy(dtype=float)
    X = np.column_stack([data[indepvars].to_numpy(dtype=float), np.ones(n)])
    names = indepvars + ['_cons']
    p = len(names)

    # Iteratively Reweighted Least Squares (IRLS) solver (Koenker-Bassett / statsmodels QuantReg)
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    max_qreg_iter = 2000
    converged = False
    step_norm = 0.0
    actual_iters = 0
    for it in range(max_qreg_iter):
        actual_iters = it + 1
        beta_prev = beta.copy()
        resid = y - X @ beta
        r_abs = np.maximum(np.abs(resid), 1e-6)
        w = np.where(resid < 0, (1.0 - quantile) / r_abs, quantile / r_abs)
        Xw = X * np.sqrt(w[:, None])
        yw = y * np.sqrt(w)
        try:
            beta = np.linalg.inv(Xw.T @ Xw) @ (Xw.T @ yw)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(Xw.T @ Xw) @ (Xw.T @ yw)
        step_norm = float(np.max(np.abs(beta - beta_prev)))
        if step_norm < 1e-9:
            converged = True
            break

    resid = y - X @ beta
    inv_XtX = np.linalg.pinv(X.T @ X)

    vce = vce.lower() if vce else 'iid'
    if vce in ('iid', 'sparsity'):
        # Hall-Sheather bandwidth & Epanechnikov kernel matching statsmodels QuantReg
        iqre = float(stats.scoreatpercentile(resid, 75) - stats.scoreatpercentile(resid, 25))
        z = stats.norm.ppf(quantile)
        h_val = (n ** (-1.0 / 3.0)) * (stats.norm.ppf(0.975) ** (2.0 / 3.0)) * (
            ((1.5 * (stats.norm.pdf(z) ** 2)) / (2.0 * (z ** 2) + 1.0)) ** (1.0 / 3.0)
        )
        h = min(float(np.std(y)), iqre / 1.34) * (stats.norm.ppf(quantile + h_val) - stats.norm.ppf(quantile - h_val))
        h = max(h, 1e-6)
        u = resid / h
        kern = np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u ** 2), 0.0)
        fhat0 = float(np.sum(kern) / (n * h))
        fhat0 = max(fhat0, 1e-6)

        # Correct asymptotic covariance: V = (q * (1 - q) / fhat0^2) * (X' X)^(-1)
        vcov = (quantile * (1.0 - quantile) / (fhat0 ** 2)) * inv_XtX
    elif vce == 'bootstrap':
        # Pairs bootstrap
        boot_betas = np.zeros((reps, p))
        rng = np.random.default_rng(42)
        for b_i in range(reps):
            boot_idx = rng.integers(0, n, size=n)
            X_b, y_b = X[boot_idx], y[boot_idx]
            b_b = np.linalg.pinv(X_b.T @ X_b) @ (X_b.T @ y_b)
            for _ in range(50):
                b_prev = b_b.copy()
                r_b = y_b - X_b @ b_b
                r_abs = np.maximum(np.abs(r_b), 1e-6)
                w_b = np.where(r_b < 0, (1.0 - quantile) / r_abs, quantile / r_abs)
                Xw_b = X_b * np.sqrt(w_b[:, None])
                yw_b = y_b * np.sqrt(w_b)
                b_b = np.linalg.pinv(Xw_b.T @ Xw_b) @ (Xw_b.T @ yw_b)
                if np.max(np.abs(b_b - b_prev)) < 1e-4:
                    break
            boot_betas[b_i] = b_b
        vcov = np.cov(boot_betas, rowvar=False)
    else:
        raise ValueError(f"Unsupported vce '{vce}' for qreg. Allowed: 'iid', 'bootstrap'.")

    se = np.sqrt(np.maximum(0, np.diag(vcov)))
    df_r = max(1, n - p)
    t_stat = np.where(se > 0, beta / se, 0.0)
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df_r))
    t_crit = stats.t.ppf(0.975, df_r)
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
        'model_type': 'qreg',
        'depvar': depvar,
        'indepvars': indepvars,
        'quantile': quantile,
        'n': n,
        'df_r': df_r,
        'converged': converged,
        'convergence_status': "converged" if converged else "maximum iterations reached without convergence",
        'iterations': actual_iters,
        'max_iter': max_qreg_iter,
        'gradient_norm': step_norm,
        'warnings': [] if converged else ["Maximum iterations reached without convergence."],
        'coefs': coef_table,
        'beta': beta,
        'vcov': vcov,
        'var_names': names,
        'data_index': data.index,
        'resid': resid
    }


# =====================================================================
# 6. Post-Estimation: Average Marginal Effects (margins)
# =====================================================================

def compute_margins(model_result, df, dydx_vars=None):
    """
    Computes Average Marginal Effects (AME) with Delta Method Standard Errors.
    Supported model types: ols, reghdfe, probit, logit, poisson.
    """
    mtype = model_result.get('model_type')
    var_names = model_result.get('var_names')
    beta = model_result.get('beta')
    vcov = model_result.get('vcov')
    depvar = model_result.get('depvar')
    n = model_result['n']
    if any('#' in v for v in var_names):
        raise ValueError("margins for models with interactions is not supported")
    linear = mtype in ('ols', 'reghdfe', 'panel_fe')
    if not linear:
        if mtype not in ('probit', 'logit', 'poisson'):
            raise ValueError(f"margins not supported for model type: {mtype}")
        if 'fitted_design' not in model_result:
            raise ValueError("refit the model to recover its estimation sample for margins")
        X_mat = np.asarray(model_result['fitted_design'], dtype=float)

    if dydx_vars is None or dydx_vars == ['*']:
        dydx_vars = [v for v in var_names if v != '_cons']
    else:
        expanded_dydx = []
        for v in dydx_vars:
            if v.startswith("i."):
                base = v[2:]
                matching = [col for col in var_names if col.endswith(f".{base}") and not col.startswith("c.")]
                if matching:
                    expanded_dydx.extend(matching)
                else:
                    expanded_dydx.append(v)
            elif v not in var_names and f'c.{v}' in var_names:
                expanded_dydx.append(f'c.{v}')
            else:
                expanded_dydx.append(v)
        dydx_vars = expanded_dydx

    margins_table = []

    for target_var in dydx_vars:
        if target_var not in var_names:
            raise ValueError(f"variable '{target_var}' not found in fitted model")
        k_idx = var_names.index(target_var)
        b_k = beta[k_idx]
        is_indicator = (not linear and not target_var.startswith('c.')
                        and np.all(np.isin(X_mat[:, k_idx], [0.0, 1.0])))
        if is_indicator:
            X0 = X_mat.copy()
            X0[:, k_idx] = 0.0
            if '.' in target_var:
                # A factor contrast sets all its indicators to the omitted level first.
                suffix = '.' + target_var.rsplit('.', 1)[1]
                for j, name in enumerate(var_names):
                    if name.endswith(suffix) and not name.startswith('c.'):
                        X0[:, j] = 0.0
            X1 = X0.copy()
            X1[:, k_idx] = 1.0

        if linear:
            # Linear model: dE[y]/dx = beta_k
            ame = float(b_k)
            se_ame = float(np.sqrt(vcov[k_idx, k_idx]))
        elif mtype == 'probit':
            if is_indicator:
                # Discrete difference for indicator variable: P(Y=1|D=1) - P(Y=1|D=0)
                eta1 = np.clip(X1 @ beta, -8.0, 8.0)
                eta0 = np.clip(X0 @ beta, -8.0, 8.0)
                P1 = stats.norm.cdf(eta1)
                P0 = stats.norm.cdf(eta0)
                ame = float(np.mean(P1 - P0))
                phi1 = stats.norm.pdf(eta1)
                phi0 = stats.norm.pdf(eta0)
                grad = np.mean(phi1[:, None] * X1 - phi0[:, None] * X0, axis=0)
                var_ame = float(grad.T @ vcov @ grad)
                se_ame = float(np.sqrt(max(0.0, var_ame)))
            else:
                eta = np.clip(X_mat @ beta, -8.0, 8.0)
                phi = stats.norm.pdf(eta)
                ame = float(np.mean(phi) * b_k)
                grad = np.mean((-b_k * eta * phi)[:, None] * X_mat, axis=0)
                grad[k_idx] += np.mean(phi)
                var_ame = float(grad.T @ vcov @ grad)
                se_ame = float(np.sqrt(max(0.0, var_ame)))
        elif mtype == 'logit':
            if is_indicator:
                # Discrete difference for indicator variable: P(Y=1|D=1) - P(Y=1|D=0)
                eta1 = np.clip(X1 @ beta, -30.0, 30.0)
                eta0 = np.clip(X0 @ beta, -30.0, 30.0)
                P1 = 1.0 / (1.0 + np.exp(-eta1))
                P0 = 1.0 / (1.0 + np.exp(-eta0))
                ame = float(np.mean(P1 - P0))
                w1 = P1 * (1.0 - P1)
                w0 = P0 * (1.0 - P0)
                grad = np.mean(w1[:, None] * X1 - w0[:, None] * X0, axis=0)
                var_ame = float(grad.T @ vcov @ grad)
                se_ame = float(np.sqrt(max(0.0, var_ame)))
            else:
                eta = np.clip(X_mat @ beta, -30.0, 30.0)
                p = 1.0 / (1.0 + np.exp(-eta))
                w = p * (1.0 - p)
                ame = float(np.mean(w) * b_k)
                grad = np.mean((b_k * w * (1.0 - 2.0 * p))[:, None] * X_mat, axis=0)
                grad[k_idx] += np.mean(w)
                var_ame = float(grad.T @ vcov @ grad)
                se_ame = float(np.sqrt(max(0.0, var_ame)))
        elif mtype == 'poisson':
            if is_indicator:
                mu1 = np.exp(np.clip(X1 @ beta, -30.0, 30.0))
                mu0 = np.exp(np.clip(X0 @ beta, -30.0, 30.0))
                ame = float(np.mean(mu1 - mu0))
                grad = np.mean(mu1[:, None] * X1 - mu0[:, None] * X0, axis=0)
                var_ame = float(grad.T @ vcov @ grad)
                se_ame = float(np.sqrt(max(0.0, var_ame)))
            else:
                eta = np.clip(X_mat @ beta, -30.0, 30.0)
                mu = np.exp(eta)
                ame = float(np.mean(mu) * b_k)
                grad = np.mean((b_k * mu)[:, None] * X_mat, axis=0)
                grad[k_idx] += np.mean(mu)
                var_ame = float(grad.T @ vcov @ grad)
                se_ame = float(np.sqrt(max(0.0, var_ame)))
        else:
            raise ValueError(f"margins not supported for model type: {mtype}")

        z_val = ame / se_ame if se_ame > 0 else 0.0
        p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_val)))
        z_crit = stats.norm.ppf(0.975)

        margins_table.append({
            'variable': target_var,
            'dy_dx': ame,
            'se': se_ame,
            'z': z_val,
            'p': p_val,
            'ci_lower': ame - z_crit * se_ame,
            'ci_upper': ame + z_crit * se_ame
        })

    return {
        'model_type': mtype,
        'depvar': depvar,
        'n': n,
        'margins': margins_table
    }


# =====================================================================
# 7. Post-Estimation: Linear Combinations (lincom)
# =====================================================================

def compute_lincom(model_result, expr_str):
    """
    Computes linear combinations of coefficients: c' beta.
    Supports expressions like 'mpg - 2*weight' or '2.rep78 - 1.rep78 + 5'.
    """
    var_names = model_result.get('var_names')
    beta = model_result.get('beta')
    vcov = model_result.get('vcov')
    df_r = model_result.get('df_r', 1000)

    # Parse linear combination into coefficient vector c and constant offset
    # Clean expression
    expr = expr_str.strip()
    c = np.zeros(len(var_names), dtype=float)
    offset = 0.0

    # Tokenize signed terms: e.g. + 2*mpg, - weight, + 5
    tokens = re.findall(r'([+-]?\s*[^+-]+)', expr)
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        sign = -1.0 if tok.startswith('-') else 1.0
        tok_clean = tok.lstrip('+-').strip()

        # Check for multiplier: e.g. 2 * mpg
        if '*' in tok_clean:
            mult_str, vname = tok_clean.split('*', 1)
            mult = float(mult_str.strip()) * sign
            vname = vname.strip()
        else:
            # Check if pure number
            try:
                val = float(tok_clean) * sign
                offset += val
                continue
            except ValueError:
                mult = sign
                vname = tok_clean

        if vname in var_names:
            idx = var_names.index(vname)
            c[idx] += mult
        elif vname in ('_cons', 'Intercept'):
            if '_cons' in var_names:
                idx = var_names.index('_cons')
                c[idx] += mult
            else:
                offset += mult
        else:
            raise ValueError(f"variable '{vname}' not found in fitted model")

    estimate = float(c.T @ beta + offset)
    var_estimate = float(c.T @ vcov @ c)
    se_estimate = float(np.sqrt(max(0.0, var_estimate)))

    t_val = estimate / se_estimate if se_estimate > 0 else 0.0
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_val), df_r))
    t_crit = stats.t.ppf(0.975, df_r)

    return {
        'expression': expr,
        'estimate': estimate,
        'se': se_estimate,
        't': t_val,
        'p': p_val,
        'ci_lower': estimate - t_crit * se_estimate,
        'ci_upper': estimate + t_crit * se_estimate,
        'df_r': df_r
    }


def compute_ologit(df, depvar, indepvars, odds_ratio=False, vce='mle', cluster_var=None, max_iter=100, tol=1e-7):
    """
    Fits Ordered Logistic Regression (proportional odds cumulative logit model).
    P(y <= j | X) = 1 / (1 + exp(-(cut_j - X beta)))
    """
    all_needed = [depvar] + indepvars
    if cluster_var and cluster_var not in all_needed:
        all_needed.append(cluster_var)

    data = df[all_needed].dropna().copy()
    n = len(data)
    if n == 0:
        raise ValueError("no observations remaining after dropping missing values")

    y = data[depvar].to_numpy()
    unq_y = np.sort(np.unique(y))
    J = len(unq_y)
    if J < 2:
        raise ValueError(f"dependent variable '{depvar}' must have at least 2 categories")

    k = len(indepvars)
    X = data[indepvars].to_numpy(dtype=float)

    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = OrderedModel(data[depvar], data[indepvars], distr='logit')
            res = mod.fit(method='bfgs', maxiter=max_iter, disp=False)

            beta = res.params[:k].to_numpy(dtype=float)
            se_beta = res.bse[:k].to_numpy(dtype=float)

            cut_params = res.params[k:].to_numpy(dtype=float)
            cuts = [float(cut_params[0])]
            for delta in cut_params[1:]:
                cuts.append(cuts[-1] + float(np.exp(np.clip(delta, -20, 20))))

            raw_cov = res.cov_params().to_numpy(dtype=float)

            # Delta method transformation matrix J_mat
            # Parameters: [beta_1, ..., beta_k, theta_0, ..., theta_{J-2}]
            # Transformed: [beta_1, ..., beta_k, c_0, ..., c_{J-2}]
            p_tot = k + J - 1
            J_mat = np.eye(p_tot)
            J_thresh = np.zeros((J - 1, J - 1))
            J_thresh[:, 0] = 1.0
            for j_idx in range(1, J - 1):
                for m_idx in range(1, j_idx + 1):
                    J_thresh[j_idx, m_idx] = float(np.exp(np.clip(cut_params[m_idx], -20, 20)))
            J_mat[k:, k:] = J_thresh

            cov_mat = J_mat @ raw_cov @ J_mat.T
            cuts_se = [float(np.sqrt(max(0.0, cov_mat[k + j, k + j]))) for j in range(J - 1)]

            ll = float(res.llf)
            ll_0 = float(res.llnull) if hasattr(res, 'llnull') and res.llnull is not None else -n * np.log(J)
            converged = bool(res.mle_retvals['converged']) if hasattr(res, 'mle_retvals') and 'converged' in res.mle_retvals else True
            estimation_method = "statsmodels_ordered_model"
            inference_status = "SUCCESS"
            inference_error = None

    except Exception:
        y_idx = np.searchsorted(unq_y, y)
        from scipy.optimize import minimize

        def transform(theta):
            b = theta[:k]
            c = np.zeros(J - 1)
            c[0] = theta[k]
            if J > 2:
                c[1:] = theta[k] + np.cumsum(np.exp(np.clip(theta[k+1:], -20, 20)))
            return b, c

        def neg_ll(theta):
            b, c = transform(theta)
            eta = X @ b
            cuts_ext = np.concatenate([[-np.inf], c, [np.inf]])
            z_hi = np.clip(cuts_ext[y_idx + 1] - eta, -30, 30)
            z_lo = np.clip(cuts_ext[y_idx] - eta, -30, 30)
            f_hi = np.where(y_idx == J - 1, 1.0, 1.0 / (1.0 + np.exp(-z_hi)))
            f_lo = np.where(y_idx == 0, 0.0, 1.0 / (1.0 + np.exp(-z_lo)))
            p = np.clip(f_hi - f_lo, 1e-15, 1.0)
            return -np.sum(np.log(p))

        b0 = np.zeros(k)
        c0 = np.zeros(J - 1)
        theta0 = np.concatenate([b0, c0])
        opt = minimize(neg_ll, theta0, method='BFGS')
        beta_arr, cuts_arr = transform(opt.x)
        beta = beta_arr.to_numpy(dtype=float) if hasattr(beta_arr, 'to_numpy') else np.asarray(beta_arr, dtype=float)
        cuts = [float(c_v) for c_v in cuts_arr]
        ll = float(-opt.fun)
        ll_0 = -n * np.log(J)
        converged = bool(opt.success)
        estimation_method = "scipy_bfgs_fallback"

        # Check design matrix rank and conditioning first
        is_collinear = (k > 0) and (np.linalg.matrix_rank(X) < k or np.linalg.cond(X) > 1e10)

        p_tot = k + J - 1
        theta_opt = opt.x
        H = np.zeros((p_tot, p_tot))
        eps = 1e-4

        for i in range(p_tot):
            for j in range(i, p_tot):
                if i == j:
                    ei = np.zeros(p_tot)
                    ei[i] = eps
                    H[i, i] = (neg_ll(theta_opt + ei) - 2.0 * neg_ll(theta_opt) + neg_ll(theta_opt - ei)) / (eps ** 2)
                else:
                    ei = np.zeros(p_tot)
                    ej = np.zeros(p_tot)
                    ei[i] = eps
                    ej[j] = eps
                    f_pp = neg_ll(theta_opt + ei + ej)
                    f_pm = neg_ll(theta_opt + ei - ej)
                    f_mp = neg_ll(theta_opt - ei + ej)
                    f_mm = neg_ll(theta_opt - ei - ej)
                    H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps ** 2)
                    H[j, i] = H[i, j]

        try:
            eigvals = np.linalg.eigvalsh(H)
            min_eig = float(np.min(eigvals))
            cond_num = float(np.linalg.cond(H))
            if (not is_collinear) and min_eig > 1e-10 and cond_num < 1e12:
                raw_cov = np.linalg.inv(H)
                J_mat = np.eye(p_tot)
                J_thresh = np.zeros((J - 1, J - 1))
                J_thresh[:, 0] = 1.0
                for j_idx in range(1, J - 1):
                    for m_idx in range(1, j_idx + 1):
                        J_thresh[j_idx, m_idx] = float(np.exp(np.clip(theta_opt[k + m_idx], -20, 20)))
                J_mat[k:, k:] = J_thresh

                cov_mat = J_mat @ raw_cov @ J_mat.T
                se_beta = np.sqrt(np.maximum(0.0, np.diag(cov_mat)[:k]))
                cuts_se = [float(np.sqrt(max(0.0, cov_mat[k + j, k + j]))) for j in range(J - 1)]
                inference_status = "SUCCESS"
                inference_error = None
            else:
                raw_cov = np.full((p_tot, p_tot), np.nan)
                cov_mat = np.full((p_tot, p_tot), np.nan)
                se_beta = np.full(k, np.nan)
                cuts_se = [float(np.nan)] * (J - 1)
                inference_status = "FAILED"
                inference_error = (
                    "Collinear or rank-deficient design matrix detected."
                    if is_collinear
                    else f"Observed Fisher information matrix is ill-conditioned (min_eig={min_eig:.2e}, cond={cond_num:.2e})."
                )
        except Exception as e:
            raw_cov = np.full((p_tot, p_tot), np.nan)
            cov_mat = np.full((p_tot, p_tot), np.nan)
            se_beta = np.full(k, np.nan)
            cuts_se = [float(np.nan)] * (J - 1)
            inference_status = "FAILED"
            inference_error = f"Hessian inversion failed: {e}"

    lr_chi2 = max(0.0, 2.0 * (ll - ll_0))
    prob_chi2 = float(1.0 - stats.chi2.cdf(lr_chi2, k))
    pseudo_r2 = max(0.0, 1.0 - (ll / ll_0)) if ll_0 != 0 else 0.0

    coef_table = []
    z_crit = stats.norm.ppf(0.975)
    for i, name in enumerate(indepvars):
        b_val = float(beta[i])
        se_val = float(se_beta[i]) if se_beta is not None and not np.isnan(se_beta[i]) else np.nan
        if np.isnan(se_val) or se_val <= 0:
            z_val = np.nan
            p_val = np.nan
            ci_l = np.nan
            ci_u = np.nan
        else:
            z_val = b_val / se_val
            p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_val)))
            ci_l = float(b_val - z_crit * se_val)
            ci_u = float(b_val + z_crit * se_val)

        if odds_ratio:
            or_val = float(np.exp(b_val))
            or_se = float(or_val * se_val) if not np.isnan(se_val) else np.nan
            or_ci_l = float(np.exp(ci_l)) if not np.isnan(ci_l) else np.nan
            or_ci_u = float(np.exp(ci_u)) if not np.isnan(ci_u) else np.nan
            coef_table.append({
                'name': name, 'coef': or_val, 'se': or_se, 'z': z_val, 'p': p_val,
                'ci_lower': or_ci_l, 'ci_upper': or_ci_u, 'is_odds_ratio': True
            })
        else:
            coef_table.append({
                'name': name, 'coef': b_val, 'se': se_val, 'z': z_val, 'p': p_val,
                'ci_lower': ci_l, 'ci_upper': ci_u, 'is_odds_ratio': False
            })

    for j in range(J - 1):
        c_val = float(cuts[j])
        c_se = float(cuts_se[j]) if cuts_se is not None and not np.isnan(cuts_se[j]) else np.nan
        if np.isnan(c_se) or c_se <= 0:
            ci_l = np.nan
            ci_u = np.nan
        else:
            ci_l = float(c_val - z_crit * c_se)
            ci_u = float(c_val + z_crit * c_se)
        coef_table.append({
            'name': f'/cut{j+1}',
            'coef': c_val,
            'se': c_se,
            'z': np.nan,
            'p': np.nan,
            'ci_lower': ci_l,
            'ci_upper': ci_u,
            'is_cutpoint': True,
            'is_odds_ratio': False
        })

    return {
        'model_type': 'ologit',
        'depvar': depvar,
        'indepvars': indepvars,
        'n': n,
        'k': k,
        'df_m': k,
        'df_r': n - k - (J - 1),
        'll': ll,
        'll_0': ll_0,
        'lr_chi2': lr_chi2,
        'prob_chi2': prob_chi2,
        'pseudo_r2': pseudo_r2,
        'coefs': coef_table,
        'vcov': cov_mat,
        'vcov_raw': raw_cov,
        'categories': [int(x) if isinstance(x, (int, float, np.integer, np.floating)) and float(x).is_integer() else x for x in unq_y],
        'cuts': cuts,
        'odds_ratio': odds_ratio,
        'converged': converged,
        'estimation_method': estimation_method,
        'inference_status': inference_status,
        'inference_error': inference_error
    }


# =====================================================================
# Negative Binomial Count Regression (NB2 MLE)
# =====================================================================

def compute_nbreg(df, depvar, indepvars, irr=False, vce='mle', cluster_var=None, max_iter=150, tol=1e-7):
    """
    Fits Negative Binomial II (NB2) regression via Maximum Likelihood Estimation.
    Mean Link: mu_i = exp(X_i beta).
    Variance: Var(y|x) = mu + alpha * mu^2.
    Parameterization: Estimates beta and ln(alpha).
    Tests H0: alpha = 0 (LR test against Poisson model).
    """
    all_needed = [depvar] + list(indepvars)
    if cluster_var and cluster_var not in all_needed:
        all_needed.append(cluster_var)

    data = df[all_needed].dropna().copy()
    n = len(data)
    if n == 0:
        raise ValueError("no observations remaining after dropping missing values")

    y = data[depvar].to_numpy(dtype=float)
    if np.any(y < 0) or not np.all(np.equal(np.mod(y, 1), 0)):
        raise ValueError(f"dependent variable '{depvar}' must contain non-negative integer counts for nbreg")

    k = len(indepvars)
    if k > 0:
        X = np.column_stack([data[indepvars].to_numpy(dtype=float), np.ones(n)])
    else:
        X = np.ones((n, 1))

    names = list(indepvars) + ['_cons']
    n_params = len(names)

    # 1. Fit Poisson model for initial values and LR test
    pois_res = compute_poisson(data, depvar, indepvars, vce='mle')
    ll_poisson = pois_res['ll']
    beta_init = pois_res['beta'].copy()

    # Initial alpha from Poisson residuals: sum(( (y - mu)^2 - y ) / mu^2) / n
    mu_init = np.exp(np.clip(X @ beta_init, -30, 30))
    raw_alpha = float(np.mean(np.maximum(0.01, ((y - mu_init)**2 - y) / np.maximum(1e-4, mu_init**2))))
    gamma_init = float(np.log(max(1e-4, min(10.0, raw_alpha))))

    params_init = np.append(beta_init, gamma_init)

    def neg_loglik_and_grad(params):
        b = params[:-1]
        gamma = params[-1]
        alpha = np.exp(np.clip(gamma, -15, 10))
        inv_alpha = 1.0 / alpha

        eta = np.clip(X @ b, -30, 30)
        mu = np.exp(eta)

        # Log-likelihood
        ll_i = (special.gammaln(y + inv_alpha) - special.gammaln(y + 1) - special.gammaln(inv_alpha)
                - (y + inv_alpha) * np.log1p(alpha * mu) + y * eta + y * gamma)
        ll = float(np.sum(ll_i))

        # Gradient
        w = (y - mu) / (1.0 + alpha * mu)
        grad_beta = X.T @ w

        term_psi = special.digamma(y + inv_alpha) - special.digamma(inv_alpha)
        grad_gamma = float(np.sum(-inv_alpha * (term_psi - np.log1p(alpha * mu)) + w))

        grad = np.append(grad_beta, grad_gamma)
        return -ll, -grad

    opt = optimize.minimize(neg_loglik_and_grad, params_init, jac=True, method='BFGS',
                            options={'maxiter': max_iter, 'gtol': tol})
    params_opt = opt.x
    beta_opt = params_opt[:-1]
    gamma_opt = params_opt[-1]
    alpha_opt = float(np.exp(gamma_opt))
    ll = float(-opt.fun)

    # Fit constant-only model for Pseudo R2
    def neg_loglik_const(params):
        b_c = params[0]
        gamma_c = params[1]
        alpha_c = np.exp(np.clip(gamma_c, -15, 10))
        inv_alpha_c = 1.0 / alpha_c
        mu_c = np.exp(np.clip(b_c, -30, 30))
        ll_i = (special.gammaln(y + inv_alpha_c) - special.gammaln(y + 1) - special.gammaln(inv_alpha_c)
                - (y + inv_alpha_c) * np.log1p(alpha_c * mu_c) + y * b_c + y * gamma_c)
        return -float(np.sum(ll_i))

    y_bar = float(np.mean(y))
    opt_0 = optimize.minimize(neg_loglik_const, [np.log(max(1e-4, y_bar)), gamma_init], method='BFGS')
    ll_0 = float(-opt_0.fun)

    # Central finite differences Hessian for parameter covariance
    p_len = len(params_opt)
    h_step = 1e-5
    H = np.zeros((p_len, p_len))
    for i in range(p_len):
        for j in range(p_len):
            if i == j:
                e_i = np.zeros(p_len); e_i[i] = h_step
                H[i, i] = (neg_loglik_and_grad(params_opt + e_i)[0] - 2 * opt.fun + neg_loglik_and_grad(params_opt - e_i)[0]) / (h_step**2)
            elif i < j:
                e_i = np.zeros(p_len); e_i[i] = h_step
                e_j = np.zeros(p_len); e_j[j] = h_step
                H[i, j] = (neg_loglik_and_grad(params_opt + e_i + e_j)[0]
                           - neg_loglik_and_grad(params_opt + e_i - e_j)[0]
                           - neg_loglik_and_grad(params_opt - e_i + e_j)[0]
                           + neg_loglik_and_grad(params_opt - e_i - e_j)[0]) / (4 * h_step**2)
                H[j, i] = H[i, j]

    try:
        inv_H = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        inv_H = np.linalg.pinv(H)

    vce = vce.lower() if vce else 'mle'
    n_clusters = None
    df_vce = n - n_params

    # Scores for robust / cluster sandwich
    mu_opt = np.exp(np.clip(X @ beta_opt, -30, 30))
    inv_a = 1.0 / alpha_opt
    w_opt = (y - mu_opt) / (1.0 + alpha_opt * mu_opt)
    scores_beta = X * w_opt[:, None]
    term_psi_opt = special.digamma(y + inv_a) - special.digamma(inv_a)
    scores_gamma = (-inv_a * (term_psi_opt - np.log1p(alpha_opt * mu_opt)) + w_opt)[:, None]
    all_scores = np.column_stack([scores_beta, scores_gamma])

    if vce == 'mle':
        vcov = inv_H
    elif vce in ('robust', 'hc1'):
        omega = all_scores.T @ all_scores
        scale = n / float(max(1, n - 1))
        vcov = scale * (inv_H @ omega @ inv_H)
    elif vce == 'cluster':
        if not cluster_var or cluster_var not in data.columns:
            raise ValueError(f"cluster variable '{cluster_var}' not found")
        clusters = data[cluster_var].to_numpy()
        from engine.estimation import compute_mle_vcov_cluster
        vcov, n_clusters = compute_mle_vcov_cluster(inv_H, all_scores, clusters)
        df_vce = n_clusters - 1
    else:
        raise ValueError(f"Unsupported vce '{vce}' for nbreg. Allowed: 'mle', 'robust', 'cluster'.")

    se_all = np.sqrt(np.maximum(0, np.diag(vcov)))
    se_beta = se_all[:-1]
    se_gamma = se_all[-1]
    se_alpha = alpha_opt * se_gamma # Delta method

    z_stat = np.where(se_beta > 0, beta_opt / se_beta, 0.0)
    p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat)))
    z_crit = stats.norm.ppf(0.975)

    coef_table = []
    for i, name in enumerate(names):
        b = float(beta_opt[i])
        s = float(se_beta[i])
        z = float(z_stat[i])
        p = float(p_val[i])
        ci_l = b - z_crit * s
        ci_u = b + z_crit * s

        if irr and name != '_cons':
            irr_val = float(np.exp(b))
            irr_se = float(irr_val * s)
            irr_ci_l = float(np.exp(ci_l))
            irr_ci_u = float(np.exp(ci_u))
            coef_table.append({
                'name': name,
                'coef': irr_val,
                'se': irr_se,
                'z': z,
                'p': p,
                'ci_lower': irr_ci_l,
                'ci_upper': irr_ci_u,
                'is_odds_ratio': True,
                'is_irr': True
            })
        else:
            coef_table.append({
                'name': name,
                'coef': b,
                'se': s,
                'z': z,
                'p': p,
                'ci_lower': ci_l,
                'ci_upper': ci_u,
                'is_odds_ratio': False,
                'is_irr': False
            })

    # Add /lnalpha and alpha rows
    lnalpha_ci_l = gamma_opt - z_crit * se_gamma
    lnalpha_ci_u = gamma_opt + z_crit * se_gamma
    coef_table.append({
        'name': '/lnalpha',
        'coef': gamma_opt,
        'se': se_gamma,
        'z': np.nan,
        'p': np.nan,
        'ci_lower': lnalpha_ci_l,
        'ci_upper': lnalpha_ci_u,
        'is_odds_ratio': False,
        'is_aux': True
    })

    alpha_ci_l = float(np.exp(lnalpha_ci_l))
    alpha_ci_u = float(np.exp(lnalpha_ci_u))
    coef_table.append({
        'name': 'alpha',
        'coef': alpha_opt,
        'se': se_alpha,
        'z': np.nan,
        'p': np.nan,
        'ci_lower': alpha_ci_l,
        'ci_upper': alpha_ci_u,
        'is_odds_ratio': False,
        'is_aux': True
    })

    # LR test of alpha=0: H0: Poisson model vs H1: Negative Binomial
    # Test statistic chibar2(01) = 2 * (ll_nb - ll_poisson)
    chi2_alpha = max(0.0, 2.0 * (ll - ll_poisson))
    p_alpha = float(0.5 * (1.0 - stats.chi2.cdf(chi2_alpha, 1)))

    lr_chi2 = max(0.0, 2.0 * (ll - ll_0))
    prob_chi2 = float(1.0 - stats.chi2.cdf(lr_chi2, k))
    pseudo_r2 = max(0.0, 1.0 - (ll / ll_0)) if ll_0 != 0 else 0.0

    return {
        'model_type': 'nbreg',
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
        'alpha': alpha_opt,
        'lnalpha': gamma_opt,
        'se_alpha': se_alpha,
        'se_lnalpha': se_gamma,
        'chi2_c': chi2_alpha,
        'p_c': p_alpha,
        'coefs': coef_table,
        'beta': beta_opt,
        'vcov': vcov,
        'var_names': names,
        'data_index': data.index,
        'irr': irr,
        'vce': vce,
        'cluster_var': cluster_var,
        'n_clusters': n_clusters
    }


# =====================================================================
# Multinomial Logistic Regression (mlogit MLE)
# =====================================================================

def compute_mlogit(df, depvar, indepvars, baseoutcome=None, rrr=False, vce='mle', cluster_var=None, max_iter=150, tol=1e-7):
    """
    Fits Multinomial Logistic Regression via Maximum Likelihood Estimation.
    P(y = j | X) = exp(X beta_j) / sum_m exp(X beta_m), with beta_base = 0.
    Supports:
      - baseoutcome(#)
      - rrr (Relative Risk Ratios)
      - robust and cluster-robust standard errors
    """
    all_needed = [depvar] + list(indepvars)
    if cluster_var and cluster_var not in all_needed:
        all_needed.append(cluster_var)

    data = df[all_needed].dropna().copy()
    n = len(data)
    if n == 0:
        raise ValueError("no observations remaining after dropping missing values")

    y_raw = data[depvar].to_numpy()
    unique_cats = sorted(np.unique(y_raw))
    J = len(unique_cats)
    if J < 2:
        raise ValueError(f"dependent variable '{depvar}' must have at least 2 distinct categories")

    # Determine base outcome
    if baseoutcome is not None:
        try:
            # try matching numeric or string
            matched = [c for c in unique_cats if str(c) == str(baseoutcome) or (isinstance(c, (int, float, np.number)) and c == float(baseoutcome))]
            if matched:
                base_val = matched[0]
            else:
                base_val = unique_cats[0]
        except Exception:
            base_val = unique_cats[0]
    else:
        # Default: minimum category
        base_val = unique_cats[0]

    non_base_cats = [c for c in unique_cats if c != base_val]
    num_eqs = len(non_base_cats) # J - 1

    k = len(indepvars)
    if k > 0:
        X = np.column_stack([data[indepvars].to_numpy(dtype=float), np.ones(n)])
    else:
        X = np.ones((n, 1))

    k_vars = X.shape[1] # k + 1
    names = list(indepvars) + ['_cons']
    n_params_total = num_eqs * k_vars

    # Indicator matrix for categories
    d_mat = np.zeros((n, num_eqs), dtype=float)
    for eq_idx, c in enumerate(non_base_cats):
        d_mat[:, eq_idx] = (y_raw == c).astype(float)

    def neg_loglik_and_grad(params):
        beta_mat = params.reshape((num_eqs, k_vars)) # (num_eqs, k_vars)
        eta = X @ beta_mat.T # (n, num_eqs)
        eta = np.clip(eta, -30, 30)
        exp_eta = np.exp(eta)
        denom = 1.0 + np.sum(exp_eta, axis=1, keepdims=True)
        P_non_base = exp_eta / denom

        ll_i = np.sum(d_mat * eta, axis=1) - np.log(denom.ravel())
        ll = float(np.sum(ll_i))

        # Gradient
        diff = d_mat - P_non_base # (n, num_eqs)
        grad_mat = diff.T @ X # (num_eqs, k_vars)
        grad = grad_mat.ravel()
        return -ll, -grad

    opt = optimize.minimize(neg_loglik_and_grad, np.zeros(n_params_total), jac=True, method='BFGS',
                            options={'maxiter': max_iter, 'gtol': tol})
    params_opt = opt.x
    beta_mat_opt = params_opt.reshape((num_eqs, k_vars))
    ll = float(-opt.fun)

    # Constant-only model for Pseudo R2
    def neg_loglik_const(params):
        c_mat = params.reshape((num_eqs, 1))
        exp_c = np.exp(np.clip(c_mat.T, -30, 30))
        denom = 1.0 + np.sum(exp_c)
        ll_0_val = 0.0
        for eq_idx in range(num_eqs):
            ll_0_val += float(np.sum(d_mat[:, eq_idx])) * float(c_mat[eq_idx, 0])
        ll_0_val -= float(n) * float(np.log(denom))
        return -ll_0_val

    opt_0 = optimize.minimize(neg_loglik_const, np.zeros(num_eqs), method='BFGS')
    ll_0 = float(-opt_0.fun)

    # Analytical Hessian blocks
    eta_opt = np.clip(X @ beta_mat_opt.T, -30, 30)
    exp_eta_opt = np.exp(eta_opt)
    denom_opt = 1.0 + np.sum(exp_eta_opt, axis=1, keepdims=True)
    P_opt = exp_eta_opt / denom_opt # (n, num_eqs)

    H = np.zeros((n_params_total, n_params_total), dtype=float)
    for j in range(num_eqs):
        for m in range(num_eqs):
            if j == m:
                w_jm = P_opt[:, j] * (1.0 - P_opt[:, j])
            else:
                w_jm = - P_opt[:, j] * P_opt[:, m]
            H_jm = (X * w_jm[:, None]).T @ X
            H[j * k_vars:(j + 1) * k_vars, m * k_vars:(m + 1) * k_vars] = H_jm

    try:
        inv_H = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        inv_H = np.linalg.pinv(H)

    vce = vce.lower() if vce else 'mle'
    n_clusters = None
    df_vce = n - n_params_total

    # Scores for sandwich VCE
    diff_opt = d_mat - P_opt # (n, num_eqs)
    scores = np.zeros((n, n_params_total), dtype=float)
    for j in range(num_eqs):
        scores[:, j * k_vars:(j + 1) * k_vars] = X * diff_opt[:, j, None]

    if vce == 'mle':
        vcov = inv_H
    elif vce in ('robust', 'hc1'):
        omega = scores.T @ scores
        scale = n / float(max(1, n - 1))
        vcov = scale * (inv_H @ omega @ inv_H)
    elif vce == 'cluster':
        if not cluster_var or cluster_var not in data.columns:
            raise ValueError(f"cluster variable '{cluster_var}' not found")
        clusters = data[cluster_var].to_numpy()
        from engine.estimation import compute_mle_vcov_cluster
        vcov, n_clusters = compute_mle_vcov_cluster(inv_H, scores, clusters)
        df_vce = n_clusters - 1
    else:
        raise ValueError(f"Unsupported vce '{vce}' for mlogit. Allowed: 'mle', 'robust', 'cluster'.")

    se_all = np.sqrt(np.maximum(0, np.diag(vcov)))
    z_crit = stats.norm.ppf(0.975)

    equations = []
    coef_table = []
    param_idx = 0
    b_dict = {}
    se_dict = {}

    for eq_idx, c in enumerate(non_base_cats):
        eq_name = str(c)
        eq_rows = []
        for v_idx, var_name in enumerate(names):
            b_val = float(params_opt[param_idx])
            s_val = float(se_all[param_idx])
            z_val = float(b_val / s_val) if s_val > 0 else 0.0
            p_val = float(2.0 * (1.0 - stats.norm.cdf(np.abs(z_val))))
            ci_l = b_val - z_crit * s_val
            ci_u = b_val + z_crit * s_val

            lookup_key = f"{eq_name}:{var_name}"
            b_dict[lookup_key] = b_val
            se_dict[lookup_key] = s_val

            if rrr and var_name != '_cons':
                rrr_val = float(np.exp(b_val))
                rrr_se = float(rrr_val * s_val)
                rrr_ci_l = float(np.exp(ci_l))
                rrr_ci_u = float(np.exp(ci_u))
                row_dict = {
                    'eq': eq_name,
                    'name': var_name,
                    'coef': rrr_val,
                    'se': rrr_se,
                    'z': z_val,
                    'p': p_val,
                    'ci_lower': rrr_ci_l,
                    'ci_upper': rrr_ci_u,
                    'is_odds_ratio': True,
                    'is_rrr': True
                }
            else:
                row_dict = {
                    'eq': eq_name,
                    'name': var_name,
                    'coef': b_val,
                    'se': s_val,
                    'z': z_val,
                    'p': p_val,
                    'ci_lower': ci_l,
                    'ci_upper': ci_u,
                    'is_odds_ratio': False,
                    'is_rrr': False
                }
            eq_rows.append(row_dict)
            coef_table.append(row_dict)
            param_idx += 1

        equations.append({
            'outcome': c,
            'rows': eq_rows
        })

    df_m = num_eqs * k
    lr_chi2 = max(0.0, 2.0 * (ll - ll_0))
    prob_chi2 = float(1.0 - stats.chi2.cdf(lr_chi2, df_m)) if df_m > 0 else 1.0
    pseudo_r2 = max(0.0, 1.0 - (ll / ll_0)) if ll_0 != 0 else 0.0

    return {
        'model_type': 'mlogit',
        'depvar': depvar,
        'indepvars': indepvars,
        'n': n,
        'df_m': df_m,
        'df_r': df_vce,
        'll': ll,
        'll_0': ll_0,
        'lr_chi2': lr_chi2,
        'prob_chi2': prob_chi2,
        'pseudo_r2': pseudo_r2,
        'baseoutcome': base_val,
        'categories': unique_cats,
        'equations': equations,
        'coefs': coef_table,
        'b_dict': b_dict,
        'se_dict': se_dict,
        'beta': params_opt,
        'vcov': vcov,
        'rrr': rrr,
        'vce': vce,
        'cluster_var': cluster_var,
        'n_clusters': n_clusters
    }


# =====================================================================
# Constrained Linear Regression (cnsreg)
# =====================================================================

def compute_cnsreg(df, depvar, indepvars, constraints_specs):
    """
    Fits Constrained OLS Regression subject to linear equality constraints R * beta = r.
    Uses Lagrange multiplier projection:
      beta_c = beta_ols - (X'X)^-1 R' (R (X'X)^-1 R')^-1 (R beta_ols - r)
      V_c = s_c^2 * [ (X'X)^-1 - (X'X)^-1 R' (R (X'X)^-1 R')^-1 R (X'X)^-1 ]
    """
    all_vars = [depvar] + list(indepvars)
    data = df[all_vars].dropna().copy()
    n = len(data)
    if n == 0:
        raise ValueError("no observations remaining after dropping missing values")

    y = data[depvar].to_numpy(dtype=float)
    X = np.column_stack([data[indepvars].to_numpy(dtype=float), np.ones(n)])
    names = list(indepvars) + ['_cons']
    p = len(names)

    inv_XX = np.linalg.inv(X.T @ X)
    b_ols = inv_XX @ (X.T @ y)

    # Build R matrix and r vector
    # constraints_specs is a list of dicts: {'coeffs': {varname: coeff}, 'rhs': val}
    R_rows = []
    r_vals = []
    for spec in constraints_specs:
        row = np.zeros(p, dtype=float)
        for vname, coef in spec.get('coeffs', {}).items():
            if vname in names:
                row[names.index(vname)] = float(coef)
        R_rows.append(row)
        r_vals.append(float(spec.get('rhs', 0.0)))

    R = np.array(R_rows, dtype=float)
    r_vec = np.array(r_vals, dtype=float)

    # Solve constrained system
    R_inv_XX_Rt = np.linalg.inv(R @ inv_XX @ R.T)
    b_c = b_ols - inv_XX @ R.T @ R_inv_XX_Rt @ (R @ b_ols - r_vec)

    e_c = y - X @ b_c
    q = len(r_vec) # number of restrictions
    df_r = max(1, n - (p - q))
    s2_c = float(e_c.T @ e_c) / df_r
    rmse = float(np.sqrt(s2_c))

    P_R = inv_XX @ R.T @ R_inv_XX_Rt @ R @ inv_XX
    vcov_c = s2_c * (inv_XX - P_R)
    se_c = np.sqrt(np.maximum(0, np.diag(vcov_c)))

    # Identify directly constrained variables (where restriction is x_j = c)
    constrained_vars = set()
    for row_idx in range(len(R)):
        nonzeros = np.where(R[row_idx] != 0)[0]
        if len(nonzeros) == 1 and R[row_idx, nonzeros[0]] == 1.0:
            constrained_vars.add(names[nonzeros[0]])

    coef_table = []
    t_crit = stats.t.ppf(0.975, df_r)
    for i, name in enumerate(names):
        b_val = float(b_c[i])
        s_val = float(se_c[i])
        is_constrained = name in constrained_vars or s_val == 0.0
        if is_constrained:
            coef_table.append({
                'name': name,
                'coef': b_val,
                'se': 0.0,
                't': np.nan,
                'p': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'constrained': True
            })
        else:
            t_val = b_val / s_val if s_val > 0 else 0.0
            p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_val), df_r))
            ci_l = b_val - t_crit * s_val
            ci_u = b_val + t_crit * s_val
            coef_table.append({
                'name': name,
                'coef': b_val,
                'se': s_val,
                't': t_val,
                'p': p_val,
                'ci_lower': ci_l,
                'ci_upper': ci_u,
                'constrained': False
            })

    return {
        'model_type': 'cnsreg',
        'depvar': depvar,
        'indepvars': indepvars,
        'n': n,
        'df_r': df_r,
        'rmse': rmse,
        'coefs': coef_table,
        'beta': b_c,
        'vcov': vcov_c,
        'var_names': names,
        'constraints': constraints_specs
    }


# =====================================================================
# Seemingly Unrelated Regression (sureg Zellner FGLS)
# =====================================================================

def compute_sureg(df, equations):
    """
    Fits Zellner's Seemingly Unrelated Regression (SUR) model via FGLS.
    equations is a list of tuples/dicts: [(dep1, [indeps1]), (dep2, [indeps2]), ...]
    """
    all_needed = set()
    for dep, indeps in equations:
        all_needed.add(dep)
        for v in indeps:
            all_needed.add(v)

    data = df[list(all_needed)].dropna().copy()
    n = len(data)
    if n == 0:
        raise ValueError("no observations remaining after dropping missing values")

    M = len(equations)
    # Step 1: OLS equation by equation
    ols_resids = []
    X_mats = []
    y_vecs = []
    k_counts = []
    for dep, indeps in equations:
        y_m = data[dep].to_numpy(dtype=float)
        if indeps:
            cols = []
            for c in indeps:
                s = data[c]
                if isinstance(s.dtype, pd.CategoricalDtype):
                    cols.append(s.cat.codes.to_numpy(dtype=float))
                else:
                    try:
                        cols.append(s.to_numpy(dtype=float))
                    except (ValueError, TypeError):
                        codes, _ = pd.factorize(s)
                        cols.append(codes.astype(float))
            X_m = np.column_stack(cols + [np.ones(n)])
        else:
            X_m = np.ones((n, 1))
        b_ols = np.linalg.lstsq(X_m, y_m, rcond=None)[0]
        e_ols = y_m - X_m @ b_ols
        ols_resids.append(e_ols)
        X_mats.append(X_m)
        y_vecs.append(y_m)
        k_counts.append(X_m.shape[1])

    # Step 2: Form residual covariance matrix Sigma (asymptotic formulation divides by N)
    Sigma = np.zeros((M, M), dtype=float)
    for j in range(M):
        for m in range(M):
            Sigma[j, m] = float(ols_resids[j].T @ ols_resids[m]) / n

    inv_Sigma = np.linalg.inv(Sigma)

    # Step 3: Stack and solve SUR GLS system
    total_params = sum(k_counts)
    A = np.zeros((total_params, total_params), dtype=float)
    b_rhs = np.zeros(total_params, dtype=float)

    offsets = [0]
    for k_cnt in k_counts:
        offsets.append(offsets[-1] + k_cnt)

    for j in range(M):
        r_start = offsets[j]
        r_end = offsets[j + 1]
        for m in range(M):
            c_start = offsets[m]
            c_end = offsets[m + 1]
            A[r_start:r_end, c_start:c_end] = inv_Sigma[j, m] * (X_mats[j].T @ X_mats[m])
            b_rhs[r_start:r_end] += inv_Sigma[j, m] * (X_mats[j].T @ y_vecs[m])

    beta_sur = np.linalg.inv(A) @ b_rhs
    vcov_sur = np.linalg.inv(A)
    se_sur = np.sqrt(np.maximum(0, np.diag(vcov_sur)))

    # Construct equation results
    eq_results = []
    coef_table = []
    b_dict = {}
    se_dict = {}
    idx = 0
    z_crit = stats.norm.ppf(0.975)

    for eq_idx, (dep, indeps) in enumerate(equations):
        eq_names = list(indeps) + ['_cons']
        eq_rows = []
        b_eq = beta_sur[offsets[eq_idx]:offsets[eq_idx + 1]]
        y_m = y_vecs[eq_idx]
        X_m = X_mats[eq_idx]
        e_sur = y_m - X_m @ b_eq
        rmse = float(np.sqrt(np.mean(e_sur ** 2)))
        tss = float(np.sum((y_m - np.mean(y_m)) ** 2))
        r2 = max(0.0, float(1.0 - (np.sum(e_sur ** 2) / tss))) if tss > 0 else 0.0

        for vname in eq_names:
            b_val = float(beta_sur[idx])
            s_val = float(se_sur[idx])
            z_val = float(b_val / s_val) if s_val > 0 else 0.0
            p_val = float(2.0 * (1.0 - stats.norm.cdf(np.abs(z_val))))
            ci_l = b_val - z_crit * s_val
            ci_u = b_val + z_crit * s_val

            lookup_key = f"{dep}:{vname}"
            b_dict[lookup_key] = b_val
            se_dict[lookup_key] = s_val

            row_dict = {
                'eq': dep,
                'name': vname,
                'coef': b_val,
                'se': s_val,
                'z': z_val,
                'p': p_val,
                'ci_lower': ci_l,
                'ci_upper': ci_u
            }
            eq_rows.append(row_dict)
            coef_table.append(row_dict)
            idx += 1

        # Equation Chi2 test
        k_slopes = len(indeps)
        if k_slopes > 0:
            sub_b = b_eq[:k_slopes]
            sub_V = vcov_sur[offsets[eq_idx]:offsets[eq_idx] + k_slopes, offsets[eq_idx]:offsets[eq_idx] + k_slopes]
            chi2_eq = float(sub_b.T @ np.linalg.pinv(sub_V) @ sub_b)
            p_eq = float(1.0 - stats.chi2.cdf(chi2_eq, k_slopes))
        else:
            chi2_eq = 0.0
            p_eq = 1.0

        eq_results.append({
            'depvar': dep,
            'indepvars': indeps,
            'parms': k_slopes,
            'rmse': rmse,
            'r2': r2,
            'chi2': chi2_eq,
            'p': p_eq,
            'rows': eq_rows
        })

    return {
        'model_type': 'sureg',
        'n': n,
        'equations': eq_results,
        'coefs': coef_table,
        'b_dict': b_dict,
        'se_dict': se_dict,
        'beta': beta_sur,
        'vcov': vcov_sur,
        'sigma': Sigma
    }


# =====================================================================
# Nonlinear Parameter Combinations (nlcom) & Tests (testnl)
# =====================================================================

def compute_nlcom(model_estimates, expr_str):
    """
    Computes nonlinear combination of estimators via the delta method:
      estimate = g(beta)
      V(g) = grad' * V * grad
      se = sqrt(V(g))
    """
    if not expr_str or not expr_str.strip():
        raise ValueError("empty expression in nlcom")
    if len(expr_str) > MAX_EXPR_LENGTH:
        raise ValueError(f"expression length exceeds maximum limit ({len(expr_str)} > {MAX_EXPR_LENGTH})")

    coefs = model_estimates['coefs']
    # If equations exist in coefs (e.g. sureg or mlogit)
    full_keys = []
    for c in coefs:
        if c.get('is_aux') or c.get('constrained'):
            continue
        if 'eq' in c:
            full_keys.append(f"{c['eq']}:{c['name']}")
        else:
            full_keys.append(c['name'])

    b_vec = np.array([c['coef'] for c in coefs if not c.get('is_aux') and not c.get('constrained')], dtype=float)
    vcov = model_estimates['vcov']

    def eval_g(b_input):
        eval_env = {}
        for idx, k_name in enumerate(full_keys):
            val = float(b_input[idx])
            eval_env[f"__param_{k_name.replace(':', '_')}__"] = val
            bare = k_name.split(":")[-1]
            eval_env[f"__param_{bare}__"] = val
            eval_env[bare] = val
            eval_env[k_name.replace(':', '_')] = val

        # Transform command syntax like _b[mpg] to python valid identifiers
        py_expr = expr_str
        for k_name in full_keys:
            py_expr = py_expr.replace(f"_b[{k_name}]", f"__param_{k_name.replace(':', '_')}__")
            bare = k_name.split(":")[-1]
            py_expr = py_expr.replace(f"_b[{bare}]", f"__param_{bare}__")

        # Whitelist math functions
        eval_env.update({
            'exp': np.exp,
            'log': np.log,
            'ln': np.log,
            'sqrt': np.sqrt,
            'abs': np.abs,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan
        })
        try:
            parsed_node = ast.parse(py_expr, mode='eval').body
            validate_ast(parsed_node)
            res_val = _eval_ast_node(parsed_node, eval_env)
            return float(res_val)
        except Exception as e:
            raise ValueError(f"Invalid or disallowed mathematical expression in nlcom: {e}")

    g_val = eval_g(b_vec)

    # Adaptive numerical gradient via central differences
    p_len = len(b_vec)
    grad = np.zeros(p_len, dtype=float)
    for i in range(p_len):
        h_i = max(abs(b_vec[i]) * 1e-6, 1e-6)
        e_i = np.zeros(p_len)
        e_i[i] = h_i
        grad[i] = (eval_g(b_vec + e_i) - eval_g(b_vec - e_i)) / (2.0 * h_i)

    var_g = float(grad.T @ vcov[:p_len, :p_len] @ grad)
    se_g = float(np.sqrt(max(0.0, var_g)))
    z_stat = float(g_val / se_g) if se_g > 0 else 0.0
    p_val = float(2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat))))
    z_crit = 1.959963984540054

    return {
        'expr': expr_str,
        'estimate': g_val,
        'se': se_g,
        'z': z_stat,
        'p': p_val,
        'ci_lower': g_val - z_crit * se_g,
        'ci_upper': g_val + z_crit * se_g
    }


def compute_testnl(model_estimates, expr_eq_str):
    """
    Computes nonlinear Wald test H0: g(beta) = r0.
    chi2(1) = (g(beta) - r0)^2 / Var(g(beta)).
    """
    if "=" in expr_eq_str:
        left, right = expr_eq_str.split("=", 1)
        r0 = float(right.strip())
        nl_res = compute_nlcom(model_estimates, left.strip())
    else:
        r0 = 0.0
        nl_res = compute_nlcom(model_estimates, expr_eq_str.strip())

    diff = nl_res['estimate'] - r0
    var_g = nl_res['se'] ** 2
    chi2_val = float((diff ** 2) / var_g) if var_g > 0 else 0.0
    p_val = float(1.0 - stats.chi2.cdf(chi2_val, 1))

    return {
        'hypothesis': expr_eq_str,
        'chi2': chi2_val,
        'df': 1,
        'p': p_val
    }


# =====================================================================
# Vector Autoregression (var)
# =====================================================================

def compute_var(df, endog_vars, lags=2):
    """
    Fits Vector Autoregression (VAR) model using statsmodels.tsa.api.VAR.
    Returns multi-equation coefficients, standard errors, and lag diagnostics.
    """
    import statsmodels.tsa.api as tsa
    data = df[list(endog_vars)].dropna().copy()
    n = len(data)
    if n < lags + 5:
        raise ValueError("insufficient observations for specified VAR lags")

    model = tsa.VAR(data)
    res = model.fit(maxlags=lags)

    equations = []
    coef_table = []
    b_dict = {}
    se_dict = {}
    z_crit = stats.norm.ppf(0.975)

    for eq_idx, target in enumerate(endog_vars):
        eq_rows = []
        params = res.params[target]
        bse = res.bse[target]
        tvals = res.tvalues[target]
        pvals = res.pvalues[target]

        for pname in res.params.index:
            b_val = float(params[pname])
            s_val = float(bse[pname])
            z_val = float(tvals[pname])
            p_val = float(pvals[pname])
            ci_l = b_val - z_crit * s_val
            ci_u = b_val + z_crit * s_val

            # Standardize lag parameter name to standard style: L1.y1 or _cons
            if pname == 'const':
                param_name = '_cons'
            elif pname.startswith('L'):
                param_name = pname.replace('L', 'L').replace('.', '.')
            else:
                param_name = pname

            lookup_key = f"{target}:{param_name}"
            b_dict[lookup_key] = b_val
            se_dict[lookup_key] = s_val

            row_dict = {
                'eq': target,
                'name': param_name,
                'coef': b_val,
                'se': s_val,
                'z': z_val,
                'p': p_val,
                'ci_lower': ci_l,
                'ci_upper': ci_u
            }
            eq_rows.append(row_dict)
            coef_table.append(row_dict)

        equations.append({
            'depvar': target,
            'rows': eq_rows,
            'rmse': float(np.sqrt(np.mean(res.resid[target] ** 2)))
        })

    return {
        'model_type': 'var',
        'n': int(res.nobs),
        'endog_vars': list(endog_vars),
        'lags': lags,
        'aic': float(res.aic),
        'bic': float(res.bic),
        'hqic': float(res.hqic),
        'fpe': float(res.fpe),
        'll': float(res.llf),
        'equations': equations,
        'coefs': coef_table,
        'b_dict': b_dict,
        'se_dict': se_dict,
        'vcov': res.cov_params().to_numpy(),
        'sigma_u': res.sigma_u.to_numpy(),
        'resid_cov': res.sigma_u.to_numpy()
    }


# =====================================================================
# Survival Analysis: Cox Proportional Hazards (stcox)
# =====================================================================

def compute_stcox(df, timevar, failvar, indepvars, method='breslow'):
    """
    Fits Cox Proportional Hazards regression using statsmodels.duration.hazard_regression.PHReg.
    Supports Breslow and Efron tied failure methods.
    """
    import statsmodels.duration.hazard_regression as hr
    all_needed = [timevar, failvar] + list(indepvars)
    data = df[all_needed].dropna().copy()
    n = len(data)
    if n == 0:
        raise ValueError("no observations remaining after dropping missing values")

    endog = data[timevar].to_numpy(dtype=float)
    status = data[failvar].to_numpy(dtype=float)
    exog = data[list(indepvars)].to_numpy(dtype=float)

    ties_str = 'breslow' if method.lower() != 'efron' else 'efron'
    ph = hr.PHReg(endog=endog, exog=exog, status=status, ties=ties_str)
    res = ph.fit()

    coef_table = []
    b_dict = {}
    se_dict = {}
    z_crit = stats.norm.ppf(0.975)

    for i, name in enumerate(indepvars):
        b_val = float(res.params[i])
        s_val = float(res.bse[i])
        z_val = float(b_val / s_val) if s_val > 0 else 0.0
        p_val = float(res.pvalues[i])
        hr_val = float(np.exp(b_val))
        hr_se = float(hr_val * s_val)
        hr_ci_l = float(np.exp(b_val - z_crit * s_val))
        hr_ci_u = float(np.exp(b_val + z_crit * s_val))

        b_dict[name] = b_val
        se_dict[name] = s_val

        coef_table.append({
            'name': name,
            'coef': b_val,
            'se': s_val,
            'hr': hr_val,
            'hr_se': hr_se,
            'z': z_val,
            'p': p_val,
            'ci_lower': hr_ci_l,
            'ci_upper': hr_ci_u
        })

    ll = float(res.llf)
    n_events = int(np.sum(status == 1))

    return {
        'model_type': 'stcox',
        'timevar': timevar,
        'failvar': failvar,
        'indepvars': indepvars,
        'n': n,
        'n_events': n_events,
        'll': ll,
        'ties': ties_str,
        'coefs': coef_table,
        'b_dict': b_dict,
        'se_dict': se_dict,
        'vcov': res.cov_params(),
        'converged': bool(getattr(res, 'converged', True))
    }


# =====================================================================
# Survival Analysis: Kaplan-Meier Survivor Table (sts list)
# =====================================================================

def compute_sts_list(df, timevar, failvar=None):
    """
    Computes the Kaplan-Meier product-limit survivor function table,
    Greenwood standard errors, and log-log confidence intervals.
    """
    all_needed = [timevar] + ([failvar] if failvar else [])
    data = df[all_needed].dropna().copy()
    if len(data) == 0:
        raise ValueError("no observations in estimation sample")

    times = data[timevar].to_numpy(dtype=float)
    if failvar and failvar in data.columns:
        status = (data[failvar].to_numpy(dtype=float) == 1).astype(int)
    else:
        status = np.ones(len(times), dtype=int)

    # Sort ascending
    order = np.argsort(times)
    times = times[order]
    status = status[order]

    unique_event_times = sorted(np.unique(times[status == 1]))
    if not unique_event_times:
        return {'table': [], 'n_obs': len(times), 'n_events': 0, 'timevar': timevar, 'failvar': failvar}

    n_total = len(times)
    table = []
    cum_surv = 1.0
    greenwood_sum = 0.0
    z_crit = 1.959963984540054  # 95% standard normal quantile

    for i, t in enumerate(unique_event_times):
        at_risk = int(np.sum(times >= t))
        events = int(np.sum((times == t) & (status == 1)))

        if i + 1 < len(unique_event_times):
            t_next = unique_event_times[i + 1]
            lost = int(np.sum((times >= t) & (times < t_next) & (status == 0)))
        else:
            lost = int(np.sum((times >= t) & (status == 0)))

        hazard = events / at_risk if at_risk > 0 else 0.0
        cum_surv *= (1.0 - hazard)

        if at_risk > events and at_risk > 0:
            greenwood_sum += events / (at_risk * (at_risk - events))
            se = float(cum_surv * np.sqrt(greenwood_sum))
        else:
            se = float(np.nan)

        if 0.0 < cum_surv < 1.0 and greenwood_sum > 0:
            var_loglog = greenwood_sum / (np.log(cum_surv) ** 2)
            sigma = float(np.sqrt(var_loglog))
            ci_lower = float(cum_surv ** np.exp(z_crit * sigma))
            ci_upper = float(cum_surv ** np.exp(-z_crit * sigma))
        else:
            ci_lower = float(np.nan)
            ci_upper = float(np.nan)

        table.append({
            'time': float(t),
            'beg_total': at_risk,
            'fail': events,
            'net_lost': lost,
            'survivor': float(cum_surv),
            'std_err': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })

    return {
        'table': table,
        'n_obs': n_total,
        'n_events': int(np.sum(status == 1)),
        'timevar': timevar,
        'failvar': failvar
    }


def format_sts_list(r):
    """Formats the Kaplan-Meier survivor table matching standard life table output."""
    lines = [
        "         Beg.                      Survivor      Std.",
        "  Time  Total   Fail   Net Lost    Function     Error     [95% Conf. Int.]",
        "-" * 70
    ]
    for row in r['table']:
        t_val = row['time']
        t_str = f"{int(t_val):6d}" if t_val == int(t_val) else f"{t_val:6.2f}"
        beg_str = f"{row['beg_total']:6d}"
        fail_str = f"{row['fail']:6d}"
        lost_str = f"{row['net_lost']:10d}"
        surv_str = f"{row['survivor']:12.4f}"
        se_str = f"{row['std_err']:10.4f}" if not np.isnan(row['std_err']) else "         ."
        cil_str = f"{row['ci_lower']:10.4f}" if not np.isnan(row['ci_lower']) else "         ."
        ciu_str = f"{row['ci_upper']:10.4f}" if not np.isnan(row['ci_upper']) else "         ."
        lines.append(f"{t_str} {beg_str} {fail_str} {lost_str} {surv_str} {se_str}  {cil_str} {ciu_str}")
    lines.append("-" * 70)
    return "\n".join(lines)


