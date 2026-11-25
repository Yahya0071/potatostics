"""
engine/causal.py - Causal Inference & Quasi-Experimental Suite
Implements Difference-in-Differences (DiD) and Propensity Score Matching (PSM).
"""

import numpy as np
from scipy import stats
from engine.estimation import compute_logit, compute_regression

def compute_did(df, depvar, treatvar, timevar, cluster_var=None):
    cols = [depvar, treatvar, timevar]
    if cluster_var and cluster_var in df.columns:
        cols.append(cluster_var)
    data = df[cols].dropna().copy()
    
    # Check binary treatment and time
    for var, label in [(treatvar, "Treatment"), (timevar, "Time/Post")]:
        vals = np.unique(data[var])
        if not np.all(np.isin(vals, [0, 1])):
            raise ValueError(f"{label} variable '{var}' must be binary (0 or 1).")
            
    # Compute 2x2 table and verify all four cells exist
    means = data.groupby([treatvar, timevar])[depvar].mean()
    for cell in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        if cell not in means.index:
            raise ValueError(f"DiD 2x2 estimation requires observations in all four cells. Missing cell (treat={cell[0]}, time={cell[1]}).")

    y00 = float(means.loc[(0, 0)])
    y01 = float(means.loc[(0, 1)])
    y10 = float(means.loc[(1, 0)])
    y11 = float(means.loc[(1, 1)])
    
    diff_ctrl = y01 - y00
    diff_treat = y11 - y10
    did_estimate = diff_treat - diff_ctrl
    
    # Interaction regression: y = b0 + b1*treat + b2*time + b3*(treat*time)
    data['interaction'] = data[treatvar] * data[timevar]
    reg_kwargs = {'robust': True}
    if cluster_var:
        reg_kwargs = {'vce': 'cluster', 'cluster_var': cluster_var}
    reg_res = compute_regression(data, depvar, [treatvar, timevar, 'interaction'], **reg_kwargs)
    
    # Interaction coefficient is the DiD estimator
    did_coef = [r for r in reg_res['coefs'] if r['name'] == 'interaction'][0]
    
    return {
        'depvar': depvar,
        'treatvar': treatvar,
        'timevar': timevar,
        'cluster_var': cluster_var,
        'n': len(data),
        'control_pre': y00,
        'control_post': y01,
        'control_diff': diff_ctrl,
        'treat_pre': y10,
        'treat_post': y11,
        'treat_diff': diff_treat,
        'did_estimate': did_estimate,
        'se': did_coef['se'],
        't': did_coef['t'],
        'p': did_coef['p'],
        'ci_lower': did_coef['ci_lower'],
        'ci_upper': did_coef['ci_upper'],
        'table_2x2': {
            'y00': y00, 'y01': y01, 'diff_ctrl': diff_ctrl,
            'y10': y10, 'y11': y11, 'diff_treat': diff_treat,
            'did': did_estimate
        },
        'regression_result': reg_res,
        'did_coef': did_coef,
        'estimator': "2x2 Difference-in-Differences"
    }

def compute_psmatch(df, treatvar, covariates, depvar, caliper=None):
    needed = [treatvar, depvar] + covariates
    data = df[needed].dropna().copy()
    
    # 1. Fit Propensity Score via Logit
    logit_res = compute_logit(data, treatvar, covariates)
    data['pscore'] = logit_res['p_pred']
    
    treated = data[data[treatvar] == 1]
    control = data[data[treatvar] == 0]
    
    if len(treated) == 0 or len(control) == 0:
        raise ValueError("Both treatment and control observations are required.")
        
    # Common support assessment
    min_ctrl, max_ctrl = float(control['pscore'].min()), float(control['pscore'].max())
    min_treat, max_treat = float(treated['pscore'].min()), float(treated['pscore'].max())
    cs_min = max(min_ctrl, min_treat)
    cs_max = min(max_ctrl, max_treat)
    
    on_supp_t = int(np.sum((treated['pscore'] >= cs_min) & (treated['pscore'] <= cs_max)))
    off_supp_t = int(len(treated) - on_supp_t)
    on_supp_c = int(np.sum((control['pscore'] >= cs_min) & (control['pscore'] <= cs_max)))
    off_supp_c = int(len(control) - on_supp_c)

    common_support = {
        'support_min': cs_min,
        'support_max': cs_max,
        'on_support_treated': on_supp_t,
        'off_support_treated': off_supp_t,
        'on_support_control': on_supp_c,
        'off_support_control': off_supp_c
    }

    # 2. Vectorized Nearest Neighbor Matching (with optional caliper)
    ctrl_pscores = control['pscore'].to_numpy()
    ctrl_y = control[depvar].to_numpy()
    treated_pscores = treated['pscore'].to_numpy()
    treated_y = treated[depvar].to_numpy()

    # Matrix of absolute differences (N_treated x N_control)
    diff_matrix = np.abs(treated_pscores[:, None] - ctrl_pscores[None, :])
    best_ctrl_indices = np.argmin(diff_matrix, axis=1)
    min_distances = np.min(diff_matrix, axis=1)

    if caliper is not None:
        caliper_val = float(caliper)
        matched_mask = min_distances <= caliper_val
        if np.sum(matched_mask) == 0:
            raise ValueError(f"No control units matched treated units within caliper of {caliper_val}.")
    else:
        matched_mask = np.ones(len(treated), dtype=bool)

    matched_treated_y = treated_y[matched_mask]
    matched_ctrl_idx = best_ctrl_indices[matched_mask]
    matched_ctrl_y = ctrl_y[matched_ctrl_idx]
    
    diffs = matched_treated_y - matched_ctrl_y
    att = float(np.mean(diffs))
    se_att = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
    t_stat = float(att / se_att) if se_att > 0 else 0.0
    p_val = float(2.0 * (1.0 - stats.norm.cdf(np.abs(t_stat))))
    mean_treated = float(np.mean(matched_treated_y))
    mean_control = float(np.mean(matched_ctrl_y))

    # 3. Covariate Balance Table
    balance_table = []
    for cov in covariates:
        cov_t = treated.iloc[np.where(matched_mask)[0]][cov].to_numpy(dtype=float)
        cov_c = control.iloc[matched_ctrl_idx][cov].to_numpy(dtype=float)
        m_t = float(np.mean(cov_t))
        m_c = float(np.mean(cov_c))
        v_t = float(np.var(cov_t, ddof=1)) if len(cov_t) > 1 else 1.0
        v_c = float(np.var(cov_c, ddof=1)) if len(cov_c) > 1 else 1.0
        pooled_sd = np.sqrt(0.5 * (v_t + v_c))
        std_diff = float(100.0 * (m_t - m_c) / pooled_sd) if pooled_sd > 0 else 0.0
        tt_res = stats.ttest_ind(cov_t, cov_c, equal_var=False)
        balance_table.append({
            'covariate': cov,
            'mean_treated': m_t,
            'mean_control': m_c,
            'std_diff_pct': std_diff,
            't_stat': float(tt_res.statistic) if not np.isnan(tt_res.statistic) else 0.0,
            'p_val': float(tt_res.pvalue) if not np.isnan(tt_res.pvalue) else 1.0
        })

    return {
        'depvar': depvar,
        'outcome_var': depvar,
        'treatvar': treatvar,
        'covariates': covariates,
        'caliper': caliper,
        'n_treated': len(treated),
        'n_control': len(control),
        'n_matched': int(np.sum(matched_mask)),
        'mean_treated': mean_treated,
        'mean_control': mean_control,
        'att': att,
        'se': se_att,
        't': t_stat,
        'p': p_val,
        'p_value': p_val,
        'ci_lower': att - 1.96 * se_att,
        'ci_upper': att + 1.96 * se_att,
        'common_support': common_support,
        'balance_table': balance_table
    }
