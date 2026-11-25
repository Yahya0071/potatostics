"""
engine/causal_modern.py - Modern Causal Inference Engine
Implements:
1. Callaway & Sant'Anna (2021) Staggered Difference-in-Differences (csdid)
   with group-time ATTs, dynamic event-study aggregations, and pre-trend test.
2. Synthetic Control Method (synth) with convex constrained optimization,
   predictor matching, trajectory calculation, and in-space placebo inference.
3. Event-Study regression estimation (eventstudy / coefplot).
4. Weak-instrument diagnostics (Montiel Olea-Pflueger effective F, Anderson-Rubin).
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


# =====================================================================
# 1. Staggered Difference-in-Differences (Callaway & Sant'Anna 2021)
# =====================================================================

def compute_csdid(df, depvar, ivar, timevar, gvar, control_type="notyettreated", n_boot=999, seed=None):
    """
    Computes Callaway & Sant'Anna (2021) Staggered DiD estimator.
    Parameters:
    - depvar: Outcome variable
    - ivar: Panel unit identifier
    - timevar: Time period variable
    - gvar: Cohort / first treatment time variable (0 or NaN for never-treated)
    - control_type: 'notyettreated' or 'nevertreated'
    """
    data = df[[depvar, ivar, timevar]].dropna().copy()
    data[gvar] = df.loc[data.index, gvar]
    
    # Never-treated units coded NaN or 0: preserve them, count them, and map NaN to 0
    never_mask = data[gvar].isna() | (data[gvar] == 0)
    n_never_treated = int(data.loc[never_mask, ivar].nunique())
    data[gvar] = data[gvar].fillna(0).astype(int)

    data[ivar] = data[ivar].astype(str)
    data[timevar] = data[timevar].astype(int)

    times = sorted(data[timevar].unique())
    cohorts = sorted([g for g in data[gvar].unique() if g > 0 and g > min(times)])

    if not cohorts:
        raise ValueError("no treated cohorts found in gvar; check treatment timing variable")

    # Pivot panel into balanced/unbalanced matrix: (unit, time) -> y
    pivot_y = data.pivot(index=ivar, columns=timevar, values=depvar)
    unit_g = data.groupby(ivar)[gvar].first()
    all_units = pivot_y.index
    N_units = len(all_units)
    unit_to_idx = {u: idx for idx, u in enumerate(all_units)}

    gt_results = []
    unit_influences = {}

    for g in cohorts:
        # Baseline period for cohort g is g - 1
        baseline_t = g - 1
        if baseline_t not in times:
            continue

        treated_units = unit_g[unit_g == g].index
        n_g = len(treated_units)
        if n_g == 0:
            continue

        for t in times:
            if t == baseline_t:
                continue

            # Determine control units
            if control_type == "nevertreated":
                control_units = unit_g[(unit_g == 0) | (unit_g > max(times))].index
            else:  # not-yet-treated: controls must have G > max(t, g-1) or G == 0
                max_t = max(t, baseline_t)
                control_units = unit_g[(unit_g == 0) | (unit_g > max_t) | (unit_g > max(times))].index

            # Drop treated units from control if any overlap
            control_units = [u for u in control_units if u not in treated_units]
            n_c = len(control_units)
            if n_c == 0:
                continue

            # Compute change in outcome: delta_y = Y_t - Y_baseline
            try:
                dy_treat = (pivot_y.loc[treated_units, t] - pivot_y.loc[treated_units, baseline_t]).dropna()
                dy_ctrl = (pivot_y.loc[control_units, t] - pivot_y.loc[control_units, baseline_t]).dropna()
            except KeyError:
                continue

            if len(dy_treat) == 0 or len(dy_ctrl) == 0:
                continue

            att_gt = float(dy_treat.mean() - dy_ctrl.mean())
            event_time = t - g

            # Unit-level influence function vector psi_gt
            psi_gt = np.zeros(N_units, dtype=float)
            n_treat_obs = len(dy_treat)
            n_ctrl_obs = len(dy_ctrl)
            mean_dy_treat = float(dy_treat.mean())
            mean_dy_ctrl = float(dy_ctrl.mean())

            for u, dy in dy_treat.items():
                psi_gt[unit_to_idx[u]] = (N_units / n_treat_obs) * (dy - mean_dy_treat)

            for u, dy in dy_ctrl.items():
                psi_gt[unit_to_idx[u]] = -(N_units / n_ctrl_obs) * (dy - mean_dy_ctrl)

            unit_influences[(g, t)] = psi_gt

            if N_units > 1:
                se_gt = float(np.sqrt(np.sum(psi_gt ** 2) / (N_units ** 2)))
            else:
                var_treat = float(np.var(dy_treat, ddof=1) / len(dy_treat)) if len(dy_treat) > 1 else 0.0
                var_ctrl = float(np.var(dy_ctrl, ddof=1) / len(dy_ctrl)) if len(dy_ctrl) > 1 else 0.0
                se_gt = float(np.sqrt(var_treat + var_ctrl))

            z_stat = att_gt / se_gt if se_gt > 0 else 0.0
            p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat)))
            z_crit = stats.norm.ppf(0.975)

            gt_results.append({
                'cohort': g,
                'time': t,
                'event_time': event_time,
                'att': att_gt,
                'se': se_gt,
                'z': z_stat,
                'p': p_val,
                'ci_lower': att_gt - z_crit * se_gt,
                'ci_upper': att_gt + z_crit * se_gt,
                'n_treat': len(dy_treat),
                'n_control': len(dy_ctrl)
            })

    if not gt_results:
        raise ValueError("could not compute any group-time ATTs; check sample coverage")

    df_gt = pd.DataFrame(gt_results)

    # Dynamic event-study aggregation (by event_time e = t - g)
    event_agg = []
    psi_event = {}
    for e in sorted(df_gt['event_time'].unique()):
        sub_e = df_gt[df_gt['event_time'] == e]
        weights = (sub_e['n_treat'] / sub_e['n_treat'].sum()).to_numpy()
        att_e = float(np.sum(sub_e['att'] * weights))

        # Influence function for aggregated ATT(e)
        Psi_e = np.zeros(N_units, dtype=float)
        for k_idx, (_, row) in enumerate(sub_e.iterrows()):
            w_k = weights[k_idx]
            Psi_e += w_k * unit_influences[(row['cohort'], row['time'])]
        psi_event[int(e)] = Psi_e

        if N_units > 1:
            se_e = float(np.sqrt(np.sum(Psi_e ** 2) / (N_units ** 2)))
        else:
            se_e = float(np.sqrt(np.sum((weights ** 2) * (sub_e['se'] ** 2))))

        z_e = att_e / se_e if se_e > 0 else 0.0
        p_e = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_e)))
        z_crit = stats.norm.ppf(0.975)

        event_agg.append({
            'event_time': int(e),
            'att': att_e,
            'se': se_e,
            'z': z_e,
            'p': p_e,
            'ci_lower': att_e - z_crit * se_e,
            'ci_upper': att_e + z_crit * se_e,
            'n_groups': len(sub_e)
        })

    df_event = pd.DataFrame(event_agg)

    # Simple overall ATT (average across post-treatment periods e >= 0)
    post_cells = df_gt[df_gt['event_time'] >= 0]
    if len(post_cells) > 0:
        w_post = (post_cells['n_treat'] / post_cells['n_treat'].sum()).to_numpy()
        overall_att = float(np.sum(post_cells['att'] * w_post))
        Psi_post = np.zeros(N_units, dtype=float)
        for k_idx, (_, row) in enumerate(post_cells.iterrows()):
            w_k = w_post[k_idx]
            Psi_post += w_k * unit_influences[(row['cohort'], row['time'])]
        if N_units > 1:
            overall_se = float(np.sqrt(np.sum(Psi_post ** 2) / (N_units ** 2)))
        else:
            overall_se = float(np.sqrt(np.sum((w_post ** 2) * (post_cells['se'] ** 2))))
    else:
        overall_att = 0.0
        overall_se = 0.0

    z_overall = overall_att / overall_se if overall_se > 0 else 0.0
    p_overall = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_overall)))
    z_crit = stats.norm.ppf(0.975)

    # Pre-trend joint test (H0: ATT(e) = 0 for all e < 0) via Callaway & Sant'Anna (2021) Sup-t Multiplier-Bootstrap
    pre_cells = df_event[df_event['event_time'] < 0]
    if len(pre_cells) > 0:
        pre_e_vals = sorted(pre_cells['event_time'].unique())
        L_pre = len(pre_e_vals)
        Psi_pre = np.column_stack([psi_event[e] for e in pre_e_vals])
        Sigma_pre = (Psi_pre.T @ Psi_pre) / (N_units ** 2)
        att_pre = pre_cells.sort_values('event_time')['att'].to_numpy()
        se_pre = pre_cells.sort_values('event_time')['se'].to_numpy()
        se_pre_safe = np.maximum(se_pre, 1e-12)

        try:
            inv_sig = np.linalg.pinv(Sigma_pre)
            chi2_pre = float(att_pre.T @ inv_sig @ att_pre)
        except Exception:
            inv_sig = np.diag(1.0 / (se_pre_safe ** 2))
            chi2_pre = float(np.sum((att_pre / se_pre_safe) ** 2))
        df_pre = L_pre
        chi2_asymp_p = float(1.0 - stats.chi2.cdf(chi2_pre, df_pre)) if df_pre > 0 else 1.0

        # Uniform Sup-t Multiplier Bootstrap (Callaway & Sant'Anna 2021, eq 3.12)
        # Avoids finite-sample upward bias from inverting high-dimensional lead covariance
        t_stat_sup = float(np.max(np.abs(att_pre / se_pre_safe)))
        B = int(n_boot) if n_boot and n_boot > 0 else 999
        boot_rng = np.random.default_rng(seed)
        # Rademacher weights V in {-1.0, 1.0}^(N x B)
        V = boot_rng.choice([-1.0, 1.0], size=(N_units, B))
        theta_star = (Psi_pre.T @ V) / N_units  # shape (L_pre, B)
        t_star = np.max(np.abs(theta_star / se_pre_safe[:, None]), axis=0)  # shape (B,)
        p_pretrend = float((1.0 + np.sum(t_star >= t_stat_sup)) / (1.0 + B))
        pretrend_method = "uniform_supt_multiplier_bootstrap"
    else:
        chi2_pre = 0.0
        df_pre = 0
        p_pretrend = 1.0
        chi2_asymp_p = 1.0
        t_stat_sup = 0.0
        pretrend_method = "uniform_supt_multiplier_bootstrap"

    return {
        'model_type': 'csdid',
        'depvar': depvar,
        'ivar': ivar,
        'timevar': timevar,
        'gvar': gvar,
        'overall_att': overall_att,
        'overall_se': overall_se,
        'overall_z': z_overall,
        'overall_p': p_overall,
        'overall_ci_lower': overall_att - z_crit * overall_se,
        'overall_ci_upper': overall_att + z_crit * overall_se,
        'pretrend_chi2': chi2_pre,
        'pretrend_df': df_pre,
        'pretrend_p': p_pretrend,
        'pretrend_p_asymp': chi2_asymp_p,
        'pretrend_supt': t_stat_sup,
        'pretrend_method': pretrend_method,
        'group_time_effects': gt_results,
        'dynamic_effects': event_agg,
        'unit_influences': unit_influences,
        'n_obs': len(data),
        'n_units': len(all_units),
        'n_never_treated': n_never_treated,
        'label': "unconditional Callaway-Sant'Anna (2021) DiD",
        'cohorts': cohorts
    }


# =====================================================================
# 2. Synthetic Control Method (synth)
# =====================================================================

def compute_synth(df, depvar, idvar, timevar, treated_id, treat_time, predictors=None, run_placebos=True, is_cancelled=None):
    """
    Computes Synthetic Control Method (Abadie, Diamond & Hainmueller).
    Finds optimal convex donor weights w >= 0, sum(w) = 1 minimizing pre-treatment predictor gap.
    Runs in-space placebo inference across donor pool for exact permutation p-value.
    """
    cols = [depvar, idvar, timevar]
    if predictors:
        cols += [p for p in predictors if p not in cols]

    data = df[cols].dropna().copy()
    data[idvar] = data[idvar].astype(str)
    treated_id = str(treated_id)

    times = sorted(data[timevar].unique())
    all_ids = sorted(data[idvar].unique())

    # Check for balanced panel
    n_expected = len(all_ids) * len(times)
    if len(data) != n_expected:
        raise ValueError("synth requires a balanced panel; missing unit-time observations detected.")

    pre_times = [t for t in times if t < treat_time]
    post_times = [t for t in times if t >= treat_time]

    if not pre_times:
        raise ValueError("no pre-treatment time periods found; check treat_time")
    if not post_times:
        raise ValueError("no post-treatment time periods found; check treat_time")

    if treated_id not in all_ids:
        raise ValueError(f"treated unit '{treated_id}' not found in id variable '{idvar}'")

    donor_ids = [uid for uid in all_ids if uid != treated_id]
    n_donors = len(donor_ids)
    if n_donors < 2:
        raise ValueError("synthetic control requires at least two donor pool units")

    # Predictors: if None, use pre-treatment outcome values
    if not predictors:
        predictors = [depvar]

    # Build X1 (treated predictors) and X0 (donor predictors)
    data_pre = data[data[timevar].isin(pre_times)]
    
    # Calculate predictor means over pre-treatment period
    agg_pre = data_pre.groupby([idvar])[predictors].mean()
    X1 = agg_pre.loc[treated_id].to_numpy(dtype=float)
    X0 = agg_pre.loc[donor_ids].to_numpy(dtype=float).T  # K x J

    # Optimize donor weights W: min ||X1 - X0 W||^2 s.t. w_j >= 0, sum(w) = 1
    def synth_objective(w):
        diff = X1 - X0 @ w
        return float(np.sum(diff ** 2))

    w_init = np.ones(n_donors) / n_donors
    bounds = [(0.0, 1.0) for _ in range(n_donors)]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    opt = minimize(synth_objective, w_init, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = np.maximum(0.0, opt.x)
    weights /= np.sum(weights)

    # Pivot outcome for all units: (id, time) -> Y
    pivot_y = data.pivot(index=idvar, columns=timevar, values=depvar)
    y_treated = pivot_y.loc[treated_id].to_numpy(dtype=float)
    y_donors = pivot_y.loc[donor_ids].to_numpy(dtype=float)  # J x T
    y_synth = weights @ y_donors  # 1 x T

    gaps = y_treated - y_synth

    # Pre-treatment MSPE and Post-treatment MSPE
    t_arr = np.array(times)
    is_pre = np.isin(t_arr, pre_times)
    is_post = np.isin(t_arr, post_times)

    pre_mspe = float(np.mean(gaps[is_pre] ** 2))
    post_mspe = float(np.mean(gaps[is_post] ** 2))
    mspe_ratio = post_mspe / pre_mspe if pre_mspe > 0 else 0.0

    trajectory = []
    for idx, t in enumerate(times):
        trajectory.append({
            'time': t,
            'treated': float(y_treated[idx]),
            'synthetic': float(y_synth[idx]),
            'gap': float(gaps[idx]),
            'is_post': bool(t >= treat_time)
        })

    weights_table = []
    for j, d_id in enumerate(donor_ids):
        if weights[j] > 0.001:
            weights_table.append({
                'donor_id': d_id,
                'weight': float(weights[j])
            })
    weights_table.sort(key=lambda x: x['weight'], reverse=True)

    # In-space placebo test across all donors
    placebo_ratios = []
    if run_placebos and n_donors <= 50:
        for p_idx, placebo_id in enumerate(donor_ids):
            if is_cancelled and is_cancelled():
                break
            other_donors = [d for d in donor_ids if d != placebo_id]
            X1_p = agg_pre.loc[placebo_id].to_numpy(dtype=float)
            X0_p = agg_pre.loc[other_donors].to_numpy(dtype=float).T
            
            def obj_p(w, x1=X1_p, x0=X0_p):
                return float(np.sum((x1 - x0 @ w) ** 2))
            
            w0 = np.ones(len(other_donors)) / len(other_donors)
            bnds = [(0.0, 1.0) for _ in range(len(other_donors))]
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            res_p = minimize(obj_p, w0, method='SLSQP', bounds=bnds, constraints=cons)
            w_p = np.maximum(0.0, res_p.x)
            w_p /= np.sum(w_p)

            y_p_synth = w_p @ pivot_y.loc[other_donors].to_numpy(dtype=float)
            y_p_true = pivot_y.loc[placebo_id].to_numpy(dtype=float)
            gap_p = y_p_true - y_p_synth

            pre_m_p = float(np.mean(gap_p[is_pre] ** 2))
            post_m_p = float(np.mean(gap_p[is_post] ** 2))
            r_p = post_m_p / pre_m_p if pre_m_p > 0 else 0.0
            placebo_ratios.append(r_p)

        all_ratios = [mspe_ratio] + placebo_ratios
        p_val_placebo = float(np.mean(np.array(all_ratios) >= mspe_ratio))
    else:
        p_val_placebo = np.nan

    return {
        'model_type': 'synth',
        'depvar': depvar,
        'idvar': idvar,
        'timevar': timevar,
        'treated_id': treated_id,
        'treat_time': treat_time,
        'pre_mspe': pre_mspe,
        'post_mspe': post_mspe,
        'mspe_ratio': mspe_ratio,
        'p_value_placebo': p_val_placebo,
        'weights': weights_table,
        'trajectory': trajectory,
        'n_donors': n_donors,
        'times': times
    }


# =====================================================================
# 3. Dynamic Event-Study Regression
# =====================================================================

def compute_event_study(df, depvar, unitvar, timevar, treat_time_var, pre_window=4, post_window=5):
    """
    Fits standard dynamic two-way fixed effects event-study regression:
    y_it = alpha_i + lambda_t + sum_{k != -1} beta_k * 1(t - E_i = k) + eps_it.
    Tests parallel trends assumption H0: beta_{-K} = ... = beta_{-2} = 0.
    """
    from engine.advanced_econometrics import compute_reghdfe

    data = df[[depvar, unitvar, timevar]].dropna().copy()
    data[treat_time_var] = df.loc[data.index, treat_time_var]
    never_treated = data[treat_time_var].isna() | (data[treat_time_var] == 0)
    data['rel_time'] = np.where(never_treated, np.nan, data[timevar] - data[treat_time_var])

    # Bin or drop observations outside [-pre_window, +post_window]
    keep_mask = never_treated | ((data['rel_time'] >= -pre_window) & (data['rel_time'] <= post_window))
    data = data[keep_mask].copy()
    never_treated_sub = data[treat_time_var].isna() | (data[treat_time_var] == 0)

    # Generate lead/lag indicators, omitting rel_time == -1
    lead_lag_vars = []
    for k in range(-pre_window, post_window + 1):
        if k == -1:
            continue  # normalized reference period
        vname = f"lead_{-k}" if k < 0 else f"lag_{k}"
        data[vname] = np.where(never_treated_sub, 0.0, (data['rel_time'] == k).astype(float))
        lead_lag_vars.append((k, vname))

    indepvars = [vname for _, vname in lead_lag_vars]
    absorb_vars = [unitvar, timevar]

    reg_res = compute_reghdfe(data, depvar, indepvars, absorb_vars, vce='cluster', cluster_var=unitvar)

    # Format event-study output table
    coef_dict = {r['name']: r for r in reg_res['coefs']}
    event_table = []
    pre_leads = []

    for k, vname in lead_lag_vars:
        if vname in coef_dict:
            c = coef_dict[vname]
            event_table.append({
                'k': k,
                'name': vname,
                'coef': c['coef'],
                'se': c['se'],
                'ci_lower': c['ci_lower'],
                'ci_upper': c['ci_upper'],
                'p': c['p']
            })
            if k < -1:
                pre_leads.append(vname)

    # Add reference period k = -1 (0 coef, 0 se)
    event_table.append({
        'k': -1,
        'name': 'ref_period',
        'coef': 0.0,
        'se': 0.0,
        'ci_lower': 0.0,
        'ci_upper': 0.0,
        'p': 1.0
    })
    event_table.sort(key=lambda x: x['k'])

    # Pre-trend joint Wald test
    if pre_leads:
        var_indices = [reg_res['var_names'].index(v) for v in pre_leads]
        sub_b = reg_res['beta'][var_indices]
        sub_V = reg_res['vcov'][np.ix_(var_indices, var_indices)]
        try:
            wald_stat = float(sub_b.T @ np.linalg.inv(sub_V) @ sub_b)
            p_wald = float(1.0 - stats.chi2.cdf(wald_stat, len(pre_leads)))
        except Exception:
            wald_stat = 0.0
            p_wald = 1.0
    else:
        wald_stat = 0.0
        p_wald = 1.0

    return {
        'model_type': 'eventstudy',
        'estimator': 'twfe',
        'label': 'TWFE Event Study',
        'depvar': depvar,
        'event_table': event_table,
        'pretrend_chi2': wald_stat,
        'pretrend_df': len(pre_leads),
        'pretrend_p': p_wald,
        'n': reg_res['n'],
        'r2': reg_res['r2_within']
    }


# =====================================================================
# 4. Weak-Instrument Diagnostics
# =====================================================================

def compute_weak_iv_diagnostics(iv_result, df):
    """
    Computes first-stage diagnostics and Montiel Olea-Pflueger (2013) effective F-statistic
    honoring vce ('unadjusted', 'robust'/'hc1', 'cluster').
    """
    endog_vars = iv_result.get('endog_vars', [])
    exog_vars = iv_result.get('exog_vars', [])
    instruments = iv_result.get('instruments', [])
    vce = iv_result.get('vce', 'unadjusted').lower()
    cluster_var = iv_result.get('cluster_var')

    if not endog_vars or not instruments:
        eff_f = float(iv_result.get('first_stage', {}).get('f_stat', 0.0))
    else:
        cols = list(dict.fromkeys(endog_vars + exog_vars + instruments + ([cluster_var] if cluster_var else [])))
        data_idx = iv_result.get('data_index')
        data = df.loc[data_idx, cols].dropna() if data_idx is not None else df[cols].dropna()
        n = len(data)
        q = len(instruments)

        first_endog = endog_vars[0]
        y_fs = data[first_endog].to_numpy(dtype=float)
        Z_excl = data[instruments].to_numpy(dtype=float)
        Z_incl = data[exog_vars].to_numpy(dtype=float) if exog_vars else np.empty((n, 0))

        # Frisch-Waugh-Lovell: residualize y_fs and Z_excl w.r.t. [Z_incl, 1]
        W = np.column_stack([Z_incl, np.ones(n)])
        b_w_y, _, _, _ = np.linalg.lstsq(W, y_fs, rcond=None)
        y_tilde = y_fs - W @ b_w_y

        Z_tilde = np.zeros_like(Z_excl)
        for j in range(q):
            b_w_z, _, _, _ = np.linalg.lstsq(W, Z_excl[:, j], rcond=None)
            Z_tilde[:, j] = Z_excl[:, j] - W @ b_w_z

        # Fit pi_hat = (Z_tilde' Z_tilde)^(-1) Z_tilde' y_tilde
        ZtZ = Z_tilde.T @ Z_tilde
        try:
            inv_ZtZ = np.linalg.inv(ZtZ)
        except np.linalg.LinAlgError:
            inv_ZtZ = np.linalg.pinv(ZtZ)

        pi_hat = inv_ZtZ @ (Z_tilde.T @ y_tilde)
        v_hat = y_tilde - Z_tilde @ pi_hat
        ssr = float(np.sum(v_hat ** 2))
        k_total = q + W.shape[1]
        df_r = max(1, n - k_total)

        if vce in ('robust', 'hc1'):
            scores = Z_tilde * v_hat[:, None]
            omega = scores.T @ scores
            c = n / df_r
            V_pi = c * (inv_ZtZ @ omega @ inv_ZtZ)

            Q_zz = ZtZ / n
            W_pi = n * V_pi
            denom = float(np.trace(W_pi @ Q_zz))
            if denom > 0:
                eff_f = float((pi_hat.T @ Q_zz @ pi_hat) / denom)
            else:
                eff_f = 0.0

        elif vce == 'cluster' and cluster_var and cluster_var in data.columns:
            clusters = data[cluster_var].to_numpy()
            unq_clust = np.unique(clusters)
            G = len(unq_clust)
            meat = np.zeros((q, q), dtype=float)
            for cl in unq_clust:
                idx_c = (clusters == cl)
                Z_c = Z_tilde[idx_c, :]
                v_c = v_hat[idx_c]
                s_c = Z_c.T @ v_c
                meat += np.outer(s_c, s_c)
            q_factor = (G / max(1, G - 1)) * ((n - 1) / df_r)
            V_pi = q_factor * (inv_ZtZ @ meat @ inv_ZtZ)

            Q_zz = ZtZ / n
            W_pi = n * V_pi
            denom = float(np.trace(W_pi @ Q_zz))
            if denom > 0:
                eff_f = float((pi_hat.T @ Q_zz @ pi_hat) / denom)
            else:
                eff_f = 0.0

        else:  # unadjusted classical
            sigma2 = ssr / df_r
            if ssr > 0 and df_r > 0:
                eff_f = float((pi_hat.T @ ZtZ @ pi_hat / q) / sigma2)
            else:
                eff_f = 0.0

    # Montiel Olea-Pflueger effective F threshold benchmarks (tau = 10% / 20% bias)
    crit_10 = 23.1
    crit_20 = 12.8

    return {
        'effective_F': eff_f,
        'crit_val_10pct': crit_10,
        'crit_val_20pct': crit_20,
        'is_strong': bool(eff_f > crit_10)
    }


def compute_did_multiplegt(df, depvar, unitvar, timevar, treatvar, dynamic=0, placebo=0, cluster_var=None, reps=50, seed=42):
    """
    de Chaisemartin & D'Haultfoeuille (2020) DID_M Estimator for Heterogeneous & Switching Treatments.
    Computes instantaneous treatment effect across switchers, dynamic horizon effects, and placebo tests.
    """
    needed_cols = [depvar, unitvar, timevar, treatvar]
    if cluster_var and cluster_var not in needed_cols:
        needed_cols.append(cluster_var)

    data = df[needed_cols].dropna().copy()
    n = len(data)
    if n == 0:
        raise ValueError("no observations in estimation sample")

    units = data[unitvar].unique()
    times = sorted(data[timevar].unique())
    n_times = len(times)
    if n_times < 2:
        raise ValueError("did_multiplegt requires at least 2 time periods")

    y_wide = data.pivot(index=unitvar, columns=timevar, values=depvar)
    d_wide = data.pivot(index=unitvar, columns=timevar, values=treatvar)

    def _calc_effects(y_w, d_w):
        did_t_list = []
        w_t_list = []
        total_switchers = 0
        total_controls = 0

        for idx in range(1, n_times):
            t_curr = times[idx]
            t_prev = times[idx - 1]
            if t_curr not in y_w.columns or t_prev not in y_w.columns:
                continue

            d_curr = d_w[t_curr]
            d_prev = d_w[t_prev]

            switchers = (d_prev == 0) & (d_curr > 0)
            controls = (d_prev == 0) & (d_curr == 0)

            n_s = switchers.sum()
            n_c = controls.sum()
            total_switchers += n_s
            total_controls += n_c

            if n_s > 0 and n_c > 0:
                dy_s = (y_w.loc[switchers, t_curr] - y_w.loc[switchers, t_prev]).mean()
                dy_c = (y_w.loc[controls, t_curr] - y_w.loc[controls, t_prev]).mean()
                did_t = dy_s - dy_c
                did_t_list.append(did_t)
                w_t_list.append(n_s)

        did_m_val = float(np.average(did_t_list, weights=w_t_list)) if did_t_list else 0.0

        dynamic_effects = {}
        for l in range(1, dynamic + 1):
            dyn_list = []
            dyn_w = []
            for idx in range(1, n_times - l):
                t_base = times[idx - 1]
                t_target = times[idx + l]

                s_mask = (d_w[t_base] == 0)
                for step in range(l + 1):
                    s_mask = s_mask & (d_w[times[idx + step]] > 0)

                c_mask = (d_w[t_base] == 0)
                for step in range(l + 1):
                    c_mask = c_mask & (d_w[times[idx + step]] == 0)

                if s_mask.sum() > 0 and c_mask.sum() > 0:
                    dy_s = (y_w.loc[s_mask, t_target] - y_w.loc[s_mask, t_base]).mean()
                    dy_c = (y_w.loc[c_mask, t_target] - y_w.loc[c_mask, t_base]).mean()
                    dyn_list.append(dy_s - dy_c)
                    dyn_w.append(s_mask.sum())

            dynamic_effects[l] = float(np.average(dyn_list, weights=dyn_w)) if dyn_list else 0.0

        placebo_effects = {}
        for p in range(1, placebo + 1):
            plac_list = []
            plac_w = []
            for idx in range(1 + p, n_times):
                t_switch = times[idx]
                t_base = times[idx - 1]
                t_prior = times[idx - 1 - p]

                switchers = (d_w[t_base] == 0) & (d_w[t_switch] > 0)
                controls = (d_w[t_base] == 0) & (d_w[t_switch] == 0)

                if switchers.sum() > 0 and controls.sum() > 0:
                    dy_s = (y_w.loc[switchers, t_base] - y_w.loc[switchers, t_prior]).mean()
                    dy_c = (y_w.loc[controls, t_base] - y_w.loc[controls, t_prior]).mean()
                    plac_list.append(dy_s - dy_c)
                    plac_w.append(switchers.sum())

            placebo_effects[p] = float(np.average(plac_list, weights=plac_w)) if plac_list else 0.0

        return did_m_val, dynamic_effects, placebo_effects, total_switchers, total_controls

    point_est, point_dyn, point_plac, n_switch, n_ctrl = _calc_effects(y_wide, d_wide)

    # Cluster bootstrap standard errors
    b_did_m = []
    b_dyn = {l: [] for l in range(1, dynamic + 1)}
    b_plac = {p: [] for p in range(1, placebo + 1)}

    n_u = len(units)
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        boot_units = rng.choice(units, size=n_u, replace=True)
        y_b = y_wide.loc[boot_units].reset_index(drop=True)
        d_b = d_wide.loc[boot_units].reset_index(drop=True)
        try:
            est_b, dyn_b, plac_b, _, _ = _calc_effects(y_b, d_b)
            b_did_m.append(est_b)
            for l in range(1, dynamic + 1):
                if l in dyn_b:
                    b_dyn[l].append(dyn_b[l])
            for p in range(1, placebo + 1):
                if p in plac_b:
                    b_plac[p].append(plac_b[p])
        except Exception:
            continue

    se_main = float(np.std(b_did_m)) if len(b_did_m) > 1 else 0.1
    z_main = point_est / se_main if se_main > 0 else 0.0
    p_main = float(2.0 * (1.0 - stats.norm.cdf(np.abs(z_main))))
    ci_l = point_est - 1.96 * se_main
    ci_u = point_est + 1.96 * se_main

    dyn_table = []
    for l in range(1, dynamic + 1):
        est_l = point_dyn.get(l, 0.0)
        se_l = float(np.std(b_dyn[l])) if len(b_dyn[l]) > 1 else se_main
        z_l = est_l / se_l if se_l > 0 else 0.0
        p_l = float(2.0 * (1.0 - stats.norm.cdf(np.abs(z_l))))
        dyn_table.append({
            'horizon': f"Effect_{l}",
            'estimate': est_l,
            'se': se_l,
            'z': z_l,
            'p': p_l,
            'ci_lower': est_l - 1.96 * se_l,
            'ci_upper': est_l + 1.96 * se_l
        })

    plac_table = []
    for p in range(1, placebo + 1):
        est_p = point_plac.get(p, 0.0)
        se_p = float(np.std(b_plac[p])) if len(b_plac[p]) > 1 else se_main
        z_p = est_p / se_p if se_p > 0 else 0.0
        p_p = float(2.0 * (1.0 - stats.norm.cdf(np.abs(z_p))))
        plac_table.append({
            'horizon': f"Placebo_{p}",
            'estimate': est_p,
            'se': se_p,
            'z': z_p,
            'p': p_p,
            'ci_lower': est_p - 1.96 * se_p,
            'ci_upper': est_p + 1.96 * se_p
        })

    return {
        'model_type': 'did_multiplegt',
        'depvar': depvar,
        'unitvar': unitvar,
        'timevar': timevar,
        'treatvar': treatvar,
        'n': n,
        'n_units': len(units),
        'n_switchers': n_switch,
        'n_controls': n_ctrl,
        'estimate': point_est,
        'se': se_main,
        'z': z_main,
        'p': p_main,
        'ci_lower': ci_l,
        'ci_upper': ci_u,
        'dynamic_effects': dyn_table,
        'placebo_effects': plac_table
    }


# =====================================================================
# 5. Goodman-Bacon (2021) Timing Decomposition (bacondecomp)
# =====================================================================

def compute_bacondecomp(df, depvar, treatvar, unitvar, timevar):
    """
    Goodman-Bacon (2021) 2x2 Decomposition of Difference-in-Differences with Variation in Treatment Timing.
    Decomposes the TWFE estimate into a variance-weighted average across:
      (a) Treated vs. Never-Treated
      (b) Earlier vs. Later Treated
      (c) Later vs. Earlier Treated
    """
    needed_cols = [depvar, treatvar, unitvar, timevar]
    data = df[needed_cols].dropna().copy()
    if len(data) == 0:
        raise ValueError("no observations in estimation sample")

    data[depvar] = pd.to_numeric(data[depvar], errors='coerce')
    data[treatvar] = pd.to_numeric(data[treatvar], errors='coerce')
    data = data.dropna()

    y = data[depvar].to_numpy(dtype=float)
    d = data[treatvar].to_numpy(dtype=float)

    id_m_d = data.groupby(unitvar)[treatvar].transform('mean').to_numpy(dtype=float)
    t_m_d = data.groupby(timevar)[treatvar].transform('mean').to_numpy(dtype=float)
    d_dm = d - id_m_d - t_m_d + np.mean(d)

    id_m_y = data.groupby(unitvar)[depvar].transform('mean').to_numpy(dtype=float)
    t_m_y = data.groupby(timevar)[depvar].transform('mean').to_numpy(dtype=float)
    y_dm = y - id_m_y - t_m_y + np.mean(y)

    denom_twfe = np.sum(d_dm ** 2)
    if denom_twfe <= 0:
        raise ValueError("Treatment variable has no variance after demeaning by unit and time.")
    twfe_estimate = float(np.sum(d_dm * y_dm) / denom_twfe)

    unit_stats = data.groupby(unitvar).agg(
        max_d=(treatvar, 'max'),
        min_d=(treatvar, 'min')
    )

    treated_rows = data[data[treatvar] > 0]
    first_treated = treated_rows.groupby(unitvar)[timevar].min().to_dict()

    timing_groups = {}
    never_treated = []
    always_treated = []

    all_units = data[unitvar].unique()
    for u in all_units:
        max_v = unit_stats.loc[u, 'max_d']
        min_v = unit_stats.loc[u, 'min_d']
        if max_v == 0:
            never_treated.append(u)
        elif min_v > 0:
            always_treated.append(u)
        else:
            t_first = first_treated.get(u)
            if t_first is not None:
                timing_groups.setdefault(t_first, []).append(u)

    sorted_cohorts = sorted(timing_groups.keys())
    all_times = sorted(data[timevar].unique())
    T_total = len(all_times)
    N_total = len(all_units)

    comparisons = []

    def _run_sub_twfe(sub_df, d_col):
        y_sub = sub_df[depvar].to_numpy(dtype=float)
        d_sub = sub_df[d_col].to_numpy(dtype=float)
        id_m = sub_df.groupby(unitvar)[d_col].transform('mean').to_numpy(dtype=float)
        t_m = sub_df.groupby(timevar)[d_col].transform('mean').to_numpy(dtype=float)
        d_sub_dm = d_sub - id_m - t_m + np.mean(d_sub)

        id_y = sub_df.groupby(unitvar)[depvar].transform('mean').to_numpy(dtype=float)
        t_y = sub_df.groupby(timevar)[depvar].transform('mean').to_numpy(dtype=float)
        y_sub_dm = y_sub - id_y - t_y + np.mean(y_sub)

        v_d = float(np.var(d_sub_dm))
        d_ss = float(np.sum(d_sub_dm ** 2))
        b = float(np.sum(d_sub_dm * y_sub_dm) / d_ss) if d_ss > 0 else 0.0
        return b, v_d

    # 1. Treated vs Never Treated
    if never_treated:
        n_u = len(never_treated)
        for k in sorted_cohorts:
            k_units = timing_groups[k]
            n_k = len(k_units)
            sub_units = set(k_units + never_treated)
            sub_df = data[data[unitvar].isin(sub_units)].copy()
            b_ku, v_ku = _run_sub_twfe(sub_df, treatvar)
            raw_w = ((n_k + n_u) / N_total) ** 2 * v_ku
            comparisons.append({
                "comparison_type": "Treated vs Never Treated",
                "group_k": str(k),
                "group_control": "Never Treated",
                "estimate": b_ku,
                "raw_weight": raw_w
            })

    # 2. Earlier vs. Later Treated & 3. Later vs. Earlier Treated
    for i in range(len(sorted_cohorts)):
        k = sorted_cohorts[i]
        k_units = timing_groups[k]
        n_k = len(k_units)
        d_bar_k = len(data[(data[unitvar].isin(k_units)) & (data[treatvar] > 0)]) / (n_k * T_total) if n_k * T_total > 0 else 0.5

        for j in range(i + 1, len(sorted_cohorts)):
            l = sorted_cohorts[j]
            l_units = timing_groups[l]
            n_l = len(l_units)
            d_bar_l = len(data[(data[unitvar].isin(l_units)) & (data[treatvar] > 0)]) / (n_l * T_total) if n_l * T_total > 0 else 0.5

            pair_units = set(k_units + l_units)

            # Earlier vs Later (t < l)
            sub_earlier = data[(data[unitvar].isin(pair_units)) & (data[timevar] < l)].copy()
            if len(sub_earlier) > 0 and sub_earlier[treatvar].nunique() > 1:
                b_early, v_early = _run_sub_twfe(sub_earlier, treatvar)
                raw_w_early = ((n_k + n_l) / N_total) ** 2 * (1.0 - d_bar_l) ** 2 * v_early
                comparisons.append({
                    "comparison_type": "Earlier vs Later Treated",
                    "group_k": f"Early ({k})",
                    "group_control": f"Late ({l})",
                    "estimate": b_early,
                    "raw_weight": raw_w_early
                })

            # Later vs Earlier (t >= k)
            sub_later = data[(data[unitvar].isin(pair_units)) & (data[timevar] >= k)].copy()
            if len(sub_later) > 0:
                sub_later['d_later'] = ((sub_later[unitvar].isin(l_units)) & (sub_later[timevar] >= l)).astype(float)
                if sub_later['d_later'].nunique() > 1:
                    b_late, v_late = _run_sub_twfe(sub_later, 'd_later')
                    raw_w_late = ((n_k + n_l) / N_total) ** 2 * (d_bar_k) ** 2 * v_late
                    comparisons.append({
                        "comparison_type": "Later vs Earlier Treated",
                        "group_k": f"Late ({l})",
                        "group_control": f"Early ({k})",
                        "estimate": b_late,
                        "raw_weight": raw_w_late
                    })

    tot_raw_w = sum(c["raw_weight"] for c in comparisons)
    if tot_raw_w > 0:
        for c in comparisons:
            c["weight"] = float(c["raw_weight"] / tot_raw_w)
    else:
        for c in comparisons:
            c["weight"] = 1.0 / len(comparisons) if comparisons else 0.0

    weighted_bacon = float(sum(c["estimate"] * c["weight"] for c in comparisons)) if comparisons else twfe_estimate

    summary = []
    by_type = {}
    for c in comparisons:
        by_type.setdefault(c["comparison_type"], []).append(c)

    for ctype, items in by_type.items():
        type_w = float(sum(it["weight"] for it in items))
        type_avg_est = float(sum(it["estimate"] * it["weight"] for it in items) / type_w) if type_w > 0 else 0.0
        summary.append({
            "comparison_type": ctype,
            "weight": type_w,
            "avg_estimate": type_avg_est,
            "count": len(items)
        })

    return {
        "model_type": "bacondecomp",
        "twfe": twfe_estimate,
        "weighted_bacon": weighted_bacon,
        "weights_sum": float(sum(c["weight"] for c in comparisons)),
        "comparisons": comparisons,
        "summary": summary,
        "n_obs": len(data),
        "n_units": N_total,
        "n_times": T_total,
        "depvar": depvar,
        "treatvar": treatvar
    }


def format_bacondecomp(r):
    """Formats Goodman-Bacon decomposition summary into an academic comparison table."""
    lines = [
        "Goodman-Bacon (2021) Difference-in-Differences Decomposition",
        f"Dep. variable: {r['depvar']} | Treatment: {r['treatvar']} | Total Obs: {r['n_obs']:,} | Panels: {r['n_units']:,}",
        f"Two-Way Fixed Effects Estimate (TWFE) = {r['twfe']:.5f}",
        f"Weighted Bacon Decomposition Average = {r['weighted_bacon']:.5f}",
        "-" * 72,
        f"{'Comparison Group':<32} {'Weight':>10} {'Avg. 2x2 DiD':>14} {'Pairs':>10}",
        "-" * 72,
    ]
    for s in r['summary']:
        lines.append(f"{s['comparison_type']:<32} {s['weight']:>10.4f} {s['avg_estimate']:>14.5f} {s['count']:>10}")
    lines.append("-" * 72)
    lines.append(f"{'Total / Weighted Sum':<32} {r['weights_sum']:>10.4f} {r['weighted_bacon']:>14.5f}")
    lines.append("-" * 72)
    return "\n".join(lines)


# =====================================================================
# 6. Doubly Robust Difference-in-Differences (Sant'Anna & Zhao 2020)
# =====================================================================

def compute_drdid(df, depvar, treatvar, timevar, covars=None, unitvar=None, trim_level=0.995, seed=42, ps_covars=None, or_covars=None):
    """
    Sant'Anna & Zhao (2020) Doubly Robust Difference-in-Differences (DRDID).
    Supports 2-period balanced panel (with unitvar) and repeated cross-sections.
    Implements inverse probability weighting with outcome regression and influence-function variance.
    Supports separate propensity score covariates (ps_covars) and outcome regression covariates (or_covars).
    """
    covars = covars or []
    ps_covars = list(ps_covars) if ps_covars is not None else list(covars)
    or_covars = list(or_covars) if or_covars is not None else list(covars)
    all_covars = sorted(set(covars + ps_covars + or_covars))

    needed_cols = [depvar, treatvar, timevar] + all_covars
    if unitvar and unitvar in df.columns:
        needed_cols.append(unitvar)
    data = df[needed_cols].dropna().copy()
    if len(data) == 0:
        raise ValueError("no observations in estimation sample")

    times = sorted(data[timevar].unique())
    if len(times) != 2:
        raise ValueError(f"drdid requires exactly 2 time periods, found {len(times)}: {times}")

    t0, t1 = times[0], times[1]

    def _build_design(source_df, c_list, n_obs):
        if not c_list:
            return np.ones((n_obs, 1))
        X_cov = source_df[c_list].to_numpy(dtype=float)
        for c_idx, c_name in enumerate(c_list):
            if np.var(X_cov[:, c_idx]) < 1e-12:
                raise ValueError(f"Covariate '{c_name}' has zero variance (constant regressor).")
        X_mat = np.column_stack([np.ones(n_obs), X_cov])
        cond_num = np.linalg.cond(X_mat)
        if cond_num > 1e12:
            raise ValueError(f"Singular or highly collinear covariates detected (condition number {cond_num:.2e}).")
        return X_mat

    # Panel Case with Unit Tracking
    if unitvar and unitvar in data.columns:
        piv_y = data.pivot(index=unitvar, columns=timevar, values=depvar)
        piv_d = data.pivot(index=unitvar, columns=timevar, values=treatvar)

        # Check for unbalanced units
        unbalanced_units = len(piv_y) - len(piv_y.dropna())
        valid_units = piv_y.dropna().index.intersection(piv_d.dropna().index)
        if len(valid_units) == 0:
            raise ValueError("No balanced panel units found across the 2 time periods.")

        dy = (piv_y.loc[valid_units, t1] - piv_y.loc[valid_units, t0]).to_numpy(dtype=float)
        d = (piv_d.loc[valid_units, t1] > 0).to_numpy(dtype=float)

        base_df = data[data[timevar] == t0].set_index(unitvar).reindex(valid_units)
        N = len(valid_units)
        n_treat = int(np.sum(d))
        n_ctrl = N - n_treat

        if n_treat == 0:
            raise ValueError("No treated units found in estimation sample.")
        if n_ctrl == 0:
            raise ValueError("No control units found in estimation sample.")

        X_ps = _build_design(base_df, ps_covars, N)
        X_or = _build_design(base_df, or_covars, N)

        def _negloglik(b):
            p = 1.0 / (1.0 + np.exp(-X_ps @ b))
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            return -np.sum(d * np.log(p) + (1.0 - d) * np.log(1.0 - p))

        b0 = np.zeros(X_ps.shape[1])
        opt_res = minimize(_negloglik, b0, method='BFGS')
        if not opt_res.success and np.linalg.norm(opt_res.jac) > 1e-2:
            raise ValueError(f"Propensity score optimization failed to converge: {opt_res.message}")

        ps = 1.0 / (1.0 + np.exp(-X_ps @ opt_res.x))
        # Check overlap
        ps_ctrl = ps[d == 0]
        ps_treat = ps[d == 1]
        if len(ps_ctrl) > 0 and len(ps_treat) > 0:
            if np.min(ps_treat) > np.max(ps_ctrl) or np.max(ps_treat) < np.min(ps_ctrl):
                raise ValueError("No common support: treatment and control groups have non-overlapping propensity scores.")

        trim_min = 1.0 - trim_level
        trim_max = trim_level
        ps = np.clip(ps, trim_min, trim_max)

        ctrl_mask = (d == 0)
        X_or_ctrl = X_or[ctrl_mask]
        dy_ctrl = dy[ctrl_mask]
        b_or = np.linalg.lstsq(X_or_ctrl, dy_ctrl, rcond=None)[0]
        mu0_hat = X_or @ b_or

        w = np.where(d == 0, ps / (1.0 - ps), 0.0)
        sum_w = np.sum(w) or 1.0
        w_norm = w / sum_w

        att = float((1.0 / n_treat) * np.sum(d * (dy - mu0_hat)) - np.sum(w_norm * (dy - mu0_hat)))

        p_d = n_treat / N
        psi = (d / p_d) * (dy - mu0_hat - att) - ((1.0 - d) * ps / ((1.0 - ps) * p_d)) * (dy - mu0_hat)
        var_att = float(np.sum(psi ** 2) / (N ** 2))
        se = float(np.sqrt(max(var_att, 1e-12)))
        z_stat = float(att / se) if se > 0 else 0.0
        p_val = float(2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat))))

        return {
            "model_type": "drdid",
            "att": att,
            "se": se,
            "z": z_stat,
            "p": p_val,
            "ci_lower": att - 1.96 * se,
            "ci_upper": att + 1.96 * se,
            "n_obs": N,
            "n_treated": n_treat,
            "n_control": n_ctrl,
            "depvar": depvar,
            "treatvar": treatvar,
            "covars": covars,
            "ps_covars": ps_covars,
            "or_covars": or_covars,
            "data_type": "panel",
            "propensity_scores": ps,
            "mu0_hat": mu0_hat,
            "b_or": b_or,
            "b_ps": opt_res.x,
            "unbalanced_dropped": unbalanced_units
        }
    else:
        # Repeated Cross Sections
        post = (data[timevar] == t1).astype(float).to_numpy()
        d = (data[treatvar] > 0).astype(float).to_numpy()
        y = data[depvar].to_numpy(dtype=float)
        N = len(data)

        n_d1_t1 = int(np.sum(d * post))
        n_d1_t0 = int(np.sum(d * (1 - post)))
        n_d0_t1 = int(np.sum((1 - d) * post))
        n_d0_t0 = int(np.sum((1 - d) * (1 - post)))

        if n_d1_t1 == 0:
            raise ValueError("No treated observations in post-treatment period (D=1, T=1).")
        if n_d1_t0 == 0:
            raise ValueError("No treated observations in pre-treatment period (D=1, T=0).")
        if n_d0_t1 == 0:
            raise ValueError("No control observations in post-treatment period (D=0, T=1).")
        if n_d0_t0 == 0:
            raise ValueError("No control observations in pre-treatment period (D=0, T=0).")

        X_ps = _build_design(data, ps_covars, N)
        X_or = _build_design(data, or_covars, N)

        def _negloglik(b):
            p = 1.0 / (1.0 + np.exp(-X_ps @ b))
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            return -np.sum(d * np.log(p) + (1.0 - d) * np.log(1.0 - p))

        b0 = np.zeros(X_ps.shape[1])
        opt_res = minimize(_negloglik, b0, method='BFGS')
        if not opt_res.success and np.linalg.norm(opt_res.jac) > 1e-2:
            raise ValueError(f"Propensity score optimization failed to converge: {opt_res.message}")

        ps = 1.0 / (1.0 + np.exp(-X_ps @ opt_res.x))
        # Check overlap
        ps_ctrl = ps[d == 0]
        ps_treat = ps[d == 1]
        if len(ps_ctrl) > 0 and len(ps_treat) > 0:
            if np.min(ps_treat) > np.max(ps_ctrl) or np.max(ps_treat) < np.min(ps_ctrl):
                raise ValueError("No common support: treatment and control groups have non-overlapping propensity scores.")

        trim_min = 1.0 - trim_level
        trim_max = trim_level
        raw_ps = ps
        ps = np.clip(raw_ps, trim_min, trim_max)

        mask_c0 = (d == 0) & (post == 0)
        mask_c1 = (d == 0) & (post == 1)
        b_or0 = np.linalg.lstsq(X_or[mask_c0], y[mask_c0], rcond=None)[0]
        b_or1 = np.linalg.lstsq(X_or[mask_c1], y[mask_c1], rcond=None)[0]
        mu0_hat = np.where(post == 1, X_or @ b_or1, X_or @ b_or0)

        residuals = y - mu0_hat
        term_t1 = np.sum(d * post * residuals) / n_d1_t1
        term_t0 = np.sum(d * (1 - post) * residuals) / n_d1_t0

        w_c1 = np.where(mask_c1, ps / (1 - ps), 0.0)
        w_c0 = np.where(mask_c0, ps / (1 - ps), 0.0)
        sw_c1 = np.sum(w_c1) or 1.0
        sw_c0 = np.sum(w_c0) or 1.0

        adj_c1 = np.sum((w_c1 / sw_c1) * residuals)
        adj_c0 = np.sum((w_c0 / sw_c0) * residuals)

        att = float((term_t1 - term_t0) - (adj_c1 - adj_c0))

        # Standardized weights for repeated cross-sections
        w_d1_t1 = (d * post) / (n_d1_t1 / N)
        w_d1_t0 = (d * (1.0 - post)) / (n_d1_t0 / N)
        w_d0_t1 = w_c1 / (sw_c1 / N)
        w_d0_t0 = w_c0 / (sw_c0 / N)

        # Each normalized mean in the traditional RC ATT (Sant'Anna & Zhao,
        # 2020, Eq. 3.3) has its own centering term.
        control_score_t1 = w_d0_t1 * (residuals - adj_c1)
        control_score_t0 = w_d0_t0 * (residuals - adj_c0)
        psi = (w_d1_t1 * (residuals - term_t1) - w_d1_t0 * (residuals - term_t0)
               - control_score_t1 + control_score_t0)

        # Propagate estimation of the two control outcome regressions. Solving
        # the transposed design avoids forming the OLS normal equations.
        for mask, outcome_weights in [(mask_c1, w_d0_t1 - w_d1_t1),
                                      (mask_c0, w_d1_t0 - w_d0_t0)]:
            gradient = X_or.T @ outcome_weights / N
            observation_weights = np.linalg.lstsq(X_or[mask].T, gradient, rcond=None)[0]
            psi[mask] += N * observation_weights * residuals[mask]

        # The odds derivative uses the original logit fit and is zero where
        # the propensity was clipped for the control weights.
        unclipped = (raw_ps > trim_min) & (raw_ps < trim_max)
        gradient = X_ps.T @ (unclipped * (control_score_t0 - control_score_t1)) / N
        ps_design = X_ps * np.sqrt(raw_ps * (1.0 - raw_ps))[:, None]
        ps_design_inverse = np.linalg.pinv(ps_design)
        ps_direction = ps_design_inverse @ (ps_design_inverse.T @ gradient)
        psi += N * (d - raw_ps) * (X_ps @ ps_direction)

        var_att = float(np.sum(psi ** 2) / (N ** 2))
        se = float(np.sqrt(max(var_att, 1e-12)))
        z_stat = float(att / se) if se > 0 else 0.0
        p_val = float(2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat))))

        return {
            "model_type": "drdid",
            "att": att,
            "se": se,
            "z": z_stat,
            "p": p_val,
            "ci_lower": att - 1.96 * se,
            "ci_upper": att + 1.96 * se,
            "n_obs": N,
            "n_treated": int(np.sum(d)),
            "n_control": int(np.sum(1 - d)),
            "n_d1_t1": n_d1_t1,
            "n_d1_t0": n_d1_t0,
            "n_d0_t1": n_d0_t1,
            "n_d0_t0": n_d0_t0,
            "depvar": depvar,
            "treatvar": treatvar,
            "covars": covars,
            "data_type": "repeated_cross_section",
            "propensity_scores": ps,
            "mu0_hat": mu0_hat,
            "b_or0": b_or0,
            "b_or1": b_or1,
            "b_ps": opt_res.x
        }


def format_drdid(r):
    """Formats Sant'Anna & Zhao (2020) Doubly Robust DiD estimation table."""
    lines = [
        "Sant'Anna & Zhao (2020) Doubly Robust Difference-in-Differences",
        f"Dep. variable: {r['depvar']} | Treatment: {r['treatvar']} | Sample ({r['data_type']}): {r['n_obs']:,}",
        f"Treated Units: {r['n_treated']:,} | Control Units: {r['n_control']:,}",
        "-" * 78,
        f"{'Parameter':<12} {'Estimate':>12} {'Std. Err.':>12} {'z':>8} {'P>|z|':>8} {'[95% Conf. Interval]':>22}",
        "-" * 78,
        f"{'ATT':<12} {r['att']:>12.5f} {r['se']:>12.5f} {r['z']:>8.2f} {r['p']:>8.4f} {r['ci_lower']:>10.4f} {r['ci_upper']:>10.4f}",
        "-" * 78,
    ]
    return "\n".join(lines)


