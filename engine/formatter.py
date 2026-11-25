"""
engine/formatter.py - Monospace Text Formatter
Formats statistical models, summaries, crosstabs, diagnostics, and time series
into clean standard ASCII tables for the results console and text logs.
"""

import numpy as np

def format_describe(dataset):
    lines = []
    lines.append(f"Contains data from: {dataset.dataset_label or 'Memory'}")
    lines.append(f" Observations: {dataset.N:>10,}")
    lines.append(f"    Variables: {dataset.K:>10,}")
    lines.append("-" * 78)
    lines.append(f"{'Variable name':<16} {'Storage type':<14} {'Display format':<16} {'Variable label'}")
    lines.append("-" * 78)
    for col in dataset.df.columns:
        dtype_str = str(dataset.df[col].dtype)
        fmt = dataset.var_formats.get(col, "%9.0g" if "int" in dtype_str else ("%9.2f" if "float" in dtype_str else "%9s"))
        lbl = dataset.var_labels.get(col, "")
        lines.append(f"{col:<16} {dtype_str:<14} {fmt:<16} {lbl}")
    lines.append("-" * 78)
    return "\n".join(lines)

def format_summarize(stats_list):
    lines = []
    lines.append(f"{'Variable':<14} | {'Obs':>8} {'Mean':>12} {'Std. dev.':>12} {'Min':>12} {'Max':>12}")
    lines.append("-" * 15 + "+" + "-" * 58)
    for s in stats_list:
        if s['obs'] == 0:
            lines.append(f"{s['var']:<14} | {0:>8} {'':>12} {'':>12} {'':>12} {'':>12}")
        else:
            lines.append(f"{s['var']:<14} | {s['obs']:>8} {s['mean']:>12.4f} {s['std']:>12.4f} {s['min']:>12.4f} {s['max']:>12.4f}")
    return "\n".join(lines)

def format_summarize_detail(stats_list):
    blocks = []
    for s in stats_list:
        lines = []
        varname = s['var']
        lines.append(f"{varname:^61}")
        lines.append("-" * 61)
        lines.append("      Percentiles      Smallest")
        sm = list(s.get('smallest', [0.0]*4))
        lg = list(s.get('largest', [0.0]*4))
        while len(sm) < 4:
            sm.append(0.0)
        while len(lg) < 4:
            lg.append(0.0)
        
        lines.append(f" 1%    {s.get('p1', 0.0):>10.4f}     {sm[0]:>10.4f}")
        lines.append(f" 5%    {s.get('p5', 0.0):>10.4f}     {sm[1]:>10.4f}")
        lines.append(f"10%    {s.get('p10', 0.0):>10.4f}     {sm[2]:>10.4f}       Obs          {s['obs']:>12,d}")
        lines.append(f"25%    {s.get('p25', 0.0):>10.4f}     {sm[3]:>10.4f}       Sum of wgt.  {int(s.get('sum_w', s['obs'])):>12,d}")
        lines.append("")
        lines.append(f"50%    {s.get('p50', 0.0):>10.4f}                      Mean         {s['mean']:>12.4f}")
        lines.append(f"                        Largest       Std. dev.    {s['std']:>12.4f}")
        lines.append(f"75%    {s.get('p75', 0.0):>10.4f}     {lg[0]:>10.4f}")
        lines.append(f"90%    {s.get('p90', 0.0):>10.4f}     {lg[1]:>10.4f}       Variance     {s.get('Var', 0.0):>12.4f}")
        lines.append(f"95%    {s.get('p95', 0.0):>10.4f}     {lg[2]:>10.4f}       Skewness     {s.get('skewness', 0.0):>12.4f}")
        lines.append(f"99%    {s.get('p99', 0.0):>10.4f}     {lg[3]:>10.4f}       Kurtosis     {s.get('kurtosis', 0.0):>12.4f}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)

def format_tabulate_1way(val_counts, total_obs, varname, val_map=None):
    lines = []
    lines.append(f"{varname:<18} | {'Freq.':>10} {'Percent':>10} {'Cum.':>10}")
    lines.append("-" * 19 + "+" + "-" * 34)
    cum = 0.0
    for val, count in val_counts.items():
        pct = (count / total_obs) * 100.0
        cum += pct
        disp_val = str(val)
        if val_map and val in val_map:
            disp_val = str(val_map[val])
        lines.append(f"{disp_val:<18} | {count:>10,} {pct:>10.2f} {cum:>10.2f}")
    lines.append("-" * 19 + "+" + "-" * 34)
    lines.append(f"{'Total':<18} | {total_obs:>10,}     100.00")
    return "\n".join(lines)

def format_tabulate_2way(ct, var1, var2):
    lines = []
    cols = list(ct.columns)
    header = f"{var1:<14} | " + " ".join(f"{str(c):>10}" for c in cols) + f" | {'Total':>10}"
    lines.append("Key: frequency")
    lines.append(header)
    lines.append("-" * 15 + "+" + "-" * (11 * len(cols) + 1) + "+" + "-" * 12)
    for idx, row in ct.iterrows():
        row_str = f"{str(idx):<14} | " + " ".join(f"{int(v):>10,}" for v in row) + f" | {int(row.sum()):>10,}"
        lines.append(row_str)
    lines.append("-" * 15 + "+" + "-" * (11 * len(cols) + 1) + "+" + "-" * 12)
    col_totals = ct.sum(axis=0)
    lines.append(f"{'Total':<14} | " + " ".join(f"{int(v):>10,}" for v in col_totals) + f" | {int(col_totals.sum()):>10,}")
    return "\n".join(lines)

def format_correlate(r):
    vars = r['vars']
    mat = r.get('matrix', r.get('corr'))
    n = r['n']
    is_cov = r.get('is_covariance', False)
    lines = [f"(obs={n:,})"]
    if is_cov:
        lines.append("(covariance matrix)")
    lines.append("")
    hdr = f"{'':<14} | " + " ".join(f"{v:>10}" for v in vars)
    lines.append(hdr)
    lines.append("-" * 15 + "+" + "-" * (11 * len(vars)))
    for i, v1 in enumerate(vars):
        row = [f"{v1:<14} |"]
        for j, v2 in enumerate(vars):
            if j <= i:
                val = mat.loc[v1, v2]
                row.append(f"{val:>10.4f}")
        lines.append(" ".join(row))
    return "\n".join(lines)

def format_regression(r):
    lines = []
    is_robust = r.get('robust', False) or r.get('vce', 'ols') != 'ols'
    vce_type = r.get('vce', 'ols')

    if not is_robust:
        # Classical ANOVA format
        lines.append(f"      Source |       SS           df       MS      Number of obs   = {r['n']:>9,}")
        lines.append(f"-------------+----------------------------------   F({r['df_m']:>2}, {r['df_r']:>5})    = {r['f_stat']:>9.2f}")
        lines.append(f"       Model | {r['ss_m']:>14.4f} {r['df_m']:>4} {r['ms_m']:>12.4f}   Prob > F        = {r['prob_f']:>9.4f}")
        lines.append(f"    Residual | {r['ss_r']:>14.4f} {r['df_r']:>4} {r['ms_r']:>12.4f}   R-squared       = {r['r2']:>9.4f}")
        lines.append(f"-------------+----------------------------------   Adj R-squared   = {r['adj_r2']:>9.4f}")
        lines.append(f"       Total | {r['ss_t']:>14.4f} {r['df_t']:>4} {r['ms_t']:>12.4f}   Root MSE        = {r['root_mse']:>9.4f}")
    else:
        # Robust / Clustered / Spatial OLS format
        if vce_type in ('spatial', 'conley'):
            title = "Linear regression (Spatial HAC / Conley)"
            se_label = "Spatial std. err."
        else:
            title = "Linear regression (Robust)"
            se_label = "Robust std. err."
        cluster_info = f"Number of clusters = {r['n_clusters']:>9,}" if r.get('n_clusters') else ""
        lines.append(f"{title:<47} Number of obs   = {r['n']:>9,}")
        if cluster_info:
            lines.append(f"{'':<47} {cluster_info}")
        lines.append(f"{'':<47} F({r['df_m']:>2}, {r['df_r']:>5})    = {r['f_stat']:>9.2f}")
        lines.append(f"{'':<47} Prob > F        = {r['prob_f']:>9.4f}")
        lines.append(f"{'':<47} R-squared       = {r['r2']:>9.4f}")
        lines.append(f"{'':<47} Root MSE        = {r['root_mse']:>9.4f}")

    lines.append("")
    dep = r['depvar']
    if not is_robust:
        se_label = "Std. err."
    lines.append(f"{dep:>12} | Coefficient  {se_label:>16}      t    P>|t|     [95% conf. interval]")
    lines.append("-" * 13 + "+" + "-" * 68)
    for row in r['coefs']:
        name = row['name']
        if row.get('omitted'):
            lines.append(f"{name:>12} | {'(omitted)':>11}")
            continue
        b = row['coef']
        se = row['se']
        t = row['t']
        p = row['p']
        ci_l = row['ci_lower']
        ci_u = row['ci_upper']
        lines.append(f"{name:>12} | {b:>11.4f} {se:>16.4f} {t:>7.2f}   {p:>5.3f}    {ci_l:>11.4f} {ci_u:>11.4f}")
    lines.append("-" * 13 + "+" + "-" * 68)
    return "\n".join(lines)

def format_logit(r):
    lines = []
    if not r.get('converged', True):
        lines.append(">>> WARNING: ESTIMATES DID NOT CONVERGE <<<")
        status = r.get('convergence_status', 'Non-convergence')
        lines.append(f">>> Status: {status} (Standard errors and inference are invalid) <<<")
        lines.append("")
    title = "Logistic regression" if not r['odds_ratio'] else "Logistic regression (Odds Ratios)"
    lines.append(f"{title:<35} Number of obs   = {r['n']:>9,}")
    if r.get('n_clusters'):
        lines.append(f"{'':<35} Number of clusters = {r['n_clusters']:>9,}")
    lines.append(f"{'':<35} LR chi2({r['df_m']})      = {r['lr_chi2']:>9.2f}")
    lines.append(f"{'':<35} Prob > chi2     = {r['prob_chi2']:>9.4f}")
    lines.append(f"Log likelihood = {r['ll']:>14.4f}     Pseudo R2        = {r['pseudo_r2']:>9.4f}")
    lines.append("")
    if r.get('vce') == 'cluster' and r.get('n_clusters'):
        c_var = r.get('cluster_var', '')
        lines.append(f"(Std. err. adjusted for {r['n_clusters']} clusters in {c_var})")
        lines.append("")
    dep = r['depvar']
    col_name = "Odds ratio" if r['odds_ratio'] else "Coefficient"
    se_label = "Std. err."
    stat_label = "z"
    p_label = "P>|z|"
    lines.append(f"{dep:>12} | {col_name:>11}  {se_label:>16} {stat_label:>6}    {p_label:>5}     [95% conf. interval]")
    lines.append("-" * 13 + "+" + "-" * 68)
    for row in r['coefs']:
        name = row['name']
        b = row['coef']
        se = row['se']
        z = row['z']
        p = row['p']
        ci_l = row['ci_lower']
        ci_u = row['ci_upper']
        lines.append(f"{name:>12} | {b:>11.4f} {se:>16.4f} {z:>6.2f}   {p:>5.3f}    {ci_l:>11.4f} {ci_u:>11.4f}")
    lines.append("-" * 13 + "+" + "-" * 68)
    return "\n".join(lines)

def format_ivregress(r):
    lines = []
    lines.append(f"Instrumental variables (2SLS) regression       Number of obs   = {r['n']:>9,}")
    if r.get('n_clusters'):
        lines.append(f"                                               Number of clusters = {r['n_clusters']:>9,}")
    lines.append(f"                                               R-squared       = {r['r2']:>9.4f}")
    lines.append(f"                                               Root MSE        = {r['root_mse']:>9.4f}")
    lines.append("")
    dep = r['depvar']
    se_label = "Robust std. err." if r.get('vce') in ('robust', 'cluster') else "Std. err."
    is_small = bool(r.get('small', False))
    stat_col = "t" if is_small else "z"
    p_col = "P>|t|" if is_small else "P>|z|"
    lines.append(f"{dep:>12} | Coefficient  {se_label:>16}   {stat_col:>4}    {p_col:>5}     [95% conf. interval]")
    lines.append("-" * 13 + "+" + "-" * 68)
    for row in r['coefs']:
        name = row['name']
        b = row['coef']
        se = row['se']
        st = row.get(stat_col, row.get('stat', row.get('z', row.get('t', 0.0))))
        p = row['p']
        ci_l = row['ci_lower']
        ci_u = row['ci_upper']
        lines.append(f"{name:>12} | {b:>11.4f} {se:>16.4f} {st:>7.2f}   {p:>5.3f}    {ci_l:>11.4f} {ci_u:>11.4f}")
    lines.append("-" * 13 + "+" + "-" * 68)
    lines.append(f"Endogenous: {', '.join(r['endog_vars'])}")
    lines.append(f"Instruments: {', '.join(r['instruments'])}")
    return "\n".join(lines)

def format_panel_fe(r):
    lines = []
    lines.append(f"Fixed-effects (within) regression               Number of obs     = {r['n']:>10,}")
    lines.append(f"Group variable: {r['panelvar']:<16}                Number of groups  = {r['num_panels']:>10,}")
    if r.get('n_clusters'):
        lines.append(f"                                                Number of clusters = {r['n_clusters']:>10,}")
    lines.append("")
    lines.append("R-squared:                                      Obs per group:")
    min_o = r.get('min_obs', 0)
    avg_o = r.get('avg_obs', 0.0)
    max_o = r.get('max_obs', 0)
    lines.append(f"     within  = {r.get('r2_within', 0.0):<8.4f}                         min = {min_o:>10}")
    lines.append(f"     between = {r.get('r2_between', 0.0):<8.4f}                         avg = {avg_o:>10.1f}")
    lines.append(f"     overall = {r.get('r2_overall', 0.0):<8.4f}                         max = {max_o:>10}")
    lines.append("")
    k = len(r.get('indepvars', []))
    df_r = r.get('df_r', 0)
    f_stat = r.get('f_stat', 0.0)
    prob_f = r.get('prob_f', 0.0)
    corr_u_xb = r.get('corr_u_xb', 0.0)
    lines.append(f"                                                F({k},{df_r}){' ' * max(0, 14 - len(f'{k},{df_r}'))}= {f_stat:>10.2f}")
    lines.append(f"corr(u_i, Xb)  = {corr_u_xb:>8.4f}                        Prob > F          = {prob_f:>10.4f}")
    lines.append("")
    dep = r['depvar']
    se_label = "Robust std. err." if r.get('vce') in ('robust', 'cluster') else "Std. err."
    lines.append(f"{dep:>12} | Coefficient  {se_label:>16}      t    P>|t|     [95% conf. interval]")
    lines.append("-" * 13 + "+" + "-" * 68)
    for row in r['coefs']:
        name = row['name']
        b = row['coef']
        se = row['se']
        t = row['t']
        p = row['p']
        ci_l = row['ci_lower']
        ci_u = row['ci_upper']
        lines.append(f"{name:>12} | {b:>11.4f} {se:>16.4f} {t:>7.2f}   {p:>5.3f}    {ci_l:>11.4f} {ci_u:>11.4f}")
    lines.append("-" * 13 + "+" + "-" * 68)
    lines.append(f"     sigma_u | {r.get('sigma_u', 0.0):>11.4f}")
    lines.append(f"     sigma_e | {r.get('sigma_e', 0.0):>11.4f}")
    lines.append(f"         rho | {r.get('rho', 0.0):>11.4f}   (fraction of variance due to u_i)")
    lines.append("-" * 13 + "+" + "-" * 68)
    df_eff = r.get('df_effects', r.get('num_panels', 1) - 1)
    f_eff = r.get('f_effects', 0.0)
    p_eff = r.get('prob_f_effects', 1.0)
    lines.append(f"F test that all u_i=0: F({df_eff}, {df_r}) = {f_eff:.2f}               Prob > F = {p_eff:.4f}")
    lines.append("-" * 78)
    return "\n".join(lines)


def format_panel_re(r):
    lines = []
    lines.append(f"Random-effects GLS regression                   Number of obs     = {r['n']:>10,}")
    lines.append(f"Group variable: {r['panelvar']:<16}                Number of groups  = {r['num_panels']:>10,}")
    if r.get('n_clusters'):
        lines.append(f"                                                Number of clusters = {r['n_clusters']:>10,}")
    lines.append("")
    lines.append("R-sq:                                           Obs per group:")
    min_o = r.get('min_obs', 0)
    avg_o = r.get('avg_obs', 0.0)
    max_o = r.get('max_obs', 0)
    lines.append(f"     within  = {r.get('r2_within', 0.0):<8.4f}                         min = {min_o:>10}")
    lines.append(f"     between = {r.get('r2_between', 0.0):<8.4f}                         avg = {avg_o:>10.1f}")
    lines.append(f"     overall = {r.get('r2_overall', 0.0):<8.4f}                         max = {max_o:>10}")
    lines.append("")
    k = r.get('df_m', len(r.get('indepvars', [])))
    chi2_val = r.get('chi2', 0.0)
    prob_chi2 = r.get('prob_chi2', 0.0)
    lines.append(f"                                                Wald chi2({k}){' ' * max(0, 7 - len(str(k)))}= {chi2_val:>10.2f}")
    lines.append(f"corr(u_i, X)   = 0 (assumed)                    Prob > chi2       = {prob_chi2:>10.4f}")
    lines.append("")
    dep = r['depvar']
    se_label = "Robust std. err." if r.get('vce') in ('robust', 'cluster') else "Std. err."
    lines.append(f"{dep:>12} |      Coef.   {se_label:>12}      z    P>|z|     [95% conf. interval]")
    lines.append("-" * 13 + "+" + "-" * 64)
    for row in r['coefs']:
        name = row['name']
        b = row['coef']
        se = row['se']
        z = row.get('z', row.get('t', 0.0))
        p = row['p']
        ci_l = row['ci_lower']
        ci_u = row['ci_upper']
        lines.append(f"{name:>12} | {b:>10.4f}   {se:>12.4f} {z:>7.2f}   {p:>5.3f}   {ci_l:>10.4f}   {ci_u:>10.4f}")
    lines.append("-" * 13 + "+" + "-" * 64)
    lines.append(f"     sigma_u | {r.get('sigma_u', 0.0):>10.4f}")
    lines.append(f"     sigma_e | {r.get('sigma_e', 0.0):>10.4f}")
    lines.append(f"         rho | {r.get('rho', 0.0):>10.4f}   (fraction of variance due to u_i)")
    lines.append("-" * 78)
    return "\n".join(lines)


def format_hausman(r):
    lines = []
    lines.append("")
    lines.append("                 ---- Coefficients ----")
    lines.append("             |      (b)          (B)            (b-B)     sqrt(diag(V_b-V_B))")
    lines.append(f"             |  {r['name1']:>10}   {r['name2']:>10}      Difference          S.E.")
    lines.append("-" * 13 + "+" + "-" * 64)
    for row in r['rows']:
        name = row['name']
        b = row['b']
        B = row['B']
        diff = row['diff']
        se_diff = row['se_diff']
        se_str = f"{se_diff:>12.4f}" if se_diff is not None else "           ."
        lines.append(f"{name:>12} | {b:>11.4f}  {B:>11.4f}     {diff:>10.4f}      {se_str}")
    lines.append("-" * 13 + "+" + "-" * 64)
    lines.append("                           b = consistent under Ho and Ha; obtained from xtreg")
    lines.append("            B = inconsistent under Ha, efficient under Ho; obtained from xtreg")
    lines.append("")
    lines.append("    Test:  Ho:  difference in coefficients not systematic")
    lines.append("")
    df = r['df']
    chi2 = r['chi2']
    p = r['p']
    lines.append(f"                  chi2({df}) = (b-B)'[(V_b-V_B)^(-1)](b-B)")
    lines.append(f"                          = {chi2:>11.2f}")
    if r.get('not_pos_def'):
        lines.append("chi2 < 0: the model fails the asymptotic assumptions of the Hausman test")
    else:
        p = r.get('p', 0.0)
        lines.append(f"                Prob>chi2 = {p:>11.4f}")
    return "\n".join(lines)

def format_wald_test(r):
    lines = []
    restrictions = r.get('restrictions', [r['hypothesis']])
    for idx, rest in enumerate(restrictions, 1):
        lines.append(f" ({idx:>2})  {rest}")
    lines.append("")
    if r['test_type'] == 'F':
        df_den = r.get('df_denom', r.get('df_den', 1))
        lines.append(f"       F({r['df_num']:>3}, {df_den:>5}) = {r['stat']:>9.2f}")
        lines.append(f"            Prob > F = {r['p_value']:>9.4f}")
    else:
        lines.append(f"       chi2({r['df']:>2}) = {r['stat']:>9.2f}")
        lines.append(f"         Prob > chi2 = {r['p_value']:>9.4f}")
    return "\n".join(lines)

# --- Diagnostics Formatters ---

def format_vif(r):
    lines = []
    lines.append(f"{'Variable':<15} | {'VIF':>10} {'1/VIF (Tolerance)':>20}")
    lines.append("-" * 16 + "+" + "-" * 32)
    for row in r['rows']:
        lines.append(f"{row['variable']:<15} | {row['vif']:>10.2f} {row['tolerance']:>20.6f}")
    lines.append("-" * 16 + "+" + "-" * 32)
    lines.append(f"{'Mean VIF':<15} | {r['mean_vif']:>10.2f}")
    return "\n".join(lines)

def format_hettest(r):
    lines = []
    lines.append("Breusch-Pagan / Cook-Weisberg test for heteroskedasticity")
    lines.append("         Ho: Constant variance")
    lines.append("         Variables: Fitted values of depvar")
    lines.append("")
    lines.append(f"         chi2({r['df']})     = {r['chi2']:>9.2f}")
    lines.append(f"         Prob > chi2 = {r['p_value']:>9.4f}")
    return "\n".join(lines)

def format_brant_test(r, detail=False):
    lines = []
    if detail and r.get('binary_models'):
        bin_mods = r['binary_models']
        indeps = r.get('indepvars', [])
        lines.append("Estimated coefficients from binary logit models (detail):")
        
        # Build header with cutoffs
        header = f"{'Variable':<16} |"
        sep_len = 18
        for bm in bin_mods:
            header += f" {bm['cutoff']:>14}"
            sep_len += 15
        lines.append("-" * max(52, sep_len))
        lines.append(header)
        lines.append("-" * 17 + "+" + "-" * (max(52, sep_len) - 18))
        
        for v in indeps:
            row_b = f"{v:<16} |"
            row_se = f"{'':<16} |"
            for bm in bin_mods:
                b_val = bm['b'].get(v, 0.0)
                se_val = bm['se'].get(v, 0.0)
                row_b += f" {b_val:>14.4f}"
                row_se += f" {'(' + f'{se_val:.4f}' + ')':>14}"
            lines.append(row_b)
            lines.append(row_se)
            
        row_cons_b = f"{'_cons':<16} |"
        row_cons_se = f"{'':<16} |"
        for bm in bin_mods:
            b_c = bm.get('b_cons', 0.0)
            se_c = bm.get('se_cons', 0.0)
            row_cons_b += f" {b_c:>14.4f}"
            row_cons_se += f" {'(' + f'{se_c:.4f}' + ')':>14}"
        lines.append(row_cons_b)
        lines.append(row_cons_se)
        lines.append("-" * max(52, sep_len))
        lines.append("")

    lines.append("Brant test of parallel regression assumption")
    lines.append("-" * 52)
    lines.append(f"{'Variable':<16} | {'chi2':>10} {'p>chi2':>10} {'df':>8}")
    lines.append("-" * 17 + "+" + "-" * 34)
    for row in r['rows']:
        lines.append(f"{row['variable']:<16} | {row['chi2']:>10.2f} {row['p']:>10.3f} {row['df']:>8}")
    lines.append("-" * 17 + "+" + "-" * 34)
    lines.append("A significant test statistic provides evidence that the parallel")
    lines.append("regression assumption has been violated.")
    return "\n".join(lines)

def format_did_multiplegt(r):
    lines = []
    lines.append("de Chaisemartin & D'Haultfoeuille (2020) DID_M Estimator")
    lines.append("Difference-in-Differences with Switching / Reversible Treatments")
    lines.append(f"Number of obs = {r['n']:>9,}   Units         = {r['n_units']:>9,}")
    lines.append(f"Switchers     = {r['n_switchers']:>9,}   Controls      = {r['n_controls']:>9,}")
    lines.append("-" * 78)
    lines.append(f"{'Estimate':<28} | {'Estimate':>10} {'Std. err.':>10} {'z':>8} {'P>|z|':>8} {'[95% conf. interval]':>16}")
    lines.append("-" * 29 + "+" + "-" * 48)
    lines.append(f"{'Average instantaneous effect':<28} | {r['estimate']:>10.4f} {r['se']:>10.4f} {r['z']:>8.2f} {r['p']:>8.4f} {r['ci_lower']:>9.4f} {r['ci_upper']:>9.4f}")

    if r.get('dynamic_effects'):
        lines.append("-" * 29 + "+" + "-" * 48)
        lines.append(f"{'Dynamic Effects':<28} |")
        for d in r['dynamic_effects']:
            lines.append(f"  {d['horizon']:<26} | {d['estimate']:>10.4f} {d['se']:>10.4f} {d['z']:>8.2f} {d['p']:>8.4f} {d['ci_lower']:>9.4f} {d['ci_upper']:>9.4f}")

    if r.get('placebo_effects'):
        lines.append("-" * 29 + "+" + "-" * 48)
        lines.append(f"{'Placebos / Pre-trends':<28} |")
        for p in r['placebo_effects']:
            lines.append(f"  {p['horizon']:<26} | {p['estimate']:>10.4f} {p['se']:>10.4f} {p['z']:>8.2f} {p['p']:>8.4f} {p['ci_lower']:>9.4f} {p['ci_upper']:>9.4f}")

    lines.append("-" * 29 + "+" + "-" * 48)
    return "\n".join(lines)

def format_svy_regression(r):
    lines = []
    lines.append("Survey: Linear regression")
    lines.append("")
    strata_info = f"Number of strata   = {r['n_strata']:>9,}"
    obs_info = f"Number of obs   = {r['n']:>9,}"
    lines.append(f"{strata_info:<38} {obs_info}")
    psu_info = f"Number of PSUs     = {r['n_psus']:>9,}"
    pop_info = f"Population size = {r['pop_size']:>9,.1f}"
    lines.append(f"{psu_info:<38} {pop_info}")
    lines.append(f"{'':<38} Design df       = {r['df_r']:>9,}")
    lines.append(f"{'':<38} F({r['df_m']:>2}, {r['df_r']:>5})    = {r['f_stat']:>9.2f}")
    lines.append(f"{'':<38} Prob > F        = {r['prob_f']:>9.4f}")
    lines.append(f"{'':<38} R-squared       = {r['r2']:>9.4f}")
    lines.append("")
    dep = r['depvar']
    lines.append(f"{dep:>12} |             Linearized")
    lines.append(f"{'':>12} | Coefficient  std. err.      t    P>|t|     [95% conf. interval]")
    lines.append("-" * 13 + "+" + "-" * 68)
    for row in r['coefs']:
        name = row['name']
        b = row['coef']
        se = row['se']
        t = row['t']
        p = row['p']
        ci_l = row['ci_lower']
        ci_u = row['ci_upper']
        lines.append(f"{name:>12} | {b:>11.4f} {se:>10.4f} {t:>7.2f}   {p:>5.3f}    {ci_l:>11.4f} {ci_u:>11.4f}")
    lines.append("-" * 13 + "+" + "-" * 68)
    return "\n".join(lines)

def format_ovtest(r):
    lines = []
    lines.append("Ramsey RESET test using powers of the fitted values")
    lines.append("         Ho: Model has no omitted variables")
    df_denom = r.get('df_denom', r.get('df_den'))
    lines.append(f"         F({r['df_num']}, {df_denom})  = {r['f_stat']:>9.2f}")
    lines.append(f"         Prob > F = {r['p_value']:>9.4f}")
    return "\n".join(lines)

def format_dwatson(r):
    lines = []
    lines.append("Durbin-Watson d-statistic to test for first-order serial correlation")
    lines.append(f"         d-statistic = {r['dw']:>9.4f}")
    lines.append("         (Values near 2 indicate no autocorrelation; < 1.5 indicates positive autocorrelation)")
    return "\n".join(lines)

# --- Time Series & Forecasting Formatters ---

def format_dfuller(r, varname):
    lines = []
    lines.append(f"Augmented Dickey-Fuller test for unit root: {varname}")
    lines.append(f"Number of obs = {r['n_obs']}, Lags used = {r['lags_used']}")
    lines.append("-" * 70)
    lines.append(f"{'':<20} {'Test Statistic':>14} {'1% Critical':>12} {'5% Critical':>12} {'10% Critical':>12}")
    lines.append("-" * 70)
    c1 = r['critical_values'].get('1%', 0)
    c5 = r['critical_values'].get('5%', 0)
    c10 = r['critical_values'].get('10%', 0)
    lines.append(f"{'Z(t)':<20} {r['stat']:>14.4f} {c1:>12.4f} {c5:>12.4f} {c10:>12.4f}")
    lines.append("-" * 70)
    lines.append(f"MacKinnon approximate p-value for Z(t) = {r['p_value']:.4f}")
    return "\n".join(lines)

def format_corrgram(r, varname):
    lines = []
    lines.append(f"Autocorrelations and Partial Autocorrelations of {varname}")
    lines.append(f"Number of obs = {r['n']}")
    lines.append("-" * 65)
    lines.append(f"{'LAG':>4} {'ACF':>10} {'PACF':>10} {'Q-Stat':>12} {'Prob > Q':>12}")
    lines.append("-" * 65)
    for row in r['rows']:
        lines.append(f"{row['lag']:>4} {row['acf']:>10.4f} {row['pacf']:>10.4f} {row['q_stat']:>12.2f} {row['p_value']:>12.4f}")
    lines.append("-" * 65)
    return "\n".join(lines)

def format_arima(r, depvar):
    lines = []
    p, d, q = r['order']
    lines.append(f"ARIMA({p},{d},{q}) regression on {depvar:<10} Number of obs = {r['n']:>8}")
    lines.append(f"Log likelihood = {r['ll']:>12.4f}               AIC           = {r['aic']:>8.2f}")
    lines.append(f"{'':<45} BIC           = {r['bic']:>8.2f}")
    lines.append("")
    lines.append(f"{depvar:>12} | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]")
    lines.append("-" * 13 + "+" + "-" * 62)
    for row in r['coefs']:
        lines.append(f"{row['name']:>12} | {row['coef']:>11.4f} {row['se']:>10.4f} {row['z']:>7.2f}   {row['p']:>5.3f}    {row['ci_lower']:>11.4f} {row['ci_upper']:>11.4f}")
    lines.append("-" * 13 + "+" + "-" * 62)
    lines.append("")
    lines.append("Out-of-sample Forecast:")
    lines.append("  Step |    Predicted       [95% Conf. Interval]")
    lines.append("-" * 7 + "+" + "-" * 40)
    forecasts = r.get('forecasts', r.get('forecast', []))
    for row in forecasts:
        mean_val = row.get('mean', row.get('predicted', 0.0))
        lines.append(f"{row['step']:>6} | {mean_val:>12.4f}     {row['ci_lower']:>9.4f} {row['ci_upper']:>11.4f}")
    return "\n".join(lines)

# --- Causal Inference Formatters ---

def format_did(r):
    lines = []
    lines.append(f"Difference-in-Differences Estimation: {r['depvar']}")
    clust_msg = f" | Clustered by: {r['cluster_var']}" if r.get('cluster_var') else ""
    lines.append(f"Treatment: {r['treatvar']} | Time/Post: {r['timevar']} | Obs = {r['n']}{clust_msg}")
    lines.append("-" * 65)
    lines.append(f"{'Outcome Means':<20} | {'Pre-Period (0)':>14} {'Post-Period (1)':>15} {'Difference':>12}")
    lines.append("-" * 21 + "+" + "-" * 43)
    c0, c1, cd = r['control_pre'], r['control_post'], r['control_diff']
    t0, t1, td = r['treat_pre'], r['treat_post'], r['treat_diff']
    lines.append(f"{'Control (0)':<20} | {c0:>14.4f} {c1:>15.4f} {cd:>12.4f}")
    lines.append(f"{'Treated (1)':<20} | {t0:>14.4f} {t1:>15.4f} {td:>12.4f}")
    lines.append("-" * 21 + "+" + "-" * 43)
    lines.append(f"{'Diff-in-Diff (DiD)':<20} | {'':<14} {'':<15} {r['did_estimate']:>12.4f}")
    lines.append("-" * 65)
    lines.append(f"DiD Estimator: {r['did_estimate']:.4f} (Std. Err: {r['se']:.4f}, t = {r['t']:.2f}, p = {r['p']:.4f})")
    lines.append(f"95% Confidence Interval: [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
    return "\n".join(lines)

def format_psmatch(r):
    lines = []
    lines.append("Propensity Score Matching: Average Treatment Effect on the Treated (ATT)")
    lines.append(f"Treatment: {r['treatvar']} | Outcome: {r['outcome_var']} | Covariates: {', '.join(r['covariates'])}")
    lines.append(f"Sample: Treated = {r['n_treated']}, Control = {r['n_control']}")
    lines.append("-" * 60)
    lines.append(f"{'Variable':<15} {'Treated':>12} {'Control':>12} {'Difference':>12} {'S.E.':>9}")
    lines.append("-" * 60)
    lines.append(f"{r['outcome_var']:<15} {r['mean_treated']:>12.4f} {r['mean_control']:>12.4f} {r['att']:>12.4f} {r['se']:>9.4f}")
    lines.append("-" * 60)
    lines.append(f"ATT Estimate: {r['att']:.4f} (t = {r['t']:.2f}, p = {r['p']:.4f})")
    lines.append(f"95% Conf. Interval: [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
    return "\n".join(lines)

# --- Phase 2D Advanced Econometrics Formatters ---

def format_reghdfe(r):
    lines = []
    lines.append(f"HDFE Linear regression (absorbed {len(r['absorb_vars'])} FEs)")
    lines.append(f"Absorbed variables: {', '.join(r['absorb_vars'])}")
    lines.append(f"Number of obs = {r['n']:>9,}   F({r['k']:>3}, {r['df_r']:>5}) = {r['f_stat']:>8.2f}")
    lines.append(f"Prob > F      = {r['prob_f']:>9.4f}   R-squared     = {r['r2_overall']:>8.4f}")
    lines.append(f"Within R2     = {r['r2_within']:>9.4f}   Degrees abs.  = {r['df_absorbed']:>8,}")
    lines.append("-" * 78)
    lines.append(f"{r['depvar']:<14} | {'Coef.':>10} {'Std. err.':>10} {'t':>8} {'P>|t|':>8} {'[95% conf. interval]':>22}")
    lines.append("-" * 15 + "+" + "-" * 62)
    for c in r['coefs']:
        if c.get('omitted'):
            lines.append(f"{c['name']:<14} | {'(omitted)':>11}")
        else:
            lines.append(f"{c['name']:<14} | {c['coef']:>10.4f} {c['se']:>10.4f} {c['t']:>8.2f} {c['p']:>8.4f} {c['ci_lower']:>11.4f} {c['ci_upper']:>10.4f}")
    lines.append("-" * 15 + "+" + "-" * 62)
    return "\n".join(lines)

def format_probit(r):
    lines = []
    if not r.get('converged', True):
        lines.append(">>> WARNING: ESTIMATES DID NOT CONVERGE <<<")
        status = r.get('convergence_status', 'Non-convergence')
        lines.append(f">>> Status: {status} (Standard errors and inference are invalid) <<<")
        lines.append("")
    lines.append("Probit regression")
    lines.append(f"Number of obs = {r['n']:>9,}   LR chi2({r['df_m']:>3})  = {r['lr_chi2']:>8.2f}")
    if r.get('n_clusters'):
        lines.append(f"Number of clusters = {r['n_clusters']:>6,}")
    lines.append(f"Log likelihood= {r['ll']:>9.4f}   Prob > chi2   = {r['prob_chi2']:>8.4f}")
    lines.append(f"                               Pseudo R2     = {r['pseudo_r2']:>8.4f}")
    if r.get('vce') == 'cluster' and r.get('n_clusters'):
        c_var = r.get('cluster_var', '')
        lines.append(f"(Std. err. adjusted for {r['n_clusters']} clusters in {c_var})")
    lines.append("-" * 78)
    lines.append(f"{r['depvar']:<14} | {'Coef.':>10} {'Std. err.':>10} {'z':>8} {'P>|z|':>8} {'[95% conf. interval]':>22}")
    lines.append("-" * 15 + "+" + "-" * 62)
    for c in r['coefs']:
        lines.append(f"{c['name']:<14} | {c['coef']:>10.4f} {c['se']:>10.4f} {c['z']:>8.2f} {c['p']:>8.4f} {c['ci_lower']:>11.4f} {c['ci_upper']:>10.4f}")
    lines.append("-" * 15 + "+" + "-" * 62)
    return "\n".join(lines)


def format_nbreg(r):
    lines = []
    lines.append("Fitting Poisson model:")
    lines.append("Fitting full model:")
    lines.append("")
    header_title = "Negative binomial regression (IRR)" if r.get('irr') else "Negative binomial regression"
    lines.append(f"{header_title:<32}                Number of obs     = {r['n']:>10,}")
    if r.get('n_clusters'):
        lines.append(f"                                                Number of clusters = {r['n_clusters']:>10,}")
    lines.append(f"                                                LR chi2({r['df_m']}){' ' * max(0, 7 - len(str(r['df_m'])))} = {r['lr_chi2']:>10.2f}")
    lines.append(f"Dispersion     = mean                           Prob > chi2       = {r['prob_chi2']:>10.4f}")
    lines.append(f"Log likelihood = {r['ll']:<12.5f}                   Pseudo R2         = {r['pseudo_r2']:>10.4f}")
    lines.append("")
    coef_label = "IRR" if r.get('irr') else "Coef."
    se_label = "Robust std. err." if r.get('vce') in ('robust', 'cluster') else "Std. err."
    lines.append(f"{r['depvar']:>12} |      {coef_label}   {se_label:>12}      z    P>|z|     [95% conf. interval]")
    lines.append("-" * 13 + "+" + "-" * 64)
    aux_printed = False
    for row in r['coefs']:
        name = row['name']
        b = row['coef']
        se = row['se']
        z = row.get('z', np.nan)
        p = row.get('p', np.nan)
        ci_l = row['ci_lower']
        ci_u = row['ci_upper']
        if row.get('is_aux'):
            if not aux_printed:
                lines.append("-" * 13 + "+" + "-" * 64)
                aux_printed = True
            lines.append(f"{name:>12} | {b:>10.4f}   {se:>12.4f}                           {ci_l:>10.4f}   {ci_u:>10.4f}")
        else:
            lines.append(f"{name:>12} | {b:>10.4f}   {se:>12.4f} {z:>7.2f}   {p:>5.3f}   {ci_l:>10.4f}   {ci_u:>10.4f}")
    lines.append("-" * 13 + "+" + "-" * 64)
    chi2_c = r.get('chi2_c', 0.0)
    p_c = r.get('p_c', 0.0)
    lines.append(f"LR test of alpha=0: chibar2(01) = {chi2_c:.2f}                Prob >= chibar2 = {p_c:.3f}")
    if r.get('cluster_var'):
        lines.append(f"({r.get('n_clusters', 0)} clusters in {r.get('cluster_var')})")
    lines.append("-" * 78)
    return "\n".join(lines)


def format_mlogit(r):
    lines = []
    lines.append(f"Multinomial logistic regression                 Number of obs     = {r['n']:>10,}")
    if r.get('n_clusters'):
        lines.append(f"                                                Number of clusters = {r['n_clusters']:>10,}")
    lines.append(f"                                                LR chi2({r['df_m']}){' ' * max(0, 7 - len(str(r['df_m'])))} = {r['lr_chi2']:>10.2f}")
    lines.append(f"                                                Prob > chi2       = {r['prob_chi2']:>10.4f}")
    lines.append(f"Log likelihood = {r['ll']:<12.5f}                   Pseudo R2         = {r['pseudo_r2']:>10.4f}")
    lines.append("")
    coef_label = "RRR" if r.get('rrr') else "Coef."
    se_label = "Robust std. err." if r.get('vce') in ('robust', 'cluster') else "Std. err."
    lines.append(f"{r['depvar']:>12} |      {coef_label}   {se_label:>12}      z    P>|z|     [95% conf. interval]")
    lines.append("-" * 13 + "+" + "-" * 64)

    # Base outcome display
    base_out = r.get('baseoutcome')
    lines.append(f"{str(base_out):>12} |  (base outcome)")
    lines.append("-" * 13 + "+" + "-" * 64)

    # Equations display
    for eq in r.get('equations', []):
        eq_name = str(eq['outcome'])
        lines.append(f"{eq_name:<12} |")
        for row in eq['rows']:
            name = row['name']
            b = row['coef']
            se = row['se']
            z = row.get('z', np.nan)
            p = row.get('p', np.nan)
            ci_l = row['ci_lower']
            ci_u = row['ci_upper']
            lines.append(f"{name:>12} | {b:>10.4f}   {se:>12.4f} {z:>7.2f}   {p:>5.3f}   {ci_l:>10.4f}   {ci_u:>10.4f}")
        lines.append("-" * 13 + "+" + "-" * 64)

    if r.get('cluster_var'):
        lines.append(f"({r.get('n_clusters', 0)} clusters in {r.get('cluster_var')})")
        lines.append("-" * 78)
    return "\n".join(lines)


def format_cnsreg(r):
    lines = []
    lines.append(f"Constrained linear regression                   Number of obs     = {r['n']:>10,}")
    lines.append(f"                                                Root MSE          = {r['rmse']:>10.4f}")
    lines.append("")
    dep = r['depvar']
    lines.append(f"{dep:>12} |      Coef.   Std. Err.      t    P>|t|     [95% conf. interval]")
    lines.append("-" * 13 + "+" + "-" * 64)
    for c in r['coefs']:
        name = c['name']
        b = c['coef']
        if c.get('constrained'):
            lines.append(f"{name:>12} | {b:>10.4f}  (constrained)")
        else:
            se = c['se']
            t = c['t']
            p = c['p']
            ci_l = c['ci_lower']
            ci_u = c['ci_upper']
            lines.append(f"{name:>12} | {b:>10.4f}   {se:>12.4f} {t:>7.2f}   {p:>5.3f}   {ci_l:>10.4f}   {ci_u:>10.4f}")
    lines.append("-" * 78)
    return "\n".join(lines)


def format_sureg(r):
    lines = []
    lines.append("Seemingly unrelated regression")
    lines.append("-" * 74)
    lines.append(f"{'Equation':<16} {'Obs':>6} {'Parms':>6} {'RMSE':>12} {'R-sq':>10} {'chi2':>10} {'P':>8}")
    lines.append("-" * 74)
    for eq in r['equations']:
        dep = eq['depvar']
        parms = eq['parms']
        rmse = eq['rmse']
        r2 = eq['r2']
        chi2 = eq['chi2']
        p = eq['p']
        lines.append(f"{dep:<16} {r['n']:>6} {parms:>6} {rmse:>12.4f} {r2:>10.4f} {chi2:>10.2f} {p:>8.4f}")
    lines.append("-" * 74)
    lines.append("")
    lines.append(f"{'':>12} |      Coef.   Std. Err.      z    P>|z|     [95% conf. interval]")
    lines.append("-" * 13 + "+" + "-" * 64)
    for eq in r['equations']:
        dep = eq['depvar']
        lines.append(f"{dep:<12} |")
        for row in eq['rows']:
            name = row['name']
            b = row['coef']
            se = row['se']
            z = row['z']
            p = row['p']
            ci_l = row['ci_lower']
            ci_u = row['ci_upper']
            lines.append(f"{name:>12} | {b:>10.4f}   {se:>12.4f} {z:>7.2f}   {p:>5.3f}   {ci_l:>10.4f}   {ci_u:>10.4f}")
        lines.append("-" * 13 + "+" + "-" * 64)
    return "\n".join(lines)


def format_nlcom(r):
    lines = []
    lines.append(f"       _nl_1:  {r['expr']}")
    lines.append("-" * 78)
    lines.append(f"{'':>12} |      Coef.   Std. Err.      z    P>|z|     [95% conf. interval]")
    lines.append("-" * 13 + "+" + "-" * 64)
    b = r['estimate']
    se = r['se']
    z = r['z']
    p = r['p']
    ci_l = r['ci_lower']
    ci_u = r['ci_upper']
    lines.append(f"{'_nl_1':>12} | {b:>10.4f}   {se:>12.4f} {z:>7.2f}   {p:>5.3f}   {ci_l:>10.4f}   {ci_u:>10.4f}")
    lines.append("-" * 78)
    return "\n".join(lines)


def format_testnl(r):
    lines = []
    lines.append(f"  (1)  {r['hypothesis']}")
    lines.append("")
    lines.append(f"               chi2({r['df']}) = {r['chi2']:>10.2f}")
    lines.append(f"           Prob > chi2 = {r['p']:>10.4f}")
    return "\n".join(lines)


def format_var(r):
    lines = []
    lines.append("Vector autoregression")
    lines.append(f"Sample: {r['n']} observations   No. of equations = {len(r['equations'])}")
    lines.append(f"Log likelihood = {r['ll']:<12.4f}   AIC = {r['aic']:>10.4f}   SBIC = {r['bic']:>10.4f}")
    lines.append("-" * 78)
    lines.append(f"{'':>12} |      Coef.   Std. Err.      z    P>|z|     [95% conf. interval]")
    lines.append("-" * 13 + "+" + "-" * 64)
    for eq in r['equations']:
        dep = eq['depvar']
        lines.append(f"{dep:<12} |")
        for row in eq['rows']:
            name = row['name']
            b = row['coef']
            se = row['se']
            z = row['z']
            p = row['p']
            ci_l = row['ci_lower']
            ci_u = row['ci_upper']
            lines.append(f"{name:>12} | {b:>10.4f}   {se:>12.4f} {z:>7.2f}   {p:>5.3f}   {ci_l:>10.4f}   {ci_u:>10.4f}")
        lines.append("-" * 13 + "+" + "-" * 64)
    return "\n".join(lines)


def format_stcox(r):
    lines = []
    lines.append("Cox proportional hazards regression")
    lines.append(f"No. of subjects = {r['n']:>8,}   No. of failures = {r['n_events']:>8,}")
    lines.append(f"Log likelihood  = {r['ll']:>10.4f}   Ties method: {r.get('ties', 'breslow').capitalize()}")
    lines.append("-" * 78)
    lines.append(f"{'_t':>12} | Haz. ratio   Std. Err.      z    P>|z|     [95% conf. interval]")
    lines.append("-" * 13 + "+" + "-" * 64)
    for c in r['coefs']:
        name = c['name']
        hr = c['hr']
        se = c['hr_se']
        z = c['z']
        p = c['p']
        ci_l = c['ci_lower']
        ci_u = c['ci_upper']
        lines.append(f"{name:>12} | {hr:>10.4f}   {se:>12.4f} {z:>7.2f}   {p:>5.3f}   {ci_l:>10.4f}   {ci_u:>10.4f}")
    lines.append("-" * 78)
    return "\n".join(lines)

def format_ologit(r):
    lines = []
    if not r.get('converged', True):
        lines.append(">>> WARNING: ESTIMATES DID NOT CONVERGE <<<")
        status = r.get('convergence_status', 'Non-convergence')
        lines.append(f">>> Status: {status} (Standard errors and inference are invalid) <<<")
        lines.append("")
    lines.append("Ordered logistic regression")
    lines.append(f"Number of obs = {r['n']:>9,}   LR chi2({r['df_m']:>3})  = {r['lr_chi2']:>8.2f}")
    lines.append(f"Log likelihood= {r['ll']:>9.4f}   Prob > chi2   = {r['prob_chi2']:>8.4f}")
    lines.append(f"                               Pseudo R2     = {r['pseudo_r2']:>8.4f}")
    lines.append("-" * 78)
    coef_label = "Odds ratio" if r.get('odds_ratio') else "Coefficient"
    lines.append(f"{r['depvar']:<14} | {coef_label:>11} {'Std. err.':>10} {'z':>8} {'P>|z|':>8} {'[95% conf. interval]':>22}")
    lines.append("-" * 15 + "+" + "-" * 62)
    for c in r['coefs']:
        if c.get('is_cutpoint'):
            lines.append(f"{c['name']:<14} | {c['coef']:>11.4f} {c['se']:>10.4f} {'':>8} {'':>8} {c['ci_lower']:>11.4f} {c['ci_upper']:>10.4f}")
        else:
            lines.append(f"{c['name']:<14} | {c['coef']:>11.4f} {c['se']:>10.4f} {c['z']:>8.2f} {c['p']:>8.4f} {c['ci_lower']:>11.4f} {c['ci_upper']:>10.4f}")
    lines.append("-" * 15 + "+" + "-" * 62)
    return "\n".join(lines)

def format_poisson(r):
    lines = []
    if not r.get('converged', True):
        lines.append(">>> WARNING: ESTIMATES DID NOT CONVERGE <<<")
        status = r.get('convergence_status', 'Non-convergence')
        lines.append(f">>> Status: {status} (Standard errors and inference are invalid) <<<")
        lines.append("")
    header_title = "Poisson regression (IRR)" if r.get('irr') else "Poisson regression"
    lines.append(header_title)
    lines.append(f"Number of obs = {r['n']:>9,}   LR chi2({r['df_m']:>3})  = {r['lr_chi2']:>8.2f}")
    lines.append(f"Log likelihood= {r['ll']:>9.4f}   Prob > chi2   = {r['prob_chi2']:>8.4f}")
    lines.append(f"                               Pseudo R2     = {r['pseudo_r2']:>8.4f}")
    lines.append("-" * 78)
    coef_label = "IRR" if r.get('irr') else "Coef."
    lines.append(f"{r['depvar']:<14} | {coef_label:>10} {'Std. err.':>10} {'z':>8} {'P>|z|':>8} {'[95% conf. interval]':>22}")
    lines.append("-" * 15 + "+" + "-" * 62)
    for c in r['coefs']:
        lines.append(f"{c['name']:<14} | {c['coef']:>10.4f} {c['se']:>10.4f} {c['z']:>8.2f} {c['p']:>8.4f} {c['ci_lower']:>11.4f} {c['ci_upper']:>10.4f}")
    lines.append("-" * 15 + "+" + "-" * 62)
    return "\n".join(lines)

def format_tobit(r):
    lines = []
    if not r.get('converged', True):
        lines.append(">>> WARNING: ESTIMATES DID NOT CONVERGE <<<")
        status = r.get('convergence_status', 'Non-convergence')
        lines.append(f">>> Status: {status} (Standard errors and inference are invalid) <<<")
        lines.append("")
    lines.append("Tobit regression")
    lines.append(f"Number of obs = {r['n']:>9,}   Uncensored obs = {r['n_uncensored']:>8,}")
    lines.append(f"Left-censored = {r['n_left']:>9,}   Right-censored = {r['n_right']:>8,}")
    lines.append(f"Log likelihood= {r['ll']:>9.4f}")
    lines.append("-" * 78)
    lines.append(f"{r['depvar']:<14} | {'Coef.':>10} {'Std. err.':>10} {'t':>8} {'P>|t|':>8} {'[95% conf. interval]':>22}")
    lines.append("-" * 15 + "+" + "-" * 62)
    for c in r['coefs']:
        lines.append(f"{c['name']:<14} | {c['coef']:>10.4f} {c['se']:>10.4f} {c['z']:>8.2f} {c['p']:>8.4f} {c['ci_lower']:>11.4f} {c['ci_upper']:>10.4f}")
    lines.append("-" * 15 + "+" + "-" * 62)
    return "\n".join(lines)

def format_qreg(r):
    lines = []
    if not r.get('converged', True):
        lines.append(">>> WARNING: ESTIMATES DID NOT CONVERGE <<<")
        status = r.get('convergence_status', 'Non-convergence')
        lines.append(f">>> Status: {status} (Standard errors and inference are invalid) <<<")
        lines.append("")
    lines.append(f"Quantile regression (.q = {r['quantile']:.2f})")
    lines.append(f"Number of obs = {r['n']:>9,}   Degrees of freedom = {r['df_r']:>6,}")
    lines.append("-" * 78)
    lines.append(f"{r['depvar']:<14} | {'Coef.':>10} {'Std. err.':>10} {'t':>8} {'P>|t|':>8} {'[95% conf. interval]':>22}")
    lines.append("-" * 15 + "+" + "-" * 62)
    for c in r['coefs']:
        if c.get('omitted'):
            lines.append(f"{c['name']:<14} | {'(omitted)':>11}")
        else:
            lines.append(f"{c['name']:<14} | {c['coef']:>10.4f} {c['se']:>10.4f} {c['t']:>8.2f} {c['p']:>8.4f} {c['ci_lower']:>11.4f} {c['ci_upper']:>10.4f}")
    lines.append("-" * 15 + "+" + "-" * 62)
    return "\n".join(lines)

def format_margins(r):
    lines = []
    lines.append("Average marginal effects (AME)")
    lines.append(f"Model: {r['model_type']} | Dep. variable: {r['depvar']} | Obs = {r['n']:,}")
    lines.append("-" * 78)
    lines.append(f"{'Variable':<14} | {'dy/dx':>10} {'Std. err.':>10} {'z':>8} {'P>|z|':>8} {'[95% conf. interval]':>22}")
    lines.append("-" * 15 + "+" + "-" * 62)
    for m in r['margins']:
        lines.append(f"{m['variable']:<14} | {m['dy_dx']:>10.4f} {m['se']:>10.4f} {m['z']:>8.2f} {m['p']:>8.4f} {m['ci_lower']:>11.4f} {m['ci_upper']:>10.4f}")
    lines.append("-" * 15 + "+" + "-" * 62)
    return "\n".join(lines)

def format_lincom(r):
    lines = []
    lines.append(f" ( 1)  {r['expression']} = 0")
    lines.append("-" * 78)
    lines.append(f"{'Linear comb.':<14} | {'Coef.':>10} {'Std. err.':>10} {'t':>8} {'P>|t|':>8} {'[95% conf. interval]':>22}")
    lines.append("-" * 15 + "+" + "-" * 62)
    lines.append(f"{'(1)':<14} | {r['estimate']:>10.4f} {r['se']:>10.4f} {r['t']:>8.2f} {r['p']:>8.4f} {r['ci_lower']:>11.4f} {r['ci_upper']:>10.4f}")
    lines.append("-" * 15 + "+" + "-" * 62)
    return "\n".join(lines)

def format_csdid(r):
    lines = []
    lines.append("Callaway & Sant'Anna (2021) Staggered Difference-in-Differences")
    lines.append(f"Outcome: {r['depvar']} | Unit: {r['ivar']} | Time: {r['timevar']} | Cohort: {r['gvar']}")
    lines.append(f"Obs: {r['n_obs']:,} | Panel units: {r['n_units']:,} | Treated cohorts: {len(r['cohorts'])}")
    lines.append("-" * 78)
    lines.append(f"Overall ATT = {r['overall_att']:.4f}  (Std. Err: {r['overall_se']:.4f}, z = {r['overall_z']:.2f}, p = {r['overall_p']:.4f})")
    lines.append(f"95% Confidence Interval: [{r['overall_ci_lower']:.4f}, {r['overall_ci_upper']:.4f}]")
    lines.append(f"Pre-trend test: chi2({r['pretrend_df']}) = {r['pretrend_chi2']:.2f}, Prob > chi2 = {r['pretrend_p']:.4f}")
    lines.append("-" * 78)
    lines.append(f"{'Event Time (e)':<16} | {'ATT':>10} {'Std. err.':>10} {'z':>8} {'P>|z|':>8} {'[95% conf. interval]':>22}")
    lines.append("-" * 17 + "+" + "-" * 60)
    for row in r['dynamic_effects']:
        lines.append(f"e = {row['event_time']:<12} | {row['att']:>10.4f} {row['se']:>10.4f} {row['z']:>8.2f} {row['p']:>8.4f} {row['ci_lower']:>11.4f} {row['ci_upper']:>10.4f}")
    lines.append("-" * 17 + "+" + "-" * 60)
    return "\n".join(lines)

def format_synth(r):
    lines = []
    lines.append("Synthetic Control Method Estimation")
    lines.append(f"Treated Unit: {r['treated_id']} | Treatment Time: {r['treat_time']} | Donor Units: {r['n_donors']}")
    lines.append(f"Pre-treatment MSPE: {r['pre_mspe']:.4f} | Post-treatment MSPE: {r['post_mspe']:.4f}")
    lines.append(f"Post/Pre MSPE Ratio: {r['mspe_ratio']:.2f}")
    if not np.isnan(r.get('p_value_placebo', np.nan)):
        lines.append(f"In-space Placebo Permutation p-value: {r['p_value_placebo']:.4f}")
    lines.append("-" * 55)
    lines.append(f"{'Donor Unit':<25} | {'Synthetic Weight':>20}")
    lines.append("-" * 26 + "+" + "-" * 28)
    for w in r['weights']:
        lines.append(f"{w['donor_id']:<25} | {w['weight']:>20.4f}")
    lines.append("-" * 55)
    return "\n".join(lines)

def format_event_study(r):
    lines = []
    lines.append("Dynamic Event-Study Regression (TWFE)")
    lines.append(f"Outcome: {r['depvar']} | Obs = {r['n']:,} | Within R2 = {r['r2']:.4f}")
    lines.append(f"Pre-trend test: chi2({r['pretrend_df']}) = {r['pretrend_chi2']:.2f}, Prob > chi2 = {r['pretrend_p']:.4f}")
    lines.append("-" * 78)
    lines.append(f"{'Period':<16} | {'Estimate':>10} {'Std. err.':>10} {'[95% conf. interval]':>22}")
    lines.append("-" * 17 + "+" + "-" * 44)
    for row in r['event_table']:
        ref_mark = " (ref)" if row['k'] == -1 else ""
        lines.append(f"k = {row['k']:<8}{ref_mark:<5} | {row['coef']:>10.4f} {row['se']:>10.4f} {row['ci_lower']:>11.4f} {row['ci_upper']:>10.4f}")
    lines.append("-" * 17 + "+" + "-" * 44)
    return "\n".join(lines)
