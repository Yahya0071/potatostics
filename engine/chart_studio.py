"""
engine/chart_studio.py - Academic Chart Studio & Vector Graphics Exporter
Themes:
- 'aer': American Economic Review (serif, black/grey, clean minimalist top/right spine removal)
- 'nature': Nature / Science (clean sans-serif, high-contrast, publication ready)
- 'economist': The Economist (signature bold typography, red accent, light gridlines)

Vector Exports: .pdf, .svg, .eps, and 300/600 DPI .png.
"""

import matplotlib
# Use Agg backend when running headless or exporting
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Colorblind-safe Tol / Okabe-Ito Palette
OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9', '#000000', '#E69F00']


def apply_academic_theme(ax, theme='aer', title=None, xlabel=None, ylabel=None):
    """Applies academic journal aesthetic styling to a matplotlib axis."""
    theme = theme.lower() if theme else 'aer'

    if theme == 'aer':
        plt.rcParams['font.family'] = 'serif'
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#222222')
        ax.spines['bottom'].set_color('#222222')
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.yaxis.grid(True, linestyle=':', alpha=0.5, color='#888888')
        ax.xaxis.grid(False)
        ax.set_facecolor('white')

    elif theme == 'nature':
        plt.rcParams['font.family'] = 'sans-serif'
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.yaxis.grid(True, linestyle='-', alpha=0.2, color='#CCCCCC')
        ax.xaxis.grid(False)
        ax.set_facecolor('white')

    elif theme == 'economist':
        plt.rcParams['font.family'] = 'sans-serif'
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#333333')
        ax.spines['bottom'].set_linewidth(1.5)
        ax.yaxis.grid(True, linestyle='-', alpha=0.6, color='#D5D5D5')
        ax.xaxis.grid(False)
        ax.set_facecolor('#F0F4F8')

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12, loc='left')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, labelpad=8)


def plot_event_study(event_result, theme='aer', save_path=None):
    """
    Plots leads-and-lags dynamic event-study with 95% confidence intervals
    and pre-trend Wald test annotation.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    table = event_result['event_table']
    periods = [r['k'] for r in table]
    coefs = [r['coef'] for r in table]
    ci_lower = [r['ci_lower'] for r in table]
    ci_upper = [r['ci_upper'] for r in table]

    err_low = np.array(coefs) - np.array(ci_lower)
    err_high = np.array(ci_upper) - np.array(coefs)

    # Reference zero line
    ax.axhline(0, color='#888888', linestyle='--', linewidth=1.0)
    # Reference event time vertical line at -0.5
    ax.axvline(-0.5, color='#D9534F', linestyle=':', linewidth=1.2, label='Treatment Event')

    # Marker and error bars
    marker_color = '#1f77b4' if theme != 'aer' else '#111111'
    ax.errorbar(
        periods, coefs, yerr=[err_low, err_high],
        fmt='o', color=marker_color, ecolor=marker_color,
        elinewidth=1.4, capsize=3.5, capthick=1.2, markersize=5.5
    )

    # Pre-trend test annotation
    p_pre = event_result.get('pretrend_p', 1.0)
    note = f"Pre-trend test p-value = {p_pre:.3f}"
    ax.text(0.03, 0.92, note, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))

    title = f"Dynamic Event Study: {event_result.get('depvar', 'Outcome')}"
    apply_academic_theme(ax, theme=theme, title=title, xlabel="Event Time (Periods Relative to Treatment)", ylabel="Estimate and 95% CI")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    return fig


def plot_coefplot(model_result, theme='aer', save_path=None):
    """Plots regression coefficient point estimates and confidence intervals."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    coefs_data = [c for c in model_result.get('coefs', []) if c['name'] not in ('_cons', 'Intercept')]
    if not coefs_data:
        coefs_data = model_result.get('coefs', [])

    names = [c['name'] for c in reversed(coefs_data)]
    betas = [c['coef'] for c in reversed(coefs_data)]
    ci_low = [c['ci_lower'] for c in reversed(coefs_data)]
    ci_high = [c['ci_upper'] for c in reversed(coefs_data)]

    y_pos = np.arange(len(names))
    err_left = np.array(betas) - np.array(ci_low)
    err_right = np.array(ci_high) - np.array(betas)

    ax.axvline(0, color='#888888', linestyle='--', linewidth=1.0)
    color = '#111111' if theme == 'aer' else '#1f77b4'

    ax.errorbar(
        betas, y_pos, xerr=[err_left, err_right],
        fmt='o', color=color, ecolor=color, elinewidth=1.5,
        capsize=3.5, markersize=5.5
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)

    title = f"Coefficient Estimates: {model_result.get('depvar', '')}"
    apply_academic_theme(ax, theme=theme, title=title, xlabel="Coefficient and 95% CI")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    return fig




def plot_forest(studies, theme='aer', save_path=None):
    """
    Plots an academic meta-analysis Forest Plot with inverse-variance study weights,
    confidence intervals, and pooled summary diamond.
    """
    import scipy.stats as stats
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, len(studies) * 0.5 + 2.0)), dpi=150)

    k = len(studies)
    thetas = np.array([s['theta'] for s in studies])
    ses = np.array([s['se'] for s in studies])
    names = [s.get('study', f'Study {i+1}') for i, s in enumerate(studies)]

    # Inverse-variance weights
    weights = 1.0 / (ses ** 2)
    weights_pct = 100.0 * weights / np.sum(weights)

    # Pooled effect
    pooled_theta = float(np.sum(weights * thetas) / np.sum(weights))
    pooled_se = float(1.0 / np.sqrt(np.sum(weights)))
    pooled_ci_low = pooled_theta - 1.96 * pooled_se
    pooled_ci_high = pooled_theta + 1.96 * pooled_se

    # Heterogeneity: Cochran's Q and I^2
    q_stat = float(np.sum(weights * (thetas - pooled_theta) ** 2))
    df_q = max(1, k - 1)
    p_q = float(1.0 - stats.chi2.cdf(q_stat, df_q))
    i_squared = max(0.0, 100.0 * (q_stat - df_q) / q_stat) if q_stat > 0 else 0.0

    # Draw null reference line
    ax.axvline(0, color='#888888', linestyle='--', linewidth=1.0)
    # Draw pooled vertical dashed line
    ax.axvline(pooled_theta, color='#2c7bb6', linestyle=':', linewidth=1.0, alpha=0.7)

    y_coords = np.arange(k, 0, -1)

    for i in range(k):
        y = y_coords[i]
        th = thetas[i]
        se = ses[i]
        ci_l = th - 1.96 * se
        ci_u = th + 1.96 * se
        wt = weights_pct[i]

        # Horizontal error bar
        ax.plot([ci_l, ci_u], [y, y], color='#222222', linewidth=1.4)
        # Study square proportional to weight
        sq_size = 4.0 + 8.0 * (wt / np.max(weights_pct))
        ax.plot(th, y, marker='s', markersize=sq_size, color='#111111', markeredgecolor='black')

    # Draw pooled diamond at y = 0
    y_diamond = 0
    diamond_pts = np.array([
        [pooled_ci_low, y_diamond],
        [pooled_theta, y_diamond + 0.28],
        [pooled_ci_high, y_diamond],
        [pooled_theta, y_diamond - 0.28]
    ])
    diamond = Polygon(diamond_pts, closed=True, facecolor='#d7191c', edgecolor='#990000', alpha=0.85, zorder=5)
    ax.add_patch(diamond)

    # Labels and ticks
    all_y = list(y_coords) + [0]
    all_names = names + ['Overall (Fixed)']
    ax.set_yticks(all_y)
    ax.set_yticklabels(all_names, fontsize=9.5)
    ax.set_ylim(-1.0, k + 1.0)

    title = f"Meta-Analysis Forest Plot (I-squared = {i_squared:.1f}%, p = {p_q:.3f})"
    apply_academic_theme(ax, theme=theme, title=title, xlabel="Effect Size and 95% CI")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    return fig




def plot_binscatter(df, yvar, xvar, byvar=None, n_bins=20, controls=None, line='linear', theme='aer', save_path=None, ax=None):
    """
    Renders publication-grade binned scatter plot (Cattaneo, Crump, Farrell & Feng 2024 / Chetty et al.).
    Computes equal-frequency quantile bins with OLS fit and HC1 robust standard errors.
    """
    import statsmodels.api as sm

    cols = [yvar, xvar]
    if byvar and byvar not in cols:
        cols.append(byvar)
    if controls:
        cols.extend([c for c in controls if c not in cols])
    data = df[cols].dropna().copy()

    if len(data) < n_bins:
        raise ValueError(f"Insufficient observations ({len(data)}) for {n_bins} bins.")

    # Frisch-Waugh-Lovell partialling out if controls specified
    if controls:
        W = sm.add_constant(data[controls])
        y_res = sm.OLS(data[yvar], W).fit().resid + data[yvar].mean()
        x_res = sm.OLS(data[xvar], W).fit().resid + data[xvar].mean()
        data[yvar] = y_res
        data[xvar] = x_res

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
        created_fig = True
    else:
        fig = ax.figure

    groups = [None] if not byvar else sorted(data[byvar].unique())
    group_results = {}
    all_binned = []

    overall_slope = 0.0
    overall_se = 0.0
    overall_r2 = 0.0

    for idx, g in enumerate(groups):
        sub_df = data if g is None else data[data[byvar] == g].copy()
        if len(sub_df) < n_bins:
            continue
        color = OKABE_ITO[idx % len(OKABE_ITO)]
        label = f"{byvar} = {g}" if g is not None else None

        # Equal-frequency quantile binning (pd.qcut)
        q_count = min(n_bins, len(sub_df) // 2)
        sub_df['bin'] = pd.qcut(sub_df[xvar], q=q_count, duplicates='drop')
        binned = sub_df.groupby('bin', observed=True).agg(
            x_mean=(xvar, 'mean'),
            y_mean=(yvar, 'mean'),
            count=(yvar, 'count')
        ).dropna().reset_index()

        # Fit OLS line with HC1 standard errors
        X_mat = sm.add_constant(sub_df[xvar])
        ols_mod = sm.OLS(sub_df[yvar], X_mat).fit(cov_type='HC1')
        slope = float(ols_mod.params[xvar])
        se = float(ols_mod.bse[xvar])
        r2 = float(ols_mod.rsquared)

        if idx == 0:
            overall_slope = slope
            overall_se = se
            overall_r2 = r2

        # Scatter points
        ax.scatter(binned['x_mean'], binned['y_mean'], color=color, s=40, label=label, alpha=0.9, edgecolors='none', zorder=4)

        # Fitted regression line
        if line == 'linear':
            x_line = np.linspace(sub_df[xvar].quantile(0.01), sub_df[xvar].quantile(0.99), 100)
            y_line = ols_mod.predict(sm.add_constant(x_line))
            ax.plot(x_line, y_line, color=color, linewidth=1.6, zorder=3)

        group_results[g if g is not None else 'all'] = {
            'binned': binned,
            'slope': slope,
            'se': se,
            'r2': r2
        }
        all_binned.append(binned)

    if byvar and len(groups) > 1:
        ax.legend(frameon=False, loc='best', fontsize=9)

    title = f"Binned Scatter: {yvar} on {xvar}"
    if controls:
        title += f" (controlling for {', '.join(controls)})"
    apply_academic_theme(ax, theme=theme, title=title, xlabel=str(xvar), ylabel=str(yvar))

    binned_combined = pd.concat(all_binned, ignore_index=True) if all_binned else pd.DataFrame()

    if save_path and created_fig:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    return {
        'bins': binned_combined,
        'slope': overall_slope,
        'se': overall_se,
        'r2': overall_r2,
        'groups': group_results,
        'figure': fig,
        'ax': ax
    }


def plot_rd(df, yvar, xvar, cutoff=0.0, bandwidth=None, n_bins=20, p=1, theme='aer', save_path=None, ax=None):
    """
    Renders publication-grade Regression Discontinuity (RD) plot
    (Calonico, Cattaneo, & Titiunik 2014, 2015 JASA).
    Computes binned sample means on each side of the cutoff, estimates local linear
    regression with HC1 robust standard errors and 95% confidence bands.
    """
    import statsmodels.api as sm

    data = df[[yvar, xvar]].dropna().copy()
    cutoff = float(cutoff)
    data['d'] = (data[xvar] >= cutoff).astype(float)
    data['x_c'] = data[xvar] - cutoff
    data['d_x_c'] = data['d'] * data['x_c']

    # Default bandwidth: Silverman rule of thumb if bandwidth is None
    if bandwidth is None:
        std_x = float(data[xvar].std())
        iqr_x = float(data[xvar].quantile(0.75) - data[xvar].quantile(0.25))
        spread = min(std_x, iqr_x / 1.349) if iqr_x > 0 else std_x
        bandwidth = float(1.06 * spread * (len(data) ** (-0.2)))
        bandwidth = max(bandwidth, 0.1 * std_x)
    else:
        bandwidth = float(bandwidth)

    # Binning on each side
    left_data = data[data[xvar] < cutoff].copy()
    right_data = data[data[xvar] >= cutoff].copy()

    n_bins_side = max(5, n_bins // 2)
    left_data['bin'] = pd.cut(left_data[xvar], bins=min(n_bins_side, max(3, len(left_data) // 5)))
    right_data['bin'] = pd.cut(right_data[xvar], bins=min(n_bins_side, max(3, len(right_data) // 5)))

    left_bins = left_data.groupby('bin', observed=True).agg(
        x_mean=(xvar, 'mean'),
        y_mean=(yvar, 'mean'),
        count=(yvar, 'count')
    ).dropna().reset_index()

    right_bins = right_data.groupby('bin', observed=True).agg(
        x_mean=(xvar, 'mean'),
        y_mean=(yvar, 'mean'),
        count=(yvar, 'count')
    ).dropna().reset_index()

    # Local linear estimation within bandwidth
    sub_rd = data[np.abs(data['x_c']) <= bandwidth].copy()
    if len(sub_rd) < 10:
        sub_rd = data.copy()

    X_rd = sm.add_constant(sub_rd[['d', 'x_c', 'd_x_c']])
    model_rd = sm.OLS(sub_rd[yvar], X_rd).fit(cov_type='HC1')
    tau = float(model_rd.params['d'])
    se_tau = float(model_rd.bse['d'])
    t_stat = tau / se_tau if se_tau > 0 else 0.0
    p_val = float(2.0 * (1.0 - stats.norm.cdf(np.abs(t_stat))))
    ci_lower = tau - 1.96 * se_tau
    ci_upper = tau + 1.96 * se_tau

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
        created_fig = True
    else:
        fig = ax.figure

    # Plot binned points
    ax.scatter(left_bins['x_mean'], left_bins['y_mean'], color=OKABE_ITO[0], s=35, alpha=0.85, zorder=4)
    ax.scatter(right_bins['x_mean'], right_bins['y_mean'], color=OKABE_ITO[1], s=35, alpha=0.85, zorder=4)

    # Plot cutoff vertical dashed line
    ax.axvline(cutoff, color='#555555', linestyle='--', linewidth=1.0, zorder=2)

    # Fitted lines & 95% confidence bands
    x_grid_left = np.linspace(max(cutoff - bandwidth, float(data[xvar].min())), cutoff, 80)
    pred_df_left = pd.DataFrame({'const': 1.0, 'd': 0.0, 'x_c': x_grid_left - cutoff, 'd_x_c': 0.0})
    pred_left = model_rd.get_prediction(pred_df_left).summary_frame(alpha=0.05)
    ax.plot(x_grid_left, pred_left['mean'], color=OKABE_ITO[0], linewidth=1.8, zorder=3)
    ax.fill_between(x_grid_left, pred_left['mean_ci_lower'], pred_left['mean_ci_upper'], color=OKABE_ITO[0], alpha=0.15, zorder=2)

    x_grid_right = np.linspace(cutoff, min(cutoff + bandwidth, float(data[xvar].max())), 80)
    pred_df_right = pd.DataFrame({'const': 1.0, 'd': 1.0, 'x_c': x_grid_right - cutoff, 'd_x_c': x_grid_right - cutoff})
    pred_right = model_rd.get_prediction(pred_df_right).summary_frame(alpha=0.05)
    ax.plot(x_grid_right, pred_right['mean'], color=OKABE_ITO[1], linewidth=1.8, zorder=3)
    ax.fill_between(x_grid_right, pred_right['mean_ci_lower'], pred_right['mean_ci_upper'], color=OKABE_ITO[1], alpha=0.15, zorder=2)

    # Annotation badge
    note = f"RD Jump tau = {tau:.3f} (SE = {se_tau:.3f}, p = {p_val:.3f})\nBandwidth h = {bandwidth:.2f}"
    ax.text(0.04, 0.90, note, transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))

    title = f"Regression Discontinuity: {yvar} on {xvar} (Cutoff c = {cutoff:0.1f})"
    apply_academic_theme(ax, theme=theme, title=title, xlabel=str(xvar), ylabel=str(yvar))

    if save_path and created_fig:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    return {
        'tau': tau,
        'se': se_tau,
        'p_value': p_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_effective': len(sub_rd),
        'bandwidth': bandwidth,
        'left_bins': left_bins,
        'right_bins': right_bins,
        'figure': fig,
        'ax': ax
    }


def plot_irf(df, response_var, shock_var, max_horizon=12, theme='aer', save_path=None, ax=None):
    """
    Renders publication-grade Impulse Response Function (IRF) exhibit
    via Jordà (2005) Local Projections with Newey-West HAC standard errors
    and dual confidence bands: 68% (inner, 1 SE) and 95% (outer, 1.96 SE).
    """
    import statsmodels.api as sm

    data = df[[response_var, shock_var]].dropna().copy()
    max_h = int(max_horizon)
    horizons = []

    for h in range(max_h + 1):
        df_h = data.copy()
        df_h['y_lead'] = df_h[response_var].shift(-h)
        valid = df_h.dropna(subset=['y_lead', shock_var])
        if len(valid) < 5:
            continue

        X_h = sm.add_constant(valid[[shock_var]])
        # Newey-West HAC standard errors with lag h + 1
        model_h = sm.OLS(valid['y_lead'], X_h).fit(cov_type='HAC', cov_kwds={'maxlags': h + 1})
        coef = float(model_h.params[shock_var])
        se = float(model_h.bse[shock_var])

        horizons.append({
            'horizon': h,
            'estimate': coef,
            'se': se,
            'ci68_lo': coef - 1.0 * se,
            'ci68_hi': coef + 1.0 * se,
            'ci95_lo': coef - 1.96 * se,
            'ci95_hi': coef + 1.96 * se
        })

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
        created_fig = True
    else:
        fig = ax.figure

    h_arr = np.array([r['horizon'] for r in horizons])
    est_arr = np.array([r['estimate'] for r in horizons])
    ci68_lo = np.array([r['ci68_lo'] for r in horizons])
    ci68_hi = np.array([r['ci68_hi'] for r in horizons])
    ci95_lo = np.array([r['ci95_lo'] for r in horizons])
    ci95_hi = np.array([r['ci95_hi'] for r in horizons])

    # Reference zero line
    ax.axhline(0, color='#666666', linestyle='--', linewidth=0.9, alpha=0.8)

    # 95% outer ribbon
    ax.fill_between(h_arr, ci95_lo, ci95_hi, color=OKABE_ITO[0], alpha=0.15, label='95% CI')
    # 68% inner ribbon (1 SE)
    ax.fill_between(h_arr, ci68_lo, ci68_hi, color=OKABE_ITO[0], alpha=0.30, label='68% CI (1 SE)')
    # Point estimates trajectory
    ax.plot(h_arr, est_arr, color=OKABE_ITO[0], linewidth=2.0, marker='o', markersize=4.5)

    ax.legend(frameon=False, loc='best', fontsize=9)
    title = f"Local Projections IRF: Response of {response_var} to {shock_var}"
    apply_academic_theme(ax, theme=theme, title=title, xlabel="Horizon (h)", ylabel=f"Response ({response_var})")

    if save_path and created_fig:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    return {
        'horizons': horizons,
        'figure': fig,
        'ax': ax
    }


def plot_synthetic_control(synth_result, theme='aer', save_path=None, ax=None):
    """
    Renders publication exhibit:
    Panel A: Treated vs. Synthetic outcome trajectories with intervention marker.
    Panel B: Difference / Gap trajectory around zero baseline.
    """
    if 'trajectory' in synth_result:
        traj = synth_result['trajectory']
        times = np.array([r['time'] for r in traj])
        treated = np.array([r['treated'] for r in traj])
        synthetic = np.array([r['synthetic'] for r in traj])
        gaps = np.array([r['gap'] for r in traj])
        treat_time = synth_result.get('treat_time', times[len(times)//2])
    else:
        times = np.array(synth_result['time'])
        treated = np.array(synth_result['treated'])
        synthetic = np.array(synth_result['synthetic'])
        gaps = np.array(synth_result['gap'])
        treat_time = synth_result.get('treat_time', times[len(times)//2])

    if ax is not None:
        fig = ax.figure
        ax.axvline(treat_time, color='#555555', linestyle='--', linewidth=1.0)
        ax.plot(times, treated, color=OKABE_ITO[1], linewidth=2.0, label='Treated')
        ax.plot(times, synthetic, color=OKABE_ITO[0], linewidth=1.8, linestyle='-.', label='Synthetic')
        ax.legend(frameon=False, loc='best', fontsize=9)
        title = f"Synthetic Control: {synth_result.get('depvar', 'Outcome')}"
        apply_academic_theme(ax, theme=theme, title=title, xlabel="Time", ylabel="Level")
        return {'figure': fig, 'ax': ax}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=150)

    # Panel A: Trajectories
    ax1.axvline(treat_time, color='#555555', linestyle='--', linewidth=1.0)
    ax1.plot(times, treated, color=OKABE_ITO[1], linewidth=2.0, label='Treated')
    ax1.plot(times, synthetic, color=OKABE_ITO[0], linewidth=1.8, linestyle='-.', label='Synthetic')
    ax1.legend(frameon=False, loc='best', fontsize=9)
    apply_academic_theme(ax1, theme=theme, title="Panel A: Outcome Trajectories", xlabel="Time", ylabel="Level")

    # Panel B: Gap
    ax2.axhline(0, color='#666666', linestyle='--', linewidth=0.9)
    ax2.axvline(treat_time, color='#555555', linestyle='--', linewidth=1.0)
    ax2.plot(times, gaps, color='#333333', linewidth=1.8, marker='o', markersize=4.0)
    apply_academic_theme(ax2, theme=theme, title="Panel B: Treatment Gap (Treated - Synthetic)", xlabel="Time", ylabel="Gap")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    return {
        'figure': fig,
        'ax_traj': ax1,
        'ax_gap': ax2
    }



