"""
engine/timeseries.py - Time Series Analysis & Forecasting Engine
Implements Augmented Dickey-Fuller (ADF) unit root testing, correlograms (ACF/PACF),
and Box-Jenkins ARIMA forecasting with confidence intervals.
"""

import inspect
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA

def compute_dfuller(series, regression='c', lags=None):
    s = pd.Series(series).dropna().to_numpy(dtype=float)
    if len(s) < 10:
        raise ValueError("Insufficient time series observations for ADF test.")
        
    adf_kwargs = {'regression': regression}
    if 'result_object' in inspect.signature(adfuller).parameters:
        adf_kwargs['result_object'] = True

    if lags is not None:
        adf_kwargs['maxlag'] = int(lags)
        adf_kwargs['autolag'] = None
    else:
        adf_kwargs['autolag'] = 'AIC'

    result = adfuller(s, **adf_kwargs)
        
    if hasattr(result, 'statistic'):
        stat = result.statistic
        pval = result.pvalue
        usedlag = getattr(result, 'lags', getattr(result, 'usedlag', 0))
        nobs = result.nobs
        crit_vals = result.critical_values
    elif len(result) == 6:
        stat, pval, usedlag, nobs, crit_vals, icbest = result
    else:
        stat, pval, usedlag, nobs, crit_vals = result
    
    return {
        'stat': float(stat),
        'p_value': float(pval),
        'lags_used': int(usedlag),
        'n_obs': int(nobs),
        'critical_values': {k: float(v) for k, v in crit_vals.items()},
        'is_stationary': bool(pval < 0.05)
    }

def compute_corrgram(series, max_lags=12):
    s = pd.Series(series).dropna().to_numpy(dtype=float)
    n = len(s)
    max_lags = min(max_lags, n // 2 - 1, 30)
    if max_lags < 2:
        max_lags = 2
        
    acf_kwargs = {'nlags': max_lags, 'qstat': True}
    if 'result_object' in inspect.signature(acf).parameters:
        acf_kwargs['result_object'] = True

    acf_res = acf(s, **acf_kwargs)
    if hasattr(acf_res, 'acf'):
        acf_vals = acf_res.acf
        q_stat = acf_res.qstat
        p_vals = acf_res.pvalues
    else:
        acf_vals, q_stat, p_vals = acf_res

    pacf_vals = pacf(s, nlags=max_lags, method='yw')
    
    rows = []
    for lag in range(1, max_lags + 1):
        rows.append({
            'lag': lag,
            'acf': float(acf_vals[lag]),
            'pacf': float(pacf_vals[lag]),
            'q_stat': float(q_stat[lag - 1]),
            'p_value': float(p_vals[lag - 1])
        })
        
    return {
        'n': n,
        'max_lags': max_lags,
        'rows': rows
    }

def compute_arima(series, p=1, d=0, q=0, steps=5):
    s = pd.Series(series).dropna()
    model = ARIMA(s, order=(p, d, q))
    res = model.fit()
    
    # In-sample summary
    coef_table = []
    param_names = res.param_names
    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    
    for name in param_names:
        b = float(params[name])
        se = float(bse[name])
        z = b / se if se > 0 else 0.0
        pv = float(pvalues[name])
        coef_table.append({
            'name': name,
            'coef': b,
            'se': se,
            'z': z,
            'p': pv,
            'ci_lower': b - 1.96 * se,
            'ci_upper': b + 1.96 * se
        })
        
    # Out of sample forecast
    forecast_res = res.get_forecast(steps=steps)
    forecast_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)
    
    forecast_table = []
    for i in range(steps):
        forecast_table.append({
            'step': i + 1,
            'predicted': float(forecast_mean.iloc[i]),
            'ci_lower': float(conf_int.iloc[i, 0]),
            'ci_upper': float(conf_int.iloc[i, 1])
        })
        
    return {
        'order': (p, d, q),
        'n': len(s),
        'aic': float(res.aic),
        'bic': float(res.bic),
        'll': float(res.llf),
        'coefs': coef_table,
        'forecast': forecast_table
    }


def compute_hp_filter(series, lamb=1600.0):
    """
    Hodrick-Prescott (1997) filter decomposing time series y_t into trend g_t and cycle c_t:
      min_{g} sum_{t=1}^T (y_t - g_t)^2 + lamb * sum_{t=2}^{T-1} ((g_{t+1} - g_t) - (g_t - g_{t-1}))^2
    Returns (cycle, trend) as numpy float64 arrays.
    """
    from statsmodels.tsa.filters.hp_filter import hpfilter
    y = np.asarray(series, dtype=float)
    if len(y) < 4:
        raise ValueError("HP filter requires at least 4 observations.")
    if lamb <= 0:
        raise ValueError(f"HP filter smoothing parameter lambda must be positive, got {lamb}")
    cycle, trend = hpfilter(y, lamb=float(lamb))
    return np.asarray(cycle, dtype=float), np.asarray(trend, dtype=float)

