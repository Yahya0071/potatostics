"""
engine/bigdata.py - High-Performance Big Data & Out-of-Core Processing Core
Part of Potatostics Big Data Suite (Phase 4).

Features:
1. Apache Arrow & Polars Zero-Copy Ingestion:
   - Streaming Parquet (.parquet), Arrow IPC / Feather (.feather, .arrow), and compressed CSV (.csv.gz).
2. Out-of-Core Streaming OLS Regression (StreamingRegressionAccumulator):
   - Computes X'X, X'y, y'y, and N in bounded-memory streaming chunks (O(K^2) RAM).
   - Allows regressions on 10M-50M+ rows and files exceeding RAM on a standard PC.
   - Computes homoskedastic and heteroskedasticity-robust (HC1) standard errors via 2-pass streaming.
3. High-Velocity Polars Aggregations:
   - Polars-accelerated multi-threaded summary statistics and group collapse.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import scipy.linalg
from engine.estimation import _order_preserving_qr_selection

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.feather as feather
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


class StreamingRegressionAccumulator:
    """
    Out-of-core online streaming accumulator for ordinary least squares regression.
    Maintains:
      - X'X: (K x K) symmetric Gram matrix
      - X'y: (K x 1) projection vector
      - y'y: scalar sum of squared dependent variable
      - sum_y: scalar sum of y (for total sum of squares)
      - N: total observation count
    Memory consumption is strictly O(K^2) regardless of whether N is 100,000 or 100,000,000.
    """

    def __init__(self, n_features=None, add_constant=True, k=None):
        if n_features is not None:
            self.n_features = n_features
            self.k = n_features + (1 if add_constant else 0)
        elif k is not None:
            self.k = k
            self.n_features = k - (1 if add_constant else 0)
        else:
            raise ValueError("n_features or k must be provided")
        self.add_constant = add_constant
        self.XtX = np.zeros((self.k, self.k), dtype=np.float64)
        self.Xty = np.zeros(self.k, dtype=np.float64)
        self.yty = 0.0
        self.sum_y = 0.0
        self.N = 0
        self.R = np.zeros((0, self.k + 1), dtype=np.float64)

    def update(self, X_chunk, y_chunk):
        """
        Updates accumulation matrices with a batch / chunk of data.
        X_chunk: (B x (K-1)) array-like
        y_chunk: (B,) array-like
        """
        X_arr = np.asarray(X_chunk, dtype=np.float64)
        y_arr = np.asarray(y_chunk, dtype=np.float64)

        # Drop NaN observations within this chunk
        mask = ~np.isnan(X_arr).any(axis=1) & ~np.isnan(y_arr)
        if not np.all(mask):
            X_arr = X_arr[mask]
            y_arr = y_arr[mask]

        b_size = len(y_arr)
        if b_size == 0:
            return

        if self.add_constant:
            ones = np.ones((b_size, 1), dtype=np.float64)
            X_arr = np.hstack([X_arr, ones])

        # Accumulate
        self.XtX += X_arr.T @ X_arr
        self.Xty += X_arr.T @ y_arr
        self.yty += float(np.sum(y_arr ** 2))
        self.sum_y += float(np.sum(y_arr))
        self.N += b_size

        # Incremental QR / TSQR accumulation
        chunk_mat = np.column_stack([X_arr, y_arr])
        if self.R.shape[0] > 0:
            stacked = np.vstack([self.R, chunk_mat])
        else:
            stacked = chunk_mat
        _, R_new = scipy.linalg.qr(stacked, mode='economic')
        self.R = R_new[:self.k + 1, :]

    def finalize(self, depvar, var_names=None, vce='classical', second_pass_generator=None):
        """
        Solves beta = (X'X)^(-1) X'y and computes degrees of freedom, R2, F-stat,
        and standard errors (classical or HC1 robust).
        """
        if isinstance(depvar, (list, tuple)):
            if var_names is not None and isinstance(var_names, str):
                vce = var_names
            var_names = depvar
            depvar = 'y'
        if var_names is None:
            var_names = [f"x{i+1}" for i in range(self.k - (1 if self.add_constant else 0))]

        if self.N <= self.k:
            raise ValueError(f"Insufficient observations (N={self.N}) for {self.k} regressors.")

        all_names = list(var_names)
        if self.add_constant and '_cons' not in all_names:
            all_names.append('_cons')

        # Stable rank detection and solve via accumulated R factor
        if self.R.shape[0] >= self.k:
            Rx = self.R[:self.k, :self.k]
            Ry = self.R[:self.k, self.k]
        else:
            Rx = np.zeros((self.k, self.k), dtype=np.float64)
            Ry = np.zeros(self.k, dtype=np.float64)

        kept, omitted_cols, _, _ = _order_preserving_qr_selection(Rx)
        rank = len(kept)
        omitted_vars = [all_names[j] for j in omitted_cols]

        beta = np.zeros(self.k, dtype=np.float64)
        inv_XtX = np.zeros((self.k, self.k), dtype=np.float64)
        cond_number = 1.0

        if rank > 0:
            S_kept = np.linalg.norm(Rx[:, kept], axis=0)
            S_kept = np.where(S_kept > 0, S_kept, 1.0)
            Xs_kept = Rx[:, kept] / S_kept
            Qs, Rs = scipy.linalg.qr(Xs_kept, mode='economic')
            beta_s = scipy.linalg.solve_triangular(Rs, Qs.T @ Ry)
            beta[kept] = beta_s / S_kept

            Rs_inv = scipy.linalg.solve_triangular(Rs, np.eye(rank))
            inv_XsXs = Rs_inv @ Rs_inv.T
            inv_R_R = inv_XsXs / np.outer(S_kept, S_kept)
            inv_XtX[np.ix_(kept, kept)] = inv_R_R

            diag_Rs = np.abs(np.diag(Rs))
            cond_number = float(np.max(diag_Rs) / np.min(diag_Rs)) if np.min(diag_Rs) > 0 else 1.0

        warnings_list = []
        if rank < self.k:
            warnings_list.append(f"Rank deficiency detected: {len(omitted_vars)} collinear regressors omitted ({', '.join(omitted_vars)}).")
        if cond_number > 1e10:
            warnings_list.append(f"High regressor condition number ({cond_number:.2e}). Results may be sensitive to collinearity.")

        # Robust calculation of Sum of Squared Residuals to prevent cancellation:
        # e'e = y'y - 2 beta' (X'y) + beta' (X'X) beta
        b_XtX_b = float(beta.T @ self.XtX @ beta)
        b_Xty = float(beta.T @ self.Xty)
        ssr = float(self.yty - 2.0 * b_Xty + b_XtX_b)
        if ssr < 0.0 or not np.isfinite(ssr):
            ssr = max(0.0, float(self.yty - b_Xty))
        if not np.isfinite(ssr):
            ssr = 0.0

        df_m = max(0, rank - (1 if self.add_constant else 0))
        df_r = max(1, self.N - rank)
        sigma2 = ssr / df_r if df_r > 0 else 0.0

        # Total sum of squares: SST = y'y - N * y_bar^2
        y_bar = self.sum_y / self.N
        sst = max(0.0, float(self.yty - self.N * (y_bar ** 2)))
        if sst <= 0:
            sst = 1e-12

        ssm = max(0.0, sst - ssr)
        r2 = max(0.0, min(1.0, 1.0 - (ssr / sst))) if sst > 0 else 0.0
        r2_adj = 1.0 - ((1.0 - r2) * (self.N - 1) / df_r) if df_r > 0 else r2

        # F-statistic
        if df_m > 0 and sigma2 > 0:
            f_stat = float((ssm / df_m) / sigma2)
            p_f = float(1.0 - stats.f.cdf(f_stat, df_m, df_r))
        else:
            f_stat = 0.0
            p_f = 1.0

        # Variance-Covariance Matrix: Classical vs Robust HC1
        vce = (vce or 'classical').lower()
        if vce in ('robust', 'hc1', 'white') and second_pass_generator is not None:
            # 2nd pass: accumulate meat matrix S = sum X_b' diag(e_b^2) X_b
            S = np.zeros((self.k, self.k), dtype=np.float64)
            gen = second_pass_generator() if callable(second_pass_generator) else second_pass_generator
            for X_chunk, y_chunk in gen:
                X_arr = np.asarray(X_chunk, dtype=np.float64)
                y_arr = np.asarray(y_chunk, dtype=np.float64)
                mask = ~np.isnan(X_arr).any(axis=1) & ~np.isnan(y_arr)
                if not np.all(mask):
                    X_arr = X_arr[mask]
                    y_arr = y_arr[mask]
                b_size = len(y_arr)
                if b_size == 0:
                    continue
                if self.add_constant:
                    ones = np.ones((b_size, 1), dtype=np.float64)
                    X_arr = np.hstack([X_arr, ones])

                # Residuals for this chunk
                e_chunk = y_arr - X_arr @ beta
                scores = X_arr * e_chunk[:, None]
                S += scores.T @ scores

            hc1_scale = float(self.N / df_r) if df_r > 0 else 1.0
            vcov = hc1_scale * (inv_XtX @ S @ inv_XtX)
        else:
            vcov = sigma2 * inv_XtX

        se = np.sqrt(np.maximum(0.0, np.diag(vcov)))
        t_vals = np.where(se > 0, np.divide(beta, se, out=np.zeros_like(beta), where=(se > 0)), 0.0)
        p_vals = 2.0 * (1.0 - stats.t.cdf(np.abs(t_vals), df_r)) if df_r > 0 else np.zeros_like(beta)
        t_crit = stats.t.ppf(0.975, df_r) if df_r > 0 else 1.96

        coefs_list = []
        for i, name in enumerate(all_names):
            is_omitted = name in omitted_vars
            b_val = float(beta[i])
            se_val = 0.0 if is_omitted else float(se[i])
            t_val = 0.0 if is_omitted else float(t_vals[i])
            p_val = 1.0 if is_omitted else float(p_vals[i])
            coefs_list.append({
                'name': name,
                'coef': b_val,
                'se': se_val,
                't': t_val,
                'p': p_val,
                'ci_lower': 0.0 if is_omitted else b_val - t_crit * se_val,
                'ci_upper': 0.0 if is_omitted else b_val + t_crit * se_val,
                'omitted': is_omitted
            })

        df_t = self.N - 1
        ms_m = ssm / df_m if df_m > 0 else 0.0
        ms_r = ssr / df_r if df_r > 0 else 0.0
        ms_t = sst / df_t if df_t > 0 else 0.0

        return {
            'model_type': 'ols',
            'streaming': True,
            'depvar': depvar,
            'indepvars': [v for v in all_names if v != '_cons'],
            'var_names': all_names,
            'n': int(self.N),
            'rank': int(rank),
            'omitted_vars': omitted_vars,
            'cond_number': float(cond_number),
            'warnings': warnings_list,
            'df_m': int(df_m),
            'df_r': int(df_r),
            'df_t': int(df_t),
            'f_stat': f_stat,
            'prob_f': p_f,
            'p_f': p_f,
            'r2': float(r2),
            'adj_r2': float(r2_adj),
            'r2_adj': float(r2_adj),
            'root_mse': float(np.sqrt(sigma2)),
            'ss_m': float(ssm),
            'ss_r': float(ssr),
            'ssr': float(ssr),
            'ss_t': float(sst),
            'ms_m': float(ms_m),
            'ms_r': float(ms_r),
            'ms_t': float(ms_t),
            'beta': beta,
            'vcov': vcov,
            'coefs': coefs_list,
            'vce': vce,
            'robust': (vce in ('robust', 'hc1', 'white'))
        }


def read_parquet_or_arrow(filepath, columns=None):
    """
    Reads an Apache Parquet (.parquet) or Feather/Arrow IPC (.feather, .arrow)
    file into a pandas DataFrame using zero-copy column projections.
    """
    filepath = str(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ('.parquet', '.pq'):
        if HAS_POLARS:
            # Polars is ultra-fast with multi-threaded IO
            pl_df = pl.read_parquet(filepath, columns=columns)
            return pl_df.to_pandas()
        elif HAS_PYARROW:
            table = pq.read_table(filepath, columns=columns)
            return table.to_pandas()
        else:
            return pd.read_parquet(filepath, columns=columns)

    elif ext in ('.feather', '.arrow', '.ipc'):
        if HAS_PYARROW:
            table = feather.read_table(filepath, columns=columns)
            return table.to_pandas()
        else:
            return pd.read_feather(filepath, columns=columns)

    else:
        raise ValueError(f"Unsupported Arrow/Parquet file extension: '{ext}'")


def write_parquet_or_arrow(df, filepath, compression='snappy'):
    """Exports a pandas DataFrame to Apache Parquet format with high compression."""
    filepath = str(filepath)
    if not filepath.lower().endswith(('.parquet', '.pq', '.arrow', '.feather')):
        filepath += '.parquet'

    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.feather', '.arrow'):
        if HAS_PYARROW:
            table = pa.Table.from_pandas(df)
            feather.write_feather(table, filepath, compression=compression)
        else:
            df.to_feather(filepath)
    else:
        if HAS_PYARROW:
            table = pa.Table.from_pandas(df)
            pq.write_table(table, filepath, compression=compression)
        else:
            df.to_parquet(filepath, compression=compression)
    return filepath


def stream_dataset_chunks(source, columns, chunk_size=100000):
    """
    Generator yielding (X_chunk, y_chunk) pairs from disk or memory.
    Source can be:
      - .parquet / .csv / .feather filepath string
      - pandas DataFrame
      - polars DataFrame
    """
    depvar = columns[0]
    indepvars = columns[1:]
    all_needed = [depvar] + indepvars

    if isinstance(source, str):
        ext = os.path.splitext(source)[1].lower()
        if ext in ('.parquet', '.pq') and HAS_PYARROW:
            pf = pq.ParquetFile(source)
            for batch in pf.iter_batches(batch_size=chunk_size, columns=all_needed):
                b_df = batch.to_pandas()
                y = b_df[depvar].to_numpy(dtype=np.float64)
                X = b_df[indepvars].to_numpy(dtype=np.float64) if indepvars else np.empty((len(y), 0))
                yield X, y

        elif ext in ('.feather', '.arrow', '.ipc') and HAS_PYARROW:
            # Native PyArrow IPC RecordBatch streaming without full materialization
            try:
                with pa.memory_map(source, 'r') as mmap_file:
                    reader = pa.ipc.open_file(mmap_file)
                    for i in range(reader.num_record_batches):
                        batch = reader.get_batch(i)
                        if all_needed:
                            cols = [batch.column(c) for c in all_needed if c in batch.schema.names]
                            names = [c for c in all_needed if c in batch.schema.names]
                            batch = pa.RecordBatch.from_arrays(cols, names=names)
                        b_df = batch.to_pandas()
                        y = b_df[depvar].to_numpy(dtype=np.float64)
                        X = b_df[indepvars].to_numpy(dtype=np.float64) if indepvars else np.empty((len(y), 0))
                        yield X, y
            except (pa.ArrowInvalid, Exception):
                with pa.memory_map(source, 'r') as mmap_file:
                    reader = pa.ipc.open_stream(mmap_file)
                    for batch in reader:
                        if all_needed:
                            cols = [batch.column(c) for c in all_needed if c in batch.schema.names]
                            names = [c for c in all_needed if c in batch.schema.names]
                            batch = pa.RecordBatch.from_arrays(cols, names=names)
                        b_df = batch.to_pandas()
                        y = b_df[depvar].to_numpy(dtype=np.float64)
                        X = b_df[indepvars].to_numpy(dtype=np.float64) if indepvars else np.empty((len(y), 0))
                        yield X, y

        elif ext in ('.csv', '.txt', '.gz'):
            for chunk_df in pd.read_csv(source, usecols=all_needed, chunksize=chunk_size):
                y = chunk_df[depvar].to_numpy(dtype=np.float64)
                X = chunk_df[indepvars].to_numpy(dtype=np.float64) if indepvars else np.empty((len(y), 0))
                yield X, y

        else:
            # Fallback to reading table and chunking
            df = read_parquet_or_arrow(source, columns=all_needed)
            n_rows = len(df)
            for start in range(0, n_rows, chunk_size):
                sub = df.iloc[start:start + chunk_size]
                y = sub[depvar].to_numpy(dtype=np.float64)
                X = sub[indepvars].to_numpy(dtype=np.float64) if indepvars else np.empty((len(y), 0))
                yield X, y

    elif HAS_POLARS and isinstance(source, pl.DataFrame):
        n_rows = len(source)
        for start in range(0, n_rows, chunk_size):
            sub = source.slice(start, chunk_size)
            y = sub[depvar].to_numpy()
            X = sub[indepvars].to_numpy() if indepvars else np.empty((len(y), 0))
            yield X, y

    elif isinstance(source, pd.DataFrame):
        n_rows = len(source)
        for start in range(0, n_rows, chunk_size):
            sub = source.iloc[start:start + chunk_size]
            y = sub[depvar].to_numpy(dtype=np.float64)
            X = sub[indepvars].to_numpy(dtype=np.float64) if indepvars else np.empty((len(y), 0))
            yield X, y

    else:
        raise TypeError(f"Unsupported data source for streaming: {type(source)}")


def compute_streaming_ols(source, depvar, indepvars, vce='classical', chunk_size=100000, is_cancelled=None):
    """
    Executes an out-of-core streaming OLS regression on large data.
    """
    cols = [depvar] + indepvars
    acc = StreamingRegressionAccumulator(n_features=len(indepvars), add_constant=True)

    # 1st pass: accumulate X'X and X'y
    for X_b, y_b in stream_dataset_chunks(source, cols, chunk_size=chunk_size):
        if is_cancelled and is_cancelled():
            raise InterruptedError("Operation cancelled by user.")
        acc.update(X_b, y_b)

    # Helper for 2nd pass if robust VCE requested
    def second_pass_gen():
        for X_b, y_b in stream_dataset_chunks(source, cols, chunk_size=chunk_size):
            if is_cancelled and is_cancelled():
                raise InterruptedError("Operation cancelled by user.")
            yield X_b, y_b

    return acc.finalize(depvar, indepvars, vce=vce, second_pass_generator=second_pass_gen)


def polars_fast_summary(df, varlist=None):
    """
    Uses Polars for multi-threaded summary statistics computation on large data.
    """
    if not HAS_POLARS:
        raise ImportError("Polars is required for fast multi-threaded summary.")

    if not isinstance(df, pl.DataFrame):
        pl_df = pl.from_pandas(df)
    else:
        pl_df = df

    if not varlist:
        varlist = [col for col, dtype in zip(pl_df.columns, pl_df.dtypes) if dtype.is_numeric()]

    results = []
    for var in varlist:
        s = pl_df[var].drop_nulls()
        n = len(s)
        if n == 0:
            continue
        mean_val = float(s.mean())
        sd_val = float(s.std()) if n > 1 else 0.0
        min_val = float(s.min())
        p25_val = float(s.quantile(0.25))
        p50_val = float(s.median())
        p75_val = float(s.quantile(0.75))
        max_val = float(s.max())

        results.append({
            'var': var,
            'obs': n,
            'n': n,
            'mean': mean_val,
            'std': sd_val,
            'sd': sd_val,
            'min': min_val,
            'p25': p25_val,
            'p50': p50_val,
            'p75': p75_val,
            'max': max_val
        })
    return results


def polars_fast_collapse(df, stats_dict, by_vars):
    """
    Polars multi-threaded aggregation for large datasets.
    stats_dict: {stat: [vars]} e.g. {'mean': ['y', 'x'], 'sum': ['v']}
    by_vars: list of grouping variables
    """
    if not HAS_POLARS:
        raise ImportError("Polars is required for fast collapse.")
    pl_df = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df
    aggs = []
    for stat, vars_to_agg in stats_dict.items():
        for v in vars_to_agg:
            if v not in pl_df.columns:
                continue
            col_expr = pl.col(v)
            if stat in ('mean', 'avg'):
                aggs.append(col_expr.mean().alias(f"{v}"))
            elif stat in ('sum', 'total'):
                aggs.append(col_expr.sum().alias(f"{v}"))
            elif stat == 'sd':
                aggs.append(col_expr.std().alias(f"{v}"))
            elif stat == 'min':
                aggs.append(col_expr.min().alias(f"{v}"))
            elif stat == 'max':
                aggs.append(col_expr.max().alias(f"{v}"))
            elif stat in ('median', 'p50'):
                aggs.append(col_expr.median().alias(f"{v}"))
            elif stat == 'count':
                aggs.append(col_expr.count().alias(f"{v}"))
    out = pl_df.group_by(by_vars).agg(aggs)
    return out.to_pandas()
