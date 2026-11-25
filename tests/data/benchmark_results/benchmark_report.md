# Potatostics — Automated Econometric Benchmark Report
**Generated at**: 2026-11-25 14:54:58 UTC
**Total Benchmarks**: 19 | **Passed**: 19 | **Failed**: 0 | **Skipped**: 0

| Benchmark Identifier | Estimator | Validation Type | Reference Standard | Max Abs Coef Diff | Max Abs SE Diff | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **`BENCH_01_OLS_CLASSICAL`** | `ols` | `INDEPENDENT_EXTERNAL` | statsmodels.regression.linear_model.OLS | 5.50e-11 | 8.23e-11 | **PASS** |
| **`BENCH_02_OLS_ROBUST_HC1`** | `ols_hc1` | `INDEPENDENT_EXTERNAL` | statsmodels.regression.linear_model.OLS(cov_type='HC1') | 5.50e-11 | 2.64e-11 | **PASS** |
| **`BENCH_03_OLS_CLUSTER`** | `ols_cluster` | `INDEPENDENT_EXTERNAL` | statsmodels.regression.linear_model.OLS(cov_type='cluster') | 5.50e-11 | 2.39e-10 | **PASS** |
| **`BENCH_04_LOGIT_MLE`** | `logit` | `INDEPENDENT_EXTERNAL` | statsmodels.discrete.discrete_model.Logit | 3.36e-12 | 5.52e-13 | **PASS** |
| **`BENCH_05_PROBIT_MLE`** | `probit` | `INDEPENDENT_EXTERNAL` | statsmodels.discrete.discrete_model.Probit | 5.33e-15 | 1.40e-14 | **PASS** |
| **`BENCH_06_POISSON_MLE`** | `poisson` | `INDEPENDENT_EXTERNAL` | statsmodels.discrete.discrete_model.Poisson | 2.08e-10 | 1.21e-11 | **PASS** |
| **`BENCH_07_QUANTILE_REGRESSION`** | `qreg` | `INDEPENDENT_EXTERNAL` | statsmodels.regression.quantile_regression.QuantReg | 1.26e-08 | 4.65e-10 | **PASS** |
| **`BENCH_08_IVREGRESS_2SLS`** | `ivregress_2sls` | `INDEPENDENT_EXTERNAL` | linearmodels.iv.model.IV2SLS | 0.00e+00 | 0.00e+00 | **PASS** |
| **`BENCH_09_PANEL_FE`** | `xtreg_fe` | `INDEPENDENT_EXTERNAL` | linearmodels.panel.model.PanelOLS(entity_effects=True) | 0.00e+00 | 1.39e-17 | **PASS** |
| **`BENCH_10_HP_FILTER`** | `tsfilter_hp` | `INTERNAL_CONSISTENCY` | statsmodels.tsa.filters.hp_filter.hpfilter | 0.00e+00 | N/A | **PASS** |
| **`BENCH_11_VAR`** | `var` | `INTERNAL_CONSISTENCY` | statsmodels.tsa.api.VAR | 0.00e+00 | 0.00e+00 | **PASS** |
| **`BENCH_12_KAPLAN_MEIER`** | `sts_list` | `ANALYTICAL_GROUND_TRUTH` | Kaplan-Meier Product-Limit & Greenwood Formula | 5.55e-17 | 2.78e-17 | **PASS** |
| **`BENCH_13_BACONDECOMP`** | `bacondecomp` | `ANALYTICAL_GROUND_TRUTH` | Goodman-Bacon (2021) Decomposition Theorem | 4.44e-16 | N/A | **PASS** |
| **`BENCH_14_DRDID`** | `drdid` | `SIMULATION_GROUND_TRUTH` | Sant'Anna & Zhao (2020) Doubly Robust DiD (Panel & RC) | 4.05e-03 | N/A | **PASS** |
| **`BENCH_15_HP_FILTER_ANALYTICAL`** | `tsfilter_hp` | `ANALYTICAL_GROUND_TRUTH` | Closed-Form Central Difference Solve: (I + lambda K'K)^(-1) y | 5.70e-11 | N/A | **PASS** |
| **`BENCH_16_VAR_ANALYTICAL_OLS`** | `var` | `ANALYTICAL_GROUND_TRUTH` | Equation-by-Equation OLS: B = (X'X)^(-1) X'Y | 5.00e-16 | N/A | **PASS** |
| **`BENCH_17_COX_BRESLOW_ANALYTICAL`** | `stcox` | `ANALYTICAL_GROUND_TRUTH` | Direct Scipy Numerical Optimization of Breslow Partial Log-Likelihood | 1.40e-07 | N/A | **PASS** |
| **`BENCH_18_ORDERED_LOGIT`** | `ologit` | `INTERNAL_CONSISTENCY` | statsmodels.miscmodels.ordinal_model.OrderedModel | 0.00e+00 | 0.00e+00 | **PASS** |
| **`BENCH_19_COX_PHREG`** | `stcox` | `INTERNAL_CONSISTENCY` | statsmodels.duration.hazard_regression.PHReg | 0.00e+00 | 0.00e+00 | **PASS** |

All benchmarks compare exact identical datasets, samples, and degrees of freedom conventions.
Untested metrics are recorded as `None` / `N/A` rather than populated with placeholder zeros.