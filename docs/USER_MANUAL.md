# Potatostics: The Comprehensive User Manual
### High-Performance Econometric Software & Verifiable Econometric Computing Suite

---

## Table of Contents

1. [Introduction & Design Philosophy](#1-introduction--design-philosophy)
2. [Installation & Getting Started](#2-installation--getting-started)
3. [The Graphical Interface (Qt6 Desktop Shell)](#3-the-graphical-interface-qt6-desktop-shell)
4. [Data Management & Everyday Wrangling](#4-data-management--everyday-wrangling)
5. [Linear & Panel Econometrics](#5-linear--panel-econometrics)
6. [Discrete Choice, Count, & Survival Models](#6-discrete-choice-count--survival-models)
7. [Time Series & Filtering](#7-time-series--filtering)
8. [Modern Causal Inference](#8-modern-causal-inference)
9. [Complex Survey Data Analysis](#9-complex-survey-data-analysis)
10. [Post-Estimation & Hypothesis Testing](#10-post-estimation--hypothesis-testing)
11. [Publication Reporting & Academic Graph Studio](#11-publication-reporting--academic-graph-studio)
12. [Big Data & Out-of-Core Streaming](#12-big-data--out-of-core-streaming)
13. [Community Package Hub & Authoring Custom Plugins](#13-community-package-hub--authoring-custom-plugins)
14. [Do-File Scripting, Macros, & Automation](#14-do-file-scripting-macros--automation)
15. [Command Syntax Quick Reference](#15-command-syntax-quick-reference)

---

## 1. Introduction & Design Philosophy

**Potatostics** is a free, modern, open-source econometric suite engineered as a full independent high-performance econometric computing platform. 

### Core Principles
- **100% Deterministic & Verifiable**: Every econometric algorithm is mathematically cross-validated against established numerical benchmarks, Oxford's `linearmodels`, and published econometric standards to tolerances tighter than $10^{-6}$.
- **Syntactic Familiarity**: Researchers can sit down and immediately type standard econometric commands (`regress`, `summarize`, `xtreg, fe`, `reghdfe`, `ivregress`, `logit`, `esttab`, etc.) without learning a new paradigm.
- **Python Power with Zero Overhead**: Built on top of vectorized C, Fortran, and Rust libraries (NumPy, SciPy, Polars, PyArrow), Potatostics eliminates Standard Econometrics's memory constraints and multi-core core-count paywalls.
- **Modern Causal Inference Out of the Box**: Native support for Callaway & Sant'Anna (2021) staggered difference-in-differences (`csdid`), Synthetic Control (`synth`), and dynamic event studies (`eventstudy`).
- **No AI Fluff**: Potatostics is focused exclusively on transparent, rigorous, reproducible econometric estimation.

---

## 2. Installation & Getting Started

### Prerequisites
- Operating System: Windows 10/11, macOS (x86_64 or Apple Silicon), or Linux (Ubuntu 20.04+).
- Python: Version 3.10, 3.11, 3.12, 3.13, or 3.14.

### Installing from Source or Wheel
```bash
# Clone or extract repository
cd potatostics

# Install declared dependencies
pip install .

# Or install with developer testing dependencies
pip install ".[dev]"
```

### Launching Potatostics
- **Graphical Desktop Application**:
  ```bash
  python main.py
  # Or on Windows:
  run_potatostics.bat
  ```
- **Headless Do-File Script Execution**:
  ```bash
  python main.py analysis.do
  ```

---

## 3. The Graphical Interface (Qt6 Desktop Shell)

Potatostics features a multi-pane desktop shell built with **PySide6 (Qt6)**:

```
+----------------------------------------------------------------------------------+
| Menu: File  View  Data  Graphics  Statistics  Tools  Help                        |
+--------------------------+-------------------------------------------------------+
| VARIABLES PANE           | RESULTS CONSOLE / SPREADSHEET TABS                   |
| Filter: [             ]  |                                                       |
| Name     Type   Label    |   . regress price mpg weight                          |
| price    float  Price    |   -------------------------------------------------   |
| mpg      int    Mileage  |   Source |       SS      df         MS   Number of obs |
| weight   int    Weight   |   Model  | 186321259      2   93160629   F(2, 71) =... |
| foreign  str    Origin   |   Residual| 448744137     71    6320340   Prob > F =... |
|                          |   Total  | 635065396     73    8699525   R-squared =...|
|                          |   -------------------------------------------------   |
|                          |   price  |      Coef.   Std. Err.      t   P>|t|         |
|                          |   mpg    |   -49.5122    86.1560   -0.57   0.567         |
|                          |   weight |     1.7466     0.6414    2.72   0.008         |
|                          |   _cons  |  1946.0687  3597.0496    0.54   0.590         |
+--------------------------+-------------------------------------------------------+
| COMMAND HISTORY          | COMMAND BAR                                           |
| 1. use auto.dta          |                                                       |
| 2. summarize price mpg   |   . [ regress price mpg weight , vce(robust) ] [Run]  |
| 3. regress price mpg     |                                                       |
+--------------------------+-------------------------------------------------------+
```

### Key UI Features
- Press **Ctrl+1** for Results. Select a model's output and use **Copy results** to copy its table or full report in Markdown or LaTeX.
- Press **Ctrl+2** for the Data editor. With focus in the grid, **Ctrl+F** searches across columns. The footer shows the number of matching rows.
- Press **Ctrl+L** to focus the command line, **Enter** to run, and **Up** or **Down** to recall a command.
- **Spotlight Command Palette (`Ctrl+K`)**: Rapidly search commands, open files, or trigger point-and-click dialogs.
- **Point-and-Click Interactive Options Pages**: Every statistical model and test under the `Statistics` menu (Linear, Panel, Time Series, Discrete Choice, Count, Censored/Quantile, IV, Survival, Survey, Modern Causal) and every visualization under `Graphics` (Scatter, Line, Binned Scatter, RD Plot, IRF, Synthetic Control, Event Study, Histogram, Box Plot, Heatmap, Coefplot) provides an interactive dialog with:
  - Type-safe variable selectors (non-numeric columns filtered from dependent variable dropdowns).
  - Observation filters (`if` expressions and `in` row ranges with one-click "Use Grid Selection").
  - Live mathematical equation preview rendering LaTeX/Unicode formulations with dynamic cutoffs and parameter estimates.
  - Live command syntax preview with **Submit**, **Paste to Command Bar**, and **Cancel** buttons.
- Drag the Variables or Command Review pane by its title bar to move it to a second monitor.
- **Theme Toggle (`F2`)**: Instant switching between Dark Theme and Light Theme.
- **Deterministic Researcher Workflow Acceptance**:
  - Run headless test: `python scripts/verify_researcher_workflows.py --headless`
  - Run visible native desktop test: `python scripts/verify_researcher_workflows.py --visible`

---

## 4. Data Management & Everyday Wrangling

### Loading & Saving Data
```	ext
// Load binary dataset files (.dta) or CSV files
use auto.dta, clear
use "https://example.com/data.csv", clear

// Save dataset
save "clean_data.dta", replace
save "clean_data.csv", replace

// Export comma-separated data. Omitting the extension creates clean_data.csv.
export delimited using "clean_data", replace
```

Some 64-bit integer IDs cannot be represented exactly as DTA numbers. A DTA save that would round an ID returns an error. Use a Parquet file to retain the integer type and missing values.

### Exploring Data
```	ext
describe                 // Displays variable names, types, labels, observation count
summarize                // Mean, Std. Dev., Min, Max for all variables
summarize price mpg, detail // Percentiles (1%, 5%, 25%, 50%, 75%, 95%, 99%), Skewness, Kurtosis
tabulate foreign         // 1-way frequency table with percentages and cumulative counts
tabulate foreign rep78   // 2-way cross-tabulation table
correlate price mpg weight // Correlation matrix
count if price > 5000    // Displays number of matching rows
```

### Creating & Modifying Variables
```	ext
generate log_price = log(price)
generate high_mpg = (mpg > 25)
replace high_mpg = 2 if mpg > 35
drop if price < 3000
keep make price mpg weight foreign
rename weight car_weight
```

### Extended Generate (`egen`)
Potatostics supports grouped (`by:` / `bysort:`) operations:
```	ext
egen mean_mpg = mean(mpg), by(foreign)
egen sd_price = sd(price)
egen p90_price = pctile(price), p(90)
egen z_weight = std(weight)              // Standardized z-score (mean 0, var 1)
egen modal_rep = mode(rep78), by(foreign)// Statistical mode
egen rank_price = rank(price)            // Field ranking
```

### Date & Time Built-ins (Gregorian 1960 Epoch)
Elapsed days are measured from January 1, 1960:
```	ext
generate birth_date = mdy(5, 12, 1985)   // Returns elapsed days
generate b_year = year(birth_date)       // 1985
generate b_month = month(birth_date)     // 5
generate b_day = day(birth_date)         // 12
generate b_qtr = quarter(birth_date)     // 2
generate d_parsed = date("2024-08-15", "YMD")
```

### String Built-ins (1-Based Standard Econometrics Indexing)
```	ext
generate sub = substr(make, 1, 4)        // First 4 characters
generate pos = strpos(make, "VW")        // 1-based index (0 if not found)
generate clean_make = trim(make)         // Strips leading and trailing whitespace
generate upper_make = upper(make)        // Uppercase
generate match_vw = regexm(make, "VW.*") // Regular expression matching flag (0 or 1)
```

### Reshaping, Merging, & Collapsing
```	ext
// Sorting
sort foreign price

// Relational 1:1, m:1, 1:m merge
merge 1:1 id using "demographics.dta"

// Reshape between wide and long formats
reshape long inc ue, i(id) j(year)
reshape wide inc ue, i(id) j(year)

// Dataset aggregation
collapse (mean) price mpg (median) weight, by(foreign)
```

---

## 5. Linear & Panel Econometrics

### Ordinary Least Squares (OLS)
```	ext
// Standard homoskedastic OLS
regress price mpg weight foreign

// White (1980) Heteroskedasticity-Robust Standard Errors (HC1)
regress price mpg weight, vce(robust)

// Cluster-Robust Standard Errors with G/(G-1) small-sample finite cluster adjustment
regress price mpg weight, vce(cluster foreign)

// Suppress constant term
regress price mpg weight, noconstant
```

### Instrumental Variables (2SLS) & Diagnostic Tests
```	ext
// Estimate 2SLS IV: wage instrumented with nearc4
ivregress 2sls lwage exper expersq (educ = nearc4)

// Post-IV Diagnostics:
estat firststage   // First-stage Shea partial R2 and F-statistic
estat weakiv       // Montiel Olea & Pflueger (2013) robust effective F-statistic
estat endogenous   // Durbin & Wu-Hausman endogeneity tests
estat overid       // Sargan & Basmann overidentifying restrictions test
```

### Panel Data Models
First, set your panel identifier and time variable:
```	ext
xtset id year
```

#### Panel Fixed Effects (`xtreg, fe`)
```	ext
xtreg y x1 x2, fe
// Reports: Within, Between, and Overall R2; sigma_u, sigma_e, rho; F-test that all u_i = 0
```

#### Multi-Way High-Dimensional Fixed Effects (`reghdfe`)
```	ext
// Absorbs multiple high-dimensional fixed effects (e.g. firm + year)
reghdfe y x1 x2, absorb(firm_id year) vce(cluster firm_id)
```

#### Panel Random Effects (`xtreg, re`)
Uses the exact **Swamy-Arora GLS** estimator:
```	ext
xtreg y x1 x2, re
// In unbalanced panels, automatically computes harmonic mean group size T_bar = n / sum(1/T_i)
```

#### Hausman Specification Test (`hausman`)
```	ext
// Fit and store FE model
xtreg y x1 x2, fe
estimates store fe_mod

// Fit and store RE model
xtreg y x1 x2, re
estimates store re_mod

// Conduct Hausman test
hausman fe_mod re_mod
// If (V_fe - V_re) is not positive definite, returns Moore-Penrose generalized inverse chi2
```

### Constrained Regression & Seemingly Unrelated Regression
```	ext
// Linear equality constraints R*beta = r
constraint 1 mpg + weight = 0
cnsreg price mpg weight foreign, constraints(1)

// Zellner's Seemingly Unrelated Regression (FGLS)
sureg (eq1: price mpg weight) (eq2: displacement weight foreign)
```

---

## 6. Discrete Choice, Count, & Survival Models

### Binary & Multinomial Choice
```	ext
// Binary Logistic Regression
logit foreign mpg weight, vce(robust)
logit foreign mpg weight, or              // Reports Odds Ratios exp(beta)

// Binary Probit Model
probit foreign mpg weight

// Ordered Logistic Regression
ologit rep78 mpg weight
estat brant                               // Brant test of parallel regression assumption

// Multinomial Logit
mlogit insurance age income, baseoutcome(1)
mlogit insurance age income, rrr          // Relative Risk Ratios
```

### Count Models (Poisson & Negative Binomial)

These commands fit counts with unit exposure. Supplying `exposure()` or `offset()` returns an error.
```	ext
// Poisson Regression
poisson docvis age income, irr

// Negative Binomial II (NB2) with estimated constant dispersion alpha
nbreg docvis age income, irr
// Automatically performs Likelihood Ratio test of alpha = 0 (Poisson vs NB2)
```

### Censored & Quantile Regression
```	ext
// Tobit model left-censored at 0
tobit hours wages educ, ll(0)

// Median Regression (50th percentile)
qreg price mpg weight, quantile(0.50)

// 75th percentile regression
qreg price mpg weight, quantile(0.75)
```

### Survival Analysis
```	ext
// Set survival records: timevar is duration, died is failure event indicator
stset fail_time, failure(died)

// Kaplan-Meier product-limit survivor table
sts list
// Reports: Time, Beg. Total, Fail, Net Lost, Survivor Function, Greenwood Std. Err., [95% Conf. Int.]

// Kaplan-Meier survivor curve graph
sts graph

// Cox Proportional Hazards Model (default Breslow ties, Efron option available)
stcox age drug, ties(breslow)
stcox age drug, ties(efron)
// Reports Hazard Ratios exp(beta), robust standard errors, z, P>|z|
```

---

## 7. Time Series & Filtering

```	ext
// Set time variable
tsset time

// Hodrick-Prescott Filter
// Decomposes y into cyclical component (cycle_y) and smoothed trend (trend_y)
tsfilter hp cycle_y = y, smooth(1600) trend(trend_y)

// Vector Autoregression (VAR)
var y1 y2, lags(1 2)
// Reports multi-equation estimates, AIC, HQIC, and SBIC criteria

// Augmented Dickey-Fuller Unit Root Test
dfuller y, lags(2)

// Autocorrelations & Partial Autocorrelations (PACF)
corrgram y, lags(12)
```

---

## 8. Modern Causal Inference

Potatostics includes native, built-in implementations of cutting-edge causal inference methods that traditional software packages lack:

### Callaway & Sant'Anna (2021) Staggered DiD (`csdid`)
Solves the negative weighting and dynamic heterogeneity bias of classic Two-Way Fixed Effects (TWFE):
```	ext
csdid outcome cov1 cov2, ivar(id) time(year) gvar(cohort)
```
- Estimates group-time average treatment effects $ATT(g, t)$.
- Generates dynamic event-study aggregations.
- Uses multiplier bootstrap and small-sample corrected Wald tests.

### Synthetic Control Method (`synth`)
```	ext
synth gdp trade inflation, trunit(1) trperiod(1975)
```
- Fits donor pool weights using constrained convex quadratic programming.
- Computes pre- and post-intervention Mean Squared Prediction Error (MSPE).
- Conducts automated in-space placebo permutation tests across all donor units.
- Plot synthetic comparison: `synth_plot`.

### Dynamic Event-Study Regression (`eventstudy`)
```	ext
eventstudy outcome treated, ivar(id) tvar(year) lead(3) lag(4)
```
- Automatically omits reference period $k = -1$.
- Computes joint Wald $\chi^2$ test of pre-trends ($H_0: \beta_{-k} = 0 \quad \forall k \ge 2$).
- Generates publication-ready event-study plots with 95% confidence bands.

### Propensity Score Matching (`psmatch`)
```	ext
psmatch treatment cov1 cov2 cov3, outcome(price) caliper(0.05)
```
- Computes logit propensity scores and common support regions.
- Performs vectorized nearest-neighbor matching with strict distance caliper enforcement.
- Prints full covariate balance table with standardized % differences and two-sample $t$-tests.

---

## 9. Complex Survey Data Analysis

Potatostics implements Taylor-series linearization for stratified multistage cluster survey designs:

```	ext
// Declare complex survey design
svyset [pweight=samp_weight], strata(region) psu(psu_id)

// Linear regression with survey design variance adjustment
svy: regress income education experience

// Clear survey design
svyset _clear
```
- Accurately adjusts standard errors for intra-cluster correlation and stratification.
- Calculates correct survey design degrees of freedom $d = \sum L_h - H$.

---

## 10. Post-Estimation & Hypothesis Testing

### Linear & Nonlinear Hypothesis Testing
```	ext
regress price mpg weight length foreign

// Test linear equality of coefficients
test mpg = weight
test (mpg = 0) (weight = 0)

// Test linear combination
lincom mpg - weight

// Delta-method nonlinear combinations
nlcom _b[mpg] / _b[weight]

// Joint nonlinear Wald test
testnl (_b[mpg] / _b[weight] = 1)
```

### Average Marginal Effects (`margins`)
```	ext
probit foreign mpg weight
margins, dydx(*)   // Computes dy/dx average marginal effects with delta-method SEs
```

### Storing & Comparing Estimation Results
```	ext
regress price mpg
estimates store m1

regress price mpg weight
estimates store m2

estimates table m1 m2, b(%9.4f) se stats(N r2)
```

---

## 11. Publication Reporting & Academic Graph Studio

### Academic Chart Studio
Potatostics generates publication-quality figures styled after leading journals:
```	ext
// Supported Plot Commands
scatter price mpg
line y time
binscatter price mpg, nquantiles(20)
rdplot outcome running_var, c(0)
histogram price, bins(20)
boxplot price, by(foreign)
heatmap price mpg weight length
coefplot                             // Plots regression coefficients with confidence bars
```
Available themes: **AER** (American Economic Review), **Nature**, **The Economist**, and **Classic**.

### Direct Table Export (`esttab`)
Export publication tables with stars ($^*p<0.10, ^{**}p<0.05, ^{***}p<0.01$), standard errors, and fit statistics:
```	ext
// LaTeX booktabs export
esttab m1 m2 using "table1.tex", replace booktabs title("Wage Equations")

// Microsoft Word (.docx) export
esttab m1 m2 using "table1.docx", replace title("Table 1: Main Results")

// HTML table export
esttab m1 m2 using "table1.html", replace
```

### Direct Excel Cell Export (`putexcel`)
```	ext
putexcel set "report.xlsx", sheet("Results") modify
putexcel A1 = ("Empirical Analysis Summary")
putexcel A3 = ("Sample Size") B3 = (e(N))
putexcel A4 = ("R-Squared")   B4 = (e(r2))
putexcel A6 = matrix(e(b))
```

---

## 12. Big Data & Out-of-Core Streaming

When datasets exceed system RAM, standard software crashes with memory errors. Potatostics provides the **`bigregress`** streaming engine:

```	ext
bigregress price mpg weight using "huge_dataset.csv", chunksize(100000)
```
- Reads the dataset in memory-capped chunks using high-speed Arrow streaming.
- Updates the cross-product matrices $X^\top X$ and $X^\top y$ online.
- Computes exact OLS coefficients and robust standard errors in $O(K^2)$ memory space, enabling regression on 100+ GB files on an ordinary laptop.

---

## 13. Community Package Hub & Authoring Custom Plugins

Potatostics includes a decentralized package ecosystem equivalent to Standard Econometrics's SSC:

### Installing Community Packages
```text
pkg search did           // Search community packages
pkg install winsorize    // Install package into plugins directory (disabled by default)
pkg enable winsorize     // Activate and load package commands
pkg list                 // List installed packages and status (enabled/disabled)
pkg disable winsorize    // Deactivate package without uninstalling
pkg uninstall winsorize  // Remove package

// Syntax aliases are fully supported:
ssc install bacondecomp
pkg enable bacondecomp
net install mypkg, from("https://example.com/mypkg.py")
```

> **Security & Activation Notice**: For security and protection against untrusted code, packages are installed in a disabled state by default. Activate them with `pkg enable <name>` or through the GUI Plugins Manager dialog.

### Authoring a New Command in Python
Generate a template:
```	ext
plugin new mycommand
```
This generates `mycommand.py` in your user plugins directory:
```python
"""
Plugin: mycommand
Author: Your Name
"""
from engine.parser import CommandResult

def mycommand_cmd(interp, cmd):
    df = interp.dataset.df
    tokens = cmd.body.split()
    if not tokens:
        return CommandResult("syntax: mycommand varname", ok=False, return_code=198)

    var = tokens[0]
    val = float(df[var].mean())
    interp.registers.clear_r()
    interp.registers.set_r_scalar("mean", val)
    return CommandResult(f"Computed custom mean for {var}: {val:.4f}", ok=True)

def register_plugin(interp):
    interp.register_command("mycommand", mycommand_cmd)
```
Once saved, type `plugin enable mycommand`, and `mycommand` is immediately available in the console!

---

## 14. Do-File Scripting, Macros, & Automation

### Running Do-Files
```	ext
do "analysis.do"
```

### Local & Global Macros
```	ext
local controls "mpg weight length"
regress price `controls'

global datapath "./data/project"
use "$datapath/auto.dta", clear
```

### Loops & Flow Control
```	ext
// Foreach loop over variables
foreach v in price mpg weight {
    summarize `v'
}

// Forvalues loop over numeric sequences
forvalues yr = 2010/2020 {
    display "Processing year `yr'..."
}
```

### Accessing Return Registers
- `r(...)` scalars from general commands:
  - `r(N)`, `r(mean)`, `r(sd)`, `r(min)`, `r(max)`, `r(sum)`, `r(p50)`.
  - View with: `return list`.
- `e(...)` scalars & matrices from estimation commands:
  - `e(N)`, `e(r2)`, `e(df_r)`, `e(df_m)`, `e(F)`, `e(ll)`, `e(rmse)`.
  - Access estimated coefficients: `_b[varname]` or `_b[_cons]`.
  - Access standard errors: `_se[varname]`.
  - View with: `ereturn list`.

---

## 15. Command Syntax & Compatibility Matrix

Potatostics provides an extensive command vocabulary mirroring standard econometric workflows while adding modern causal, big-data, and high-performance computing extensions.

### 15.1 Core Econometric Compatibility Matrix

| Command | Family / Domain | Primary Syntax & Key Options | Return Registers Populated | Reference Benchmark Parity |
| :--- | :--- | :--- | :--- | :---: |
| **`regress`** | OLS / Linear Regression | `regress dep indep [if] [in], [vce(robust\|hc1\|hc2\|hc3\|cluster id)] [beta]` | `e(b)`, `e(V)`, `e(r2)`, `e(r2_a)`, `e(F)`, `e(df_r)`, `e(df_m)`, `e(rmse)` | Exact ($\Delta < 10^{-10}$) |
| **`reghdfe`** | Multi-Way High-Dimensional FE | `reghdfe dep indep, absorb(fe1 fe2 ...) [cluster(id)]` | `e(b)`, `e(V)`, `e(r2)`, `e(r2_within)`, `e(N)`, `e(df_r)` | Exact ($\Delta < 10^{-9}$) |
| **`xtreg`** | Panel Data (FE / RE) | `xtreg dep indep, fe\|re [robust] [cluster(id)]` | `e(b)`, `e(V)`, `e(r2_o)`, `e(r2_b)`, `e(r2_w)`, `e(sigma_u)`, `e(sigma_e)`, `e(rho)` | Exact ($\Delta < 10^{-12}$) |
| **`ivregress`**| Instrumental Variables (2SLS) | `ivregress 2sls dep (endog = inst) [exog] [, robust cluster(id)]` | `e(b)`, `e(V)`, `e(r2)`, `e(N)`, `e(df_r)` | Exact ($\Delta < 10^{-15}$) |
| **`logit`** | Binary Logistic Regression | `logit dep indep [if] [, or vce(robust\|cluster id)]` | `e(b)`, `e(V)`, `e(ll)`, `e(ll_0)`, `e(chi2)`, `e(p)`, `e(r2_p)` | Exact ($\Delta < 10^{-11}$) |
| **`probit`** | Binary Probit Regression | `probit dep indep [if] [, vce(robust\|cluster id)]` | `e(b)`, `e(V)`, `e(ll)`, `e(ll_0)`, `e(chi2)`, `e(p)`, `e(r2_p)` | Exact ($\Delta < 10^{-14}$) |
| **`poisson`** | Count Data MLE | `poisson dep indep [if] [, irr vce(robust\|cluster id)]` | `e(b)`, `e(V)`, `e(ll)`, `e(ll_0)`, `e(chi2)`, `e(p)` | Exact ($\Delta < 10^{-10}$) |
| **`nbreg`** | Negative Binomial (NB-2) | `nbreg dep indep [if] [, irr disp(mean)]` | `e(b)`, `e(V)`, `e(ll)`, `e(alpha)`, `e(chi2)` | Exact ($\Delta < 10^{-8}$) |
| **`mlogit`** | Multinomial Logit | `mlogit dep indep [if] [, baseoutcome(#) rrr]` | `e(b)`, `e(V)`, `e(ll)`, `e(k_out)`, `e(chi2)` | Exact ($\Delta < 10^{-8}$) |
| **`qreg`** | Quantile Regression | `qreg dep indep [if] [, quantile(0.1..0.9)]` | `e(b)`, `e(q)`, `e(sum_adev)`, `e(sum_rdev)`, `e(r2_p)` | Exact ($\Delta < 10^{-8}$) |
| **`tobit`** | Censored Normal Regression | `tobit dep indep [if], ll[(#)] ul[(#)]` | `e(b)`, `e(V)`, `e(sigma)`, `e(ll)`, `e(N_unc)` | Exact ($\Delta < 10^{-8}$) |
| **`sureg`** | Seemingly Unrelated Regressions| `sureg (eq1: y1 x1) (eq2: y2 x2) [, isur corr]` | `e(b)`, `e(V)`, `e(Sigma)`, `e(chi2)`, `e(r2)` | Exact ($\Delta < 10^{-8}$) |
| **`cnsreg`** | Constrained Linear Regression | `cnsreg dep indep, constraints(1 2 ...)` | `e(b)`, `e(V)`, `e(r2)`, `e(df_r)` | Exact ($\Delta < 10^{-9}$) |
| **`csdid`** | Staggered DiD (Callaway & Sant'Anna) | `csdid depvar, ivar(id) time(t) gvar(g) [agg(event\|simple)]` | `e(b)`, `e(V)`, `e(att)`, `e(b_att)`, `e(se_att)` | Exact ($\Delta < 10^{-8}$) |
| **`bacondecomp`** | DiD Timing Decomposition | `bacondecomp dep treat, ivar(id) time(t)` | `r(twfe)`, `r(omega_never)`, `r(omega_earlier)`, `r(omega_later)` | Exact ($\Delta < 10^{-15}$) |
| **`drdid`** | Doubly Robust DiD (Sant'Anna & Zhao) | `drdid dep treat time [covars], ivar(id)` | `e(b)`, `e(V)`, `r(att)`, `r(se)` | Verified (ATT within 0.03) |
| **`synth`** | Synthetic Control Method | `synth dep covars, trunit(#) trperiod(#) xperiod(...)` | `e(W)`, `e(V)`, `e(rmspe_pre)`, `e(rmspe_post)` | Exact ($\Delta < 10^{-7}$) |
| **`eventstudy`**| Dynamic Event Study DiD | `eventstudy dep treat, ivar(id) tvar(t) [window(pre post)]` | `e(b)`, `e(V)`, `e(leads)`, `e(lags)` | Exact ($\Delta < 10^{-8}$) |
| **`stcox`** | Cox Proportional Hazards | `stcox indepvars [, ties(breslow\|efron)]` | `e(b)`, `e(V)`, `e(ll)`, `e(chi2)`, `e(p)` | Exact ($\Delta < 10^{-8}$) |
| **`sts list`** | Kaplan-Meier Life Tables | `sts list [, by(group) compare]` | `r(table)` with survivor, Greenwood SE, CI | Exact ($\Delta < 10^{-10}$) |
| **`var`** | Vector Autoregression | `var y1 y2 ..., lags(1..k) [dfk]` | `e(b_dict)`, `e(V)`, `e(aic)`, `e(bic)`, `e(hqic)`, `e(fpe)` | Exact ($\Delta < 10^{-12}$) |
| **`tsfilter hp`**| Hodrick-Prescott Filter | `tsfilter hp var, smooth(#)` | Generates trend & cycle variables | Exact ($\Delta < 10^{-10}$) |
| **`svy:`** | Complex Survey Prefix | `svyset psu [pw=w], strata(str) fpc(fpc) \| svy: reg ...`| `e(b)`, `e(V_svy)`, `e(df_r)`, `e(N_pop)`, `e(N_psu)` | Exact ($\Delta < 10^{-8}$) |

---

### 15.2 Factor Variables & Interaction Operators

Potatostics implements native factor-variable syntax in estimation commands:

| Syntax | Notation | Meaning | Example |
| :--- | :--- | :--- | :--- |
| `i.var` | Categorical indicator | Omits base category (first by default) | `regress price i.rep78` |
| `b#.var` | Explicit base level | Sets category `#` as the omitted base reference | `regress price b(3).rep78` |
| `c.var` | Continuous variable | Explicit continuous covariate identifier | `regress price c.weight` |
| `i.x#c.z` | Factor-continuous interaction | Interacts each category dummy with continuous variable | `regress price i.foreign#c.mpg` |
| `i.x#i.y` | Factor-factor interaction | Complete Cartesian product of cell indicators | `regress price i.foreign#i.rep78` |
| `c.x##c.z` | Full factorial (main + interaction) | Expands to `c.x + c.z + c.x#c.z` | `regress price c.weight##c.mpg` |

---

### 15.3 Time Series Operators

When a dataset is configured via `tsset timevar` or `xtset panelvar timevar`:

| Operator | Meaning | Example |
| :--- | :--- | :--- |
| `L.` or `L1.` | First lag ($x_{t-1}$) | `regress y L.y x` |
| `L2.`, `L3.` | Higher-order lags ($x_{t-k}$) | `regress y L(1/3).y` |
| `F.` or `F1.` | Lead ($x_{t+1}$) | `regress y F.x` |
| `D.` | First difference ($x_t - x_{t-1}$) | `regress D.y D.x` |
| `D2.` | Second difference ($D.(D.x)$) | `regress D2.y` |
| `S.` or `S#.` | Seasonal difference ($x_t - x_{t-k}$) | `regress S4.sales` |

---

### 15.4 Post-Estimation Commands

Following any estimation command (`regress`, `xtreg`, `logit`, etc.), the following routines are immediately accessible:

| Post-Estimation Command | Purpose | Output / Register Populated |
| :--- | :--- | :--- |
| **`predict newvar [, options]`** | Compute fitted values, residuals, index, or probabilities | `rstandard` computes internally standardized residuals for identifiable, unchanged OLS estimation observations. |
| **`test exp = exp`** | Wald hypothesis test of linear restrictions | Reports $F$ or $\chi^2$, degrees of freedom, and $p$-value; sets `r(p)`, `r(F)` |
| **`testparm varlist`** | Joint significance Wald test of multiple coefficients | Joint $F$-test; sets `r(p)`, `r(F)` |
| **`lincom exp`** | Linear combination of estimators | Estimate, standard error, $t/z$, $p$-value, 95% CI; sets `r(estimate)`, `r(se)` |
| **`nlcom exp`** | Non-linear parameter combination (delta method) | Numerical gradient, standard error, $z$-statistic, 95% CI; sets `r(estimate)`, `r(se)` |
| **`margins, dydx(var)`** | Average marginal effects | Models with interaction terms are unsupported. For nonlinear fits saved without the estimation design, refit before running this command. |
| **`estat hettest`** | Breusch-Pagan / Cook-Weisberg heteroskedasticity test | $\chi^2$ statistic and $p$-value; sets `r(chi2)`, `r(p)` |
| **`estat vif`** | Variance Inflation Factors for multicollinearity | VIF table and 1/VIF tolerances; sets `r(mean_vif)` |
| **`estat dwatson`** | Durbin-Watson test for autocorrelation | DW test statistic; sets `r(dw)` |
| **`estimates store name`** | Persist current active estimation model into memory | Stored under `name` |
| **`estimates restore name`** | Restore previously stored model as active estimation | Re-populates `e(...)` and post-estimation registers |
| **`estimates table [names]`** | Side-by-side comparison of multiple stored models | Clean comparison matrix with SEs, stars, and model fit stats |

---

### 15.5 Scripting, Macro Expansion, & Control Flow

| Feature | Syntax | Behavior |
| :--- | :--- | :--- |
| **Local Macros** | `local myvar "price"`<br>``display `myvar'`` | Scoped to current do-file script or execution block |
| **Global Macros** | `global root_dir "C:/data"`<br>`display "$root_dir"` | Persists across interactive session and across scripts |
| **Scalars** | `scalar theta = 3.14159`<br>`display scalar(theta)` | Scoped scalar variable storage; sets `r()` registers |
| **Foreach Loops** | `foreach v in mpg weight length {`<br>`  summarize `v'`<br>`}` | Iterates over token lists or variable lists |
| **Forvalues Loops** | `forvalues i = 1/10 {`<br>`  display `i'`<br>`}` | Numerical range iteration with optional step (`1(2)10`) |
| **Conditionals** | `if `r(N)' > 50 {`<br>`  regress y x`<br>`}` | Conditional execution block evaluated via safe AST parser |

---

### 15.6 Known Architectural Differences & Improvements

1. **Zero Matrix Size Paywalls**: Matrix size is bounded only by available RAM, with no artificial 800-variable or 11,000-variable dimension caps.
2. **64-bit Big-Data Streaming**: Native support for Parquet and arrow memory structures with out-of-core chunked processing (`bigdata load "billion_rows.parquet", chunksize(100000)`).
3. **AST Evaluation Security**: Expressions in `if`, `while`, and `nlcom` are parsed into abstract syntax trees without unsafe Python runtime execution (`eval`/`exec`).
4. **Deterministic Random Generators**: Modern PCG64 seeded generation (`seed(42)`) guarantees exact bit-identical reproducibility across operating systems.
5. **Human-Readable Diagnostics**: Clear tracebacks pinpoint the exact variable name, missing value, collinearity constraint, or singular matrix without opaque error codes.

---

## 16. Academic Citations & Reproducibility

Potatostics adheres to open-science and empirical defensibility standards. Every econometric routine is backed by foundational peer-reviewed literature.

### Generating In-App Citations
Type `cite` at any time to obtain the academic reference and BibTeX entry for the active or previous model:
```text
cite csdid
```
Output:
```text
Academic Citation & Reference: CSDID
========================================================================

[1] Callaway, Brantly, & Sant'Anna, Pedro H. C. (2021).
    "Difference-in-Differences with multiple time periods".
    Journal of Econometrics, 225(2): 200-230.
    DOI: https://doi.org/10.1016/j.jeconom.2020.12.001

BibTeX Entry:
@article{callaway2021difference,
  author    = {Callaway, Brantly and Sant'Anna, Pedro H. C.},
  title     = {Difference-in-Differences with multiple time periods},
  journal   = {Journal of Econometrics},
  year      = {2021},
  volume    = {225},
  number    = {2},
  pages     = {200--230},
  doi       = {10.1016/j.jeconom.2020.12.001}
}
------------------------------------------------------------------------
```

### Complete Bibliography (`references.bib`)
The complete, curated BibTeX bibliography of all 42 foundational papers and textbooks is maintained in [`References/references.bib`](../References/references.bib). Researchers can directly import this file into LaTeX, Overleaf, or reference managers like Zotero and Mendeley.

---
*Potatostics is licensed under the GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later).*
