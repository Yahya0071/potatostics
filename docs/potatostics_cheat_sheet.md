# Potatostics: Comprehensive Command Reference & Econometric Cheat Sheet

Welcome to **Potatostics**, the open-source statistical analysis software and econometric workstation. This cheat sheet provides applied researchers, data scientists, and students with a reference for all syntax, statistical estimators, causal inference methods, and publication tools.

---

## 1. General Syntax Structure

All Potatostics commands follow a standard grammar:

```	ext
[prefix :] command [varlist] [= exp] [if exp] [in range] [using filename] [, options]
```

### Examples:
```	ext
regress price mpg weight if foreign == 1, robust
bysort foreign (price): summarize mpg weight
bigregress price mpg weight using "census_50m.parquet", robust chunksize(250000)
```

---

## 2. Keyboard Shortcuts & Desktop Shell

| Shortcut | Action | Description |
| :--- | :--- | :--- |
| **`Ctrl + K`** | **Spotlight Command Palette** | Fuzzy search launcher for all commands, models, variables, and actions |
| **`Ctrl + Enter`** | **Run Command / Script** | Executes the active command line or selected `.do` file lines |
| **`Ctrl + O`** | **Open Dataset** | Opens `.dta`, `.csv`, `.parquet`, or `.feather` data file |
| **`Ctrl + S`** | **Save Dataset** | Exports active memory dataset to CSV or Parquet |
| **`Ctrl + L`** | **Clear Console** | Clears the results console |
| **`Ctrl + T`** | **Toggle Theme** | Switches between Dark Mode and Light Mode |

---

## 3. Data Ingestion & Storage

| Task | Syntax | Description |
| :--- | :--- | :--- |
| **Load Standard Econometrics (.dta)** | `use "data.dta" [, clear]` | Ingests binary Standard Econometrics dataset (Standard Econometrics .dta format) with value labels |
| **Load CSV** | `use "data.csv" [, clear]` | Reads standard comma-separated file |
| **Load Parquet** | `use "data.parquet", clear` | Zero-copy fast ingestion via PyArrow / Polars |
| **Save Parquet** | `save "output.parquet", replace` | Exports compressed, high-performance columnar storage |
| **Save CSV** | `save "output.csv", replace` | Exports comma-separated tabular data |
| **Inspect Data** | `describe` | Lists all variables, data types, formats, and dataset labels |
| **Count Rows** | `count [if exp]` | Returns observation count in register `r(N)` |
| **Browse / View** | `list [varlist] [in 1/20]` | Displays observation rows in console grid |

---

## 4. Data Transformation & Wrangling

### Variable Creation & Manipulation
```	ext
generate ln_price = log(price)               // Create new variable with math expressions
replace ln_price = 0 if price <= 0           // Conditionally update existing variable
egen mean_mpg = mean(mpg)                    // Extended generate: mean, sd, total, min, max
egen age_group = cut(age), at(0, 18, 65, 100)// Bin numeric values into categories
drop if missing(price)                       // Remove matching rows
keep make price mpg weight                   // Retain only specified variables
recode rep78 (1 2 = 1) (3 = 2) (4 5 = 3)     // Recode numeric values and ranges
duplicates drop id year, force               // Remove duplicate observation rows
```

### Categorical & String Pipelines
```	ext
encode maker, gen(maker_id)                  // Convert string categories into labeled integer factor
decode foreign_code, gen(foreign_str)        // Convert labeled integers back into string names
label define f_lbl 0 "Domestic" 1 "Foreign"  // Define value label mappings
label values foreign f_lbl                   // Attach label to variable
label variable price "Price in 2024 USD"     // Attach descriptive variable label
```

### Relational & Reshaping Operations
```	ext
// Group-by processing with internal _n (current index) and _N (group total)
bysort region (price): generate rank = _n

// Relational merges (1:1, 1:m, m:1) producing automatic _merge diagnosis
merge 1:1 id year using "firm_fundamentals.csv"

// Wide to Long and Long to Wide Pivots
reshape long inc, i(id) j(year)
reshape wide inc, i(id) j(year)

// Fast Aggregation
collapse (mean) price mpg (sum) sales, by(region year)
```

---

## 5. Statistical Estimation Suite

### Linear Models & High-Dimensional Fixed Effects
```	ext
// Classical Ordinary Least Squares
regress price mpg weight

// Heteroskedasticity-Robust Standard Errors (White / HC1)
regress price mpg weight, robust

// Clustered Standard Errors by Firm or State
regress price mpg weight, vce(cluster state_id)

// High-Dimensional Multi-Way Fixed Effects (Alternating Projections)
reghdfe price mpg weight, absorb(firm_id year_id) vce(cluster firm_id)

// Out-of-Core Big Data Streaming OLS (10M - 50M+ rows directly from disk)
bigregress price mpg weight using "bigdata.parquet", robust chunksize(100000)
```

### Nonlinear & Discrete Choice Estimators
```	ext
// Binary Logit (Maximum Likelihood Estimation)
logit foreign price mpg weight

// Binary Logistic Odds Ratios (e^b)
logistic foreign price mpg weight

// Binary Probit Model (Gaussian CDF Link MLE)
probit foreign price mpg weight, robust

// Poisson Count Model (with Incidence Rate Ratios)
poisson citations experience publications, irr robust

// Tobit Censored Normal Regression (Lower limit = 0)
tobit donations income age, ll(0)

// Quantile Regression (Median / 50th percentile or arbitrary quantile)
qreg wage education experience, quantile(0.75)
```

### Endogeneity & Panel Estimators
```	ext
// Two-Stage Least Squares (2SLS Instrumental Variables)
ivregress 2sls wage (education = father_educ mother_educ) experience age

// Panel Data Setup & Fixed Effects Within Estimator
xtset firm_id year
xtreg investment capital sales, fe vce(cluster firm_id)
```

---

## 6. Cutting-Edge Causal Inference Suite

### Callaway & Sant'Anna (2021) Staggered DiD (`csdid`)
Avoids heterogeneous treatment bias in staggered adoption designs:
```	ext
csdid outcome, ivar(unit_id) timevar(year) gvar(cohort_year) control(notyettreated)
```
- Reports group-time average treatment effects: $ATT(g, t)$.
- Aggregates dynamic effects: $ATT(e = t - g)$.
- Computes joint Wald pre-trend parallel trends test.

### Synthetic Control Method (`synth`)
Constrained convex counterfactual optimization ($w_j \ge 0, \sum w_j = 1$):
```	ext
synth gdp trade inflation, trunit(1) trperiod(1990) unit(country_id) time(year)
```
- Calculates pre/post-treatment MSPE and optimal donor pool weights.
- Executes in-space placebo permutation test for exact non-parametric $p$-value.

### Dynamic Event Study (`eventstudy`)
```	ext
eventstudy outcome, unit(id) time(year) treattime(cohort) pre(4) post(5) savegraph("event.png")
```

---

## 7. Diagnostics & Post-Estimation

```	ext
// Multicollinearity
estat vif

// Heteroskedasticity (Breusch-Pagan / Cook-Weisberg test)
estat hettest

// Model Misspecification (Ramsey RESET omitted variable test)
estat ovtest

// Autocorrelation (Durbin-Watson statistic)
estat dwatson

// Weak-Instrument Diagnostic (Montiel Olea-Pflueger effective F-statistic)
estat weakiv

// Average Marginal Effects with Delta-Method SEs
margins, dydx(*)

// Linear Combinations of Estimators
lincom mpg - 2*weight

// Post-Estimation Wald Hypothesis Restriction Test
test mpg = weight
test (mpg = 0) (weight = 0)

// Fitted Values and Residuals
predict y_hat, xb
predict e_hat, resid
```

---

## 8. Publication Hub (`esttab` & Vector Figures)

### Multi-Model Comparison Tables
Store fitted models in memory and generate publication comparison tables:

```	ext
regress price mpg
estimates store Model1

regress price mpg weight
estimates store Model2

reghdfe price mpg weight, absorb(state year)
estimates store Model3

// Console Monospace Comparison
esttab Model1 Model2 Model3

// LaTeX booktabs export (\toprule, \midrule, \bottomrule)
esttab using "table.tex", booktabs title("Empirical Determinants of Price")

// Microsoft Word (.docx) native table export
esttab using "results.docx", docx title("Regression Results")

// CSV Export
esttab using "comparison.csv", csv
```

### Academic Vector Figures (`coefplot`)
```	ext
// Coefficient Plot with American Economic Review (AER) theme
coefplot Model2, theme(aer) save("coefplot_aer.pdf")

// Coefficient Plot with Nature theme
coefplot Model2, theme(nature) save("coefplot_nature.png")
```

---

## 9. Scripting, Loops, & Return Memory

### Local and Global Macros
```	ext
local controls "mpg weight length"
regress price `controls'

global datadir "./data"
use "$datadir/census.dta", clear
```

### Control Flow Loops
```	ext
// Loop over list of variables
foreach v of varlist price mpg weight {
    summarize `v'
}

// Loop over numeric sequence
forvalues y = 2015/2022 {
    count if year == `y'
}
```

### Return Registers (`r()` and `e()`)
```	ext
summarize price
display r(mean)
display r(sd)

regress price mpg weight
display e(r2)
display e(N)
display _b[mpg]
display _se[mpg]

return list
ereturn list
```

---

## 10. Command assistant (rule-based)

Type `ai <natural language prompt>` in the command bar or press `Ctrl+K`:

```	ext
// Natural Language Estimation
ai "regress price on mpg and weight with robust standard errors"

// Contextual Follow-Up (Short-Term Memory)
ai "what was the R2?"
ai "add robust standard errors"
ai "explain that"

// Persistent Project Codebooks (Long-Term Memory)
ai "codebook set price = Vehicle transaction price in 2024 USD"
ai "codebook"
ai "memory status"
```

---

*Potatostics - Open-Source Econometric Platform.*
