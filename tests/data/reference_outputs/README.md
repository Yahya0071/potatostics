# External Benchmark Cross-Check Logs

This directory (`tests/data/reference_outputs/`) is used to store raw reference log files generated from real econometric sessions for automated cross-checking against Potatostics.

## Instructions: How to Generate the Reference Log

1. Launch your reference statistical software.
2. Run the following commands:

```text
// Set up text log
sysuse auto, clear
log using "auto_benchmarks.log", text replace

// 1. OLS Regression
regress price mpg weight

// 2. Clustered Logit
logit foreign mpg weight, vce(cluster rep78)

// 3. Panel Fixed Effects Regression
xtset rep78
xtreg price mpg weight, fe

// Close log
log close
```

3. Save the resulting `auto_benchmarks.log` in this directory (`tests/data/reference_outputs/auto_benchmarks.log`).

## Automated Cross-Check Test

When `auto_benchmarks.log` (or any `.log` file in this directory) is present:
- `tests/test_stage17_reference_crosscheck.py` automatically parses the reference regression output tables.
- It extracts the estimated coefficients (`Coefficient` / `Coef.`) and standard errors (`Std. err.` / `Std. Err.`).
- It executes the identical model specifications in Potatostics on `tests/data/auto.dta`.
- It verifies that the coefficients and standard errors match within 4 decimal places ($\le 10^{-4}$ tolerance).

If no log file is present, the test is automatically skipped with a descriptive message directing the tester to this README.
