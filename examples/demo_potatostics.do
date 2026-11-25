// =============================================================================
// Potatostics Master Demonstration Script (demo_potatostics.do)
// Showcases full econometric workflow: data management, advanced estimators,
// causal inference, multi-model LaTeX/Word reporting, and big data streaming.
// =============================================================================

// 1. Data Ingestion & Summary Statistics
use "sample_panel_auto.csv", clear
describe
summarize price mpg weight sales
correlate price mpg weight sales
tabulate maker

// 2. Data Wrangling & Transformations
egen mean_price = mean(price)
generate high_price = price > mean_price
encode maker, gen(maker_code)

// 3. Multi-Model Estimation Pipeline
// Model 1: Standard OLS
regress price mpg weight foreign
estimates store M1_OLS

// Model 2: Heteroskedasticity-Robust OLS (HC1)
regress price mpg weight foreign, robust
estimates store M2_Robust

// Model 3: High-Dimensional Multi-Way Fixed Effects (Unit + Year)
reghdfe price sales treated, absorb(unit_id year) vce(cluster unit_id)
estimates store M3_HDFE

// Model 4: Binary Probit Discrete Choice Model
probit foreign mpg price
estimates store M4_Probit

// 4. Multi-Model Publication Tables (esttab)
// Monospace Comparison in Console
esttab M1_OLS M2_Robust M3_HDFE

// Native LaTeX booktabs Export
esttab using "demo_table.tex", booktabs title("Vehicle Price Determinants")

// Native Microsoft Word (.docx) Table Export
esttab using "demo_table.docx", docx title("Empirical Results Summary")

// Standard CSV Spreadsheet Export
esttab using "demo_table.csv", csv

// 5. Modern Causal Inference Suite
// Callaway & Sant'Anna (2021) Staggered Difference-in-Differences
csdid price, ivar(unit_id) timevar(year) gvar(cohort) control(notyettreated)

// Dynamic Event Study with 95% Confidence Intervals & Vector Plot Export
eventstudy price, unit(unit_id) time(year) treattime(cohort) pre(3) post(4) savegraph("demo_eventstudy.png")

// 6. High-Performance Big Data & Parquet Streaming
regress price mpg weight using "sample_panel_auto.parquet", robust

// End of demonstration script
