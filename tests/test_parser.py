"""
tests/test_parser.py - Comprehensive Unit and Regression Test Suite for Phase 2A
Covers:
1. Lexer & Grammar: Top-level comma splitting outside quotes and parens (paths, function args).
2. Option parser: Flags, argument options, nested parentheses.
3. Qualifiers: 'if exp' and 'in range' expressions and combination parsing.
4. Security & AST Whitelist: Rejection of __import__, open, eval, lambda, attribute access.
5. Whitelisted functions: inrange, inlist, missing, log, ln, exp, sqrt, abs, round, floor, ceil.
6. Execution with qualifiers across commands (summarize, regress, logit, tabulate, count, list, replace, generate).
7. Factor-variable notation (i.var, ib#.var, c.var, i.a#i.b, c.x##i.d).
8. Time series tsset and lag/lead/difference operators (L., L2., D., F., gap detection).
9. Variance options: vce(robust), vce(hc2), vce(hc3), vce(cluster var) compared to statsmodels.
10. Error behavior and structured CommandResult (ok, return_code, error).
"""

import os
import unittest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.parser import (
    parse_in_range,
    parse_command_line,
    evaluate_dta_expression,
    expand_factor_variables,
    CommandResult,
)

SAMPLE_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data.csv")


class TestParserGrammar(unittest.TestCase):
    """Tests for Phase 2A top-level tokenization and grammar parsing."""

    def test_01_top_level_comma_options(self):
        cmd = parse_command_line("regress price mpg weight, robust")
        self.assertEqual(cmd.name, "regress")
        self.assertEqual(cmd.body, "price mpg weight")
        self.assertTrue(cmd.has_option("robust"))

    def test_02_comma_in_quotes_file_path(self):
        cmd = parse_command_line('use "C:\\data,folder\\my,file.csv", clear')
        self.assertEqual(cmd.name, "use")
        self.assertEqual(cmd.body, '"C:\\data,folder\\my,file.csv"')
        self.assertTrue(cmd.has_option("clear"))

    def test_03_comma_in_function_args(self):
        cmd = parse_command_line("generate z = round(x, 0.1)")
        self.assertEqual(cmd.name, "generate")
        self.assertEqual(cmd.body, "z = round(x, 0.1)")
        self.assertEqual(cmd.options, {})

    def test_04_comma_in_function_args_with_options(self):
        cmd = parse_command_line("generate z = round(x, 0.1), replace")
        self.assertEqual(cmd.name, "generate")
        self.assertEqual(cmd.body, "z = round(x, 0.1)")
        self.assertTrue(cmd.has_option("replace"))

    def test_05_comma_in_multiple_functions(self):
        cmd = parse_command_line("generate w = max(a, b) + min(c, d)")
        self.assertEqual(cmd.name, "generate")
        self.assertEqual(cmd.body, "w = max(a, b) + min(c, d)")

    def test_06_nested_parentheses_options(self):
        cmd = parse_command_line("regress y x, vce(cluster id)")
        self.assertEqual(cmd.get_option("vce"), "cluster id")

    def test_07_deeply_nested_parentheses_in_options(self):
        cmd = parse_command_line("cmd arg, opt(((a, b)))")
        self.assertEqual(cmd.get_option("opt"), "((a, b))")

    def test_08_quotes_with_embedded_spaces_and_brackets(self):
        cmd = parse_command_line('use "my file name [1] (test).csv", clear')
        self.assertEqual(cmd.name, "use")
        self.assertEqual(cmd.body, '"my file name [1] (test).csv"')
        self.assertTrue(cmd.has_option("clear"))


class TestParserQualifiers(unittest.TestCase):
    """Tests for Reference 'in' and 'if' qualifier extraction."""

    def test_09_in_range_single_index(self):
        start, end = parse_in_range("5")
        self.assertEqual((start, end), (5, 5))

    def test_10_in_range_from_to(self):
        start, end = parse_in_range("2/8")
        self.assertEqual((start, end), (2, 8))

    def test_11_in_range_negative_index(self):
        start, end = parse_in_range("-3/-1")
        self.assertEqual((start, end), (-3, -1))

    def test_12_in_range_first_last(self):
        start, end = parse_in_range("1/l", total_n=50)
        self.assertEqual((start, end), (1, 50))
        start_f, end_f = parse_in_range("f/10")
        self.assertEqual((start_f, end_f), (1, 10))

    def test_13_if_condition_extraction(self):
        cmd = parse_command_line("regress price mpg if foreign == 1")
        self.assertEqual(cmd.name, "regress")
        self.assertEqual(cmd.body, "price mpg")
        self.assertEqual(cmd.if_expr, "foreign == 1")

    def test_14_if_and_in_combined(self):
        cmd = parse_command_line("summarize price mpg if mpg > 20 in 2/9, detail")
        self.assertEqual(cmd.name, "summarize")
        self.assertEqual(cmd.body, "price mpg")
        self.assertEqual(cmd.if_expr, "mpg > 20")
        self.assertEqual(cmd.in_range, (2, 9))
        self.assertTrue(cmd.has_option("detail"))

    def test_15_using_file_clause(self):
        cmd = parse_command_line('merge 1:1 id using "other_file.dta", keep(match)')
        self.assertEqual(cmd.name, "merge")
        self.assertEqual(cmd.body, "1:1 id")
        self.assertEqual(cmd.using_file, "other_file.dta")
        self.assertEqual(cmd.get_option("keep"), "match")

    def test_16_using_file_unquoted(self):
        cmd = parse_command_line("use rawdata.dta, clear")
        self.assertEqual(cmd.name, "use")
        self.assertEqual(cmd.body, "rawdata.dta")
        self.assertTrue(cmd.has_option("clear"))


class TestSecurityASTWhitelist(unittest.TestCase):
    """Verifies that malicious or arbitrary code injection is strictly blocked."""

    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})

    def test_17_reject_import_injection(self):
        with self.assertRaises(ValueError):
            evaluate_dta_expression("__import__('os').system('dir')", self.df)

    def test_18_reject_eval_injection(self):
        with self.assertRaises(ValueError):
            evaluate_dta_expression("eval('1 + 1')", self.df)

    def test_19_reject_open_file_injection(self):
        with self.assertRaises(ValueError):
            evaluate_dta_expression("open('secret.txt').read()", self.df)

    def test_20_reject_attribute_access(self):
        with self.assertRaises(ValueError):
            evaluate_dta_expression("x.__class__.__bases__", self.df)

    def test_21_reject_lambda_execution(self):
        with self.assertRaises(ValueError):
            evaluate_dta_expression("(lambda: 1)()", self.df)

    def test_22_whitelist_dta_functions_inrange(self):
        res = evaluate_dta_expression("inrange(x, 1, 2)", self.df)
        self.assertEqual(list(res), [True, True, False])

    def test_23_whitelist_dta_functions_inlist(self):
        res = evaluate_dta_expression("inlist(x, 1, 3)", self.df)
        self.assertEqual(list(res), [True, False, True])

    def test_24_whitelist_dta_functions_missing(self):
        df_miss = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        res = evaluate_dta_expression("missing(x)", df_miss)
        self.assertEqual(list(res), [False, True, False])


class TestInterpreterQualifiers(unittest.TestCase):
    """Tests 'if' and 'in' execution across at least 5 different statistical and data commands."""

    def setUp(self):
        self.ds = Dataset()
        self.interp = Interpreter(self.ds)
        self.interp.execute(f'use "{SAMPLE_CSV}", clear')

    def test_25_summarize_with_if_and_in(self):
        out = self.interp.execute("summarize price mpg if foreign == 0 in 1/5")
        self.assertTrue(out.ok)
        self.assertIn("Obs", out)
        self.assertIn("price", out)

    def test_26_regress_with_if(self):
        out = self.interp.execute("regress price mpg if weight < 3000")
        self.assertTrue(out.ok)
        self.assertIn("R-squared", out)

    def test_27_logit_with_if(self):
        out = self.interp.execute("logit foreign price mpg if weight < 4000")
        self.assertTrue(out.ok)
        self.assertIn("Logistic regression", out)

    def test_28_tabulate_with_if(self):
        out = self.interp.execute("tabulate foreign if mpg > 20")
        self.assertTrue(out.ok)
        self.assertIn("Freq.", out)

    def test_29_count_with_if(self):
        out = self.interp.execute("count if price > 5000")
        self.assertTrue(out.ok)
        # Verify count returned a number
        self.assertTrue(any(c.isdigit() for c in out))

    def test_30_list_with_if_and_in(self):
        out = self.interp.execute("list price mpg if foreign == 1 in 1/10")
        self.assertTrue(out.ok)

    def test_31_replace_with_if(self):
        out = self.interp.execute("replace price = 9999 if price > 10000")
        self.assertTrue(out.ok)
        self.assertTrue((self.ds.df["price"] <= 9999).all())

    def test_32_generate_with_if(self):
        out = self.interp.execute("generate high_mpg = 1 if mpg > 25")
        self.assertTrue(out.ok)
        self.assertIn("high_mpg", self.ds.df.columns)
        self.assertTrue(pd.isna(self.ds.df.loc[self.ds.df["mpg"] <= 25, "high_mpg"]).all())


class TestFunctionsAndTimeSeries(unittest.TestCase):
    """Tests generate/replace math functions and time-series lag/lead/difference operators."""

    def setUp(self):
        self.ds = Dataset()
        self.interp = Interpreter(self.ds)
        self.interp.execute(f'use "{SAMPLE_CSV}", clear')

    def test_33_math_functions(self):
        out = self.interp.execute("generate l_price = log(price) + sqrt(mpg) + abs(weight) + exp(0)")
        self.assertTrue(out.ok)
        self.assertIn("l_price", self.ds.df.columns)

    def test_34_rounding_and_truncation(self):
        out = self.interp.execute("generate r = round(mpg, 1) + floor(mpg) + ceil(mpg)")
        self.assertTrue(out.ok)
        self.assertIn("r", self.ds.df.columns)

    def test_35_special_variables_n_and_N(self):
        out1 = self.interp.execute("generate row_id = _n")
        out2 = self.interp.execute("generate row_total = _N")
        self.assertTrue(out1.ok)
        self.assertTrue(out2.ok)
        self.assertEqual(list(self.ds.df["row_id"]), list(range(1, self.ds.N + 1)))
        self.assertEqual(list(self.ds.df["row_total"]), [self.ds.N] * self.ds.N)

    def test_36_tsset_and_lag_operator(self):
        # Add sequential time variable
        self.ds.df["time"] = list(range(1, self.ds.N + 1))
        out_ts = self.interp.execute("tsset time")
        self.assertTrue(out_ts.ok)

        out_gen = self.interp.execute("generate l_price = L.price")
        self.assertTrue(out_gen.ok)
        self.assertTrue(pd.isna(self.ds.df["l_price"].iloc[0]))
        self.assertEqual(self.ds.df["l_price"].iloc[1], self.ds.df["price"].iloc[0])

    def test_37_lead_and_difference_operators(self):
        self.ds.df["time"] = list(range(1, self.ds.N + 1))
        self.interp.execute("tsset time")

        self.interp.execute("generate d_price = D.price")
        self.interp.execute("generate f_price = F.price")
        self.assertEqual(self.ds.df["d_price"].iloc[1], self.ds.df["price"].iloc[1] - self.ds.df["price"].iloc[0])
        self.assertEqual(self.ds.df["f_price"].iloc[0], self.ds.df["price"].iloc[1])

    def test_38_tsset_gap_detection(self):
        # Create non-consecutive time values with a gap
        self.ds.df["time_gap"] = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
        out_ts = self.interp.execute("tsset time_gap")
        self.assertTrue(out_ts.ok)
        self.assertIn("gap", out_ts.lower())


class TestFactorVariableNotation(unittest.TestCase):
    """Tests Reference factor variable expansion: i.var, ib#.var, c.var, i.a#i.b, c.x##i.d."""

    def setUp(self):
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            "y": np.random.randn(n) * 10 + 50,
            "mpg": np.random.uniform(15, 35, n),
            "foreign": np.random.choice([0, 1], n),
            "rep78": np.random.choice([1, 2, 3, 4], n),
        })

    def test_39_factor_dummy_expansion_default_base(self):
        terms, new_df, _ = expand_factor_variables(["i.foreign"], self.df)
        self.assertIn("1.foreign", terms)
        self.assertNotIn("0.foreign", terms)  # Base category omitted

    def test_single_level_factor_requires_observed_values_for_fit_and_predict(self):
        frame = pd.DataFrame({"y": [1, 3, 5, 7, 200, 200, 200, 200],
                              "x": range(8), "g": [1.0] * 4 + [np.nan] * 4})
        for terms in ("x i.g", "x c.x#i.g", "c.x##i.g"):
            with self.subTest(terms=terms):
                dataset = Dataset()
                dataset.load_dataframe(frame)
                interp = Interpreter(dataset)
                result = interp.execute(f"regress y {terms}")
                self.assertTrue(result.ok, result.output)
                self.assertEqual(interp.registers.get_e("N"), 4)
                self.assertTrue(interp.execute("predict fitted, xb").ok)
                np.testing.assert_allclose(dataset.df.fitted, [1, 3, 5, 7] + [np.nan] * 4)
        dataset.load_dataframe(frame.assign(g=np.nan))
        self.assertFalse(interp.execute("regress y x i.g").ok)

    def test_40_factor_base_level_selection(self):
        # ib1 selects level 1 as base, so level 0 is included
        terms, new_df, _ = expand_factor_variables(["ib1.foreign"], self.df)
        self.assertIn("0.foreign", terms)
        self.assertNotIn("1.foreign", terms)

    def test_41_continuous_and_factor_interaction(self):
        terms, new_df, _ = expand_factor_variables(["c.mpg#i.foreign"], self.df)
        self.assertIn("c.mpg#1.foreign", terms)

    def test_42_factorial_interaction_expansion(self):
        # c.mpg##i.foreign expands to: c.mpg, 1.foreign, and c.mpg#1.foreign
        terms, new_df, _ = expand_factor_variables(["c.mpg##i.foreign"], self.df)
        self.assertTrue("c.mpg" in terms or "mpg" in terms)
        self.assertIn("1.foreign", terms)
        self.assertIn("c.mpg#1.foreign", terms)

    def test_43_regression_factor_matches_statsmodels_formula(self):
        ds = Dataset()
        ds.load_dataframe(self.df.copy())
        interp = Interpreter(ds)

        out = interp.execute("regress y c.mpg##i.foreign")
        self.assertTrue(out.ok)
        self.assertIn("c.mpg#1.foreign", out)

        # Compare with statsmodels OLS formula
        mod = ols("y ~ mpg * C(foreign, Treatment(reference=0))", data=self.df).fit()
        # Verify R-squared matches to 4 decimal places
        r2_potatostics = interp.estimates.last_model["r2"]
        self.assertAlmostEqual(r2_potatostics, mod.rsquared, places=4)


class TestVarianceAndClustering(unittest.TestCase):
    """Tests vce(robust), vce(hc2), vce(hc3), and vce(cluster var) against statsmodels."""

    def setUp(self):
        np.random.seed(123)
        n = 200
        cluster_ids = np.random.choice(range(1, 21), size=n)  # 20 clusters
        x = np.random.uniform(10, 50, n)
        e = np.random.normal(0, 2 + 0.1 * x)  # heteroskedastic
        y = 5.0 + 2.5 * x + e

        self.df = pd.DataFrame({"y": y, "x": x, "cid": cluster_ids})
        self.ds = Dataset()
        self.ds.load_dataframe(self.df.copy())
        self.interp = Interpreter(self.ds)

    def test_44_vce_robust_hc1_matches_statsmodels(self):
        out = self.interp.execute("regress y x, vce(robust)")
        self.assertTrue(out.ok)
        self.assertIn("Linear regression (Robust)", out)

        # Reference statsmodels HC1
        X = sm.add_constant(self.df["x"])
        ref = sm.OLS(self.df["y"], X).fit(cov_type="HC1")

        b_pot = self.interp.estimates.last_model["coefs"][0]["coef"]
        se_pot = self.interp.estimates.last_model["coefs"][0]["se"]
        self.assertAlmostEqual(b_pot, ref.params["x"], places=6)
        self.assertAlmostEqual(se_pot, ref.bse["x"], places=6)

    def test_45_vce_hc2_and_hc3(self):
        out_hc2 = self.interp.execute("regress y x, vce(hc2)")
        self.assertTrue(out_hc2.ok)
        X = sm.add_constant(self.df["x"])
        ref_hc2 = sm.OLS(self.df["y"], X).fit(cov_type="HC2")
        se_hc2 = self.interp.estimates.last_model["coefs"][0]["se"]
        self.assertAlmostEqual(se_hc2, ref_hc2.bse["x"], places=6)

        out_hc3 = self.interp.execute("regress y x, vce(hc3)")
        self.assertTrue(out_hc3.ok)
        ref_hc3 = sm.OLS(self.df["y"], X).fit(cov_type="HC3")
        se_hc3 = self.interp.estimates.last_model["coefs"][0]["se"]
        self.assertAlmostEqual(se_hc3, ref_hc3.bse["x"], places=6)

    def test_46_vce_cluster_matches_statsmodels(self):
        out_cl = self.interp.execute("regress y x, vce(cluster cid)")
        self.assertTrue(out_cl.ok)
        self.assertIn("Number of clusters", out_cl)

        # Reference statsmodels cluster-robust
        X = sm.add_constant(self.df["x"])
        ref_cl = sm.OLS(self.df["y"], X).fit(cov_type="cluster", cov_kwds={"groups": self.df["cid"]})

        se_pot = self.interp.estimates.last_model["coefs"][0]["se"]
        self.assertAlmostEqual(se_pot, ref_cl.bse["x"], places=6)

    def test_47_vce_cluster_on_logit(self):
        self.df["binary_y"] = (self.df["y"] > np.median(self.df["y"])).astype(int)
        self.ds.load_dataframe(self.df.copy())
        out_logit = self.interp.execute("logit binary_y x, vce(cluster cid)")
        self.assertTrue(out_logit.ok)
        self.assertIn("Number of clusters", out_logit)


class TestStructuredErrorsAndReturnCodes(unittest.TestCase):
    """Tests error handling, return codes, and structured CommandResult objects."""

    def setUp(self):
        self.ds = Dataset()
        self.interp = Interpreter(self.ds)
        self.interp.execute(f'use "{SAMPLE_CSV}", clear')

    def test_48_syntax_error_returns_structured_result(self):
        res = self.interp.execute("regress")
        self.assertIsInstance(res, CommandResult)
        self.assertFalse(res.ok)
        self.assertEqual(res.return_code, 198)

    def test_49_variable_not_found_returns_111(self):
        res = self.interp.execute("regress price nonexistent_var")
        self.assertIsInstance(res, CommandResult)
        self.assertFalse(res.ok)
        self.assertEqual(res.return_code, 111)
        self.assertIn("variable 'nonexistent_var' not found", res)

    def test_50_command_result_backwards_compatibility(self):
        # CommandResult subclasses str so standard string methods work seamlessly
        res = self.interp.execute("summarize price")
        self.assertTrue(res.startswith("Variable"))
        self.assertIn("5609.4000", res)
        self.assertEqual(len(res), len(str(res)))


if __name__ == "__main__":
    unittest.main()
