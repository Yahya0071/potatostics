"""
tests/test_data_ops.py - Comprehensive Unit & Regression Test Suite for Phase 2B
Covers:
1. by: and bysort: prefix execution, group-local _n, _N, and lagged index evaluation
2. merge 1:1, 1:m, m:1, _merge statuses (1, 2, 3), uniqueness checks (r(459)), keep/assert/generate/replace
3. append using with type union and generate() source flag
4. reshape long and reshape wide round-trip conversions
5. collapse with mean, median, sd, min, max, count, percentiles
6. egen suite (mean, total, sd, rank, group, tag, rowtotal, rowmean, rowmiss, rownonmiss, cut)
7. encode and decode categorical pipeline with attached value labels
8. label management (define, values, variable, list, dir, drop)
9. recode value and range rules
10. duplicates report, duplicates tag, duplicates drop
"""

import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from engine.dataset import Dataset
from engine.interpreter import Interpreter

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV = os.path.join(os.path.dirname(TEST_DIR), "sample_data.csv")


class TestByPrefix(unittest.TestCase):
    """Tests for by: and bysort: prefix execution."""

    def setUp(self):
        self.ds = Dataset()
        self.interp = Interpreter(self.ds)
        # Create small test dataset with groups
        df = pd.DataFrame({
            "group": ["A", "A", "A", "B", "B"],
            "year": [2020, 2021, 2022, 2020, 2021],
            "sales": [100.0, 150.0, 200.0, 50.0, 80.0]
        })
        self.ds.load_dataframe(df)

    def test_01_bysort_descriptive(self):
        res = self.interp.execute("bysort group: summarize sales")
        self.assertTrue(res.ok)
        self.assertIn("group = A", res)
        self.assertIn("group = B", res)

    def test_02_bysort_local_n_and_N(self):
        res_n = self.interp.execute("bysort group (year): generate rank = _n")
        self.assertTrue(res_n.ok)
        self.assertEqual(list(self.ds.df["rank"]), [1, 2, 3, 1, 2])

        res_N = self.interp.execute("bysort group: generate total_in_group = _N")
        self.assertTrue(res_N.ok)
        self.assertEqual(list(self.ds.df["total_in_group"]), [3, 3, 3, 2, 2])

    def test_03_bysort_lag_index(self):
        res = self.interp.execute("bysort group (year): generate lag_sales = sales[_n-1]")
        self.assertTrue(res.ok)
        self.assertTrue(pd.isna(self.ds.df["lag_sales"].iloc[0]))
        self.assertEqual(self.ds.df["lag_sales"].iloc[1], 100.0)
        self.assertEqual(self.ds.df["lag_sales"].iloc[2], 150.0)
        # Group B first element is also NaN (does not bleed from Group A)
        self.assertTrue(pd.isna(self.ds.df["lag_sales"].iloc[3]))
        self.assertEqual(self.ds.df["lag_sales"].iloc[4], 50.0)

    def test_grouped_regression_preserves_failed_group_as_non_model_output(self):
        result = self.interp.execute("bysort group: regress sales year")
        first, second = result.sections()
        self.assertTrue(result.ok)
        self.assertEqual(first.command, "by group = A: regress sales year")
        self.assertTrue(first.is_model)
        self.assertIn("Number of obs   =         3", first.text)
        self.assertEqual(second.command, "by group = B: regress sales year")
        self.assertFalse(second.is_model)
        self.assertIn("Insufficient observations (2)", second.text)
        self.assertEqual(result.output, "\n".join(part.text for part in result.sections()))

        quiet = self.interp.execute("bysort group: quietly regress sales year")
        border = "-" * 78
        self.assertEqual(quiet.output, f"{border}\n -> group = A\n{border}\n\n\n{border}\n -> group = B\n{border}\n\n")
        self.assertFalse(any(part.is_model for part in quiet.sections()))

    def test_grouped_script_keeps_sections_and_original_whitespace(self):
        self.ds.load_dataframe(pd.concat([self.ds.df, pd.DataFrame({
            "group": ["B", "B"], "year": [2022, 2023], "sales": [100.0, 140.0],
        })], ignore_index=True))
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "group_models.do"
            script.write_text("if 1 {\nregress sales year\nsummarize sales\n}\n", encoding="utf-8")
            command = f'do "{script}"'
            result = self.interp.script_runner.run_script(f"if 1 {{\nbysort group: {command}\n}}")
            expected = []
            border = "-" * 78
            for label, group_data in self.ds.df.groupby("group"):
                group_dataset = Dataset()
                group_dataset.load_dataframe(group_data)
                group_script = Interpreter(group_dataset).execute(command)
                expected += [border, f" -> group = {label}", border, group_script.output.strip(), ""]

        self.assertTrue(result.ok)
        self.assertEqual(result.output, "\n".join(expected))
        self.assertEqual(result.output, "\n".join(part.text for part in result.sections()))
        self.assertEqual([(part.command, part.is_model) for part in result.sections()], [
            ("by group = A: regress sales year", True),
            ("by group = A: summarize sales", False),
            ("by group = B: regress sales year", True),
            ("by group = B: summarize sales", False),
        ])


class TestMerge(unittest.TestCase):
    """Tests for relational dataset merging (1:1, 1:m, m:1)."""

    def setUp(self):
        self.ds = Dataset()
        self.interp = Interpreter(self.ds)
        self.master_df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "wage": [25.0, 30.0, 18.0, 42.0],
            "dept": ["HR", "IT", "IT", "Sales"]
        })
        self.ds.load_dataframe(self.master_df)

        self.using_path = os.path.join(TEST_DIR, "test_using_merge.csv")
        self.using_df = pd.DataFrame({
            "id": [2, 3, 4, 5],
            "bonus": [5000, 2000, 8000, 1000],
            "dept_name": ["Information Tech", "Information Tech", "Global Sales", "Executive"]
        })
        self.using_df.to_csv(self.using_path, index=False)

    def tearDown(self):
        if os.path.exists(self.using_path):
            os.remove(self.using_path)

    def test_04_merge_1to1_standard(self):
        res = self.interp.execute(f'merge 1:1 id using "{self.using_path}"')
        self.assertTrue(res.ok)
        self.assertIn("_merge==1", res)
        self.assertIn("_merge==2", res)
        self.assertIn("_merge==3", res)
        # 1 master only (id 1), 1 using only (id 5), 3 matched (id 2, 3, 4)
        status_counts = self.ds.df["_merge"].value_counts().to_dict()
        self.assertEqual(status_counts[1], 1)
        self.assertEqual(status_counts[2], 1)
        self.assertEqual(status_counts[3], 3)

    def test_05_merge_1to1_master_duplicates_error(self):
        # Introduce duplicate id in master
        dup_df = pd.concat([self.master_df, self.master_df.iloc[[0]]], ignore_index=True)
        self.ds.load_dataframe(dup_df)
        res = self.interp.execute(f'merge 1:1 id using "{self.using_path}"')
        self.assertFalse(res.ok)
        self.assertEqual(res.return_code, 459)
        self.assertIn("master data", res.error)

    def test_06_merge_1to1_using_duplicates_error(self):
        # Create using file with duplicate keys
        dup_using = pd.concat([self.using_df, self.using_df.iloc[[0]]], ignore_index=True)
        dup_using_path = os.path.join(TEST_DIR, "dup_using.csv")
        dup_using.to_csv(dup_using_path, index=False)

        res = self.interp.execute(f'merge 1:1 id using "{dup_using_path}"')
        if os.path.exists(dup_using_path):
            os.remove(dup_using_path)
        self.assertFalse(res.ok)
        self.assertEqual(res.return_code, 459)
        self.assertIn("using data", res.error)

    def test_07_merge_keep_and_assert_options(self):
        res_assert = self.interp.execute(f'merge 1:1 id using "{self.using_path}", assert(match)')
        self.assertFalse(res_assert.ok)
        self.assertEqual(res_assert.return_code, 9)

        # Reset master and merge with keep(match)
        self.ds.load_dataframe(self.master_df)
        res_keep = self.interp.execute(f'merge 1:1 id using "{self.using_path}", keep(match)')
        self.assertTrue(res_keep.ok)
        self.assertEqual(len(self.ds.df), 3)
        self.assertTrue((self.ds.df["_merge"] == 3).all())


class TestAppend(unittest.TestCase):
    """Tests for dataset appending."""

    def setUp(self):
        self.ds = Dataset()
        self.interp = Interpreter(self.ds)
        self.ds.load_dataframe(pd.DataFrame({"x": [1, 2], "y": ["A", "B"]}))
        self.append_path = os.path.join(TEST_DIR, "test_append.csv")
        pd.DataFrame({"x": [3, 4], "y": ["C", "D"], "z": [10.5, 20.5]}).to_csv(self.append_path, index=False)

    def tearDown(self):
        if os.path.exists(self.append_path):
            os.remove(self.append_path)

    def test_08_append_with_generate_flag(self):
        res = self.interp.execute(f'append using "{self.append_path}", generate(source)')
        self.assertTrue(res.ok)
        self.assertEqual(len(self.ds.df), 4)
        self.assertEqual(list(self.ds.df["source"]), [0, 0, 1, 1])
        self.assertIn("z", self.ds.df.columns)
        self.assertTrue(pd.isna(self.ds.df["z"].iloc[0]))


class TestReshape(unittest.TestCase):
    """Tests for reshape long and reshape wide."""

    def setUp(self):
        self.ds = Dataset()
        self.interp = Interpreter(self.ds)
        # Wide data
        self.wide_df = pd.DataFrame({
            "id": [1, 2],
            "inc80": [1000, 2000],
            "inc81": [1200, 2200],
            "inc82": [1400, 2400]
        })
        self.ds.load_dataframe(self.wide_df)

    def test_09_reshape_long(self):
        res = self.interp.execute("reshape long inc, i(id) j(year)")
        self.assertTrue(res.ok)
        self.assertEqual(len(self.ds.df), 6)
        self.assertIn("inc", self.ds.df.columns)
        self.assertIn("year", self.ds.df.columns)
        # Verify first row
        self.assertEqual(self.ds.df.iloc[0]["inc"], 1000)

    def test_10_reshape_round_trip(self):
        self.interp.execute("reshape long inc, i(id) j(year)")
        res_wide = self.interp.execute("reshape wide inc, i(id) j(year)")
        self.assertTrue(res_wide.ok)
        self.assertIn("inc80", self.ds.df.columns)
        self.assertIn("inc81", self.ds.df.columns)
        self.assertIn("inc82", self.ds.df.columns)
        self.assertEqual(len(self.ds.df), 2)


class TestCollapse(unittest.TestCase):
    """Tests for collapse data aggregation."""

    def setUp(self):
        self.ds = Dataset()
        self.interp = Interpreter(self.ds)
        self.ds.load_dataframe(pd.DataFrame({
            "group": ["A", "A", "B", "B", "B"],
            "val": [10.0, 20.0, 30.0, 40.0, 50.0],
            "weight": [1.0, 2.0, 3.0, 4.0, 5.0]
        }))

    def test_11_collapse_mean_and_sd(self):
        res = self.interp.execute("collapse (mean) val (sum) weight, by(group)")
        self.assertTrue(res.ok)
        self.assertEqual(len(self.ds.df), 2)
        # Group A mean of val: 15.0; sum of weight: 3.0
        row_a = self.ds.df[self.ds.df["group"] == "A"].iloc[0]
        self.assertEqual(row_a["val"], 15.0)
        self.assertEqual(row_a["weight"], 3.0)


class TestEgen(unittest.TestCase):
    """Tests for egen extended generate functions."""

    def setUp(self):
        self.ds = Dataset()
        self.interp = Interpreter(self.ds)
        self.ds.load_dataframe(pd.DataFrame({
            "x": [10.0, 20.0, 30.0, 40.0],
            "y": [1.0, np.nan, 3.0, 4.0],
            "cat": ["A", "A", "B", "B"]
        }))

    def test_12_egen_mean_total_sd(self):
        self.interp.execute("egen mean_x = mean(x)")
        self.interp.execute("egen sum_x = total(x)")
        self.interp.execute("egen sd_x = sd(x)")
        self.assertEqual(self.ds.df["mean_x"].iloc[0], 25.0)
        self.assertEqual(self.ds.df["sum_x"].iloc[0], 100.0)

    def test_13_egen_row_functions(self):
        self.interp.execute("egen r_tot = rowtotal(x y)")
        self.interp.execute("egen r_miss = rowmiss(x y)")
        self.assertEqual(self.ds.df["r_tot"].iloc[0], 11.0)
        self.assertEqual(self.ds.df["r_tot"].iloc[1], 20.0)  # y is NaN, so rowtotal is 20
        self.assertEqual(self.ds.df["r_miss"].iloc[1], 1)
        self.assertEqual(self.ds.df["r_miss"].iloc[0], 0)

    def test_14_egen_group_and_tag(self):
        self.interp.execute("egen gid = group(cat)")
        self.interp.execute("egen first_occ = tag(cat)")
        self.assertEqual(list(self.ds.df["gid"]), [1, 1, 2, 2])
        self.assertEqual(list(self.ds.df["first_occ"]), [1, 0, 1, 0])


class TestEncodeDecodeAndLabels(unittest.TestCase):
    """Tests for categorical encode, decode, and label commands."""

    def setUp(self):
        self.ds = Dataset()
        self.interp = Interpreter(self.ds)
        self.ds.load_dataframe(pd.DataFrame({
            "country": ["USA", "Germany", "USA", "Japan", "Germany"],
            "code": [1, 2, 1, 3, 2]
        }))

    def test_15_encode_and_decode(self):
        res_enc = self.interp.execute("encode country, gen(country_num)")
        self.assertTrue(res_enc.ok)
        self.assertTrue(pd.api.types.is_numeric_dtype(self.ds.df["country_num"]))
        # Verify value labels mapping exists
        self.assertIn("country_num", self.ds.val_labels)
        labels = self.ds.val_labels["country_num"]
        self.assertEqual(labels[1], "Germany")  # Alphabetical 1st

        # Decode back to string
        res_dec = self.interp.execute("decode country_num, gen(country_str)")
        self.assertTrue(res_dec.ok)
        self.assertEqual(list(self.ds.df["country"]), list(self.ds.df["country_str"]))

    def test_16_label_management(self):
        res_def = self.interp.execute('label define yesno 0 "No" 1 "Yes"')
        self.assertTrue(res_def.ok)
        self.assertIn("yesno", self.ds.val_labels)

        res_var = self.interp.execute('label variable country "Origin Country"')
        self.assertTrue(res_var.ok)
        self.assertEqual(self.ds.var_labels["country"], "Origin Country")

        res_dir = self.interp.execute("label dir")
        self.assertIn("yesno", res_dir)

        res_list = self.interp.execute("label list yesno")
        self.assertIn("Yes", res_list)

        res_drop = self.interp.execute("label drop yesno")
        self.assertTrue(res_drop.ok)
        self.assertNotIn("yesno", self.ds.val_labels)


class TestRecodeAndDuplicates(unittest.TestCase):
    """Tests for recode and duplicates commands."""

    def setUp(self):
        self.ds = Dataset()
        self.interp = Interpreter(self.ds)
        self.ds.load_dataframe(pd.DataFrame({
            "age": [15, 25, 45, 65, 80],
            "id": [1, 1, 2, 3, 3]
        }))

    def test_17_recode_ranges(self):
        res = self.interp.execute("recode age (min/18 = 1) (19/64 = 2) (65/max = 3), gen(age_cat)")
        self.assertTrue(res.ok)
        self.assertEqual(list(self.ds.df["age_cat"]), [1.0, 2.0, 2.0, 3.0, 3.0])

    def test_18_duplicates_report_tag_and_drop(self):
        res_rep = self.interp.execute("duplicates report id")
        self.assertTrue(res_rep.ok)
        self.assertIn("Surplus", res_rep)

        res_tag = self.interp.execute("duplicates tag id, gen(is_dup)")
        self.assertTrue(res_tag.ok)
        # IDs 1 and 3 appear twice (1 surplus copy each), ID 2 appears once (0 surplus)
        self.assertEqual(list(self.ds.df["is_dup"]), [1, 1, 0, 1, 1])

        res_drop = self.interp.execute("duplicates drop id, force")
        self.assertTrue(res_drop.ok)
        self.assertEqual(len(self.ds.df), 3)
        self.assertEqual(list(self.ds.df["id"]), [1, 2, 3])

    def test_19_merge_m1_relationship(self):
        # Multiple master observations per country, 1 in using
        master_m = pd.DataFrame({"firm": [1, 2, 3, 4], "country": ["US", "US", "DE", "DE"]})
        self.ds.load_dataframe(master_m)
        macro_path = os.path.join(TEST_DIR, "macro_using.csv")
        pd.DataFrame({"country": ["US", "DE"], "gdp_growth": [2.1, 1.4]}).to_csv(macro_path, index=False)

        res = self.interp.execute(f'merge m:1 country using "{macro_path}"')
        if os.path.exists(macro_path):
            os.remove(macro_path)
        self.assertTrue(res.ok)
        self.assertIn("gdp_growth", self.ds.df.columns)
        self.assertEqual(len(self.ds.df), 4)
        self.assertTrue((self.ds.df["_merge"] == 3).all())

    def test_20_merge_nogenerate_and_custom_generate(self):
        master = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        self.ds.load_dataframe(master)
        using_p = os.path.join(TEST_DIR, "u_temp.csv")
        pd.DataFrame({"id": [2, 3], "bonus": [100, 200]}).to_csv(using_p, index=False)

        res_nogen = self.interp.execute(f'merge 1:1 id using "{using_p}", nogenerate')
        self.assertTrue(res_nogen.ok)
        self.assertNotIn("_merge", self.ds.df.columns)

        self.ds.load_dataframe(master)
        res_custom = self.interp.execute(f'merge 1:1 id using "{using_p}", generate(my_status)')
        if os.path.exists(using_p):
            os.remove(using_p)
        self.assertTrue(res_custom.ok)
        self.assertIn("my_status", self.ds.df.columns)
        self.assertNotIn("_merge", self.ds.df.columns)

    def test_21_merge_replace_option(self):
        master = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        self.ds.load_dataframe(master)
        using_p = os.path.join(TEST_DIR, "u_rep.csv")
        pd.DataFrame({"id": [2, 3], "val": [999, 888]}).to_csv(using_p, index=False)

        res_rep = self.interp.execute(f'merge 1:1 id using "{using_p}", replace')
        if os.path.exists(using_p):
            os.remove(using_p)
        self.assertTrue(res_rep.ok)
        # For id 2, val was replaced by 999 from using
        id2_val = self.ds.df.loc[self.ds.df["id"] == 2, "val"].iloc[0]
        self.assertEqual(id2_val, 999)

    def test_22_append_keep_columns(self):
        master = pd.DataFrame({"x": [1, 2], "y": ["A", "B"]})
        self.ds.load_dataframe(master)
        using_p = os.path.join(TEST_DIR, "u_app.csv")
        pd.DataFrame({"x": [3], "y": ["C"], "omit_col": [999]}).to_csv(using_p, index=False)

        res = self.interp.execute(f'append using "{using_p}", keep(x y)')
        if os.path.exists(using_p):
            os.remove(using_p)
        self.assertTrue(res.ok)
        self.assertNotIn("omit_col", self.ds.df.columns)
        self.assertEqual(len(self.ds.df), 3)

    def test_23_collapse_percentiles_min_max(self):
        df = pd.DataFrame({
            "group": ["A"] * 10,
            "score": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        self.ds.load_dataframe(df)
        res = self.interp.execute("collapse (min) score (p50) score (max) score, by(group)")
        self.assertTrue(res.ok)
        self.assertEqual(len(self.ds.df), 1)

    def test_24_egen_rowmean_rowmin_rowmax(self):
        df = pd.DataFrame({"a": [10, 20], "b": [30, 40], "c": [50, 60]})
        self.ds.load_dataframe(df)
        self.interp.execute("egen row_avg = rowmean(a b c)")
        self.interp.execute("egen row_lowest = rowmin(a b c)")
        self.interp.execute("egen row_highest = rowmax(a b c)")
        self.assertEqual(self.ds.df["row_avg"].iloc[0], 30.0)
        self.assertEqual(self.ds.df["row_lowest"].iloc[0], 10.0)
        self.assertEqual(self.ds.df["row_highest"].iloc[0], 50.0)

    def test_25_egen_cut_bins(self):
        df = pd.DataFrame({"val": [1, 5, 10, 15, 20, 25, 30, 35]})
        self.ds.load_dataframe(df)
        res = self.interp.execute("egen bucket = cut(val), group(4)")
        self.assertTrue(res.ok)
        self.assertIn("bucket", self.ds.df.columns)
        self.assertEqual(len(self.ds.df["bucket"].unique()), 4)

    def test_26_recode_single_values_and_else(self):
        df = pd.DataFrame({"status": [1, 2, 9, 99]})
        self.ds.load_dataframe(df)
        res = self.interp.execute("recode status (1 = 10) (2 = 20) (9 = 90), gen(new_status)")
        self.assertTrue(res.ok)
        self.assertEqual(list(self.ds.df["new_status"]), [10.0, 20.0, 90.0, 99.0])

    def test_27_duplicates_list_shows_rows(self):
        df = pd.DataFrame({"x": [1, 1, 2]})
        self.ds.load_dataframe(df)
        res = self.interp.execute("duplicates list x")
        self.assertTrue(res.ok)
        self.assertIn("1", res)

    def test_28_label_variable_update(self):
        df = pd.DataFrame({"x": [1, 2]})
        self.ds.load_dataframe(df)
        res = self.interp.execute('label variable x "My Variable"')
        self.assertTrue(res.ok)
        self.assertEqual(self.ds.var_labels["x"], "My Variable")

    def test_29_bysort_multiple_byvars(self):
        df = pd.DataFrame({
            "region": ["East", "East", "West", "West"],
            "year": [2021, 2022, 2021, 2022],
            "gdp": [100, 120, 200, 240]
        })
        self.ds.load_dataframe(df)
        res = self.interp.execute("bysort region year: generate row_n = _n")
        self.assertTrue(res.ok)
        self.assertEqual(list(self.ds.df["row_n"]), [1, 1, 1, 1])

    def test_30_data_ops_error_when_no_dataset(self):
        ds_empty = Dataset()
        interp_empty = Interpreter(ds_empty)
        res_merge = interp_empty.execute("merge 1:1 id using test.csv")
        self.assertFalse(res_merge.ok)
        self.assertEqual(res_merge.return_code, 111)

        res_collapse = interp_empty.execute("collapse (mean) x, by(g)")
        self.assertFalse(res_collapse.ok)
        self.assertEqual(res_collapse.return_code, 111)


if __name__ == "__main__":
    unittest.main()
