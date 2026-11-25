"""
tests/test_stage53_data_integrity.py

Comprehensive regression tests for Potatostics Stage 53:
- Priority 0.1: Correct exact numeric Find & Replace (no tolerance-based corruption of 100000.5)
- Priority 0.2: Preservation of missing values (None, np.nan, pd.NA) during string replacement
- Priority 0.3: Transactional replacement operations
- Priority 1.1: Strict integer-cell validation (rejecting fractional floats like '1.9', text, overflow)
- Priority 1.2: Strict boolean-cell validation (explicit tokens only; rejecting unknown tokens)
- Priority 1.3: Synchronized sorted and filtered views (re-evaluating filter/sort on relevant cell edits)
- Priority 1.4: Multi-cell paste validation and transactional rollback
"""

import os
import unittest
import numpy as np
import pandas as pd

from PySide6 import QtWidgets
from PySide6.QtCore import Qt

# Ensure headless Qt
app = QtWidgets.QApplication.instance()
if app is None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["POTATOSTICS_TESTING"] = "1"
    app = QtWidgets.QApplication([])

from engine.dataset import Dataset
from gui_qt.data_grid import VirtualPandasModel, DataGridWidget, perform_find_replace


class TestStage53DataIntegrity(unittest.TestCase):
    def setUp(self):
        self.ds = Dataset()

    def test_exact_integer_search_and_replacement_preserve_decimal_precision(self):
        df = pd.DataFrame({"x": [1, 2]})
        for find, replacement in [
            ("1.00000000000000001", "9"),
            ("1", "9.00000000000000001"),
        ]:
            with self.subTest(find=find, replacement=replacement):
                self.ds.load_dataframe(df.copy())
                self.assertEqual(perform_find_replace(self.ds, find, replacement, exact_match=True), 0)
                pd.testing.assert_frame_equal(self.ds.df, df)

        self.ds.load_dataframe(df.copy())
        large_integer = 9007199254740993
        self.assertEqual(perform_find_replace(self.ds, "1.0", f"{large_integer}.0", exact_match=True), 1)
        self.assertEqual(self.ds.df["x"].tolist(), [large_integer, 2])

    def test_exact_float_search_validates_query_in_storage_dtype(self):
        for dtype in ("float32", "Float32"):
            with self.subTest(dtype=dtype):
                df = pd.DataFrame({"x": pd.Series([0.1, 0.2, None], dtype=dtype)})
                self.ds.load_dataframe(df.copy())
                self.assertEqual(perform_find_replace(self.ds, "0.1", "0.3", exact_match=True), 1)
                df.iloc[0, 0] = np.float32(0.3)
                pd.testing.assert_frame_equal(self.ds.df, df)

        df = pd.DataFrame({"x": [0.0, np.inf, 9007199254740992.0]})
        for find in ("1e-999", "1e999", "9007199254740993", "9007199254740993.0"):
            with self.subTest(find=find):
                self.ds.load_dataframe(df.copy())
                self.assertEqual(perform_find_replace(self.ds, find, "9", exact_match=True), 0)
                pd.testing.assert_frame_equal(self.ds.df, df)

        self.ds.load_dataframe(df.copy())
        self.assertEqual(perform_find_replace(self.ds, "inf", "9", exact_match=True), 1)
        self.assertEqual(self.ds.df["x"].tolist(), [0.0, 9.0, 9007199254740992.0])

    # =========================================================================
    # Priority 0.1: Exact Numeric Find & Replace
    # =========================================================================

    def test_01_exact_numeric_find_replace_does_not_corrupt_nearby_numbers(self):
        """
        Verifies exact numeric matching does NOT modify nearby numbers within float tolerance.
        Replacing 100000 with 50 must change 100000.0 but leave 100000.5 and 100001.0 intact.
        """
        df = pd.DataFrame({
            "val": [100000.0, 100000.5, 100001.0, -10.0, 0.0, np.nan]
        })
        self.ds.load_dataframe(df)

        # Replace 100000 with 50
        count1 = perform_find_replace(self.ds, find_text="100000", replace_text="50")
        self.assertEqual(count1, 1, "Only exact match 100000.0 should be replaced")
        self.assertEqual(self.ds.df["val"].iloc[0], 50.0)
        self.assertEqual(self.ds.df["val"].iloc[1], 100000.5, "100000.5 must NOT be changed!")
        self.assertEqual(self.ds.df["val"].iloc[2], 100001.0, "100001.0 must NOT be changed!")
        self.assertEqual(self.ds.df["val"].iloc[3], -10.0)
        self.assertEqual(self.ds.df["val"].iloc[4], 0.0)
        self.assertTrue(np.isnan(self.ds.df["val"].iloc[5]))

        # Replace -10 with -5
        count2 = perform_find_replace(self.ds, find_text="-10", replace_text="-5")
        self.assertEqual(count2, 1)
        self.assertEqual(self.ds.df["val"].iloc[3], -5.0)

        # Replace 0 with 7
        count3 = perform_find_replace(self.ds, find_text="0", replace_text="7")
        self.assertEqual(count3, 1)
        self.assertEqual(self.ds.df["val"].iloc[4], 7.0)

        # Replace . (missing) with 99
        count4 = perform_find_replace(self.ds, find_text=".", replace_text="99")
        self.assertEqual(count4, 1)
        self.assertEqual(self.ds.df["val"].iloc[5], 99.0)

    def test_02_numeric_replacement_nullable_and_integer_dtypes(self):
        """Verifies integer, nullable integer (Int64), large numbers and nonfinites."""
        df = pd.DataFrame({
            "int_col": pd.Series([10, 20, 30, 10], dtype="int64"),
            "nullable_int": pd.Series([100, pd.NA, 200, 100], dtype="Int64"),
            "large_num": [1e18, 1e18 + 1, 1e20, -1e18],
            "nonfinites": [np.inf, -np.inf, np.nan, 0.0]
        })
        self.ds.load_dataframe(df)

        # In integer column, replace 10 with 99
        c_int = perform_find_replace(self.ds, find_text="10", replace_text="99")
        self.assertEqual(c_int, 2)
        self.assertEqual(list(self.ds.df["int_col"]), [99, 20, 30, 99])
        self.assertEqual(self.ds.df["int_col"].dtype, np.int64, "Integer dtype must be preserved")

        # In nullable integer, replace 100 with 500, preserving NA
        c_null = perform_find_replace(self.ds, find_text="100", replace_text="500")
        self.assertEqual(c_null, 2)
        self.assertEqual(self.ds.df["nullable_int"].iloc[0], 500)
        self.assertTrue(pd.isna(self.ds.df["nullable_int"].iloc[1]))
        self.assertEqual(self.ds.df["nullable_int"].dtype, "Int64")

        # Nonfinite inf replacement
        c_inf = perform_find_replace(self.ds, find_text="inf", replace_text="1234.5")
        self.assertEqual(c_inf, 1)
        self.assertEqual(self.ds.df["nonfinites"].iloc[0], 1234.5)
        self.assertEqual(self.ds.df["nonfinites"].iloc[1], -np.inf)
        self.assertTrue(np.isnan(self.ds.df["nonfinites"].iloc[2]))

    # =========================================================================
    # Priority 0.2: Preservation of Missing Values During String Replacement
    # =========================================================================

    def test_03_string_replacement_strictly_preserves_missing_values(self):
        """
        Verifies that None, np.nan, and pd.NA are NEVER converted to literal 'None', 'nan', '<NA>'
        during string Find & Replace.
        """
        df = pd.DataFrame({
            "obj_col": ["apple", None, "orange", np.nan, "apple pie"],
            "str_col": pd.Series(["apple", pd.NA, "orange", "apple", "apple pie"], dtype="string")
        })
        self.ds.load_dataframe(df)

        # Initial missing mask verification
        obj_na_before = self.ds.df["obj_col"].isna().copy()
        str_na_before = self.ds.df["str_col"].isna().copy()

        # Find 'apple', replace with 'pear' (substring / regex replacement)
        count = perform_find_replace(self.ds, find_text="apple", replace_text="pear", exact_match=False)
        self.assertGreater(count, 0)

        # Verify missing masks were NOT corrupted into literal strings
        self.assertTrue((self.ds.df["obj_col"].isna() == obj_na_before).all(), "obj_col missing values must remain missing!")
        self.assertTrue((self.ds.df["str_col"].isna() == str_na_before).all(), "str_col missing values must remain missing!")

        # Verify literal string representations
        self.assertEqual(self.ds.df["obj_col"].iloc[0], "pear")
        self.assertTrue(pd.isna(self.ds.df["obj_col"].iloc[1]))
        self.assertEqual(self.ds.df["obj_col"].iloc[2], "orange")
        self.assertTrue(np.isnan(self.ds.df["obj_col"].iloc[3]))
        self.assertEqual(self.ds.df["obj_col"].iloc[4], "pear pie")

        self.assertEqual(self.ds.df["str_col"].iloc[0], "pear")
        self.assertTrue(pd.isna(self.ds.df["str_col"].iloc[1]))

    def test_04_categorical_and_unicode_and_regex_metacharacters(self):
        """Verifies categorical columns, Unicode/Arabic text, regex metacharacters, and backslashes."""
        df = pd.DataFrame({
            "cat_col": pd.Series(pd.Categorical(["apple", None, "orange", "apple"])),
            "arabic_col": ["مرحبا بالعالم", None, "بيانات احصائية", "مرحبا"],
            "metachar_col": ["price ($100)", "formula [a+b]", None, "C:\\Program Files\\Potatostics"]
        })
        self.ds.load_dataframe(df)

        # 1. Categorical replacement preserves category dtype and missing values
        c_cat = perform_find_replace(self.ds, find_text="apple", replace_text="banana", exact_match=True)
        self.assertEqual(c_cat, 2)
        self.assertEqual(self.ds.df["cat_col"].iloc[0], "banana")
        self.assertTrue(pd.isna(self.ds.df["cat_col"].iloc[1]))
        self.assertEqual(self.ds.df["cat_col"].iloc[2], "orange")
        self.assertEqual(self.ds.df["cat_col"].iloc[3], "banana")
        self.assertTrue(isinstance(self.ds.df["cat_col"].dtype, pd.CategoricalDtype))

        # 2. Unicode and Arabic replacement
        c_ar = perform_find_replace(self.ds, find_text="مرحبا", replace_text="أهلاً", exact_match=False)
        self.assertEqual(c_ar, 2)
        self.assertIn("أهلاً بالعالم", self.ds.df["arabic_col"].iloc[0])
        self.assertTrue(pd.isna(self.ds.df["arabic_col"].iloc[1]))
        self.assertEqual(self.ds.df["arabic_col"].iloc[3], "أهلاً")

        # 3. Metacharacters and backslashes (must not be corrupted by re.sub escape handling)
        c_meta = perform_find_replace(self.ds, find_text="($100)", replace_text="($250)", exact_match=False)
        self.assertEqual(c_meta, 1)
        self.assertEqual(self.ds.df["metachar_col"].iloc[0], "price ($250)")

        c_slash = perform_find_replace(self.ds, find_text="C:\\Program Files", replace_text="D:\\Analysis\\Custom", exact_match=False)
        self.assertEqual(c_slash, 1)
        self.assertEqual(self.ds.df["metachar_col"].iloc[3], "D:\\Analysis\\Custom\\Potatostics")

    # =========================================================================
    # Priority 0.3: Transactional Replacement
    # =========================================================================

    def test_05_transactional_failure_leaves_dataset_untouched(self):
        """Verifies that an error in replacement leaves canonical dataset 100% untouched."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        self.ds.load_dataframe(df)
        df_orig = self.ds.df.copy()

        # Attempt find_replace with an invalid condition or type
        # Text in numeric column must be rejected without mutating column
        c = perform_find_replace(self.ds, find_text="2.0", replace_text="invalid_text_str")
        self.assertEqual(c, 0, "Incompatible type replacement must be rejected")
        pd.testing.assert_frame_equal(self.ds.df, df_orig)

    # =========================================================================
    # Priority 1.1: Strict Integer-Cell Validation
    # =========================================================================

    def test_06_integer_cell_validation_rejects_fractions_and_text(self):
        """Verifies setData rejects '1.9' without silent truncation, rejects text, and bounds."""
        df = pd.DataFrame({
            "age": pd.Series([25, 30, 35], dtype="int32")
        })
        self.ds.load_dataframe(df)
        model = VirtualPandasModel(self.ds)

        idx0 = model.index(0, 0)

        # 1. Reject fractional string '1.9' (MUST NOT silently truncate to 1!)
        res_frac = model.setData(idx0, "1.9", Qt.ItemDataRole.EditRole)
        self.assertFalse(res_frac, "setData must return False for fractional float in integer column")
        self.assertEqual(self.ds.df["age"].iloc[0], 25, "Original cell value must remain unchanged")

        # 2. Reject text
        res_text = model.setData(idx0, "thirty", Qt.ItemDataRole.EditRole)
        self.assertFalse(res_text)
        self.assertEqual(self.ds.df["age"].iloc[0], 25)

        # 3. Reject integer overflow for int32
        res_overflow = model.setData(idx0, "999999999999999999999", Qt.ItemDataRole.EditRole)
        self.assertFalse(res_overflow)
        self.assertEqual(self.ds.df["age"].iloc[0], 25)

        # 4. Accept valid integer
        res_valid = model.setData(idx0, "42", Qt.ItemDataRole.EditRole)
        self.assertTrue(res_valid)
        self.assertEqual(self.ds.df["age"].iloc[0], 42)

    # =========================================================================
    # Priority 1.2: Strict Boolean-Cell Validation
    # =========================================================================

    def test_07_boolean_cell_validation_rejects_unknown_tokens(self):
        """Verifies boolean cells reject arbitrary tokens like 'maybe' and accept explicit true/false."""
        df = pd.DataFrame({
            "flag": [True, False, True]
        })
        self.ds.load_dataframe(df)
        model = VirtualPandasModel(self.ds)

        idx = model.index(0, 0)

        # 1. Reject unknown tokens
        self.assertFalse(model.setData(idx, "maybe", Qt.ItemDataRole.EditRole))
        self.assertEqual(self.ds.df["flag"].iloc[0], True)

        self.assertFalse(model.setData(idx, "unknown", Qt.ItemDataRole.EditRole))
        self.assertEqual(self.ds.df["flag"].iloc[0], True)

        # 2. Accept explicit False tokens
        self.assertTrue(model.setData(idx, "0", Qt.ItemDataRole.EditRole))
        self.assertEqual(self.ds.df["flag"].iloc[0], False)

        self.assertTrue(model.setData(idx, "false", Qt.ItemDataRole.EditRole))
        self.assertEqual(self.ds.df["flag"].iloc[0], False)

        # 3. Accept explicit True tokens
        self.assertTrue(model.setData(idx, "yes", Qt.ItemDataRole.EditRole))
        self.assertEqual(self.ds.df["flag"].iloc[0], True)

    # =========================================================================
    # Priority 1.3: Synchronized Sorted and Filtered Views
    # =========================================================================

    def test_08_sorted_and_filtered_views_remain_synchronized(self):
        """
        Verifies editing a cell that participates in an active filter or sort
        correctly updates visible row mappings and preserves canonical observation order.
        """
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David"],
            "score": [80, 95, 70, 85]
        }, index=[10, 20, 10, 20])  # Duplicate pandas index labels to verify positional iloc safety!
        self.ds.load_dataframe(df)
        model = VirtualPandasModel(self.ds)

        # 1. Apply filter 'david'
        model.set_filter("david")
        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(model.data(model.index(0, 0)), "David")

        # Edit visible cell to 'Daniel' -> leaves filter
        model.setData(model.index(0, 0), "Daniel")
        self.assertEqual(self.ds.df["name"].iloc[3], "Daniel", "Positional update must hit canonical row 3 despite duplicate index")
        self.assertEqual(model.rowCount(), 0, "Row must leave active filter 'david'")

        # Clear filter
        model.set_filter("")
        self.assertEqual(model.rowCount(), 4)

        # 2. Sort by score ascending (col 1)
        model.sort_by_column(1)
        # Order should be: Charlie (70), Alice (80), Daniel (85), Bob (95)
        self.assertEqual(model.data(model.index(0, 0)), "Charlie")
        self.assertEqual(model.data(model.index(3, 0)), "Bob")

        # Edit Charlie's score (visible row 0) from 70 to 100 -> Charlie should move to the bottom!
        model.setData(model.index(0, 1), "100")
        self.assertEqual(self.ds.df["score"].iloc[2], 100, "Canonical Charlie score must be 100")
        # New sorted order: Alice (80), Daniel (85), Bob (95), Charlie (100)
        self.assertEqual(model.data(model.index(0, 0)), "Alice")
        self.assertEqual(model.data(model.index(3, 0)), "Charlie")

    # =========================================================================
    # Priority 1.4: Multi-Cell Paste Reliability
    # =========================================================================

    def test_09_multicell_paste_transactional_validation(self):
        """Verifies multi-cell paste applies valid blocks and aborts invalid blocks without corruption."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [10.0, 20.0, 30.0]
        })
        self.ds.load_dataframe(df)
        widget = DataGridWidget(self.ds)

        # 1. Valid rectangular paste
        valid_text = "100\t50.5\n200\t60.5"
        QtWidgets.QApplication.clipboard().setText(valid_text)
        pasted = widget.paste_clipboard(start_row=0, start_col=0)
        self.assertEqual(pasted, 4)
        self.assertEqual(self.ds.df["a"].iloc[0], 100)
        self.assertEqual(self.ds.df["b"].iloc[0], 50.5)
        self.assertEqual(self.ds.df["a"].iloc[1], 200)
        self.assertEqual(self.ds.df["b"].iloc[1], 60.5)

        # 2. Partially invalid paste (row 2 col a is '1.9' which is invalid integer)
        df_before = self.ds.df.copy()
        invalid_text = "300\t70.5\n1.9\t80.5"  # 1.9 invalid for int column
        QtWidgets.QApplication.clipboard().setText(invalid_text)
        pasted_invalid = widget.paste_clipboard(start_row=1, start_col=0)
        self.assertEqual(pasted_invalid, 0, "Paste with invalid integer cell must be completely aborted")
        pd.testing.assert_frame_equal(self.ds.df, df_before, "Dataset must remain completely untouched on paste failure")

    # =========================================================================
    # Part 1: Large-Integer Precision Boundary Tests
    # =========================================================================

    def test_10_large_integer_precision_boundary(self):
        """
        Verifies exact integer parsing and matching without 64-bit float mantissa rounding
        for numbers at and beyond 2^53 (e.g. 2^53-1, 2^53, 2^53+1, int64.max, uint64.max).
        """
        val_2_53_minus_1 = 9007199254740991
        val_2_53 = 9007199254740992
        val_2_53_plus_1 = 9007199254740993
        val_max_int64 = int(np.iinfo(np.int64).max)
        val_max_uint64 = int(np.iinfo(np.uint64).max)

        df = pd.DataFrame({
            "large_int": [val_2_53, val_2_53_plus_1, val_max_int64],
            "nullable_large": pd.Series([val_2_53_plus_1, val_2_53, pd.NA], dtype="Int64"),
            "u64": pd.Series([val_max_uint64, 42, 100], dtype="uint64"),
        })
        self.ds.load_dataframe(df)

        # 1. Searching for 2^53+1 (9007199254740993) must NOT match 2^53 (9007199254740992)
        # In float64, float(9007199254740993) == 9007199254740992.0!
        c = perform_find_replace(self.ds, find_text=str(val_2_53_plus_1), replace_text=str(val_2_53_minus_1))
        # Matches row 1 in 'large_int' and row 0 in 'nullable_large' -> exactly 2 cells
        self.assertEqual(c, 2)
        # Verify row 0 in 'large_int' (2^53) was completely untouched!
        self.assertEqual(int(self.ds.df["large_int"].iloc[0]), val_2_53)
        # Verify row 1 in 'large_int' was updated to 2^53-1
        self.assertEqual(int(self.ds.df["large_int"].iloc[1]), val_2_53_minus_1)
        # Verify nullable row 0 was updated to 2^53-1 and row 1 remained 2^53
        self.assertEqual(int(self.ds.df["nullable_large"].iloc[0]), val_2_53_minus_1)
        self.assertEqual(int(self.ds.df["nullable_large"].iloc[1]), val_2_53)
        self.assertTrue(pd.isna(self.ds.df["nullable_large"].iloc[2]))

        # 2. Replacing uint64.max with 9999
        c_u64 = perform_find_replace(self.ds, find_text=str(val_max_uint64), replace_text="9999")
        self.assertEqual(c_u64, 1)
        self.assertEqual(int(self.ds.df["u64"].iloc[0]), 9999)
        self.assertEqual(int(self.ds.df["u64"].iloc[1]), 42)

    # =========================================================================
    # Part 2: Duplicate-Index Positional Safety
    # =========================================================================

    def test_11_duplicate_index_replacement_safety(self):
        """
        Verifies replacement in DataFrames with non-unique index labels uses positional
        indexing (.iloc / positional masks) and never corrupts non-matching rows sharing an index label.
        """
        df = pd.DataFrame({
            "fruit": ["apple", "banana", "cherry", "apple tart"],
            "score": [10, 20, 30, 40]
        }, index=[5, 5, 6, 6])  # Duplicate index labels!
        self.ds.load_dataframe(df)

        # Exact match replace 'apple' -> 'orange'
        c = perform_find_replace(self.ds, find_text="apple", replace_text="orange", exact_match=True)
        self.assertEqual(c, 1)
        # Row 0 (index 5) is 'orange'
        self.assertEqual(self.ds.df["fruit"].iloc[0], "orange")
        # Row 1 (index 5) MUST STILL BE 'banana', NOT corrupted into 'orange'!
        self.assertEqual(self.ds.df["fruit"].iloc[1], "banana")
        # Row 2 (index 6) is 'cherry'
        self.assertEqual(self.ds.df["fruit"].iloc[2], "cherry")
        # Row 3 (index 6) is 'apple tart' (not exact match)
        self.assertEqual(self.ds.df["fruit"].iloc[3], "apple tart")

    # =========================================================================
    # Part 3: Preserving Precision When Setting Missing in Large-Integer Columns
    # =========================================================================

    def test_12_editing_missing_in_large_integer_preserves_precision(self):
        """
        Verifies that clearing or setting missing values in a standard integer column
        converts the column to pandas nullable integer (Int64), preserving exact large-integer precision.
        """
        val_large = 9007199254740993
        df = pd.DataFrame({
            "id": [val_large, 100]
        })
        self.ds.load_dataframe(df)
        model = VirtualPandasModel(self.ds)

        # Clear row 1 (index (1, 0)) by entering '.'
        res = model.setData(model.index(1, 0), ".")
        self.assertTrue(res)
        # Row 0 must retain exact integer 9007199254740993
        self.assertEqual(int(self.ds.df["id"].iloc[0]), val_large)
        self.assertEqual(self.ds.df["id"].iloc[0], val_large)
        # Row 1 must be pd.NA
        self.assertTrue(pd.isna(self.ds.df["id"].iloc[1]))
        # Column dtype must be Int64, NOT float64
        self.assertEqual(str(self.ds.df["id"].dtype), "Int64")

    # =========================================================================
    # Part 4: Genuinely Transactional Multi-Column Rollback
    # =========================================================================

    def test_13_multi_column_transactional_rollback(self):
        """
        Verifies that if a find token matches across mixed-type columns, but the replacement
        cannot be converted to the required type in ANY matched column, the ENTIRE operation
        is aborted and no column is modified.
        """
        df = pd.DataFrame({
            "num_col": [1, 2, 3],
            "str_col": ["alpha", "2", "beta"]
        })
        self.ds.load_dataframe(df)
        df_orig = df.copy()

        # '2' matches in both num_col (row 1) and str_col (row 1).
        # 'invalid_num' cannot be converted to integer for num_col.
        # Entire operation must be rejected with 0 replacements and 0 dataset modifications!
        c = perform_find_replace(self.ds, find_text="2", replace_text="invalid_num")
        self.assertEqual(c, 0, "Mixed-column incompatible replacement must be rejected")
        pd.testing.assert_frame_equal(self.ds.df, df_orig, "Dataset must remain completely untouched")

    # =========================================================================
    # Part 5: Accurate Replacement Counts (No-Op Returns 0)
    # =========================================================================

    def test_14_no_op_replacement_count(self):
        """
        Verifies that replacing a value with itself (no-op) returns 0 changed cells,
        distinguishing matched cells from genuinely changed cells.
        """
        df = pd.DataFrame({
            "str_col": ["apple", "banana", "apple"],
            "int_col": [10, 20, 10],
            "flt_col": [1.5, 2.5, 1.5]
        })
        self.ds.load_dataframe(df)
        df_orig = df.copy()

        # String replace with itself
        c_str = perform_find_replace(self.ds, find_text="apple", replace_text="apple", match_case=True)
        self.assertEqual(c_str, 0, "No-op string replacement must return 0 changed cells")

        # Integer replace with itself
        c_int = perform_find_replace(self.ds, find_text="10", replace_text="10")
        self.assertEqual(c_int, 0, "No-op integer replacement must return 0 changed cells")

        # Float replace with itself
        c_flt = perform_find_replace(self.ds, find_text="1.5", replace_text="1.5")
        self.assertEqual(c_flt, 0, "No-op float replacement must return 0 changed cells")

        # Missing replace with missing
        c_na = perform_find_replace(self.ds, find_text=".", replace_text=".")
        self.assertEqual(c_na, 0, "No-op missing replacement must return 0 changed cells")

        pd.testing.assert_frame_equal(self.ds.df, df_orig)


if __name__ == "__main__":
    unittest.main()
