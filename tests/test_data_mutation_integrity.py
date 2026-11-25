"""
tests/test_data_mutation_integrity.py

Consolidated Data Mutation Integrity & Product Reliability Test Suite.
Verifies all core data mutation invariants across Potatostics:
- Invariant 1: Blank-row insertion preserves all dtypes, large-integer precision (>2^53), metadata, and row ordering.
- Invariant 2: Authoritative read-only enforcement blocks all GUI mutation surfaces (setData, paste, add_row, add_var, replace_all).
- Invariant 3: Atomic commit with rollback in Find & Replace when an unexpected failure occurs during publication.
- Invariant 4: Mixed object-column preservation: non-string objects are not silently coerced during text replacement without explicit option.
- Invariant 5: Strict numeric precision boundaries (2^53-1, 2^53, 2^53+1, int64.min/max, uint64.max).
- Invariant 6: Positional safety on duplicate and non-monotonic DataFrame indices.
- Invariant 7: Missing value preservation and ExtensionDtype conversions.
- Invariant 8: Synchronized sorted and filtered views.
- Invariant 9: Transactional multi-cell clipboard paste with rollback.
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd
from PySide6 import QtWidgets

from engine.dataset import Dataset
from gui_qt.data_grid import (
    DataGridWidget,
    VirtualPandasModel,
    perform_find_replace,
    insert_blank_observation,
)

# Ensure offscreen Qt application
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(["--platform", "offscreen"])


class TestDataMutationIntegrity(unittest.TestCase):
    def setUp(self):
        self.ds = Dataset()

    def test_csv_preserves_nullable_wide_integer_literals(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "identifiers.csv"
            path.write_text(
                "signed,unsigned,minimum,beyond,mixed,wage\n"
                f"{2**53 + 1},{2**64 - 1},{-2**63},{2**65},{2**63 + 1},10.25\n"
                ",,,,,\n"
                f"{2**53 + 3},2,1,1,-1,0.1\n",
                encoding="utf-8",
            )
            self.ds.load_csv(path)
        frame = self.ds.df
        for name, expected in (("signed", 2**53 + 1), ("unsigned", 2**64 - 1), ("minimum", -2**63)):
            self.assertTrue(pd.api.types.is_integer_dtype(frame[name]))
            self.assertEqual(int(frame[name].iloc[0]), expected)
        self.assertEqual(int(frame.signed.iloc[2]), 2**53 + 3)
        self.assertEqual(frame.beyond.iloc[0], str(2**65))
        self.assertEqual(frame.mixed.iloc[0], str(2**63 + 1))
        self.assertTrue(frame.iloc[1].isna().all())
        self.assertEqual(frame.wage.iloc[[0, 2]].tolist(), [10.25, 0.1])

    def test_dta_rejects_inexact_signed_and_unsigned_without_overwrite(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "identifiers.dta"
            self.ds.load_dataframe(pd.DataFrame({"original": [1]}))
            self.ds.save_dta(path)
            previous = path.read_bytes()
            for dtype, value in (("Int64", 2**60 + 1), ("Int64", -2**60 - 1), ("UInt64", 2**64 - 1)):
                frame = pd.DataFrame({"id": pd.array([str(value), None], dtype=dtype)})
                self.ds.load_dataframe(frame)
                with self.assertRaisesRegex(ValueError, "column 'id'"):
                    self.ds.save_dta(path)
                self.assertEqual(path.read_bytes(), previous)
                self.assertIsNone(self.ds.filename)
                pd.testing.assert_frame_equal(self.ds.df, frame)

    def test_append_csv_preserves_using_ids_with_missing_values(self):
        from engine.interpreter import Interpreter

        self.ds.load_dataframe(pd.DataFrame({"id": [1], "text": ["original"]}))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "more.csv"
            path.write_text(f"id,text\n{2**53 + 1},new\n,blank\n", encoding="utf-8")
            result = Interpreter(self.ds).execute(f'append using "{path}"')
        self.assertTrue(result.ok, str(result))
        self.assertEqual(int(self.ds.df.id.iloc[1]), 2**53 + 1)
        self.assertTrue(pd.isna(self.ds.df.id.iloc[2]))

    def test_csv_backed_project_preserves_signed_and_unsigned_ids(self):
        from unittest.mock import patch
        from engine.interpreter import Interpreter
        from engine.session_manager import load_session_bundle, save_session_bundle

        frame = pd.DataFrame({
            "id": pd.array([str(2**53 + 1), None], dtype="Int64"),
            "uid": np.array([2**64 - 1, 1], dtype="uint64"),
        })
        self.ds.load_dataframe(frame)
        interp = Interpreter(self.ds)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "analysis.potproj"
            with patch("engine.session_manager.HAVE_PYARROW", False):
                save_session_bundle(path, interp)
            self.ds.clear()
            load_session_bundle(path, interp)
        pd.testing.assert_frame_equal(self.ds.df, frame)

    def test_dta_preserves_representable_nullable_large_integers(self):
        frame = pd.DataFrame({
            "signed": pd.array([str(2**60), None, str(-2**63)], dtype="Int64"),
            "unsigned": pd.array([str(2**63), None, str(2**53 + 2)], dtype="UInt64"),
        })
        self.ds.load_dataframe(frame)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "identifiers.dta"
            self.ds.save_dta(path)
            pd.testing.assert_frame_equal(self.ds.df, frame)
            self.ds.load_dta(path)
        for name in frame:
            self.assertTrue(self.ds.df[name].isna().equals(frame[name].isna()))
            self.assertEqual([int(v) for v in self.ds.df[name].dropna()], [int(v) for v in frame[name].dropna()])

    def test_dta_partial_writer_failure_keeps_original_file(self):
        from unittest.mock import patch

        def fail_after_open(frame, path, **kwargs):
            Path(path).write_bytes(b"partial")
            raise OSError("disk full")

        self.ds.load_dataframe(pd.DataFrame({"id": [42]}))
        from engine.dataset import _fmt_name
        with TemporaryDirectory() as folder:
            path = Path(folder) / "original.dta"
            path.write_bytes(b"existing")
            with patch.object(pd.DataFrame, "to_" + _fmt_name, fail_after_open), self.assertRaisesRegex(OSError, "disk full"):
                self.ds.save_dta(path)
            self.assertEqual(path.read_bytes(), b"existing")
            self.assertEqual(list(Path(folder).iterdir()), [path])
            self.assertIsNone(self.ds.filename)

    def test_set_obs_keeps_existing_integer_ids_exact(self):
        from engine.interpreter import Interpreter

        original = pd.DataFrame({
            "id": np.array([2**53 + 1, 2**53 + 3], dtype="int64"),
            "uid": np.array([2**64 - 1, 2**64 - 2], dtype="uint64"),
            "flag": np.array([True, False]),
            "rate": np.array([0.25, 1.5], dtype="float32"),
        })
        self.ds.load_dataframe(original)
        interp = Interpreter(self.ds)
        self.assertTrue(interp.execute("set obs 5").ok)
        self.assertEqual(self.ds.N, 5)
        for col in original:
            self.assertEqual(self.ds.df[col].iloc[:2].tolist(), original[col].tolist())
            self.assertTrue(self.ds.df[col].iloc[2:].isna().all())
        self.assertEqual(self.ds.df.dtypes.astype(str).tolist(), ["Int64", "UInt64", "boolean", "float32"])

    def test_failed_project_overwrite_keeps_previous_session(self):
        import pyarrow
        from engine.interpreter import Interpreter
        from engine.session_manager import load_session_bundle, save_session_bundle

        interp = Interpreter(self.ds)
        self.ds.load_dataframe(pd.DataFrame({"id": [2**53 + 1, 2**53 + 3]}))
        original = self.ds.df.copy()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "analysis.potproj"
            save_session_bundle(path, interp)
            good_archive = path.read_bytes()
            other = Path(folder) / "other.csv"
            other.write_text("id\ntext-id\n", encoding="utf-8")
            self.assertTrue(interp.execute(f'append using "{other}"').ok)
            with self.assertRaises((pyarrow.ArrowInvalid, pyarrow.ArrowTypeError)):
                save_session_bundle(path, interp)
            self.assertEqual(path.read_bytes(), good_archive)
            load_session_bundle(path, interp)
        pd.testing.assert_frame_equal(self.ds.df, original)

    def test_opening_empty_project_replaces_data_macros_and_registers(self):
        from engine.interpreter import Interpreter
        from engine.session_manager import save_session_bundle, load_session_bundle

        interp = Interpreter(self.ds)
        interp.macro_mgr.set_global("saved", "one")
        interp.macro_mgr.set_local("saved_local", "two")
        interp.registers.set_r_scalar("saved_result", 7)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "empty.potproj"
            save_session_bundle(path, interp)
            self.ds.load_dataframe(pd.DataFrame({"previous": [1, 2]}))
            interp.macro_mgr.set_global("previous", "stale")
            interp.macro_mgr.set_local("previous_local", "stale")
            interp.registers.set_e_scalar("previous_result", 123)
            load_session_bundle(path, interp)
        self.assertTrue(self.ds.is_empty)
        self.assertEqual(interp.macro_mgr.globals, {"saved": "one"})
        self.assertEqual(interp.macro_mgr.locals, {"saved_local": "two"})
        self.assertIsNone(interp.registers.get_e("previous_result"))
        self.assertEqual(interp.registers.get_r("saved_result"), 7)

    def test_project_roundtrip_preserves_empty_shapes(self):
        from engine.interpreter import Interpreter
        from engine.session_manager import load_session_bundle, save_session_bundle

        interp = Interpreter(self.ds)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "empty.potproj"
            for frame in (pd.DataFrame({"id": pd.Series([], dtype="Int64")}),
                          pd.DataFrame(index=pd.RangeIndex(3))):
                with self.subTest(shape=frame.shape):
                    self.ds.load_dataframe(frame)
                    save_session_bundle(path, interp)
                    self.ds.load_dataframe(pd.DataFrame({"previous": [5]}))
                    load_session_bundle(path, interp)
                    pd.testing.assert_frame_equal(self.ds.df, frame)

    def test_project_restores_panel_and_time_declarations(self):
        import json
        from zipfile import ZipFile
        from engine.interpreter import Interpreter
        from engine.session_manager import load_session_bundle, save_session_bundle

        source = Interpreter(self.ds)
        self.ds.load_dataframe(pd.DataFrame({"id": [1, 1, 2], "year": [1, 2, 1]}))
        self.assertTrue(source.execute("xtset id year").ok)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "panel.potproj"
            save_session_bundle(path, source)
            target = Interpreter(Dataset())
            load_session_bundle(path, target)
            self.assertTrue(target.execute("xtset").ok)
            self.assertEqual((target.panelvar, target.timevar), ("id", "year"))
            legacy = Path(folder) / "legacy.potproj"
            with ZipFile(path) as current, ZipFile(legacy, "w") as previous:
                for member in current.namelist():
                    payload = current.read(member)
                    if member == "manifest.json":
                        manifest = json.loads(payload)
                        del manifest["panelvar"], manifest["timevar"]
                        payload = json.dumps(manifest)
                    previous.writestr(member, payload)
            load_session_bundle(legacy, target)
            self.assertEqual((target.panelvar, target.timevar), (None, None))
            self.assertFalse(target.execute("xtset").ok)

    # =========================================================================
    # Invariant 1: Blank-Row Insertion Dtype & Precision Preservation
    # =========================================================================

    def test_01_blank_observation_insertion_comprehensive_dtypes(self):
        """
        Verifies that adding a blank observation preserves all dtypes, including
        large integers (> 2^53), signed/unsigned integer variants, floats, booleans,
        strings, categoricals, datetimes, timedeltas, and non-unique index labels.
        """
        val_2_53 = 9007199254740992
        val_2_53_p1 = 9007199254740993
        max_i64 = int(np.iinfo(np.int64).max)
        min_i64 = int(np.iinfo(np.int64).min)
        max_u64 = int(np.iinfo(np.uint64).max)

        df = pd.DataFrame({
            "i64_large": [val_2_53, val_2_53_p1],
            "i64_bounds": [min_i64, max_i64],
            "u64": [max_u64, 42],
            "i32": [100, 200],
            "i16": [10, 20],
            "i8": [1, 2],
            "u32": [1000, 2000],
            "u16": [100, 200],
            "u8": [10, 20],
            "null_i64": pd.Series([val_2_53_p1, pd.NA], dtype="Int64"),
            "null_u64": pd.Series([max_u64, pd.NA], dtype="UInt64"),
            "f32": [1.5, 2.5],
            "f64": [1e15, 2e15],
            "b_std": [True, False],
            "b_null": pd.Series([True, pd.NA], dtype="boolean"),
            "obj": ["alpha", 123],
            "str_null": pd.Series(["hello", pd.NA], dtype="string"),
            "cat": pd.Categorical(["A", "B"]),
            "dt": [pd.Timestamp("2026-01-01"), pd.NaT]
        }, index=[5, 5])  # Duplicate index labels to test positional safety
        df["i64_large"] = df["i64_large"].astype("int64")
        df["i64_bounds"] = df["i64_bounds"].astype("int64")
        df["u64"] = df["u64"].astype("uint64")
        df["i32"] = df["i32"].astype("int32")
        df["i16"] = df["i16"].astype("int16")
        df["i8"] = df["i8"].astype("int8")
        df["u32"] = df["u32"].astype("uint32")
        df["u16"] = df["u16"].astype("uint16")
        df["u8"] = df["u8"].astype("uint8")
        df["f32"] = df["f32"].astype("float32")
        df["f64"] = df["f64"].astype("float64")
        df["b_std"] = df["b_std"].astype(bool)

        self.ds.load_dataframe(df)
        self.ds.var_labels["i64_large"] = "Large Identifier"
        self.ds.val_label_names["cat"] = "cat_lbl"

        widget = DataGridWidget(self.ds)
        # Execute GUI Add Row action
        widget._add_row()

        res = self.ds.df
        self.assertEqual(len(res), 3, "New row must be appended")

        # 1. Exact value preservation on pre-existing rows
        self.assertEqual(int(res["i64_large"].iloc[0]), val_2_53)
        self.assertEqual(int(res["i64_large"].iloc[1]), val_2_53_p1)
        self.assertTrue(pd.isna(res["i64_large"].iloc[2]), "New row must be missing NA")
        self.assertEqual(str(res["i64_large"].dtype), "Int64", "Column must be promoted to nullable Int64")

        self.assertEqual(int(res["i64_bounds"].iloc[0]), min_i64)
        self.assertEqual(int(res["i64_bounds"].iloc[1]), max_i64)
        self.assertTrue(pd.isna(res["i64_bounds"].iloc[2]))

        self.assertEqual(int(res["u64"].iloc[0]), max_u64)
        self.assertEqual(int(res["u64"].iloc[1]), 42)
        self.assertTrue(pd.isna(res["u64"].iloc[2]))
        self.assertEqual(str(res["u64"].dtype), "UInt64")

        # 2. Dtype invariants
        self.assertEqual(str(res["i32"].dtype), "Int32")
        self.assertEqual(str(res["u32"].dtype), "UInt32")
        self.assertEqual(str(res["f32"].dtype), "float32")
        self.assertEqual(str(res["f64"].dtype), "float64")
        self.assertEqual(str(res["b_std"].dtype), "boolean")
        self.assertEqual(str(res["cat"].dtype), "category")
        self.assertEqual(res["dt"].dtype, df["dt"].dtype)

        # 3. Index and metadata preservation
        self.assertEqual(res.index.tolist(), [5, 5, 6])
        self.assertEqual(self.ds.var_labels.get("i64_large"), "Large Identifier")
        self.assertEqual(self.ds.val_label_names.get("cat"), "cat_lbl")

    # =========================================================================
    # Invariant 2: Authoritative Read-Only Mutation Enforcement
    # =========================================================================

    def test_02_readonly_mutation_enforcement_blocks_all_surfaces(self):
        """
        Verifies that when the grid is in a read-only state (e.g. background worker running),
        ALL data mutation paths are blocked: setData, paste_clipboard, _add_row, _add_var,
        and _on_replace_all. Once read-only is cleared, mutations succeed.
        """
        df = pd.DataFrame({
            "a": [1, 2],
            "b": ["foo", "bar"]
        })
        self.ds.load_dataframe(df)
        df_orig = df.copy()

        widget = DataGridWidget(self.ds)
        model = widget.model

        # 1. Engage Read-Only state
        widget.set_readonly(True)
        self.assertTrue(widget.is_readonly)
        self.assertTrue(model.is_readonly)

        # Path A: In-place cell edit via setData
        res_set = model.setData(model.index(0, 0), "99")
        self.assertFalse(res_set, "setData must return False in read-only state")
        self.assertEqual(self.ds.df["a"].iloc[0], 1)

        # Path B: Clipboard paste
        QtWidgets.QApplication.clipboard().setText("99\tnew_val")
        pasted = widget.paste_clipboard(start_row=0, start_col=0)
        self.assertEqual(pasted, 0, "paste_clipboard must return 0 in read-only state")
        self.assertEqual(self.ds.df["a"].iloc[0], 1)

        # Path C: Add Row
        widget._add_row()
        self.assertEqual(len(self.ds.df), 2, "_add_row must not append in read-only state")

        # Path D: Add Variable
        widget._add_var()  # Even if called directly, blocked by is_readonly
        self.assertEqual(len(self.ds.df.columns), 2, "_add_var must not add columns in read-only state")

        # Path E: Find & Replace
        widget.txt_find.setText("foo")
        widget.txt_replace.setText("baz")
        widget._on_replace_all()
        self.assertEqual(self.ds.df["b"].iloc[0], "foo", "_on_replace_all must not mutate in read-only state")

        pd.testing.assert_frame_equal(self.ds.df, df_orig, "Dataset must remain completely untouched in read-only mode")

        # 2. Release Read-Only state
        widget.set_readonly(False)
        self.assertFalse(widget.is_readonly)

        # Verify mutations succeed again
        res_set2 = model.setData(model.index(0, 0), "99")
        self.assertTrue(res_set2, "setData must succeed once editable")
        self.assertEqual(self.ds.df["a"].iloc[0], 99)

        pasted2 = widget.paste_clipboard(start_row=1, start_col=0)
        self.assertGreater(pasted2, 0, "paste_clipboard must succeed once editable")

    # =========================================================================
    # Invariant 3: Atomic Commit with Rollback on Commit-Time Failure
    # =========================================================================

    def test_03_find_replace_commit_phase_failure_injection_rollback(self):
        """
        Simulates an unexpected failure during the column-commit phase of perform_find_replace.
        Verifies that any partially assigned columns are rolled back and the dataset remains
        100% identical to its pre-operation state.
        """
        class FailingDataFrame(pd.DataFrame):
            def __setitem__(self, key, value):
                if key == "col_second":
                    raise RuntimeError("Simulated catastrophic failure during commit of second column")
                super().__setitem__(key, value)

        df = FailingDataFrame({
            "col_first": ["target_val", "other"],
            "col_second": ["target_val", "keep"],
            "num_col": [10, 20]
        })
        self.ds.df = df
        df_backup = pd.DataFrame(df).copy()

        count = perform_find_replace(self.ds, find_text="target_val", replace_text="replaced_val")
        self.assertEqual(count, 0, "Commit-phase failure must report 0 modified cells")

        # Complete dataset verification: col_first MUST be rolled back to target_val!
        self.assertEqual(self.ds.df["col_first"].iloc[0], "target_val", "col_first must roll back cleanly")
        self.assertEqual(self.ds.df["col_second"].iloc[0], "target_val", "col_second must remain target_val")
        pd.testing.assert_frame_equal(pd.DataFrame(self.ds.df), df_backup, "Entire DataFrame must match original state byte-for-byte")

    # =========================================================================
    # Invariant 4: Mixed Object-Column Element Type Preservation
    # =========================================================================

    def test_04_mixed_object_column_preserves_non_string_objects(self):
        """
        Verifies that in mixed object columns, text replacement only modifies genuine string
        elements and preserves integers, floats, booleans, and datetimes without coercing them to strings.
        """
        t_stamp = pd.Timestamp("2026-11-18")
        df = pd.DataFrame({
            "mixed": [123, "hello123", None, True, 45.6, t_stamp]
        }, dtype=object)
        self.ds.load_dataframe(df)

        # Substring replacement '12' -> 'X'
        c = perform_find_replace(self.ds, find_text="12", replace_text="X", exact_match=False)
        self.assertEqual(c, 1, "Only the string element 'hello123' should be modified")

        res_s = self.ds.df["mixed"]
        # Element 0: int 123 must remain integer, NOT coerced to 'X3'!
        self.assertTrue(isinstance(res_s.iloc[0], int))
        self.assertEqual(res_s.iloc[0], 123)

        # Element 1: string 'hello123' becomes 'helloX3'
        self.assertTrue(isinstance(res_s.iloc[1], str))
        self.assertEqual(res_s.iloc[1], "helloX3")

        # Element 2: None remains None
        self.assertTrue(res_s.iloc[2] is None or pd.isna(res_s.iloc[2]))

        # Element 3: bool True must remain boolean
        self.assertTrue(isinstance(res_s.iloc[3], bool))
        self.assertEqual(res_s.iloc[3], True)

        # Element 4: float 45.6 must remain float
        self.assertTrue(isinstance(res_s.iloc[4], float))
        self.assertEqual(res_s.iloc[4], 45.6)

        # Element 5: Timestamp must remain Timestamp
        self.assertTrue(isinstance(res_s.iloc[5], pd.Timestamp))
        self.assertEqual(res_s.iloc[5], t_stamp)

        # Explicit coerce_mixed=True allows intentional conversion
        c_forced = perform_find_replace(self.ds, find_text="12", replace_text="X", exact_match=False, coerce_mixed=True)
        self.assertGreater(c_forced, 0)
        self.assertEqual(self.ds.df["mixed"].iloc[0], "X3", "Explicit coerce_mixed converts integer to string")

    # =========================================================================
    # Invariant 5: Large-Integer Boundaries & Numeric Precision
    # =========================================================================

    def test_05_large_integer_precision_boundaries(self):
        """Tests exact 64-bit integer replacement at and beyond 2^53 limits."""
        val_2_53_m1 = 9007199254740991
        val_2_53 = 9007199254740992
        val_2_53_p1 = 9007199254740993
        max_i64 = int(np.iinfo(np.int64).max)
        max_u64 = int(np.iinfo(np.uint64).max)

        df = pd.DataFrame({
            "large_int": [val_2_53, val_2_53_p1, max_i64],
            "nullable_large": pd.Series([val_2_53_p1, val_2_53, pd.NA], dtype="Int64"),
            "u64": pd.Series([max_u64, 42, 100], dtype="uint64"),
        })
        self.ds.load_dataframe(df)

        c = perform_find_replace(self.ds, find_text=str(val_2_53_p1), replace_text=str(val_2_53_m1))
        self.assertEqual(c, 2)
        self.assertEqual(int(self.ds.df["large_int"].iloc[0]), val_2_53)
        self.assertEqual(int(self.ds.df["large_int"].iloc[1]), val_2_53_m1)
        self.assertEqual(int(self.ds.df["nullable_large"].iloc[0]), val_2_53_m1)
        self.assertEqual(int(self.ds.df["nullable_large"].iloc[1]), val_2_53)

        c_u64 = perform_find_replace(self.ds, find_text=str(max_u64), replace_text="9999")
        self.assertEqual(c_u64, 1)
        self.assertEqual(int(self.ds.df["u64"].iloc[0]), 9999)

    # =========================================================================
    # Invariant 6: Positional Safety on Duplicate Indices
    # =========================================================================

    def test_06_duplicate_index_positional_safety(self):
        """Verifies duplicate index labels do not cause sibling observation collisions."""
        df = pd.DataFrame({
            "fruit": ["apple", "banana", "cherry", "apple tart"],
            "score": [10, 20, 30, 40]
        }, index=[5, 5, 6, 6])
        self.ds.load_dataframe(df)

        c = perform_find_replace(self.ds, find_text="apple", replace_text="orange", exact_match=True)
        self.assertEqual(c, 1)
        self.assertEqual(self.ds.df["fruit"].iloc[0], "orange")
        self.assertEqual(self.ds.df["fruit"].iloc[1], "banana", "Sibling index 5 must not be overwritten!")
        self.assertEqual(self.ds.df["fruit"].iloc[2], "cherry")
        self.assertEqual(self.ds.df["fruit"].iloc[3], "apple tart")

    # =========================================================================
    # Invariant 7: Missing Value Preservation on Integer Edit
    # =========================================================================

    def test_07_editing_missing_in_large_integer_preserves_precision(self):
        """Verifies setting missing in integer column converts to Int64 without float64 precision loss."""
        val_large = 9007199254740993
        df = pd.DataFrame({"id": [val_large, 100]})
        self.ds.load_dataframe(df)
        model = VirtualPandasModel(self.ds)

        res = model.setData(model.index(1, 0), ".")
        self.assertTrue(res)
        self.assertEqual(int(self.ds.df["id"].iloc[0]), val_large)
        self.assertTrue(pd.isna(self.ds.df["id"].iloc[1]))
        self.assertEqual(str(self.ds.df["id"].dtype), "Int64")

    # =========================================================================
    # Invariant 8: Multi-Column Transactional Rollback
    # =========================================================================

    def test_08_multi_column_transactional_rollback(self):
        """Verifies mixed-column replacement aborts completely if any column fails conversion."""
        df = pd.DataFrame({
            "num_col": [1, 2, 3],
            "str_col": ["alpha", "2", "beta"]
        })
        self.ds.load_dataframe(df)
        df_orig = df.copy()

        c = perform_find_replace(self.ds, find_text="2", replace_text="invalid_num")
        self.assertEqual(c, 0)
        pd.testing.assert_frame_equal(self.ds.df, df_orig)

    # =========================================================================
    # Invariant 9: No-Op Replacement Count
    # =========================================================================

    def test_09_no_op_replacement_count(self):
        """Verifies replacing a value with itself strictly reports 0 modified cells."""
        df = pd.DataFrame({
            "str_col": ["apple", "banana", "apple"],
            "int_col": [10, 20, 10],
            "flt_col": [1.5, 2.5, 1.5]
        })
        self.ds.load_dataframe(df)
        df_orig = df.copy()

        self.assertEqual(perform_find_replace(self.ds, "apple", "apple", match_case=True), 0)
        self.assertEqual(perform_find_replace(self.ds, "10", "10"), 0)
        self.assertEqual(perform_find_replace(self.ds, "1.5", "1.5"), 0)
        self.assertEqual(perform_find_replace(self.ds, ".", "."), 0)
        pd.testing.assert_frame_equal(self.ds.df, df_orig)

    # =========================================================================
    # Invariant 10: Multi-Cell Paste Transactional Validation
    # =========================================================================

    def test_10_multicell_paste_transactional_validation(self):
        """Verifies multi-cell paste applies valid blocks and aborts invalid blocks atomically."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [10.0, 20.0, 30.0]
        })
        self.ds.load_dataframe(df)
        widget = DataGridWidget(self.ds)

        valid_text = "100\t50.5\n200\t60.5"
        QtWidgets.QApplication.clipboard().setText(valid_text)
        pasted = widget.paste_clipboard(start_row=0, start_col=0)
        self.assertEqual(pasted, 4)
        self.assertEqual(self.ds.df["a"].iloc[0], 100)
        self.assertEqual(self.ds.df["b"].iloc[0], 50.5)

        df_before = self.ds.df.copy()
        invalid_text = "300\t70.5\n1.9\t80.5"
        QtWidgets.QApplication.clipboard().setText(invalid_text)
        pasted_invalid = widget.paste_clipboard(start_row=1, start_col=0)
        self.assertEqual(pasted_invalid, 0)
        pd.testing.assert_frame_equal(self.ds.df, df_before)

    # =========================================================================
    # Invariant 11: Synchronized Sorted and Filtered Views
    # =========================================================================

    def test_11_sorted_and_filtered_views_remain_synchronized(self):
        """Verifies sorting and filtering remain synchronized during cell edits."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David"],
            "score": [80, 95, 70, 85]
        }, index=[10, 20, 10, 20])
        self.ds.load_dataframe(df)
        model = VirtualPandasModel(self.ds)

        model.set_filter("david")
        self.assertEqual(model.rowCount(), 1)
        model.setData(model.index(0, 0), "Daniel")
        self.assertEqual(self.ds.df["name"].iloc[3], "Daniel")
        self.assertEqual(model.rowCount(), 0)

        model.set_filter("")
        self.assertEqual(model.rowCount(), 4)
        model.sort_by_column(1)
        self.assertEqual(model.data(model.index(0, 0)), "Charlie")
        model.setData(model.index(0, 1), "100")
        self.assertEqual(self.ds.df["score"].iloc[2], 100)
        self.assertEqual(model.data(model.index(0, 0)), "Alice")

    # =========================================================================
    # Invariant 12: Numeric Categorical Editing & Type Preservation (Task 1)
    # =========================================================================

    def test_12_numeric_categorical_editing_and_type_preservation(self):
        """
        Verifies dtype-aware categorical editing:
        - Editing existing category preserves underlying Python/pandas type (int, float, bool, etc.).
        - Entering '1' into [1, 2, 1] retains integer category 1 (never string '1').
        - Adding legitimate new category ('3') preserves category dtype without corrupting observations.
        - Invalid text ('abc', '1.5') cannot become a category in numeric categorical; rejected.
        - Ordered categories retain ordering.
        - Missing values ('.') set code -1 without adding NaN category.
        - Value labels resolve text and underlying integer codes.
        """
        # 1. Integer Categorical preservation
        s = pd.Series(pd.Categorical([1, 2, 1], categories=[1, 2]))
        df = pd.DataFrame({"cat_int": s})
        self.ds.load_dataframe(df)
        model = VirtualPandasModel(self.ds)

        # Edit cell 0 to "1"
        ok = model.setData(model.index(0, 0), "1")
        self.assertTrue(ok)
        val0 = self.ds.df["cat_int"].iloc[0]
        self.assertEqual(val0, 1)
        self.assertIsInstance(val0, (int, np.integer))
        self.assertEqual(list(self.ds.df["cat_int"].cat.categories), [1, 2])
        self.assertEqual(str(self.ds.df["cat_int"].cat.categories.dtype), "int64")

        # Edit cell 0 to legitimate new numeric category "3"
        ok = model.setData(model.index(0, 0), "3")
        self.assertTrue(ok)
        self.assertEqual(self.ds.df["cat_int"].iloc[0], 3)
        self.assertEqual(list(self.ds.df["cat_int"].cat.categories), [1, 2, 3])
        self.assertEqual(str(self.ds.df["cat_int"].cat.categories.dtype), "int64")

        # Try entering invalid text "abc" into numeric categorical
        ok = model.setData(model.index(0, 0), "abc")
        self.assertFalse(ok, "Incompatible text must be rejected for numeric categorical")
        self.assertEqual(self.ds.df["cat_int"].iloc[0], 3)
        self.assertEqual(list(self.ds.df["cat_int"].cat.categories), [1, 2, 3])

        # Try entering fractional "3.5" into integer categorical
        ok = model.setData(model.index(0, 0), "3.5")
        self.assertFalse(ok, "Fractional float must be rejected for integer categorical")

        # Set missing value "."
        ok = model.setData(model.index(0, 0), ".")
        self.assertTrue(ok)
        self.assertTrue(pd.isna(self.ds.df["cat_int"].iloc[0]))
        self.assertEqual(list(self.ds.df["cat_int"].cat.categories), [1, 2, 3])

        # 2. Ordered category semantics
        s_ord = pd.Series(pd.Categorical([10, 20, 10], categories=[10, 20], ordered=True))
        df_ord = pd.DataFrame({"cat_ord": s_ord})
        self.ds.load_dataframe(df_ord)
        model_ord = VirtualPandasModel(self.ds)

        ok = model_ord.setData(model_ord.index(0, 0), "10")
        self.assertTrue(ok)
        self.assertTrue(self.ds.df["cat_ord"].cat.ordered)
        self.assertEqual(list(self.ds.df["cat_ord"].cat.categories), [10, 20])

        ok = model_ord.setData(model_ord.index(0, 0), "30")
        self.assertTrue(ok)
        self.assertTrue(self.ds.df["cat_ord"].cat.ordered)
        self.assertEqual(list(self.ds.df["cat_ord"].cat.categories), [10, 20, 30])

        # 3. Value label resolution
        df_lbl = pd.DataFrame({"gender": pd.Series(pd.Categorical([1, 2, 1]))})
        self.ds.load_dataframe(df_lbl)
        self.ds.val_label_names["gender"] = "lbl_gender"
        self.ds.val_labels["lbl_gender"] = {1: "Male", 2: "Female"}
        model_lbl = VirtualPandasModel(self.ds)

        # Enter label string "Female"
        ok = model_lbl.setData(model_lbl.index(0, 0), "Female")
        self.assertTrue(ok)
        self.assertEqual(self.ds.df["gender"].iloc[0], 2)
        self.assertEqual(model_lbl.data(model_lbl.index(0, 0)), "Female")

    # =========================================================================
    # Invariant 13: Safe Floating-Point Precision & Boundary Enforcement (Task 2)
    # =========================================================================

    def test_13_safe_floating_point_overflow_underflow_and_precision(self):
        """
        Verifies floating-point validation across float32 and float64:
        - Ordinary decimals (0.1, 123.456) accepted with normal IEEE rounding.
        - Float32 maximum finite value accepted.
        - Overflow (1e100) rejected; positive infinity never silently stored.
        - Underflow (1e-50) rejected from silently underflowing to zero.
        - Large integers exceeding float32 exact limit (16777217) rejected.
        - Large integers within float32 exact limit (16777216) accepted.
        - Explicit NaN and infinity handled per policy.
        - Float64 boundary values and precision limits enforced.
        """
        df = pd.DataFrame({
            "f32": np.array([1.5, 2.5], dtype=np.float32),
            "f64": [100.0, 200.0]
        })
        self.ds.load_dataframe(df)
        model = VirtualPandasModel(self.ds)

        # 1. Ordinary decimal
        ok = model.setData(model.index(0, 0), "0.1")
        self.assertTrue(ok)
        self.assertAlmostEqual(float(self.ds.df["f32"].iloc[0]), 0.1, places=6)

        # 2. Maximum finite float32
        max_f32_str = str(np.finfo(np.float32).max)
        ok = model.setData(model.index(0, 0), max_f32_str)
        self.assertTrue(ok)

        # 3. Float32 overflow rejection
        f32_before = self.ds.df["f32"].iloc[0]
        ok = model.setData(model.index(0, 0), "1e100")
        self.assertFalse(ok, "Overflow '1e100' must be rejected for float32")
        self.assertEqual(self.ds.df["f32"].iloc[0], f32_before, "Value must remain unchanged")

        # 4. Float32 underflow rejection
        ok = model.setData(model.index(0, 0), "1e-50")
        self.assertFalse(ok, "Material underflow '1e-50' to zero must be rejected for float32")
        self.assertEqual(self.ds.df["f32"].iloc[0], f32_before)

        # 5. Float32 integer precision limit: 2^24 = 16,777,216
        ok_exact = model.setData(model.index(0, 0), "16777216")
        self.assertTrue(ok_exact, "Exact integer 16,777,216 must be accepted in float32")
        self.assertEqual(int(self.ds.df["f32"].iloc[0]), 16777216)

        ok_loss = model.setData(model.index(0, 0), "16777217")
        self.assertFalse(ok_loss, "Inexact integer 16,777,217 must be rejected in float32 for precision loss")
        self.assertEqual(int(self.ds.df["f32"].iloc[0]), 16777216)

        # 6. Explicit infinity and NaN
        ok = model.setData(model.index(0, 0), "inf")
        self.assertTrue(ok)
        self.assertTrue(np.isposinf(self.ds.df["f32"].iloc[0]))

        ok = model.setData(model.index(0, 0), "-inf")
        self.assertTrue(ok)
        self.assertTrue(np.isneginf(self.ds.df["f32"].iloc[0]))

        ok = model.setData(model.index(0, 0), "nan")
        self.assertTrue(ok)
        self.assertTrue(np.isnan(self.ds.df["f32"].iloc[0]))

        # 7. Float64 boundary values and precision limits
        ok = model.setData(model.index(0, 1), "1e300")
        self.assertTrue(ok)
        self.assertEqual(self.ds.df["f64"].iloc[0], 1e300)

        ok = model.setData(model.index(0, 1), "1e400")
        self.assertFalse(ok, "Float64 overflow beyond 1e308 must be rejected")

        # 2^53 = 9007199254740992 is exact in float64; 2^53 + 1 loses precision
        ok = model.setData(model.index(0, 1), "9007199254740992")
        self.assertTrue(ok)

        ok = model.setData(model.index(0, 1), "9007199254740993")
        self.assertFalse(ok, "Inexact integer > 2^53 must be rejected in float64 for precision loss")

    # =========================================================================
    # Invariant 14: Index Identity Preservation & Boundary Safety (Task 3)
    # =========================================================================

    def test_14_index_identity_preservation_large_integers_and_boundaries(self):
        """
        Verifies index safety during row insertion:
        - RangeIndex datasets preserve RangeIndex positional semantics.
        - Nonconsecutive integer indices preserve labels and integer dtype.
        - Duplicate integer labels are preserved without modifying identities.
        - Signed int64 near maximum (max_i64 - 1) appends max_i64 without float conversion.
        - Signed int64 AT maximum (max_i64) avoids overflow and float coercion.
        - Unsigned uint64 at maximum avoids overflow and float coercion.
        - String indices generate unique labels without modifying existing labels.
        - Index names are preserved across row insertions.
        """
        # 1. RangeIndex
        df_range = pd.DataFrame({"x": [10, 20]})
        df_range.index.name = "pos_idx"
        res_range = insert_blank_observation(df_range)
        self.assertIsInstance(res_range.index, pd.RangeIndex)
        self.assertEqual(len(res_range), 3)
        self.assertEqual(list(res_range.index), [0, 1, 2])
        self.assertEqual(res_range.index.name, "pos_idx")

        # 2. Nonconsecutive and duplicate integer indices
        df_dups = pd.DataFrame({"x": [10, 20, 30]}, index=[100, 100, 200])
        res_dups = insert_blank_observation(df_dups)
        self.assertEqual(res_dups.index.tolist(), [100, 100, 200, 201])
        self.assertEqual(str(res_dups.index.dtype), "int64")

        # 3. Signed int64 near maximum
        max_i64 = np.iinfo(np.int64).max
        df_near_max = pd.DataFrame({"x": [1, 2]}, index=pd.Index([max_i64 - 2, max_i64 - 1], dtype=np.int64))
        res_near_max = insert_blank_observation(df_near_max)
        self.assertEqual(res_near_max.index.dtype, np.int64)
        self.assertEqual(res_near_max.index.tolist(), [max_i64 - 2, max_i64 - 1, max_i64])

        # 4. Signed int64 AT maximum (must never cast to float64!)
        df_at_max = pd.DataFrame({"x": [1, 2]}, index=pd.Index([max_i64 - 1, max_i64], dtype=np.int64))
        res_at_max = insert_blank_observation(df_at_max)
        self.assertEqual(res_at_max.index.dtype, np.int64)
        self.assertEqual(res_at_max.index.tolist()[:2], [max_i64 - 1, max_i64])
        self.assertNotEqual(str(res_at_max.index.dtype), "float64")
        self.assertEqual(len(res_at_max), 3)

        # 5. Unsigned uint64 AT maximum
        max_u64 = np.iinfo(np.uint64).max
        df_u64 = pd.DataFrame({"x": [1, 2]}, index=pd.Index([max_u64 - 1, max_u64], dtype=np.uint64))
        res_u64 = insert_blank_observation(df_u64)
        self.assertEqual(res_u64.index.dtype, np.uint64)
        self.assertEqual(res_u64.index.tolist()[:2], [max_u64 - 1, max_u64])
        self.assertNotEqual(str(res_u64.index.dtype), "float64")

        # 6. String indices
        df_str = pd.DataFrame({"x": [1, 2]}, index=["obs_1", "obs_2"])
        res_str = insert_blank_observation(df_str)
        self.assertEqual(res_str.index.tolist(), ["obs_1", "obs_2", "obs_3"])

    # =========================================================================
    # Invariant 15: Find & Replace & Clipboard Integration (Tasks 1 & 2)
    # =========================================================================

    def test_15_find_replace_and_clipboard_with_floats_and_categoricals(self):
        """
        Verifies consistent validation across clipboard paste and Find & Replace:
        - Pasting overflow value (1e100) into float32 fails transactionally (0 cells modified).
        - Pasting lossy integer (16777217) into float32 fails transactionally.
        - Pasting integer into integer categorical preserves integer categories.
        - Find & Replace replacing float32 with overflow (1e100) fails transactionally (returns 0).
        - Find & Replace replacing integer categorical with new integer preserves int category type.
        - Find & Replace replacing integer categorical with invalid text fails transactionally.
        """
        df = pd.DataFrame({
            "f32": np.array([1.5, 2.5], dtype=np.float32),
            "cat": pd.Series(pd.Categorical([1, 2]))
        })
        self.ds.load_dataframe(df)
        widget = DataGridWidget(self.ds)

        # 1. Clipboard paste overflow into float32
        df_backup = self.ds.df.copy()
        QtWidgets.QApplication.clipboard().setText("1e100")
        pasted = widget.paste_clipboard(start_row=0, start_col=0)
        self.assertEqual(pasted, 0, "Pasting overflow into float32 must fail transactionally")
        pd.testing.assert_frame_equal(self.ds.df, df_backup)

        # 2. Clipboard paste lossy integer into float32
        QtWidgets.QApplication.clipboard().setText("16777217")
        pasted = widget.paste_clipboard(start_row=0, start_col=0)
        self.assertEqual(pasted, 0, "Pasting lossy integer into float32 must fail transactionally")
        pd.testing.assert_frame_equal(self.ds.df, df_backup)

        # 3. Clipboard paste valid integer into integer categorical
        QtWidgets.QApplication.clipboard().setText("3")
        pasted = widget.paste_clipboard(start_row=0, start_col=1)
        self.assertEqual(pasted, 1)
        self.assertEqual(self.ds.df["cat"].iloc[0], 3)
        self.assertEqual(list(self.ds.df["cat"].cat.categories), [1, 2, 3])
        self.assertEqual(str(self.ds.df["cat"].cat.categories.dtype), "int64")

        # 4. Find & Replace float32 with overflow
        df_before_fr = self.ds.df.copy()
        n = perform_find_replace(self.ds, find_text="2.5", replace_text="1e100", exact_match=True)
        self.assertEqual(n, 0, "Replacing with overflow in float32 must return 0")
        pd.testing.assert_frame_equal(self.ds.df, df_before_fr)

        # 5. Find & Replace integer categorical with valid integer
        n = perform_find_replace(self.ds, find_text="2", replace_text="4", exact_match=True)
        self.assertEqual(n, 1)
        self.assertEqual(self.ds.df["cat"].iloc[1], 4)
        self.assertEqual(list(self.ds.df["cat"].cat.categories), [1, 2, 3, 4])
        self.assertEqual(str(self.ds.df["cat"].cat.categories.dtype), "int64")

        # 6. Find & Replace integer categorical with incompatible text
        df_before_bad = self.ds.df.copy()
        n = perform_find_replace(self.ds, find_text="4", replace_text="invalid_text", exact_match=True)
        self.assertEqual(n, 0, "Replacing integer categorical with invalid text must return 0")
        pd.testing.assert_frame_equal(self.ds.df, df_before_bad)

        # 7. Find & Replace categorical with missing ('.')
        n_miss = perform_find_replace(self.ds, find_text="3", replace_text=".", exact_match=True)
        self.assertEqual(n_miss, 1, "Replacing categorical with '.' must succeed")
        self.assertTrue(pd.isna(self.ds.df["cat"].iloc[0]))
        self.assertEqual(self.ds.df["cat"].iloc[1], 4)

    # =========================================================================
    # Invariant 16: Session Persistence Dtype Fidelity
    # =========================================================================

    def test_16_session_persistence_numeric_categorical_fidelity(self):
        """Verifies session save and reload preserves numeric categorical dtypes."""
        import tempfile
        from engine.interpreter import Interpreter
        from engine.session_manager import save_session_bundle, load_session_bundle

        interp1 = Interpreter(Dataset())
        df = pd.DataFrame({
            "num_cat": pd.Series(pd.Categorical([10, 20, 10], categories=[10, 20])),
            "f32": np.array([1.5, 2.5, 3.5], dtype=np.float32)
        })
        interp1.dataset.load_dataframe(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = os.path.join(tmpdir, "test_session.potproj")
            save_session_bundle(bundle_path, interp1)

            interp2 = Interpreter(Dataset())
            load_session_bundle(bundle_path, interp2)

            res_df = interp2.dataset.df
            self.assertEqual(str(res_df["num_cat"].dtype), "category")
            self.assertEqual(list(res_df["num_cat"].cat.categories), [10, 20])
            self.assertEqual(res_df["num_cat"].tolist(), [10, 20, 10])
            self.assertEqual(str(res_df["f32"].dtype), "float32")


    # =========================================================================
    # Invariant 17: RangeIndex Custom Progression & Name Preservation
    # =========================================================================

    def test_17_blank_observation_custom_rangeindex_preservation(self):
        """
        Verifies that inserting a blank observation into a DataFrame with a
        custom RangeIndex (start!=0 or step!=1) preserves start, step, and name,
        rather than resetting to default RangeIndex(0, N, 1).
        """
        df = pd.DataFrame(
            {"val": [10, 20]},
            index=pd.RangeIndex(start=100, stop=104, step=2, name="custom_idx")
        )
        res = insert_blank_observation(df)
        self.assertIsInstance(res.index, pd.RangeIndex)
        self.assertEqual(res.index.start, 100)
        self.assertEqual(res.index.step, 2)
        self.assertEqual(res.index.stop, 106)
        self.assertEqual(res.index.name, "custom_idx")
        self.assertEqual(list(res.index), [100, 102, 104])
        self.assertEqual(len(res), 3)

    # =========================================================================
    # Invariant 18: Lexical Underflow to Zero Rejection
    # =========================================================================

    def test_18_lexical_underflow_validation(self):
        """
        Verifies that lexical underflow inputs (e.g. 1e-5000) are rejected with
        an informative error rather than silently becoming 0.0, while actual
        zeros ('0', '0.0', '0e-5') remain valid.
        """
        from gui_qt.data_grid import validate_float_input

        # Non-zero underflow should be rejected for float64
        ok, val, err = validate_float_input("1e-5000", np.float64)
        self.assertFalse(ok)
        self.assertIn("underflows float64 to zero", err)

        # Non-zero underflow should be rejected for float32
        ok32, val32, err32 = validate_float_input("1e-5000", np.float32)
        self.assertFalse(ok32)
        self.assertIn("underflows float32 to zero", err32)

        # Subnormal underflow in float32
        ok_f32_uf, _, err_f32_uf = validate_float_input("1e-46", np.float32)
        self.assertFalse(ok_f32_uf)
        self.assertIn("underflows float32 to zero", err_f32_uf)

        # Genuine zeros must be accepted
        for zero_str in ["0", "0.0", "-0.0", "+0.0", "0e-10", "0.00e5"]:
            ok_z, val_z, _ = validate_float_input(zero_str, np.float64)
            self.assertTrue(ok_z, f"Zero representation '{zero_str}' should be accepted")
            self.assertEqual(val_z, 0.0)

    # =========================================================================
    # Invariant 19: Publication-Quality Table Console Exporters
    # =========================================================================

    def test_19_console_table_latex_and_markdown_exporters(self):
        """
        Verifies that console table exporters:
        1. Extract the latest table without stitching earlier tables.
        2. Dynamically determine column count (no hardcoded lrrrrr).
        3. Properly escape LaTeX special characters (_, %, &, #, $).
        4. Escape Markdown pipes and preserve multi-column confidence intervals.
        """
        from gui_qt.console import (
            escape_latex,
            escape_markdown,
            extract_latest_table_lines,
            format_table_as_latex,
            format_table_as_markdown
        )
        from engine.interpreter import Interpreter

        interp = Interpreter(Dataset())
        df = pd.DataFrame({
            "wage_usd": [10.5, 20.2, 15.0, 30.1, 25.4],
            "educ_yr": [12, 16, 14, 18, 16]
        })
        interp.dataset.load_dataframe(df)

        out_sum = interp.execute("summarize wage_usd educ_yr").output
        out_reg = interp.execute("regress wage_usd educ_yr").output

        # Stitched historical console text
        console_text = f"{out_sum}\n\n. regress wage_usd educ_yr\n{out_reg}"

        # 1. Isolation: Only the regression table should be extracted
        extracted = extract_latest_table_lines(console_text)
        self.assertTrue(len(extracted) >= 4)
        joined_ext = "\n".join(extracted)
        self.assertIn("wage_usd", joined_ext)
        self.assertNotIn("Std. dev.", joined_ext, "Summarize table must not be mixed into regress table")

        # 2. LaTeX formatting with dynamic columns and escaping
        ltx = format_table_as_latex(extracted)
        self.assertIsNotNone(ltx)
        self.assertIn("\\begin{tabular}{lrrrrrr}", ltx, "Regress table should have 7 columns (l + 6 r)")
        self.assertIn("wage\\_usd", ltx, "Underscore must be escaped in LaTeX")
        self.assertIn("educ\\_yr", ltx)
        self.assertIn("\\multicolumn{2}{c}{[95\\% conf. interval]}", ltx, "CI header should span 2 columns")
        self.assertIn("\\toprule", ltx)
        self.assertIn("\\bottomrule", ltx)

        # 3. Markdown formatting
        md = format_table_as_markdown(extracted)
        self.assertIsNotNone(md)
        self.assertIn("| wage_usd |", md)
        self.assertIn("|:---|---:|---:|---:|---:|---:|---:|", md, "Markdown table must have 7 aligned columns")
        self.assertIn("P>\\|t\\|", md, "Pipe in P>|t| must be escaped in Markdown")

        # 4. Escaping unit tests
        self.assertEqual(escape_latex("cost_$ & val_%"), "cost\\_\\$ \\& val\\_\\%")
        self.assertEqual(escape_latex("a\\b{c}"), r"a\textbackslash{}b\{c\}")
        self.assertEqual(escape_markdown("a|b|c"), "a\\|b\\|c")


if __name__ == "__main__":
    unittest.main()
