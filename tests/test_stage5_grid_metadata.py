"""
tests/test_stage5_grid_metadata.py - Tests for Stage 5 Data Grid & Metadata Preservation
"""

import unittest
import numpy as np
import pandas as pd

from PySide6 import QtWidgets

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from gui_qt.data_grid import DataGridWidget, perform_find_replace

app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(["--platform", "offscreen"])


class TestStage5GridMetadata(unittest.TestCase):
    def setUp(self):
        self.ds = Dataset()
        self.interp = Interpreter(self.ds)

    def test_data_grid_apply_filter(self):
        """DataGridWidget._apply_filter must filter visible rows without altering underlying dataset."""
        df = pd.DataFrame({
            "make": ["Buick", "Cadillac", "Chevrolet", "Ford", "Honda"],
            "price": [4099, 14500, 3299, 4500, 5200]
        })
        self.ds.load_dataframe(df)
        grid = DataGridWidget(self.ds)
        
        self.assertEqual(grid.model.rowCount(), 5)
        
        # Apply filter "c" -> matches Buick, Cadillac, Chevrolet (3 rows)
        grid._apply_filter("c")
        self.assertEqual(grid.model.rowCount(), 3)
        self.assertIn("3 of 5 rows matched", grid.lbl_status.text())

        # Clear filter
        grid._apply_filter("")
        self.assertEqual(grid.model.rowCount(), 5)

        # Underlying dataset must remain unchanged
        self.assertEqual(len(self.ds.df), 5)

    def test_perform_find_replace_comprehensive(self):
        """Find & Replace must support backslashes, regex characters, categories, numbers, and missing values."""
        df = pd.DataFrame({
            "path": ["C:\\old\\dir\\file1.txt", "D:\\old\\dir\\file2.txt", "keep_me"],
            "num": [10.0, 20.0, np.nan],
            "cat": pd.Categorical(["A", "B", "A"])
        })
        self.ds.load_dataframe(df)

        # 1. Backslash replacement string
        replaced = perform_find_replace(self.ds, "old\\dir", "new\\target\\path")
        self.assertEqual(replaced, 2)
        self.assertIn("C:\\new\\target\\path\\file1.txt", self.ds.df["path"].iloc[0])

        # 2. Missing value replacement
        rep_missing = perform_find_replace(self.ds, ".", "999")
        self.assertEqual(rep_missing, 1)
        self.assertEqual(self.ds.df["num"].iloc[2], 999.0)

        # 3. Categorical column replacement (exact match)
        rep_cat = perform_find_replace(self.ds, "A", "Alpha", exact_match=True)
        self.assertEqual(rep_cat, 2)
        self.assertEqual(self.ds.df["cat"].iloc[0], "Alpha")

        # 4. Numeric replacement
        rep_num = perform_find_replace(self.ds, "10", "15")
        self.assertEqual(rep_num, 1)
        self.assertEqual(self.ds.df["num"].iloc[0], 15.0)

    def test_metadata_preservation_across_rename(self):
        """Renaming a variable must preserve its variable label and value label bindings."""
        df = pd.DataFrame({
            "foreign": [0, 1, 0, 1],
            "price": [4000, 5000, 6000, 7000]
        })
        self.ds.load_dataframe(df)
        self.ds.var_labels["foreign"] = "Car origin"
        self.ds.val_label_names["foreign"] = "origin_lbl"
        self.ds.val_labels["origin_lbl"] = {0: "Domestic", 1: "Foreign"}

        res = self.interp.execute("rename foreign origin")
        self.assertTrue(res.ok)
        self.assertIn("origin", self.ds.df.columns)
        self.assertNotIn("foreign", self.ds.df.columns)

        # Labels must be transferred to new variable name
        self.assertEqual(self.ds.var_labels.get("origin"), "Car origin")
        self.assertEqual(self.ds.val_label_names.get("origin"), "origin_lbl")

    def test_dataset_supported_formats_documentation(self):
        """Dataset docstring must accurately list supported formats and omit unsupported claims."""
        doc = Dataset.__doc__
        self.assertIsNotNone(doc)
        self.assertNotIn("Excel", doc, "Dataset docstring should not claim unsupported Excel format")
        self.assertIn("DTA", doc)
        self.assertIn("CSV", doc)


if __name__ == "__main__":
    unittest.main()
