import os
import tempfile
import unittest
import pandas as pd
from engine.dataset import Dataset
from gui_qt.data_grid import VirtualPandasModel
from PySide6.QtCore import Qt

class TestDTAValueLabelsAndFormats(unittest.TestCase):
    def test_p1_non_matching_label_table_name_and_shared_tables(self):
        """
        DTA value label tables where variable name != label table name,
        and multiple variables sharing a single label table.
        """
        ds = Dataset()
        # Create dataset with shared label mapping
        df = pd.DataFrame({
            'sex': [1, 2, 1, 2],
            'head_sex': [2, 1, 1, 2],
            'status': [0, 1, -1, 0]
        })
        ds.load_dataframe(df)

        # Non-matching label table name
        ds.val_label_names['sex'] = 'gender_lbl'
        ds.val_label_names['head_sex'] = 'gender_lbl' # shared table
        ds.val_label_names['status'] = 'status_lbl'

        ds.val_labels['gender_lbl'] = {1: 'Male', 2: 'Female'}
        ds.val_labels['status_lbl'] = {-1: 'Refused', 0: 'Inactive', 1: 'Active'}
        ds.var_formats = {'sex': '%8.0g', 'head_sex': '%8.0g', 'status': '%10.0g'}

        with tempfile.TemporaryDirectory() as tmpdir:
            dta_path = os.path.join(tmpdir, "test_labels.dta")
            ds.save_dta(dta_path)

            # Reload into a fresh dataset
            ds2 = Dataset()
            ds2.load_dta(dta_path)

            # Assert variables preserved their label associations
            self.assertEqual(len(ds2.df), 4)
            # Verify sex and head_sex have label mappings
            lbl_sex = ds2.val_label_names.get('sex')
            lbl_head = ds2.val_label_names.get('head_sex')
            self.assertTrue(bool(lbl_sex))
            self.assertTrue(bool(lbl_head))

            # Verify label content
            map_sex = ds2.val_labels.get(lbl_sex)
            self.assertIsNotNone(map_sex)
            self.assertEqual(map_sex[1], 'Male')
            self.assertEqual(map_sex[2], 'Female')

            map_status = ds2.val_labels.get(ds2.val_label_names.get('status'))
            self.assertIsNotNone(map_status)
            self.assertEqual(map_status[-1], 'Refused')

            # Verify GUI DataGrid resolves both variables to labeled text
            model = VirtualPandasModel(ds2)
            # Row 0: sex is 1 -> 'Male'
            self.assertEqual(model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole), 'Male')
            # Row 0: head_sex is 2 -> 'Female'
            self.assertEqual(model.data(model.index(0, 1), Qt.ItemDataRole.DisplayRole), 'Female')
            # Row 2: status is -1 -> 'Refused'
            self.assertEqual(model.data(model.index(2, 2), Qt.ItemDataRole.DisplayRole), 'Refused')

    def test_p1_dta_formats_loaded_from_metadata(self):
        """Auto.dta formats must be accurately read from file, not default %9.0g."""
        auto_path = os.path.join(os.path.dirname(__file__), "data", "auto.dta")
        if not os.path.exists(auto_path):
            self.skipTest("auto.dta fixture missing")

        ds = Dataset()
        ds.load_dta(auto_path)

        # In auto.dta: make is %17s, mpg is %8.0g, etc.
        self.assertIn('make', ds.var_formats)
        self.assertEqual(ds.var_formats['make'], '%17s')
        self.assertEqual(ds.var_formats['foreign'], '%8.0g')
        self.assertEqual(ds.val_label_names['foreign'], 'foreign')
        self.assertEqual(ds.val_labels['foreign'][0], 'Domestic')
        self.assertEqual(ds.val_labels['foreign'][1], 'Foreign')

if __name__ == '__main__':
    unittest.main()
