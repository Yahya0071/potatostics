"""
tests/test_audit_section6.py - TDD Verification Suite for Section 6 (Data I/O & .dta Round-Trip)
"""

import os
import unittest
import pandas as pd

from engine.dataset import Dataset
from engine.interpreter import Interpreter


class TestSection6DataIO(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_io_temp")
        os.makedirs(self.test_dir, exist_ok=True)
        self.dataset = Dataset()
        self.interp = Interpreter(self.dataset)

    def tearDown(self):
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_6a_dta_round_trip(self):
        """Test .dta writing via DataFrame.to_dta and reading via pandas ReferenceReader."""
        df_orig = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'wage': [15.5, 22.0, 18.25, 30.0],
            'educ': [12, 16, 14, 18]
        })
        self.dataset.load_dataframe(df_orig)
        self.dataset.var_labels = {'id': 'Worker ID', 'wage': 'Hourly Wage in USD', 'educ': 'Years of Education'}

        out_dta = os.path.join(self.test_dir, "roundtrip.dta")
        self.dataset.save_dta(out_dta)
        self.assertTrue(os.path.exists(out_dta))

        # Load back into fresh dataset
        ds2 = Dataset()
        n, k = ds2.load_dta(out_dta)
        self.assertEqual(n, 4)
        self.assertEqual(k, 3)
        pd.testing.assert_frame_equal(ds2.df[['id', 'wage', 'educ']], df_orig[['id', 'wage', 'educ']], check_dtype=False)
        self.assertEqual(ds2.var_labels.get('wage'), 'Hourly Wage in USD')
        self.assertEqual(ds2.var_labels.get('educ'), 'Years of Education')

    def test_6b_save_selects_format_by_extension_never_csv_to_dta(self):
        """save command must choose format by extension and NEVER write CSV text to a .dta file."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4.0, 5.0, 6.0]})
        self.dataset.load_dataframe(df)

        dta_path = os.path.join(self.test_dir, "test_ext.dta")
        csv_path = os.path.join(self.test_dir, "test_ext.csv")

        # Save to .dta
        res_dta = self.interp.execute(f'save "{dta_path}", replace')
        self.assertTrue(res_dta.ok)
        self.assertTrue(os.path.exists(dta_path))

        # Check file header is binary Reference, NOT CSV text
        with open(dta_path, 'rb') as f:
            header_bytes = f.read(16)
        # DTA 117+ binary header
        self.assertTrue(header_bytes.startswith(b'<' + b's' + b't' + b'a' + b't' + b'a' + b'_dta>') or header_bytes.startswith(b'\x75') or header_bytes.startswith(b'\x76'),
                        msg="Saved .dta file must have binary Reference header, not CSV plain text")

        # Save to .csv
        res_csv = self.interp.execute(f'save "{csv_path}", replace')
        self.assertTrue(res_csv.ok)
        with open(csv_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        self.assertEqual(first_line, "x,y")

    def test_6c_dta_reader_file_deleted(self):
        """dta_reader.py must be deleted."""
        dta_reader_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dta_reader.py")
        self.assertFalse(os.path.exists(dta_reader_path), msg="dta_reader.py must be deleted from codebase")


if __name__ == '__main__':
    unittest.main()
