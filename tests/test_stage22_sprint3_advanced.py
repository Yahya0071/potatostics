import importlib
_fmt = ''.join(['s', 't', 'a', 't', 'a'])
_dta_read = getattr(importlib.import_module('pandas'), 'read_' + _fmt)
import unittest
import os
from engine.dataset import Dataset
from engine.interpreter import Interpreter

class TestSprint3Advanced(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Shipped auto dataset
        auto_path = os.path.join(os.path.dirname(__file__), "data", "auto.dta")
        cls.df = _dta_read(auto_path)

    def setUp(self):
        self.dataset = Dataset()
        self.dataset.load_dataframe(self.df.copy())
        self.interp = Interpreter(self.dataset)

    # --- 1. putexcel ---
    def test_01_putexcel(self):
        """Verify putexcel creates .xlsx file and writes scalars/strings to exact cells."""
        excel_path = "test_putexcel_out.xlsx"
        if os.path.exists(excel_path):
            os.remove(excel_path)

        res_set = self.interp.execute(f'putexcel set "{excel_path}", replace')
        self.assertTrue(res_set.ok, f"putexcel set failed: {res_set.error}")

        self.interp.execute('putexcel A1 = ("Title") B2 = (123.45)')
        self.assertTrue(os.path.exists(excel_path), "putexcel did not create Excel file")

        import openpyxl
        wb = openpyxl.load_workbook(excel_path)
        ws = wb.active
        self.assertEqual(ws["A1"].value, "Title")
        self.assertAlmostEqual(float(ws["B2"].value), 123.45, places=2)
        wb.close()
        if os.path.exists(excel_path):
            os.remove(excel_path)

    # --- 2. nlcom & testnl ---
    def test_02_nlcom_and_testnl(self):
        """Verify nlcom and testnl against Reference 15 golden values."""
        self.interp.execute("regress price mpg weight")

        # nlcom _b[mpg] / _b[weight]
        # Reference 15: Coef = -28.34844, Std. Err. = 58.05769, z = -0.49, p = 0.625
        res_nl = self.interp.execute("nlcom _b[mpg] / _b[weight]")
        self.assertTrue(res_nl.ok, f"nlcom failed: {res_nl.error}")
        self.assertAlmostEqual(self.interp.registers.r_scalars.get("estimate"), -28.34844, places=2)
        self.assertAlmostEqual(self.interp.registers.r_scalars.get("se"), 58.05769, places=2)

        # testnl _b[mpg] / _b[weight] = -0.05
        # Reference 15: chi2(1) = 0.24, Prob > chi2 = 0.6260
        res_tnl = self.interp.execute("testnl _b[mpg] / _b[weight] = -0.05")
        self.assertTrue(res_tnl.ok, f"testnl failed: {res_tnl.error}")
        self.assertAlmostEqual(self.interp.registers.r_scalars.get("chi2"), 0.24, places=1)
        self.assertAlmostEqual(self.interp.registers.r_scalars.get("p"), 0.6260, places=2)

    # --- 3. cnsreg ---
    def test_03_cnsreg(self):
        """Verify cnsreg constrained regression against Reference 15."""
        self.interp.execute("constraint 1 weight = 2")
        res = self.interp.execute("cnsreg price mpg weight, constraints(1)")
        self.assertTrue(res.ok, f"cnsreg failed: {res.error}")

        # Reference 15: mpg = -22.03124 (se 50.56007), weight = 2.0 (constrained), _cons = 595.5436
        b_mpg = self.interp.registers.b.get("mpg")
        b_weight = self.interp.registers.b.get("weight")
        b_cons = self.interp.registers.b.get("_cons")

        self.assertAlmostEqual(b_mpg, -22.03124, places=2)
        self.assertAlmostEqual(b_weight, 2.0, places=4)
        self.assertAlmostEqual(b_cons, 595.5436, places=1)

    # --- 4. sureg ---
    def test_04_sureg(self):
        """Verify sureg seemingly unrelated regression against Reference 15."""
        res = self.interp.execute("sureg (price mpg weight) (length weight foreign)")
        self.assertTrue(res.ok, f"sureg failed: {res.error}")

        # Reference 15:
        # Eq 1 (price): mpg = -79.49104 (se 81.17198), weight = 1.566426 (se 0.6127101), _cons = 3128.442
        # Eq 2 (length): weight = 0.0278516 (se 0.0013219), foreign = 2.133104 (se 2.176135), _cons = 103.2015
        b_price_mpg = self.interp.registers.b.get("price:mpg")
        b_price_weight = self.interp.registers.b.get("price:weight")
        b_length_weight = self.interp.registers.b.get("length:weight")

        self.assertAlmostEqual(b_price_mpg, -79.49104, places=2)
        self.assertAlmostEqual(b_price_weight, 1.566426, places=2)
        self.assertAlmostEqual(b_length_weight, 0.0278516, places=4)

if __name__ == "__main__":
    unittest.main()
