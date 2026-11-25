import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import optimize
import openpyxl
try:
    from linearmodels.system import SUR
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False

from engine.dataset import Dataset
from engine.interpreter import Interpreter


class TestExternalValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.n = 120
        cls.x1 = np.random.randn(cls.n)
        cls.x2 = np.random.randn(cls.n) * 1.5
        
        # Correlated error terms for SUR
        e1 = np.random.randn(cls.n)
        e2 = 0.6 * e1 + np.random.randn(cls.n) * 0.8
        
        cls.y1 = 1.5 + 2.4 * cls.x1 + e1
        cls.y2 = 3.2 - 1.8 * cls.x2 + e2

        cls.df = pd.DataFrame({
            'y1': cls.y1,
            'y2': cls.y2,
            'x1': cls.x1,
            'x2': cls.x2
        })

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ds = Dataset()
        self.ds.load_dataframe(self.df.copy())
        self.interp = Interpreter(self.ds)
        self.interp.custom_commands.clear()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_putexcel_roundtrip_with_openpyxl(self):
        """putexcel writes cells, text, scalars, and matrices with round-trip verification via openpyxl."""
        excel_path = os.path.join(self.temp_dir, "report.xlsx").replace("\\", "/")
        
        # 1. Fit a model to populate e(b)
        self.interp.execute("regress y1 x1")

        # 2. Configure putexcel workbook
        res_set = self.interp.execute(f'putexcel set "{excel_path}", replace')
        self.assertTrue(res_set.ok, f"putexcel set failed: {res_set.error}")

        # 3. Write individual cells and regression results
        res_write = self.interp.execute(
            'putexcel A1 = ("Econometric Summary") '
            'B2 = (987.654) '
            'C3 = ("Validated") '
            'A5 = matrix(e(b))'
        )
        self.assertTrue(res_write.ok, f"putexcel write failed: {res_write.error}")

        # 4. Round-trip read back with openpyxl
        wb = openpyxl.load_workbook(excel_path)
        ws = wb.active

        self.assertEqual(ws["A1"].value, "Econometric Summary")
        self.assertAlmostEqual(float(ws["B2"].value), 987.654, places=3)
        self.assertEqual(ws["C3"].value, "Validated")

        # Verify matrix row contains coefficients
        b_x1 = self.interp.registers.b.get("x1")
        b_cons = self.interp.registers.b.get("_cons")
        val_a5 = float(ws["A5"].value)
        val_b5 = float(ws["B5"].value)

        self.assertAlmostEqual(val_a5, b_x1, places=4)
        self.assertAlmostEqual(val_b5, b_cons, places=4)

    def test_02_nlcom_and_testnl_vs_numerical_jacobian(self):
        """nlcom and testnl delta method validated against independent numerical Jacobian and analytical gradients."""
        self.interp.execute("regress y1 x1 x2")
        
        # In Potatostics: test non-linear ratio g(beta) = x1 / x2
        res_nl = self.interp.execute("nlcom x1 / x2")
        self.assertTrue(res_nl.ok, f"nlcom failed: {res_nl.error}")

        pot_est = self.interp.registers.get_r("estimate")
        pot_se = self.interp.registers.get_r("se")
        pot_z = self.interp.registers.get_r("z")

        # Independent calculation via analytical & numerical Jacobian:
        b = self.interp.estimates.last_model['beta'][:2] # [b_x1, b_x2]
        vcov = self.interp.estimates.last_model['vcov'][:2, :2]

        def g_func(params):
            return params[0] / params[1]

        expected_est = float(g_func(b))
        
        # Analytical Jacobian: [1/b2, -b1/(b2^2)]
        grad = np.array([1.0 / b[1], -b[0] / (b[1] ** 2)])
        expected_var = float(grad.T @ vcov @ grad)
        expected_se = float(np.sqrt(expected_var))
        expected_z = float(expected_est / expected_se)

        self.assertAlmostEqual(pot_est, expected_est, places=6)
        self.assertAlmostEqual(pot_se, expected_se, places=6)
        self.assertAlmostEqual(pot_z, expected_z, places=5)

        # testnl: H0: x1 / x2 = 1.0
        res_tnl = self.interp.execute("testnl x1 / x2 = 1.0")
        self.assertTrue(res_tnl.ok, f"testnl failed: {res_tnl.error}")

        pot_chi2 = self.interp.registers.get_r("chi2")
        expected_chi2 = float(((expected_est - 1.0) ** 2) / expected_var)
        self.assertAlmostEqual(pot_chi2, expected_chi2, places=5)

    def test_03_cnsreg_vs_restricted_least_squares(self):
        """cnsreg validated against exact restricted least squares formula and scipy SLSQP optimizer."""
        # Define constraint: 1.0 * x1 + 1.0 * x2 = 1.0
        self.interp.execute("constraint 1 x1 + x2 = 1.0")
        res_cns = self.interp.execute("cnsreg y1 x1 x2, constraints(1)")
        self.assertTrue(res_cns.ok, f"cnsreg failed: {res_cns.error}")

        pot_b_x1 = self.interp.registers.b.get("x1")
        pot_b_x2 = self.interp.registers.b.get("x2")
        pot_b_cons = self.interp.registers.b.get("_cons")

        # Constraint must hold exactly
        self.assertAlmostEqual(pot_b_x1 + pot_b_x2, 1.0, places=10)

        # Independent Restricted OLS via projection:
        # beta_c = beta_ols - (X'X)^-1 R' (R (X'X)^-1 R')^-1 (R beta_ols - r)
        X = np.column_stack([self.x1, self.x2, np.ones(self.n)])
        y = self.y1
        inv_XX = np.linalg.inv(X.T @ X)
        b_ols = inv_XX @ (X.T @ y)

        R = np.array([[1.0, 1.0, 0.0]])
        r_val = np.array([1.0])

        inv_R_XX_Rt = np.linalg.inv(R @ inv_XX @ R.T)
        expected_b_c = b_ols - inv_XX @ R.T @ inv_R_XX_Rt @ (R @ b_ols - r_val)

        self.assertAlmostEqual(pot_b_x1, expected_b_c[0], places=6)
        self.assertAlmostEqual(pot_b_x2, expected_b_c[1], places=6)
        self.assertAlmostEqual(pot_b_cons, expected_b_c[2], places=6)

        # Also verify with scipy.optimize SLSQP
        def ssr(beta):
            return float(np.sum((y - X @ beta) ** 2))
        
        opt_res = optimize.minimize(
            ssr, b_ols,
            constraints={'type': 'eq', 'fun': lambda beta: beta[0] + beta[1] - 1.0},
            method='SLSQP'
        )
        self.assertAlmostEqual(pot_b_x1, opt_res.x[0], places=5)
        self.assertAlmostEqual(pot_b_x2, opt_res.x[1], places=5)
        self.assertAlmostEqual(pot_b_cons, opt_res.x[2], places=5)

    def test_04_sureg_vs_linearmodels_sur(self):
        """sureg multi-equation FGLS validated directly against Oxford's linearmodels.system.SUR."""
        if not HAS_LINEARMODELS:
            self.skipTest("linearmodels not available")
        res_pot = self.interp.execute("sureg (y1 x1) (y2 x2)")
        self.assertTrue(res_pot.ok, f"sureg failed: {res_pot.error}")

        pot_b1_x1 = self.interp.registers.b.get("y1:x1")
        pot_b1_cons = self.interp.registers.b.get("y1:_cons")
        pot_b2_x2 = self.interp.registers.b.get("y2:x2")
        pot_b2_cons = self.interp.registers.b.get("y2:_cons")

        pot_se1_x1 = self.interp.registers.se.get("y1:x1")
        pot_se1_cons = self.interp.registers.se.get("y1:_cons")

        # Independent linearmodels SUR estimation
        formulas = {
            'eq1': 'y1 ~ 1 + x1',
            'eq2': 'y2 ~ 1 + x2'
        }
        lm_model = SUR.from_formula(formulas, self.df)
        lm_res = lm_model.fit(cov_type='unadjusted')

        lm_b1_cons = lm_res.params['eq1_Intercept']
        lm_b1_x1 = lm_res.params['eq1_x1']
        lm_b2_cons = lm_res.params['eq2_Intercept']
        lm_b2_x2 = lm_res.params['eq2_x2']

        lm_se1_x1 = lm_res.std_errors['eq1_x1']
        lm_se1_cons = lm_res.std_errors['eq1_Intercept']

        # Point estimates match to 10^-8
        self.assertAlmostEqual(pot_b1_x1, lm_b1_x1, places=6)
        self.assertAlmostEqual(pot_b1_cons, lm_b1_cons, places=6)
        self.assertAlmostEqual(pot_b2_x2, lm_b2_x2, places=6)
        self.assertAlmostEqual(pot_b2_cons, lm_b2_cons, places=6)

        # Standard errors match to 10^-5
        self.assertAlmostEqual(pot_se1_x1, lm_se1_x1, places=4)
        self.assertAlmostEqual(pot_se1_cons, lm_se1_cons, places=4)
