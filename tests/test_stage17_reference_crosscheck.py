"""
tests/test_stage17_reference_crosscheck.py - Reference Output Cross-Check and Parser Verification
Cross-checks Potatostics regression, logit cluster, and xtreg fe results against
official Reference log files placed in tests/data/reference_outputs/.
"""

import os
import glob
import unittest

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.reference_log_parser import parse_reference_log, compare_models_to_reference

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REFERENCE_OUTPUTS_DIR = os.path.join(DATA_DIR, "reference_outputs")
AUTO_PATH = os.path.join(DATA_DIR, "auto.dta")


SAMPLE_REFERENCE_LOG = """
------------------------------------------------------------------------------------------------------------------------
      name:  <unnamed>
       log:  auto_benchmarks.log
  log type:  text
------------------------------------------------------------------------------------------------------------------------

. regress price mpg weight

      Source |       SS           df       MS      Number of obs   =        74
-------------+----------------------------------   F(2, 71)        =     14.74
       Model |   186321280         2  93160640.1   Prob > F        =    0.0000
    Residual |   448744116        71  6320339.67   R-squared       =    0.2934
-------------+----------------------------------   Adj R-squared   =    0.2735
       Total |   635065396        73  8699525.97   Root MSE        =    2514.0

------------------------------------------------------------------------------
       price | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
         mpg |   -49.51222    86.15604    -0.57   0.567    -221.3025     122.278
      weight |    1.746559    .6413538     2.72   0.008     .4677402    3.025378
       _cons |    1946.069    3597.05      0.54   0.590    -5226.245    9118.382
------------------------------------------------------------------------------

. logit foreign mpg weight, vce(cluster rep78)

Iteration 0:  Log pseudolikelihood = -42.401149  
Iteration 1:  Log pseudolikelihood = -25.267793  
Iteration 2:  Log pseudolikelihood = -22.784403  
Iteration 3:  Log pseudolikelihood = -22.678438  
Iteration 4:  Log pseudolikelihood = -22.678037  
Iteration 5:  Log pseudolikelihood = -22.678037  

Logistic regression                                     Number of obs =     69
                                                        Wald chi2(2)  =  47.52
                                                        Prob > chi2   = 0.0000
Log pseudolikelihood = -22.678037                       Pseudo R2     = 0.4652

                                (Std. err. adjusted for 5 clusters in rep78)
------------------------------------------------------------------------------
             |               Robust
     foreign | Coefficient  std. err.      z    P>|z|     [95% conf. interval]
-------------+----------------------------------------------------------------
         mpg |   -.1524337    .0271167    -5.62   0.000    -.2055815   -.0992859
      weight |   -.0042459    .0008051    -5.27   0.000    -.0058238    -.002668
       _cons |    14.21691    2.031818     7.00   0.000     10.23462    18.19921
------------------------------------------------------------------------------

. xtset rep78
panel variable:  rep78 (unbalanced)

. xtreg price mpg weight, fe

Fixed-effects (within) regression               Number of obs     =         69
Group variable: rep78                           Number of groups  =          5

R-squared:                                      Obs per group:
     within  = 0.3591                           min =          2
     between = 0.0002                           avg =       13.8
     overall = 0.2994                           max =         30

                                                F(2,62)          =      17.37
corr(u_i, Xb)  =  -0.8325                        Prob > F          =     0.0000

------------------------------------------------------------------------------
       price | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
-------------+----------------------------------------------------------------
         mpg |   -63.09709     87.4528    -0.72   0.473    -237.9127    111.7185
      weight |    2.093078    .6369062     3.29   0.002     .8199216    3.366235
       _cons |    1143.134    3565.726     0.32   0.750     -5984.65    8270.918
------------------------------------------------------------------------------
     sigma_u |   1110.2497
     sigma_e |    2424.057
         rho |   .17335689   (fraction of variance due to u_i)
------------------------------------------------------------------------------
F test that all u_i=0: F(4, 62) = 1.66                       Prob > F = 0.1707
------------------------------------------------------------------------------
"""


class TestStage17ReferenceCrossCheck(unittest.TestCase):
    def setUp(self):
        self.ds = Dataset()
        if os.path.exists(AUTO_PATH):
            self.ds.load_dta(AUTO_PATH)
        self.interp = Interpreter(self.ds)

    def test_01_reference_log_parser_and_sample_crosscheck(self):
        """Verify parser accurately extracts tables and compares to 4 decimals."""
        models = parse_reference_log(SAMPLE_REFERENCE_LOG)
        self.assertIn("regress", models)
        self.assertIn("logit", models)
        self.assertIn("xtreg", models)

        # 1. Regress cross-check
        reg_tbl = models["regress"]["table"]
        self.assertAlmostEqual(reg_tbl["mpg"]["coef"], -49.51222, places=4)
        self.assertAlmostEqual(reg_tbl["mpg"]["se"], 86.15604, places=4)
        self.assertAlmostEqual(reg_tbl["weight"]["coef"], 1.746559, places=4)
        self.assertAlmostEqual(reg_tbl["weight"]["se"], 0.6413538, places=4)
        self.assertAlmostEqual(reg_tbl["_cons"]["coef"], 1946.069, places=3)

        # 2. Compare against Potatostics execution
        r_pot = self.interp.execute("regress price mpg weight")
        self.assertTrue(r_pot.ok)
        diffs = compare_models_to_reference(self.interp, reg_tbl, tolerance=1e-4)
        self.assertEqual(diffs, [], f"Discrepancies found in regress: {diffs}")

        # 3. Clustered logit comparison
        logit_tbl = models["logit"]["table"]
        r_pot_logit = self.interp.execute("logit foreign mpg weight, vce(cluster rep78)")
        self.assertTrue(r_pot_logit.ok)
        diffs_logit = compare_models_to_reference(self.interp, logit_tbl, tolerance=1e-4)
        self.assertEqual(diffs_logit, [], f"Discrepancies found in logit: {diffs_logit}")

        # 4. xtreg fe comparison
        xt_tbl = models["xtreg"]["table"]
        self.interp.execute("xtset rep78")
        r_pot_xt = self.interp.execute("xtreg price mpg weight, fe")
        self.assertTrue(r_pot_xt.ok)
        diffs_xt = compare_models_to_reference(self.interp, xt_tbl, tolerance=1e-4)
        self.assertEqual(diffs_xt, [], f"Discrepancies found in xtreg: {diffs_xt}")

    def test_02_filesystem_reference_log_crosscheck_or_skip(self):
        """Check for external Reference log in tests/data/reference_outputs/; skip if absent."""
        log_files = glob.glob(os.path.join(REFERENCE_OUTPUTS_DIR, "*.log")) + glob.glob(os.path.join(REFERENCE_OUTPUTS_DIR, "*.txt"))
        if not log_files:
            self.skipTest("No Reference output log found in tests/data/reference_outputs/. See tests/data/reference_outputs/README.md for instructions.")

        for log_path in log_files:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            models = parse_reference_log(content)
            if not models:
                continue

            if "regress" in models:
                r_pot = self.interp.execute("regress price mpg weight")
                if r_pot.ok:
                    diffs = compare_models_to_reference(self.interp, models["regress"]["table"], tolerance=1e-4)
                    self.assertEqual(diffs, [], f"Reference log {os.path.basename(log_path)} regress mismatch: {diffs}")

            if "logit" in models:
                r_pot = self.interp.execute("logit foreign mpg weight, vce(cluster rep78)")
                if r_pot.ok:
                    diffs = compare_models_to_reference(self.interp, models["logit"]["table"], tolerance=1e-4)
                    self.assertEqual(diffs, [], f"Reference log {os.path.basename(log_path)} logit mismatch: {diffs}")

            if "xtreg" in models:
                self.interp.execute("xtset rep78")
                r_pot = self.interp.execute("xtreg price mpg weight, fe")
                if r_pot.ok:
                    diffs = compare_models_to_reference(self.interp, models["xtreg"]["table"], tolerance=1e-4)
                    self.assertEqual(diffs, [], f"Reference log {os.path.basename(log_path)} xtreg mismatch: {diffs}")


if __name__ == "__main__":
    unittest.main()
