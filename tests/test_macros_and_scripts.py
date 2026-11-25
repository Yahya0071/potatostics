"""
tests/test_macros_and_scripts.py - Comprehensive Unit Tests for Phase 2C:
1. Macro evaluation & expansion (locals, globals, nested macros).
2. Return registers: r(), e(), _b[], _se[], _rc, return list, ereturn list.
3. Script prefixes: quietly, capture, noisily.
4. Script loops and flow: foreach (in and of varlist), forvalues, if/else blocks.
5. Comment stripping and line continuations (///, /* */, //, *).
6. Do-file execution via `do` and `run`.
"""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from engine.dataset import Dataset
from engine.interpreter import Interpreter
from engine.script_runner import strip_comments_and_join_continuations, parse_forvalues_range


class TestMacrosAndReturnRegisters(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'price': np.random.uniform(3000, 15000, n),
            'mpg': np.random.uniform(12, 40, n),
            'weight': np.random.uniform(1800, 4800, n),
            'rep78': np.random.choice([1, 2, 3, 4, 5], n),
            'rep_flag': np.random.choice([0, 1], n)
        })
        self.dataset = Dataset()
        self.dataset.load_dataframe(df)
        self.interpreter = Interpreter(self.dataset)

    def test_01_local_verbatim_and_expansion(self):
        self.interpreter.execute("local myvar price")
        self.assertEqual(self.interpreter.macro_mgr.get_local("myvar"), "price")
        res_sum = self.interpreter.execute("summarize `myvar'")
        self.assertTrue(res_sum.ok)
        self.assertIn("price", res_sum.output)
        res_di = self.interpreter.execute('display "`myvar\'"')
        self.assertIn("price", res_di.output)

    def test_02_local_expression_evaluation(self):
        self.interpreter.execute("local sum = 10 + 25 * 2")
        self.assertEqual(self.interpreter.macro_mgr.get_local("sum"), "60")

    def test_03_global_macro_and_expansion(self):
        self.interpreter.execute("global g_title Best Model")
        self.assertEqual(self.interpreter.macro_mgr.get_global("g_title"), "Best Model")
        res = self.interpreter.execute("display $g_title")
        self.assertIn("Best Model", res.output)

        res2 = self.interpreter.execute("display ${g_title}")
        self.assertIn("Best Model", res2.output)

    def test_04_nested_macro_expansion(self):
        self.interpreter.execute("local i 1")
        self.interpreter.execute("local var_1 mpg")
        # In `var_`i'', `i' expands first to 1, then `var_1' expands to mpg
        expanded = self.interpreter.macro_mgr.expand("`var_`i''")
        self.assertEqual(expanded, "mpg")

    def test_05_summarize_populates_r_registers(self):
        self.interpreter.execute("summarize price")
        regs = self.interpreter.registers
        self.assertEqual(regs.r_scalars.get('N'), 100)
        self.assertAlmostEqual(regs.r_scalars.get('mean'), self.dataset.df['price'].mean(), places=4)
        self.assertAlmostEqual(regs.r_scalars.get('min'), self.dataset.df['price'].min(), places=4)
        self.assertAlmostEqual(regs.r_scalars.get('max'), self.dataset.df['price'].max(), places=4)

        # return list formatting
        res_list = self.interpreter.execute("return list")
        self.assertIn("r(mean)", res_list.output)
        self.assertIn("r(N)", res_list.output)

    def test_06_regress_populates_e_registers_and_b_se(self):
        res = self.interpreter.execute("regress price mpg weight")
        self.assertTrue(res.ok)
        regs = self.interpreter.registers

        self.assertEqual(regs.e_scalars.get('N'), 100)
        self.assertIn('r2', regs.e_scalars)
        self.assertIn('F', regs.e_scalars)
        self.assertEqual(regs.e_macros.get('cmd'), 'regress')
        self.assertEqual(regs.e_macros.get('depvar'), 'price')

        # _b and _se access
        self.assertIn('mpg', regs.b)
        self.assertIn('mpg', regs.se)
        self.assertIn('_cons', regs.b)

        # ereturn list formatting
        res_elist = self.interpreter.execute("ereturn list")
        self.assertIn("e(r2)", res_elist.output)
        self.assertIn("e(N)", res_elist.output)

    def test_07_display_expressions_with_r_and_b(self):
        self.interpreter.execute("summarize mpg")
        res1 = self.interpreter.execute('display "Mean mpg is: " r(mean)')
        self.assertIn("Mean mpg is:", res1.output)

        self.interpreter.execute("regress price mpg")
        res2 = self.interpreter.execute('display "Beta: " _b[mpg] " / SE: " _se[mpg]')
        self.assertIn("Beta:", res2.output)
        self.assertIn("/ SE:", res2.output)

    def test_08_expression_referencing_registers_in_generate(self):
        self.interpreter.execute("summarize price")
        self.interpreter.execute("generate price_dev = price - r(mean)")
        self.assertIn('price_dev', self.dataset.df.columns)
        self.assertAlmostEqual(self.dataset.df['price_dev'].mean(), 0.0, places=4)

    def test_09_capture_prefix_traps_errors(self):
        # Without capture, referencing non-existent variable yields error
        res_err = self.interpreter.execute("regress price non_existent_var")
        self.assertFalse(res_err.ok)

        # With capture, command succeeds silently and records _rc
        res_cap = self.interpreter.execute("capture regress price non_existent_var")
        self.assertTrue(res_cap.ok)
        self.assertEqual(self.interpreter.registers.rc, 111)

        # Reference _rc in expression
        res_rc = self.interpreter.execute("display _rc")
        self.assertIn("111", res_rc.output)

    def test_10_quietly_prefix_silences_output(self):
        res = self.interpreter.execute("quietly summarize price")
        self.assertTrue(res.ok)
        self.assertEqual(res.output, "")
        # But registers are still populated!
        self.assertEqual(self.interpreter.registers.r_scalars.get('N'), 100)


class TestScriptRunnerAndControlFlow(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'x3': np.random.randn(n),
            'y': np.random.randn(n)
        })
        self.dataset = Dataset()
        self.dataset.load_dataframe(df)
        self.interpreter = Interpreter(self.dataset)

    def test_11_comment_stripping_and_continuations(self):
        script = """
        * This is a comment line
        // Another comment
        summarize x1 ///
            x2 x3
        /* Multiline
           comment
        */
        count
        """
        lines = strip_comments_and_join_continuations(script)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0][1], "summarize x1 x2 x3")
        self.assertEqual(lines[1][1], "count")

    def test_12_forvalues_range_parsing(self):
        self.assertEqual(parse_forvalues_range("1/4"), [1, 2, 3, 4])
        self.assertEqual(parse_forvalues_range("1(2)7"), [1, 3, 5, 7])
        self.assertEqual(parse_forvalues_range("10(-2)4"), [10, 8, 6, 4])

    def test_13_foreach_in_loop(self):
        script = """
        foreach v in x1 x2 {
            gen `v'_sq = `v' * `v'
        }
        """
        res = self.interpreter.script_runner.run_script(script)
        self.assertTrue(res.ok)
        self.assertIn('x1_sq', self.dataset.df.columns)
        self.assertIn('x2_sq', self.dataset.df.columns)

    def test_14_foreach_of_varlist_loop(self):
        script = """
        foreach v of varlist x* {
            gen `v'_abs = abs(`v')
        }
        """
        res = self.interpreter.script_runner.run_script(script)
        self.assertTrue(res.ok)
        self.assertIn('x1_abs', self.dataset.df.columns)
        self.assertIn('x2_abs', self.dataset.df.columns)
        self.assertIn('x3_abs', self.dataset.df.columns)

    def test_15_forvalues_loop(self):
        script = """
        forvalues i = 1/3 {
            gen z`i' = `i' * 10
        }
        """
        res = self.interpreter.script_runner.run_script(script)
        self.assertTrue(res.ok)
        self.assertIn('z1', self.dataset.df.columns)
        self.assertIn('z2', self.dataset.df.columns)
        self.assertIn('z3', self.dataset.df.columns)
        self.assertEqual(self.dataset.df['z3'].iloc[0], 30)

    def test_16_if_else_control_block(self):
        script = """
        local flag 1
        if `flag' == 1 {
            gen created_by_if = 100
        }
        else {
            gen created_by_else = 200
        }
        """
        res = self.interpreter.script_runner.run_script(script)
        self.assertTrue(res.ok)
        self.assertIn('created_by_if', self.dataset.df.columns)
        self.assertNotIn('created_by_else', self.dataset.df.columns)

    def test_17_do_file_execution(self):
        with tempfile.NamedTemporaryFile('w', suffix='.do', delete=False) as f:
            f.write("""
            * Sample Analysis Do-file
            gen y_scaled = y * 100
            regress y_scaled x1 x2
            local my_r2 = e(r2)
            display "R-squared from do-file is: " `my_r2'
            """)
            temp_path = f.name

        try:
            res = self.interpreter.execute(f'do "{temp_path}"')
            self.assertTrue(res.ok)
            self.assertIn('y_scaled', self.dataset.df.columns)
            self.assertIn("R-squared from do-file is:", res.output)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == '__main__':
    unittest.main()
