"""
tests/test_results_console_export.py

Permanent regression test suite for Potatostics publication-quality table exports:
- Verifies dynamic LaTeX (booktabs) and Markdown table generation.
- Enforces structural table boundary detection and isolation.
- Prevents unrelated statistical tables (e.g. ANOVA + Coefficients) from being merged.
- Protects borderless two-column and multi-column tables from value loss.
- Rejects non-table text with pipes (e.g. 'Status | pending') from being formatted as tables.
- Preserves exact statistical numerical fidelity, variable names, and missing-value notation.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from PySide6 import QtWidgets, QtGui

from engine.interpreter import Interpreter
from engine.dataset import Dataset
from gui_qt.console import (
    extract_table_blocks_from_text,
    is_valid_table_block,
    extract_diagnostics_from_text,
    format_table_as_latex,
    format_table_as_markdown,
    format_full_report_as_latex,
    format_full_report_as_markdown,
    ResultsConsole,
    StatisticalResultRecord,
    StatisticalReport
)

app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(["--platform", "offscreen"])


class TestResultsConsoleExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate genuine interpreter outputs for fixtures
        cls.ds = Dataset()
        df = pd.DataFrame({
            "price": [5000, 6000, 4500, 7000, 8000, 5500, 9000],
            "mpg": [20, 25, 18, 30, 15, 22, 14],
            "weight": [3000, 2500, 3200, 2200, 3800, 2900, 4000],
            "foreign": [0, 1, 0, 1, 0, 0, 1],
            "wage_usd": [15.5, 22.0, 18.2, 35.0, 28.4, 19.1, 40.2],
            "tax_%": [5.0, 7.5, 5.0, 10.0, 8.5, 6.0, 12.0],
            "cost_$": [100, 150, 120, 250, 200, 130, 300]
        })
        cls.ds.load_dataframe(df)
        cls.interp = Interpreter(cls.ds)

        cls.out_sum = cls.interp.execute("summarize price mpg weight").output
        cls.out_tab = cls.interp.execute("tabulate foreign").output
        cls.out_reg = cls.interp.execute("regress price mpg weight").output
        cls.out_robust = cls.interp.execute("regress price mpg weight, robust").output
        cls.out_reg2 = cls.interp.execute("regress price mpg").output

    # =========================================================================
    # Test 1: Normal Two-Column Table
    # =========================================================================
    def test_01_normal_two_column_table(self):
        """Verifies standard 2-column table with borders formats with exactly 2 columns."""
        table_text = (
            "Metric         | Value\n"
            "---------------+------\n"
            "Observations   | 100\n"
            "R-squared      | 0.854\n"
            "---------------+------\n"
        )
        ltx = format_table_as_latex(table_text)
        self.assertIsNotNone(ltx)
        self.assertIn("\\begin{tabular}{lr}", ltx)
        self.assertIn("Observations & 100 \\\\", ltx)
        self.assertIn("R-squared & 0.854 \\\\", ltx)

        md = format_table_as_markdown(table_text)
        self.assertIsNotNone(md)
        self.assertIn("| Metric | Value |", md)
        self.assertIn("| Observations | 100 |", md)

    # =========================================================================
    # Test 2: Borderless Two-Column Table (Defect B)
    # =========================================================================
    def test_02_borderless_two_column_table(self):
        """
        Verifies borderless 2-column tables retain both columns and values
        without discarding the numeric column (Defect B fix).
        """
        borderless_text = (
            "variable | mean\n"
            "x        | 1.23\n"
            "y        | 4.56\n"
        )
        ltx = format_table_as_latex(borderless_text)
        self.assertIsNotNone(ltx, "Borderless table must not be rejected")
        self.assertIn("\\begin{tabular}{lr}", ltx, "Must declare 2 columns (l + r)")
        self.assertIn("variable & mean \\\\", ltx)
        self.assertIn("x & 1.23 \\\\", ltx, "Numeric value '1.23' must not be dropped")
        self.assertIn("y & 4.56 \\\\", ltx, "Numeric value '4.56' must not be dropped")

        md = format_table_as_markdown(borderless_text)
        self.assertIsNotNone(md)
        self.assertIn("| variable | mean |", md)
        self.assertIn("| x | 1.23 |", md)
        self.assertIn("| y | 4.56 |", md)

    # =========================================================================
    # Test 3: Three-Column Descriptive Table
    # =========================================================================
    def test_03_three_column_descriptive_table(self):
        """Verifies genuine tabulate output formats with 4 columns (stub + 3 data)."""
        ltx = format_table_as_latex(self.out_tab)
        self.assertIsNotNone(ltx)
        self.assertIn("\\begin{tabular}{lrrr}", ltx)
        self.assertIn("foreign & Freq. & Percent & Cum. \\\\", ltx)
        self.assertIn("0 & 4 & 57.14 & 57.14 \\\\", ltx)
        self.assertIn("Total & 7 & 100.00", ltx)

        md = format_table_as_markdown(self.out_tab)
        self.assertIsNotNone(md)
        self.assertIn("| foreign | Freq. | Percent | Cum. |", md)
        self.assertIn("| 0 | 4 | 57.14 | 57.14 |", md)

    # =========================================================================
    # Test 4: Six/Seven-Column Regression Table
    # =========================================================================
    def test_04_six_column_regression_table(self):
        """Verifies robust regression table formats with 7 total columns."""
        ltx = format_table_as_latex(self.out_robust)
        self.assertIsNotNone(ltx)
        self.assertIn("\\begin{tabular}{lrrrrrr}", ltx)
        self.assertIn("price & Coefficient & Robust std. err. & t & P>|t|", ltx)
        self.assertIn("mpg & 1034.0161 & 132.3453 & 7.81 & 0.001", ltx)
        self.assertIn("\\_cons & -45918.6931 & 6628.9118 & -6.93 & 0.002", ltx)

    # =========================================================================
    # Test 5: Confidence Interval with Two Numerical Endpoints
    # =========================================================================
    def test_05_confidence_interval_endpoints(self):
        """Verifies CI header spans 2 columns in LaTeX and has [Low] / [High] in Markdown."""
        ltx = format_table_as_latex(self.out_robust)
        self.assertIn("\\multicolumn{2}{c}{[95\\% conf. interval]}", ltx)
        self.assertIn("666.5667 & 1401.4656 \\\\", ltx)

        md = format_table_as_markdown(self.out_robust)
        self.assertIn("[95% conf. interval] [Low] | [95% conf. interval] [High]", md)
        self.assertIn("666.5667 | 1401.4656 |", md)

    # =========================================================================
    # Test 6: Full OLS Output Containing ANOVA and Coefficient Sections (Defect A)
    # =========================================================================
    def test_06_full_ols_table_boundary_isolation(self):
        """
        Verifies full regression output generates two distinct, independent tables
        (ANOVA and Coefficient) without merging unrelated columns (Defect A fix).
        """
        ltx = format_table_as_latex(self.out_reg)
        self.assertIsNotNone(ltx)

        # Must have exactly 2 independent table environments
        self.assertEqual(ltx.count("\\begin{table}[htbp]"), 2)
        self.assertEqual(ltx.count("\\end{table}"), 2)

        # Table 1: ANOVA (4 columns: Source, SS, df, MS)
        self.assertIn("\\begin{tabular}{lrrr}", ltx)
        self.assertIn("Source & SS & df & MS \\\\", ltx)
        self.assertIn("Model &", ltx)
        self.assertIn("Residual &", ltx)
        self.assertIn("Total &", ltx)

        # Table 2: Coefficients (7 columns)
        self.assertIn("\\begin{tabular}{lrrrrrr}", ltx)
        self.assertIn("price & Coefficient & Std. err. & t & P>|t|", ltx)
        self.assertIn("mpg & 1034.0161 & 186.1726 & 5.55 & 0.005", ltx)

        # Markdown: Must have 2 distinct markdown tables separated by blank lines
        md = format_table_as_markdown(self.out_reg)
        self.assertIsNotNone(md)
        self.assertIn("| Source | SS | df | MS |", md)
        self.assertIn("| price | Coefficient | Std. err. | t | P>\\|t\\| |", md)

    # =========================================================================
    # Test 7: Selected Coefficient Section Only
    # =========================================================================
    def test_07_selected_coefficient_section_only(self):
        """Verifies selecting only the coefficient block exports only that single table."""
        coef_lines = [l for l in self.out_reg.splitlines() if "price" in l or "mpg" in l or "weight" in l or "_cons" in l or "---" in l]
        coef_text = "\n".join(coef_lines)

        ltx = format_table_as_latex(coef_text)
        self.assertIsNotNone(ltx)
        self.assertEqual(ltx.count("\\begin{table}[htbp]"), 1)
        self.assertNotIn("Source", ltx)
        self.assertIn("mpg & 1034.0161", ltx)

    # =========================================================================
    # Test 8: Multiple Commands in Console History
    # =========================================================================
    def test_08_multiple_commands_in_console_history(self):
        """Verifies console history with multiple commands keeps tables distinct."""
        history = f". summarize price mpg\n{self.out_sum}\n\n. regress price mpg weight\n{self.out_reg}"
        blocks = extract_table_blocks_from_text(history)
        # Summarize + ANOVA + Coef = 3 distinct valid table blocks
        self.assertTrue(len(blocks) >= 3)

        ltx = format_table_as_latex(history)
        self.assertEqual(ltx.count("\\begin{table}[htbp]"), 3)

    # =========================================================================
    # Test 9: Plain Text Containing Pipes (Defect C)
    # =========================================================================
    def test_09_plain_text_pipe_rejection(self):
        """
        Verifies ordinary sentences with pipes are rejected as tables (Defect C fix).
        """
        bad_texts = [
            "Status | pending",
            "Notes | The study was conducted in 2024.",
            "| Single cell text |",
            "Diagnostic output: | all clear |",
            "Key | Value\nSentence without pipes.",
        ]
        for bad in bad_texts:
            self.assertIsNone(format_table_as_latex(bad), f"Must reject: {bad}")
            self.assertIsNone(format_table_as_markdown(bad), f"Must reject: {bad}")

    # =========================================================================
    # Test 10: Special Characters in Variable Names
    # =========================================================================
    def test_10_special_characters_escaping(self):
        """Verifies _, %, &, #, $, {, }, ~, ^ are escaped cleanly in single pass."""
        table_with_special = (
            "Variable      | Coef.   Std. err.\n"
            "--------------+------------------\n"
            "wage_usd      | 15.50   1.20\n"
            "tax_%         |  2.50   0.30\n"
            "cost_$        | 99.00   5.00\n"
            "firm_&_co     |  1.10   0.15\n"
            "tag_#1        |  0.80   0.05\n"
            "group_{A}     |  3.20   0.40\n"
            "path\\_var     |  4.10   0.50\n"
        )
        ltx = format_table_as_latex(table_with_special)
        self.assertIsNotNone(ltx)
        self.assertIn("wage\\_usd & 15.50", ltx)
        self.assertIn("tax\\_\\% & 2.50", ltx)
        self.assertIn("cost\\_\\$ & 99.00", ltx)
        self.assertIn("firm\\_\\&\\_co & 1.10", ltx)
        self.assertIn("tag\\_\\#1 & 0.80", ltx)
        self.assertIn("group\\_\\{A\\} & 3.20", ltx)

        # Markdown pipes in P>|t| must be escaped
        md = format_table_as_markdown(self.out_robust)
        self.assertIn("P>\\|t\\|", md)

    # =========================================================================
    # Test 11: Missing Values
    # =========================================================================
    def test_11_missing_values_preservation(self):
        """Verifies '.' and 'NA' missing notation is preserved in table exports."""
        table_with_missing = (
            "Variable | Coef.   Std. err.\n"
            "---------+------------------\n"
            "x1       | 10.50   1.20\n"
            "x2       |     .      .\n"
            "x3       |    NA     NA\n"
        )
        ltx = format_table_as_latex(table_with_missing)
        self.assertIsNotNone(ltx)
        self.assertIn("x2 & . & . \\\\", ltx)
        self.assertIn("x3 & NA & NA \\\\", ltx)

        md = format_table_as_markdown(table_with_missing)
        self.assertIn("| x2 | . | . |", md)
        self.assertIn("| x3 | NA | NA |", md)

    # =========================================================================
    # Test 12: Empty Selections
    # =========================================================================
    def test_12_empty_selections(self):
        """Verifies empty and whitespace strings return None without raising exceptions."""
        self.assertIsNone(format_table_as_latex(""))
        self.assertIsNone(format_table_as_latex("   \n\n  "))
        self.assertIsNone(format_table_as_markdown(""))
        self.assertIsNone(format_table_as_markdown(None))

    # =========================================================================
    # Test 13: Partial Table Selection
    # =========================================================================
    def test_13_partial_table_selection(self):
        """Verifies a subset of data rows (e.g. 2 variables) exports correctly."""
        partial_text = (
            "Variable | Obs    Mean\n"
            "---------+------------\n"
            "price    |  74 6165.25\n"
            "mpg      |  74   21.30\n"
        )
        ltx = format_table_as_latex(partial_text)
        self.assertIsNotNone(ltx)
        self.assertIn("price & 74 & 6165.25 \\\\", ltx)
        self.assertIn("mpg & 74 & 21.30 \\\\", ltx)

    # =========================================================================
    # Test 14: Column-Count Consistency
    # =========================================================================
    def test_14_column_count_consistency(self):
        """Verifies every line in the rendered tabular has matching column counts."""
        ltx = format_table_as_latex(self.out_sum)
        lines = [l for l in ltx.splitlines() if "&" in l]
        # Summarize has 6 columns (stub + 5 data)
        for line in lines:
            col_count = len(line.split("&"))
            self.assertEqual(col_count, 6, f"Row should have 6 columns: {line}")

    # =========================================================================
    # Test 15: Numerical Fidelity Between Interpreter and Exported Table
    # =========================================================================
    def test_15_numerical_fidelity(self):
        """Verifies exact bit-for-bit numerical parity between interpreter and exports."""
        ltx = format_table_as_latex(self.out_reg)
        self.assertIn("1034.0161", ltx)
        self.assertIn("186.1726", ltx)
        self.assertIn("5.55", ltx)
        self.assertIn("0.005", ltx)
        self.assertIn("517.1182", ltx)
        self.assertIn("1550.9141", ltx)
        self.assertIn("-45918.6931", ltx)

        md = format_table_as_markdown(self.out_reg)
        self.assertIn("1034.0161", md)
        self.assertIn("-45918.6931", md)

    # =========================================================================
    # Test 16: LaTeX Compilation Verification Gate
    # =========================================================================
    def test_16_latex_compiler_verification_gate(self):
        """
        Attempts LaTeX compilation if pdflatex is installed;
        otherwise marks compilation NOT VERIFIED truthfully.
        """
        pdflatex_path = shutil.which("pdflatex")
        if not pdflatex_path:
            # Documented truthfulness: mark NOT VERIFIED when compiler is absent
            print("\n[NOTE] pdflatex not found on host: LaTeX compilation marked NOT VERIFIED (skipped).")
            self.skipTest("pdflatex not available in this test environment")

        import tempfile
        import os
        ltx_table = format_table_as_latex(self.out_reg)
        doc = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\begin{document}\n"
            f"{ltx_table}\n"
            "\\end{document}\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_file = os.path.join(tmpdir, "table_test.tex")
            with open(tex_file, "w", encoding="utf-8") as f:
                f.write(doc)
            res = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "table_test.tex"],
                cwd=tmpdir,
                capture_output=True,
                text=True
            )
            self.assertEqual(res.returncode, 0, f"LaTeX compilation failed: {res.stdout}")

    # =========================================================================
    # Test 17: Ambiguous Table Structure Rejection (Schema Mismatch)
    # =========================================================================
    def test_17_ambiguous_table_structure_rejection(self):
        """
        Verifies ambiguous schemas such as 'a | b \n x | 1 2 3' are strictly rejected
        and neither silently padded with empty columns nor silently discarded.
        """
        ambig_text = "a | b\nx | 1 2 3"
        self.assertFalse(is_valid_table_block(ambig_text.splitlines()))
        self.assertIsNone(format_table_as_latex(ambig_text))
        self.assertIsNone(format_table_as_markdown(ambig_text))

        # Even with borders, mismatched token counts must be rejected
        ambig_bordered = (
            "a   | b\n"
            "----+------\n"
            "x   | 1 2 3\n"
        )
        self.assertFalse(is_valid_table_block(ambig_bordered.splitlines()))
        self.assertIsNone(format_table_as_latex(ambig_bordered))
        self.assertIsNone(format_table_as_markdown(ambig_bordered))

    # =========================================================================
    # Test 18: Inconsistent Data Row Column Counts Rejection
    # =========================================================================
    def test_18_inconsistent_data_rows_rejection(self):
        """Verifies tables with inconsistent column counts across data rows are rejected."""
        inconsistent_text = (
            "Variable | Stat1 Stat2\n"
            "---------+------------\n"
            "x        | 1.0   2.0\n"
            "y        | 1.0   2.0   3.0\n"
        )
        self.assertFalse(is_valid_table_block(inconsistent_text.splitlines()))
        self.assertIsNone(format_table_as_latex(inconsistent_text))
        self.assertIsNone(format_table_as_markdown(inconsistent_text))

    # =========================================================================
    # Test 19: Full Model Report Diagnostics Preservation (Classical OLS)
    # =========================================================================
    def test_19_full_model_report_diagnostics_classical_ols(self):
        """
        Verifies Full Model Report export retains all model diagnostics displayed
        beside the ANOVA table in classical OLS without altering coefficient estimates.
        """
        ltx = format_full_report_as_latex(self.out_reg)
        self.assertIsNotNone(ltx)
        self.assertIn("Model Diagnostics", ltx)
        self.assertIn("Number of obs", ltx)
        self.assertIn("Prob > F", ltx)
        self.assertIn("R-squared", ltx)
        self.assertIn("Adj R-squared", ltx)
        self.assertIn("Root MSE", ltx)

        # Numerical fidelity in diagnostics
        self.assertIn("21.69", ltx)
        self.assertIn("0.8734", ltx)

        # Markdown full report
        md = format_full_report_as_markdown(self.out_reg)
        self.assertIsNotNone(md)
        self.assertIn("### Model Diagnostics", md)
        self.assertIn("| Number of obs |", md)
        self.assertIn("| R-squared |", md)
        self.assertIn("| Adj R-squared |", md)
        self.assertIn("| 21.69 |", md)

    # =========================================================================
    # Test 20: Classical vs Robust Model Diagnostics Distinction
    # =========================================================================
    def test_20_classical_vs_robust_model_diagnostics(self):
        """
        Verifies that robust regression omits Adj R-squared in compliance with robust VCE conventions,
        while retaining F, Prob > F, R-squared, and Root MSE.
        """
        ltx = format_full_report_as_latex(self.out_robust)
        self.assertIsNotNone(ltx)
        self.assertIn("Model Diagnostics", ltx)
        self.assertIn("Number of obs", ltx)
        self.assertIn("R-squared", ltx)
        self.assertIn("Root MSE", ltx)
        self.assertNotIn("Adj R-squared", ltx)

        md = format_full_report_as_markdown(self.out_robust)
        self.assertIsNotNone(md)
        self.assertIn("### Model Diagnostics", md)
        self.assertNotIn("Adj R-squared", md)

    # =========================================================================
    # Test 21: Model Diagnostics Regex Extraction Parity
    # =========================================================================
    def test_21_extract_diagnostics_regex_accuracy(self):
        """Verifies exact extraction of all diagnostics from formatted output."""
        diags_classical = extract_diagnostics_from_text(self.out_reg)
        diag_keys_classical = [k for k, v in diags_classical]
        self.assertIn("Number of obs", diag_keys_classical)
        self.assertIn("R-squared", diag_keys_classical)
        self.assertIn("Adj R-squared", diag_keys_classical)
        self.assertIn("Root MSE", diag_keys_classical)

        diags_robust = extract_diagnostics_from_text(self.out_robust)
        diag_keys_robust = [k for k, v in diags_robust]
        self.assertIn("Number of obs", diag_keys_robust)
        self.assertIn("R-squared", diag_keys_robust)
        self.assertNotIn("Adj R-squared", diag_keys_robust)

    # =========================================================================
    # Test 22: Export Scope Distinction (Coefficient Table vs Full Report)
    # =========================================================================
    def test_22_export_scope_distinction(self):
        """
        Verifies clear boundary between 'Coefficient Table Only' and 'Full Model Report'.
        Coefficient-only export must NOT contain Model Diagnostics.
        Full report export MUST contain Model Diagnostics.
        """
        ltx_coef = format_table_as_latex(self.out_reg, full_report=False)
        self.assertIsNotNone(ltx_coef)
        self.assertNotIn("Model Diagnostics", ltx_coef)

        ltx_full = format_full_report_as_latex(self.out_reg)
        self.assertIsNotNone(ltx_full)
        self.assertIn("Model Diagnostics", ltx_full)

        md_coef = format_table_as_markdown(self.out_reg, full_report=False)
        self.assertIsNotNone(md_coef)
        self.assertNotIn("### Model Diagnostics", md_coef)

        md_full = format_full_report_as_markdown(self.out_reg)
        self.assertIsNotNone(md_full)
        self.assertIn("### Model Diagnostics", md_full)

    # =========================================================================
    # Test 23: Consecutive Regressions Model-Scoped Diagnostics (Scenario B)
    # =========================================================================
    def test_23_consecutive_regressions_diagnostics_isolation(self):
        """
        Verifies that when two regressions run consecutively in ResultsConsole:
        - Selecting Model 2 table exports Model 2's diagnostics (R2=0.1009, F=0.56, Root MSE=1707.4833),
          NEVER Model 1's diagnostics (R2=0.9156, F=21.69).
        - Selecting Model 1 table exports Model 1's diagnostics (R2=0.9156, F=21.69),
          NEVER Model 2's diagnostics.
        """
        console = ResultsConsole()
        console.append_command("regress price mpg weight", execution_id=1)
        console.append_output(self.out_reg, execution_id=1, command_text="regress price mpg weight")

        console.append_command("regress price mpg", execution_id=2)
        console.append_output(self.out_reg2, execution_id=2, command_text="regress price mpg")

        # Select Model 2's output
        rec2 = console._execution_records[1]
        cursor = console.textCursor()
        cursor.setPosition(rec2.char_start)
        cursor.setPosition(rec2.char_end, QtGui.QTextCursor.MoveMode.KeepAnchor)
        console.setTextCursor(cursor)

        records, target_text, is_sel = console._resolve_target_records()
        self.assertTrue(is_sel)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].command_text, "regress price mpg")

        # Copy full report for Model 2
        ltx_m2 = format_full_report_as_latex(target_text, extra_diagnostics=records[0].diagnostics)
        self.assertIsNotNone(ltx_m2)
        self.assertIn("0.1009", ltx_m2, "Model 2 R-squared (0.1009) must be present")
        self.assertIn("1707.4833", ltx_m2, "Model 2 Root MSE (1707.4833) must be present")
        self.assertNotIn("0.9156", ltx_m2, "Model 1 R-squared (0.9156) must NOT contaminate Model 2")
        self.assertNotIn("21.69", ltx_m2, "Model 1 F-stat (21.69) must NOT contaminate Model 2")

        # Select Model 1's output
        rec1 = console._execution_records[0]
        cursor.setPosition(rec1.char_start)
        cursor.setPosition(rec1.char_end, QtGui.QTextCursor.MoveMode.KeepAnchor)
        console.setTextCursor(cursor)

        records1, target_text1, is_sel1 = console._resolve_target_records()
        self.assertTrue(is_sel1)
        self.assertEqual(len(records1), 1)
        self.assertEqual(records1[0].command_text, "regress price mpg weight")

        ltx_m1 = format_full_report_as_latex(target_text1, extra_diagnostics=records1[0].diagnostics)
        self.assertIsNotNone(ltx_m1)
        self.assertIn("0.9156", ltx_m1, "Model 1 R-squared (0.9156) must be present")
        self.assertIn("21.69", ltx_m1, "Model 1 F-stat (21.69) must be present")
        self.assertNotIn("0.1009", ltx_m1, "Model 2 R-squared (0.1009) must NOT contaminate Model 1")

    # =========================================================================
    # Test 24: Default No-Selection Export Resolves to Latest Model (Scenario C)
    # =========================================================================
    def test_24_no_selection_defaults_to_latest_model(self):
        """
        Verifies that when no selection is made, export actions resolve strictly to
        the latest model report rather than blindly dumping the entire console history.
        """
        console = ResultsConsole()
        console.append_command("regress price mpg weight", execution_id=1)
        console.append_output(self.out_reg, execution_id=1, command_text="regress price mpg weight")

        console.append_command("regress price mpg", execution_id=2)
        console.append_output(self.out_reg2, execution_id=2, command_text="regress price mpg")

        # Ensure no selection
        cursor = console.textCursor()
        cursor.clearSelection()
        console.setTextCursor(cursor)

        records, target_text, is_sel = console._resolve_target_records()
        self.assertFalse(is_sel)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].command_text, "regress price mpg")

        # Should export strictly the latest model (Model 2)
        ltx = format_full_report_as_latex(target_text, extra_diagnostics=records[0].diagnostics)
        self.assertIsNotNone(ltx)
        self.assertIn("0.1009", ltx)
        self.assertNotIn("0.9156", ltx)

    # =========================================================================
    # Test 25: Multi-Model Selection Exports Independent Scoped Reports (Scenario D)
    # =========================================================================
    def test_25_multi_model_selection_independent_reports(self):
        """
        Verifies that when selection spans multiple models, each model receives its own
        diagnostics without merging into a single contaminated block.
        """
        console = ResultsConsole()
        console.append_command("regress price mpg weight", execution_id=1)
        console.append_output(self.out_reg, execution_id=1, command_text="regress price mpg weight")

        console.append_command("regress price mpg", execution_id=2)
        console.append_output(self.out_reg2, execution_id=2, command_text="regress price mpg")

        # Select entire document spanning both models
        cursor = console.textCursor()
        cursor.select(QtGui.QTextCursor.SelectionType.Document)
        console.setTextCursor(cursor)

        records, target_text, is_sel = console._resolve_target_records()
        self.assertTrue(is_sel)
        self.assertEqual(len(records), 2)

        # Multi-command string segmentation directly
        ltx = format_table_as_latex(console.toPlainText(), full_report=True)
        self.assertIsNotNone(ltx)
        self.assertEqual(ltx.count("Model Diagnostics"), 2)
        self.assertIn("0.9156", ltx)
        self.assertIn("0.1009", ltx)

    # =========================================================================
    # Test 26: Interleaved Invalid Command Between Regressions (Scenario E)
    # =========================================================================
    def test_26_interleaved_invalid_command_between_regressions(self):
        """
        Verifies that an error/invalid command between regressions does not
        corrupt execution records or model diagnostic scoping.
        """
        console = ResultsConsole()
        console.append_command("regress price mpg weight", execution_id=1)
        console.append_output(self.out_reg, execution_id=1, command_text="regress price mpg weight")

        console.append_command("invalid syntax command", execution_id=2)
        console.append_error("unrecognized command: invalid syntax", execution_id=2, command_text="invalid syntax command")

        console.append_command("regress price mpg", execution_id=3)
        console.append_output(self.out_reg2, execution_id=3, command_text="regress price mpg")

        # No selection should still resolve to Model 2
        cursor = console.textCursor()
        cursor.clearSelection()
        console.setTextCursor(cursor)

        records, target_text, is_sel = console._resolve_target_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].command_text, "regress price mpg")
        self.assertIn("0.1009", [v for k, v in records[0].diagnostics if k == "R-squared"])

    # =========================================================================
    # Test 27: Interleaved Descriptive Command Between Regressions (Scenario F)
    # =========================================================================
    def test_27_interleaved_summarize_between_regressions(self):
        """
        Verifies that running 'summarize' between regressions does not contaminate
        diagnostics or table detection.
        """
        console = ResultsConsole()
        console.append_command("regress price mpg weight", execution_id=1)
        console.append_output(self.out_reg, execution_id=1, command_text="regress price mpg weight")

        console.append_command("summarize price mpg", execution_id=2)
        console.append_output(self.out_sum, execution_id=2, command_text="summarize price mpg")

        console.append_command("regress price mpg", execution_id=3)
        console.append_output(self.out_reg2, execution_id=3, command_text="regress price mpg")

        # Summarize record has no regression diagnostics
        rec_sum = console._execution_records[1]
        self.assertEqual(len(rec_sum.diagnostics), 0)

        # Model 2 record retains strictly its own diagnostics
        rec_m2 = console._execution_records[2]
        self.assertIn("0.1009", [v for k, v in rec_m2.diagnostics if k == "R-squared"])

    # =========================================================================
    # Test 28: Defect 5.1 Regression Test (Trailing Invalid Candidate Block)
    # =========================================================================
    def test_28_trailing_invalid_candidate_does_not_drop_diagnostics(self):
        """
        Verifies that when text ends with an invalid candidate block (e.g. 'Status | pending'),
        diagnostics attach to the last VALID table block rather than being dropped (Defect 5.1 fix).
        """
        corrupted_text = (
            f"{self.out_reg}\n\n"
            "Status | pending\n"
            "Diagnostic output | continuing\n"
        )
        ltx = format_full_report_as_latex(corrupted_text)
        self.assertIsNotNone(ltx)
        self.assertIn("Model Diagnostics", ltx, "Model Diagnostics must NOT be dropped")
        self.assertIn("R-squared", ltx)
        self.assertIn("0.9156", ltx)

        md = format_full_report_as_markdown(corrupted_text)
        self.assertIsNotNone(md)
        self.assertIn("### Model Diagnostics", md, "Model Diagnostics must NOT be dropped in Markdown")

    # =========================================================================
    # Test 29: Explicit Empty Diagnostics Preservation
    # =========================================================================
    def test_29_explicit_empty_diagnostics_preservation(self):
        """
        Verifies that passing extra_diagnostics=() or [] does not fall back to text scraping.
        """
        ltx = format_table_as_latex(self.out_reg, full_report=True, extra_diagnostics=())
        self.assertIsNotNone(ltx)
        self.assertNotIn("Model Diagnostics", ltx, "Explicit empty diagnostics must be respected")

        md = format_table_as_markdown(self.out_reg, full_report=True, extra_diagnostics=[])
        self.assertIsNotNone(md)
        self.assertNotIn("### Model Diagnostics", md, "Explicit empty diagnostics must be respected")

    # =========================================================================
    # Test 30: StatisticalResultRecord and StatisticalReport Provenance Container
    # =========================================================================
    def test_30_statistical_provenance_containers(self):
        """Verifies dataclass contracts, properties, and formatting helpers."""
        rep = StatisticalReport(
            command_text="regress price mpg",
            table_text=self.out_reg2,
            diagnostics=[("R-squared", "0.1009"), ("Root MSE", "1707.4833")],
            model_name="m2"
        )
        ltx = rep.format_latex(full_report=True)
        self.assertIsNotNone(ltx)
        self.assertIn("0.1009", ltx)
        self.assertIn("1707.4833", ltx)

        md = rep.format_markdown(full_report=True)
        self.assertIsNotNone(md)
        self.assertIn("0.1009", md)

        rec = StatisticalResultRecord(
            execution_id=42,
            command_text="regress price mpg",
            char_start=0,
            char_end=100,
            output_text=self.out_reg2,
            diagnostics=[("R-squared", "0.1009")]
        )
        self.assertEqual(rec.execution_id, 42)
        self.assertEqual(rec.command_text, "regress price mpg")
        self.assertEqual(len(rec.diagnostics), 1)


    def test_script_and_reopened_project_export_each_models_own_diagnostics(self):
        from gui_qt.main_window import PotatosticsQtWindow
        from engine.session_manager import save_session_bundle

        window = PotatosticsQtWindow()
        self.addCleanup(window.close)
        window.dataset.load_dataframe(self.ds.df.copy())
        console = window.console
        console.append_output("Research notes 🥔")

        def select_coefficients(output):
            text = console.toPlainText()
            block = extract_table_blocks_from_text(output)[-1]
            start = text.index(block[0], text.index(output))
            end = text.index(block[-2], start) + len(block[-2])
            cursor = console.textCursor()
            # Qt indexes UTF-16 units, including for a transcript with emoji.
            cursor.setPosition(len(text[:start].encode("utf-16-le")) // 2)
            cursor.setPosition(len(text[:end].encode("utf-16-le")) // 2,
                               QtGui.QTextCursor.MoveMode.KeepAnchor)
            console.setTextCursor(cursor)

        def check_reports():
            for output, wanted, other in [(self.out_reg, "0.9156", "0.1009"),
                                          (self.out_reg2, "0.1009", "0.9156")]:
                select_coefficients(output)
                for copy in (console._copy_full_report_as_latex, console._copy_full_report_as_markdown):
                    copy()
                    report = app.clipboard().text()
                    self.assertIn(wanted, report)
                    self.assertNotIn(other, report)

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "models.do"
            script.write_text("if 1 {\nregress price mpg weight\n}\nregress price mpg\nsummarize price\n",
                              encoding="utf-8")
            result = window.run_command_sync(f'do "{script}"')
            self.assertTrue(result.ok, result)
            check_reports()
            path = str(Path(tmp) / "new.potproj")
            self.assertTrue(window.save_session(path))
            self.assertTrue(window.open_session(path))
            check_reports()
            cursor = console.textCursor()
            cursor.clearSelection()
            console.setTextCursor(cursor)
            console._copy_full_report_as_markdown()
            self.assertIn("0.1009", app.clipboard().text())

            legacy = str(Path(tmp) / "legacy.potproj")
            save_session_bundle(legacy, window.interpreter, console_text=console.toPlainText())
            self.assertTrue(window.open_session(legacy))
            select_coefficients(self.out_reg2)
            console._copy_full_report_as_markdown()
            self.assertIn("mpg", app.clipboard().text())
            self.assertNotIn("Model Diagnostics", app.clipboard().text())
            self.assertNotIn("0.9156", app.clipboard().text())

    def test_export_preserves_grouped_observation_counts(self):
        for count in ("1,234", "1,023,456"):
            output = self.out_reg.replace("Number of obs   =         7", f"Number of obs   = {count}")
            self.assertIn(("Number of obs", count), extract_diagnostics_from_text(output))
            self.assertIn(count, format_full_report_as_markdown(output))
            self.assertIn(count, format_full_report_as_latex(output))

    # =========================================================================
    # Phase 1: Comprehensive Multi-Model Reporting Test Suite (15 Test Cases)
    # =========================================================================

    def test_p1_01_single_ols_command_result(self):
        """1. Single OLS CommandResult: formats single model with exact coefficients and diagnostics."""
        res = self.interp.execute("regress price mpg weight")
        md = format_full_report_as_markdown(res)
        self.assertIsNotNone(md)
        self.assertIn("0.9156", md)
        self.assertIn("1034.0161", md)
        self.assertIn("10.0710", md)
        ltx = format_full_report_as_latex(res)
        self.assertIsNotNone(ltx)
        self.assertIn("0.9156", ltx)

    def test_p1_02_two_model_do_file_command_result(self):
        """2. Two-model do-file CommandResult: preserves both models with their own diagnostics."""
        from engine.script_runner import ScriptRunner
        runner = ScriptRunner(self.interp)
        res = runner.run_script("regress price mpg weight\nregress price mpg\n", is_do_file=True)
        md = format_full_report_as_markdown(res)
        self.assertIsNotNone(md)
        # Assert both 0.9156 and 0.1009 appear
        self.assertIn("0.9156", md)
        self.assertIn("0.1009", md)
        # Verify both coefficient tables are preserved
        self.assertIn("weight", md)
        self.assertIn("10.0710", md)
        self.assertIn("-92.3994", md)

        # LaTeX format
        ltx = format_full_report_as_latex(res)
        self.assertIsNotNone(ltx)
        self.assertIn("0.9156", ltx)
        self.assertIn("0.1009", ltx)

    def test_p1_03_three_model_do_file(self):
        """3. Three-model do-file: formats all three models independently without diagnostic cross-talk."""
        from engine.script_runner import ScriptRunner
        runner = ScriptRunner(self.interp)
        res = runner.run_script("regress price mpg weight\nregress price mpg\nregress price weight\n", is_do_file=True)
        md = format_full_report_as_markdown(res)
        self.assertIsNotNone(md)
        self.assertIn("0.9156", md)
        self.assertIn("0.1009", md)
        # Check that weight regression R-squared is also present
        res_w = self.interp.execute("regress price weight")
        r2_w = [v for k, v in extract_diagnostics_from_text(res_w.output) if k == "R-squared"][0]
        self.assertIn(r2_w, md)

    def test_p1_04_direct_plain_string_output_one_model(self):
        """4. Direct plain-string output from one model preserves existing backward-compatibility."""
        md = format_full_report_as_markdown(self.out_reg)
        self.assertIsNotNone(md)
        self.assertIn("0.9156", md)
        self.assertIn("1034.0161", md)

    def test_p1_05_ambiguous_concatenated_output_raises(self):
        """5. Ambiguous concatenated output without boundaries raises AmbiguousReportingError."""
        from gui_qt.console import AmbiguousReportingError
        # Two coefficient tables glued together with no headers, no ANOVA, no prompts
        tbl1 = (
            "       price | Coefficient         Std. err.      t    P>|t|     [95% conf. interval]\n"
            "-------------+--------------------------------------------------------------------\n"
            "         mpg |   1034.0161         186.1726    5.55   0.005       517.1182   1550.9141\n"
            "       _cons | -45918.6931        8778.0419   -5.23   0.006    -70290.4447 -21546.9415\n"
            "-------------+--------------------------------------------------------------------\n"
        )
        tbl2 = (
            "       price | Coefficient         Std. err.      t    P>|t|     [95% conf. interval]\n"
            "-------------+--------------------------------------------------------------------\n"
            "      weight |     10.0710           1.6211    6.21   0.003         5.5701     14.5718\n"
            "       _cons | -45918.6931        8778.0419   -5.23   0.006    -70290.4447 -21546.9415\n"
            "-------------+--------------------------------------------------------------------\n"
        )
        ambiguous_text = tbl1 + "\n\n" + tbl2
        with self.assertRaises(AmbiguousReportingError):
            format_full_report_as_markdown(ambiguous_text)

    def test_p1_06_structured_output_section_sequence(self):
        """6. Structured OutputSection sequence: formats list of OutputSections independently."""
        from engine.parser import OutputSection
        sec1 = OutputSection("regress price mpg weight", self.out_reg, True)
        sec2 = OutputSection("regress price mpg", self.out_reg2, True)
        md = format_full_report_as_markdown([sec1, sec2])
        self.assertIsNotNone(md)
        self.assertIn("0.9156", md)
        self.assertIn("0.1009", md)

    def test_p1_07_different_model_sample_sizes(self):
        """7. Different model sample sizes: each model retains its own correct observation count."""
        from engine.script_runner import ScriptRunner
        runner = ScriptRunner(self.interp)
        res = runner.run_script("regress price mpg weight\nregress price mpg if mpg > 18\n", is_do_file=True)
        md = format_full_report_as_markdown(res)
        self.assertIsNotNone(md)
        self.assertIn("| Number of obs | 7 |", md)
        self.assertIn("| Number of obs | 4 |", md)

    def test_p1_08_different_predictor_counts(self):
        """8. Different predictor counts: model 1 with 2 predictors, model 2 with 1 predictor."""
        from engine.script_runner import ScriptRunner
        runner = ScriptRunner(self.interp)
        res = runner.run_script("regress price mpg weight\nregress price mpg\n", is_do_file=True)
        md = format_full_report_as_markdown(res)
        sections = md.split("### Model Diagnostics")
        self.assertEqual(len(sections), 3)  # 2 models produce 2 diagnostic sections
        # Section 1 contains weight, Section 2 does not contain weight in its table
        self.assertIn("weight", sections[0])
        self.assertNotIn("weight", sections[1])

    def test_p1_09_classical_and_robust_ols(self):
        """9. Classical and robust OLS: formats classical (with ANOVA) and robust (without ANOVA) cleanly."""
        from engine.script_runner import ScriptRunner
        runner = ScriptRunner(self.interp)
        res = runner.run_script("regress price mpg weight\nregress price mpg weight, robust\n", is_do_file=True)
        md = format_full_report_as_markdown(res)
        self.assertIsNotNone(md)
        self.assertIn("0.9156", md)

    def test_p1_10_invalid_command_between_regressions(self):
        """10. Invalid command between two regressions: does not disrupt valid model report formatting."""
        from engine.script_runner import ScriptRunner
        runner = ScriptRunner(self.interp)
        res = runner.run_script("regress price mpg weight\nnonexistent_cmd_xyz\nregress price mpg\n", is_do_file=True, nostop=True)
        md = format_full_report_as_markdown(res)
        self.assertIsNotNone(md)
        self.assertIn("0.9156", md)
        self.assertIn("0.1009", md)

    def test_p1_11_summarize_between_regressions(self):
        """11. Summarize between two regressions: summarize does not contaminate regression diagnostics."""
        from engine.script_runner import ScriptRunner
        runner = ScriptRunner(self.interp)
        res = runner.run_script("regress price mpg weight\nsummarize price\nregress price mpg\n", is_do_file=True)
        md = format_full_report_as_markdown(res)
        self.assertIsNotNone(md)
        self.assertIn("0.9156", md)
        self.assertIn("0.1009", md)

    def test_p1_12_nested_do_file(self):
        """12. Nested script containing several estimates."""
        from engine.script_runner import ScriptRunner
        with tempfile.TemporaryDirectory() as tmp:
            sub = Path(tmp) / "sub.do"
            sub.write_text("regress price mpg\n", encoding="utf-8")
            main_script = f"regress price mpg weight\ndo \"{sub}\"\n"
            runner = ScriptRunner(self.interp)
            res = runner.run_script(main_script, is_do_file=True)
            md = format_full_report_as_markdown(res)
            self.assertIsNotNone(md)
            self.assertIn("0.9156", md)
            self.assertIn("0.1009", md)

    def test_p1_13_multi_model_with_warnings(self):
        """13. Multi-model output containing warnings or notes retains clean table formatting."""
        from engine.script_runner import ScriptRunner
        runner = ScriptRunner(self.interp)
        # foreign has few observations
        res = runner.run_script("regress price mpg foreign\nregress price mpg\n", is_do_file=True)
        md = format_full_report_as_markdown(res)
        self.assertIsNotNone(md)
        self.assertIn("mpg", md)

    def test_p1_14_explicit_empty_diagnostics(self):
        """14. Explicit empty diagnostics formats table without diagnostics block."""
        md = format_full_report_as_markdown(self.out_reg, extra_diagnostics=[])
        self.assertIsNotNone(md)
        self.assertNotIn("Model Diagnostics", md)

    def test_p1_15_missing_or_unsupported_model_diagnostics(self):
        """15. Missing or unsupported model diagnostics: table formats cleanly without error."""
        text_no_diag = (
            "       price | Coefficient         Std. err.      t    P>|t|     [95% conf. interval]\n"
            "-------------+--------------------------------------------------------------------\n"
            "         mpg |   1034.0161         186.1726    5.55   0.005       517.1182   1550.9141\n"
            "       _cons | -45918.6931        8778.0419   -5.23   0.006    -70290.4447 -21546.9415\n"
            "-------------+--------------------------------------------------------------------\n"
        )
        md = format_full_report_as_markdown(text_no_diag)
        self.assertIsNotNone(md)
        self.assertIn("1034.0161", md)


if __name__ == "__main__":
    unittest.main()
