r"""
engine/exporters.py - Multi-Model Academic Exporter (esttab)
Supports:
1. Side-by-side multi-model regression tables with significance stars.
2. Formats standard errors in parentheses or t-statistics.
3. Summary rows: Observations, R-squared, Adjusted R-squared, F-statistic, etc.
4. Native LaTeX booktabs export (\toprule, \midrule, \bottomrule).
5. Native Word (.docx) table export via python-docx.
6. HTML table export with modern responsive styling.
7. CSV export.
"""

import os
import numpy as np
import pandas as pd


class ModelTableExporter:
    """Exports stored estimation models to LaTeX booktabs, Word, HTML, or CSV."""

    def __init__(self, models_dict, options=None):
        self.models = models_dict
        self.options = options or {}

    def _get_unique_vars(self):
        all_vars = []
        for m in self.models.values():
            for row in m.get('coefs', []):
                name = row['name']
                if name not in all_vars and name not in ('_cons', 'Intercept'):
                    all_vars.append(name)
        if any(any(r['name'] in ('_cons', 'Intercept') for r in m.get('coefs', [])) for m in self.models.values()):
            all_vars.append('_cons')
        return all_vars

    def to_ascii_text(self):
        """Generates monospace text comparison table."""
        if not self.models:
            return "no estimation models available to compare"

        all_vars = self._get_unique_vars()
        col_w = 16
        lines = []

        header = f"{'Variable':<14} | " + " ".join(f"{name:^{col_w}}" for name in self.models.keys())
        lines.append(header)
        lines.append("-" * 15 + "+" + "-" * (col_w * len(self.models) + len(self.models)))

        for v in all_vars:
            coef_strs = []
            se_strs = []
            for m in self.models.values():
                coef_dict = {r['name']: r for r in m.get('coefs', [])}
                if v in coef_dict:
                    row = coef_dict[v]
                    b = row['coef']
                    p = row.get('p', 1.0)
                    se = row.get('se', 0.0)
                    stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
                    coef_strs.append(f"{b:10.4f}{stars:<3}")
                    se_strs.append(f"({se:10.4f})   ")
                else:
                    coef_strs.append(f"{'':^{col_w}}")
                    se_strs.append(f"{'':^{col_w}}")
            lines.append(f"{v:<14} | " + " ".join(f"{s:^{col_w}}" for s in coef_strs))
            lines.append(f"{'':<14} | " + " ".join(f"{s:^{col_w}}" for s in se_strs))

        lines.append("-" * 15 + "+" + "-" * (col_w * len(self.models) + len(self.models)))

        # Summary statistics
        obs_strs = [f"{m.get('n', 0):^{col_w},}" for m in self.models.values()]
        lines.append(f"{'Observations':<14} | " + " ".join(obs_strs))

        r2_strs = []
        for m in self.models.values():
            r2_val = m.get('r2', m.get('pseudo_r2', m.get('r2_within', 0.0)))
            r2_strs.append(f"{r2_val:^{col_w}.4f}")
        lines.append(f"{'R-squared':<14} | " + " ".join(r2_strs))

        lines.append("-" * 15 + "+" + "-" * (col_w * len(self.models) + len(self.models)))
        lines.append("* p<0.10, ** p<0.05, *** p<0.01; standard errors in parentheses")
        return "\n".join(lines)

    def to_latex_booktabs(self, title="Regression Results", label="tab:regression"):
        """Generates production-grade LaTeX table with booktabs formatting."""
        all_vars = self._get_unique_vars()
        n_models = len(self.models)
        align = "l" + "c" * n_models

        lines = [
            "% Requires \\usepackage{booktabs} in document preamble",
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{title}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{align}}}",
            "\\toprule"
        ]

        # Model headers
        headers = [" "] + [m_name.replace("_", "\\_") for m_name in self.models.keys()]
        lines.append(" & ".join(headers) + " \\\\")
        lines.append("\\midrule")

        # Coefficients and standard errors
        for v in all_vars:
            v_disp = v.replace("_", "\\_")
            c_line = [v_disp]
            s_line = [" "]
            for m in self.models.values():
                coef_dict = {r['name']: r for r in m.get('coefs', [])}
                if v in coef_dict:
                    row = coef_dict[v]
                    b = row['coef']
                    p = row.get('p', 1.0)
                    se = row.get('se', 0.0)
                    stars = "^{***}" if p < 0.01 else ("^{**}" if p < 0.05 else ("^{*}" if p < 0.10 else ""))
                    c_line.append(f"{b:.4f}{stars}")
                    s_line.append(f"({se:.4f})")
                else:
                    c_line.append(" ")
                    s_line.append(" ")
            lines.append(" & ".join(c_line) + " \\\\")
            lines.append(" & ".join(s_line) + " \\\\[0.4ex]")

        lines.append("\\midrule")
        lines.append("Observations & " + " & ".join(f"{m.get('n', 0):,}" for m in self.models.values()) + " \\\\")

        r2_vals = []
        for m in self.models.values():
            r2_val = m.get('r2', m.get('pseudo_r2', m.get('r2_within', 0.0)))
            r2_vals.append(f"{r2_val:.4f}")
        lines.append("R-squared & " + " & ".join(r2_vals) + " \\\\")

        lines.append("\\bottomrule")
        lines.append(f"\\multicolumn{{{n_models + 1}}}{{l}}{{\\footnotesize * p$<$0.10, ** p$<$0.05, *** p$<$0.01. Standard errors in parentheses.}}")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    def to_html(self, title="Regression Results"):
        """Generates clean HTML table with modern CSS styling."""
        all_vars = self._get_unique_vars()
        lines = [
            "<style>",
            "table.potatostics-table { border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 14px; margin: 16px 0; width: 100%; }",
            "table.potatostics-table th, table.potatostics-table td { padding: 8px 14px; text-align: center; }",
            "table.potatostics-table th:first-child, table.potatostics-table td:first-child { text-align: left; }",
            "table.potatostics-table thead tr { border-top: 2px solid #222; border-bottom: 1px solid #222; }",
            "table.potatostics-table tbody tr.se-row { color: #555; }",
            "table.potatostics-table tbody tr.mid-sep { border-top: 1px solid #ddd; }",
            "table.potatostics-table tfoot tr { border-top: 1px solid #222; border-bottom: 2px solid #222; }",
            "table.potatostics-table caption { font-weight: bold; font-size: 16px; margin-bottom: 8px; text-align: left; }",
            "</style>",
            "<table class='potatostics-table'>",
            f"<caption>{title}</caption>",
            "<thead><tr><th>Variable</th>" + "".join(f"<th>{name}</th>" for name in self.models.keys()) + "</tr></thead>",
            "<tbody>"
        ]

        for v in all_vars:
            c_row = [f"<td><strong>{v}</strong></td>"]
            s_row = ["<td></td>"]
            for m in self.models.values():
                coef_dict = {r['name']: r for r in m.get('coefs', [])}
                if v in coef_dict:
                    row = coef_dict[v]
                    b = row['coef']
                    p = row.get('p', 1.0)
                    se = row.get('se', 0.0)
                    stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
                    c_row.append(f"<td>{b:.4f}{stars}</td>")
                    s_row.append(f"<td>({se:.4f})</td>")
                else:
                    c_row.append("<td></td>")
                    s_row.append("<td></td>")
            lines.append("<tr>" + "".join(c_row) + "</tr>")
            lines.append("<tr class='se-row'>" + "".join(s_row) + "</tr>")

        lines.append("<tr class='mid-sep'><td>Observations</td>" + "".join(f"<td>{m.get('n', 0):,}</td>" for m in self.models.values()) + "</tr>")
        r2_vals = [f"<td>{m.get('r2', m.get('pseudo_r2', m.get('r2_within', 0.0))):.4f}</td>" for m in self.models.values()]
        lines.append("<tr><td>R-squared</td>" + "".join(r2_vals) + "</tr>")
        lines.append("</tbody>")
        lines.append(f"<tfoot><tr><td colspan='{len(self.models)+1}' style='text-align:left; font-size:12px; color:#666;'>* p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01; standard errors in parentheses</td></tr></tfoot>")
        lines.append("</table>")
        return "\n".join(lines)

    def to_docx(self, filepath, title="Regression Results"):
        """Exports professional Word (.docx) table using python-docx."""
        try:
            import docx
            from docx.shared import Pt
            from docx.enum.table import WD_TABLE_ALIGNMENT
        except ImportError:
            raise ImportError("python-docx is required for docx export")

        doc = docx.Document()
        doc.add_heading(title, level=2)

        all_vars = self._get_unique_vars()
        n_cols = len(self.models) + 1
        table = doc.add_table(rows=1, cols=n_cols)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.style = 'Light Shading Accent 1' if 'Light Shading Accent 1' in doc.styles else 'Table Grid'

        # Header row
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = "Variable"
        for i, m_name in enumerate(self.models.keys(), 1):
            hdr_cells[i].text = str(m_name)

        # Coefficient rows
        for v in all_vars:
            row_c = table.add_row().cells
            row_s = table.add_row().cells
            row_c[0].text = v
            row_s[0].text = ""

            for i, m in enumerate(self.models.values(), 1):
                coef_dict = {r['name']: r for r in m.get('coefs', [])}
                if v in coef_dict:
                    r = coef_dict[v]
                    b = r['coef']
                    p = r.get('p', 1.0)
                    se = r.get('se', 0.0)
                    stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
                    row_c[i].text = f"{b:.4f}{stars}"
                    row_s[i].text = f"({se:.4f})"
                else:
                    row_c[i].text = ""
                    row_s[i].text = ""

        # Summary rows
        row_n = table.add_row().cells
        row_n[0].text = "Observations"
        for i, m in enumerate(self.models.values(), 1):
            row_n[i].text = f"{m.get('n', 0):,}"

        row_r2 = table.add_row().cells
        row_r2[0].text = "R-squared"
        for i, m in enumerate(self.models.values(), 1):
            r2_val = m.get('r2', m.get('pseudo_r2', m.get('r2_within', 0.0)))
            row_r2[i].text = f"{r2_val:.4f}"

        p_note = doc.add_paragraph("* p<0.10, ** p<0.05, *** p<0.01. Standard errors in parentheses.")
        p_note.runs[0].font.size = Pt(9)
        p_note.runs[0].font.italic = True

        doc.save(filepath)
        return f"(Word table exported to {filepath})"

    def to_csv(self, filepath):
        """Exports clean formatted CSV comparison table."""
        all_vars = self._get_unique_vars()
        rows = []

        header = ["Variable"] + list(self.models.keys())
        rows.append(header)

        for v in all_vars:
            r_c = [v]
            r_s = [""]
            for m in self.models.values():
                coef_dict = {r['name']: r for r in m.get('coefs', [])}
                if v in coef_dict:
                    row = coef_dict[v]
                    b = row['coef']
                    p = row.get('p', 1.0)
                    se = row.get('se', 0.0)
                    stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
                    r_c.append(f"{b:.4f}{stars}")
                    r_s.append(f"({se:.4f})")
                else:
                    r_c.append("")
                    r_s.append("")
            rows.append(r_c)
            rows.append(r_s)

        row_n = ["Observations"] + [f"{m.get('n', 0):,}" for m in self.models.values()]
        rows.append(row_n)

        r2_vals = [f"{m.get('r2', m.get('pseudo_r2', m.get('r2_within', 0.0))):.4f}" for m in self.models.values()]
        row_r2 = ["R-squared"] + r2_vals
        rows.append(row_r2)

        df_out = pd.DataFrame(rows[1:], columns=rows[0])
        df_out.to_csv(filepath, index=False)
        return f"(CSV table exported to {filepath})"

    def export(self, filepath):
        """Exports the table to a file based on file extension (.tex, .docx, .html, .csv)."""
        ext = os.path.splitext(filepath)[1].lower()
        dir_name = os.path.dirname(os.path.abspath(filepath))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if ext in (".tex", ".latex"):
            content = self.to_latex_booktabs()
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return f"(LaTeX table exported to {filepath})"
        elif ext == ".docx":
            return self.to_docx(filepath)
        elif ext in (".html", ".htm"):
            content = self.to_html()
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return f"(HTML table exported to {filepath})"
        elif ext == ".csv":
            return self.to_csv(filepath)
        else:
            content = self.to_latex_booktabs()
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return f"(LaTeX table exported to {filepath})"


class DemographicTableExporter:
    """Table 1 / Demographic and Balance Table Exporter (Demographic summary table)."""

    def __init__(self, df, varlist=None, by_var=None, val_labels=None, val_label_names=None):
        import scipy.stats as stats
        self.stats = stats
        self.df = df.copy()
        self.by_var = by_var
        self.val_labels = val_labels or {}
        self.val_label_names = val_label_names or {}

        if by_var and by_var in self.df.columns:
            self.df = self.df.dropna(subset=[by_var])

        if not varlist:
            self.raw_varlist = [c for c in self.df.columns if c != by_var]
        else:
            self.raw_varlist = varlist

        self.parsed_vars = []
        for v in self.raw_varlist:
            if v.startswith("i."):
                col = v[2:]
                if col in self.df.columns:
                    self.parsed_vars.append({'name': col, 'categorical': True})
            else:
                col = v
                if col in self.df.columns:
                    is_cat = (
                        pd.api.types.is_object_dtype(self.df[col]) or
                        str(self.df[col].dtype) == 'category' or
                        (pd.api.types.is_numeric_dtype(self.df[col]) and self.df[col].nunique() <= 5)
                    )
                    self.parsed_vars.append({'name': col, 'categorical': is_cat})

        self.groups = []
        self.group_labels = {}
        if self.by_var and self.by_var in self.df.columns:
            self.groups = sorted(self.df[self.by_var].dropna().unique())
            lbl_map = self.val_labels.get(self.val_label_names.get(self.by_var, "")) or self.val_labels.get(self.by_var, {})
            for g in self.groups:
                g_key = int(g) if isinstance(g, (int, float, np.integer, np.floating)) and float(g).is_integer() else g
                self.group_labels[g] = str(lbl_map.get(g, lbl_map.get(g_key, g)))

        self.rows = self._build_rows()

    def _get_val_label(self, var, val):
        lbl_map = self.val_labels.get(self.val_label_names.get(var, "")) or self.val_labels.get(var, {})
        v_key = int(val) if isinstance(val, (int, float, np.integer, np.floating)) and float(val).is_integer() else val
        return str(lbl_map.get(val, lbl_map.get(v_key, val)))

    def _format_p(self, p):
        if p is None or np.isnan(p):
            return ""
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    def _build_rows(self):
        rows = []
        n_total = len(self.df)

        for item in self.parsed_vars:
            var = item['name']
            is_cat = item['categorical']

            if not is_cat:
                # Continuous variable
                # Total
                s_tot = self.df[var].dropna()
                mean_tot, sd_tot = s_tot.mean(), s_tot.std()
                tot_str = f"{mean_tot:,.2f} ({sd_tot:,.2f})"

                group_vals = {}
                p_val = None
                if self.groups:
                    group_series = []
                    for g in self.groups:
                        s_g = self.df[self.df[self.by_var] == g][var].dropna()
                        group_vals[g] = f"{s_g.mean():,.2f} ({s_g.std():,.2f})"
                        group_series.append(s_g)
                    try:
                        if len(self.groups) == 2:
                            _, p_val = self.stats.ttest_ind(group_series[0], group_series[1], equal_var=False)
                        elif len(self.groups) > 2:
                            _, p_val = self.stats.f_oneway(*group_series)
                    except Exception:
                        p_val = None

                rows.append({
                    'type': 'continuous',
                    'label': var,
                    'group_vals': group_vals,
                    'total': tot_str,
                    'p': self._format_p(p_val)
                })

            else:
                # Categorical variable
                # Calculate p-value across groups
                p_val = None
                if self.groups:
                    try:
                        ct = pd.crosstab(self.df[var], self.df[self.by_var])
                        _, p_val, _, _ = self.stats.chi2_contingency(ct)
                    except Exception:
                        p_val = None

                rows.append({
                    'type': 'cat_header',
                    'label': var,
                    'group_vals': {g: "" for g in self.groups},
                    'total': "",
                    'p': self._format_p(p_val)
                })

                levels = sorted(self.df[var].dropna().unique())
                for lvl in levels:
                    lvl_lbl = self._get_val_label(var, lvl)
                    n_tot_lvl = (self.df[var] == lvl).sum()
                    pct_tot = (n_tot_lvl / n_total * 100) if n_total > 0 else 0
                    lvl_tot_str = f"{n_tot_lvl:,} ({pct_tot:.1f}%)"

                    group_vals = {}
                    for g in self.groups:
                        df_g = self.df[self.df[self.by_var] == g]
                        n_g = len(df_g)
                        n_g_lvl = (df_g[var] == lvl).sum()
                        pct_g = (n_g_lvl / n_g * 100) if n_g > 0 else 0
                        group_vals[g] = f"{n_g_lvl:,} ({pct_g:.1f}%)"

                    rows.append({
                        'type': 'cat_level',
                        'label': f"  {lvl_lbl}",
                        'group_vals': group_vals,
                        'total': lvl_tot_str,
                        'p': ""
                    })

        return rows

    def to_ascii_text(self):
        col_w = 22
        lines = []
        n_total = len(self.df)

        # Header
        hdr = f"{'Variable':<26}"
        if self.groups:
            for g in self.groups:
                g_lbl = self.group_labels[g]
                n_g = (self.df[self.by_var] == g).sum()
                col_txt = f"{g_lbl} (N={n_g:,})"
                hdr += f" {col_txt:^{col_w}}"
        hdr += f" {'Total (N=' + f'{n_total:,}' + ')':^{col_w}}"
        if self.groups:
            hdr += f" {'p-value':^10}"

        sep = "-" * len(hdr)
        lines.append(sep)
        lines.append(hdr)
        lines.append(sep)

        for r in self.rows:
            line = f"{r['label']:<26}"
            if self.groups:
                for g in self.groups:
                    val_str = r['group_vals'].get(g, "")
                    line += f" {val_str:^{col_w}}"
            line += f" {r['total']:^{col_w}}"
            if self.groups:
                p_str = r['p']
                line += f" {p_str:^10}"
            lines.append(line)

        lines.append(sep)
        return "\n".join(lines)

    def to_dataframe(self):
        cols = ["Variable"]
        if self.groups:
            for g in self.groups:
                g_lbl = self.group_labels[g]
                n_g = (self.df[self.by_var] == g).sum()
                cols.append(f"{g_lbl} (N={n_g})")
        cols.append(f"Total (N={len(self.df)})")
        if self.groups:
            cols.append("p-value")

        data_rows = []
        for r in self.rows:
            row = [r['label']]
            if self.groups:
                for g in self.groups:
                    row.append(r['group_vals'].get(g, ""))
            row.append(r['total'])
            if self.groups:
                row.append(r['p'])
            data_rows.append(row)

        return pd.DataFrame(data_rows, columns=cols)

    def to_csv(self, filepath):
        df_out = self.to_dataframe()
        df_out.to_csv(filepath, index=False)
        return f"(Demographic table exported to {filepath})"

    def to_latex_booktabs(self, title="Demographic and Summary Statistics", label="tab:demographics"):
        df_out = self.to_dataframe()
        cols = df_out.columns.tolist()
        align = "l" + "c" * (len(cols) - 1)
        lines = [
            "% Requires \\usepackage{booktabs} in document preamble",
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{title}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{align}}}",
            "\\toprule",
            " & ".join(cols) + " \\\\",
            "\\midrule"
        ]
        for _, row in df_out.iterrows():
            escaped = [str(x).replace("%", "\\%").replace("&", "\\&").replace("_", "\\_") for x in row]
            lines.append(" & ".join(escaped) + " \\\\")
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        return "\n".join(lines)

    def to_docx(self, filepath):
        try:
            import docx
        except ImportError:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.to_ascii_text())
            return f"(Word export fell back to text file: {filepath})"

        df_out = self.to_dataframe()
        doc = docx.Document()
        table = doc.add_table(rows=len(df_out) + 1, cols=len(df_out.columns))
        table.style = 'Table Grid'

        # Header
        for c_idx, col in enumerate(df_out.columns):
            cell = table.cell(0, c_idx)
            cell.text = str(col)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.bold = True

        # Data
        for r_idx, row in df_out.iterrows():
            for c_idx, val in enumerate(row):
                table.cell(r_idx + 1, c_idx).text = str(val)

        doc.save(filepath)
        return f"(Word table exported to {filepath})"

    def export(self, filepath):
        ext = os.path.splitext(filepath)[1].lower()
        dir_name = os.path.dirname(os.path.abspath(filepath))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if ext in (".csv",):
            return self.to_csv(filepath)
        elif ext in (".tex", ".latex"):
            content = self.to_latex_booktabs()
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return f"(LaTeX table exported to {filepath})"
        elif ext in (".docx",):
            return self.to_docx(filepath)
        elif ext in (".html", ".htm"):
            df_out = self.to_dataframe()
            html = df_out.to_html(index=False, classes="table table-striped")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)
            return f"(HTML table exported to {filepath})"
        else:
            return self.to_csv(filepath)


