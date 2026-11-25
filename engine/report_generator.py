"""
engine/report_generator.py - Native Word (.docx) & HTML Report Generator
Generates publication-ready academic reports containing summary statistics,
regression tables, and model comparisons using python-docx.
"""

import os
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

def generate_docx_report(filepath, dataset, estimates_store):
    doc = Document()
    
    # Page setup
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Title
    title = doc.add_heading("Potatostics Econometric Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.add_run(f"Dataset: {os.path.basename(dataset.filename or 'Active In-Memory Data')} | Observations: {dataset.N:,} | Variables: {dataset.K:,}").italic = True
    
    # 1. Dataset Overview
    doc.add_heading("1. Summary Statistics", level=1)
    
    # Summary stats table
    table_su = doc.add_table(rows=1, cols=6)
    table_su.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table_su.rows[0].cells
    hdr_titles = ["Variable", "Obs", "Mean", "Std. Dev.", "Min", "Max"]
    for i, t in enumerate(hdr_titles):
        hdr_cells[i].text = t
        hdr_cells[i].paragraphs[0].runs[0].bold = True

    from engine.estimation import compute_summarize
    stats_data = compute_summarize(dataset.df)
    for s in stats_data:
        row_cells = table_su.add_row().cells
        row_cells[0].text = s['var']
        row_cells[1].text = f"{s['obs']:,}"
        row_cells[2].text = f"{s['mean']:.4f}"
        row_cells[3].text = f"{s['std']:.4f}"
        row_cells[4].text = f"{s['min']:.4f}"
        row_cells[5].text = f"{s['max']:.4f}"

    # 2. Estimation Models
    doc.add_heading("2. Econometric Estimation Models", level=1)
    
    if estimates_store.store:
        # Comparison Table
        models = estimates_store.store
        all_vars = []
        for m in models.values():
            for row in m['coefs']:
                if row['name'] not in all_vars and row['name'] != '_cons':
                    all_vars.append(row['name'])
        all_vars.append('_cons')

        num_cols = len(models) + 1
        table_est = doc.add_table(rows=1, cols=num_cols)
        table_est.alignment = WD_TABLE_ALIGNMENT.CENTER
        
        # Header
        h_cells = table_est.rows[0].cells
        h_cells[0].text = "Variable"
        h_cells[0].paragraphs[0].runs[0].bold = True
        for idx, m_name in enumerate(models.keys()):
            h_cells[idx + 1].text = m_name
            h_cells[idx + 1].paragraphs[0].runs[0].bold = True

        for v in all_vars:
            r_c = table_est.add_row().cells
            r_c[0].text = v
            r_c[0].paragraphs[0].runs[0].bold = True
            
            # Standard error row
            r_se = table_est.add_row().cells
            r_se[0].text = " "
            
            for idx, (m_name, m) in enumerate(models.items()):
                coef_dict = {r['name']: r for r in m['coefs']}
                if v in coef_dict:
                    row = coef_dict[v]
                    b = row['coef']
                    p = row['p']
                    se = row['se']
                    stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
                    r_c[idx + 1].text = f"{b:.4f}{stars}"
                    r_se[idx + 1].text = f"({se:.4f})"
                else:
                    r_c[idx + 1].text = ""
                    r_se[idx + 1].text = ""

        # Footer rows
        r_n = table_est.add_row().cells
        r_n[0].text = "Observations"
        r_n[0].paragraphs[0].runs[0].bold = True
        for idx, m in enumerate(models.values()):
            r_n[idx + 1].text = f"{m['n']:,}"

        r_r2 = table_est.add_row().cells
        r_r2[0].text = "R-squared"
        r_r2[0].paragraphs[0].runs[0].bold = True
        for idx, m in enumerate(models.values()):
            r2_val = m.get('r2', m.get('pseudo_r2', m.get('r2_within', 0.0)))
            r_r2[idx + 1].text = f"{r2_val:.4f}"

        p_note = doc.add_paragraph()
        p_note.add_run("* p<0.10, ** p<0.05, *** p<0.01. Standard errors in parentheses.").italic = True
    elif estimates_store.last_model:
        m = estimates_store.last_model
        p = doc.add_paragraph()
        p.add_run(f"Model: {m['model_type'].upper()} on {m['depvar']} | Observations: {m['n']:,}").bold = True
        
        table_m = doc.add_table(rows=1, cols=5)
        table_m.alignment = WD_TABLE_ALIGNMENT.CENTER
        h = table_m.rows[0].cells
        for idx, t in enumerate(["Variable", "Coefficient", "Std. Err.", "t / z", "P-value"]):
            h[idx].text = t
            h[idx].paragraphs[0].runs[0].bold = True
            
        for r in m['coefs']:
            rc = table_m.add_row().cells
            rc[0].text = r['name']
            rc[1].text = f"{r['coef']:.4f}"
            rc[2].text = f"{r['se']:.4f}"
            rc[3].text = f"{r.get('t', r.get('z', 0.0)):.2f}"
            rc[4].text = f"{r['p']:.4f}"
    else:
        doc.add_paragraph("No estimation models currently stored in memory.")

    doc.save(filepath)
    return f"(report successfully exported to '{filepath}')"
