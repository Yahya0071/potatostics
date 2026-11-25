"""
gui_qt/console.py - Interactive Results Console with Academic Table Exporters
Provides:
- Colored monospaced terminal output for regression tables, ANOVAs, and summaries
- Right-click export menu: Copy as LaTeX (booktabs), Word Table, Markdown, CSV
- Command echo highlighting (. command in bold cyan) and error highlighting
- Structural table boundary detection and publication-quality table formatting
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any
from PySide6 import QtGui, QtWidgets
from engine.parser import OutputSection


@dataclass
class StatisticalResultRecord:
    """
    Provenance tracking record for command execution output in ResultsConsole.
    Associates text ranges, model snapshots, and model diagnostics with execution ID.
    """
    execution_id: Any = None
    command_text: str = ""
    char_start: int = 0
    char_end: int = 0
    output_text: str = ""
    model_snapshot: dict[str, Any] | None = None
    diagnostics: list[tuple[str, str]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    is_model: bool = False


@dataclass
class StatisticalReport:
    """
    Scoped container for an academic statistical report with verified model identity.
    """
    command_text: str = ""
    table_text: str = ""
    diagnostics: list[tuple[str, str]] = field(default_factory=list)
    model_name: str = ""
    is_valid: bool = True

    def format_latex(self, full_report: bool = True) -> str | None:
        return format_table_as_latex(self.table_text, full_report=full_report, extra_diagnostics=self.diagnostics)

    def format_markdown(self, full_report: bool = True) -> str | None:
        return format_table_as_markdown(self.table_text, full_report=full_report, extra_diagnostics=self.diagnostics)

_LATEX_MAP = {
    '\\': r'\textbackslash{}',
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
}
_LATEX_RE = re.compile(r'[\\&%\$#_{}~^]')


def escape_latex(s: str) -> str:
    """Escapes LaTeX special characters in table cell content in a single pass."""
    return _LATEX_RE.sub(lambda m: _LATEX_MAP[m.group(0)], s)


def escape_markdown(s: str) -> str:
    """Escapes Markdown special characters in table cell content."""
    return s.replace('|', r'\|')


def tokenize_header(header_right: str) -> list[str]:
    """Tokenizes right-hand side of table header into column labels."""
    patterns = [
        r'\[95%\s*conf\.?\s*interval\]',
        r'\[95%\s*Conf\.\s*Interval\]',
        r'Robust\s+std\.\s+err\.',
        r'Std\.\s+[Ee]rr\.',
        r'Std\.\s+dev\.',
        r'Haz\.\s+ratio',
        r'P>\|[tz]\|'
    ]
    combined_pat = '|'.join(f'({p})' for p in patterns) + r'|\S+'
    return [m.group(0) for m in re.finditer(combined_pat, header_right, re.IGNORECASE)]


def extract_table_blocks_from_text(text: str) -> list[list[str]]:
    """
    Extracts distinct table candidate blocks from text.
    Separates blocks by blank lines, command prompts, or ordinary non-table prose lines.
    Preserves multi-table outputs (e.g. ANOVA + Coefficient tables) as distinct blocks.
    """
    if not text or not text.strip():
        return []

    lines = text.splitlines()
    blocks = []
    current_block = []

    for line in lines:
        stripped = line.strip()
        # Empty line or command line marks a block boundary
        if not stripped or stripped.startswith('.'):
            if current_block:
                blocks.append(current_block)
                current_block = []
            continue

        is_border = bool(re.match(r'^[-\+=]{3,}', stripped)) or (set(stripped) <= {'-', '+', ' ', '|', '='} and len(stripped) >= 3)
        has_pipe = '|' in line

        if has_pipe or is_border:
            current_block.append(line)
        else:
            if current_block:
                blocks.append(current_block)
                current_block = []

    if current_block:
        blocks.append(current_block)

    return blocks


def extract_diagnostics_from_text(text: str) -> list[tuple[str, str]]:
    """
    Extracts regression/model diagnostics from console text.
    Preserves exact displayed numbers without rounding or loss.
    """
    pattern = re.compile(
        r'((?:Number of obs|F\s*\([^)=\n]+\)|Prob\s*>\s*[Ff]|Prob\s*>\s*chi2|Wald chi2\s*\([^)=\n]+\)|R-squared|Adj R-squared|Root MSE|Log likelihood|Pseudo R2|AIC|BIC))\s*=\s*'
        r'([-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?)',
        re.IGNORECASE
    )
    diagnostics = []
    seen = set()
    for match in pattern.finditer(text):
        raw_key = match.group(1).strip()
        clean_key = re.sub(r'\s+', ' ', raw_key)
        clean_key = re.sub(r'\(\s+', '(', clean_key)
        clean_key = re.sub(r'\s+\)', ')', clean_key)
        clean_key = re.sub(r',\s*', ', ', clean_key)
        val = match.group(2).strip()
        if clean_key.lower() not in seen:
            diagnostics.append((clean_key, val))
            seen.add(clean_key.lower())
    return diagnostics


def is_valid_table_block(table_lines: list[str]) -> bool:
    """
    Validates whether a candidate block represents a genuine statistical/tabular structure:
    1. At least 2 content rows (or header + separator + data row).
    2. Column delimiter '|' on every non-border line.
    3. Consistent column count and tabular/numerical evidence.
    4. Rejects single-line text or arbitrary non-table prose containing pipes (e.g. 'Status | pending').
    5. Rejects ambiguous schemas (e.g. 'a | b' followed by 'x | 1 2 3').
    """
    if not table_lines:
        return False

    content_rows = []
    has_sep = False

    for l in table_lines:
        stripped = l.strip()
        is_border = bool(re.match(r'^[-\+=]{3,}', stripped)) or (set(stripped) <= {'-', '+', ' ', '|', '='} and len(stripped) >= 3)
        if is_border and '|' not in l:
            has_sep = True
            continue
        if '|' in l:
            stub, right = l.split('|', 1)
            content_rows.append((stub.strip(), right.strip()))

    # Single-line text with a pipe is never a table (Defect C)
    if len(content_rows) < 2:
        return False

    first_stub_lower = content_rows[0][0].lower()
    first_right_lower = content_rows[0][1].lower()
    is_anova = first_stub_lower in ('source', 'model', 'residual') or 'number of obs' in first_right_lower

    if is_anova:
        data_rows = []
        for stub, right in content_rows:
            if stub.lower() == 'source':
                continue
            tokens = right.split()
            if tokens:
                data_rows.append(tokens)
        if not data_rows:
            return False
        for r in data_rows:
            if len(r) < 3:
                return False
        return True

    # Check for numerical content or recognized statistical keywords
    stat_keywords = {
        'variable', 'source', 'obs', 'mean', 'std', 'dev', 'min', 'max',
        'coefficient', 'coef', 'se', 'robust', 't', 'z', 'p', 'freq',
        'percent', 'cum', 'ss', 'df', 'ms', 'model', 'residual', 'total',
        'param', 'estimate', 'conf', 'interval', 'item', 'price', 'quantity'
    }

    has_stat_header = False
    header_tokens_raw = (content_rows[0][0] + " " + content_rows[0][1]).lower().split()
    for tok in header_tokens_raw:
        clean_tok = re.sub(r'[^a-z]', '', tok)
        if clean_tok in stat_keywords:
            has_stat_header = True
            break

    has_numeric_data = False
    for stub, right in content_rows[1:]:
        if re.search(r'[-+]?\d*\.?\d+', right) or re.search(r'\b\.\b', right):
            has_numeric_data = True
            break

    # If no border line and neither stat header nor numeric data, reject as prose
    if not has_sep and not (has_stat_header or has_numeric_data):
        return False

    # Structural consistency validation:
    header_right = content_rows[0][1]
    header_tokens = tokenize_header(header_right)
    if not header_tokens:
        return False

    data_rows = []
    for stub, right in content_rows[1:]:
        tokens = right.split()
        if tokens:
            data_rows.append((stub, tokens))

    if not data_rows:
        return False

    # Check non-total rows for consistency (tabulate 'Total' row omits cumulative pct)
    non_total_rows = [tokens for stub, tokens in data_rows if stub.lower() != 'total']
    target_rows = non_total_rows if non_total_rows else [tokens for stub, tokens in data_rows]

    col_counts = [len(tokens) for tokens in target_rows]
    if len(set(col_counts)) > 1:
        return False

    body_cols = col_counts[0]
    has_ci = bool(header_tokens and re.search(r'\[95%', header_tokens[-1], re.I))

    if has_ci:
        if body_cols != len(header_tokens) + 1:
            return False
    else:
        if body_cols != len(header_tokens):
            return False

    return True


def parse_table_structure(table_lines: list[str]):
    """
    Parses a validated table block into structured rows:
    Returns (is_anova, header_info, data_rows, body_cols) or None if invalid.
    """
    if not is_valid_table_block(table_lines):
        return None

    parsed_items = []
    is_anova = False

    for l in table_lines:
        stripped = l.strip()
        is_border = bool(re.match(r'^[-\+=]{3,}', stripped)) or (set(stripped) <= {'-', '+', ' ', '|', '='} and len(stripped) >= 3)
        if is_border and '|' not in l:
            parsed_items.append(('SEP', []))
            continue
        if '|' not in l:
            continue
        stub, right = l.split('|', 1)
        stub = stub.strip()
        right = right.strip()
        if stub.lower() in ('source', 'model', 'residual') or 'number of obs' in right.lower():
            is_anova = True
        parsed_items.append(('ROW', stub, right))

    if is_anova:
        # ANOVA table format: Source | SS df MS
        data_rows = []
        header_cells = ['Source', 'SS', 'df', 'MS']
        for item in parsed_items:
            if item[0] == 'ROW':
                stub, right = item[1], item[2]
                if stub.lower() == 'source':
                    continue
                tokens = right.split()
                if len(tokens) >= 3:
                    data_rows.append([stub, tokens[0], tokens[1], tokens[2]])
                else:
                    data_rows.append([stub] + tokens)
        return is_anova, header_cells, data_rows, 3

    content_rows = [item for item in parsed_items if item[0] == 'ROW']
    if not content_rows:
        return None

    header_stub, header_right = content_rows[0][1], content_rows[0][2]
    header_tokens = tokenize_header(header_right)

    data_rows = []
    data_col_counts = []

    for item in content_rows[1:]:
        stub, right = item[1], item[2]
        vals = right.split()
        if vals:
            data_rows.append([stub] + vals)
            if stub.lower() != 'total':
                data_col_counts.append(len(vals))

    if not data_rows:
        return None

    body_cols = data_col_counts[0] if data_col_counts else len(data_rows[0]) - 1
    return is_anova, [header_stub, header_right, header_tokens], data_rows, body_cols


def format_single_table_as_latex(table_lines: list[str], full_report: bool = False, model_diagnostics: list[tuple[str, str]] | None = None) -> str | None:
    """Formats a single validated table block as a LaTeX booktabs table."""
    struct = parse_table_structure(table_lines)
    if not struct:
        return None

    is_anova, header_info, data_rows, body_cols = struct
    total_cols = 1 + body_cols

    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\begin{{tabular}}{{l{'r' * body_cols}}}",
        "\\toprule"
    ]

    if is_anova:
        latex_lines.append(" & ".join([escape_latex(h) for h in header_info]) + " \\\\")
        latex_lines.append("\\midrule")
        for row in data_rows:
            if row[0].lower() == 'total':
                latex_lines.append("\\midrule")
            cells = [escape_latex(c) for c in row]
            while len(cells) < 4:
                cells.append("")
            latex_lines.append(" & ".join(cells[:4]) + " \\\\")
        latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        return "\n".join(latex_lines)

    header_stub, header_right, tokens = header_info
    latex_cells = [escape_latex(header_stub)]

    if body_cols > 0 and len(tokens) == body_cols - 1 and re.search(r'\[95%', tokens[-1], re.I):
        for tok in tokens[:-1]:
            latex_cells.append(escape_latex(tok))
        ci_tok = tokens[-1]
        latex_cells.append(f"\\multicolumn{{2}}{{c}}{{{escape_latex(ci_tok)}}}")
    else:
        for tok in tokens:
            latex_cells.append(escape_latex(tok))
        while len(latex_cells) < total_cols:
            latex_cells.append("")

    latex_lines.append(" & ".join(latex_cells[:total_cols]) + " \\\\")
    latex_lines.append("\\midrule")

    for row in data_rows:
        stub = row[0]
        if stub.lower() == 'total' and len(data_rows) > 3:
            latex_lines.append("\\midrule")
        cells = [escape_latex(c) for c in row]
        while len(cells) < total_cols:
            cells.append("")
        latex_lines.append(" & ".join(cells[:total_cols]) + " \\\\")

    if full_report and model_diagnostics:
        latex_lines.append("\\midrule")
        latex_lines.append(f"\\multicolumn{{{total_cols}}}{{l}}{{\\textbf{{Model Diagnostics}}}} \\\\")
        latex_lines.append("\\midrule")
        for diag_k, diag_v in model_diagnostics:
            k_esc = escape_latex(diag_k)
            v_esc = escape_latex(diag_v)
            if body_cols > 1:
                latex_lines.append(f"{k_esc} & \\multicolumn{{{body_cols}}}{{r}}{{{v_esc}}} \\\\")
            else:
                latex_lines.append(f"{k_esc} & {v_esc} \\\\")

    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(latex_lines)


def format_single_table_as_markdown(table_lines: list[str], full_report: bool = False, model_diagnostics: list[tuple[str, str]] | None = None) -> str | None:
    """Formats a single validated table block as a standard Markdown table."""
    struct = parse_table_structure(table_lines)
    if not struct:
        return None

    is_anova, header_info, data_rows, body_cols = struct
    total_cols = 1 + body_cols
    md_lines = []

    if is_anova:
        md_lines.append("| " + " | ".join([escape_markdown(h) for h in header_info]) + " |")
        md_lines.append("|:---|---:|---:|---:|")
        for row in data_rows:
            cells = [escape_markdown(c) for c in row]
            while len(cells) < 4:
                cells.append("")
            md_lines.append("| " + " | ".join(cells[:4]) + " |")
        return "\n".join(md_lines)

    header_stub, header_right, tokens = header_info
    md_cells = [escape_markdown(header_stub)]

    if body_cols > 0 and len(tokens) == body_cols - 1 and re.search(r'\[95%', tokens[-1], re.I):
        for tok in tokens[:-1]:
            md_cells.append(escape_markdown(tok))
        ci_tok = tokens[-1]
        md_cells.append(escape_markdown(f"{ci_tok} [Low]"))
        md_cells.append(escape_markdown(f"{ci_tok} [High]"))
    else:
        for tok in tokens:
            md_cells.append(escape_markdown(tok))
        while len(md_cells) < total_cols:
            md_cells.append("")

    md_lines.append("| " + " | ".join(md_cells[:total_cols]) + " |")
    md_lines.append("|:---|" + "|".join(["---:"] * (total_cols - 1)) + "|")

    for row in data_rows:
        cells = [escape_markdown(c) for c in row]
        while len(cells) < total_cols:
            cells.append("")
        md_lines.append("| " + " | ".join(cells[:total_cols]) + " |")

    if full_report and model_diagnostics:
        md_lines.append("")
        md_lines.append("### Model Diagnostics")
        md_lines.append("")
        md_lines.append("| Diagnostic | Value |")
        md_lines.append("|:---|---:|")
        for diag_k, diag_v in model_diagnostics:
            md_lines.append(f"| {escape_markdown(diag_k)} | {escape_markdown(diag_v)} |")

    return "\n".join(md_lines)


class AmbiguousReportingError(ValueError):
    """Raised when unsegmented plain text contains multiple model outputs without trustworthy boundaries."""
    pass


MODEL_HEADER_PATTERN = re.compile(
    r'(?m)^(?=\s*Source\s*\||'
    r'Linear regression|'
    r'Probit regression|'
    r'Logit regression|'
    r'Poisson regression|'
    r'Tobit regression|'
    r'Quantile regression|'
    r'HDFE Estimation|'
    r'High-dimensional|'
    r'Instrumental variables|'
    r'Difference-in-Differences|'
    r'Panel regression)'
)


def resolve_input_sections(
    text_or_target: Any,
    full_report: bool = False,
    extra_diagnostics: list[tuple[str, str]] | tuple | None = None
) -> list[tuple[str, list[tuple[str, str]] | None, str]]:
    """
    Resolves input into structured (section_text, explicit_diagnostics, command_label) tuples.
    Supports:
    1. CommandResult with output_sections.
    2. Sequence of OutputSection objects.
    3. Sequence of StatisticalResultRecord objects.
    4. Plain text / list of lines with command-prompt or model-header segmentation.
    5. Raises AmbiguousReportingError if unsegmented plain text has multiple model tables without boundaries.
    """
    if text_or_target is None:
        return []

    # Case 1: Structured CommandResult (which subclasses str, check attributes before isinstance(..., str))
    if hasattr(text_or_target, "output_sections") and text_or_target.output_sections:
        sections = []
        for sec in text_or_target.output_sections:
            sec_text = getattr(sec, "text", "")
            if sec_text and sec_text.strip():
                diags = extract_diagnostics_from_text(sec_text) if full_report else None
                cmd = getattr(sec, "command", "")
                sections.append((sec_text, diags, cmd))
        if sections:
            return sections

    if hasattr(text_or_target, "sections") and callable(text_or_target.sections):
        raw_secs = text_or_target.sections()
        if raw_secs and (len(raw_secs) > 1 or raw_secs[0].command != "" or raw_secs[0].is_model):
            sections = []
            for sec in raw_secs:
                sec_text = getattr(sec, "text", "")
                if sec_text and sec_text.strip():
                    diags = extract_diagnostics_from_text(sec_text) if full_report else None
                    cmd = getattr(sec, "command", "")
                    sections.append((sec_text, diags, cmd))
            if sections:
                return sections

    # Case 2: Explicit sequence of OutputSection or StatisticalResultRecord objects
    if isinstance(text_or_target, (list, tuple)) and text_or_target and not isinstance(text_or_target[0], str):
        sections = []
        for item in text_or_target:
            if hasattr(item, "output_text"):
                item_text = item.output_text
                item_diags = item.diagnostics if full_report else None
                cmd = getattr(item, "command_text", "")
            elif hasattr(item, "text"):
                item_text = item.text
                item_diags = extract_diagnostics_from_text(item_text) if full_report else None
                cmd = getattr(item, "command", "")
            else:
                continue
            if item_text and item_text.strip():
                sections.append((item_text, item_diags, cmd))
        if sections:
            return sections

    # Case 3: List of string lines
    if isinstance(text_or_target, list) and (not text_or_target or isinstance(text_or_target[0], str)):
        text = "\n".join(text_or_target)
    else:
        text = str(text_or_target)

    if not text or not text.strip():
        return []

    if extra_diagnostics is not None:
        return [(text, list(extra_diagnostics), "")]

    # Multi-command prompt segmentation: . regress ...
    if re.search(r'(?m)^\.\s+\w+', text):
        prompt_segments = [s for s in re.split(r'(?m)^(?=\.\s+)', text) if s.strip()]
        if len(prompt_segments) > 1:
            return [(s, extract_diagnostics_from_text(s) if full_report else None, "") for s in prompt_segments]

    # Model header segmentation (e.g. do-file concatenated output with multiple models)
    header_segments = [s for s in re.split(MODEL_HEADER_PATTERN, text) if s.strip()]
    if len(header_segments) > 1:
        valid_segs = [s for s in header_segments if extract_table_blocks_from_text(s)]
        if len(valid_segs) > 1:
            return [(s, extract_diagnostics_from_text(s) if full_report else None, "") for s in valid_segs]

    # Ambiguity check for plain text:
    blocks = extract_table_blocks_from_text(text)
    valid_blocks = [b for b in blocks if is_valid_table_block(b)]
    if len(valid_blocks) > 1 and full_report:
        structures = [parse_table_structure(b) for b in valid_blocks]
        coef_tables = [s for s in structures if s and not s[0]]
        anova_tables = [s for s in structures if s and s[0]]
        if len(coef_tables) > 1 or len(anova_tables) > 1:
            raise AmbiguousReportingError(
                "Ambiguous multi-model output: text contains multiple model tables without "
                "trustworthy command or header boundaries. Provide a CommandResult, "
                "OutputSection sequence, or pre-segmented text."
            )

    return [(text, extract_diagnostics_from_text(text) if full_report else None, "")]


def _format_single_section_latex(text: str, full_report: bool = False, diagnostics: list[tuple[str, str]] | None = None) -> str | None:
    blocks = extract_table_blocks_from_text(text)
    if not blocks:
        return None

    if diagnostics is None and full_report:
        diagnostics = extract_diagnostics_from_text(text)

    valid_blocks_with_idx = []
    for i, b in enumerate(blocks):
        if is_valid_table_block(b):
            valid_blocks_with_idx.append((i, b))

    if not valid_blocks_with_idx:
        return None

    target_diag_idx = valid_blocks_with_idx[-1][0]
    valid_tables = []
    for i, b in enumerate(blocks):
        attach = diagnostics if (full_report and i == target_diag_idx and diagnostics) else None
        tbl = format_single_table_as_latex(b, full_report=full_report, model_diagnostics=attach)
        if tbl:
            valid_tables.append(tbl)

    if not valid_tables:
        return None

    return "\n\n".join(valid_tables)


def _format_single_section_markdown(text: str, full_report: bool = False, diagnostics: list[tuple[str, str]] | None = None) -> str | None:
    blocks = extract_table_blocks_from_text(text)
    if not blocks:
        return None

    if diagnostics is None and full_report:
        diagnostics = extract_diagnostics_from_text(text)

    valid_blocks_with_idx = []
    for i, b in enumerate(blocks):
        if is_valid_table_block(b):
            valid_blocks_with_idx.append((i, b))

    if not valid_blocks_with_idx:
        return None

    target_diag_idx = valid_blocks_with_idx[-1][0]
    valid_tables = []
    for i, b in enumerate(blocks):
        attach = diagnostics if (full_report and i == target_diag_idx and diagnostics) else None
        tbl = format_single_table_as_markdown(b, full_report=full_report, model_diagnostics=attach)
        if tbl:
            valid_tables.append(tbl)

    if not valid_tables:
        return None

    return "\n\n".join(valid_tables)


def format_table_as_latex(text_or_lines, full_report: bool = False, extra_diagnostics: list[tuple[str, str]] | tuple | None = None) -> str | None:
    """
    Formats text containing one or multiple tables as publication-quality LaTeX booktabs.
    Preserves multiple independent tables without mixing their schemas.
    When full_report=True, includes applicable model diagnostics.
    Returns None if no valid table structure is detected.
    """
    sections = resolve_input_sections(text_or_lines, full_report=full_report, extra_diagnostics=extra_diagnostics)
    if not sections:
        return None

    formatted_sections = []
    for sec_text, sec_diags, _ in sections:
        res = _format_single_section_latex(sec_text, full_report=full_report, diagnostics=sec_diags)
        if res:
            formatted_sections.append(res)

    return "\n\n".join(formatted_sections) if formatted_sections else None


def format_table_as_markdown(text_or_lines, full_report: bool = False, extra_diagnostics: list[tuple[str, str]] | tuple | None = None) -> str | None:
    """
    Formats text containing one or multiple tables as standard Markdown tables.
    Preserves multiple independent tables without mixing their schemas.
    When full_report=True, includes applicable model diagnostics.
    Returns None if no valid table structure is detected.
    """
    sections = resolve_input_sections(text_or_lines, full_report=full_report, extra_diagnostics=extra_diagnostics)
    if not sections:
        return None

    formatted_sections = []
    for sec_text, sec_diags, _ in sections:
        res = _format_single_section_markdown(sec_text, full_report=full_report, diagnostics=sec_diags)
        if res:
            formatted_sections.append(res)

    return "\n\n".join(formatted_sections) if formatted_sections else None


def format_full_report_as_latex(text_or_lines, extra_diagnostics: list[tuple[str, str]] | None = None) -> str | None:
    """Formats full regression report including model diagnostics as LaTeX booktabs."""
    return format_table_as_latex(text_or_lines, full_report=True, extra_diagnostics=extra_diagnostics)


def format_full_report_as_markdown(text_or_lines, extra_diagnostics: list[tuple[str, str]] | None = None) -> str | None:
    """Formats full regression report including model diagnostics as Markdown."""
    return format_table_as_markdown(text_or_lines, full_report=True, extra_diagnostics=extra_diagnostics)


def extract_latest_table_lines(text: str) -> list[str]:
    """
    Extracts the latest table block from console text.
    Maintained for direct single-table queries.
    """
    blocks = extract_table_blocks_from_text(text)
    for b in reversed(blocks):
        if is_valid_table_block(b):
            return b
    return []


class ResultsConsole(QtWidgets.QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.setPlaceholderText("Run a command to see its results here.")
        self.document().setDocumentMargin(12)
        
        # Monospace font
        font = QtGui.QFont("Consolas", 10)
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        self.setFont(font)
        
        # Color formats
        self._fmt_cmd = QtGui.QTextCharFormat()
        self._fmt_cmd.setForeground(QtGui.QColor("#4EC9B0"))
        self._fmt_cmd.setFontWeight(QtGui.QFont.Weight.Bold)

        self._fmt_err = QtGui.QTextCharFormat()
        self._fmt_err.setForeground(QtGui.QColor("#F44747"))
        self._fmt_err.setFontWeight(QtGui.QFont.Weight.Bold)

        self._fmt_normal = QtGui.QTextCharFormat()

        # Execution tracking and model provenance
        self._execution_records: list[StatisticalResultRecord] = []
        self._pending_command: tuple[str, int, Any] | None = None

    def set_theme(self, dark: bool):
        """Recolor command and error spans without disturbing transcript positions."""
        command = QtGui.QColor("#4EC9B0" if dark else "#086751")
        error = QtGui.QColor("#FF8585" if dark else "#A52334")
        previous = {self._fmt_cmd.foreground().color().name(): command,
                    self._fmt_err.foreground().color().name(): error}
        self._fmt_cmd.setForeground(command)
        self._fmt_err.setForeground(error)
        spans = []
        block = self.document().begin()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                if fragment.isValid():
                    old = fragment.charFormat().foreground().color()
                    new = previous.get(old.name())
                    if new is not None and new != old:
                        spans.append((fragment.position(), fragment.length(), new))
                it += 1
            block = block.next()
        cursor = QtGui.QTextCursor(self.document())
        cursor.beginEditBlock()
        for start, length, color in spans:
            cursor.setPosition(start)
            cursor.setPosition(start + length, QtGui.QTextCursor.MoveMode.KeepAnchor)
            fmt = QtGui.QTextCharFormat()
            fmt.setForeground(color)
            cursor.mergeCharFormat(fmt)
        cursor.endEditBlock()

    def append_command(self, cmd_text: str, execution_id: Any = None):
        cursor = self.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        char_start = cursor.position()
        cursor.insertText(f". {cmd_text}\n", self._fmt_cmd)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
        self._pending_command = (cmd_text, char_start, execution_id)

    def append_output(self, output_text: str, execution_id: Any = None, command_result: Any = None, model_snapshot: dict[str, Any] | None = None, command_text: str = "", *, is_error=False):
        cursor = self.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        if self._pending_command:
            p_cmd, _, p_id = self._pending_command
            if execution_id is None or p_id == execution_id:
                command_text = p_cmd
                self._pending_command = None

        sections = getattr(command_result, "output_sections", None)
        snapshot = model_snapshot if model_snapshot is not None else getattr(command_result, "model_snapshot", None)
        if sections is None:
            sections = (OutputSection(command_text, output_text, snapshot is not None and not is_error),)
        fmt = self._fmt_err if is_error else self._fmt_normal
        for index, part in enumerate(sections):
            if index:
                cursor.insertText("\n", fmt)
            start = cursor.position()
            cursor.insertText(part.text, fmt)
            self._execution_records.append(StatisticalResultRecord(
                execution_id=execution_id, command_text=part.command, char_start=start,
                char_end=cursor.position(), output_text=part.text,
                model_snapshot=snapshot if part.is_model else None,
                diagnostics=extract_diagnostics_from_text(part.text), is_model=part.is_model,
            ))
        cursor.insertText("\n\n", fmt)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def append_error(self, err_text: str, execution_id: Any = None, command_text: str = "", command_result=None):
        self.append_output(err_text, execution_id, command_result=command_result,
                           command_text=command_text, is_error=True)

    def session_ranges(self):
        # Positions refer to console.txt in Qt UTF-16 units. The text itself is stored once.
        return [[r.char_start, r.char_end, r.command_text, r.is_model]
                for r in self._execution_records if r.char_end > r.char_start]

    def setPlainText(self, text):
        super().setPlainText(text)
        self._execution_records.clear()
        self._pending_command = None

    def restore_transcript(self, text, ranges):
        self.setPlainText(text)
        cursor = self.textCursor()
        limit = self.document().characterCount() - 1
        records = []
        previous_end = 0
        if not isinstance(ranges, list):
            return
        for item in ranges:
            if not isinstance(item, list) or len(item) != 4:
                return
            start, end, command, is_model = item
            if (type(start) is not int or type(end) is not int
                    or not previous_end <= start < end <= limit
                    or not isinstance(command, str) or type(is_model) is not bool):
                return
            cursor.setPosition(start)
            cursor.setPosition(end, QtGui.QTextCursor.MoveMode.KeepAnchor)
            output = cursor.selection().toPlainText()
            records.append(StatisticalResultRecord(
                command_text=command, char_start=start, char_end=end, output_text=output,
                diagnostics=extract_diagnostics_from_text(output), is_model=is_model,
            ))
            previous_end = end
        self._execution_records = records

    def clear(self):
        super().clear()
        self._execution_records.clear()
        self._pending_command = None

    def add_export_actions(self, menu):
        menu.addAction("Table as Markdown", self._copy_as_markdown)
        menu.addAction("Table as LaTeX", self._copy_as_latex)
        menu.addSeparator()
        menu.addAction("Model report as Markdown", self._copy_full_report_as_markdown)
        menu.addAction("Model report as LaTeX", self._copy_full_report_as_latex)

    def contextMenuEvent(self, event):
        menu = self.createStandardContextMenu()
        menu.addSeparator()
        self.add_export_actions(menu)
        menu.addSeparator()
        act_clear = menu.addAction("Clear results")
        act_clear.triggered.connect(self.clear)

        menu.exec(event.globalPos())

    def _notify_user(self, message: str, is_warning: bool = False):
        """Displays clear user feedback via main window status bar and cursor tooltip."""
        win = self.window()
        if win and hasattr(win, "statusBar") and win.statusBar():
            win.statusBar().showMessage(message, 4000)
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), message, self)

    def _resolve_target_records(self, model_only=False) -> tuple[list[StatisticalResultRecord], str, bool]:
        """
        Resolves the target records and text for an export action:
        1. If a text selection exists:
           - Finds all records overlapping with [selectionStart, selectionEnd].
           - Returns (matching_records, selected_text, True).
        2. If NO text selection exists:
           - Resolves strictly to the latest record containing a model report or statistical table.
           - Prevents whole-console dumps and cross-model contamination.
           - Returns ([latest_record], latest_record.output_text, False) if found,
             or ([], self.toPlainText(), False) if no records exist.
        """
        cursor = self.textCursor()
        if cursor.hasSelection():
            sel_start = cursor.selectionStart()
            sel_end = cursor.selectionEnd()
            sel_text = cursor.selectedText().replace('\u2029', '\n')
            matching = [
                r for r in self._execution_records
                if r.char_start < sel_end and r.char_end > sel_start
            ]
            return matching, sel_text, True

        for rec in reversed(self._execution_records):
            if model_only and not rec.is_model:
                continue
            if rec.is_model or rec.diagnostics or extract_table_blocks_from_text(rec.output_text):
                return [rec], rec.output_text, False

        return [], self.toPlainText(), False

    def _copy_as_latex(self):
        records, text, is_sel = self._resolve_target_records()
        if records:
            tables = []
            for rec in records:
                rec_text = text if (is_sel and len(records) == 1) else rec.output_text
                tbl = format_table_as_latex(rec_text, full_report=False)
                if tbl:
                    tables.append(tbl)
            if tables:
                ltx = "\n\n".join(tables)
                QtWidgets.QApplication.clipboard().setText(ltx)
                self._notify_user("Copied table as LaTeX (booktabs) to clipboard.")
                return

        ltx = format_table_as_latex(text, full_report=False)
        if ltx:
            QtWidgets.QApplication.clipboard().setText(ltx)
            self._notify_user("Copied table as LaTeX (booktabs) to clipboard.")
        else:
            fallback = text if is_sel else self.toPlainText()
            QtWidgets.QApplication.clipboard().setText(fallback)
            self._notify_user("No valid statistical table detected. Plain text copied to clipboard.", is_warning=True)

    def _copy_as_markdown(self):
        records, text, is_sel = self._resolve_target_records()
        if records:
            tables = []
            for rec in records:
                rec_text = text if (is_sel and len(records) == 1) else rec.output_text
                tbl = format_table_as_markdown(rec_text, full_report=False)
                if tbl:
                    tables.append(tbl)
            if tables:
                md = "\n\n".join(tables)
                QtWidgets.QApplication.clipboard().setText(md)
                self._notify_user("Copied table as Markdown to clipboard.")
                return

        md = format_table_as_markdown(text, full_report=False)
        if md:
            QtWidgets.QApplication.clipboard().setText(md)
            self._notify_user("Copied table as Markdown to clipboard.")
        else:
            fallback = text if is_sel else self.toPlainText()
            QtWidgets.QApplication.clipboard().setText(fallback)
            self._notify_user("No valid statistical table detected. Plain text copied to clipboard.", is_warning=True)

    def _copy_full_report_as_latex(self):
        self._copy_model_report("LaTeX")

    def _copy_full_report_as_markdown(self):
        self._copy_model_report("Markdown")

    def _copy_model_report(self, format_name):
        latex = format_name == "LaTeX"
        report_formatter = format_full_report_as_latex if latex else format_full_report_as_markdown
        table_formatter = format_table_as_latex if latex else format_table_as_markdown
        records, text, selected = self._resolve_target_records(model_only=True)
        models = [record for record in records if record.is_model]
        reports = []
        for model in models:
            body = report_formatter(model.output_text, extra_diagnostics=model.diagnostics)
            if not body:
                break
            command = " ".join(model.command_text.splitlines())
            heading = r"\noindent\texttt{" + escape_latex(command) + r"}\par" if latex else "    " + command
            reports.append(f"{heading}\n\n{body}" if command else body)
        if reports and len(reports) == len(models):
            QtWidgets.QApplication.clipboard().setText("\n\n".join(reports))
            self._notify_user(f"Copied full model report as {format_name} to clipboard.")
            return

        # Old session files contain plain text with no trustworthy model boundaries.
        if not selected:
            text = "\n".join(extract_latest_table_lines(self.toPlainText())) or text
        table = table_formatter(text)
        QtWidgets.QApplication.clipboard().setText(table or text)
        message = ("Copied table only. A full model report is unavailable for this output."
                   if table else "No valid model table detected. Plain text copied to clipboard.")
        self._notify_user(message, is_warning=True)
