"""
engine/parser.py - Command Grammar Parser and Whitelisted AST Evaluator
Implements:
1. Top-level clause splitting: command [varlist] [if exp] [in range] [using file] [, options]
2. Top-level comma splitting outside quotes, parentheses, brackets, and braces.
3. Whitelisted AST expression translator for command expressions (strictly prohibits eval/exec/query injection).
4. generate/replace functions: log, ln, exp, sqrt, abs, round, int, floor, ceil, max, min, mod, missing, inlist, inrange.
5. Special variables (_n, _N) and time-series lag/lead/diff operators (L., L2., D., F.).
6. Factor-variable expansion (i.var, ib#.var, c.var, i.a#i.b, c.x##i.d).
7. Structured command execution result (CommandResult).
"""

import ast
import re
import math
import datetime
from typing import NamedTuple
import numpy as np
import pandas as pd


class OutputSection(NamedTuple):
    """Displayed output of one script command, without retaining its fitted model."""
    command: str
    text: str
    is_model: bool


class CommandResult(str):
    """
    Subclass of str providing backward-compatible string returns
    while exposing structured fields (output, ok, error, return_code, model_snapshot, command_text).
    """
    def __new__(cls, output="", ok=True, error=None, return_code=0, model_snapshot=None, command_text=""):
        obj = str.__new__(cls, output)
        obj.output = output
        obj.ok = ok
        obj.error = error
        obj.return_code = return_code
        obj.model_snapshot = model_snapshot
        obj.command_text = command_text
        obj.output_sections = None
        return obj

    def sections(self):
        if self.output_sections is not None:
            return self.output_sections
        return (OutputSection(self.command_text, self.output, self.ok and self.model_snapshot is not None),)

    @classmethod
    def from_sections(cls, sections, failure=None):
        result = cls("\n".join(part.text for part in sections),
                     ok=failure is None,
                     error=failure.error if failure is not None else None,
                     return_code=failure.return_code if failure is not None else 0)
        result.output_sections = sections
        return result

    def __repr__(self):
        return f"CommandResult(ok={self.ok}, return_code={self.return_code}, output={repr(self.output)})"


class ParsedCommand:
    """Represents a fully decomposed command."""
    def __init__(self, verb, body="", if_expr=None, in_range=None, using_file=None, options=None, raw=""):
        self.verb = verb.lower() if verb else ""
        self.body = body.strip()
        self.if_expr = if_expr.strip() if if_expr else None
        self.in_range = in_range  # tuple (start_1based, end_1based) or None
        self.using_file = using_file.strip() if using_file else None
        self.options = options if options is not None else {}
        self.raw = raw

    @property
    def name(self):
        return self.verb

    @property
    def command(self):
        return self.verb

    @property
    def if_exp(self):
        return self.if_expr

    @property
    def using(self):
        return self.using_file

    def has_option(self, name):
        return name.lower() in self.options

    def get_option(self, name, default=None):
        val = self.options.get(name.lower(), default)
        if isinstance(val, str):
            val = val.strip('\'"')
        return val

    def __repr__(self):
        return (f"ParsedCommand(verb='{self.verb}', body='{self.body}', "
                f"if='{self.if_expr}', in={self.in_range}, using='{self.using_file}', "
                f"options={self.options})")


def split_top_level(text, delimiter=','):
    """
    Splits text by delimiter only when the delimiter appears at the top level,
    outside single/double quotes and parentheses/brackets/braces.
    Returns (before_delimiter, after_delimiter). If delimiter not found, after is "".
    """
    in_single = False
    in_double = False
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0

    for idx, ch in enumerate(text):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == '(':
                paren_depth += 1
            elif ch == ')':
                paren_depth = max(0, paren_depth - 1)
            elif ch == '[':
                bracket_depth += 1
            elif ch == ']':
                bracket_depth = max(0, bracket_depth - 1)
            elif ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth = max(0, brace_depth - 1)
            elif ch == delimiter and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                return text[:idx].strip(), text[idx + 1:].strip()

    return text.strip(), ""


def parse_options(options_str):
    """
    Parses an options string into a dictionary supporting:
    - boolean flags: robust, clear, replace, detail
    - options with arguments: vce(robust), vce(cluster id), arima(1,0,0)
    - nested parentheses: step(1, (2, 3))
    """
    if not options_str or not options_str.strip():
        return {}

    opts = {}
    i = 0
    n = len(options_str)

    while i < n:
        # Skip whitespace and commas between options
        while i < n and (options_str[i].isspace() or options_str[i] == ','):
            i += 1
        if i >= n:
            break

        # Read option name (alphanumeric and underscores)
        name_start = i
        while i < n and (options_str[i].isalnum() or options_str[i] in ('_', '-')):
            i += 1
        opt_name = options_str[name_start:i].lower()
        if not opt_name:
            i += 1
            continue

        # Check if immediately followed by '('
        while i < n and options_str[i].isspace():
            i += 1

        if i < n and options_str[i] == '(':
            # Parse argument within balanced parentheses
            i += 1
            paren_depth = 1
            arg_start = i
            while i < n and paren_depth > 0:
                if options_str[i] == '(':
                    paren_depth += 1
                elif options_str[i] == ')':
                    paren_depth -= 1
                i += 1
            arg_end = i - 1 if paren_depth == 0 else i
            arg_val = options_str[arg_start:arg_end].strip()
            opts[opt_name] = arg_val
        else:
            opts[opt_name] = True

    return opts


def parse_in_range(in_str, total_n=None):
    """
    Parses an 'in' range clause argument into 1-based (start, end) tuple.
    Examples:
    - '1/10' -> (1, 10)
    - '5' -> (5, 5)
    - '-10/-1' -> (-10, -1)
    - '1/l' or '1/L' -> (1, total_n) if total_n else (1, 'L')
    - 'f/10' -> (1, 10)
    """
    s = in_str.strip().lower()
    if '/' in s:
        parts = s.split('/', 1)
        p1, p2 = parts[0].strip(), parts[1].strip()
        
        # start
        if p1 in ('f', '1'):
            start = 1
        elif p1 in ('l', '-1'):
            start = total_n if total_n else -1
        else:
            try:
                start = int(p1)
            except ValueError:
                raise ValueError(f"invalid range start in 'in {in_str}'")

        # end
        if p2 == 'l':
            end = total_n if total_n else 'L'
        elif p2 == 'f':
            end = 1
        else:
            try:
                end = int(p2)
            except ValueError:
                raise ValueError(f"invalid range end in 'in {in_str}'")
        return (start, end)
    else:
        # Single observation
        if s in ('l', '-1'):
            val = total_n if total_n else -1
        elif s == 'f':
            val = 1
        else:
            try:
                val = int(s)
            except ValueError:
                raise ValueError(f"invalid observation number in 'in {in_str}'")
        return (val, val)


def parse_command_line(cmd_line):
    """
    Decomposes a full command line:
    command [body] [using file] [if exp] [in range] [, options]
    """
    cmd_line = cmd_line.strip()
    if not cmd_line or cmd_line.startswith("*") or cmd_line.startswith("//"):
        return ParsedCommand(verb="", raw=cmd_line)

    # 1. Separate options from command body at top-level comma
    head, options_part = split_top_level(cmd_line, ',')
    options = parse_options(options_part)

    # 2. Tokenize outside quotes to isolate verb, using, if, in
    in_single = False
    in_double = False
    paren_depth = 0
    tokens_with_indices = []
    
    current_word = []
    word_start = 0
    
    for idx, ch in enumerate(head):
        if ch == "'" and not in_double:
            in_single = not in_single
            current_word.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            current_word.append(ch)
        elif not in_single and not in_double:
            if ch == '(':
                paren_depth += 1
                current_word.append(ch)
            elif ch == ')':
                paren_depth = max(0, paren_depth - 1)
                current_word.append(ch)
            elif ch.isspace() and paren_depth == 0:
                if current_word:
                    tokens_with_indices.append((''.join(current_word), word_start, idx))
                    current_word = []
                word_start = idx + 1
            else:
                if not current_word:
                    word_start = idx
                current_word.append(ch)
        else:
            current_word.append(ch)
            
    if current_word:
        tokens_with_indices.append((''.join(current_word), word_start, len(head)))

    if not tokens_with_indices:
        return ParsedCommand(verb="", raw=cmd_line)

    verb = tokens_with_indices[0][0]

    # Special handling for generate/replace with '=':
    # e.g. gen x = round(price, 0.1) if price > 0
    # e.g. replace x = 10 in 1/5
    # The 'using', 'if', 'in' must only be searched AFTER the '=' expression or in remaining clauses.
    if_expr = None
    in_range = None
    using_file = None

    # Search for top-level keywords 'using', 'if', 'in' outside parens/quotes
    # We find keywords from right-to-left or left-to-right outside parens
    # Keyword candidates: 'if', 'in', 'using'
    clause_positions = []
    for tok, start, end in tokens_with_indices[1:]:
        tok_low = tok.lower()
        if tok_low in ('if', 'in', 'using'):
            clause_positions.append((tok_low, start, end))

    # Sort clauses by appearance in string
    clause_positions.sort(key=lambda x: x[1])

    if not clause_positions:
        # No if/in/using clauses
        body = head[tokens_with_indices[0][2]:].strip()
        return ParsedCommand(verb=verb, body=body, options=options, raw=cmd_line)

    # The body is everything between verb and the first clause
    first_clause_start = clause_positions[0][1]
    body = head[tokens_with_indices[0][2]:first_clause_start].strip()

    # Extract each clause text
    for i, (kw, start, end) in enumerate(clause_positions):
        next_start = clause_positions[i + 1][1] if i + 1 < len(clause_positions) else len(head)
        clause_content = head[end:next_start].strip()
        
        if kw == 'if':
            if_expr = clause_content
        elif kw == 'in':
            in_range = parse_in_range(clause_content)
        elif kw == 'using':
            using_file = clause_content.strip('\'"')

    return ParsedCommand(
        verb=verb,
        body=body,
        if_expr=if_expr,
        in_range=in_range,
        using_file=using_file,
        options=options,
        raw=cmd_line
    )


# =====================================================================
# Whitelisted AST Expression Evaluator
# =====================================================================

ALLOWED_AST_NODES = (
    ast.Expression, ast.UnaryOp, ast.BinOp, ast.Compare, ast.BoolOp,
    ast.Call, ast.Name, ast.Constant, ast.Load,
    # Unary operators
    ast.Not, ast.USub, ast.UAdd,
    # Binary operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    # Comparisons
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    # Boolean logic
    ast.And, ast.Or
)

def _dta_round(x, y=1.0):
    """
    round(x, y) rounds to the nearest multiple of y: y * round(x / y).
    Ties round away from zero (e.g. round(2.5, 5) == 5, round(-2.5, 5) == -5).
    """
    if y == 0:
        return x
    z = x / y
    if isinstance(z, pd.Series):
        rounded_z = np.sign(z) * np.floor(np.abs(z) + 0.5 + 1e-15)
        return rounded_z * y
    else:
        z_flt = float(z)
        sign = 1.0 if z_flt >= 0 else -1.0
        rounded_z = sign * np.floor(abs(z_flt) + 0.5 + 1e-15)
        return float(rounded_z * float(y))


def _dta_max(*args):
    """max() ignores missing (.) values unless all arguments are missing."""
    if not args:
        return np.nan
    is_series = any(isinstance(a, pd.Series) for a in args)
    if is_series:
        ref_idx = next(a.index for a in args if isinstance(a, pd.Series))
        res = None
        for a in args:
            a_s = a if isinstance(a, pd.Series) else pd.Series(a, index=ref_idx)
            if res is None:
                res = a_s.copy()
            else:
                res = np.fmax(res, a_s)
        return res
    else:
        valid = [float(a) for a in args if not pd.isna(a)]
        return max(valid) if valid else np.nan


def _dta_min(*args):
    """min() ignores missing (.) values unless all arguments are missing."""
    if not args:
        return np.nan
    is_series = any(isinstance(a, pd.Series) for a in args)
    if is_series:
        ref_idx = next(a.index for a in args if isinstance(a, pd.Series))
        res = None
        for a in args:
            a_s = a if isinstance(a, pd.Series) else pd.Series(a, index=ref_idx)
            if res is None:
                res = a_s.copy()
            else:
                res = np.fmin(res, a_s)
        return res
    else:
        valid = [float(a) for a in args if not pd.isna(a)]
        return min(valid) if valid else np.nan


def _dta_compare(left, right, op_type):
    """
    Implements reference comparison semantics.
    For strings:
      Missing values evaluate as "" (empty string). Comparisons are alphabetical.
    For numerics:
      Missing values (.) evaluate as +infinity.
    """
    is_left_s = isinstance(left, pd.Series)
    is_right_s = isinstance(right, pd.Series)

    is_left_str = isinstance(left, str) or (
        is_left_s and (pd.api.types.is_string_dtype(left) or pd.api.types.is_object_dtype(left))
    )
    is_right_str = isinstance(right, str) or (
        is_right_s and (pd.api.types.is_string_dtype(right) or pd.api.types.is_object_dtype(right))
    )

    if is_left_str or is_right_str:
        if is_left_s or is_right_s:
            idx = left.index if is_left_s else right.index
            if is_left_s:
                l_s = left.fillna("").astype(str)
            else:
                l_s = pd.Series("" if pd.isna(left) else str(left), index=idx)
            if is_right_s:
                r_s = right.fillna("").astype(str)
            else:
                r_s = pd.Series("" if pd.isna(right) else str(right), index=idx)

            if op_type == 'eq':
                res = l_s == r_s
            elif op_type == 'ne':
                res = l_s != r_s
            elif op_type == 'lt':
                res = l_s < r_s
            elif op_type == 'le':
                res = l_s <= r_s
            elif op_type == 'gt':
                res = l_s > r_s
            elif op_type == 'ge':
                res = l_s >= r_s
            else:
                raise ValueError(f"unknown op {op_type}")
            return pd.Series(res, index=idx, dtype=bool)
        else:
            l_val = "" if pd.isna(left) else str(left)
            r_val = "" if pd.isna(right) else str(right)
            if op_type == 'eq':
                return l_val == r_val
            elif op_type == 'ne':
                return l_val != r_val
            elif op_type == 'lt':
                return l_val < r_val
            elif op_type == 'le':
                return l_val <= r_val
            elif op_type == 'gt':
                return l_val > r_val
            elif op_type == 'ge':
                return l_val >= r_val
            else:
                raise ValueError(f"unknown op {op_type}")

    # Numeric comparison
    if is_left_s or is_right_s:
        idx = left.index if is_left_s else right.index
        l_arr = left.to_numpy(dtype=float) if is_left_s else np.full(len(idx), float(left))
        r_arr = right.to_numpy(dtype=float) if is_right_s else np.full(len(idx), float(right))
        
        l_nan = np.isnan(l_arr)
        r_nan = np.isnan(r_arr)
        
        l_val = np.where(l_nan, np.inf, l_arr)
        r_val = np.where(r_nan, np.inf, r_arr)
        
        if op_type == 'eq':
            res = (l_nan & r_nan) | (~l_nan & ~r_nan & (l_arr == r_arr))
        elif op_type == 'ne':
            res = ~((l_nan & r_nan) | (~l_nan & ~r_nan & (l_arr == r_arr)))
        elif op_type == 'lt':
            res = (l_val < r_val) & ~(l_nan & r_nan)
        elif op_type == 'le':
            res = (l_val < r_val) | (l_nan & r_nan) | (~l_nan & ~r_nan & (l_arr == r_arr))
        elif op_type == 'gt':
            res = (l_val > r_val) & ~(l_nan & r_nan)
        elif op_type == 'ge':
            res = (l_val > r_val) | (l_nan & r_nan) | (~l_nan & ~r_nan & (l_arr == r_arr))
        else:
            raise ValueError(f"unknown op {op_type}")
            
        return pd.Series(res, index=idx, dtype=bool)
    else:
        l_nan = pd.isna(left)
        r_nan = pd.isna(right)
        
        if l_nan and r_nan:
            if op_type in ('eq', 'le', 'ge'):
                return True
            return False
            
        l_val = np.inf if l_nan else float(left)
        r_val = np.inf if r_nan else float(right)
        
        if op_type == 'eq':
            return l_val == r_val
        elif op_type == 'ne':
            return l_val != r_val
        elif op_type == 'lt':
            return l_val < r_val
        elif op_type == 'le':
            return l_val <= r_val
        elif op_type == 'gt':
            return l_val > r_val
        elif op_type == 'ge':
            return l_val >= r_val
        else:
            raise ValueError(f"unknown op {op_type}")


_DTA_EPOCH = datetime.date(1960, 1, 1)

def _dta_mdy(m, d, y):
    if isinstance(m, pd.Series) or isinstance(d, pd.Series) or isinstance(y, pd.Series):
        s_m = pd.Series(m)
        s_d = pd.Series(d)
        s_y = pd.Series(y)
        res = []
        for mi, di, yi in zip(s_m, s_d, s_y):
            if pd.isna(mi) or pd.isna(di) or pd.isna(yi):
                res.append(np.nan)
            else:
                try:
                    dt = datetime.date(int(yi), int(mi), int(di))
                    res.append(int((dt - _DTA_EPOCH).days))
                except Exception:
                    res.append(np.nan)
        return pd.Series(res, index=s_m.index if isinstance(m, pd.Series) else (s_d.index if isinstance(d, pd.Series) else s_y.index))
    else:
        if pd.isna(m) or pd.isna(d) or pd.isna(y):
            return np.nan
        dt = datetime.date(int(y), int(m), int(d))
        return int((dt - _DTA_EPOCH).days)

def _dta_year(d):
    if isinstance(d, pd.Series):
        return d.apply(lambda x: int((_DTA_EPOCH + datetime.timedelta(days=int(x))).year) if pd.notna(x) else np.nan)
    return int((_DTA_EPOCH + datetime.timedelta(days=int(d))).year) if pd.notna(d) else np.nan

def _dta_month(d):
    if isinstance(d, pd.Series):
        return d.apply(lambda x: int((_DTA_EPOCH + datetime.timedelta(days=int(x))).month) if pd.notna(x) else np.nan)
    return int((_DTA_EPOCH + datetime.timedelta(days=int(d))).month) if pd.notna(d) else np.nan

def _dta_day(d):
    if isinstance(d, pd.Series):
        return d.apply(lambda x: int((_DTA_EPOCH + datetime.timedelta(days=int(x))).day) if pd.notna(x) else np.nan)
    return int((_DTA_EPOCH + datetime.timedelta(days=int(d))).day) if pd.notna(d) else np.nan

def _dta_quarter(d):
    if isinstance(d, pd.Series):
        return d.apply(lambda x: int(((_DTA_EPOCH + datetime.timedelta(days=int(x))).month - 1) // 3 + 1) if pd.notna(x) else np.nan)
    return int(((_DTA_EPOCH + datetime.timedelta(days=int(d))).month - 1) // 3 + 1) if pd.notna(d) else np.nan

def _dta_date_parse(s, fmt=None):
    if isinstance(s, pd.Series):
        return s.apply(lambda x: int((pd.to_datetime(str(x)).date() - _DTA_EPOCH).days) if pd.notna(x) else np.nan)
    if pd.isna(s):
        return np.nan
    dt = pd.to_datetime(str(s)).date()
    return int((dt - _DTA_EPOCH).days)

def _dta_trim(s):
    if isinstance(s, pd.Series):
        return s.astype(str).str.strip()
    return str(s).strip() if pd.notna(s) else ""

def _dta_itrim(s):
    if isinstance(s, pd.Series):
        return s.astype(str).apply(lambda x: re.sub(r'\s+', ' ', str(x).strip()) if pd.notna(x) else "")
    return re.sub(r'\s+', ' ', str(s).strip()) if pd.notna(s) else ""

def _dta_lower(s):
    if isinstance(s, pd.Series):
        return s.astype(str).str.lower()
    return str(s).lower() if pd.notna(s) else ""

def _dta_upper(s):
    if isinstance(s, pd.Series):
        return s.astype(str).str.upper()
    return str(s).upper() if pd.notna(s) else ""

def _dta_proper(s):
    if isinstance(s, pd.Series):
        return s.astype(str).str.title()
    return str(s).title() if pd.notna(s) else ""

def _dta_substr(s, n1, n2=None):
    def sub_single(val, start, length):
        if pd.isna(val):
            return ""
        st = str(val)
        start = int(start)
        if start > 0:
            py_start = start - 1
        elif start < 0:
            py_start = len(st) + start
        else:
            py_start = 0
        if py_start < 0:
            py_start = 0
        if length is None:
            return st[py_start:]
        length = int(length)
        if length <= 0:
            return ""
        return st[py_start:py_start + length]

    if isinstance(s, pd.Series):
        return s.apply(lambda val: sub_single(val, n1, n2))
    return sub_single(s, n1, n2)

def _dta_strpos(s1, s2):
    def pos_single(haystack, needle):
        if pd.isna(haystack) or pd.isna(needle):
            return 0
        idx = str(haystack).find(str(needle))
        return idx + 1 if idx >= 0 else 0

    if isinstance(s1, pd.Series):
        return s1.apply(lambda val: pos_single(val, s2))
    return pos_single(s1, s2)

def _dta_regexm(s, regex):
    def match_single(val, pattern):
        if pd.isna(val) or pd.isna(pattern):
            return 0
        return 1 if re.search(str(pattern), str(val)) else 0

    if isinstance(s, pd.Series):
        return s.apply(lambda val: match_single(val, regex))
    return match_single(s, regex)


WHITELISTED_FUNCTIONS = {
    'log': np.log,
    'ln': np.log,
    'exp': np.exp,
    'sqrt': np.sqrt,
    'abs': np.abs,
    'round': _dta_round,
    'int': np.trunc,
    'floor': np.floor,
    'ceil': np.ceil,
    'max': _dta_max,
    'min': _dta_min,
    'mod': np.mod,
    'missing': pd.isna,
    'inlist': lambda x, *args: pd.Series(x).isin(args) if hasattr(x, '__iter__') and not isinstance(x, str) else (x in args),
    'inrange': lambda x, a, b: (x >= a) & (x <= b),
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'mdy': _dta_mdy,
    'year': _dta_year,
    'month': _dta_month,
    'day': _dta_day,
    'quarter': _dta_quarter,
    'date': _dta_date_parse,
    'trim': _dta_trim,
    'itrim': _dta_itrim,
    'lower': _dta_lower,
    'upper': _dta_upper,
    'proper': _dta_proper,
    'substr': _dta_substr,
    'strpos': _dta_strpos,
    'regexm': _dta_regexm
}

# AST Security & Resource Exhaustion Thresholds
MAX_EXPR_LENGTH = 10000
MAX_AST_NODES = 500
MAX_AST_DEPTH = 30
MAX_INT_VALUE = 10**100
MAX_INT_BITS = 4096
MAX_EXPONENT = 1000

WHITELISTED_FUNCTION_ARITY = {
    'log': (1, 1),
    'ln': (1, 1),
    'exp': (1, 1),
    'sqrt': (1, 1),
    'abs': (1, 1),
    'round': (1, 2),
    'int': (1, 1),
    'floor': (1, 1),
    'ceil': (1, 1),
    'max': (1, 100),
    'min': (1, 100),
    'mod': (2, 2),
    'missing': (1, 1),
    'inlist': (2, 255),
    'inrange': (3, 3),
    'sin': (1, 1),
    'cos': (1, 1),
    'tan': (1, 1),
    'mdy': (3, 3),
    'year': (1, 1),
    'month': (1, 1),
    'day': (1, 1),
    'quarter': (1, 1),
    'date': (2, 3),
    'trim': (1, 1),
    'itrim': (1, 1),
    'lower': (1, 1),
    'upper': (1, 1),
    'proper': (1, 1),
    'substr': (2, 3),
    'strpos': (2, 2),
    'regexm': (2, 2)
}


def _get_ast_depth(node, current_depth=1):
    max_d = current_depth
    for child in ast.iter_child_nodes(node):
        d = _get_ast_depth(child, current_depth + 1)
        if d > max_d:
            max_d = d
    return max_d


def preprocess_dta_expression(expr_str):
    """
    Translates operator syntax into Python/AST-compatible syntax:
    - '~=' -> '!='
    - '^' -> '**'
    - '!' or '~' -> ' not '
    - '&' -> ' and '
    - '|' -> ' or '
    - Standalone '.' -> '_dta_missing_'
    - Handles time series dot notation: L.var -> _ts_L1_var_, etc.
    Extracts quoted string literals before regex substitutions and restores them afterwards.
    """
    s = expr_str.strip()

    # 1. Mask quoted string literals ("..." and '...')
    str_literals = []
    def mask_str(match):
        idx = len(str_literals)
        str_literals.append(match.group(0))
        return f"__STR_LIT_{idx}__"

    s = re.sub(r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')', mask_str, s)

    # 2. Time series operators pattern: e.g. L.var, L2.var, D.var, F.var, L1.var
    def replace_ts(match):
        op = match.group(1).upper()
        order = match.group(2) if match.group(2) else "1"
        var = match.group(3)
        return f"_ts_{op}{order}_{var}_"

    s = re.sub(r'\b([LDFldf])(\d*)\.([a-zA-Z0-9_]+)', replace_ts, s)

    # 3. Return / estimation register symbols:
    s = re.sub(r'_b\[([a-zA-Z0-9_]+)\]', r'_est_b_\1_', s)
    s = re.sub(r'_se\[([a-zA-Z0-9_]+)\]', r'_est_se_\1_', s)
    s = re.sub(r'\br\(([a-zA-Z0-9_]+)\)', r'_ret_r_\1_', s)
    s = re.sub(r'\be\(([a-zA-Z0-9_]+)\)', r'_ret_e_\1_', s)

    # 4. Standalone '.' (numeric missing) not part of float like 0.5
    s = re.sub(r'(?<![0-9a-zA-Z_])\.(?![0-9a-zA-Z_])', '_dta_missing_', s)

    # 5. Replace operators outside quotes
    s = re.sub(r'~=', '!=', s)
    s = re.sub(r'\^', '**', s)
    s = re.sub(r'&', ' and ', s)
    s = re.sub(r'\|', ' or ', s)
    s = re.sub(r'[!~](?!=)', ' not ', s)

    # 6. Unmask quoted string literals
    for idx, lit in enumerate(str_literals):
        s = s.replace(f"__STR_LIT_{idx}__", lit)

    return s


def validate_ast(node):
    """Recursively validates that the AST contains ONLY whitelisted, safe nodes and respects resource bounds."""
    # 1. Total node count
    nodes = list(ast.walk(node))
    if len(nodes) > MAX_AST_NODES:
        raise ValueError(f"expression exceeds maximum AST node complexity limit ({len(nodes)} > {MAX_AST_NODES})")

    # 2. Maximum AST depth
    depth = _get_ast_depth(node)
    if depth > MAX_AST_DEPTH:
        raise ValueError(f"expression exceeds maximum nesting depth limit ({depth} > {MAX_AST_DEPTH})")

    for sub in nodes:
        if not isinstance(sub, ALLOWED_AST_NODES):
            raise ValueError(f"disallowed syntax or operator: {type(sub).__name__}")

        if isinstance(sub, ast.Constant):
            if isinstance(sub.value, int):
                if abs(sub.value) > MAX_INT_VALUE or sub.value.bit_length() > MAX_INT_BITS:
                    raise ValueError(f"integer literal exceeds maximum allowed size ({MAX_INT_BITS} bits)")

        elif isinstance(sub, ast.Call):
            if not isinstance(sub.func, ast.Name):
                raise ValueError("disallowed function call: attribute access and method calls are prohibited")
            func_name = sub.func.id.lower()
            if func_name not in WHITELISTED_FUNCTIONS:
                raise ValueError(f"unknown or disallowed function: '{sub.func.id}'")
            if sub.keywords:
                raise ValueError("keyword arguments are not permitted in expressions")
            arity = WHITELISTED_FUNCTION_ARITY.get(func_name)
            if arity:
                min_a, max_a = arity
                num_a = len(sub.args)
                if num_a < min_a or num_a > max_a:
                    if min_a == max_a:
                        raise ValueError(f"function '{func_name}' expects exactly {min_a} argument(s), got {num_a}")
                    else:
                        raise ValueError(f"function '{func_name}' expects between {min_a} and {max_a} arguments, got {num_a}")

        elif isinstance(sub, ast.BinOp):
            if isinstance(sub.op, ast.Pow):
                if isinstance(sub.right, ast.Constant) and isinstance(sub.right.value, (int, float)):
                    if abs(sub.right.value) > MAX_EXPONENT:
                        raise ValueError(f"exponent magnitude ({sub.right.value}) exceeds maximum limit ({MAX_EXPONENT})")


def evaluate_dta_expression(expr_str, df=None, time_context=None, scalar_context=None):
    """
    Safely evaluates a command expression against a DataFrame or scalar context using a whitelisted AST visitor.
    Returns a pandas Series or scalar value.
    """
    if not expr_str or not expr_str.strip():
        raise ValueError("empty expression")

    if len(expr_str) > MAX_EXPR_LENGTH:
        raise ValueError(f"expression length exceeds maximum limit ({len(expr_str)} > {MAX_EXPR_LENGTH})")

    clean_expr = preprocess_dta_expression(expr_str)

    try:
        parsed_ast = ast.parse(clean_expr, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"invalid syntax in expression: '{expr_str}'") from e

    validate_ast(parsed_ast)

    # Build evaluation environment with df columns and whitelisted functions
    env = {}
    for name, func in WHITELISTED_FUNCTIONS.items():
        env[name] = func

    if df is not None and not df.empty:
        n_rows = len(df)
        env['_n'] = pd.Series(np.arange(1, n_rows + 1), index=df.index)
        env['_N'] = n_rows
        # Add dataset columns
        for col in df.columns:
            env[col] = df[col]
    else:
        env['_n'] = 1
        env['_N'] = 0

    env['_dta_missing_'] = np.nan

    # Inject scalar / return / register context if provided
    if scalar_context:
        for k, v in scalar_context.items():
            env[k] = v
            # Also register preprocessed variable names
            if k.startswith("r(") and k.endswith(")"):
                env[f"_ret_r_{k[2:-1]}_"] = v
            elif k.startswith("e(") and k.endswith(")"):
                env[f"_ret_e_{k[2:-1]}_"] = v
            elif k.startswith("_b[") and k.endswith("]"):
                env[f"_est_b_{k[3:-1]}_"] = v
            elif k.startswith("_se[") and k.endswith("]"):
                env[f"_est_se_{k[4:-1]}_"] = v

    # Pre-evaluate and inject time series lag/lead/diff variables if referenced
    if df is not None and time_context is not None:
        ts_pattern = re.findall(r'_ts_([LDF])(\d+)_([a-zA-Z0-9_]+)_', clean_expr)
        for op, order_str, var in ts_pattern:
            order = int(order_str)
            key = f"_ts_{op}{order}_{var}_"
            if key not in env:
                env[key] = compute_time_series_series(df, var, op, order, time_context)

    # Safely evaluate AST using custom evaluator
    return _eval_ast_node(parsed_ast.body, env)


def _eval_ast_node(node, env):
    """Safely evaluates an AST node within a sanitized environment."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int):
            if abs(node.value) > MAX_INT_VALUE or node.value.bit_length() > MAX_INT_BITS:
                raise ValueError(f"integer literal exceeds maximum allowed size ({MAX_INT_BITS} bits)")
        return node.value

    elif isinstance(node, ast.Name):
        name = node.id
        if name in env:
            return env[name]
        raise ValueError(f"variable '{name}' not found")

    elif isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand, env)
        if isinstance(node.op, ast.Not):
            return ~operand if isinstance(operand, pd.Series) else not operand
        elif isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return +operand

    elif isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, env)
        right = _eval_ast_node(node.right, env)
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
                    if abs(left) > 1e150 and abs(right) > 1e150:
                        raise ValueError("arithmetic overflow in multiplication")
                return left * right
            elif isinstance(node.op, ast.Div):
                # Division by zero returns NaN in standard fashion
                with np.errstate(divide='ignore', invalid='ignore'):
                    res = left / right
                    if isinstance(res, (float, np.floating)) and np.isinf(res):
                        return np.nan
                    elif isinstance(res, pd.Series):
                        res = res.replace([np.inf, -np.inf], np.nan)
                    return res
            elif isinstance(node.op, ast.FloorDiv):
                return left // right
            elif isinstance(node.op, ast.Mod):
                return left % right
            elif isinstance(node.op, ast.Pow):
                if isinstance(right, (int, float, np.number)) and abs(right) > MAX_EXPONENT:
                    raise ValueError(f"exponent magnitude ({right}) exceeds maximum allowed limit ({MAX_EXPONENT})")
                if isinstance(left, (int, float, np.number)) and abs(left) > 1e100 and right > 1:
                    raise ValueError("computation exceeds numeric range limits")
                try:
                    with np.errstate(over='raise', invalid='ignore'):
                        return left ** right
                except (OverflowError, FloatingPointError):
                    raise ValueError("arithmetic overflow in exponentiation")
        except OverflowError:
            raise ValueError("arithmetic overflow in expression")

    elif isinstance(node, ast.Compare):
        left = _eval_ast_node(node.left, env)
        result = None
        current_left = left
        for op, comp_node in zip(node.ops, node.comparators):
            right = _eval_ast_node(comp_node, env)
            if isinstance(op, ast.Eq):
                comp_res = _dta_compare(current_left, right, 'eq')
            elif isinstance(op, ast.NotEq):
                comp_res = _dta_compare(current_left, right, 'ne')
            elif isinstance(op, ast.Lt):
                comp_res = _dta_compare(current_left, right, 'lt')
            elif isinstance(op, ast.LtE):
                comp_res = _dta_compare(current_left, right, 'le')
            elif isinstance(op, ast.Gt):
                comp_res = _dta_compare(current_left, right, 'gt')
            elif isinstance(op, ast.GtE):
                comp_res = _dta_compare(current_left, right, 'ge')
            else:
                raise ValueError(f"disallowed comparison operator: {type(op).__name__}")

            result = comp_res if result is None else (result & comp_res)
            current_left = right
        return result

    elif isinstance(node, ast.BoolOp):
        values = [_eval_ast_node(v, env) for v in node.values]
        if isinstance(node.op, ast.And):
            res = values[0]
            for v in values[1:]:
                res = res & v
            return res
        elif isinstance(node.op, ast.Or):
            res = values[0]
            for v in values[1:]:
                res = res | v
            return res

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("disallowed function call: attribute access and method calls are prohibited")
        func_name = node.func.id.lower()
        if func_name not in WHITELISTED_FUNCTIONS:
            raise ValueError(f"unknown or disallowed function: '{func_name}'")
        if node.keywords:
            raise ValueError("keyword arguments are not permitted in expressions")
        arity = WHITELISTED_FUNCTION_ARITY.get(func_name)
        if arity:
            min_a, max_a = arity
            num_a = len(node.args)
            if num_a < min_a or num_a > max_a:
                if min_a == max_a:
                    raise ValueError(f"function '{func_name}' expects exactly {min_a} argument(s), got {num_a}")
                else:
                    raise ValueError(f"function '{func_name}' expects between {min_a} and {max_a} arguments, got {num_a}")
        func = WHITELISTED_FUNCTIONS[func_name]
        args = [_eval_ast_node(arg, env) for arg in node.args]

        if func_name in ('log', 'ln'):
            with np.errstate(invalid='ignore', divide='ignore'):
                val = args[0]
                res = np.log(np.where(val > 0, val, np.nan))
                return pd.Series(res, index=val.index) if isinstance(val, pd.Series) else res
        elif func_name == 'sqrt':
            with np.errstate(invalid='ignore'):
                val = args[0]
                res = np.sqrt(np.where(val >= 0, val, np.nan))
                return pd.Series(res, index=val.index) if isinstance(val, pd.Series) else res
        elif func_name == 'round':
            return func(*args)
        elif func_name == 'inlist':
            return func(args[0], *args[1:])
        elif func_name == 'inrange':
            return func(args[0], args[1], args[2])
        elif func_name in ('max', 'min'):
            return func(*args)
        else:
            return func(*args)

    elif isinstance(node, (ast.Attribute, ast.Subscript)):
        raise ValueError(f"disallowed syntax: {type(node).__name__} is strictly prohibited in expressions")

    raise ValueError(f"disallowed syntax: {type(node).__name__}")


def evaluate_if_in_qualifiers(df, if_expr=None, in_range=None, time_context=None, scalar_context=None):
    """
    Evaluates 'if exp' and 'in range' qualifiers against a DataFrame.
    Returns a boolean mask of length len(df) and the filtered subset df_sub.
    """
    n = len(df)
    mask = pd.Series(True, index=df.index)

    # 1. Apply 'in range'
    if in_range is not None:
        start_1based, end_1based = in_range
        # Resolve 'L' or negative indices
        start_idx = start_1based - 1 if start_1based > 0 else n + start_1based
        if end_1based == 'L' or end_1based is None:
            end_idx = n
        elif end_1based > 0:
            end_idx = min(end_1based, n)
        else:
            end_idx = n + end_1based + 1

        in_mask = pd.Series(False, index=df.index)
        if start_idx < end_idx and start_idx < n:
            in_mask.iloc[max(0, start_idx):end_idx] = True
        mask = mask & in_mask

    # 2. Apply 'if exp'
    if if_expr is not None and if_expr.strip():
        cond_series = evaluate_dta_expression(if_expr, df, time_context=time_context, scalar_context=scalar_context)
        if not isinstance(cond_series, pd.Series):
            cond_series = pd.Series(bool(cond_series), index=df.index)
        # Treat NaN or non-boolean values as False
        cond_bool = cond_series.fillna(False).astype(bool)
        mask = mask & cond_bool

    return mask, df[mask]


# =====================================================================
# Time Series Lag/Lead/Diff Operator Engine
# =====================================================================

def compute_time_series_series(df, varname, op, order=1, time_context=None):
    """
    Computes time-series operators (L., L2., D., F.) taking into account
    panel grouping and time gaps.
    """
    if varname not in df.columns:
        raise ValueError(f"variable '{varname}' not found")

    if time_context is None or not time_context.get('timevar'):
        raise ValueError("time variable not set; use tsset or xtset")

    timevar = time_context.get('timevar')
    panelvar = time_context.get('panelvar', None)

    if timevar not in df.columns:
        raise ValueError(f"time variable '{timevar}' not found in dataset")
    if panelvar and panelvar not in df.columns:
        raise ValueError(f"panel variable '{panelvar}' not found in dataset")

    op = op.upper()
    col = df[varname].to_numpy(dtype=float)
    times = df[timevar].to_numpy()

    result = np.full(len(df), np.nan, dtype=float)

    if panelvar:
        panels = df[panelvar].to_numpy()
        # Build dictionary lookup: (panel_id, time) -> val
        lookup = {}
        for p, t, v in zip(panels, times, col):
            lookup[(p, t)] = v

        for idx, (p, t, current_val) in enumerate(zip(panels, times, col)):
            if op == 'L':
                target_t = t - order
                result[idx] = lookup.get((p, target_t), np.nan)
            elif op == 'F':
                target_t = t + order
                result[idx] = lookup.get((p, target_t), np.nan)
            elif op == 'D':
                # Order-k difference: (1 - L)^k x_t = sum_{j=0}^k (-1)^j * comb(k, j) * x_{t-j}
                val = 0.0
                valid = True
                for j in range(order + 1):
                    target_t = t - j
                    v = lookup.get((p, target_t), np.nan)
                    if np.isnan(v):
                        valid = False
                        break
                    val += ((-1) ** j) * math.comb(order, j) * v
                result[idx] = val if valid else np.nan
    else:
        # Single time series
        lookup = dict(zip(times, col))
        for idx, (t, current_val) in enumerate(zip(times, col)):
            if op == 'L':
                target_t = t - order
                result[idx] = lookup.get(target_t, np.nan)
            elif op == 'F':
                target_t = t + order
                result[idx] = lookup.get(target_t, np.nan)
            elif op == 'D':
                val = 0.0
                valid = True
                for j in range(order + 1):
                    target_t = t - j
                    v = lookup.get(target_t, np.nan)
                    if np.isnan(v):
                        valid = False
                        break
                    val += ((-1) ** j) * math.comb(order, j) * v
                result[idx] = val if valid else np.nan

    return pd.Series(result, index=df.index)


# =====================================================================
# Factor-Variable Notation Expander
# =====================================================================

def expand_estimation_terms(varlist, df):
    """Exclude observations with missing factors, including those with no dummy column."""
    terms, frame, factors = expand_factor_variables(varlist, df)
    if factors:
        frame = frame.dropna(subset=list(factors))
    return terms, frame, factors


def expand_factor_variables(varlist, df):
    """
    Expands factor-variable notation into generated dummy and interaction columns:
    - i.var: dummies with base level omitted (ib#.var, ib(last).var, ib(first).var)
    - c.var: continuous variable as-is
    - i.a#i.b: interaction of factor variables
    - c.x#i.d: interaction between continuous and factor
    - c.x##i.d: full factorial (main effects + interaction)
    - i.a##i.b: full factorial of two factor variables

    Note on syntax convention:
    In Potatostics, a bare variable in interaction terms (e.g. `x#i.d`) is treated
    as continuous (equivalent to `c.x#i.d`), whereas classic packages treat bare variables as factor variables.
    Three-way and higher interactions are not supported and will raise a ValueError.

    Returns (expanded_varlist, working_df, factor_info)
    """
    working_df = df.copy()
    expanded_varlist = []
    factor_info = {}

    for raw_term in varlist:
        term = raw_term.strip()
        if not term:
            continue

        # Check for 3-way or higher interactions
        clean_hashes = term.replace('##', '#')
        if clean_hashes.count('#') >= 2:
            raise ValueError(f"Three-way and higher interactions are not supported: '{term}'")

        # 1. Full factorial: term1##term2
        if '##' in term:
            parts = term.split('##', 1)
            t1, t2 = parts[0].strip(), parts[1].strip()
            # Expands to t1, t2, t1#t2
            sub_terms = [t1, t2, f"{t1}#{t2}"]
            sub_expanded, working_df, info = expand_factor_variables(sub_terms, working_df)
            factor_info.update(info)
            for st in sub_expanded:
                if st not in expanded_varlist:
                    expanded_varlist.append(st)
            continue

        # 2. Single interaction: term1#term2
        if '#' in term:
            parts = term.split('#', 1)
            t1, t2 = parts[0].strip(), parts[1].strip()
            
            # Expand t1 and t2 separately
            e1_list, working_df, info1 = _expand_single_factor_term(t1, working_df)
            e2_list, working_df, info2 = _expand_single_factor_term(t2, working_df)
            factor_info.update(info1)
            factor_info.update(info2)

            for c1 in e1_list:
                for c2 in e2_list:
                    inter_name = f"{c1}#{c2}"
                    if inter_name not in working_df.columns:
                        working_df[inter_name] = working_df[c1] * working_df[c2]
                    if inter_name not in expanded_varlist:
                        expanded_varlist.append(inter_name)
            continue

        # 3. Single term (i.var or c.var or standard var)
        terms, working_df, info = _expand_single_factor_term(term, working_df)
        factor_info.update(info)
        for t in terms:
            if t not in expanded_varlist:
                expanded_varlist.append(t)

    return expanded_varlist, working_df, factor_info


def _expand_single_factor_term(term, df):
    """Expands a single term: i.var, ib#.var, c.var, or plain var."""
    term = term.strip()
    
    # Continuous: c.var or plain var
    if term.startswith("c."):
        actual_var = term[2:]
        if actual_var not in df.columns:
            raise ValueError(f"variable '{actual_var}' not found")
        if term not in df.columns:
            df[term] = df[actual_var]
        return [term], df, {}

    # Factor variable: i.var or ib<base>.var
    match_factor = re.match(r'^(i|ib[a-zA-Z0-9_()#]+)\.([a-zA-Z0-9_]+)$', term, re.IGNORECASE)
    if match_factor:
        prefix, varname = match_factor.group(1).lower(), match_factor.group(2)
        if varname not in df.columns:
            raise ValueError(f"variable '{varname}' not found")

        # Determine levels
        levels = sorted(df[varname].dropna().unique())
        if len(levels) == 0:
            return [], df, {varname: {'base': None, 'levels': levels}}

        # Determine base level
        base = levels[0]  # default is first (lowest)
        if prefix.startswith("ib"):
            spec = prefix[2:].strip("().#")
            if spec == "last":
                base = levels[-1]
            elif spec == "first":
                base = levels[0]
            elif spec == "freq":
                base = df[varname].value_counts().index[0]
            else:
                try:
                    target_num = float(spec)
                    for lvl in levels:
                        if abs(lvl - target_num) < 1e-9:
                            base = lvl
                            break
                except ValueError:
                    base = levels[0]

        # Generate dummy columns: e.g. 1.foreign, 2.foreign
        # Missing values in factor variable must produce NaN, NOT 0.0 (which would code as base level)
        generated_cols = []
        for lvl in levels:
            if lvl == base:
                continue  # base level omitted
            
            # Formatting level integer, float, or string
            try:
                lvl_num = float(lvl)
                lvl_str = f"{int(lvl_num)}" if lvl_num.is_integer() else f"{lvl}"
            except (ValueError, TypeError):
                lvl_str = f"{lvl}"
            col_name = f"{lvl_str}.{varname}"
            
            if col_name not in df.columns:
                df[col_name] = np.where(df[varname].isna(), np.nan, (df[varname] == lvl).astype(float))
            generated_cols.append(col_name)

        return generated_cols, df, {varname: {'base': base, 'levels': levels}}

    # Plain variable
    if term not in df.columns:
        raise ValueError(f"variable '{term}' not found")
    return [term], df, {}
