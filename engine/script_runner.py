"""
engine/script_runner.py - Do-file Runner, Control Flow Blocks, and Script Prefixes.
Supports:
1. Prefixes: quietly (qui), noisily (noi), capture (cap).
2. Loops: foreach ... { }, forvalues ... { }.
3. Conditionals: if ... { } else { }.
4. Macro expansions: `local', $global, ${global}.
5. Comments and continuations: *, //, /* */, ///.
6. Execution of .do files with error handling and _rc trapping.
"""

import re
import shlex
import pandas as pd
from engine.parser import CommandResult, evaluate_dta_expression


def strip_comments_and_join_continuations(raw_script):
    """
    Strips /* ... */ block comments, // line comments, and lines starting with *.
    Joins lines ending with ///.
    Returns a list of logical lines with their original line numbers.
    """
    # 1. Remove /* ... */ comments (can span multiple lines)
    # Using regex with DOTALL flag
    clean_text = re.sub(r'/\*.*?\*/', ' ', raw_script, flags=re.DOTALL)

    raw_lines = clean_text.splitlines()
    logical_lines = []
    
    current_acc = []
    start_lineno = 1

    for lineno, line in enumerate(raw_lines, 1):
        line_str = line.strip()

        # If current accumulator is empty, check if line is purely comment
        if not current_acc:
            start_lineno = lineno
            if not line_str or line_str.startswith("*") or line_str.startswith("//"):
                continue

        # Check for /// continuation
        # Pattern: anything before '///' is kept, anything after is discarded
        match_cont = re.search(r'^(.*?)\s*///\s*(.*)$', line)
        if match_cont:
            before_cont = match_cont.group(1).rstrip()
            current_acc.append(before_cont)
            continue  # Continue to next line to join

        # Line without continuation: check if it ends with // comment
        # Strip trailing // comments if outside quotes
        in_dquote = False
        in_squote = False
        comment_idx = None
        for i in range(len(line)):
            ch = line[i]
            if ch == '"' and not in_squote:
                in_dquote = not in_dquote
            elif ch == "'" and not in_dquote:
                in_squote = not in_squote
            elif not in_dquote and not in_squote:
                if ch == '/' and i + 1 < len(line) and line[i+1] == '/':
                    comment_idx = i
                    break
        
        if comment_idx is not None:
            line_str = line[:comment_idx].rstrip()
        else:
            line_str = line.rstrip()

        if current_acc:
            current_acc.append(line_str)
            full_line = " ".join(part.strip() for part in current_acc if part.strip())
            current_acc = []
            if full_line:
                logical_lines.append((start_lineno, full_line))
        else:
            if line_str.strip():
                logical_lines.append((lineno, line_str.strip()))

    if current_acc:
        full_line = " ".join(current_acc).strip()
        if full_line:
            logical_lines.append((start_lineno, full_line))

    return logical_lines


def parse_forvalues_range(spec):
    """
    Parses forvalues range specs:
    - 1/5 -> [1, 2, 3, 4, 5]
    - 1(1)5 -> [1, 2, 3, 4, 5]
    - 10(-2)2 -> [10, 8, 6, 4, 2]
    - 1 2 to 5 -> [1, 2, 3, 4, 5]
    """
    spec = spec.strip()
    # Pattern: start(step)stop
    m_step = re.match(r'^([+-]?\d+(?:\.\d+)?)\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*\)\s*([+-]?\d+(?:\.\d+)?)$', spec)
    if m_step:
        start = float(m_step.group(1))
        step = float(m_step.group(2))
        stop = float(m_step.group(3))
        vals = []
        curr = start
        if step > 0:
            while curr <= stop + 1e-9:
                vals.append(int(curr) if curr.is_integer() else curr)
                curr += step
        elif step < 0:
            while curr >= stop - 1e-9:
                vals.append(int(curr) if curr.is_integer() else curr)
                curr += step
        return vals

    # Pattern: start/stop (default step 1)
    m_slash = re.match(r'^([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)$', spec)
    if m_slash:
        start = float(m_slash.group(1))
        stop = float(m_slash.group(2))
        step = 1.0 if stop >= start else -1.0
        vals = []
        curr = start
        if step > 0:
            while curr <= stop + 1e-9:
                vals.append(int(curr) if curr.is_integer() else curr)
                curr += step
        else:
            while curr >= stop - 1e-9:
                vals.append(int(curr) if curr.is_integer() else curr)
                curr += step
        return vals

    raise ValueError(f"invalid forvalues range specification: '{spec}'")


class ScriptRunner:
    """
    Executes scripts, do-files, and multi-line blocks with macro expansion,
    loops, and control flow.
    """
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.macro_mgr = interpreter.macro_mgr
        self.registers = interpreter.registers

    def run_script(self, script_text, quiet=False, nostop=False, is_do_file=False):
        """
        Executes a script text line-by-line, handling blocks and macros.
        Returns CommandResult with accumulated output.
        """
        if is_do_file:
            self.macro_mgr.push_scope()
        if hasattr(self.interpreter, '_script_depth'):
            self.interpreter._script_depth += 1
        else:
            self.interpreter._script_depth = 1

        try:
            logical_lines = strip_comments_and_join_continuations(script_text)
            sections = []

            def must_stop(result, ignore_errors):
                if result.ok:
                    return False
                # A user break also stops a capture block or a nostop do-file.
                return not ignore_errors or (result.return_code == 1 and self.interpreter.check_cancelled())

            idx = 0
            n_lines = len(logical_lines)

            while idx < n_lines:
                if hasattr(self.interpreter, 'check_cancelled') and self.interpreter.check_cancelled():
                    cancelled = CommandResult("(Execution cancelled by user)", ok=False, return_code=1)
                    sections.extend(cancelled.sections())
                    return CommandResult.from_sections(sections, cancelled)
                lineno, line = logical_lines[idx]

                # Handle block statements (foreach, forvalues, if, quietly {, etc.)
                # 1. foreach
                m_foreach = re.match(r'^foreach\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(in|of\s+varlist|of\s+newlist)\s+(.*?)\s*\{$', line, re.IGNORECASE)
                if m_foreach:
                    varname = m_foreach.group(1)
                    kind = m_foreach.group(2).lower()
                    rest = m_foreach.group(3)
                    
                    block_lines, next_idx = self._extract_block(logical_lines, idx)
                    idx = next_idx

                    # Expand macro in items
                    rest = self.macro_mgr.expand(rest)
                    
                    # Resolve items
                    if "varlist" in kind:
                        # Varlist expansion from dataset
                        items = self.interpreter.expand_varlist_pattern(rest)
                    else:
                        try:
                            items = shlex.split(rest)
                        except Exception:
                            items = rest.split()

                    for item in items:
                        self.macro_mgr.set_local(varname, item)
                        sub_res = self._execute_sub_block(block_lines, quiet=quiet, nostop=nostop)
                        if sub_res.output:
                            sections.extend(sub_res.sections())
                        if must_stop(sub_res, nostop):
                            return CommandResult.from_sections(sections, sub_res)
                    continue

                # 2. forvalues
                m_forval = re.match(r'^forvalues\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*?)\s*\{$', line, re.IGNORECASE)
                if m_forval:
                    varname = m_forval.group(1)
                    range_spec = self.macro_mgr.expand(m_forval.group(2))
                    
                    block_lines, next_idx = self._extract_block(logical_lines, idx)
                    idx = next_idx

                    vals = parse_forvalues_range(range_spec)
                    for val in vals:
                        self.macro_mgr.set_local(varname, str(val))
                        sub_res = self._execute_sub_block(block_lines, quiet=quiet, nostop=nostop)
                        if sub_res.output:
                            sections.extend(sub_res.sections())
                        if must_stop(sub_res, nostop):
                            return CommandResult.from_sections(sections, sub_res)
                    continue

                # 3. if block: if <exp> { ... } [else { ... }]
                m_if = re.match(r'^if\s+(.*?)\s*\{$', line, re.IGNORECASE)
                if m_if:
                    cond_expr = self.macro_mgr.expand(m_if.group(1))
                    if_block, next_idx = self._extract_block(logical_lines, idx)
                    idx = next_idx
                    
                    else_block = None
                    if idx < n_lines and logical_lines[idx][1].strip().lower().startswith("else"):
                        else_line = logical_lines[idx][1].strip()
                        if "{" in else_line:
                            else_block, next_idx = self._extract_block(logical_lines, idx)
                            idx = next_idx

                    # Evaluate if condition
                    cond_val = self._evaluate_scalar_condition(cond_expr)
                    if cond_val:
                        sub_res = self._execute_sub_block(if_block, quiet=quiet, nostop=nostop)
                        if sub_res.output:
                            sections.extend(sub_res.sections())
                        if must_stop(sub_res, nostop):
                            return CommandResult.from_sections(sections, sub_res)
                    elif else_block is not None:
                        sub_res = self._execute_sub_block(else_block, quiet=quiet, nostop=nostop)
                        if sub_res.output:
                            sections.extend(sub_res.sections())
                        if must_stop(sub_res, nostop):
                            return CommandResult.from_sections(sections, sub_res)
                    continue

                # 4. quietly { ... } or capture { ... } block
                m_qui_blk = re.match(r'^(quietly|qui|capture|cap)\s*\{$', line, re.IGNORECASE)
                if m_qui_blk:
                    pfx = m_qui_blk.group(1).lower()
                    block_lines, next_idx = self._extract_block(logical_lines, idx)
                    idx = next_idx
                    
                    is_qui = pfx.startswith("qui")
                    is_cap = pfx.startswith("cap")
                    
                    sub_res = self._execute_sub_block(block_lines, quiet=(quiet or is_qui), nostop=(nostop or is_cap))
                    if is_cap:
                        self.registers.rc = sub_res.return_code
                    if sub_res.output and not is_qui:
                        sections.extend(sub_res.sections())
                    if must_stop(sub_res, is_cap or nostop):
                        return CommandResult.from_sections(sections, sub_res)
                    continue

                # 5. Regular single command line
                idx += 1
                expanded_cmd = self.macro_mgr.expand(line)
                
                # Check for single-line display / di
                res = self.interpreter.execute_with_prefixes(expanded_cmd, force_quiet=quiet)
                if res.output:
                    sections.extend(res.sections())
                
                if must_stop(res, nostop):
                    return CommandResult.from_sections(sections, res)

            return CommandResult.from_sections(sections)

        finally:
            if hasattr(self.interpreter, '_script_depth'):
                self.interpreter._script_depth = max(0, self.interpreter._script_depth - 1)
            if is_do_file:
                self.macro_mgr.pop_scope()

    def _extract_block(self, logical_lines, start_idx):
        """Extracts lines inside { ... } balancing braces."""
        block = []
        depth = 0
        idx = start_idx
        n_lines = len(logical_lines)

        while idx < n_lines:
            lineno, line = logical_lines[idx]
            for ch in line:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
            
            if idx > start_idx:
                # Remove trailing '}' from the closing line if depth just became 0
                if depth == 0 and line.strip() == '}':
                    pass
                else:
                    block.append((lineno, line))

            idx += 1
            if depth == 0:
                break

        return block, idx

    def _execute_sub_block(self, block_lines, quiet=False, nostop=False):
        """Executes a list of (lineno, line) pairs."""
        sub_text = "\n".join(line for _, line in block_lines)
        return self.run_script(sub_text, quiet=quiet, nostop=nostop, is_do_file=False)

    def _evaluate_scalar_condition(self, expr_str):
        """Evaluates a scalar truth condition (truthy != 0 and not NaN)."""
        try:
            ctx = self.registers.get_context_dict()
            val = evaluate_dta_expression(expr_str, df=None, scalar_context=ctx)
            if pd.isna(val) or val == 0 or val is False:
                return False
            return True
        except Exception:
            # Check simple boolean comparison
            return False
