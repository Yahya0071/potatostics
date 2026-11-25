"""
engine/macros.py - Macro definitions, expansion, and return registers (r(), e(), _b[], _se[]).
Implements local macros (`name'), global macros ($name / ${name}), nested expansion,
return registers (r(), e()), and return/ereturn listings.
"""

import re
import numpy as np
import pandas as pd


class ReturnRegisters:
    """Stores r() and e() returned results, plus _b, _se, and _rc."""
    def __init__(self):
        self.r_scalars = {}
        self.r_macros = {}
        self.r_matrices = {}
        
        self.e_scalars = {}
        self.e_macros = {}
        self.e_matrices = {}
        
        self.b = {}      # _b[varname]
        self.se = {}     # _se[varname]
        self.rc = 0      # _rc

    def clear_r(self):
        self.r_scalars.clear()
        self.r_macros.clear()
        self.r_matrices.clear()

    def clear_e(self):
        self.e_scalars.clear()
        self.e_macros.clear()
        self.e_matrices.clear()
        self.b.clear()
        self.se.clear()

    def set_r_scalar(self, name, val):
        try:
            self.r_scalars[name] = float(val) if not pd.isna(val) else np.nan
        except (ValueError, TypeError):
            self.r_macros[name] = str(val)

    def set_scalar(self, name, val):
        m_r = re.match(r'^r\((.*?)\)$', name)
        if m_r:
            self.set_r_scalar(m_r.group(1), val)
            return
        m_e = re.match(r'^e\((.*?)\)$', name)
        if m_e:
            self.set_e_scalar(m_e.group(1), val)
            return
        self.set_r_scalar(name, val)

    def set_r_macro(self, name, val):
        self.r_macros[name] = str(val)

    def set_e_scalar(self, name, val):
        try:
            self.e_scalars[name] = float(val) if not pd.isna(val) else np.nan
        except (ValueError, TypeError):
            self.e_macros[name] = str(val)

    def set_e_macro(self, name, val):
        self.e_macros[name] = str(val)

    def set_estimates(self, b_dict, se_dict, vcov=None):
        self.b = {str(k): float(v) for k, v in b_dict.items()}
        self.se = {str(k): float(v) for k, v in se_dict.items()}
        if vcov is not None:
            self.e_matrices['V'] = vcov
        self.e_matrices['b'] = self.b

    def get_r(self, name):
        if name in self.r_scalars:
            return self.r_scalars[name]
        if name in self.r_macros:
            return self.r_macros[name]
        return self.r_matrices.get(name, None)

    def get_e(self, name):
        if name in self.e_scalars:
            return self.e_scalars[name]
        if name in self.e_macros:
            return self.e_macros[name]
        return self.e_matrices.get(name, None)

    def get_symbol_value(self, symbol):
        """Resolves r(name), e(name), _b[var], _se[var], or _rc."""
        symbol = symbol.strip()
        if symbol == "_rc":
            return self.rc

        # _b[var] or _b.var
        m_b = re.match(r'^_b\[(.*?)\]$', symbol)
        if m_b:
            var = m_b.group(1).strip()
            return self.b.get(var, np.nan)

        # _se[var] or _se.var
        m_se = re.match(r'^_se\[(.*?)\]$', symbol)
        if m_se:
            var = m_se.group(1).strip()
            return self.se.get(var, np.nan)

        # r(name)
        m_r = re.match(r'^r\((.*?)\)$', symbol)
        if m_r:
            name = m_r.group(1).strip()
            if name in self.r_scalars:
                return self.r_scalars[name]
            if name in self.r_macros:
                return self.r_macros[name]
            return np.nan

        # e(name)
        m_e = re.match(r'^e\((.*?)\)$', symbol)
        if m_e:
            name = m_e.group(1).strip()
            if name in self.e_scalars:
                return self.e_scalars[name]
            if name in self.e_macros:
                return self.e_macros[name]
            return np.nan

        return None

    def get_context_dict(self):
        """Returns a dict of all scalars and macros suitable for expression evaluation."""
        ctx = {"_rc": self.rc}
        for k, v in self.r_scalars.items():
            ctx[f"r({k})"] = v
        for k, v in self.e_scalars.items():
            ctx[f"e({k})"] = v
        for k, v in self.b.items():
            ctx[f"_b[{k}]"] = v
        for k, v in self.se.items():
            ctx[f"_se[{k}]"] = v
        return ctx

    def format_return_list(self):
        """Formats the output of 'return list'."""
        lines = []
        if self.r_scalars:
            lines.append("scalars:")
            for k in sorted(self.r_scalars.keys()):
                v = self.r_scalars[k]
                lines.append(f"                 r({k}) =  {v}")
        if self.r_macros:
            lines.append("macros:")
            for k in sorted(self.r_macros.keys()):
                lines.append(f"                 r({k}) : \"{self.r_macros[k]}\"")
        if self.r_matrices:
            lines.append("matrices:")
            for k in sorted(self.r_matrices.keys()):
                lines.append(f"                 r({k}) : matrix")
        if not lines:
            return "no saved results found"
        return "\n".join(lines)

    def format_ereturn_list(self):
        """Formats the output of 'ereturn list'."""
        lines = []
        if self.e_scalars:
            lines.append("scalars:")
            for k in sorted(self.e_scalars.keys()):
                v = self.e_scalars[k]
                lines.append(f"                 e({k}) =  {v}")
        if self.e_macros:
            lines.append("macros:")
            for k in sorted(self.e_macros.keys()):
                lines.append(f"                 e({k}) : \"{self.e_macros[k]}\"")
        if self.e_matrices:
            lines.append("matrices:")
            for k in sorted(self.e_matrices.keys()):
                lines.append(f"                 e({k}) : matrix")
        if not lines:
            return "no estimation results found"
        return "\n".join(lines)


class MacroManager:
    """
    Manages local and global macros with support for nested/recursive expansion.
    """
    def __init__(self, registers=None):
        self.globals = {}
        self.local_stack = [{}]  # stack of dicts for local scopes
        self.registers = registers if registers is not None else ReturnRegisters()

    @property
    def locals(self):
        return self.local_stack[-1]

    def push_scope(self):
        """Pushes a new local macro scope (e.g. entering a do-file or function)."""
        self.local_stack.append({})

    def pop_scope(self):
        """Pops the topmost local macro scope."""
        if len(self.local_stack) > 1:
            self.local_stack.pop()
        else:
            self.local_stack[0].clear()

    def set_local(self, name, value):
        self.locals[name.strip()] = str(value)

    def get_local(self, name):
        name = name.strip()
        # Search backwards from the most local scope to parent scopes
        for scope in reversed(self.local_stack):
            if name in scope:
                return scope[name]
        return ""

    def set_global(self, name, value):
        self.globals[name.strip()] = str(value)

    def get_global(self, name):
        name = name.strip()
        return self.globals.get(name, "")

    def set_scalar(self, name, value):
        self.registers.set_scalar(name, value)

    @property
    def return_scalars(self):
        return {f"r({k})": v for k, v in self.registers.r_scalars.items()}

    def expand(self, text):
        """
        Expands local macros `name' and global macros $name / ${name}.
        Performs recursive expansion from inside-out for nested macros.
        """
        if not text:
            return ""

        max_iterations = 20
        curr_text = text

        for _ in range(max_iterations):
            prev_text = curr_text

            # 1. Expand innermost local macros: `[^`']+?'
            # e.g. in `var_`i'' it will first match `i' -> 1, becoming `var_1'
            def replace_local(match):
                varname = match.group(1)
                return self.get_local(varname)

            curr_text = re.sub(r'`([^`\']+?)\'', replace_local, curr_text)

            # 2. Expand global macros with braces: ${name}
            def replace_global_braced(match):
                varname = match.group(1)
                return self.get_global(varname)

            curr_text = re.sub(r'\$\{([a-zA-Z0-9_]+)\}', replace_global_braced, curr_text)

            # 3. Expand global macros without braces: $name
            def replace_global_unbraced(match):
                varname = match.group(1)
                return self.get_global(varname)

            curr_text = re.sub(r'\$([a-zA-Z_][a-zA-Z0-9_]*)', replace_global_unbraced, curr_text)

            if curr_text == prev_text:
                break

        return curr_text
