"""
engine/copilot.py - Command assistant (rule-based)
Translates plain English queries into statistical commands, executes them,
and generates plain-language econometric interpretations.
"""

import re

def _match_any(patterns, text):
    for p in patterns:
        if re.search(r'\b' + re.escape(p) + r'\b', text, re.IGNORECASE):
            return True
    return False

class CopilotAssistant:
    def __init__(self, interpreter):
        self.interpreter = interpreter

    def ask(self, query):
        query_clean = query.strip().strip('\'"')
        q_lower = query_clean.lower()
        cols = list(self.interpreter.dataset.df.columns) if not self.interpreter.dataset.is_empty else []
        
        # 1. Identify intent using word-boundary matching
        # Regression
        if _match_any(["regress", "regression", "ols", "fit model"], q_lower):
            return self._handle_regression(q_lower, cols)
        # Logit
        elif _match_any(["logit", "logistic", "binary choice", "odds"], q_lower):
            return self._handle_logit(q_lower, cols)
        # Correlation (map 'relationship between' and 'relationship' to correlate)
        elif _match_any(["relationship between", "correlation", "correlate", "relationship"], q_lower):
            return self._handle_correlate(q_lower, cols)
        # Summarize (word-boundary ensures 'mean' does not match 'meaningful')
        elif _match_any(["summarize", "summary", "stats", "mean", "average", "describe"], q_lower):
            return self._handle_summarize(q_lower, cols)
        # Scatter Plot
        elif _match_any(["scatter", "plot", "graph"], q_lower):
            return self._handle_plot(q_lower, cols)
        # Tabulate
        elif _match_any(["tabulate", "frequency", "frequencies", "crosstab"], q_lower):
            return self._handle_tabulate(q_lower, cols)
        else:
            return (
                f"[Command assistant (rule-based)] I understand natural language requests such as:\n"
                f"  - 'regress price on mpg and weight'\n"
                f"  - 'logit foreign on price and mpg'\n"
                f"  - 'summarize price and mpg'\n"
                f"  - 'show correlation between price and weight'\n"
                f"  - 'plot price vs mpg'\n"
                f"Please rephrase your query with your dataset variables ({', '.join(cols[:5])}...)."
            )

    def _find_variables(self, text, cols):
        found = []
        for c in cols:
            if re.search(r'\b' + re.escape(c) + r'\b', text, re.IGNORECASE):
                found.append(c)
        return found

    def _handle_regression(self, text, cols):
        vars_found = self._find_variables(text, cols)
        if len(vars_found) < 2:
            return "[Command assistant (rule-based)] Please specify a dependent variable and at least one independent variable."
            
        depvar = vars_found[0]
        indeps = vars_found[1:]
        robust = bool(re.search(r'\b(robust|hc1)\b', text, re.IGNORECASE))
        
        cmd = f"regress {depvar} {' '.join(indeps)}" + (", robust" if robust else "")
        res_text = self.interpreter.execute(cmd)
        if "regression error:" in res_text or "insufficient" in res_text.lower():
            return res_text
            
        m = self.interpreter.estimates.last_model
        if not m or m.get('depvar') != depvar:
            return res_text
            
        # Natural language interpretation with associational wording
        interpretation = [
            "[Command assistant (rule-based) Copilot Automated Interpretation]",
            f"- Model: Regressed {depvar} on {', '.join(indeps)}.",
            f"- Goodness of Fit: The model explains {m['r2']*100:.1f}% of the variation in {depvar} (R2 = {m['r2']:.4f}, F = {m['f_stat']:.2f}, p = {m['prob_f']:.4f})."
        ]
        for row in m['coefs']:
            if row['name'] != '_cons':
                sig = "statistically significant" if row['p'] < 0.05 else "not statistically significant"
                assoc = "is associated with an average increase in" if row['coef'] > 0 else "is associated with an average decrease in"
                interpretation.append(
                    f"- {row['name']}: A 1-unit increase {assoc} {depvar} of {abs(row['coef']):.2f} ({sig}, p = {row['p']:.4f})."
                )
                
        return res_text + "\n\n" + "\n".join(interpretation)

    def _handle_logit(self, text, cols):
        vars_found = self._find_variables(text, cols)
        if len(vars_found) < 2:
            return "[Command assistant (rule-based)] Please specify a binary dependent variable and regressors."
        cmd = f"logit {vars_found[0]} {' '.join(vars_found[1:])}"
        res_text = self.interpreter.execute(cmd)
        if "logit error:" in res_text or "insufficient" in res_text.lower():
            return res_text
            
        m = self.interpreter.estimates.last_model
        if not m or m.get('depvar') != vars_found[0]:
            return res_text
            
        interpretation = [
            "[Command assistant (rule-based) Automated Interpretation]",
            f"- Fitted logistic regression on binary outcome '{vars_found[0]}'.",
            f"- Pseudo R2 = {m['pseudo_r2']:.4f}, LR chi2 = {m['lr_chi2']:.2f} (p = {m['prob_chi2']:.4f})."
        ]
        return res_text + "\n\n" + "\n".join(interpretation)

    def _handle_summarize(self, text, cols):
        vars_found = self._find_variables(text, cols)
        cmd = f"summarize {' '.join(vars_found)}" if vars_found else "summarize"
        res = self.interpreter.execute(cmd)
        out = res.output if hasattr(res, 'output') else str(res)
        return f"[Command assistant (rule-based)] Executed: {cmd}\n\n{out}"

    def _handle_correlate(self, text, cols):
        vars_found = self._find_variables(text, cols)
        cmd = f"correlate {' '.join(vars_found)}" if vars_found else "correlate"
        res = self.interpreter.execute(cmd)
        out = res.output if hasattr(res, 'output') else str(res)
        return f"[Command assistant (rule-based)] Executed: {cmd}\n\n{out}"

    def _handle_plot(self, text, cols):
        vars_found = self._find_variables(text, cols)
        if len(vars_found) >= 2:
            res = self.interpreter.execute(f"scatter {vars_found[0]} {vars_found[1]}")
            out = res.output if hasattr(res, 'output') else str(res)
            return f"[Command assistant (rule-based)] Executed: scatter {vars_found[0]} {vars_found[1]}\n\n{out}"
        elif len(vars_found) == 1:
            res = self.interpreter.execute(f"histogram {vars_found[0]}")
            out = res.output if hasattr(res, 'output') else str(res)
            return f"[Command assistant (rule-based)] Executed: histogram {vars_found[0]}\n\n{out}"
        return "[Command assistant (rule-based)] Please name the variables to plot."

    def _handle_tabulate(self, text, cols):
        vars_found = self._find_variables(text, cols)
        if vars_found:
            cmd = f"tabulate {' '.join(vars_found[:2])}"
            res = self.interpreter.execute(cmd)
            out = res.output if hasattr(res, 'output') else str(res)
            return f"[Command assistant (rule-based)] Executed: {cmd}\n\n{out}"
        return "[Command assistant (rule-based)] Please name the variable to tabulate."
