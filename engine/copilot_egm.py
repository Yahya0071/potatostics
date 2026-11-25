"""
engine/copilot_egm.py - Command assistant (rule-based) Dual-Memory Architecture
Part of Potatostics Statistical Intelligence Core.

Architecture:
1. Goldfish Memory Buffer (Short-Term):
   - Sliding context window holding immediate session trajectory:
     - Recent commands entered
     - Recent execution outputs and return codes
     - Active fitted models, R-squared, coefficients, and standard errors
     - Contextual pronoun/anaphora resolution ("add robust", "what was the R2", "fix the error")
2. Elephant Memory (Long-Term):
   - Persistent JSON-backed storage preserving:
     - Project Codebooks: variable definitions, measurement units, domain constraints
     - Workflow patterns & macro templates across sessions
     - User econometric preferences (e.g. preferred VCE, significance levels)
3. Offline Inference:
   - High-precision deterministic econometric interpretation engine
   - Automated economic interpretation & endogeneity checks
"""

import os
import re
import json


class GoldfishMemoryBuffer:
    """Short-term sliding window session memory buffer."""

    def __init__(self, max_turns=20):
        self.max_turns = max_turns
        self.history = []  # list of dicts: {'role': 'user'|'assistant'|'system', 'content': str, 'meta': dict}
        self.last_command = None
        self.last_result = None
        self.last_model = None

    def record_turn(self, role, content, meta=None):
        turn = {
            'role': role,
            'content': content,
            'meta': meta or {}
        }
        self.history.append(turn)
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def record_command(self, cmd_str, result_obj, model_dict=None):
        self.last_command = cmd_str
        self.last_result = result_obj
        if model_dict:
            self.last_model = model_dict
        self.record_turn('system', f"Executed: {cmd_str}", {
            'ok': getattr(result_obj, 'ok', True),
            'return_code': getattr(result_obj, 'return_code', 0),
            'has_model': bool(model_dict)
        })

    def get_recent_context(self, k=5):
        return self.history[-k:]

    def clear(self):
        self.history.clear()
        self.last_command = None
        self.last_result = None
        self.last_model = None


class ElephantLongTermMemory:
    """Persistent project-level vector and codebook memory."""

    def __init__(self, storage_path=None, enable_query_logging=False):
        if storage_path is None:
            storage_path = os.path.join(os.path.expanduser("~"), ".potatostics", "assistant_memory.json")
        self.storage_path = storage_path
        self.enable_query_logging = enable_query_logging
        self.data = {
            'codebooks': {},      # {project_key: {varname: description}}
            'user_preferences': {
                'default_vce': 'robust',
                'significance_alpha': 0.05,
                'default_table_format': 'booktabs',
                'enable_query_logging': False
            },
            'saved_patterns': {}, # {intent: command_template}
            'project_notes': {}
        }
        self.load()

    def load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self.data.update(loaded)
                    if 'user_preferences' in loaded:
                        self.enable_query_logging = loaded['user_preferences'].get('enable_query_logging', False)
            except Exception:
                pass

    def save(self):
        try:
            # Do not persist queries without an explicit visible opt-in
            if not self.enable_query_logging:
                if 'query_history' in self.data:
                    del self.data['query_history']
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def set_codebook_entry(self, project_id, varname, description):
        if project_id not in self.data['codebooks']:
            self.data['codebooks'][project_id] = {}
        self.data['codebooks'][project_id][varname] = description
        self.save()

    def get_codebook(self, project_id):
        return self.data['codebooks'].get(project_id, {})

    def learn_pattern(self, intent, command_template):
        self.data['saved_patterns'][intent.lower()] = command_template
        self.save()

    def find_pattern(self, query):
        q_clean = query.lower()
        for intent, tmpl in self.data['saved_patterns'].items():
            if intent in q_clean:
                return tmpl
        return None


def _match_any(patterns, text):
    for p in patterns:
        if re.search(r'\b' + re.escape(p) + r'\b', text, re.IGNORECASE):
            return True
    return False


class SessionMemoryCopilot:
    """
    Command assistant (rule-based) with dual-memory session awareness.
    Resolves contextual requests using short-term Goldfish memory,
    retains domain codebooks via Elephant long-term storage,
    and performs automated interpretations.
    """

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.goldfish = GoldfishMemoryBuffer(max_turns=25)
        self.elephant = ElephantLongTermMemory()

    def ask(self, query):
        """Processes a natural language prompt with dual-memory awareness."""
        query_clean = query.strip().strip('\'"')
        q_lower = query_clean.lower()
        self.goldfish.record_turn('user', query_clean)

        cols = list(self.interpreter.dataset.df.columns) if not self.interpreter.dataset.is_empty else []

        # 1. Memory Management & Inspection Commands
        if q_lower in ("memory", "memory status", "mem"):
            return self._format_memory_status()
        if q_lower in ("memory clear", "clear memory"):
            self.goldfish.clear()
            return "[Command assistant (rule-based)] Goldfish short-term memory cleared. Elephant project memory preserved."
        if q_lower.startswith("codebook"):
            return self._handle_codebook_query(q_lower, query_clean, cols)

        # 2. Contextual Follow-Up Queries (Goldfish short-term resolution)
        # "What was the R2?" / "What was the F stat?" / "Show me the fit"
        if any(p in q_lower for p in ["what was the r2", "what is the r2", "r-squared", "model fit"]):
            return self._handle_context_fit_query()

        # "Add robust" / "with robust standard errors" / "cluster standard errors"
        if ("robust" in q_lower or "cluster" in q_lower) and self.goldfish.last_command:
            return self._handle_context_add_option(q_lower)

        # "Explain that" / "interpret coefficients" / "what does that mean"
        if any(p in q_lower for p in ["explain", "interpret", "what does that mean", "economic significance"]):
            return self._handle_interpretation_request()

        # 3. Known Intent Matching using word-boundary checks
        # Regression
        if _match_any(["regress", "regression", "ols", "fit model"], q_lower):
            return self._handle_regression(q_lower, cols)

        # Staggered DiD / Causal
        if _match_any(["staggered", "callaway", "csdid", "sant'anna"], q_lower):
            return self._handle_csdid(q_lower, cols)

        # Synthetic Control
        if _match_any(["synthetic control", "synth", "counterfactual"], q_lower):
            return self._handle_synth(q_lower, cols)

        # Event Study
        if _match_any(["event study", "eventstudy", "dynamic effect", "leads and lags"], q_lower):
            return self._handle_event_study(q_lower, cols)

        # Multi-model table / esttab
        if _match_any(["esttab", "export table", "comparison table", "latex table", "word table"], q_lower):
            return self._handle_esttab(q_lower)

        # Correlation (map 'relationship between' to correlate)
        if _match_any(["relationship between", "correlation", "correlate", "relationship"], q_lower):
            return self._handle_correlate(q_lower, cols)

        # Summarize (word-boundary ensures 'mean' does not match 'meaningful')
        if _match_any(["summarize", "summary", "stats", "mean", "average", "describe"], q_lower):
            return self._handle_summarize(q_lower, cols)

        # Tabulate
        if _match_any(["tabulate", "frequency", "frequencies", "crosstab"], q_lower):
            return self._handle_tabulate(q_lower, cols)

        # 4. Fallback Guidance
        return (
            "[Command assistant (rule-based)] I am your rule-based econometric assistant.\n"
            "You can ask me to:\n"
            "  - Fit models: 'regress price on mpg and weight with robust standard errors'\n"
            "  - Ask contextual follow-ups: 'what was the R2?', 'add robust SE', 'explain results'\n"
            "  - Modern causal inference: 'run staggered DiD on y with unit and time'\n"
            "  - Export: 'create a LaTeX booktabs comparison table'\n"
            f"Active variables: {', '.join(cols[:8])}{'...' if len(cols) > 8 else ''}"
        )

    def _find_variables(self, text, cols):
        found = []
        for c in cols:
            if re.search(r'\b' + re.escape(c) + r'\b', text, re.IGNORECASE):
                found.append(c)
        return found

    def _handle_context_fit_query(self):
        m = self.interpreter.estimates.last_model
        if not m:
            return "[Command assistant (rule-based)] No active estimation model found in session memory."
        r2 = m.get('r2', m.get('pseudo_r2', m.get('r2_within', 'N/A')))
        r2_str = f"{r2:.4f} ({r2*100:.1f}%)" if isinstance(r2, float) else str(r2)
        f_val = m.get('f_stat', 'N/A')
        f_str = f"{f_val:.2f}" if isinstance(f_val, float) else str(f_val)
        n_obs = m.get('n', 'N/A')
        depvar = m.get('depvar', 'outcome')
        return (
            f"[Command assistant (rule-based) Model Fit Report]\n"
            f"- Dependent Variable: {depvar}\n"
            f"- Sample Size: {n_obs:,} observations\n"
            f"- R-squared: {r2_str}\n"
            f"- F-statistic: {f_str} (p = {m.get('prob_f', m.get('p_f', 0.0)):.4f})"
        )

    def _handle_context_add_option(self, q_lower):
        last_cmd = self.goldfish.last_command
        if not last_cmd or not last_cmd.startswith("regress"):
            return "[Command assistant (rule-based)] Contextual modification applies to a previously run regression command."
        
        # Check if already has robust or vce option
        if re.search(r'\b(robust|vce)\b', last_cmd, re.IGNORECASE):
            return f"[Command assistant (rule-based)] Previous command already includes a VCE/robust specification: `{last_cmd}`"
        
        # Append robust
        if "," in last_cmd:
            new_cmd = f"{last_cmd} robust"
        else:
            new_cmd = f"{last_cmd}, robust"
            
        res = self.interpreter.execute(new_cmd)
        self.goldfish.record_command(new_cmd, res, self.interpreter.estimates.last_model)
        return f"[Command assistant (rule-based) Updated Command]: `{new_cmd}`\n\n{res.output}"

    def _handle_interpretation_request(self):
        m = self.interpreter.estimates.last_model
        if not m or not m.get('coefs'):
            return "[Command assistant (rule-based)] No active regression model in memory to interpret."
        return self._generate_econometric_interpretation(m)

    def _generate_econometric_interpretation(self, m):
        depvar = m.get('depvar', 'dependent variable')
        r2 = m.get('r2', 0.0)
        lines = [
            "[Command assistant (rule-based) Copilot Automated Interpretation]",
            "1. Goodness of Fit:",
            f"   The model accounts for approximately {r2*100:.1f}% of the variance in {depvar} (R2 = {r2:.4f}, N = {m.get('n', 0):,}).",
            "2. Coefficient Effects & Significance:"
        ]
        for row in m.get('coefs', []):
            name = row['name']
            if name in ('_cons', 'Intercept'):
                continue
            b = row['coef']
            p = row.get('p', 1.0)
            sig_text = "statistically significant at the 1% level" if p < 0.01 else (
                "statistically significant at the 5% level" if p < 0.05 else (
                    "marginally significant at the 10% level" if p < 0.10 else "not statistically significant"
                )
            )
            assoc = "is associated with an average increase in" if b > 0 else "is associated with an average decrease in"
            lines.append(
                f"   • {name} (beta = {b:.4f}, p = {p:.4f}): A 1-unit change in {name} {assoc} {depvar} of {abs(b):.4f} units on average ({sig_text})."
            )
        lines.append("3. Econometric Considerations: Check for potential omitted variable bias, reverse causality, or measurement error before drawing structural causal conclusions.")
        return "\n".join(lines)

    def _handle_regression(self, text, cols):
        vars_found = self._find_variables(text, cols)
        if len(vars_found) < 2:
            return "[Copilot] Please specify a dependent variable and at least one independent variable from your dataset."
        depvar = vars_found[0]
        indeps = vars_found[1:]
        robust = "robust" in text
        cmd = f"regress {depvar} {' '.join(indeps)}" + (", robust" if robust else "")
        res = self.interpreter.execute(cmd)
        m = self.interpreter.estimates.last_model
        self.goldfish.record_command(cmd, res, m)
        if not res.ok:
            return res.output
        interp = self._generate_econometric_interpretation(m)
        return f"{res.output}\n\n{interp}"

    def _handle_csdid(self, text, cols):
        vars_found = self._find_variables(text, cols)
        if len(vars_found) < 4:
            return "[Copilot] Staggered DiD requires: outcome (depvar), unit ID (ivar), time variable, and cohort/treatment timing variable (gvar)."
        dep, ivar, timevar, gvar = vars_found[0], vars_found[1], vars_found[2], vars_found[3]
        cmd = f"csdid {dep}, ivar({ivar}) timevar({timevar}) gvar({gvar})"
        res = self.interpreter.execute(cmd)
        self.goldfish.record_command(cmd, res, self.interpreter.estimates.last_model)
        return res.output

    def _handle_synth(self, text, cols):
        vars_found = self._find_variables(text, cols)
        if not vars_found:
            return "[Copilot] Synthetic Control requires an outcome variable. Syntax: synth depvar, trunit(id) trperiod(time)"
        cmd = f"synth {vars_found[0]}, trunit(1) trperiod(2015)"
        return f"[Copilot Suggestion]: Run synthetic control with your target unit and period:\n`{cmd}`"

    def _handle_event_study(self, text, cols):
        vars_found = self._find_variables(text, cols)
        if len(vars_found) < 4:
            return "[Copilot] Event study requires: depvar, unit(var), time(var), and treattime(var)."
        dep, unit, time, treat = vars_found[0], vars_found[1], vars_found[2], vars_found[3]
        cmd = f"eventstudy {dep}, unit({unit}) time({time}) treattime({treat})"
        res = self.interpreter.execute(cmd)
        self.goldfish.record_command(cmd, res, self.interpreter.estimates.last_model)
        return res.output

    def _handle_esttab(self, text):
        fmt = "booktabs" if "latex" in text or "tex" in text else ("docx" if "word" in text or "docx" in text else "plain")
        cmd = f"esttab, {fmt}" if fmt != "plain" else "esttab"
        res = self.interpreter.execute(cmd)
        return res.output

    def _handle_summarize(self, text, cols):
        vars_found = self._find_variables(text, cols)
        cmd = f"summarize {' '.join(vars_found)}" if vars_found else "summarize"
        res = self.interpreter.execute(cmd)
        self.goldfish.record_command(cmd, res)
        return res.output

    def _handle_correlate(self, text, cols):
        vars_found = self._find_variables(text, cols)
        cmd = f"correlate {' '.join(vars_found)}" if vars_found else "correlate"
        res = self.interpreter.execute(cmd)
        self.goldfish.record_command(cmd, res)
        return res.output

    def _handle_tabulate(self, text, cols):
        vars_found = self._find_variables(text, cols)
        if not vars_found:
            return "[Copilot] Please specify a categorical variable to tabulate."
        cmd = f"tabulate {' '.join(vars_found[:2])}"
        res = self.interpreter.execute(cmd)
        self.goldfish.record_command(cmd, res)
        return res.output

    def _format_memory_status(self):
        lines = [
            "Command assistant (rule-based) Memory Status:",
            "-" * 50,
            f"• Goldfish Buffer: {len(self.goldfish.history)} recent session interactions",
            f"• Last Command: {self.goldfish.last_command or '(none)'}",
            f"• Active Model in Memory: {bool(self.interpreter.estimates.last_model)}",
            f"• Elephant Codebooks: {len(self.elephant.data['codebooks'])} projects recorded",
            f"• Default User VCE: {self.elephant.data['user_preferences'].get('default_vce', 'robust')}",
            "-" * 50
        ]
        return "\n".join(lines)

    def _handle_codebook_query(self, text, orig_text, cols):
        project = "default"
        # e.g. "codebook set price = Vehicle price in USD"
        m = re.match(r'codebook\s+set\s+([a-zA-Z0-9_]+)\s*=\s*(.*)', orig_text, re.IGNORECASE)
        if m:
            varname, desc = m.group(1), m.group(2).strip()
            self.elephant.set_codebook_entry(project, varname, desc)
            return f"[Command assistant (rule-based)] Saved codebook entry: `{varname}` = {desc}"

        cb = self.elephant.get_codebook(project)
        if not cb:
            return "[Command assistant (rule-based)] No persistent codebook entries recorded yet. Set one with: `ai codebook set varname = Description`."
        lines = ["Persistent Project Codebook:"]
        for k, v in cb.items():
            lines.append(f"  • {k}: {v}")
        return "\n".join(lines)


# Backward-compatible aliases
ElephantGoldfishCopilot = SessionMemoryCopilot
RuleBasedAssistant = SessionMemoryCopilot
