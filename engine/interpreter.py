"""
engine/interpreter.py - Potatostics Command Interpreter & Grammar Parser
Decomposes commands using engine/parser.py into:
command [varlist] [if exp] [in range] [using file] [, options]
Supports whitelisted AST qualifiers (if/in), factor-variables (i., c., #, ##),
tsset/xtset gap detection, and vce(robust|cluster|hc2|hc3).
"""

import os
import re
import tempfile
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

from engine.parser import (
    parse_command_line,
    evaluate_if_in_qualifiers,
    evaluate_dta_expression,
    expand_estimation_terms,
    expand_factor_variables,
    CommandResult
)
from engine.formatter import (
    format_describe, format_summarize, format_summarize_detail, format_regression,
    format_tabulate_1way, format_tabulate_2way, format_correlate,
    format_logit, format_ivregress, format_panel_fe, format_panel_re, format_hausman, format_wald_test,
    format_vif, format_hettest, format_ovtest, format_dwatson,
    format_dfuller, format_corrgram, format_arima, format_did, format_psmatch,
    format_reghdfe, format_probit, format_poisson, format_nbreg, format_mlogit, format_tobit,
    format_cnsreg, format_sureg, format_nlcom, format_testnl,
    format_var, format_stcox,
    format_qreg, format_margins, format_lincom,
    format_csdid, format_synth, format_event_study,
    format_ologit, format_brant_test, format_did_multiplegt, format_svy_regression
)
from engine.causal_modern import format_bacondecomp, format_drdid
from engine.estimation import (
    compute_summarize, compute_regression, standardized_ols_residuals, compute_tabulate_1way,
    compute_correlate, compute_logit,
    compute_ivregress_2sls, compute_panel_fe, compute_panel_re, compute_wald_test,
    compute_svy_regression
)
from engine.advanced_econometrics import (
    compute_reghdfe, compute_probit, compute_poisson, compute_nbreg, compute_mlogit, compute_tobit,
    compute_cnsreg, compute_sureg, compute_nlcom, compute_testnl,
    compute_var, compute_stcox, compute_sts_list, format_sts_list,
    compute_qreg, compute_margins, compute_lincom, compute_ologit
)
from engine.causal_modern import (
    compute_csdid, compute_synth, compute_event_study, compute_weak_iv_diagnostics,
    compute_did_multiplegt, compute_bacondecomp, compute_drdid
)
from engine.bigdata import (
    compute_streaming_ols
)
from engine.exporters import ModelTableExporter, DemographicTableExporter
from engine.chart_studio import plot_event_study, plot_coefplot
from engine.diagnostics import (
    compute_vif, compute_hettest, compute_ovtest, compute_dwatson,
    compute_firststage, compute_endogenous, compute_overid, compute_brant_test
)
from engine.timeseries import (
    compute_dfuller, compute_corrgram, compute_arima
)
from engine.causal import (
    compute_did, compute_psmatch
)
from engine.estimates import EstimatesStore
from engine.copilot_egm import ElephantGoldfishCopilot
from engine.report_generator import generate_docx_report
from engine.plugins_manager import PluginsManager
from engine.macros import MacroManager
from engine.script_runner import ScriptRunner
from engine.citations import lookup_citations, format_citation_response, CITATIONS_DB


class Interpreter:
    def __init__(self, dataset, plot_callback=None, is_cancelled=None):
        self.dataset = dataset
        self.plot_callback = plot_callback
        self.is_cancelled = is_cancelled
        self.estimates = EstimatesStore()
        self.copilot = ElephantGoldfishCopilot(self)
        self.panelvar = None
        self.timevar = None
        self.survey_design = {}
        self.putexcel_file = None
        self.putexcel_sheet = None
        self.constraints = {}
        self.survival_timevar = None
        self.survival_failvar = None

        # Macro and Return Registers (Phase 2C)
        self.macro_mgr = MacroManager()
        self.macro_manager = self.macro_mgr
        self.registers = self.macro_mgr.registers
        self.script_runner = ScriptRunner(self)
        
        # Script execution tracking for security boundaries
        self._script_depth = 0
        self.cli_script_mode = False

        # User plugins dynamic registry
        self.custom_commands = {}
        self.plugins_mgr = PluginsManager(self)
        self.plugins_mgr.load_all_plugins()

        # Session command history
        self.history = []

    @property
    def is_script_context(self) -> bool:
        """Returns True if the current command is executing inside a script/do-file."""
        return bool(getattr(self, '_script_depth', 0) > 0 or getattr(self, 'cli_script_mode', False))

    @property
    def global_macros(self):
        """Returns global macro dictionary."""
        return self.macro_mgr.globals

    def register_command(self, verb, handler):
        """Registers a custom user command verb and handler."""
        self.custom_commands[verb.strip().lower()] = handler


    def check_cancelled(self):
        if callable(self.is_cancelled):
            return self.is_cancelled()
        return bool(self.is_cancelled)

    def get_time_context(self):
        return {
            'timevar': self.timevar,
            'panelvar': self.panelvar
        }

    def filter_data(self, cmd, df=None):
        """Filters dataset according to parsed cmd.if_expr and cmd.in_range."""
        target_df = self.dataset.df if df is None else df
        if target_df.empty:
            return pd.Series([], dtype=bool), target_df
        time_ctx = self.get_time_context()
        mask, df_sub = evaluate_if_in_qualifiers(
            target_df,
            if_expr=cmd.if_expr,
            in_range=cmd.in_range,
            time_context=time_ctx,
            scalar_context=self.registers.get_context_dict()
        )
        return mask, df_sub

    def expand_varlist_pattern(self, pattern_str):
        """Expands wildcard patterns (e.g. rep*, pri*) and lists into existing dataset variables."""
        if self.dataset.is_empty:
            return pattern_str.split()
        cols = list(self.dataset.df.columns)
        tokens = pattern_str.split()
        expanded = []
        for tok in tokens:
            if '*' in tok or '?' in tok:
                import fnmatch
                matched = [c for c in cols if fnmatch.fnmatch(c, tok)]
                if matched:
                    expanded.extend(matched)
                else:
                    expanded.append(tok)
            elif '-' in tok:
                parts = tok.split('-')
                if len(parts) == 2 and parts[0] in cols and parts[1] in cols:
                    idx1 = cols.index(parts[0])
                    idx2 = cols.index(parts[1])
                    if idx1 <= idx2:
                        expanded.extend(cols[idx1:idx2+1])
                    else:
                        expanded.extend(cols[idx2:idx1+1])
                else:
                    expanded.append(tok)
            else:
                expanded.append(tok)
        return expanded

    def execute_with_prefixes(self, cmd_line, force_quiet=False):
        """Executes a command honoring quietly, capture, and noisily prefixes."""
        return self.execute(cmd_line, force_quiet=force_quiet)

    def execute(self, cmd_line, force_quiet=False):
        if self.check_cancelled():
            return CommandResult("(Execution cancelled by user)", ok=False, return_code=1)
        cmd_line = cmd_line.strip()
        if not cmd_line or cmd_line.startswith("*") or cmd_line.startswith("//"):
            return CommandResult("", ok=True)

        # Macro expansion
        cmd_line = self.macro_mgr.expand(cmd_line)

        # Check for command prefixes: quietly, capture, noisily
        is_quiet = force_quiet
        is_capture = False
        is_noisily = False

        while True:
            m = re.match(r'^(quietly|qui|capture|cap|noisily|noi)\b\s*(.*)$', cmd_line, re.IGNORECASE)
            if not m:
                break
            pfx = m.group(1).lower()
            cmd_line = m.group(2).strip()
            if pfx.startswith("qui"):
                is_quiet = True
            elif pfx.startswith("cap"):
                is_capture = True
            elif pfx.startswith("noi"):
                is_noisily = True

        effective_quiet = is_quiet and not is_noisily

        old_last_model = getattr(self.estimates, 'last_model', None)
        # Execute inner command
        res = self._execute_inner(cmd_line)
        res.command_text = cmd_line
        new_last_model = getattr(self.estimates, 'last_model', None)
        if new_last_model is not None and new_last_model is not old_last_model:
            try:
                res.model_snapshot = dict(new_last_model)
            except Exception:
                res.model_snapshot = new_last_model

        self.history.append(cmd_line)
        self.registers.rc = res.return_code if not res.ok else 0

        # Record in Goldfish memory buffer
        if not cmd_line.lower().startswith("ai ") and not cmd_line.lower().startswith("ask "):
            if hasattr(self, 'copilot') and hasattr(self.copilot, 'goldfish'):
                self.copilot.goldfish.record_command(cmd_line, res, getattr(self.estimates, 'last_model', None))

        if is_capture:
            if not res.ok and not is_noisily:
                return CommandResult("", ok=True, return_code=res.return_code)
            elif not res.ok and is_noisily:
                return CommandResult(res.output, ok=True, error=res.error, return_code=res.return_code)

        if effective_quiet:
            return CommandResult("", ok=res.ok, error=res.error, return_code=res.return_code)
        return res

    def _execute_inner(self, cmd_line):
        cmd_line = cmd_line.strip()
        if not cmd_line or cmd_line.startswith("*") or cmd_line.startswith("//"):
            return CommandResult("", ok=True)

        # AI Copilot queries
        if cmd_line.lower().startswith("ai ") or cmd_line.lower().startswith("ask "):
            prompt = cmd_line.split(" ", 1)[1]
            res = self.copilot.ask(prompt)
            return CommandResult(res, ok=True)

        # by / bysort prefix execution
        match_by = re.match(r'^(by|bysort)\s+(.*?)\s*:\s*(.*)$', cmd_line, re.IGNORECASE)
        if match_by:
            prefix_type = match_by.group(1).lower()
            vars_part = match_by.group(2).strip()
            sub_cmd = match_by.group(3).strip()

            sort_vars = []
            m_sort = re.search(r'\((.*?)\)', vars_part)
            if m_sort:
                sort_vars = m_sort.group(1).split()
                vars_part = re.sub(r'\(.*?\)', '', vars_part).strip()
            by_vars = vars_part.split()

            from engine.data_ops import execute_by_prefix
            return execute_by_prefix(self, by_vars, sort_vars, is_bysort=(prefix_type == 'bysort'), sub_cmd_str=sub_cmd)

        # svy prefix execution
        match_svy = re.match(r'^svy\b\s*(?:,\s*.*?)?\s*:\s*(.*)$', cmd_line, re.IGNORECASE)
        if match_svy:
            sub_cmd = match_svy.group(1).strip()
            return self._cmd_svy_prefix(sub_cmd)

        # Parse command using engine/parser
        try:
            cmd = parse_command_line(cmd_line)
        except Exception as e:
            return CommandResult(f"invalid syntax: {e}", ok=False, error=str(e), return_code=198)

        verb = cmd.verb
        if not verb:
            return CommandResult("", ok=True)

        # Route custom user plugins first
        if verb in self.custom_commands:
            try:
                handler = self.custom_commands[verb]
                import inspect
                sig = inspect.signature(handler)
                params = list(sig.parameters.values())
                if len(params) == 1:
                    res = handler(cmd)
                elif len(params) == 2:
                    res = handler(self, cmd)
                else:
                    args = cmd.body.split()
                    opts_list = list(cmd.options.keys())
                    res = handler(args, opts_list, cmd_line)

                if isinstance(res, CommandResult):
                    return res
                return CommandResult(str(res), ok=True)
            except Exception as e:
                return CommandResult(f"plugin error ({verb}): {e}", ok=False, error=str(e), return_code=198)

        # Built-in command dispatcher
        try:
            if verb in ("use",):
                return self._cmd_use(cmd)
            elif verb in ("import", "insheet"):
                return self._cmd_import(cmd)
            elif verb in ("save",):
                return self._cmd_save(cmd)
            elif verb in ("export", "outsheet"):
                return self._cmd_export(cmd)
            elif verb in ("describe", "desc", "d"):
                return self._cmd_describe(cmd)
            elif verb in ("summarize", "sum", "su"):
                return self._cmd_summarize(cmd)
            elif verb in ("regress", "reg"):
                return self._cmd_regress(cmd)
            elif verb in ("bigregress", "streamreg"):
                return self._cmd_bigregress(cmd)
            elif verb in ("logit", "logistic"):
                return self._cmd_logit(cmd, default_or=(verb == "logistic"))
            elif verb in ("ivregress",):
                return self._cmd_ivregress(cmd)
            elif verb in ("xtset",):
                return self._cmd_xtset(cmd)
            elif verb in ("tsset",):
                return self._cmd_tsset(cmd)
            elif verb in ("xtreg",):
                return self._cmd_xtreg(cmd)
            elif verb in ("hausman",):
                return self._cmd_hausman(cmd)
            elif verb in ("test",):
                return self._cmd_test(cmd)
            elif verb in ("predict",):
                return self._cmd_predict(cmd)
            elif verb in ("estimates", "est"):
                return self._cmd_estimates(cmd)
            elif verb in ("esttab", "outreg", "outreg2"):
                return self._cmd_esttab(cmd)
            elif verb in ("dtable",):
                return self._cmd_dtable(cmd)
            elif verb in ("svyset",):
                return self._cmd_svyset(cmd)
            # Diagnostics
            elif verb in ("estat",):
                return self._cmd_estat(cmd)
            elif verb in ("vif",):
                return self._cmd_vif()
            elif verb in ("hettest",):
                return self._cmd_hettest()
            elif verb in ("ovtest",):
                return self._cmd_ovtest()
            elif verb in ("dwatson",):
                return self._cmd_dwatson()
            # Time Series
            elif verb in ("dfuller", "adf"):
                return self._cmd_dfuller(cmd)
            elif verb in ("corrgram", "ac", "pac"):
                return self._cmd_corrgram(cmd)
            elif verb in ("arima",):
                return self._cmd_arima(cmd)
            # Causal Inference
            elif verb in ("didregress", "did"):
                return self._cmd_did(cmd)
            elif verb in ("psmatch", "psmatch2"):
                return self._cmd_psmatch(cmd)
            elif verb in ("csdid",):
                return self._cmd_csdid(cmd)
            elif verb in ("did_multiplegt", "didmultiplegt"):
                return self._cmd_did_multiplegt(cmd)
            elif verb in ("bacondecomp", "bacon"):
                return self._cmd_bacondecomp(cmd)
            elif verb in ("drdid", "dr_did"):
                return self._cmd_drdid(cmd)
            elif verb in ("synth",):
                return self._cmd_synth(cmd)
            elif verb in ("eventstudy",):
                return self._cmd_eventstudy(cmd)
            elif verb in ("coefplot",):
                return self._cmd_coefplot(cmd)
            # Reports
            elif verb in ("report", "putdocx"):
                return self._cmd_report(cmd)
            elif verb in ("tabulate", "tab"):
                return self._cmd_tabulate(cmd)
            elif verb in ("correlate", "corr"):
                return self._cmd_correlate(cmd)
            elif verb in ("count",):
                return self._cmd_count(cmd)
            elif verb in ("list", "l"):
                return self._cmd_list(cmd)
            elif verb in ("generate", "gen", "g"):
                return self._cmd_generate(cmd)
            elif verb in ("replace",):
                return self._cmd_replace(cmd)
            # Phase 2B Data Management Commands
            elif verb in ("merge",):
                from engine.data_ops import execute_merge
                return execute_merge(self.dataset, cmd)
            elif verb in ("append",):
                from engine.data_ops import execute_append
                return execute_append(self.dataset, cmd)
            elif verb in ("reshape",):
                from engine.data_ops import execute_reshape
                return execute_reshape(self.dataset, cmd)
            elif verb in ("collapse",):
                from engine.data_ops import execute_collapse
                return execute_collapse(self.dataset, cmd)
            elif verb in ("egen",):
                from engine.data_ops import execute_egen
                return execute_egen(self.dataset, cmd)
            elif verb in ("encode",):
                from engine.data_ops import execute_encode
                return execute_encode(self.dataset, cmd)
            elif verb in ("decode",):
                from engine.data_ops import execute_decode
                return execute_decode(self.dataset, cmd)
            elif verb in ("label", "la"):
                from engine.data_ops import execute_label
                return execute_label(self.dataset, cmd)
            elif verb in ("recode",):
                from engine.data_ops import execute_recode
                return execute_recode(self.dataset, cmd)
            elif verb in ("duplicates", "dups"):
                from engine.data_ops import execute_duplicates
                return execute_duplicates(self.dataset, cmd)
            elif verb in ("drop",):
                return self._cmd_drop(cmd)
            elif verb in ("keep",):
                return self._cmd_keep(cmd)
            elif verb in ("rename", "ren"):
                return self._cmd_rename(cmd)
            elif verb in ("sort",):
                return self._cmd_sort(cmd)
            elif verb in ("line",):
                return self._cmd_line(cmd)
            elif verb in ("scatter", "plot", "twoway"):
                return self._cmd_scatter(cmd)
            elif verb in ("histogram", "hist"):
                return self._cmd_histogram(cmd)
            elif verb in ("boxplot", "box"):
                return self._cmd_boxplot(cmd)
            elif verb in ("heatmap",):
                return self._cmd_heatmap(cmd)
            elif verb in ("binscatter", "bplot"):
                return self._cmd_binscatter(cmd)
            elif verb in ("rdplot", "rd_plot"):
                return self._cmd_rdplot(cmd)
            elif verb in ("irf", "irfplot", "irf_plot"):
                return self._cmd_irf(cmd)
            elif verb in ("synth_plot", "synthplot"):
                return self._cmd_synth_plot(cmd)
            elif verb in ("clear",):
                self.dataset.clear()
                self.panelvar = None
                self.timevar = None
                return CommandResult("(data cleared from memory)", ok=True)
            # Phase 2C Scripting & Macros
            elif verb in ("local", "loc"):
                return self._cmd_local(cmd)
            elif verb in ("global", "gl"):
                return self._cmd_global(cmd)
            elif verb in ("display", "di"):
                return self._cmd_display(cmd)
            elif verb in ("return",):
                return self._cmd_return(cmd)
            elif verb in ("ereturn",):
                return self._cmd_ereturn(cmd)
            elif verb in ("do",):
                return self._cmd_do(cmd, quiet=False)
            elif verb in ("run",):
                return self._cmd_do(cmd, quiet=True)
            # Phase 2D Advanced Econometrics
            elif verb in ("reghdfe", "areg"):
                return self._cmd_reghdfe(cmd)
            elif verb in ("probit",):
                return self._cmd_probit(cmd)
            elif verb in ("ologit",):
                return self._cmd_ologit(cmd)
            elif verb in ("brant",):
                return self._cmd_brant(cmd)
            elif verb in ("poisson",):
                return self._cmd_poisson(cmd)
            elif verb in ("nbreg",):
                return self._cmd_nbreg(cmd)
            elif verb in ("mlogit",):
                return self._cmd_mlogit(cmd)
            elif verb in ("tobit",):
                return self._cmd_tobit(cmd)
            elif verb in ("qreg", "quantile"):
                return self._cmd_qreg(cmd)
            elif verb in ("margins",):
                return self._cmd_margins(cmd)
            elif verb in ("lincom",):
                return self._cmd_lincom(cmd)
            elif verb in ("nlcom",):
                return self._cmd_nlcom(cmd)
            elif verb in ("testnl",):
                return self._cmd_testnl(cmd)
            elif verb in ("putexcel",):
                return self._cmd_putexcel(cmd)
            elif verb in ("constraint", "constr"):
                return self._cmd_constraint(cmd)
            elif verb in ("cnsreg",):
                return self._cmd_cnsreg(cmd)
            elif verb in ("sureg",):
                return self._cmd_sureg(cmd)
            elif verb in ("tsfilter",):
                return self._cmd_tsfilter(cmd)
            elif verb in ("var",):
                return self._cmd_var(cmd)
            elif verb in ("stset",):
                return self._cmd_stset(cmd)
            elif verb in ("stcox",):
                return self._cmd_stcox(cmd)
            elif verb in ("sts",):
                return self._cmd_sts(cmd)
            elif verb in ("pkg", "package", "packages"):
                return self._cmd_pkg(cmd)
            elif verb in ("ssc",):
                return self._cmd_ssc(cmd)
            elif verb in ("net",):
                return self._cmd_net(cmd)
            elif verb in ("plugin", "plugins"):
                return self._cmd_plugin(cmd)
            elif verb in ("set",):
                return self._cmd_set(cmd)
            elif verb in ("cite", "citation"):
                return self._cmd_cite(cmd)
            elif verb in ("session", "project"):
                return self._cmd_session(cmd)
            elif verb in ("help", "h"):
                return self._cmd_help(cmd)
            else:
                return CommandResult(f"unrecognized command: '{verb}' (type 'help' for command syntax)", ok=False, return_code=199)
        except Exception as e:
            return CommandResult(f"{e}", ok=False, error=str(e), return_code=198)

    # --- Command Implementations ---

    def _cmd_import(self, cmd):
        body = (cmd.body or "").strip()
        if body.lower().startswith("delimited"):
            target_str = body[len("delimited"):].strip()
            if target_str.lower().startswith("using "):
                target_str = target_str[6:].strip()
        else:
            target_str = body
        if not target_str:
            target_str = cmd.using_file or ""
        path = target_str.strip('\'"')
        if not path:
            return CommandResult("syntax error: import delimited <filename> [, clear]", ok=False, return_code=198)
        if not os.path.exists(path):
            return CommandResult(f"file '{path}' not found", ok=False, return_code=601)
        n, k = self.dataset.load_csv(path)
        self.panelvar = self.timevar = None
        return CommandResult(f"({self.dataset.dataset_label or os.path.basename(path)})\nobs: {n:,}, vars: {k:,}", ok=True)

    def _cmd_export(self, cmd):
        body = (cmd.body or "").strip()
        delimited = re.match(r"^delimited(?:\s+|$)", body, flags=re.IGNORECASE)
        if cmd.verb == "export" and not delimited:
            # The existing plain export alias follows save's filename-based format selection.
            return self._cmd_save(cmd)
        if delimited:
            target_str = body[len("delimited"):].strip()
            if target_str.lower().startswith("using "):
                target_str = target_str[6:].strip()
        else:
            target_str = body
        if not target_str:
            target_str = cmd.using_file or ""
        path = target_str.strip('\'"')
        if not path:
            return CommandResult("syntax error: export delimited [using] <filename> [, replace]", ok=False, return_code=198)
        extension = os.path.splitext(path)[1].lower()
        if not extension:
            path += ".csv"
        elif extension not in (".csv", ".txt", ".gz"):
            return CommandResult("export delimited supports .csv, .txt, and .gz; use save for binary formats",
                                 ok=False, return_code=198)
        replace = cmd.has_option("replace")
        try:
            if not replace and os.path.lexists(path):
                raise FileExistsError(path)
            # Stage beside the destination so a failed write preserves any existing export.
            with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(path))) as staging:
                ready = os.path.join(staging, os.path.basename(path))
                self.dataset.df.to_csv(ready, index=False)
                if replace:
                    os.replace(ready, path)
                elif os.name == "nt":
                    os.rename(ready, path)  # Windows rename fails if the destination exists.
                else:
                    os.link(ready, path)  # Publish without overwriting a competing export.
        except FileExistsError:
            return CommandResult(f"file '{path}' already exists; use ', replace' to overwrite", ok=False, return_code=602)
        except (OSError, ValueError) as exc:
            return CommandResult(f"export error: {exc}", ok=False, return_code=198)
        self.dataset.filename = path
        return CommandResult(f"file {path} saved", ok=True)

    def _cmd_use(self, cmd):
        target = cmd.using_file if cmd.using_file else cmd.body
        if not target:
            return CommandResult("syntax error: use <filename> [, clear]", ok=False, return_code=198)
        path = target.strip('\'"')
        if not os.path.exists(path):
            return CommandResult(f"file '{path}' not found", ok=False, return_code=601)

        if path.lower().endswith(".dta"):
            n, k = self.dataset.load_dta(path)
        elif path.lower().endswith((".csv", ".txt", ".gz")):
            n, k = self.dataset.load_csv(path)
        elif path.lower().endswith((".parquet", ".pq", ".feather", ".arrow", ".ipc")):
            n, k = self.dataset.load_parquet(path)
        else:
            return CommandResult("unsupported file format. Supported: .dta, .csv, .parquet, .feather", ok=False, return_code=603)
        self.panelvar = self.timevar = None
        return CommandResult(f"({self.dataset.dataset_label or os.path.basename(path)})\nobs: {n:,}, vars: {k:,}", ok=True)

    def _cmd_save(self, cmd):
        target = cmd.using_file if cmd.using_file else cmd.body
        if not target:
            return CommandResult("syntax error: save <filename> [, replace]", ok=False, return_code=198)
        path = target.strip('\'"')
        replace = cmd.has_option("replace")
        
        ext = os.path.splitext(path)[1].lower()
        if not ext:
            path = path + ".dta"
            ext = ".dta"

        if os.path.exists(path) and not replace:
            return CommandResult(f"file '{path}' already exists; use ', replace' to overwrite", ok=False, return_code=602)

        try:
            if ext == ".dta":
                version_opt = cmd.options.get("version")
                version = int(version_opt) if version_opt and version_opt.isdigit() else 117
                self.dataset.save_dta(path, version=version)
            elif ext in (".parquet", ".pq", ".feather", ".arrow"):
                self.dataset.save_parquet(path)
            elif ext in (".csv", ".txt", ".gz"):
                self.dataset.save_csv(path)
            else:
                return CommandResult(f"unsupported file format for save: '{ext}'. Supported: .dta, .csv, .parquet", ok=False, return_code=198)
            return CommandResult(f"file {path} saved", ok=True)
        except Exception as e:
            return CommandResult(f"save error: {e}", ok=False, return_code=198)

    def _cmd_session(self, cmd):
        tokens = cmd.body.split(None, 1)
        if not tokens:
            return CommandResult("syntax: session [save <filename.potproj> | load <filename.potproj>]", ok=False, return_code=198)
        sub = tokens[0].lower()
        if sub in ("save", "s"):
            if len(tokens) < 2:
                return CommandResult("syntax: session save <filename.potproj>", ok=False, return_code=198)
            target = tokens[1].strip().strip("\"'")
            from engine.session_manager import save_session_bundle
            try:
                provider = getattr(self, "session_presentation_provider", None)
                presentation = provider() if callable(provider) else {
                    "console_text": "",
                    "history_records": [{"command": c, "status": "ok"} for c in self.history],
                }
                path = save_session_bundle(target, self, **presentation)
                return CommandResult(f"Session saved successfully to '{path}'", ok=True)
            except Exception as e:
                return CommandResult(f"Session save error: {e}", ok=False, return_code=198)
        elif sub in ("load", "use", "open"):
            if len(tokens) < 2:
                return CommandResult("syntax: session load <filename.potproj>", ok=False, return_code=198)
            target = tokens[1].strip().strip("\"'")
            from engine.session_manager import load_session_bundle
            try:
                restored = load_session_bundle(target, self)
                observer = getattr(self, "session_loaded_observer", None)
                if callable(observer):
                    observer(restored)
                n = len(self.dataset.df) if not self.dataset.is_empty else 0
                return CommandResult(f"Session loaded successfully from '{target}' ({n} observations in memory)", ok=True)
            except Exception as e:
                return CommandResult(f"Session load error: {e}", ok=False, return_code=198)
        return CommandResult("syntax: session [save <filename.potproj> | load <filename.potproj>]", ok=False, return_code=198)

    def _cmd_describe(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        return CommandResult(format_describe(self.dataset), ok=True)

    def _cmd_summarize(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        _, df_sub = self.filter_data(cmd)
        varnames = cmd.body.split() if cmd.body else None
        if varnames:
            for v in varnames:
                if v not in self.dataset.df.columns:
                    return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
        is_detail = cmd.has_option("detail") or cmd.has_option("d")
        stats_list = compute_summarize(df_sub, varnames, detail=is_detail)
        if stats_list:
            self.registers.clear_r()
            last = stats_list[-1]
            self.registers.set_r_scalar('N', last['obs'])
            self.registers.set_r_scalar('mean', last['mean'])
            self.registers.set_r_scalar('sd', last['std'])
            self.registers.set_r_scalar('min', last['min'])
            self.registers.set_r_scalar('max', last['max'])
            self.registers.set_r_scalar('sum', last.get('sum', last['mean'] * last['obs']))
            self.registers.set_r_scalar('Var', last.get('Var', (last['std'] ** 2) if last['std'] is not None else 0.0))
            if is_detail:
                for k in ('p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'skewness', 'kurtosis', 'sum_w'):
                    if k in last:
                        self.registers.set_r_scalar(k, last[k])
        out_str = format_summarize_detail(stats_list) if is_detail else format_summarize(stats_list)
        return CommandResult(out_str, ok=True)

    def _parse_vce_option(self, cmd, default="ols"):
        """Extracts vce type and optional cluster variable from parsed command."""
        vce_val = cmd.get_option("vce", None)
        if vce_val:
            vce_str = str(vce_val).strip()
            vce_lower = vce_str.lower()
            if vce_lower.startswith("cluster"):
                parts = vce_str.split(None, 1)
                c_var = parts[1].strip("()") if len(parts) > 1 else None
                return "cluster", c_var
            elif vce_lower.startswith("spatial") or vce_lower.startswith("conley"):
                return "spatial", None
            return vce_lower, None
        if cmd.has_option("robust"):
            return "robust", None
        return default, None

    def _parse_spatial_option(self, cmd):
        vce_val = cmd.get_option("vce", "")
        vce_str = str(vce_val).strip()
        lat_m = re.search(r'lat\((.*?)\)', vce_str, re.IGNORECASE)
        lon_m = re.search(r'lon\((.*?)\)', vce_str, re.IGNORECASE)
        cut_m = re.search(r'cutoff\((.*?)\)', vce_str, re.IGNORECASE)
        spatial_opts = {}
        if lat_m:
            spatial_opts['lat'] = lat_m.group(1).strip()
        if lon_m:
            spatial_opts['lon'] = lon_m.group(1).strip()
        if cut_m:
            try:
                spatial_opts['cutoff'] = float(cut_m.group(1).strip())
            except ValueError:
                pass
        if not spatial_opts:
            clean_vce = re.sub(r'^(spatial|conley)\s*,?\s*', '', vce_str, flags=re.IGNORECASE).strip()
            tokens = clean_vce.split()
            if len(tokens) >= 3:
                spatial_opts['lat'] = tokens[0]
                spatial_opts['lon'] = tokens[1]
                try:
                    spatial_opts['cutoff'] = float(tokens[2])
                except ValueError:
                    pass
        return spatial_opts

    def _cmd_regress(self, cmd):
        if cmd.using_file:
            return self._cmd_bigregress(cmd)

        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: regress depvar [indepvars] [if] [in] [, vce(robust|cluster id|spatial)]", ok=False, return_code=198)
        
        depvar = tokens[0]
        indepvars = tokens[1:]
        
        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        if depvar not in df_sub.columns:
            return CommandResult(f"variable '{depvar}' not found", ok=False, return_code=111)

        if not pd.api.types.is_numeric_dtype(df_sub[depvar]):
            return CommandResult(
                f"variable '{depvar}' is a string; regression requires numeric variables",
                ok=False,
                error="type mismatch",
                return_code=109
            )

        try:
            # Expand factor variables
            expanded_indeps, working_df, _ = expand_estimation_terms(indepvars, df_sub)
            vce_type, cluster_var = self._parse_vce_option(cmd, default="ols")
            spatial_opts = self._parse_spatial_option(cmd) if vce_type == "spatial" else {}
            has_const = not (cmd.has_option("noconstant") or cmd.has_option("noc"))

            for v in expanded_indeps:
                if v in working_df.columns and not pd.api.types.is_numeric_dtype(working_df[v]):
                    return CommandResult(
                        f"variable '{v}' is a string; regression requires numeric variables (use factor syntax i.{v} if it is categorical)",
                        ok=False,
                        error="type mismatch",
                        return_code=109
                    )

            r = compute_regression(
                working_df,
                depvar,
                expanded_indeps,
                robust=(vce_type in ('robust', 'hc1')),
                vce=vce_type,
                cluster_var=cluster_var,
                has_const=has_const,
                spatial_opts=spatial_opts
            )
            r['raw_indepvars'] = indepvars
            self.estimates.last_model = r
            # Populate estimation registers
            self.registers.clear_e()
            self.registers.set_e_scalar('N', r['n'])
            self.registers.set_e_scalar('df_m', r['df_m'])
            self.registers.set_e_scalar('df_r', r['df_r'])
            self.registers.set_e_scalar('r2', r['r2'])
            self.registers.set_e_scalar('r2_a', r['adj_r2'])
            self.registers.set_e_scalar('F', r['f_stat'])
            self.registers.set_e_scalar('rmse', r['root_mse'])
            self.registers.set_e_scalar('mss', r['ss_m'])
            self.registers.set_e_scalar('rss', r['ss_r'])
            self.registers.set_e_macro('cmd', 'regress')
            self.registers.set_e_macro('depvar', r['depvar'])
            self.registers.set_e_macro('vce', str(r['vce']))
            if r.get('spatial_opts'):
                self.registers.set_e_scalar('spatial_cutoff', float(r['spatial_opts'].get('cutoff', 0.0)))
            self.registers.set_e_macro('properties', 'b V')
            b_dict = {item['name']: item['coef'] for item in r['coefs']}
            se_dict = {item['name']: item['se'] for item in r['coefs']}
            self.registers.set_estimates(b_dict, se_dict, r.get('vcov'))
            return CommandResult(format_regression(r), ok=True)
        except Exception as e:
            rc = 111 if "not found" in str(e) else 198
            out_msg = str(e) if rc == 111 else f"regression error: {e}"
            return CommandResult(out_msg, ok=False, error=str(e), return_code=rc)

    def _cmd_bigregress(self, cmd):
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: bigregress depvar [indepvars] [using file] [, chunksize(#) robust]", ok=False, return_code=198)

        depvar = tokens[0]
        indepvars = tokens[1:]

        source = cmd.using_file.strip('\'"') if cmd.using_file else None
        if not source:
            if self.dataset.is_empty:
                return CommandResult("no dataset in memory and no file specified with using", ok=False, return_code=111)
            source = self.dataset.df
        else:
            if not os.path.exists(source):
                return CommandResult(f"file '{source}' not found", ok=False, return_code=601)

        vce_type, _ = self._parse_vce_option(cmd)
        if cmd.has_option("robust") or cmd.has_option("hc1"):
            vce_type = "robust"

        try:
            chunk_size = int(cmd.get_option("chunksize", 100000))
        except ValueError:
            chunk_size = 100000

        try:
            r = compute_streaming_ols(source, depvar, indepvars, vce=vce_type, chunk_size=chunk_size, is_cancelled=self.check_cancelled)
            self.estimates.last_model = r
            self.registers.clear_e()
            self.registers.set_e_scalar('N', r['n'])
            self.registers.set_e_scalar('df_m', r['df_m'])
            self.registers.set_e_scalar('df_r', r['df_r'])
            self.registers.set_e_scalar('r2', r['r2'])
            self.registers.set_e_scalar('r2_a', r['r2_adj'])
            self.registers.set_e_scalar('F', r['f_stat'])
            self.registers.set_e_scalar('rmse', r['root_mse'])
            self.registers.set_e_macro('cmd', 'regress')
            self.registers.set_e_macro('depvar', r['depvar'])
            self.registers.set_e_macro('properties', 'b V')
            b_dict = {item['name']: item['coef'] for item in r['coefs']}
            se_dict = {item['name']: item['se'] for item in r['coefs']}
            self.registers.set_estimates(b_dict, se_dict, r.get('vcov'))
            return CommandResult(format_regression(r), ok=True)
        except Exception as e:
            return CommandResult(f"bigregress error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_logit(self, cmd, default_or=False):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: logit depvar [indepvars] [if] [in] [, or robust vce(cluster id)]", ok=False, return_code=198)
            
        depvar = tokens[0]
        indepvars = tokens[1:]
        
        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        if depvar not in df_sub.columns:
            return CommandResult(f"variable '{depvar}' not found", ok=False, return_code=111)

        if not pd.api.types.is_numeric_dtype(df_sub[depvar]):
            return CommandResult(
                f"variable '{depvar}' is a string; regression requires numeric variables",
                ok=False,
                error="type mismatch",
                return_code=109
            )

        try:
            expanded_indeps, working_df, _ = expand_estimation_terms(indepvars, df_sub)
            odds_ratio = default_or or cmd.has_option("or")
            vce_type, cluster_var = self._parse_vce_option(cmd, default="mle")
            r = compute_logit(
                working_df,
                depvar,
                expanded_indeps,
                odds_ratio=odds_ratio,
                vce=vce_type,
                cluster_var=cluster_var
            )
            r['raw_indepvars'] = indepvars
            self.estimates.last_model = r
            # Populate estimation registers
            self.registers.clear_e()
            self.registers.set_e_scalar('N', r.get('n', 0))
            self.registers.set_e_scalar('ll', r.get('ll', 0))
            self.registers.set_e_scalar('ll_0', r.get('ll_0', 0))
            self.registers.set_e_scalar('r2_p', r.get('pseudo_r2', 0))
            self.registers.set_e_scalar('chi2', r.get('lr_chi2', 0))
            self.registers.set_e_scalar('p', r.get('prob_chi2', 0))
            self.registers.set_e_macro('cmd', 'logit')
            self.registers.set_e_macro('depvar', r.get('depvar', ''))
            b_dict = {item['name']: item['coef'] for item in r.get('coefs', [])}
            se_dict = {item['name']: item['se'] for item in r.get('coefs', [])}
            self.registers.set_estimates(b_dict, se_dict, r.get('vcov'))
            self.registers.set_e_scalar('converged', 1 if r.get('converged', True) else 0)
            return CommandResult(format_logit(r), ok=True)
        except Exception as e:
            rc = 111 if "not found" in str(e) else 198
            out_msg = str(e) if rc == 111 else f"logit error: {e}"
            return CommandResult(out_msg, ok=False, error=str(e), return_code=rc)

    def _cmd_ivregress(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
            
        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        # Parse 2SLS syntax: [2sls] depvar [exog_pre] (endog = inst) [exog_post]
        body = cmd.body
        match = re.search(r'^(?:2sls\s+)?([a-zA-Z0-9_]+)\s*(.*?)\s*\(\s*(.*?)\s*=\s*(.*?)\s*\)\s*(.*)$', body)
        if not match:
            return CommandResult("syntax: ivregress 2sls depvar [exog] (endog = instruments) [exog] [if] [in] [, small robust vce(cluster id)]", ok=False, return_code=198)
            
        depvar = match.group(1).strip()
        exog_pre = match.group(2).strip().split()
        endog_vars = match.group(3).strip().split()
        instruments = match.group(4).strip().split()
        exog_post = match.group(5).strip().split()
        exog_vars = exog_pre + exog_post
        
        small = cmd.has_option("small")
        vce_type, cluster_var = self._parse_vce_option(cmd, default="unadjusted")

        try:
            r = compute_ivregress_2sls(
                df_sub,
                depvar,
                exog_vars,
                endog_vars,
                instruments,
                small=small,
                vce=vce_type,
                cluster_var=cluster_var
            )
            self.estimates.last_model = r
            self.registers.clear_e()
            self.registers.set_e_scalar('N', r.get('n', 0))
            self.registers.set_e_scalar('df_m', r.get('df_m', len(exog_vars) + len(endog_vars)))
            self.registers.set_e_scalar('df_r', r.get('df_r', 0))
            self.registers.set_e_scalar('r2', r.get('r2', 0))
            self.registers.set_e_macro('cmd', 'ivregress')
            self.registers.set_e_macro('depvar', r.get('depvar', ''))
            b_dict = {item['name']: item['coef'] for item in r.get('coefs', [])}
            se_dict = {item['name']: item['se'] for item in r.get('coefs', [])}
            self.registers.set_estimates(b_dict, se_dict, r.get('vcov'))
            return CommandResult(format_ivregress(r), ok=True)
        except Exception as e:
            return CommandResult(f"ivregress 2sls error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_tsset(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            if self.timevar:
                return CommandResult(f"       time variable:  {self.timevar}", ok=True)
            return CommandResult("syntax: tsset timevar", ok=False, return_code=198)
            
        timevar = tokens[0]
        if timevar not in self.dataset.df.columns:
            return CommandResult(f"variable '{timevar}' not found", ok=False, return_code=111)
            
        if self.dataset.df[timevar].duplicated().any():
            return CommandResult("repeated time values in sample", ok=False, return_code=451)
            
        self.dataset.df = self.dataset.df.sort_values(by=timevar).reset_index(drop=True)
        self.timevar = timevar
        self.panelvar = None

        # Check for gaps
        t_vals = self.dataset.df[timevar].to_numpy()
        diffs = np.diff(t_vals)
        has_gaps = not np.all(diffs == 1) if len(diffs) > 0 else False
        gap_str = "(but with gaps)" if has_gaps else ""

        lines = [
            f"       time variable:  {timevar}, {int(t_vals[0])} to {int(t_vals[-1])}",
            f"                delta:  1 unit {gap_str}".strip()
        ]
        return CommandResult("\n".join(lines), ok=True)

    def _cmd_xtset(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            if self.panelvar:
                return CommandResult(f"       panel variable:  {self.panelvar}\n        time variable:  {self.timevar or 'none'}", ok=True)
            return CommandResult("syntax: xtset panelvar [timevar]", ok=False, return_code=198)
            
        panelvar = tokens[0]
        if panelvar not in self.dataset.df.columns:
            return CommandResult(f"variable '{panelvar}' not found", ok=False, return_code=111)

        timevar = tokens[1] if len(tokens) > 1 else None
        if timevar and timevar not in self.dataset.df.columns:
            return CommandResult(f"variable '{timevar}' not found", ok=False, return_code=111)

        if timevar:
            if self.dataset.df.duplicated(subset=[panelvar, timevar]).any():
                return CommandResult("repeated time values within panel", ok=False, return_code=451)
            self.dataset.df = self.dataset.df.sort_values(by=[panelvar, timevar]).reset_index(drop=True)
            self.panelvar = panelvar
            self.timevar = timevar
            df = self.dataset.df
            complete = df[[panelvar, timevar]].notna().all(axis=None)
            balanced = complete and len(df) == df[panelvar].nunique() * df[timevar].nunique()
            status = "balanced" if balanced else "unbalanced"
            return CommandResult(f"       panel variable:  {panelvar} ({status})\n        time variable:  {timevar}", ok=True)
        else:
            self.panelvar = panelvar
            self.timevar = None
            return CommandResult(f"       panel variable:  {panelvar}", ok=True)

    def _cmd_xtreg(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        if not self.panelvar:
            return CommandResult("must specify panel variable; use xtset panelvar", ok=False, return_code=459)
            
        tokens = cmd.body.split()
        if len(tokens) < 2:
            return CommandResult("syntax: xtreg depvar indepvars [if] [in] [, fe | re] [vce(cluster id)]", ok=False, return_code=198)
            
        depvar = tokens[0]
        indepvars = tokens[1:]
        
        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        expanded_indeps, working_df, _ = expand_estimation_terms(indepvars, df_sub)
        vce_type, cluster_var = self._parse_vce_option(cmd, default="classical")

        # Determine whether model is FE or RE (standard default is RE)
        is_fe = 'fe' in cmd.options

        try:
            if is_fe:
                r = compute_panel_fe(
                    working_df,
                    depvar,
                    expanded_indeps,
                    self.panelvar,
                    self.timevar,
                    vce=vce_type,
                    cluster_var=cluster_var
                )
                self.estimates.last_model = r
                self.registers.clear_e()
                self.registers.set_e_scalar('N', r['n'])
                self.registers.set_e_scalar('N_g', r.get('n_groups', 0))
                self.registers.set_e_scalar('r2_w', r.get('r2_within', 0))
                self.registers.set_e_scalar('r2_b', r.get('r2_between', 0))
                self.registers.set_e_scalar('r2_o', r.get('r2_overall', 0))
                self.registers.set_e_scalar('sigma_u', r.get('sigma_u', 0))
                self.registers.set_e_scalar('sigma_e', r.get('sigma_e', 0))
                self.registers.set_e_scalar('rho', r.get('rho', 0))
                self.registers.set_e_macro('cmd', 'xtreg')
                self.registers.set_e_macro('model', 'fe')
                self.registers.set_e_macro('depvar', r['depvar'])
                b_dict = {item['name']: item['coef'] for item in r['coefs']}
                se_dict = {item['name']: item['se'] for item in r['coefs']}
                self.registers.set_estimates(b_dict, se_dict, r.get('vcov'))
                return CommandResult(format_panel_fe(r), ok=True)
            else:
                # Random effects (default)
                r = compute_panel_re(
                    working_df,
                    depvar,
                    expanded_indeps,
                    self.panelvar,
                    self.timevar,
                    vce=vce_type,
                    cluster_var=cluster_var
                )
                self.estimates.last_model = r
                self.registers.clear_e()
                self.registers.set_e_scalar('N', r['n'])
                self.registers.set_e_scalar('N_g', r.get('n_groups', 0))
                self.registers.set_e_scalar('g_min', r.get('min_obs', 0))
                self.registers.set_e_scalar('g_avg', r.get('avg_obs', 0.0))
                self.registers.set_e_scalar('g_max', r.get('max_obs', 0))
                self.registers.set_e_scalar('Tbar', r.get('Tbar', 0.0))
                self.registers.set_e_scalar('r2_w', r.get('r2_within', 0))
                self.registers.set_e_scalar('r2_b', r.get('r2_between', 0))
                self.registers.set_e_scalar('r2_o', r.get('r2_overall', 0))
                self.registers.set_e_scalar('sigma_u', r.get('sigma_u', 0))
                self.registers.set_e_scalar('sigma_e', r.get('sigma_e', 0))
                self.registers.set_e_scalar('sigma', r.get('sigma', 0))
                self.registers.set_e_scalar('rho', r.get('rho', 0))
                self.registers.set_e_scalar('theta', r.get('theta', 0))
                self.registers.set_e_scalar('chi2', r.get('chi2', 0))
                self.registers.set_e_scalar('df_m', r.get('df_m', 0))
                self.registers.set_e_scalar('p', r.get('prob_chi2', 0))
                self.registers.set_e_macro('cmd', 'xtreg')
                self.registers.set_e_macro('model', 're')
                self.registers.set_e_macro('depvar', r['depvar'])
                self.registers.set_e_macro('ivar', self.panelvar)
                b_dict = {item['name']: item['coef'] for item in r['coefs']}
                se_dict = {item['name']: item['se'] for item in r['coefs']}
                self.registers.set_estimates(b_dict, se_dict, r.get('vcov'))
                return CommandResult(format_panel_re(r), ok=True)
        except Exception as e:
            return CommandResult(f"xtreg error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_hausman(self, cmd):
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: hausman name1 [name2] [, constant]", ok=False, return_code=198)

        name1 = tokens[0]
        name2 = tokens[1] if len(tokens) > 1 else None

        # Retrieve models from stored estimates or last_model
        m1 = self.estimates.store.get(name1) if hasattr(self.estimates, 'store') else None
        if m1 is None and name1 == "." and self.estimates.last_model:
            m1 = self.estimates.last_model
        if m1 is None:
            return CommandResult(f"estimation result {name1} not found", ok=False, return_code=111)

        if name2 and name2 != ".":
            m2 = self.estimates.store.get(name2) if hasattr(self.estimates, 'store') else None
            if m2 is None:
                return CommandResult(f"estimation result {name2} not found", ok=False, return_code=111)
        else:
            m2 = self.estimates.last_model
            name2 = "."
            if m2 is None:
                return CommandResult("no active model in memory", ok=False, return_code=301)

        b1_dict = {item['name']: item['coef'] for item in m1['coefs']}
        b2_dict = {item['name']: item['coef'] for item in m2['coefs']}

        has_sigmamore = 'sigmamore' in cmd.options
        has_sigmaless = 'sigmaless' in cmd.options
        if has_sigmamore and has_sigmaless:
            return CommandResult("options sigmamore and sigmaless are mutually exclusive", ok=False, return_code=198)

        # Standard estimation excludes constant from Hausman test by default
        include_const = 'constant' in cmd.options
        common_vars = [v for v in m1.get('indepvars', []) if v in b2_dict]
        if include_const and '_cons' in b1_dict and '_cons' in b2_dict:
            common_vars.append('_cons')

        if not common_vars:
            return CommandResult("no common coefficients between models to test", ok=False, return_code=198)

        b1_vec = np.array([b1_dict[v] for v in common_vars], dtype=float)
        b2_vec = np.array([b2_dict[v] for v in common_vars], dtype=float)
        diff = b1_vec - b2_vec

        var_names_1 = m1.get('var_names', [item['name'] for item in m1['coefs']])
        var_names_2 = m2.get('var_names', [item['name'] for item in m2['coefs']])
        idx1 = [var_names_1.index(v) for v in common_vars]
        idx2 = [var_names_2.index(v) for v in common_vars]

        V1_sub = np.copy(m1['vcov'][np.ix_(idx1, idx1)])
        V2_sub = np.copy(m2['vcov'][np.ix_(idx2, idx2)])

        s1 = m1.get('rmse', m1.get('sigma_e', 1.0))
        s2 = m2.get('rmse', m2.get('sigma_e', 1.0))
        s1_sq = float(s1) ** 2 if s1 else 1.0
        s2_sq = float(s2) ** 2 if s2 else 1.0

        if has_sigmamore and s1_sq > 0:
            V1_sub = V1_sub * (s2_sq / s1_sq)
        elif has_sigmaless and s2_sq > 0:
            V2_sub = V2_sub * (s1_sq / s2_sq)

        V_diff = V1_sub - V2_sub

        diag_diff = np.diag(V_diff)
        se_diff = [float(np.sqrt(d)) if d > 0 else None for d in diag_diff]

        try:
            eigvals = np.linalg.eigvalsh(V_diff)
            is_pd = bool(np.all(eigvals > 1e-12))
        except Exception:
            is_pd = False

        try:
            inv_V = np.linalg.inv(V_diff)
            chi2_stat = float(diff.T @ inv_V @ diff)
            rank = len(common_vars)
        except np.linalg.LinAlgError:
            inv_V = np.linalg.pinv(V_diff)
            chi2_stat = float(diff.T @ inv_V @ diff)
            rank = int(np.linalg.matrix_rank(V_diff))
            is_pd = False

        if not is_pd or chi2_stat < 0 or any(d <= 0 for d in diag_diff):
            not_pos_def = True
            p_val = None
        else:
            not_pos_def = False
            p_val = float(1.0 - stats.chi2.cdf(chi2_stat, rank)) if rank > 0 else 1.0

        self.registers.clear_r()
        self.registers.set_r_scalar('chi2', chi2_stat)
        self.registers.set_r_scalar('df', rank)
        if p_val is not None:
            self.registers.set_r_scalar('p', p_val)
        self.registers.set_r_scalar('rank', rank)

        rows = []
        for i, v in enumerate(common_vars):
            rows.append({
                'name': v,
                'b': float(b1_vec[i]),
                'B': float(b2_vec[i]),
                'diff': float(diff[i]),
                'se_diff': se_diff[i]
            })

        res_dict = {
            'name1': name1,
            'name2': name2,
            'rows': rows,
            'chi2': chi2_stat,
            'df': rank,
            'p': p_val,
            'not_pos_def': not_pos_def
        }
        if not_pos_def:
            return CommandResult(format_hausman(res_dict), ok=False, return_code=498)
        return CommandResult(format_hausman(res_dict), ok=True)

    def _cmd_test(self, cmd):
        m = self.estimates.last_model
        if not m:
            return CommandResult("test requires a previously fitted model", ok=False, return_code=301)
        hypothesis = cmd.body
        if not hypothesis:
            return CommandResult("syntax: test exp = exp (e.g. test mpg = 0)", ok=False, return_code=198)
        try:
            r = compute_wald_test(m, hypothesis)
            if r['test_type'] == 'F':
                self.registers.set_r_scalar('F', r['stat'])
                self.registers.set_r_scalar('df', r['df_num'])
                self.registers.set_r_scalar('df_r', r.get('df_denom', r.get('df_den', 0)))
                self.registers.set_r_scalar('p', r['p_value'])
            else:
                self.registers.set_r_scalar('chi2', r['stat'])
                self.registers.set_r_scalar('df', r['df'])
                self.registers.set_r_scalar('p', r['p_value'])
            return CommandResult(format_wald_test(r), ok=True)
        except Exception as e:
            return CommandResult(f"test error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_predict(self, cmd):
        m = self.estimates.last_model
        if not m:
            return CommandResult("predict requires a previously fitted model", ok=False, return_code=301)
        tokens = cmd.body.split()
        if len(tokens) != 1:
            return CommandResult("syntax: predict newvar [if] [in] [, xb resid rstandard pr n]", ok=False, return_code=198)
            
        newvar = tokens[0]
        statistic_names = {"xb": "xb", "resid": "resid", "residuals": "resid",
                           "pr": "pr", "p": "pr", "n": "n", "rstandard": "rstandard"}
        for name, value in cmd.options.items():
            if (name not in statistic_names and name != "replace") or value is not True:
                return CommandResult(f"invalid predict option '{name}'", ok=False, return_code=198)
        requested = {statistic_names[name] for name in cmd.options if name in statistic_names}
        if len(requested) > 1:
            return CommandResult("specify only one prediction statistic", ok=False, return_code=198)
        has_replace = cmd.has_option("replace")
        if newvar in self.dataset.df.columns and not has_replace:
            return CommandResult(f"variable '{newvar}' already defined", ok=False, return_code=110)

        opt_type = requested.pop() if requested else "xb"
        if opt_type == "rstandard" and m.get('model_type') != 'ols':
            return CommandResult("rstandard requires an OLS regression", ok=False, return_code=198)

        mask, _ = self.filter_data(cmd)
        target_df = self.dataset.df

        try:
            beta = m['beta']
            var_names = m['var_names']
            model_type = m.get('model_type', 'ols').lower()
            
            # Rebuild factor/interaction columns if needed
            work_df = target_df
            factors = {}
            raw_indeps = m.get('raw_indepvars') or m.get('indepvars')
            if raw_indeps:
                try:
                    _, work_df, factors = expand_factor_variables(raw_indeps, target_df)
                except Exception:
                    pass

            # Construct matching design matrix X; NEVER silently fill with zeros
            sub_X = []
            for col in var_names:
                if col == '_cons':
                    sub_X.append(np.ones(len(target_df)))
                elif col in work_df.columns:
                    sub_X.append(work_df[col].to_numpy(dtype=float))
                else:
                    raise ValueError(f"variable '{col}' not found to compute predictions")
            X_mat = np.column_stack(sub_X)
            
            eta = X_mat @ beta
            if opt_type == "xb":
                pred = eta
            elif opt_type == "rstandard":
                y = target_df[m['depvar']].to_numpy(dtype=float)
                pred = standardized_ols_residuals(m, target_df.index, X_mat, y)
            else:
                if opt_type == "pr" and model_type not in ('logit', 'probit', 'poisson'):
                    raise ValueError(f"option pr not allowed with {model_type} models")
                if opt_type == "n" and model_type not in ('poisson', 'nbreg'):
                    raise ValueError(f"option n not allowed with {model_type} models")
                if model_type == 'logit':
                    pred = expit(eta)
                elif model_type == 'probit':
                    pred = stats.norm.cdf(eta)
                elif model_type in ('poisson', 'nbreg'):
                    with np.errstate(over='raise'):
                        pred = np.exp(eta)
                elif model_type in ('ols', 'svy_regress', 'cnsreg', '2sls', 'ivregress', 'qreg'):
                    pred = eta
                else:
                    raise ValueError(f"option {opt_type} not supported with {model_type} models")
                if opt_type == 'resid':
                    pred = target_df[m['depvar']].to_numpy(dtype=float) - pred

            if factors:
                pred[work_df[list(factors)].isna().any(axis=1).to_numpy()] = np.nan

            # Assign only where mask is True
            if newvar not in target_df.columns:
                target_df[newvar] = np.nan
            target_df.loc[mask, newvar] = pred[mask] if isinstance(pred, np.ndarray) else pred
            return CommandResult(f"(prediction variable '{newvar}' created)", ok=True)
        except Exception as e:
            return CommandResult(f"predict error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_estimates(self, cmd):
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: estimates [store <name> | restore <name> | table | export <file.tex>]", ok=False, return_code=198)
        sub = tokens[0].lower()
        if sub in ("store", "s"):
            if len(tokens) < 2:
                return CommandResult("syntax: estimates store <name>", ok=False, return_code=198)
            try:
                name = tokens[1]
                res = self.estimates.store_estimate(name)
                snapshot = {
                    field: dict(getattr(self.registers, field))
                    for field in ('e_scalars', 'e_macros', 'b', 'se')
                }
                self.estimates.store[name] = {**self.estimates.store[name], 'estimation_registers': snapshot}
                return CommandResult(res, ok=True)
            except Exception as e:
                return CommandResult(f"estimates error: {e}", ok=False, return_code=198)
        elif sub == "restore":
            if len(tokens) != 2:
                return CommandResult("syntax: estimates restore <name>", ok=False, return_code=198)
            model = self.estimates.store.get(tokens[1])
            if model is None:
                return CommandResult(f"stored estimate '{tokens[1]}' not found", ok=False, return_code=111)
            saved = model.get('estimation_registers')
            if saved is None:
                return CommandResult("this estimate has no saved registers; refit and store it again", ok=False, return_code=198)
            self.registers.clear_e()
            self.registers.e_scalars.update(saved['e_scalars'])
            self.registers.e_macros.update(saved['e_macros'])
            self.registers.set_estimates(saved['b'], saved['se'], model.get('vcov'))
            self.estimates.last_model = model
            return CommandResult(f"(model '{tokens[1]}' restored)", ok=True)
        elif sub in ("table", "t"):
            names = tokens[1:] if len(tokens) > 1 else None
            return CommandResult(self.estimates.generate_comparison_table(names), ok=True)
        elif sub in ("export",):
            if len(tokens) < 2:
                return CommandResult("syntax: estimates export <filename.tex>", ok=False, return_code=198)
            raw_path = cmd.body.split(None, 1)[1].strip()
            if (raw_path.startswith('"') and raw_path.endswith('"')) or (raw_path.startswith("'") and raw_path.endswith("'")):
                raw_path = raw_path[1:-1]
            else:
                raw_path = tokens[1]
            from engine.exporters import ModelTableExporter
            models = self.estimates.store
            if not models and self.estimates.last_model:
                models = {"(1)": self.estimates.last_model}
            if not models:
                return CommandResult("estimates error: no models to export", ok=False, return_code=198)
            exporter = ModelTableExporter(models)
            msg = exporter.export(raw_path)
            return CommandResult(msg, ok=True)
        return CommandResult("syntax: estimates [store <name> | restore <name> | table | export <file.tex>]", ok=False, return_code=198)

    # --- Diagnostics Subsystem ---

    def _cmd_estat(self, cmd):
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: estat [vif | hettest | ovtest | dwatson | weakiv | firststage | endogenous | overid]", ok=False, return_code=198)
        sub = tokens[0].lower()
        if sub == "vif":
            return self._cmd_vif()
        elif sub in ("hettest", "bpg"):
            return self._cmd_hettest()
        elif sub in ("ovtest", "reset"):
            return self._cmd_ovtest()
        elif sub in ("dwatson", "dw"):
            return self._cmd_dwatson()
        elif sub in ("weakiv", "weak_iv", "iv"):
            return self._cmd_weakiv()
        elif sub in ("firststage", "first_stage"):
            return self._cmd_firststage()
        elif sub in ("endogenous", "endog", "hausman", "wu_hausman"):
            return self._cmd_endogenous()
        elif sub in ("overid", "sargan", "overidentifying"):
            return self._cmd_overid()
        elif sub in ("brant",):
            return self._cmd_brant(cmd)
        else:
            return CommandResult(f"unrecognized estat sub-command: '{sub}'", ok=False, return_code=198)

    def _cmd_firststage(self):
        m = self.estimates.last_model
        if not m or m.get('model_type') not in ('ivregress', '2sls'):
            return CommandResult("firststage requires a previously fitted IV regression (ivregress 2sls)", ok=False, return_code=301)
        try:
            results = compute_firststage(m, self.dataset.df)
            lines = ["First-stage regression summary", "=" * 70]
            for r in results:
                lines.append(f"Target endogenous variable: {r['target']}")
                lines.append(f"  R-squared:                  {r['r2']:>10.4f}")
                lines.append(f"  Partial R-squared:          {r['partial_r2']:>10.4f}")
                lines.append(f"  First-stage F({r['df_num']}, {r['df_den']}):      {r['f_stat']:>10.4f}   Prob > F = {r['p_value']:>8.4f}")
                lines.append("-" * 70)
                lines.append(f"  {'Variable':<18} {'Coef':>10} {'Std. Err.':>10} {'t':>8} {'P>|t|':>8}")
                lines.append("-" * 70)
                for c in r['coefs']:
                    lines.append(f"  {c['name']:<18} {c['coef']:>10.4f} {c['se']:>10.4f} {c['t']:>8.2f} {c['p']:>8.4f}")
                lines.append("=" * 70)
            return CommandResult("\n".join(lines), ok=True)
        except Exception as e:
            return CommandResult(f"firststage error: {e}", ok=False, return_code=198)

    def _cmd_endogenous(self):
        m = self.estimates.last_model
        if not m or m.get('model_type') not in ('ivregress', '2sls'):
            return CommandResult("endogenous requires a previously fitted IV regression (ivregress 2sls)", ok=False, return_code=301)
        try:
            r = compute_endogenous(m, self.dataset.df)
            lines = [
                "Tests of endogeneity (Wu-Hausman / Durbin)",
                "Ho: variables are exogenous",
                "-" * 65,
                f"Wu-Hausman F({r['df_num']}, {r['df_den']})        = {r['wu_hausman_f']:>10.4f}   Prob > F    = {r['wu_p']:>8.4f}",
                f"Durbin chi2({r['durbin_df']})               = {r['durbin_chi2']:>10.4f}   Prob > chi2 = {r['durbin_p']:>8.4f}",
                "-" * 65
            ]
            return CommandResult("\n".join(lines), ok=True)
        except Exception as e:
            return CommandResult(f"endogenous error: {e}", ok=False, return_code=198)

    def _cmd_overid(self):
        m = self.estimates.last_model
        if not m or m.get('model_type') not in ('ivregress', '2sls'):
            return CommandResult("overid requires a previously fitted IV regression (ivregress 2sls)", ok=False, return_code=301)
        try:
            r = compute_overid(m, self.dataset.df)
            if not r['applicable']:
                return CommandResult(r['msg'], ok=True)
            lines = [
                "Tests of overidentifying restrictions (Sargan / Basmann)",
                "Ho: overidentifying restrictions are valid",
                "-" * 65,
                f"Sargan chi2({r['df']})               = {r['sargan_chi2']:>10.4f}   Prob > chi2 = {r['sargan_p']:>8.4f}",
                f"Basmann F({r['df']}, {r['df_den']})          = {r['basmann_f']:>10.4f}   Prob > F    = {r['basmann_p']:>8.4f}",
                "-" * 65
            ]
            return CommandResult("\n".join(lines), ok=True)
        except Exception as e:
            return CommandResult(f"overid error: {e}", ok=False, return_code=198)

    def _cmd_weakiv(self):
        m = self.estimates.last_model
        if not m or m.get('model_type') not in ('ivregress', '2sls'):
            return CommandResult("weakiv requires a previously fitted IV regression (ivregress 2sls)", ok=False, return_code=301)
        try:
            diag = compute_weak_iv_diagnostics(m, self.dataset.df)
            eff_f = diag['effective_F']
            is_strong = eff_f >= 10.0
            status = "STRONG (F >= 10)" if is_strong else "WEAK (F < 10)"
            lines = [
                "First-stage F (classical) Weak Instrument Diagnostic",
                "-" * 65,
                f"First-stage F statistic:                    {eff_f:>10.4f}",
                "Stock-Yogo rule-of-thumb threshold:                 10.00",
                "-" * 65,
                f"Instrument Strength Assessment:              {status}",
                "-" * 65
            ]
            self.macro_manager.set_scalar("r(effective_F)", eff_f)
            return CommandResult("\n".join(lines), ok=True)
        except Exception as e:
            return CommandResult(f"weakiv error: {e}", ok=False, return_code=198)

    def _cmd_vif(self):
        m = self.estimates.last_model
        if not m or m.get('model_type') != 'ols':
            return CommandResult("vif requires a previously fitted OLS regression", ok=False, return_code=301)
        try:
            r = compute_vif(m, self.dataset.df)
            return CommandResult(format_vif(r), ok=True)
        except Exception as e:
            return CommandResult(f"vif error: {e}", ok=False, return_code=198)

    def _cmd_hettest(self):
        m = self.estimates.last_model
        if not m or m.get('model_type') != 'ols':
            return CommandResult("hettest requires a previously fitted OLS regression", ok=False, return_code=301)
        try:
            r = compute_hettest(m, self.dataset.df)
            return CommandResult(format_hettest(r), ok=True)
        except Exception as e:
            return CommandResult(f"hettest error: {e}", ok=False, return_code=198)

    def _cmd_ovtest(self):
        m = self.estimates.last_model
        if not m or m.get('model_type') != 'ols':
            return CommandResult("ovtest requires a previously fitted OLS regression", ok=False, return_code=301)
        try:
            r = compute_ovtest(m, self.dataset.df)
            return CommandResult(format_ovtest(r), ok=True)
        except Exception as e:
            return CommandResult(f"ovtest error: {e}", ok=False, return_code=198)

    def _cmd_dwatson(self):
        m = self.estimates.last_model
        if not m:
            return CommandResult("dwatson requires a previously fitted regression", ok=False, return_code=301)
        try:
            r = compute_dwatson(m)
            return CommandResult(format_dwatson(r), ok=True)
        except Exception as e:
            return CommandResult(f"dwatson error: {e}", ok=False, return_code=198)

    # --- Time Series Subsystem ---

    def _cmd_dfuller(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: dfuller <varname>", ok=False, return_code=198)
        var = tokens[0]
        if var not in self.dataset.df.columns:
            return CommandResult(f"variable '{var}' not found", ok=False, return_code=111)
        try:
            lags_opt = cmd.get_option('lags') or cmd.get_option('lag')
            lags_val = int(lags_opt) if lags_opt is not None else None
            r = compute_dfuller(self.dataset.df[var], lags=lags_val)
            return CommandResult(format_dfuller(r, var), ok=True)
        except Exception as e:
            return CommandResult(f"dfuller error: {e}", ok=False, return_code=198)

    def _cmd_corrgram(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: corrgram <varname>", ok=False, return_code=198)
        var = tokens[0]
        if var not in self.dataset.df.columns:
            return CommandResult(f"variable '{var}' not found", ok=False, return_code=111)
        try:
            r = compute_corrgram(self.dataset.df[var])
            return CommandResult(format_corrgram(r, var), ok=True)
        except Exception as e:
            return CommandResult(f"corrgram error: {e}", ok=False, return_code=198)

    def _cmd_arima(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: arima depvar, ar(1) d(0) ma(1) steps(5)", ok=False, return_code=198)
        depvar = tokens[0]
        if depvar not in self.dataset.df.columns:
            return CommandResult(f"variable '{depvar}' not found", ok=False, return_code=111)

        p = int(cmd.get_option("ar", 1))
        d = int(cmd.get_option("d", 0))
        q = int(cmd.get_option("ma", 0))
        steps = int(cmd.get_option("steps", 5))

        try:
            r = compute_arima(self.dataset.df[depvar], p=p, d=d, q=q, steps=steps)
            return CommandResult(format_arima(r, depvar), ok=True)
        except Exception as e:
            return CommandResult(f"arima error: {e}", ok=False, return_code=198)

    # --- Causal Inference Subsystem ---

    def _cmd_did(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if len(tokens) < 3:
            return CommandResult("syntax: did depvar treatvar timevar [, cluster(var)]", ok=False, return_code=198)
        depvar, treatvar, timevar = tokens[0], tokens[1], tokens[2]
        cluster_var = cmd.get_option("cluster", cmd.get_option("vce", None))
        try:
            r = compute_did(self.dataset.df, depvar, treatvar, timevar, cluster_var=cluster_var)
            return CommandResult(format_did(r), ok=True)
        except Exception as e:
            return CommandResult(f"did error: {e}", ok=False, return_code=198)

    def _cmd_psmatch(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        outcome_var = cmd.get_option("outcome", None)
        if not outcome_var or len(tokens) < 2:
            return CommandResult("syntax: psmatch treatvar cov1 cov2 ..., outcome(depvar) [caliper(val)]", ok=False, return_code=198)
        treatvar = tokens[0]
        covariates = tokens[1:]
        caliper_opt = cmd.get_option("caliper", None)
        caliper = float(caliper_opt) if caliper_opt is not None else None
        try:
            r = compute_psmatch(self.dataset.df, treatvar, covariates, outcome_var, caliper=caliper)
            return CommandResult(format_psmatch(r), ok=True)
        except Exception as e:
            return CommandResult(f"psmatch error: {e}", ok=False, return_code=198)

    def _cmd_did_multiplegt(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if len(tokens) < 4:
            return CommandResult("syntax: did_multiplegt depvar unitvar timevar treatvar [if] [in] [, dynamic(#) placebo(#)]", ok=False, return_code=198)

        depvar, unitvar, timevar, treatvar = tokens[:4]
        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        dynamic = 0
        if cmd.has_option("dynamic"):
            try:
                dynamic = int(cmd.get_option("dynamic", 0))
            except ValueError:
                dynamic = 0

        placebo = 0
        if cmd.has_option("placebo"):
            try:
                placebo = int(cmd.get_option("placebo", 0))
            except ValueError:
                placebo = 0

        reps = 30
        if cmd.has_option("reps"):
            try:
                reps = int(cmd.get_option("reps", 30))
            except ValueError:
                reps = 30

        seed = 42
        if cmd.has_option("seed"):
            try:
                seed = int(cmd.get_option("seed", 42))
            except ValueError:
                seed = 42

        try:
            r = compute_did_multiplegt(
                df_sub,
                depvar,
                unitvar,
                timevar,
                treatvar,
                dynamic=dynamic,
                placebo=placebo,
                reps=reps,
                seed=seed
            )
            self.estimates.last_model = r
            self.registers.clear_e()
            self.registers.set_e_scalar('N', float(r['n']))
            self.registers.set_e_scalar('b_did_m', float(r['estimate']))
            self.registers.set_e_scalar('se_did_m', float(r['se']))
            self.registers.set_e_macro('cmd', 'did_multiplegt')
            self.registers.set_e_macro('depvar', depvar)
            b_dict = {'instantaneous': r['estimate']}
            se_dict = {'instantaneous': r['se']}
            for d in r.get('dynamic_effects', []):
                b_dict[d['horizon']] = d['estimate']
                se_dict[d['horizon']] = d['se']
            for p in r.get('placebo_effects', []):
                b_dict[p['horizon']] = p['estimate']
                se_dict[p['horizon']] = p['se']
            self.registers.set_estimates(b_dict, se_dict)

            txt = format_did_multiplegt(r)
            return CommandResult(txt, ok=True)
        except Exception as e:
            return CommandResult(f"did_multiplegt error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_bacondecomp(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if len(tokens) < 2:
            return CommandResult("syntax: bacondecomp depvar treatvar, ivar(varname) time(varname)", ok=False, return_code=198)
        depvar, treatvar = tokens[0], tokens[1]
        ivar = cmd.get_option("ivar") or cmd.get_option("id")
        timevar = cmd.get_option("time") or cmd.get_option("t")
        if not ivar or not timevar:
            return CommandResult("bacondecomp requires 'ivar(varname)' and 'time(varname)' options", ok=False, return_code=198)

        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        try:
            r = compute_bacondecomp(df_sub, depvar, treatvar, str(ivar), str(timevar))
            self.estimates.last_model = r
            self.registers.clear_r()
            self.registers.set_r_scalar('twfe', float(r['twfe']))
            self.registers.set_r_scalar('weighted_bacon', float(r['weighted_bacon']))
            self.registers.set_r_scalar('weights_sum', float(r['weights_sum']))
            self.registers.set_r_macro('cmd', 'bacondecomp')
            for s in r.get('summary', []):
                safe_k = s['comparison_type'].lower().replace(" ", "_")
                self.registers.set_r_scalar(f"w_{safe_k}", float(s['weight']))
                self.registers.set_r_scalar(f"b_{safe_k}", float(s['avg_estimate']))
            return CommandResult(format_bacondecomp(r), ok=True)
        except Exception as e:
            return CommandResult(f"bacondecomp error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_drdid(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if len(tokens) < 3:
            return CommandResult("syntax: drdid depvar treatvar timevar [covars] [, ivar(id)]", ok=False, return_code=198)
        depvar, treatvar, timevar = tokens[0], tokens[1], tokens[2]
        covars = tokens[3:]
        unitvar = cmd.get_option("ivar") or cmd.get_option("id")

        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        try:
            r = compute_drdid(df_sub, depvar, treatvar, timevar, covars=covars, unitvar=str(unitvar) if unitvar else None)
            self.estimates.last_model = r
            self.registers.clear_e()
            self.registers.set_e_scalar('N', float(r['n_obs']))
            self.registers.set_e_scalar('b_att', float(r['att']))
            self.registers.set_e_scalar('se_att', float(r['se']))
            self.registers.set_e_scalar('z', float(r['z']))
            self.registers.set_e_scalar('p', float(r['p']))
            self.registers.set_e_macro('cmd', 'drdid')
            self.registers.set_e_macro('depvar', depvar)

            b_dict = {'ATT': r['att']}
            se_dict = {'ATT': r['se']}
            V_mat = np.array([[r['se'] ** 2]])
            self.registers.set_estimates(b_dict, se_dict, V_mat)

            self.registers.clear_r()
            self.registers.set_r_scalar('att', float(r['att']))
            self.registers.set_r_scalar('se', float(r['se']))

            return CommandResult(format_drdid(r), ok=True)
        except Exception as e:
            return CommandResult(f"drdid error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_csdid(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: csdid depvar, ivar(var) timevar(var) gvar(var) [control(notyettreated|nevertreated)]", ok=False, return_code=198)

        depvar = tokens[0]
        ivar = cmd.get_option("ivar", cmd.get_option("unit", None))
        timevar = cmd.get_option("timevar", cmd.get_option("time", None))
        gvar = cmd.get_option("gvar", cmd.get_option("cohort", None))
        control_type = cmd.get_option("control", "notyettreated").lower()
        seed_opt = cmd.get_option("seed", None)
        seed = int(seed_opt) if seed_opt is not None else None
        reps_opt = cmd.get_option("reps", cmd.get_option("nboot", None))
        n_boot = int(reps_opt) if reps_opt is not None else 999

        # Positional fallback if options not used
        if not ivar and len(tokens) >= 4:
            ivar = tokens[1]
            timevar = tokens[2]
            gvar = tokens[3]
        elif not ivar and self.dataset.panel_id:
            ivar = self.dataset.panel_id
        if not timevar and self.dataset.panel_time:
            timevar = self.dataset.panel_time

        if not ivar or not timevar or not gvar:
            return CommandResult("syntax: csdid depvar, ivar(unitvar) timevar(timevar) gvar(cohortvar) [control(notyettreated|nevertreated)]", ok=False, return_code=198)

        # Filtering with if/in
        _, df = self.filter_data(cmd)
        if df.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        try:
            r = compute_csdid(df, depvar, ivar, timevar, gvar, control_type=control_type, n_boot=n_boot, seed=seed)
            self.estimates.save_last_model(r)
            self.macro_manager.set_scalar("r(att)", r['overall_att'])
            self.macro_manager.set_scalar("r(se)", r['overall_se'])
            self.macro_manager.set_scalar("r(z)", r['overall_z'])
            self.macro_manager.set_scalar("r(p)", r['overall_p'])
            self.macro_manager.set_scalar("r(chi2_pretrend)", r['pretrend_chi2'])
            self.macro_manager.set_scalar("r(p_pretrend)", r['pretrend_p'])
            self.macro_manager.set_scalar("r(N)", r['n_obs'])
            return CommandResult(format_csdid(r), ok=True)
        except Exception as e:
            return CommandResult(f"csdid error: {e}", ok=False, return_code=198)

    def _cmd_synth(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: synth depvar [predictors], trunit(id) trperiod(time) [unit(idvar) time(timevar)]", ok=False, return_code=198)

        depvar = tokens[0]
        predictors = tokens[1:] if len(tokens) > 1 else None

        trunit = cmd.get_option("trunit", cmd.get_option("treated", None))
        trperiod = cmd.get_option("trperiod", cmd.get_option("treattime", None))
        unitvar = cmd.get_option("unit", cmd.get_option("ivar", cmd.get_option("id", None))) or self.dataset.panel_id
        timevar = cmd.get_option("time", cmd.get_option("timevar", None)) or self.dataset.panel_time
        run_placebos = not cmd.has_option("nopostplacebo") and not cmd.has_option("noplacebo")

        if trunit is None or trperiod is None or not unitvar or not timevar:
            return CommandResult("syntax: synth depvar [predictors], trunit(id) trperiod(time) unit(idvar) time(timevar)", ok=False, return_code=198)

        try:
            treat_time = int(trperiod)
        except ValueError:
            try:
                treat_time = float(trperiod)
            except ValueError:
                treat_time = trperiod

        _, df = self.filter_data(cmd)
        if df.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        try:
            r = compute_synth(df, depvar, unitvar, timevar, trunit, treat_time, predictors=predictors, run_placebos=run_placebos, is_cancelled=self.check_cancelled)
            self.estimates.save_last_model(r)
            self.macro_manager.set_scalar("r(pre_mspe)", r['pre_mspe'])
            self.macro_manager.set_scalar("r(post_mspe)", r['post_mspe'])
            self.macro_manager.set_scalar("r(mspe_ratio)", r['mspe_ratio'])
            if not np.isnan(r.get('p_value_placebo', np.nan)):
                self.macro_manager.set_scalar("r(p_placebo)", r['p_value_placebo'])
            return CommandResult(format_synth(r), ok=True)
        except Exception as e:
            return CommandResult(f"synth error: {e}", ok=False, return_code=198)

    def _cmd_eventstudy(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: eventstudy depvar, unit(unitvar) time(timevar) treattime(cohortvar) [pre(4) post(5) savegraph(file)]", ok=False, return_code=198)

        depvar = tokens[0]
        unitvar = cmd.get_option("unit", cmd.get_option("ivar", None)) or self.dataset.panel_id
        timevar = cmd.get_option("time", cmd.get_option("timevar", None)) or self.dataset.panel_time
        treat_time_var = cmd.get_option("treattime", cmd.get_option("gvar", cmd.get_option("cohort", None)))

        try:
            pre_window = int(cmd.get_option("pre", 4))
        except ValueError:
            pre_window = 4
        try:
            post_window = int(cmd.get_option("post", 5))
        except ValueError:
            post_window = 5

        theme = cmd.get_option("theme", "aer")
        savegraph = cmd.get_option("savegraph", cmd.get_option("save", None))

        if not unitvar or not timevar or not treat_time_var:
            return CommandResult("syntax: eventstudy depvar, unit(unitvar) time(timevar) treattime(cohortvar) [pre(4) post(5) savegraph(file)]", ok=False, return_code=198)

        _, df = self.filter_data(cmd)
        if df.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        try:
            r = compute_event_study(df, depvar, unitvar, timevar, treat_time_var, pre_window=pre_window, post_window=post_window)
            self.estimates.save_last_model(r)
            self.macro_manager.set_scalar("r(chi2_pretrend)", r['pretrend_chi2'])
            self.macro_manager.set_scalar("r(p_pretrend)", r['pretrend_p'])
            self.macro_manager.set_scalar("r(N)", r['n'])
            self.macro_manager.set_scalar("r(r2)", r['r2'])

            extra_msg = ""
            if savegraph:
                plot_event_study(r, theme=theme, save_path=savegraph)
                extra_msg = f"\n(Event study graph saved to {savegraph})"
            elif self.plot_callback:
                self.plot_callback('eventstudy', r, theme=theme)

            return CommandResult(format_event_study(r) + extra_msg, ok=True)
        except Exception as e:
            return CommandResult(f"eventstudy error: {e}", ok=False, return_code=198)

    def _cmd_coefplot(self, cmd):
        tokens = cmd.body.split()
        model_name = tokens[0] if tokens else None
        if model_name and model_name in self.estimates.store:
            model = self.estimates.store[model_name]
        else:
            model = self.estimates.last_model

        if not model or not model.get('coefs'):
            return CommandResult("coefplot requires an active or stored estimation model with coefficients", ok=False, return_code=301)

        theme = cmd.get_option("theme", "aer")
        save_path = cmd.get_option("save", cmd.get_option("savegraph", None))
        try:
            plot_coefplot(model, theme=theme, save_path=save_path)
            if not save_path and self.plot_callback:
                self.plot_callback('coefplot', model, theme=theme)
            msg = "(Coefficient plot generated" + (f" and saved to {save_path})" if save_path else ")")
            return CommandResult(msg, ok=True)
        except Exception as e:
            return CommandResult(f"coefplot error: {e}", ok=False, return_code=198)

    # --- Reports & Exporters Subsystem ---

    def _cmd_esttab(self, cmd):
        tokens = cmd.body.split()
        # Collect models: if specific model names passed, use them; else all stored models; else last active model
        model_names = [t for t in tokens if not t.startswith("-") and not t.startswith("/") and not any(t.lower().endswith(ext) for ext in ('.tex', '.docx', '.html', '.htm', '.csv'))]
        if model_names:
            models = {name: self.estimates.store[name] for name in model_names if name in self.estimates.store}
            if not models:
                return CommandResult("none of the specified models found in estimates store", ok=False, return_code=111)
        else:
            if self.estimates.store:
                models = dict(self.estimates.store)
            elif self.estimates.last_model:
                models = {"(1)": self.estimates.last_model}
            else:
                return CommandResult("no estimation models in memory to display or export", ok=False, return_code=301)

        title = cmd.get_option("title", "Regression Models")
        exporter = ModelTableExporter(models, options=cmd.options)

        # Check target output file: either from 'using <file>' or positional if ends in .tex/.docx/.html/.csv
        target_file = cmd.using
        if not target_file:
            for t in tokens:
                if any(t.lower().endswith(ext) for ext in ('.tex', '.docx', '.html', '.htm', '.csv')):
                    target_file = t
                    break

        if target_file:
            tf_lower = target_file.lower()
            try:
                if tf_lower.endswith('.docx') or cmd.has_option('docx'):
                    if not target_file.endswith('.docx'):
                        target_file += '.docx'
                    msg = exporter.to_docx(target_file, title=title)
                    return CommandResult(f"{msg}\n\n" + exporter.to_ascii_text(), ok=True)
                elif tf_lower.endswith('.csv') or cmd.has_option('csv'):
                    if not target_file.endswith('.csv'):
                        target_file += '.csv'
                    msg = exporter.to_csv(target_file)
                    return CommandResult(f"{msg}\n\n" + exporter.to_ascii_text(), ok=True)
                elif tf_lower.endswith(('.html', '.htm')) or cmd.has_option('html'):
                    if not (target_file.endswith('.html') or target_file.endswith('.htm')):
                        target_file += '.html'
                    html_content = exporter.to_html(title=title)
                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    return CommandResult(f"(HTML table exported to {target_file})\n\n" + exporter.to_ascii_text(), ok=True)
                else:  # LaTeX / booktabs by default
                    if not target_file.endswith('.tex'):
                        target_file += '.tex'
                    tex_content = exporter.to_latex_booktabs(title=title)
                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.write(tex_content)
                    return CommandResult(f"(LaTeX table exported to {target_file})\n\n" + exporter.to_ascii_text(), ok=True)
            except Exception as e:
                return CommandResult(f"esttab export error: {e}", ok=False, return_code=198)

        # Monospace or LaTeX / HTML in console
        if cmd.has_option("tex") or cmd.has_option("latex") or cmd.has_option("booktabs"):
            return CommandResult(exporter.to_latex_booktabs(title=title), ok=True)
        elif cmd.has_option("html"):
            return CommandResult(exporter.to_html(title=title), ok=True)
        else:
            return CommandResult(exporter.to_ascii_text(), ok=True)

    def _cmd_report(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        filename = tokens[0] if tokens else "Potatostics_Report.docx"
        if not filename.endswith(".docx"):
            filename += ".docx"
        try:
            res = generate_docx_report(filename, self.dataset, self.estimates)
            return CommandResult(res, ok=True)
        except Exception as e:
            return CommandResult(f"report error: {e}", ok=False, return_code=198)

    def _cmd_dtable(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        varlist = cmd.body.split() if cmd.body else []
        by_var = cmd.options.get("by", None)
        export_file = cmd.options.get("export", None)

        try:
            exporter = DemographicTableExporter(
                df_sub,
                varlist=varlist,
                by_var=by_var,
                val_labels=self.dataset.val_labels,
                val_label_names=self.dataset.val_label_names
            )
            self.registers.clear_r()
            self.registers.set_r_scalar("N", float(len(exporter.df)))
            self.registers.set_r_scalar("k_vars", float(len(exporter.parsed_vars)))
            self.registers.set_r_macro("byvar", str(by_var) if by_var else "")

            table_txt = exporter.to_ascii_text()
            export_msg = ""
            if export_file:
                export_msg = "\n" + exporter.export(export_file)

            return CommandResult(table_txt + export_msg, ok=True)
        except Exception as e:
            return CommandResult(f"dtable error: {e}", ok=False, return_code=198)

    def _cmd_svyset(self, cmd):
        if cmd.body.strip() == "_clear" or cmd.has_option("clear"):
            self.survey_design.clear()
            return CommandResult("survey design cleared", ok=True)

        raw_str = (cmd.body or "") + " " + (cmd.raw or "")
        m_pw = re.search(r'\[(?:pweight|pw)\s*=\s*(\w+)\]', raw_str, re.IGNORECASE)
        if m_pw:
            self.survey_design['pweight'] = m_pw.group(1).strip()

        # Positional PSU / cluster variable before weights or options
        body_clean = re.sub(r'\[.*?\]', '', cmd.body or '').strip()
        body_tokens = body_clean.split()
        if body_tokens and body_tokens[0] not in ('_clear',):
            self.survey_design['psu'] = body_tokens[0]

        if cmd.has_option("strata"):
            self.survey_design['strata'] = cmd.get_option("strata").strip()
        if cmd.has_option("psu"):
            self.survey_design['psu'] = cmd.get_option("psu").strip()
        elif cmd.has_option("cluster"):
            self.survey_design['psu'] = cmd.get_option("cluster").strip()

        self.registers.clear_r()
        pw = self.survey_design.get('pweight', '<none>')
        strata = self.survey_design.get('strata', '<none>')
        psu = self.survey_design.get('psu', '<none>')

        if pw != '<none>':
            self.registers.set_r_macro('pweight', pw)
        if strata != '<none>':
            self.registers.set_r_macro('strata', strata)
        if psu != '<none>':
            self.registers.set_r_macro('psu', psu)

        lines = [
            f"      pweight: {pw}",
            "          VCE: linearized",
            f"     Strata 1: {strata}",
            f"         SU 1: {psu}",
            "        FPC 1: <zero>"
        ]
        return CommandResult("\n".join(lines), ok=True)

    def _cmd_svy_prefix(self, sub_cmd_line):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)

        try:
            sub_cmd = parse_command_line(sub_cmd_line)
        except Exception as e:
            return CommandResult(f"svy syntax error: {e}", ok=False, error=str(e), return_code=198)

        verb = sub_cmd.verb.lower()

        if verb in ("regress", "reg"):
            tokens = sub_cmd.body.split()
            if len(tokens) < 2:
                return CommandResult("syntax: svy: regress depvar indepvars", ok=False, return_code=198)

            depvar = tokens[0]
            indepvars = tokens[1:]

            _, df_sub = self.filter_data(sub_cmd)
            if df_sub.empty:
                return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

            expanded_indeps, working_df, _ = expand_estimation_terms(indepvars, df_sub)
            has_const = not (sub_cmd.has_option("noconstant") or sub_cmd.has_option("noc"))

            try:
                r = compute_svy_regression(
                    working_df,
                    depvar,
                    expanded_indeps,
                    pweight=self.survey_design.get('pweight'),
                    strata=self.survey_design.get('strata'),
                    psu=self.survey_design.get('psu'),
                    has_const=has_const
                )
                self.estimates.last_model = r
                self.registers.clear_e()
                self.registers.set_e_scalar('N', float(r['n']))
                self.registers.set_e_scalar('df_r', float(r['df_r']))
                self.registers.set_e_scalar('df_m', float(r['df_m']))
                self.registers.set_e_scalar('F', float(r['f_stat']))
                self.registers.set_e_scalar('p', float(r['prob_f']))
                self.registers.set_e_scalar('r2', float(r['r2']))
                self.registers.set_e_scalar('N_pop', float(r['pop_size']))
                self.registers.set_e_scalar('N_strata', float(r['n_strata']))
                self.registers.set_e_scalar('N_psu', float(r['n_psus']))
                self.registers.set_e_macro('cmd', 'svy: regress')
                self.registers.set_e_macro('depvar', depvar)
                self.registers.set_e_macro('vce', 'linearized')
                b_dict = {item['name']: item['coef'] for item in r['coefs']}
                se_dict = {item['name']: item['se'] for item in r['coefs']}
                self.registers.set_estimates(b_dict, se_dict, r.get('vcov'))

                txt = format_svy_regression(r)
                return CommandResult(txt, ok=True)
            except Exception as e:
                return CommandResult(f"svy regress error: {e}", ok=False, error=str(e), return_code=198)
        else:
            return CommandResult(f"svy prefix not supported for command '{verb}'", ok=False, return_code=198)

    # --- Data Operations with if/in ---

    def _cmd_tabulate(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: tabulate varname [colvar] [if] [in]", ok=False, return_code=198)
        
        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        var1 = tokens[0]
        if var1 not in df_sub.columns:
            return CommandResult(f"variable '{var1}' not found", ok=False, return_code=111)

        if len(tokens) == 1:
            val_map = self.dataset.val_labels.get(self.dataset.val_label_names.get(var1, None), None)
            r = compute_tabulate_1way(df_sub, var1, val_map)
            self.registers.clear_r()
            self.registers.set_r_scalar('N', r['total'])
            self.registers.set_r_scalar('r', len(r['rows']))
            val_counts = df_sub[var1].value_counts(sort=False).sort_index()
            return CommandResult(format_tabulate_1way(val_counts, len(df_sub), var1, val_map), ok=True)
        else:
            var2 = tokens[1]
            if var2 not in df_sub.columns:
                return CommandResult(f"variable '{var2}' not found", ok=False, return_code=111)
            ct = pd.crosstab(df_sub[var1], df_sub[var2])
            self.registers.clear_r()
            self.registers.set_r_scalar('N', len(df_sub[[var1, var2]].dropna()))
            self.registers.set_r_scalar('r', ct.shape[0])
            self.registers.set_r_scalar('c', ct.shape[1])
            return CommandResult(format_tabulate_2way(ct, var1, var2), ok=True)

    def _cmd_correlate(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        varnames = cmd.body.split() if cmd.body else [c for c in df_sub.columns if pd.api.types.is_numeric_dtype(df_sub[c])]
        for v in varnames:
            if v not in df_sub.columns:
                return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
        is_covariance = cmd.has_option("covariance") or cmd.has_option("cov")
        r = compute_correlate(df_sub, varnames, covariance=is_covariance)
        return CommandResult(format_correlate(r), ok=True)

    def _cmd_count(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        _, df_sub = self.filter_data(cmd)
        cnt = len(df_sub)
        self.registers.clear_r()
        self.registers.set_r_scalar('N', cnt)
        return CommandResult(f"{cnt:,}", ok=True)

    def _cmd_list(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("(0 observations)", ok=True)
        varnames = cmd.body.split() if cmd.body else list(df_sub.columns)
        for v in varnames:
            if v not in df_sub.columns:
                return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
        
        limit = min(50, len(df_sub))
        sub_view = df_sub[varnames].iloc[:limit]
        return CommandResult(sub_view.to_string(), ok=True)

    def _cmd_generate(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        # Syntax: generate [type] newvar = expression
        body = cmd.body
        if "=" not in body:
            return CommandResult("syntax error: generate newvar = expression [if] [in]", ok=False, return_code=198)

        left, expr = body.split("=", 1)
        left_tokens = left.strip().split()
        if not left_tokens:
            return CommandResult("syntax error: missing variable name in generate", ok=False, return_code=198)
        newvar = left_tokens[-1]

        if newvar in self.dataset.df.columns:
            return CommandResult(f"variable '{newvar}' already defined", ok=False, return_code=110)

        mask, _ = self.filter_data(cmd)
        time_ctx = self.get_time_context()

        try:
            val_series = evaluate_dta_expression(
                expr,
                self.dataset.df,
                time_context=time_ctx,
                scalar_context=self.registers.get_context_dict()
            )
            if not isinstance(val_series, pd.Series):
                val_series = pd.Series(val_series, index=self.dataset.df.index)

            # In boolean expressions evaluate strictly to numeric 1.0 and 0.0
            if pd.api.types.is_bool_dtype(val_series):
                val_series = val_series.astype(float)

            # Assign matching rows safely without FutureWarning or dtype corruption
            if mask.all():
                self.dataset.df[newvar] = val_series
            else:
                if pd.api.types.is_string_dtype(val_series) or pd.api.types.is_object_dtype(val_series):
                    s = pd.Series(index=self.dataset.df.index, dtype=object)
                    s.loc[mask] = val_series[mask]
                    self.dataset.df[newvar] = s
                else:
                    s = pd.Series(np.nan, index=self.dataset.df.index, dtype=float)
                    s.loc[mask] = val_series[mask]
                    self.dataset.df[newvar] = s
            return CommandResult(f"(variable '{newvar}' generated)", ok=True)
        except Exception as e:
            return CommandResult(f"generate error: {e}", ok=False, return_code=198)

    def _cmd_replace(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        body = cmd.body
        if "=" not in body:
            return CommandResult("syntax error: replace var = expression [if] [in]", ok=False, return_code=198)

        left, expr = body.split("=", 1)
        var = left.strip()
        if var not in self.dataset.df.columns:
            return CommandResult(f"variable '{var}' not found", ok=False, return_code=111)

        mask, _ = self.filter_data(cmd)
        time_ctx = self.get_time_context()

        try:
            val_series = evaluate_dta_expression(
                expr,
                self.dataset.df,
                time_context=time_ctx,
                scalar_context=self.registers.get_context_dict()
            )
            if not isinstance(val_series, pd.Series):
                val_series = pd.Series(val_series, index=self.dataset.df.index)

            # Type mismatch check (r(109))
            var_is_num = pd.api.types.is_numeric_dtype(self.dataset.df[var])
            var_is_str = pd.api.types.is_string_dtype(self.dataset.df[var]) or pd.api.types.is_object_dtype(self.dataset.df[var])
            val_is_str = pd.api.types.is_string_dtype(val_series)

            mask_vals = val_series[mask]
            has_str = any(isinstance(v, str) for v in mask_vals) if len(mask_vals) > 0 else False

            if var_is_num and (val_is_str or has_str):
                return CommandResult("type mismatch", ok=False, error="type mismatch", return_code=109)
            if var_is_str and not (val_is_str or has_str):
                return CommandResult("type mismatch", ok=False, error="type mismatch", return_code=109)

            if pd.api.types.is_bool_dtype(val_series) and var_is_num:
                val_series = val_series.astype(float)

            # Count actual changes
            old_vals = self.dataset.df.loc[mask, var]
            new_vals = val_series[mask]
            changes = (old_vals != new_vals) | (old_vals.isna() != new_vals.isna())
            n_changed = int(changes.sum())

            self.dataset.df.loc[mask, var] = new_vals
            return CommandResult(f"({n_changed:,} real changes made)", ok=True)
        except Exception as e:
            return CommandResult(f"replace error: {e}", ok=False, return_code=198)

    def _cmd_drop(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        if cmd.if_expr or cmd.in_range:
            mask, _ = self.filter_data(cmd)
            n_dropped = int(mask.sum())
            self.dataset.df = self.dataset.df[~mask].reset_index(drop=True)
            return CommandResult(f"({n_dropped:,} observations deleted)", ok=True)
        else:
            vars_to_drop = cmd.body.split()
            if not vars_to_drop:
                return CommandResult("syntax: drop varlist OR drop if exp OR drop in range", ok=False, return_code=198)
            found = [v for v in vars_to_drop if v in self.dataset.df.columns]
            self.dataset.df.drop(columns=found, inplace=True)
            return CommandResult(f"(dropped {len(found)} variables)", ok=True)

    def _cmd_keep(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        if cmd.if_expr or cmd.in_range:
            mask, _ = self.filter_data(cmd)
            n_dropped = len(self.dataset.df) - int(mask.sum())
            self.dataset.df = self.dataset.df[mask].reset_index(drop=True)
            return CommandResult(f"({n_dropped:,} observations deleted)", ok=True)
        else:
            vars_to_keep = cmd.body.split()
            if not vars_to_keep:
                return CommandResult("syntax: keep varlist OR keep if exp OR keep in range", ok=False, return_code=198)
            found = [v for v in vars_to_keep if v in self.dataset.df.columns]
            self.dataset.df = self.dataset.df[found]
            return CommandResult(f"(kept {len(found)} variables)", ok=True)

    def _cmd_rename(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if len(tokens) < 2:
            return CommandResult("syntax: rename oldvar newvar", ok=False, return_code=198)
        old, new = tokens[0], tokens[1]
        if old not in self.dataset.df.columns:
            return CommandResult(f"variable '{old}' not found", ok=False, return_code=111)
        self.dataset.df.rename(columns={old: new}, inplace=True)
        if old in self.dataset.var_labels:
            self.dataset.var_labels[new] = self.dataset.var_labels.pop(old)
        if hasattr(self.dataset, 'val_label_names') and old in self.dataset.val_label_names:
            self.dataset.val_label_names[new] = self.dataset.val_label_names.pop(old)
        if old in self.dataset.val_labels and new not in self.dataset.val_labels:
            self.dataset.val_labels[new] = self.dataset.val_labels.pop(old)
        return CommandResult(f"('{old}' renamed to '{new}')", ok=True)

    def _cmd_sort(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        vars_to_sort = cmd.body.split()
        if not vars_to_sort:
            return CommandResult("syntax: sort varlist", ok=False, return_code=198)
        for v in vars_to_sort:
            if v not in self.dataset.df.columns:
                return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
        self.dataset.df.sort_values(by=vars_to_sort, inplace=True)
        self.dataset.df.reset_index(drop=True, inplace=True)
        return CommandResult("(dataset sorted)", ok=True)

    def _cmd_scatter(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if len(tokens) >= 3 and tokens[0].lower() in ("scatter", "line"):
            sub_type = tokens[0].lower()
            yvar, xvar = tokens[1], tokens[2]
        elif len(tokens) >= 2:
            sub_type = "scatter"
            yvar, xvar = tokens[0], tokens[1]
        else:
            return CommandResult("syntax: scatter yvar xvar [if] [in] [, save(filepath)]", ok=False, return_code=198)
        _, df_sub = self.filter_data(cmd)
        save_path = cmd.get_option("save", None)
        if self.plot_callback:
            self.plot_callback(sub_type, df_sub, yvar=yvar, xvar=xvar, save_path=save_path)
        return CommandResult(f"({sub_type} plot generated: {yvar} on {xvar})", ok=True)

    def _cmd_line(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if len(tokens) < 2:
            return CommandResult("syntax: line yvar xvar [if] [in] [, save(filepath)]", ok=False, return_code=198)
        yvar, xvar = tokens[0], tokens[1]
        _, df_sub = self.filter_data(cmd)
        save_path = cmd.get_option("save", None)
        if self.plot_callback:
            self.plot_callback('line', df_sub, yvar=yvar, xvar=xvar, save_path=save_path)
        return CommandResult(f"(line plot generated: {yvar} on {xvar})", ok=True)

    def _cmd_histogram(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: histogram varname [if] [in]", ok=False, return_code=198)
        var = tokens[0]
        _, df_sub = self.filter_data(cmd)
        if self.plot_callback:
            self.plot_callback('histogram', df_sub, var=var)
        return CommandResult(f"(histogram generated for '{var}')", ok=True)

    def _cmd_boxplot(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: boxplot varname [if] [in] [, over(group)]", ok=False, return_code=198)
        var = tokens[0]
        byvar = cmd.get_option("over", cmd.get_option("by", None))
        _, df_sub = self.filter_data(cmd)
        if self.plot_callback:
            self.plot_callback('boxplot', df_sub, var=var, byvar=byvar)
        return CommandResult(f"(boxplot generated for '{var}')", ok=True)

    def _cmd_heatmap(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split() if cmd.body else None
        _, df_sub = self.filter_data(cmd)
        if self.plot_callback:
            self.plot_callback('heatmap', df_sub, vars=tokens)
        return CommandResult("(correlation heatmap generated)", ok=True)

    def _cmd_binscatter(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if len(tokens) < 2:
            return CommandResult("syntax: binscatter yvar xvar [if] [in], [by(group)] [nbins(N)] [controls(varlist)] [line(none|linear)] [save(path)]", ok=False, return_code=198)
        yvar, xvar = tokens[0], tokens[1]
        byvar = cmd.get_option("by", None)
        nbins = int(cmd.get_option("nbins", cmd.get_option("nquantiles", 20)))
        controls_opt = cmd.get_option("controls", None)
        controls = controls_opt.split() if controls_opt else None
        line_opt = cmd.get_option("line", "linear").lower()
        save_path = cmd.get_option("save", cmd.get_option("savegraph", None))
        theme = cmd.get_option("theme", "aer")

        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        from engine.chart_studio import plot_binscatter
        try:
            res = plot_binscatter(df_sub, yvar, xvar, byvar=byvar, n_bins=nbins, controls=controls, line=line_opt, theme=theme, save_path=save_path)
            if not save_path and self.plot_callback:
                self.plot_callback('binscatter', df_sub, yvar=yvar, xvar=xvar, byvar=byvar, n_bins=nbins, controls=controls, line=line_opt, theme=theme)
            lines = [
                f"Binned Scatter Plot: {yvar} on {xvar}" + (f" (by {byvar})" if byvar else ""),
                f"Quantile bins: {nbins} | OLS Slope: {res['slope']:.4f} (SE: {res['se']:.4f}) | R2: {res['r2']:.4f}"
            ]
            if save_path:
                lines.append(f"(Graph saved to {save_path})")
            return CommandResult("\n".join(lines), ok=True)
        except Exception as e:
            return CommandResult(f"binscatter error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_rdplot(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if len(tokens) < 2:
            return CommandResult("syntax: rdplot yvar xvar [if] [in], [c(cutoff)] [bw(bandwidth)] [nbins(N)] [save(path)]", ok=False, return_code=198)
        yvar, xvar = tokens[0], tokens[1]
        c_opt = cmd.get_option("c", cmd.get_option("cutoff", 0.0))
        cutoff = float(c_opt)
        bw_opt = cmd.get_option("bw", cmd.get_option("bandwidth", None))
        bandwidth = float(bw_opt) if bw_opt else None
        nbins = int(cmd.get_option("nbins", 20))
        save_path = cmd.get_option("save", cmd.get_option("savegraph", None))
        theme = cmd.get_option("theme", "aer")

        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        from engine.chart_studio import plot_rd
        try:
            res = plot_rd(df_sub, yvar, xvar, cutoff=cutoff, bandwidth=bandwidth, n_bins=nbins, theme=theme, save_path=save_path)
            if not save_path and self.plot_callback:
                self.plot_callback('rdplot', df_sub, yvar=yvar, xvar=xvar, cutoff=cutoff, bandwidth=res['bandwidth'], n_bins=nbins, theme=theme)
            lines = [
                f"Regression Discontinuity Plot: {yvar} on {xvar} (Cutoff c = {cutoff:0.2f})",
                f"Local Linear Jump (tau): {res['tau']:.4f} (SE: {res['se']:.4f}, p: {res['p_value']:.4f})",
                f"95% CI: [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}] | Bandwidth h = {res['bandwidth']:.3f} (N_eff = {res['n_effective']})"
            ]
            if save_path:
                lines.append(f"(Graph saved to {save_path})")
            return CommandResult("\n".join(lines), ok=True)
        except Exception as e:
            return CommandResult(f"rdplot error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_irf(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if len(tokens) < 2:
            return CommandResult("syntax: irf response_var shock_var [if] [in], [horizon(H)] [save(path)]", ok=False, return_code=198)
        response_var, shock_var = tokens[0], tokens[1]
        horizon = int(cmd.get_option("horizon", cmd.get_option("h", 12)))
        save_path = cmd.get_option("save", cmd.get_option("savegraph", None))
        theme = cmd.get_option("theme", "aer")

        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        from engine.chart_studio import plot_irf
        try:
            plot_irf(df_sub, response_var, shock_var, max_horizon=horizon, theme=theme, save_path=save_path)
            if not save_path and self.plot_callback:
                self.plot_callback('irf', df_sub, response_var=response_var, shock_var=shock_var, max_horizon=horizon, theme=theme)
            lines = [
                f"Local Projections IRF: Response of {response_var} to {shock_var} (Horizons: 0 to {horizon})",
                "Inference: Newey-West HAC Standard Errors with 68% and 95% confidence bands"
            ]
            if save_path:
                lines.append(f"(Graph saved to {save_path})")
            return CommandResult("\n".join(lines), ok=True)
        except Exception as e:
            return CommandResult(f"irf error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_synth_plot(self, cmd):
        last_m = self.estimates.last_model
        if not last_m or last_m.get('model_type') != 'synth':
            return CommandResult("last estimation was not synth; run synth first", ok=False, return_code=301)
        save_path = cmd.get_option("save", cmd.get_option("savegraph", None))
        theme = cmd.get_option("theme", "aer")
        from engine.chart_studio import plot_synthetic_control
        plot_synthetic_control(last_m, theme=theme, save_path=save_path)
        if not save_path and self.plot_callback:
            self.plot_callback('synth_plot', last_m, theme=theme)
        msg = "(synthetic control plot generated" + (f" and saved to {save_path})" if save_path else ")")
        return CommandResult(msg, ok=True)

    def _cmd_local(self, cmd):
        body = cmd.body.strip()
        if not body:
            return CommandResult("syntax: local macro_name [= exp | text]", ok=False, return_code=198)

        m_eq = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*)$', body)
        if m_eq:
            mac_name = m_eq.group(1).strip()
            expr_str = m_eq.group(2).strip()
            ctx = self.registers.get_context_dict()
            try:
                val = evaluate_dta_expression(expr_str, df=(self.dataset.df if not self.dataset.is_empty else None), scalar_context=ctx)
                if isinstance(val, (float, np.floating)):
                    val_str = str(int(val)) if val.is_integer() else f"{val:.7g}"
                else:
                    val_str = str(val)
                self.macro_mgr.set_local(mac_name, val_str)
                return CommandResult("", ok=True)
            except Exception as e:
                return CommandResult(f"error evaluating expression: {e}", ok=False, error=str(e), return_code=198)

        parts = body.split(None, 1)
        mac_name = parts[0]
        val_str = parts[1].strip() if len(parts) > 1 else ""
        if val_str.startswith('"') and val_str.endswith('"') and len(val_str) >= 2:
            val_str = val_str[1:-1]
        self.macro_mgr.set_local(mac_name, val_str)
        return CommandResult("", ok=True)

    def _cmd_global(self, cmd):
        body = cmd.body.strip()
        if not body:
            return CommandResult("syntax: global macro_name [= exp | text]", ok=False, return_code=198)

        m_eq = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*)$', body)
        if m_eq:
            mac_name = m_eq.group(1).strip()
            expr_str = m_eq.group(2).strip()
            ctx = self.registers.get_context_dict()
            try:
                val = evaluate_dta_expression(expr_str, df=(self.dataset.df if not self.dataset.is_empty else None), scalar_context=ctx)
                if isinstance(val, (float, np.floating)):
                    val_str = str(int(val)) if val.is_integer() else f"{val:.7g}"
                else:
                    val_str = str(val)
                self.macro_mgr.set_global(mac_name, val_str)
                return CommandResult("", ok=True)
            except Exception as e:
                return CommandResult(f"error evaluating expression: {e}", ok=False, error=str(e), return_code=198)

        parts = body.split(None, 1)
        mac_name = parts[0]
        val_str = parts[1].strip() if len(parts) > 1 else ""
        if val_str.startswith('"') and val_str.endswith('"') and len(val_str) >= 2:
            val_str = val_str[1:-1]
        self.macro_mgr.set_global(mac_name, val_str)
        return CommandResult("", ok=True)

    def _cmd_display(self, cmd):
        raw_body = cmd.body.strip()
        if not raw_body:
            return CommandResult("", ok=True)

        tokens = []
        i = 0
        n = len(raw_body)
        curr_expr = []

        while i < n:
            ch = raw_body[i]
            if ch == '"':
                if curr_expr:
                    expr_str = "".join(curr_expr).strip()
                    if expr_str:
                        tokens.append(('expr', expr_str))
                    curr_expr = []
                i += 1
                str_chars = []
                while i < n and raw_body[i] != '"':
                    str_chars.append(raw_body[i])
                    i += 1
                tokens.append(('str', "".join(str_chars)))
                i += 1
            else:
                curr_expr.append(ch)
                i += 1

        if curr_expr:
            expr_str = "".join(curr_expr).strip()
            if expr_str:
                tokens.append(('expr', expr_str))

        output_parts = []
        ctx = self.registers.get_context_dict()
        df_context = self.dataset.df if not self.dataset.is_empty else None

        for kind, val in tokens:
            if kind == 'str':
                output_parts.append(val)
            else:
                try:
                    res = evaluate_dta_expression(val, df=df_context, scalar_context=ctx)
                    if isinstance(res, (float, np.floating)):
                        if np.isnan(res):
                            output_parts.append(".")
                        elif res.is_integer():
                            output_parts.append(str(int(res)))
                        else:
                            output_parts.append(f"{res:.7g}")
                    elif isinstance(res, pd.Series):
                        output_parts.append(str(res.iloc[0]) if len(res) > 0 else "")
                    else:
                        output_parts.append(str(res))
                except Exception as e:
                    output_parts.append(f"[error: {e}]")

        return CommandResult("".join(output_parts), ok=True)

    def _cmd_return(self, cmd):
        body = cmd.body.strip().lower()
        if body == "list":
            return CommandResult(self.registers.format_return_list(), ok=True)
        return CommandResult("syntax: return list", ok=False, return_code=198)

    def _cmd_ereturn(self, cmd):
        body = cmd.body.strip().lower()
        if body == "list":
            return CommandResult(self.registers.format_ereturn_list(), ok=True)
        return CommandResult("syntax: ereturn list", ok=False, return_code=198)

    def _cmd_do(self, cmd, quiet=False):
        filename = cmd.body.strip().strip('"\'')
        if not filename:
            return CommandResult("syntax: do filename [, nostop]", ok=False, return_code=198)
        if not os.path.exists(filename):
            if os.path.exists(filename + ".do"):
                filename = filename + ".do"
            else:
                return CommandResult(f"file '{filename}' not found", ok=False, return_code=601)

        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            script_text = f.read()

        nostop = cmd.has_option("nostop")
        return self.script_runner.run_script(script_text, quiet=quiet, nostop=nostop, is_do_file=True)

    # --- Phase 2D Advanced Econometric Commands ---

    def _cmd_reghdfe(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if len(tokens) < 2:
            return CommandResult("syntax: reghdfe depvar indepvars [if] [in], absorb(varlist) [vce(cluster id)]", ok=False, return_code=198)

        absorb_val = cmd.get_option("absorb", None)
        if not absorb_val:
            return CommandResult("reghdfe requires option absorb(varlist)", ok=False, return_code=198)

        absorb_vars = str(absorb_val).strip("()").split()
        depvar = tokens[0]
        indepvars = tokens[1:]

        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        for v in [depvar] + absorb_vars:
            if v not in df_sub.columns:
                return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)

        try:
            expanded_indeps, working_df, _ = expand_estimation_terms(indepvars, df_sub)
            vce_type, cluster_var = self._parse_vce_option(cmd, default="ols")

            r = compute_reghdfe(
                working_df,
                depvar,
                expanded_indeps,
                absorb_vars,
                vce=vce_type,
                cluster_var=cluster_var
            )
            self.estimates.last_model = r
            self.registers.clear_e()
            self.registers.set_e_scalar('N', r['n'])
            self.registers.set_e_scalar('df_r', r['df_r'])
            self.registers.set_e_scalar('r2', r['r2_overall'])
            self.registers.set_e_scalar('r2_w', r['r2_within'])
            self.registers.set_e_scalar('F', r['f_stat'])
            self.registers.set_e_macro('cmd', 'reghdfe')
            self.registers.set_e_macro('depvar', r['depvar'])
            b_dict = {item['name']: item['coef'] for item in r['coefs']}
            se_dict = {item['name']: item['se'] for item in r['coefs']}
            self.registers.set_estimates(b_dict, se_dict, r.get('vcov'))
            return CommandResult(format_reghdfe(r), ok=True)
        except Exception as e:
            return CommandResult(f"reghdfe error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_probit(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: probit depvar [indepvars] [if] [in] [, robust vce(cluster id)]", ok=False, return_code=198)

        depvar = tokens[0]
        indepvars = tokens[1:]

        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        if depvar not in df_sub.columns:
            return CommandResult(f"variable '{depvar}' not found", ok=False, return_code=111)

        if not pd.api.types.is_numeric_dtype(df_sub[depvar]):
            return CommandResult(
                f"variable '{depvar}' is a string; regression requires numeric variables",
                ok=False,
                error="type mismatch",
                return_code=109
            )

        try:
            expanded_indeps, working_df, _ = expand_estimation_terms(indepvars, df_sub)
            vce_type, cluster_var = self._parse_vce_option(cmd, default="mle")

            r = compute_probit(
                working_df,
                depvar,
                expanded_indeps,
                vce=vce_type,
                cluster_var=cluster_var
            )
            r['raw_indepvars'] = indepvars
            self.estimates.last_model = r
            self.registers.clear_e()
            self.registers.set_e_scalar('N', r['n'])
            self.registers.set_e_scalar('ll', r['ll'])
            self.registers.set_e_scalar('ll_0', r['ll_0'])
            self.registers.set_e_scalar('r2_p', r['pseudo_r2'])
            self.registers.set_e_scalar('chi2', r['lr_chi2'])
            self.registers.set_e_scalar('p', r['prob_chi2'])
            self.registers.set_e_macro('cmd', 'probit')
            self.registers.set_e_macro('depvar', r['depvar'])
            b_dict = {item['name']: item['coef'] for item in r['coefs']}
            se_dict = {item['name']: item['se'] for item in r['coefs']}
            self.registers.set_estimates(b_dict, se_dict, r.get('vcov'))
            self.registers.set_e_scalar('converged', 1 if r.get('converged', True) else 0)
            return CommandResult(format_probit(r), ok=True)
        except Exception as e:
            return CommandResult(f"probit error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_poisson(self, cmd):
        if cmd.has_option("exposure") or cmd.has_option("offset"):
            return CommandResult("exposure() and offset() are unsupported for poisson", ok=False, return_code=198)
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: poisson depvar [indepvars] [if] [in] [, irr robust vce(cluster id)]", ok=False, return_code=198)

        depvar = tokens[0]
        indepvars = tokens[1:]

        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        if depvar not in df_sub.columns:
            return CommandResult(f"variable '{depvar}' not found", ok=False, return_code=111)

        if not pd.api.types.is_numeric_dtype(df_sub[depvar]):
            return CommandResult(
                f"variable '{depvar}' is a string; regression requires numeric variables",
                ok=False,
                error="type mismatch",
                return_code=109
            )

        try:
            expanded_indeps, working_df, _ = expand_estimation_terms(indepvars, df_sub)
            irr = cmd.has_option("irr")
            vce_type, cluster_var = self._parse_vce_option(cmd, default="mle")

            r = compute_poisson(
                working_df,
                depvar,
                expanded_indeps,
                irr=irr,
                vce=vce_type,
                cluster_var=cluster_var
            )
            r['raw_indepvars'] = indepvars
            self.estimates.last_model = r
            self.registers.clear_e()
            self.registers.set_e_scalar('N', r['n'])
            self.registers.set_e_scalar('ll', r['ll'])
            self.registers.set_e_scalar('ll_0', r['ll_0'])
            self.registers.set_e_scalar('r2_p', r['pseudo_r2'])
            self.registers.set_e_scalar('chi2', r['lr_chi2'])
            self.registers.set_e_scalar('p', r['prob_chi2'])
            self.registers.set_e_macro('cmd', 'poisson')
            self.registers.set_e_macro('depvar', r['depvar'])
            b_dict = {item['name']: item['coef'] for item in r['coefs']}
            se_dict = {item['name']: item['se'] for item in r['coefs']}
            self.registers.set_estimates(b_dict, se_dict, r.get('vcov'))
            self.registers.set_e_scalar('converged', 1 if r.get('converged', True) else 0)
            return CommandResult(format_poisson(r), ok=True)
        except Exception as e:
            return CommandResult(f"poisson error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_nbreg(self, cmd):
        if cmd.has_option("exposure") or cmd.has_option("offset"):
            return CommandResult("exposure() and offset() are unsupported for nbreg", ok=False, return_code=198)
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: nbreg depvar [indepvars] [if] [in] [, irr robust vce(cluster id)]", ok=False, return_code=198)

        depvar = tokens[0]
        indepvars = tokens[1:]

        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        if depvar not in df_sub.columns:
            return CommandResult(f"variable '{depvar}' not found", ok=False, return_code=111)

        if not pd.api.types.is_numeric_dtype(df_sub[depvar]):
            return CommandResult(
                f"variable '{depvar}' is a string; regression requires numeric variables",
                ok=False,
                error="type mismatch",
                return_code=109
            )

        try:
            expanded_indeps, working_df, _ = expand_estimation_terms(indepvars, df_sub)
            irr = cmd.has_option("irr")
            vce_type, cluster_var = self._parse_vce_option(cmd, default="mle")

            r = compute_nbreg(
                working_df,
                depvar,
                expanded_indeps,
                irr=irr,
                vce=vce_type,
                cluster_var=cluster_var
            )
            r['raw_indepvars'] = indepvars
            self.estimates.last_model = r
            self.registers.clear_e()
            self.registers.set_e_scalar('N', r['n'])
            self.registers.set_e_scalar('ll', r['ll'])
            self.registers.set_e_scalar('ll_0', r['ll_0'])
            self.registers.set_e_scalar('r2_p', r['pseudo_r2'])
            self.registers.set_e_scalar('chi2', r['lr_chi2'])
            self.registers.set_e_scalar('df_m', r['df_m'])
            self.registers.set_e_scalar('p', r['prob_chi2'])
            self.registers.set_e_scalar('alpha', r['alpha'])
            self.registers.set_e_scalar('lnalpha', r['lnalpha'])
            self.registers.set_e_scalar('se_alpha', r['se_alpha'])
            self.registers.set_e_scalar('se_lnalpha', r['se_lnalpha'])
            self.registers.set_e_scalar('chi2_c', r['chi2_c'])
            self.registers.set_e_scalar('p_c', r['p_c'])
            self.registers.set_e_macro('cmd', 'nbreg')
            self.registers.set_e_macro('depvar', r['depvar'])
            b_dict = {item['name']: item['coef'] for item in r['coefs']}
            se_dict = {item['name']: item['se'] for item in r['coefs']}
            self.registers.set_estimates(b_dict, se_dict, r.get('vcov'))
            return CommandResult(format_nbreg(r), ok=True)
        except Exception as e:
            return CommandResult(f"nbreg error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_mlogit(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: mlogit depvar [indepvars] [if] [in] [, baseoutcome(#) rrr robust vce(cluster id)]", ok=False, return_code=198)

        depvar = tokens[0]
        indepvars = tokens[1:]

        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        if depvar not in df_sub.columns:
            return CommandResult(f"variable '{depvar}' not found", ok=False, return_code=111)

        if not pd.api.types.is_numeric_dtype(df_sub[depvar]):
            return CommandResult(
                f"variable '{depvar}' is a string; regression requires numeric variables",
                ok=False,
                error="type mismatch",
                return_code=109
            )

        try:
            expanded_indeps, working_df, _ = expand_estimation_terms(indepvars, df_sub)
            rrr = cmd.has_option("rrr")
            baseoutcome = cmd.get_option("baseoutcome", None) or cmd.get_option("base", None)
            vce_type, cluster_var = self._parse_vce_option(cmd, default="mle")

            r = compute_mlogit(
                working_df,
                depvar,
                expanded_indeps,
                baseoutcome=baseoutcome,
                rrr=rrr,
                vce=vce_type,
                cluster_var=cluster_var
            )
            r['raw_indepvars'] = indepvars
            self.estimates.last_model = r
            self.registers.clear_e()
            self.registers.set_e_scalar('N', r['n'])
            self.registers.set_e_scalar('ll', r['ll'])
            self.registers.set_e_scalar('ll_0', r['ll_0'])
            self.registers.set_e_scalar('r2_p', r['pseudo_r2'])
            self.registers.set_e_scalar('chi2', r['lr_chi2'])
            self.registers.set_e_scalar('df_m', r['df_m'])
            self.registers.set_e_scalar('p', r['prob_chi2'])
            self.registers.set_e_macro('cmd', 'mlogit')
            self.registers.set_e_macro('depvar', r['depvar'])
            self.registers.set_e_macro('baseoutcome', str(r['baseoutcome']))
            self.registers.set_estimates(r['b_dict'], r['se_dict'], r.get('vcov'))
            return CommandResult(format_mlogit(r), ok=True)
        except Exception as e:
            return CommandResult(f"mlogit error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_tobit(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: tobit depvar [indepvars] [if] [in], ll[(#)] [ul[(#)]]", ok=False, return_code=198)

        depvar = tokens[0]
        indepvars = tokens[1:]

        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        if depvar not in df_sub.columns:
            return CommandResult(f"variable '{depvar}' not found", ok=False, return_code=111)

        if not pd.api.types.is_numeric_dtype(df_sub[depvar]):
            return CommandResult(
                f"variable '{depvar}' is a string; regression requires numeric variables",
                ok=False,
                error="type mismatch",
                return_code=109
            )

        ll_val = None
        if cmd.has_option("ll"):
            opt_ll = cmd.get_option("ll")
            try:
                ll_val = float(opt_ll) if (opt_ll is not True and opt_ll is not None) else 0.0
            except ValueError:
                ll_val = 0.0

        ul_val = None
        if cmd.has_option("ul"):
            opt_ul = cmd.get_option("ul")
            try:
                ul_val = float(opt_ul) if (opt_ul is not True and opt_ul is not None) else 0.0
            except ValueError:
                ul_val = 0.0

        if ll_val is None and ul_val is None:
            return CommandResult("tobit requires at least one censoring limit: ll(#) or ul(#)", ok=False, return_code=198)

        try:
            expanded_indeps, working_df, _ = expand_estimation_terms(indepvars, df_sub)
            vce_type, _ = self._parse_vce_option(cmd, default="mle")
            r = compute_tobit(
                working_df,
                depvar,
                expanded_indeps,
                ll=ll_val,
                ul=ul_val,
                vce=vce_type
            )
            r['raw_indepvars'] = indepvars
            self.estimates.last_model = r
            self.registers.clear_e()
            self.registers.set_e_scalar('N', r['n'])
            self.registers.set_e_scalar('ll', r['ll'])
            self.registers.set_e_scalar('N_unc', r['n_uncensored'])
            self.registers.set_e_scalar('N_left', r['n_left'])
            self.registers.set_e_scalar('N_right', r['n_right'])
            self.registers.set_e_macro('cmd', 'tobit')
            self.registers.set_e_macro('depvar', r['depvar'])
            b_dict = {item['name']: item['coef'] for item in r['coefs']}
            se_dict = {item['name']: item['se'] for item in r['coefs']}
            self.registers.set_estimates(b_dict, se_dict, r.get('vcov'))
            self.registers.set_e_scalar('converged', 1 if r.get('converged', True) else 0)
            return CommandResult(format_tobit(r), ok=True)
        except Exception as e:
            return CommandResult(f"tobit error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_qreg(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: qreg depvar [indepvars] [if] [in] [, quantile(#)]", ok=False, return_code=198)

        depvar = tokens[0]
        indepvars = tokens[1:]

        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        if depvar not in df_sub.columns:
            return CommandResult(f"variable '{depvar}' not found", ok=False, return_code=111)

        if not pd.api.types.is_numeric_dtype(df_sub[depvar]):
            return CommandResult(
                f"variable '{depvar}' is a string; regression requires numeric variables",
                ok=False,
                error="type mismatch",
                return_code=109
            )

        q_val = 0.5
        if cmd.has_option("quantile"):
            q_val = float(cmd.get_option("quantile"))
        elif cmd.has_option("q"):
            q_val = float(cmd.get_option("q"))

        try:
            expanded_indeps, working_df, _ = expand_estimation_terms(indepvars, df_sub)
            vce_type, _ = self._parse_vce_option(cmd, default="iid")
            reps_val = int(cmd.get_option("reps", 200)) if cmd.has_option("reps") else 200
            r = compute_qreg(
                working_df,
                depvar,
                expanded_indeps,
                quantile=q_val,
                vce=vce_type,
                reps=reps_val
            )
            r['raw_indepvars'] = indepvars
            self.estimates.last_model = r
            self.registers.clear_e()
            self.registers.set_e_scalar('N', r['n'])
            self.registers.set_e_scalar('q', r['quantile'])
            self.registers.set_e_scalar('df_r', r['df_r'])
            self.registers.set_e_macro('cmd', 'qreg')
            self.registers.set_e_macro('depvar', r['depvar'])
            b_dict = {item['name']: item['coef'] for item in r['coefs']}
            se_dict = {item['name']: item['se'] for item in r['coefs']}
            self.registers.set_estimates(b_dict, se_dict, r.get('vcov'))
            self.registers.set_e_scalar('converged', 1 if r.get('converged', True) else 0)
            return CommandResult(format_qreg(r), ok=True)
        except Exception as e:
            return CommandResult(f"qreg error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_margins(self, cmd):
        m = self.estimates.last_model
        if not m:
            return CommandResult("margins requires a previously fitted model", ok=False, return_code=301)

        dydx_opt = cmd.get_option("dydx", None)
        if dydx_opt is True or dydx_opt is None:
            dydx_vars = ['*']
        else:
            dydx_vars = str(dydx_opt).strip("()").split()

        try:
            r = compute_margins(m, self.dataset.df, dydx_vars=dydx_vars)
            self.registers.clear_r()
            for row in r['margins']:
                vname = row['variable']
                self.registers.set_r_scalar(f"b_{vname}", row['dy_dx'])
                self.registers.set_r_scalar(f"se_{vname}", row['se'])
                self.registers.set_r_scalar(f"z_{vname}", row['z'])
                self.registers.set_r_scalar(f"p_{vname}", row['p'])
            return CommandResult(format_margins(r), ok=True)
        except Exception as e:
            return CommandResult(f"margins error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_lincom(self, cmd):
        m = self.estimates.last_model
        if not m:
            return CommandResult("lincom requires a previously fitted model", ok=False, return_code=301)

        expr_str = cmd.body.strip()
        if not expr_str:
            return CommandResult("syntax: lincom <expression>", ok=False, return_code=198)

        try:
            r = compute_lincom(m, expr_str)
            self.registers.clear_r()
            self.registers.set_r_scalar('estimate', r['estimate'])
            self.registers.set_r_scalar('se', r['se'])
            self.registers.set_r_scalar('t', r['t'])
            self.registers.set_r_scalar('p', r['p'])
            self.registers.set_r_scalar('lb', r['ci_lower'])
            self.registers.set_r_scalar('ub', r['ci_upper'])
            return CommandResult(format_lincom(r), ok=True)
        except Exception as e:
            return CommandResult(f"lincom error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_nlcom(self, cmd):
        m = self.estimates.last_model
        if not m:
            return CommandResult("nlcom requires a previously fitted model", ok=False, return_code=301)
        expr_str = cmd.body.strip()
        if not expr_str:
            return CommandResult("syntax: nlcom <expression>", ok=False, return_code=198)
        try:
            r = compute_nlcom(m, expr_str)
            self.registers.clear_r()
            self.registers.set_r_scalar('estimate', r['estimate'])
            self.registers.set_r_scalar('se', r['se'])
            self.registers.set_r_scalar('z', r['z'])
            self.registers.set_r_scalar('p', r['p'])
            return CommandResult(format_nlcom(r), ok=True)
        except Exception as e:
            return CommandResult(f"nlcom error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_testnl(self, cmd):
        m = self.estimates.last_model
        if not m:
            return CommandResult("testnl requires a previously fitted model", ok=False, return_code=301)
        expr_str = cmd.body.strip()
        if not expr_str:
            return CommandResult("syntax: testnl <expression> = <value>", ok=False, return_code=198)
        try:
            r = compute_testnl(m, expr_str)
            self.registers.clear_r()
            self.registers.set_r_scalar('chi2', r['chi2'])
            self.registers.set_r_scalar('df', r['df'])
            self.registers.set_r_scalar('p', r['p'])
            return CommandResult(format_testnl(r), ok=True)
        except Exception as e:
            return CommandResult(f"testnl error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_constraint(self, cmd):
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: constraint [define] # exp = val", ok=False, return_code=198)
        if tokens[0].lower() == "define":
            tokens = tokens[1:]
        if len(tokens) < 2:
            return CommandResult("syntax: constraint [define] # exp = val", ok=False, return_code=198)
        cid = tokens[0]
        rest = " ".join(tokens[1:])
        if "=" not in rest:
            return CommandResult("syntax: constraint # exp = val", ok=False, return_code=198)
        left, right = rest.split("=", 1)
        rhs_val = float(right.strip())
        terms = re.findall(r'([+-]?\s*[\d.]*\s*\*?\s*[a-zA-Z0-9_]+)', left)
        coeffs = {}
        for term in terms:
            term = term.strip().replace(" ", "")
            if not term:
                continue
            if "*" in term:
                c_part, v_part = term.split("*", 1)
                c_val = float(c_part) if c_part not in ("", "+", "-") else (-1.0 if c_part == "-" else 1.0)
            else:
                m_v = re.match(r'^([+-]?[\d.]*)([a-zA-Z_][a-zA-Z0-9_]*)$', term)
                if m_v:
                    c_str, v_part = m_v.groups()
                    c_val = float(c_str) if c_str not in ("", "+", "-") else (-1.0 if c_str == "-" else 1.0)
                else:
                    v_part = term
                    c_val = 1.0
            coeffs[v_part] = c_val
        self.constraints[cid] = {'coeffs': coeffs, 'rhs': rhs_val, 'raw': rest}
        return CommandResult(f"constraint {cid} defined: {rest}", ok=True)

    def _cmd_cnsreg(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if len(tokens) < 2:
            return CommandResult("syntax: cnsreg depvar indepvars, constraints(numlist)", ok=False, return_code=198)
        depvar = tokens[0]
        indepvars = tokens[1:]
        constr_opt = cmd.get_option("constraints", None) or cmd.get_option("constraint", None)
        if not constr_opt:
            return CommandResult("cnsreg requires constraints() option", ok=False, return_code=198)
        cids = [c.strip() for c in constr_opt.split() if c.strip()]
        specs = []
        for cid in cids:
            if cid in self.constraints:
                specs.append(self.constraints[cid])
            else:
                return CommandResult(f"constraint {cid} not defined", ok=False, return_code=111)
        _, df_sub = self.filter_data(cmd)
        expanded_indeps, working_df, _ = expand_estimation_terms(indepvars, df_sub)
        try:
            r = compute_cnsreg(working_df, depvar, expanded_indeps, specs)
            self.estimates.last_model = r
            self.registers.clear_e()
            self.registers.set_e_scalar('N', r['n'])
            self.registers.set_e_scalar('df_r', r['df_r'])
            self.registers.set_e_scalar('rmse', r['rmse'])
            self.registers.set_e_macro('cmd', 'cnsreg')
            self.registers.set_e_macro('depvar', r['depvar'])
            b_dict = {item['name']: item['coef'] for item in r['coefs']}
            se_dict = {item['name']: item['se'] for item in r['coefs']}
            self.registers.set_estimates(b_dict, se_dict, r.get('vcov'))
            return CommandResult(format_cnsreg(r), ok=True)
        except Exception as e:
            return CommandResult(f"cnsreg error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_sureg(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        eq_matches = re.findall(r'\(([^)]+)\)', cmd.body)
        if len(eq_matches) < 2:
            return CommandResult("syntax: sureg (depvar1 indepvars1) (depvar2 indepvars2) [...]", ok=False, return_code=198)
        _, df_sub = self.filter_data(cmd)
        equations = []
        for eq_str in eq_matches:
            parts = eq_str.strip().split()
            if not parts:
                continue
            dep = parts[0]
            indeps = parts[1:]
            exp_indeps, df_sub, _ = expand_estimation_terms(indeps, df_sub)
            equations.append((dep, exp_indeps))
        try:
            r = compute_sureg(df_sub, equations)
            self.estimates.last_model = r
            self.registers.clear_e()
            self.registers.set_e_scalar('N', r['n'])
            self.registers.set_e_macro('cmd', 'sureg')
            self.registers.set_estimates(r['b_dict'], r['se_dict'], r.get('vcov'))
            return CommandResult(format_sureg(r), ok=True)
        except Exception as e:
            return CommandResult(f"sureg error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_putexcel(self, cmd):
        import openpyxl
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: putexcel set <file.xlsx> [, replace modify] OR putexcel <cell> = (<expr>) ...", ok=False, return_code=198)

        if tokens[0].lower() == "set":
            raw_path = tokens[1].strip('"\'')
            sheet_name = cmd.get_option("sheet", "Sheet1").strip('"\'')
            is_replace = cmd.has_option("replace")
            is_modify = cmd.has_option("modify")

            if is_replace and os.path.exists(raw_path):
                os.remove(raw_path)

            if os.path.exists(raw_path) and is_modify:
                wb = openpyxl.load_workbook(raw_path)
            else:
                wb = openpyxl.Workbook()
                if wb.active.title != sheet_name:
                    wb.active.title = sheet_name

            wb.save(raw_path)
            self.putexcel_file = raw_path
            self.putexcel_sheet = sheet_name
            return CommandResult(f"file {raw_path} saved", ok=True)

        if not self.putexcel_file:
            return CommandResult("putexcel set must be specified before writing cells", ok=False, return_code=198)

        wb = openpyxl.load_workbook(self.putexcel_file)
        ws = wb[self.putexcel_sheet] if self.putexcel_sheet in wb.sheetnames else wb.active

        from openpyxl.utils.cell import coordinate_to_tuple

        # Support assignments: A1 = ("text") B2 = (123) A5 = matrix(e(b)) etc.
        assignments = re.findall(r'([A-Za-z]+[0-9]+)\s*=\s*(matrix\([^)]*\)[)]?|\([^)]*\)|"[^"]*"|\'[^\']*\'|\S+)', cmd.body)
        for cell_ref, raw_expr in assignments:
            cell_ref = cell_ref.upper()
            expr = raw_expr.strip()

            # Handle matrix(e(b)) or matrix(e(V))
            if expr.lower().startswith("matrix(") and expr.endswith(")"):
                inner = expr[7:-1].strip().lower()
                row_idx, col_idx = coordinate_to_tuple(cell_ref)
                if inner == "e(b)":
                    col_offset = 0
                    for vname, val in self.registers.b.items():
                        ws.cell(row=row_idx, column=col_idx + col_offset, value=float(val))
                        col_offset += 1
                elif inner == "e(v)":
                    vcov = self.registers.vcov
                    if vcov is not None:
                        for r_i in range(vcov.shape[0]):
                            for c_i in range(vcov.shape[1]):
                                ws.cell(row=row_idx + r_i, column=col_idx + c_i, value=float(vcov[r_i, c_i]))
                continue

            # Strip enclosing parentheses if present
            if expr.startswith("(") and expr.endswith(")"):
                expr = expr[1:-1].strip()

            if (expr.startswith('"') and expr.endswith('"')) or (expr.startswith("'") and expr.endswith("'")):
                val = expr[1:-1]
            elif expr in self.registers.e_scalars:
                val = self.registers.e_scalars[expr]
            elif expr in self.registers.r_scalars:
                val = self.registers.r_scalars[expr]
            elif expr.startswith("e(") and expr.endswith(")") and expr[2:-1] in self.registers.e_scalars:
                val = self.registers.e_scalars[expr[2:-1]]
            elif expr.startswith("r(") and expr.endswith(")") and expr[2:-1] in self.registers.r_scalars:
                val = self.registers.r_scalars[expr[2:-1]]
            else:
                try:
                    val = float(expr)
                except ValueError:
                    val = expr
            ws[cell_ref] = val

        wb.save(self.putexcel_file)
        return CommandResult(f"written {len(assignments)} cells to {self.putexcel_file}", ok=True)

    def _cmd_tsfilter(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        if not self.timevar:
            return CommandResult("time variable not set; use tsset timevar", ok=False, return_code=459)

        # Syntax: tsfilter hp newvar = varname [, smooth(#) trend(trendvar)]
        from engine.timeseries import compute_hp_filter
        body = cmd.body.strip()
        if body.lower().startswith("hp"):
            body = body[2:].strip()

        if "=" not in body:
            return CommandResult("syntax: tsfilter hp newvar = varname [, smooth(#) trend(trendvar)]", ok=False, return_code=198)

        left, right = body.split("=", 1)
        newvar = left.strip().split()[-1]
        varname = right.strip().split()[0]

        df = self.dataset.df
        if varname not in df.columns:
            return CommandResult(f"variable '{varname}' not found", ok=False, return_code=111)

        smooth_val = float(cmd.get_option("smooth", 1600.0))
        trendvar = cmd.get_option("trend", None)

        y_series = df[varname].to_numpy(dtype=float)
        cycle, trend = compute_hp_filter(y_series, lamb=smooth_val)

        df[newvar] = cycle
        if trendvar:
            df[trendvar] = trend

        trend_msg = f", trend saved in '{trendvar}'" if trendvar else ""
        return CommandResult(f"(filter hp: cyclical component saved in '{newvar}'{trend_msg})", ok=True)

    def _cmd_var(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        if not self.timevar:
            return CommandResult("time variable not set; use tsset timevar", ok=False, return_code=459)

        tokens = cmd.body.split()
        if len(tokens) < 2:
            return CommandResult("syntax: var depvarlist [if] [in] [, lags(numlist)]", ok=False, return_code=198)

        depvars = tokens
        lags_opt = cmd.get_option("lags", "2")
        try:
            # If lags is e.g. "1 2" or "2"
            lag_tokens = [int(x) for x in lags_opt.split() if x.isdigit()]
            maxlags = max(lag_tokens) if lag_tokens else 2
        except Exception:
            maxlags = 2

        _, df_sub = self.filter_data(cmd)
        try:
            r = compute_var(df_sub, depvars, lags=maxlags)
            self.estimates.last_model = r
            self.registers.clear_e()
            self.registers.set_e_scalar('N', r['n'])
            self.registers.set_e_scalar('aic', r['aic'])
            self.registers.set_e_scalar('bic', r['bic'])
            self.registers.set_e_scalar('hqic', r['hqic'])
            self.registers.set_e_scalar('ll', r['ll'])
            self.registers.set_e_macro('cmd', 'var')
            self.registers.set_estimates(r['b_dict'], r['se_dict'], r.get('vcov'))
            return CommandResult(format_var(r), ok=True)
        except Exception as e:
            return CommandResult(f"var error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_stset(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: stset timevar [, failure(failvar)]", ok=False, return_code=198)

        timevar = tokens[0]
        df = self.dataset.df
        if timevar not in df.columns:
            return CommandResult(f"variable '{timevar}' not found", ok=False, return_code=111)

        failvar = cmd.get_option("failure", None) or cmd.get_option("fail", None)
        if failvar and failvar not in df.columns:
            return CommandResult(f"failure variable '{failvar}' not found", ok=False, return_code=111)

        # Generate standard standard survival variables
        df['_t0'] = 0.0
        df['_t'] = df[timevar].astype(float)
        if failvar:
            df['_d'] = (df[failvar] == 1).astype(int)
        else:
            df['_d'] = 1
        df['_st'] = 1

        self.survival_timevar = timevar
        self.survival_failvar = failvar

        n_obs = len(df)
        n_failures = int(df['_d'].sum())
        lines = [
            "-> stset " + timevar + (f", failure({failvar})" if failvar else ""),
            "",
            f"     failure event:  {failvar} != 0" if failvar else "     failure event:  (all observations fail)",
            f"obs. time interval:  (0, {timevar}]",
            " exit on or before:  failure",
            "-" * 60,
            f"    {n_obs:>10,}  total observations",
            f"    {n_failures:>10,}  failures in single-record/single-failure data",
            f"    {n_obs:>10,}  observations remaining, representing",
            f"    {n_failures:>10,}  failures",
            "-" * 60
        ]
        return CommandResult("\n".join(lines), ok=True)

    def _cmd_sts(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        if not self.survival_timevar:
            return CommandResult("data not stset; use stset timevar", ok=False, return_code=119)

        tokens = cmd.body.split()
        subcmd = tokens[0].lower() if tokens else "list"
        if subcmd not in ("list", "graph", "generate"):
            subcmd = "list"

        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        try:
            r = compute_sts_list(
                df_sub,
                timevar=self.survival_timevar,
                failvar=self.survival_failvar if self.survival_failvar else '_d'
            )
            return CommandResult(format_sts_list(r), ok=True)
        except Exception as e:
            return CommandResult(f"sts error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_stcox(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        if not self.survival_timevar:
            return CommandResult("data not stset; use stset timevar", ok=False, return_code=119)

        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: stcox varlist [, ties(breslow|efron)]", ok=False, return_code=198)

        indepvars = tokens
        ties_method = cmd.get_option("ties", "breslow")

        _, df_sub = self.filter_data(cmd)
        try:
            r = compute_stcox(
                df_sub,
                timevar=self.survival_timevar,
                failvar=self.survival_failvar if self.survival_failvar else '_d',
                indepvars=indepvars,
                method=ties_method
            )
            self.estimates.last_model = r
            self.registers.clear_e()
            self.registers.set_e_scalar('N', r['n'])
            self.registers.set_e_scalar('N_fail', r['n_events'])
            self.registers.set_e_scalar('ll', r['ll'])
            self.registers.set_e_macro('cmd', 'stcox')
            self.registers.set_estimates(r['b_dict'], r['se_dict'], r.get('vcov'))
            return CommandResult(format_stcox(r), ok=True)
        except Exception as e:
            return CommandResult(f"stcox error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_ologit(self, cmd):
        if self.dataset.is_empty:
            return CommandResult("no dataset in memory", ok=False, return_code=111)
        tokens = cmd.body.split()
        if not tokens or len(tokens) < 2:
            return CommandResult("syntax: ologit depvar indepvars [if] [in] [, or robust vce(cluster id)]", ok=False, return_code=198)

        depvar = tokens[0]
        indepvars = tokens[1:]

        _, df_sub = self.filter_data(cmd)
        if df_sub.empty:
            return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

        if depvar not in df_sub.columns:
            return CommandResult(f"variable '{depvar}' not found", ok=False, return_code=111)

        if not pd.api.types.is_numeric_dtype(df_sub[depvar]):
            return CommandResult(
                f"variable '{depvar}' is a string; regression requires numeric variables",
                ok=False,
                error="type mismatch",
                return_code=109
            )

        try:
            expanded_indeps, working_df, _ = expand_estimation_terms(indepvars, df_sub)
            vce_type, cluster_var = self._parse_vce_option(cmd, default="mle")
            odds_ratio = cmd.has_option("or")

            r = compute_ologit(
                working_df,
                depvar,
                expanded_indeps,
                odds_ratio=odds_ratio,
                vce=vce_type,
                cluster_var=cluster_var
            )
            r['raw_indepvars'] = indepvars
            self.estimates.last_model = r
            self.registers.clear_e()
            self.registers.set_e_scalar('N', float(r['n']))
            self.registers.set_e_scalar('ll', float(r['ll']))
            self.registers.set_e_scalar('ll_0', float(r['ll_0']))
            self.registers.set_e_scalar('r2_p', float(r['pseudo_r2']))
            self.registers.set_e_scalar('chi2', float(r['lr_chi2']))
            self.registers.set_e_scalar('p', float(r['prob_chi2']))
            self.registers.set_e_macro('cmd', 'ologit')
            self.registers.set_e_macro('depvar', r['depvar'])
            b_dict = {item['name']: item['coef'] for item in r['coefs']}
            se_dict = {item['name']: item['se'] for item in r['coefs']}
            self.registers.set_estimates(b_dict, se_dict, r.get('vcov'))
            return CommandResult(format_ologit(r), ok=True)
        except Exception as e:
            return CommandResult(f"ologit error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_brant(self, cmd=None):
        m = self.estimates.last_model
        if not m or m.get('model_type') != 'ologit':
            return CommandResult("estat brant can only be calculated after ologit", ok=False, return_code=301)
        try:
            r = compute_brant_test(m, self.dataset.df)
            self.registers.clear_r()
            self.registers.set_r_scalar('chi2', float(r['chi2']))
            self.registers.set_r_scalar('p', float(r['p']))
            self.registers.set_r_scalar('df', float(r['df']))
            is_detail = cmd.has_option("detail") or cmd.has_option("d") if cmd else False
            return CommandResult(format_brant_test(r, detail=is_detail), ok=True)
        except Exception as e:
            return CommandResult(f"estat brant error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_plugin(self, cmd):
        tokens = cmd.body.split()
        sub = tokens[0].lower() if tokens else "list"
        if sub == "list":
            plugins = self.plugins_mgr.list_plugins()
            lines = [
                ">>> " + self.plugins_mgr.SECURITY_WARNING + " <<<",
                "",
                f"{'Plugin Name':<20} {'Status':<12} {'Version':<10} {'Path'}",
                "-" * 75
            ]
            if not plugins:
                lines.append("(No user plugins found in " + self.plugins_mgr.plugins_dir + ")")
            else:
                for p in plugins:
                    lines.append(f"{p['name']:<20} {p['status']:<12} {p['version']:<10} {p['path']}")
            return CommandResult("\n".join(lines), ok=True)
        elif sub == "enable":
            if self.is_script_context and not self.plugins_mgr.get_allow_script_plugins():
                msg = (
                    "Security restriction: 'plugin enable' cannot be executed from a do-file or script.\n"
                    "Plugins execute arbitrary Python code. To enable plugins, type 'plugin enable <name>' "
                    "interactively in the console, or configure 'set allow_script_plugins on'."
                )
                return CommandResult(msg, ok=False, error=msg, return_code=198)
            if len(tokens) < 2:
                return CommandResult("syntax: plugin enable <name>", ok=False, return_code=198)
            name = tokens[1]
            try:
                self.plugins_mgr.enable_plugin(name)
                loaded = self.plugins_mgr.load_all_plugins()
                return CommandResult(f"Plugin '{name}' enabled. (Active plugins: {len(loaded)})", ok=True)
            except Exception as e:
                return CommandResult(f"plugin enable error: {e}", ok=False, error=str(e), return_code=198)
        elif sub == "disable":
            if len(tokens) < 2:
                return CommandResult("syntax: plugin disable <name>", ok=False, return_code=198)
            name = tokens[1]
            try:
                self.plugins_mgr.disable_plugin(name)
                return CommandResult(f"Plugin '{name}' disabled. Restart application or reload plugins.", ok=True)
            except Exception as e:
                return CommandResult(f"plugin disable error: {e}", ok=False, error=str(e), return_code=198)
        elif sub == "new":
            if self.is_script_context and not self.plugins_mgr.get_allow_script_plugins():
                msg = (
                    "Security restriction: 'plugin new' cannot be executed from a do-file or script.\n"
                    "Authoring plugins must be performed interactively in the console or GUI."
                )
                return CommandResult(msg, ok=False, error=msg, return_code=198)
            if len(tokens) < 2:
                return CommandResult("syntax: plugin new <name>", ok=False, return_code=198)
            name = tokens[1]
            try:
                path = self.plugins_mgr.create_template(name)
                return CommandResult(f"Plugin template '{name}' created at {path}.\nEdit the file and run 'plugin enable {name}' to activate.", ok=True)
            except Exception as e:
                return CommandResult(f"plugin new error: {e}", ok=False, error=str(e), return_code=198)
        else:
            return CommandResult(f"unknown plugin subcommand '{sub}'. Usage: plugin list | enable <name> | disable <name> | new <name>", ok=False, return_code=198)

    def _cmd_pkg(self, cmd):
        tokens = cmd.body.split()
        sub = tokens[0].lower() if tokens else "list"
        if sub in ("list",):
            return self._cmd_plugin(cmd)
        elif sub in ("search", "find"):
            if len(tokens) < 2:
                return CommandResult("syntax: pkg search <keyword>", ok=False, return_code=198)
            kw = tokens[1]
            results = self.plugins_mgr.search_packages(kw)
            if not results:
                return CommandResult(f"No packages found matching '{kw}'.", ok=True)
            lines = [
                f"Found {len(results)} package(s) matching '{kw}':",
                "-" * 70
            ]
            for r in results:
                lines.append(f"{r['name']:<18} v{r.get('version', '1.0'):<8} (Author: {r.get('author', 'Community')})")
                lines.append(f"  {r.get('description', '')}")
                lines.append(f"  -> To install: pkg install {r['name']}")
                lines.append("-" * 70)
            return CommandResult("\n".join(lines), ok=True)
        elif sub in ("enable",):
            if self.is_script_context and not self.plugins_mgr.get_allow_script_plugins():
                msg = (
                    "Security restriction: 'pkg enable' cannot be executed from a do-file or script.\n"
                    "Plugins execute arbitrary Python code with full user privileges.\n"
                    "To activate this package, type 'pkg enable <name>' interactively in the console command bar,\n"
                    "or configure 'set allow_script_plugins on' (WARNING: allows shared scripts to run arbitrary code)."
                )
                return CommandResult(msg, ok=False, error=msg, return_code=198)
            if len(tokens) < 2:
                return CommandResult("syntax: pkg enable <name>", ok=False, return_code=198)
            name = tokens[1]
            try:
                self.plugins_mgr.enable_plugin(name)
                loaded = self.plugins_mgr.load_all_plugins()
                return CommandResult(f"Package '{name}' enabled and loaded. (Active plugins: {len(loaded)})", ok=True)
            except Exception as e:
                return CommandResult(f"pkg enable error: {e}", ok=False, error=str(e), return_code=198)
        elif sub in ("disable",):
            if len(tokens) < 2:
                return CommandResult("syntax: pkg disable <name>", ok=False, return_code=198)
            name = tokens[1]
            try:
                self.plugins_mgr.disable_plugin(name)
                return CommandResult(f"Package '{name}' disabled.", ok=True)
            except Exception as e:
                return CommandResult(f"pkg disable error: {e}", ok=False, error=str(e), return_code=198)
        elif sub in ("install", "add"):
            if self.is_script_context and not self.plugins_mgr.get_allow_script_plugins():
                msg = (
                    f"Security restriction: 'pkg {sub}' cannot be executed from a do-file or script.\n"
                    f"Plugins execute arbitrary Python code. To install packages:\n"
                    f"  1. Type 'pkg {sub} <name>' interactively into the console command bar, OR\n"
                    f"  2. Confirm installation in the Package Hub dialog (GUI), OR\n"
                    f"  3. Configure 'set allow_script_plugins on' (WARNING: allows shared scripts to run arbitrary code)."
                )
                return CommandResult(msg, ok=False, error=msg, return_code=198)
            if len(tokens) < 2:
                return CommandResult("syntax: pkg install <name|filepath|url> [, sha256(<hash>)]", ok=False, return_code=198)
            source = tokens[1].strip('"\'')
            expected_sha = cmd.get_option("sha256", None)
            try:
                r = self.plugins_mgr.install_plugin(source, expected_sha256=expected_sha, enable=False)
                msg = (
                    f"Package '{r['name']}' successfully installed (disabled).\n"
                    f"  Location: {r['path']}\n"
                    f"  Size:     {r['size']} bytes\n"
                    f"  SHA-256:  {r['sha256']}\n"
                    f"SECURITY NOTE: Downloaded code does NOT execute automatically.\n"
                    f"To activate this package and register its command, type:\n"
                    f"  pkg enable {r['name']}"
                )
                return CommandResult(msg, ok=True)
            except Exception as e:
                return CommandResult(f"pkg install error: {e}", ok=False, error=str(e), return_code=198)
        elif sub in ("uninstall", "remove", "delete"):
            if len(tokens) < 2:
                return CommandResult("syntax: pkg uninstall <name>", ok=False, return_code=198)
            name = tokens[1]
            try:
                self.plugins_mgr.uninstall_plugin(name)
                return CommandResult(f"Package '{name}' successfully uninstalled.", ok=True)
            except Exception as e:
                return CommandResult(f"pkg uninstall error: {e}", ok=False, error=str(e), return_code=198)
        else:
            return CommandResult(f"unknown pkg subcommand '{sub}'. Usage: pkg search | install | enable | disable | list | uninstall", ok=False, return_code=198)

    def _cmd_ssc(self, cmd):
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: ssc install <packagename>", ok=False, return_code=198)
        sub = tokens[0].lower()
        if sub == "install":
            if len(tokens) < 2:
                return CommandResult("syntax: ssc install <packagename>", ok=False, return_code=198)
            pkg_name = tokens[1]
            cmd.body = f"install {pkg_name}"
            return self._cmd_pkg(cmd)
        elif sub == "describe":
            if len(tokens) < 2:
                return CommandResult("syntax: ssc describe <packagename>", ok=False, return_code=198)
            cmd.body = f"search {tokens[1]}"
            return self._cmd_pkg(cmd)
        return CommandResult("syntax: ssc install <packagename>", ok=False, return_code=198)

    def _cmd_net(self, cmd):
        if self.is_script_context and not self.plugins_mgr.get_allow_script_plugins():
            msg = (
                "Security restriction: 'net install' cannot be executed from a do-file or script.\n"
                "To install packages, type 'net install <name>' interactively in the console,\n"
                "or configure 'set allow_script_plugins on'."
            )
            return CommandResult(msg, ok=False, error=msg, return_code=198)
        tokens = cmd.body.split()
        if not tokens or tokens[0].lower() != "install":
            return CommandResult("syntax: net install <packagename> [, from(source) sha256(<hash>)]", ok=False, return_code=198)
        if len(tokens) < 2:
            return CommandResult("syntax: net install <packagename> [, from(source) sha256(<hash>)]", ok=False, return_code=198)
        pkg_name = tokens[1].strip('"\'')
        from_source = (cmd.get_option("from", None) or pkg_name).strip('"\'')
        expected_sha = cmd.get_option("sha256", None)
        try:
            r = self.plugins_mgr.install_plugin(from_source, target_name=pkg_name, enable=False, expected_sha256=expected_sha)
            msg = (
                f"Package '{r['name']}' successfully installed (disabled).\n"
                f"  Location: {r['path']}\n"
                f"  Size:     {r['size']} bytes\n"
                f"  SHA-256:  {r['sha256']}\n"
                f"To activate and run this package, execute: pkg enable {r['name']}"
            )
            return CommandResult(msg, ok=True)
        except Exception as e:
            return CommandResult(f"net install error: {e}", ok=False, error=str(e), return_code=198)

    def _cmd_set(self, cmd):
        tokens = cmd.body.split()
        if not tokens:
            return CommandResult("syntax: set <setting> <value>", ok=False, return_code=198)
        key = tokens[0].lower()
        val = tokens[1].lower() if len(tokens) > 1 else ""

        if key == "obs":
            try:
                new_n = int(val)
                if new_n < 0:
                    return CommandResult("obs must be a positive integer", ok=False, return_code=198)
                curr_n = len(self.dataset.df) if not self.dataset.is_empty else 0
                if self.dataset.is_empty:
                    self.dataset.load_dataframe(pd.DataFrame(index=range(new_n)))
                elif new_n > curr_n:
                    frame = self.dataset.df.reset_index(drop=True)
                    for col, dtype in frame.dtypes.items():
                        if isinstance(dtype, np.dtype) and dtype.kind in "iub":
                            # Use nullable integers/booleans before inserting missing rows.
                            frame[col] = frame[col].convert_dtypes()
                    self.dataset.df = frame.reindex(pd.RangeIndex(new_n))
                elif new_n < curr_n:
                    self.dataset.df = self.dataset.df.iloc[:new_n].copy()
                return CommandResult(f"number of observations (_N) was {curr_n}, now {new_n}", ok=True)
            except ValueError:
                return CommandResult("syntax: set obs #", ok=False, return_code=198)

        if key in ("allow_script_plugins", "allow_script_plugin"):
            if self.is_script_context:
                msg = (
                    "Security restriction: 'set allow_script_plugins on' cannot be executed from a do-file or script.\n"
                    "Settings that weaken script execution safety may only be changed from the interactive console or the Qt settings dialog."
                )
                return CommandResult(msg, ok=False, error=msg, return_code=198)

            if val in ("on", "1", "true"):
                self.plugins_mgr.set_allow_script_plugins(True)
                msg = (
                    "allow_script_plugins set to ON.\n"
                    "SECURITY WARNING: Shared or untrusted do-files can now install and execute "
                    "arbitrary Python plugins with full user privileges. Only enable if you trust all scripts run in this session."
                )
                return CommandResult(msg, ok=True)
            elif val in ("off", "0", "false"):
                self.plugins_mgr.set_allow_script_plugins(False)
                return CommandResult("allow_script_plugins set to OFF.", ok=True)
            else:
                return CommandResult("syntax: set allow_script_plugins on | off", ok=False, return_code=198)
        return CommandResult(f"setting '{key}' configured to '{val}'", ok=True)

    def _cmd_cite(self, cmd):
        target = cmd.body.strip().lower() if cmd and cmd.body else ""
        if not target:
            m = self.estimates.last_model
            if m:
                target = m.get('model_type') or m.get('cmd') or ""
            else:
                avail = ", ".join(sorted(CITATIONS_DB.keys()))
                msg = f"Available citation keys: {avail}\nUsage: cite <command> (e.g., cite csdid, cite synth, cite ivregress)"
                return CommandResult(msg, ok=True)

        entries = lookup_citations(target)
        if not entries:
            avail = ", ".join(sorted(CITATIONS_DB.keys()))
            msg = f"No specific citation found for '{target}'.\nAvailable citation keys: {avail}"
            return CommandResult(msg, ok=True)

        self.registers.clear_r()
        self.registers.set_r_macro('citation', f"{entries[0]['author']} ({entries[0]['year']})")
        self.registers.set_r_macro('doi', entries[0].get('doi', ''))
        self.registers.set_r_macro('bibtex', entries[0].get('bibtex', ''))

        output_str = format_citation_response(target, entries)
        return CommandResult(output_str, ok=True)


    def _cmd_help(self, cmd):
        doc = (
            "========================================================================================\n"
            "                       Potatostics Command Syntax & Reference\n"
            "========================================================================================\n"
            "General Syntax: command [varlist] [if exp] [in range] [using file] [, options]\n\n"
            "Data Management:\n"
            "  use <file> [, clear]             Load .dta or .csv dataset\n"
            "  save <file> [, replace]          Save dataset to CSV\n"
            "  describe                         Display variables, types, labels\n"
            "  summarize [vars] [if] [in]       Summary statistics (mean, std, min, max)\n"
            "  tabulate var1 [var2] [if] [in]   One-way or two-way cross-tabulation\n"
            "  correlate [vars] [if] [in]       Correlation matrix\n"
            "  generate newvar = exp [if] [in]  Create variable with functions & math\n"
            "  replace var = exp [if] [in]      Modify variable values conditionally\n"
            "  drop varlist | drop if | drop in Drop variables or observations\n"
            "  keep varlist | keep if | keep in Keep variables or observations\n"
            "  count [if] [in]                  Count matching observations\n"
            "  list [vars] [if] [in]            Display dataset observations\n"
            "  by/bysort vars [(sort)]: cmd     Execute command by groups with _n and _N\n"
            "  merge 1:1|1:m|m:1 vars using f   Relational dataset merge with _merge table\n"
            "  append using <file> [, gen()]    Append dataset from disk\n"
            "  reshape long|wide stub, i() j()  Pivot between long and wide formats\n"
            "  collapse (stat) vars, by()       Aggregate dataset (mean, sd, p50, min, max...)\n"
            "  egen newvar = func(args)         Extended generate (mean, total, rank, cut...)\n"
            "  encode stringvar, gen() [label]  Convert string categories to labeled integers\n"
            "  decode numvar, gen()             Convert labeled integers back to strings\n"
            "  label define|val|var|dir|drop    Manage variable labels and value labels\n"
            "  recode var (rule) [, gen()]      Recode values and ranges\n"
            "  duplicates report|tag|drop,force Report, flag, or remove duplicate rows\n\n"
            "Estimation & Models:\n"
            "  regress dep indeps [if] [in] [, robust vce(cluster id|hc2|hc3)]\n"
            "  reghdfe dep indeps [if] [in], absorb(fe1 fe2...) [vce(cluster id)]\n"
            "  logit dep indeps [if] [in] [, or robust vce(cluster id)]\n"
            "  probit dep indeps [if] [in] [, robust vce(cluster id)]\n"
            "  poisson dep indeps [if] [in] [, irr robust vce(cluster id)]\n"
            "  tobit dep indeps [if] [in], ll(#) [ul(#)]\n"
            "  qreg dep indeps [if] [in] [, quantile(#)]\n"
            "  ivregress 2sls dep (endog = inst) exog [if] [in] [, small robust vce(cluster id)]\n"
            "  xtreg dep indeps [if] [in], fe [vce(cluster id)]\n"
            "  margins [, dydx(*)]              Average marginal effects with delta method SE\n"
            "  lincom exp                       Linear combination of estimators (e.g. b1 - 2*b2)\n"
            "  test var1 = var2 | test var1 = 0 Post-estimation Wald restriction test\n"
            "  predict newvar [if] [in] [, xb resid pr]\n\n"
            "Scripting & Return Registers:\n"
            "  local / global name [=] exp      Define and expand local (`x') and global ($x) macros\n"
            "  foreach var in|of ... { ... }    Loop over items or dataset varlists\n"
            "  forvalues var = 1/N { ... }      Loop over numeric ranges\n"
            "  quietly / capture / noisily cmd  Script execution prefixes\n"
            "  do script.do / run script.do     Execute batch do-file scripts\n"
            "  return list / ereturn list       Display saved scalar/matrix registers\n\n"
            "Factor Variables & Time Series:\n"
            "  i.var, ib#.var, c.var, i.a#i.b, c.x##i.d in regression models\n"
            "  tsset timevar                    Set single time-series\n"
            "  xtset panelvar [timevar]         Set panel dataset\n"
            "  L.var, L2.var, D.var, F.var      Lag, lead, and difference operators\n\n"
            "Causal Inference & Modern Methods:\n"
            "  csdid depvar, ivar() time() gvar() [seed(#) reps(#)] Callaway-Sant'Anna DiD\n"
            "  synth depvar indeps, trunit() trperiod() [unit() time()] Synthetic control\n"
            "  eventstudy depvar, unit() time() treattime() [pre() post()] Dynamic event study\n"
            "  did outcome treated post         Classic 2x2 Difference-in-Differences\n"
            "  psmatch treated controls, outcome(y) Propensity Score Matching (ATT)\n\n"
            "Graphics & Econometric Exhibits:\n"
            "  scatter y x [if] [in] [, save(f)]   Scatter plot with OLS trendline\n"
            "  histogram var [if] [in] [, bins(N)] Distribution histogram\n"
            "  boxplot var [if] [in] [, over(grp)] Distribution box plot\n"
            "  binscatter y x [if] [in] [, by(grp) nbins(N) controls(vars) line(none|linear) save(f)]\n"
            "  rdplot y x [if] [in] [, c(cutoff) bw(h) nbins(N) save(f)] Sharp RD local linear exhibit\n"
            "  irf y shock [if] [in] [, horizon(H) save(f)] Local projections impulse response (Jordà)\n"
            "  synth_plot [, save(f)]              Synthetic control trajectory & gap exhibits\n"
            "  coefplot [, save(f)]                Model coefficients and confidence intervals\n\n"
            "Diagnostics & Comparison:\n"
            "  estat vif | hettest | ovtest | dwatson\n"
            "  estimates store <name>           Save model in memory\n"
            "  esttab                           Compare stored models side-by-side\n"
            "  putdocx <file.docx>              Generate native Word report\n"
            "  cite [cmd]                       Display peer-reviewed academic citation, DOI & BibTeX\n"
            "  ai \"<plain english query>\"       AI Econometric Copilot\n"
            "========================================================================================"
        )
        return CommandResult(doc, ok=True)
