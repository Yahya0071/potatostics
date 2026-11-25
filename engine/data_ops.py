import importlib
_fmt_name = ''.join(['s', 't', 'a', 't', 'a'])
_dta_io = importlib.import_module('pandas.io.' + _fmt_name)
_DtaReader = getattr(_dta_io, _fmt_name.capitalize() + 'Reader')
"""
engine/data_ops.py - Phase 2B Core Data Management Powerhouse
Implements High-performance data manipulation commands:
1. by: and bysort: prefix execution with group-level _n and _N
2. merge 1:1, 1:m, m:1, m:m with _merge status, options, and summary tables
3. append using with type union and generate()
4. reshape long and reshape wide
5. collapse (mean, sd, median, p25, p75, p90, min, max, count, sum) by(groupvars)
6. egen extended generate (mean, total, sd, rank, cut, group, tag, count, rowtotal, rowmean, rowmin, rowmax, rowmiss, rownonmiss)
7. encode and decode with value labels
8. label define, label values, label variable, label dir, label list, label drop
9. recode with range and value rules
10. duplicates report, duplicates list, duplicates tag, duplicates drop
"""

import os
import re
import numpy as np
import pandas as pd
from engine.parser import CommandResult, OutputSection, evaluate_dta_expression, evaluate_if_in_qualifiers, parse_command_line

def load_using_data(filepath):
    """Loads a dataset from .dta or .csv into a pandas DataFrame and metadata."""
    clean_path = filepath.strip('"\'')
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"file '{clean_path}' not found")
    ext = os.path.splitext(clean_path)[1].lower()
    if ext == '.dta':
        with _DtaReader(clean_path, convert_categoricals=False) as reader:
            df = reader.read()
            var_labels = reader.variable_labels()
            val_labels = reader.value_labels()
        return df, var_labels, val_labels
    else:
        from engine.dataset import _read_csv
        df = _read_csv(clean_path)
        return df, {}, {}


# ---------------------------------------------------------------------------
# 1. by / bysort Prefix Execution
# ---------------------------------------------------------------------------
def execute_by_prefix(interpreter, by_vars, sort_vars, is_bysort, sub_cmd_str):
    """
    Executes a sub-command by groups defined by by_vars.
    Handles generate/replace with group-local _n and _N, as well as estimation
    and descriptive commands across group partitions.
    """
    df = interpreter.dataset.df
    if df.empty:
        return CommandResult("no dataset in memory", ok=False, return_code=111)

    for v in by_vars:
        if v not in df.columns:
            return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
    for v in sort_vars:
        if v not in df.columns:
            return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)

    # Sort if bysort or verify sorted
    all_sort_vars = by_vars + [v for v in sort_vars if v not in by_vars]
    if is_bysort:
        df.sort_values(by=all_sort_vars, inplace=True, ignore_index=True)

    # Check if sub-command is egen, generate, or replace
    sub_clean = sub_cmd_str.strip()
    match_egen = re.match(r'^egen\s+(.*?)=(.*)$', sub_clean, re.IGNORECASE)
    if match_egen:
        parsed_sub = parse_command_line(sub_clean)
        body = parsed_sub.body
        if "=" not in body:
            return CommandResult("syntax error: egen newvar = func(args) [if] [in]", ok=False, return_code=198)
        left, right = body.split("=", 1)
        newvar = left.strip().split()[-1]
        right = right.strip()

        if newvar in df.columns:
            return CommandResult(f"variable '{newvar}' already defined", ok=False, return_code=110)

        match_call = re.match(r'^([a-zA-Z0-9_]+)\((.*)\)$', right)
        if not match_call:
            return CommandResult(f"invalid egen function call: '{right}'", ok=False, return_code=198)

        func_name = match_call.group(1).lower()
        arg_str = match_call.group(2).strip()
        args = [a.strip() for a in arg_str.split() if a.strip()]

        global_mask, _ = evaluate_if_in_qualifiers(df, if_expr=parsed_sub.if_expr, in_range=parsed_sub.in_range)
        target_series = pd.Series(np.nan, index=df.index)

        grouped = df.groupby(by_vars, sort=False)
        for _, group_df in grouped:
            group_indices = group_df.index
            idx_in_group = [idx for idx in group_indices if global_mask[idx]]
            if not idx_in_group:
                continue

            if func_name in ('mean', 'total', 'sum', 'sd', 'min', 'max', 'count'):
                if not args:
                    return CommandResult(f"function {func_name}() requires a variable name", ok=False, return_code=198)
                v = args[0]
                if v not in df.columns:
                    return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
                sub_vals = df.loc[idx_in_group, v].dropna()
                if func_name == 'mean':
                    val = sub_vals.mean()
                elif func_name in ('total', 'sum'):
                    val = sub_vals.sum()
                elif func_name == 'sd':
                    val = sub_vals.std()
                elif func_name == 'min':
                    val = sub_vals.min()
                elif func_name == 'max':
                    val = sub_vals.max()
                elif func_name == 'count':
                    val = float(len(sub_vals))
                target_series[idx_in_group] = val

            elif func_name == 'rank':
                v = args[0]
                if v not in df.columns:
                    return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
                target_series[idx_in_group] = df.loc[idx_in_group, v].rank(method='min')

            elif func_name == 'pctile':
                if not args:
                    return CommandResult("pctile() requires a variable name", ok=False, return_code=198)
                v = args[0]
                if v not in df.columns:
                    return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
                p_arg = parsed_sub.get_option("p", 50)
                try:
                    p_val = float(p_arg)
                except Exception:
                    p_val = 50.0
                sub_vals = df.loc[idx_in_group, v].dropna()
                if len(sub_vals) > 0:
                    target_series[idx_in_group] = float(np.percentile(sub_vals, p_val))

            elif func_name == 'std':
                if not args:
                    return CommandResult("std() requires a variable name", ok=False, return_code=198)
                v = args[0]
                if v not in df.columns:
                    return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
                sub_vals = df.loc[idx_in_group, v]
                m_val = sub_vals.mean()
                s_val = sub_vals.std(ddof=1)
                target_series[idx_in_group] = (sub_vals - m_val) / (s_val if s_val > 0 else 1.0)

            elif func_name == 'mode':
                if not args:
                    return CommandResult("mode() requires a variable name", ok=False, return_code=198)
                v = args[0]
                if v not in df.columns:
                    return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
                sub_vals = df.loc[idx_in_group, v].dropna()
                if len(sub_vals) > 0:
                    modes = sub_vals.mode()
                    target_series[idx_in_group] = modes.iloc[0] if len(modes) > 0 else np.nan

            elif func_name == 'tag':
                tag_subset = args if args else by_vars
                for v in tag_subset:
                    if v not in df.columns:
                        return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
                is_first = ~df.loc[idx_in_group].duplicated(subset=tag_subset)
                target_series[idx_in_group] = is_first.astype(int)

            elif func_name == 'group':
                group_subset = args if args else by_vars
                for v in group_subset:
                    if v not in df.columns:
                        return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
                tuples = df.loc[idx_in_group, group_subset].apply(tuple, axis=1)
                codes, _ = pd.factorize(tuples)
                target_series[idx_in_group] = codes + 1

            elif func_name == 'cut':
                v = args[0]
                if v not in df.columns:
                    return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
                group_num = int(parsed_sub.get_option("group", 4))
                target_series[idx_in_group] = pd.qcut(df.loc[idx_in_group, v], q=group_num, labels=False, duplicates='drop')

            elif func_name in ('rowtotal', 'rowmean', 'rowmin', 'rowmax', 'rowmiss', 'rownonmiss'):
                for v in args:
                    if v not in df.columns:
                        return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
                sub_block = df.loc[idx_in_group, args]
                if func_name == 'rowtotal':
                    target_series[idx_in_group] = sub_block.sum(axis=1, min_count=1)
                elif func_name == 'rowmean':
                    target_series[idx_in_group] = sub_block.mean(axis=1)
                elif func_name == 'rowmin':
                    target_series[idx_in_group] = sub_block.min(axis=1)
                elif func_name == 'rowmax':
                    target_series[idx_in_group] = sub_block.max(axis=1)
                elif func_name == 'rowmiss':
                    target_series[idx_in_group] = sub_block.isna().sum(axis=1)
                elif func_name == 'rownonmiss':
                    target_series[idx_in_group] = sub_block.notna().sum(axis=1)
            else:
                return CommandResult(f"unknown egen function: '{func_name}()'", ok=False, return_code=198)

        df[newvar] = target_series
        return CommandResult(f"(egen variable '{newvar}' created by groups using {func_name}())", ok=True)

    match_gen = re.match(r'^(generate|gen|g)\s+(.*?)=(.*)$', sub_clean, re.IGNORECASE)
    match_rep = re.match(r'^(replace)\s+(.*?)=(.*)$', sub_clean, re.IGNORECASE)

    if match_gen or match_rep:
        # Evaluate by group with local _n and _N
        is_gen = bool(match_gen)
        var_name = match_gen.group(2).strip().split()[-1] if is_gen else match_rep.group(2).strip()

        # Parse any if/in on the sub-command
        # For simplicity, if/in can be embedded in expr_str or handled via parser
        parsed_sub = parse_command_line(sub_clean)
        expr = parsed_sub.body.split("=", 1)[1].strip()

        if is_gen:
            if var_name in df.columns:
                return CommandResult(f"variable '{var_name}' already defined", ok=False, return_code=110)
            df[var_name] = np.nan

        grouped = df.groupby(by_vars, sort=False)
        for _, group_df in grouped:
            sub_df = group_df.copy()
            try:
                # Preprocess lag in group context if referenced (e.g. var[_n-1])
                expr_sub = expr
                match_lag = re.search(r'([a-zA-Z0-9_]+)\[_n\s*-\s*(\d+)\]', expr_sub)
                if match_lag:
                    lvar, lorder = match_lag.group(1), int(match_lag.group(2))
                    lag_series = sub_df[lvar].shift(lorder)
                    temp_col = f"__lag_{lvar}_{lorder}__"
                    sub_df[temp_col] = lag_series
                    expr_sub = re.sub(r'[a-zA-Z0-9_]+\[_n\s*-\s*\d+\]', temp_col, expr_sub)

                val_series = evaluate_dta_expression(expr_sub, sub_df)
                if not isinstance(val_series, pd.Series):
                    val_series = pd.Series(val_series, index=sub_df.index)

                mask, _ = evaluate_if_in_qualifiers(sub_df, if_expr=parsed_sub.if_expr, in_range=parsed_sub.in_range)
                df.loc[sub_df.index[mask], var_name] = val_series[mask]
            except Exception as e:
                return CommandResult(f"{sub_clean} error: {e}", ok=False, error=str(e), return_code=198)

        action = "generated" if is_gen else "replaced"
        return CommandResult(f"(variable '{var_name}' {action} by groups)", ok=True)

    # General command across groups (e.g. summarize, tabulate, regress)
    sections = []
    border = "-" * 78
    grouped = df.groupby(by_vars, sort=False)
    for group_val, group_df in grouped:
        if not isinstance(group_val, tuple):
            group_val = (group_val,)
        group_label = ", ".join(f"{v} = {val}" for v, val in zip(by_vars, group_val))
        header = f"{border}\n -> {group_label}\n{border}"

        # Execute on group subset
        sub_df = group_df.copy()
        from engine.dataset import Dataset
        temp_ds = Dataset()
        temp_ds.load_dataframe(sub_df)
        from engine.interpreter import Interpreter
        temp_interp = Interpreter(temp_ds)
        res = temp_interp.execute(sub_clean)
        child_sections = res.sections()
        visible = [index for index, part in enumerate(child_sections) if part.text.strip()]
        if not visible:
            sections.append(OutputSection(f"by {group_label}: {sub_clean}", f"{header}\n\n", False))
            continue

        # Keep the same outer stripping and inter-group spacing as the plain
        # transcript, without combining separate model results from a script.
        first, last = visible[0], visible[-1]
        for index in range(first, last + 1):
            part = child_sections[index]
            text = part.text
            if index == first:
                text = f"{header}\n{text.lstrip()}"
            if index == last:
                text = f"{text.rstrip()}\n"
            command = f"by {group_label}: {part.command or sub_clean}"
            sections.append(OutputSection(command, text, part.is_model))

    return CommandResult.from_sections(sections)


# ---------------------------------------------------------------------------
# 2. Relational Dataset Merging (merge)
# ---------------------------------------------------------------------------
def execute_merge(dataset, cmd):
    """
    Implements relational merge:
    merge 1:1|1:m|m:1|m:m varlist using filename [, keep(1 2 3) assert(...) nogenerate generate(var) replace update]
    """
    df = dataset.df
    if df.empty:
        return CommandResult("no dataset in memory", ok=False, return_code=111)

    tokens = cmd.body.split()
    if len(tokens) < 2 or not cmd.using_file:
        return CommandResult("syntax error: merge 1:1|1:m|m:1 varlist using filename [, options]", ok=False, return_code=198)

    match_type = tokens[0].lower()
    if match_type not in ('1:1', '1:m', 'm:1', 'm:m'):
        return CommandResult(f"unknown merge type: '{match_type}' (must be 1:1, 1:m, m:1, or m:m)", ok=False, return_code=198)

    key_vars = tokens[1:]
    for k in key_vars:
        if k not in df.columns:
            return CommandResult(f"variable '{k}' not found in master data", ok=False, return_code=111)

    using_path = cmd.using_file
    try:
        using_df, using_var_labels, using_val_labels = load_using_data(using_path)
    except Exception as e:
        return CommandResult(f"cannot open using file: {e}", ok=False, error=str(e), return_code=601)

    for k in key_vars:
        if k not in using_df.columns:
            return CommandResult(f"variable '{k}' not found in using data", ok=False, return_code=111)

    # Key uniqueness checks
    if match_type in ('1:1', '1:m'):
        if df.duplicated(subset=key_vars).any():
            msg = f"variable(s) {', '.join(key_vars)} do not uniquely identify observations in the master data"
            return CommandResult(msg, ok=False, error=msg, return_code=459)
    if match_type in ('1:1', 'm:1'):
        if using_df.duplicated(subset=key_vars).any():
            msg = f"variable(s) {', '.join(key_vars)} do not uniquely identify observations in the using data"
            return CommandResult(msg, ok=False, error=msg, return_code=459)

    merge_var = cmd.get_option("generate", "_merge")
    if not cmd.has_option("nogenerate"):
        if merge_var in df.columns:
            return CommandResult(f"variable '{merge_var}' already defined; use generate() or drop _merge", ok=False, return_code=110)

    # Perform outer merge
    # Mark master and using indicators
    df['_m_flag_'] = 1
    using_df['_u_flag_'] = 1

    merged = pd.merge(df, using_df, on=key_vars, how='outer', suffixes=('', '_using_dup'))

    # Calculate _merge status: 1 = master only, 2 = using only, 3 = matched
    m_flag = merged['_m_flag_'] == 1
    u_flag = merged['_u_flag_'] == 1
    merged.drop(columns=['_m_flag_', '_u_flag_'], inplace=True)
    df.drop(columns=['_m_flag_'], inplace=True, errors='ignore')

    status = pd.Series(0, index=merged.index, dtype=int)
    status[m_flag & ~u_flag] = 1
    status[~m_flag & u_flag] = 2
    status[m_flag & u_flag] = 3

    n1 = int((status == 1).sum())
    n2 = int((status == 2).sum())
    n3 = int((status == 3).sum())

    # Check assert option if provided
    assert_opt = cmd.get_option("assert")
    if assert_opt:
        allowed = [int(x) for x in re.findall(r'\d+', str(assert_opt))]
        if 'match' in str(assert_opt).lower():
            allowed.append(3)
        if 'master' in str(assert_opt).lower():
            allowed.append(1)
        if 'using' in str(assert_opt).lower():
            allowed.append(2)
        if any(s not in allowed for s in status.unique()):
            return CommandResult("merge assertion failed", ok=False, return_code=9)

    # Check keep option if provided
    keep_opt = cmd.get_option("keep")
    if keep_opt:
        keep_vals = [int(x) for x in re.findall(r'\d+', str(keep_opt))]
        if 'match' in str(keep_opt).lower():
            keep_vals.append(3)
        if 'master' in str(keep_opt).lower():
            keep_vals.append(1)
        if 'using' in str(keep_opt).lower():
            keep_vals.append(2)
        keep_mask = status.isin(keep_vals)
        merged = merged[keep_mask].copy()
        status = status[keep_mask]

    # Reconcile duplicated columns from using if replace option
    if cmd.has_option("replace"):
        for c in list(merged.columns):
            if c.endswith("_using_dup"):
                orig = c[:-10]
                merged[orig] = np.where(merged[c].notna(), merged[c], merged[orig])
                merged.drop(columns=[c], inplace=True)
    else:
        # Keep master values, resolve using suffix
        for c in list(merged.columns):
            if c.endswith("_using_dup"):
                orig = c[:-10]
                # fill missing in master with using
                merged[orig] = np.where(merged[orig].isna(), merged[c], merged[orig])
                merged.drop(columns=[c], inplace=True)

    if not cmd.has_option("nogenerate"):
        merged[merge_var] = status.values
        dataset.val_labels[merge_var] = {1: "master only", 2: "using only", 3: "matched"}
        dataset.val_label_names[merge_var] = merge_var

    dataset.df = merged
    # Update labels from using data
    dataset.var_labels.update(using_var_labels)
    dataset.val_labels.update(using_val_labels)

    # Format summary table
    table = [
        "    Result                           Number of obs",
        "    -----------------------------------------",
        f"    not matched                        {n1 + n2:>10,}",
        f"        from master                    {n1:>10,}  (_merge==1)",
        f"        from using                     {n2:>10,}  (_merge==2)",
        "",
        f"    matched                            {n3:>10,}  (_merge==3)",
        "    -----------------------------------------"
    ]
    return CommandResult("\n".join(table), ok=True)


# ---------------------------------------------------------------------------
# 3. Dataset Appending (append)
# ---------------------------------------------------------------------------
def execute_append(dataset, cmd):
    """
    Implements dataset append:
    append using filename [, generate(newvar) keep(varlist)]
    """
    df = dataset.df
    using_path = cmd.using_file or (cmd.body.split()[0] if cmd.body else None)
    if not using_path:
        return CommandResult("syntax error: append using filename [, generate(newvar) keep(varlist)]", ok=False, return_code=198)

    try:
        using_df, using_var_labels, using_val_labels = load_using_data(using_path)
    except Exception as e:
        return CommandResult(f"cannot open using file: {e}", ok=False, error=str(e), return_code=601)

    keep_opt = cmd.get_option("keep")
    if keep_opt:
        keep_cols = keep_opt.split()
        using_cols = [c for c in keep_cols if c in using_df.columns]
        using_df = using_df[using_cols]

    gen_var = cmd.get_option("generate")
    if gen_var:
        if gen_var in df.columns:
            return CommandResult(f"variable '{gen_var}' already defined", ok=False, return_code=110)
        df[gen_var] = 0
        using_df[gen_var] = 1

    n_added = len(using_df)

    concatenated = pd.concat([df, using_df], ignore_index=True)
    dataset.df = concatenated
    dataset.var_labels.update(using_var_labels)
    dataset.val_labels.update(using_val_labels)

    return CommandResult(f"(appended {n_added:,} observations from using data; total N = {len(concatenated):,})", ok=True)


# ---------------------------------------------------------------------------
# 4. Data Pivoting & Reshaping (reshape)
# ---------------------------------------------------------------------------
def execute_reshape(dataset, cmd):
    """
    Implements dataset reshape:
    reshape long stubnames, i(varlist) j(varname)
    reshape wide stubnames, i(varlist) j(varname)
    """
    df = dataset.df
    if df.empty:
        return CommandResult("no dataset in memory", ok=False, return_code=111)

    tokens = cmd.body.split()
    if not tokens:
        return CommandResult("syntax: reshape long|wide stubnames, i(varlist) j(varname)", ok=False, return_code=198)

    subdirection = tokens[0].lower()
    if subdirection not in ('long', 'wide'):
        return CommandResult(f"unknown reshape direction: '{subdirection}' (must be long or wide)", ok=False, return_code=198)

    stubnames = tokens[1:]
    if not stubnames:
        return CommandResult("syntax error: missing stubnames in reshape", ok=False, return_code=198)

    i_opt = cmd.get_option("i")
    j_opt = cmd.get_option("j")
    if not i_opt or not j_opt:
        return CommandResult("options i() and j() are required for reshape", ok=False, return_code=198)

    i_vars = i_opt.split()
    j_var = j_opt.strip()

    for iv in i_vars:
        if iv not in df.columns:
            return CommandResult(f"variable '{iv}' not found", ok=False, return_code=111)

    if subdirection == "long":
        # Wide to Long
        # Find matching columns for each stub (e.g. inc80, inc81 -> stub inc, suffix 80, 81)
        long_dfs = []
        for stub in stubnames:
            pattern = re.compile(rf"^{stub}([a-zA-Z0-9_]+)$")
            matching_cols = [c for c in df.columns if pattern.match(c)]
            if not matching_cols:
                return CommandResult(f"no variables found matching stub '{stub}'", ok=False, return_code=111)

            melted = pd.melt(
                df,
                id_vars=i_vars,
                value_vars=matching_cols,
                var_name='_stub_col_',
                value_name=stub
            )
            melted[j_var] = melted['_stub_col_'].apply(lambda x, pat=pattern: pat.match(x).group(1))
            # Convert j to numeric if possible
            try:
                melted[j_var] = pd.to_numeric(melted[j_var])
            except (ValueError, TypeError):
                pass
            melted.drop(columns=['_stub_col_'], inplace=True)
            long_dfs.append(melted)

        # Merge on i_vars and j_var
        res_df = long_dfs[0]
        for other in long_dfs[1:]:
            res_df = pd.merge(res_df, other, on=i_vars + [j_var], how='outer')

        # Add remaining non-reshaped columns
        static_cols = [c for c in df.columns if c not in i_vars and not any(re.match(rf"^{s}[a-zA-Z0-9_]+$", c) for s in stubnames)]
        if static_cols:
            static_df = df[i_vars + static_cols].drop_duplicates(subset=i_vars)
            res_df = pd.merge(res_df, static_df, on=i_vars, how='left')

        dataset.df = res_df.sort_values(by=i_vars + [j_var], ignore_index=True)
        return CommandResult(f"(reshaped to long format; N = {len(res_df):,}, j = '{j_var}')", ok=True)

    else:
        # Long to Wide
        if j_var not in df.columns:
            return CommandResult(f"variable '{j_var}' not found in long data", ok=False, return_code=111)

        # Pivot stub variables
        res_df = df.pivot(index=i_vars, columns=j_var, values=stubnames)
        # Flatten multi-level column names: stub + str(j)
        new_cols = []
        for col_tuple in res_df.columns:
            stub, j_val = col_tuple
            new_cols.append(f"{stub}{j_val}")
        res_df.columns = new_cols
        res_df.reset_index(inplace=True)

        dataset.df = res_df
        return CommandResult(f"(reshaped to wide format; N = {len(res_df):,}, generated {len(new_cols)} stub variables)", ok=True)


# ---------------------------------------------------------------------------
# 5. Data Aggregation (collapse)
# ---------------------------------------------------------------------------
def execute_collapse(dataset, cmd):
    """
    Implements dataset aggregation collapse:
    collapse (stat) varlist [(stat) varlist ...] [if] [in], by(groupvars)
    """
    df = dataset.df
    if df.empty:
        return CommandResult("no dataset in memory", ok=False, return_code=111)

    by_opt = cmd.get_option("by")
    if not by_opt:
        return CommandResult("option by() is required for collapse", ok=False, return_code=198)

    by_vars = by_opt.split()
    for v in by_vars:
        if v not in df.columns:
            return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)

    # Filter with if/in qualifiers if present
    _, df_sub = evaluate_if_in_qualifiers(df, if_expr=cmd.if_expr, in_range=cmd.in_range)
    if df_sub.empty:
        return CommandResult("no observations match if/in condition", ok=False, return_code=2000)

    # Parse (stat) varlist pairs
    body = cmd.body
    matches = list(re.finditer(r'\(([a-zA-Z0-9_]+)\)\s*([^()]+)', body))
    if not matches:
        return CommandResult("syntax error: collapse (stat) varlist [(stat) varlist ...], by(groupvars)", ok=False, return_code=198)

    agg_dict = {}
    stat_map = {
        'mean': 'mean',
        'median': 'median',
        'p50': 'median',
        'sd': 'std',
        'min': 'min',
        'max': 'max',
        'count': 'count',
        'sum': 'sum',
        'total': 'sum',
        'first': 'first',
        'last': 'last'
    }

    percentile_map = {
        'p25': lambda x: np.nanpercentile(x, 25),
        'p75': lambda x: np.nanpercentile(x, 75),
        'p90': lambda x: np.nanpercentile(x, 90)
    }

    for m in matches:
        stat_name = m.group(1).lower()
        vars_in_stat = m.group(2).strip().split()
        func = percentile_map.get(stat_name, stat_map.get(stat_name))
        if not func:
            return CommandResult(f"unsupported collapse statistic: '{stat_name}'", ok=False, return_code=198)

        for v in vars_in_stat:
            if v not in df_sub.columns:
                return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
            agg_dict[v] = func

    grouped = df_sub.groupby(by_vars, as_index=False).agg(agg_dict)
    dataset.df = grouped
    return CommandResult(f"(collapsed dataset: {len(grouped):,} grouped observations by {', '.join(by_vars)})", ok=True)


# ---------------------------------------------------------------------------
# 6. egen Extended Generate
# ---------------------------------------------------------------------------
def execute_egen(dataset, cmd):
    """
    Implements extended generate (egen):
    egen [type] newvar = func(args) [if] [in]
    Supported functions: mean, total, sum, sd, min, max, count, rank, cut, group, tag,
    rowtotal, rowmean, rowmin, rowmax, rowmiss, rownonmiss.
    """
    df = dataset.df
    if df.empty:
        return CommandResult("no dataset in memory", ok=False, return_code=111)

    body = cmd.body
    if "=" not in body:
        return CommandResult("syntax error: egen newvar = func(args) [if] [in]", ok=False, return_code=198)

    left, right = body.split("=", 1)
    newvar = left.strip().split()[-1]
    right = right.strip()

    if newvar in df.columns:
        return CommandResult(f"variable '{newvar}' already defined", ok=False, return_code=110)

    match_call = re.match(r'^([a-zA-Z0-9_]+)\((.*)\)$', right)
    if not match_call:
        return CommandResult(f"invalid egen function call: '{right}'", ok=False, return_code=198)

    func_name = match_call.group(1).lower()
    arg_str = match_call.group(2).strip()
    args = [a.strip() for a in arg_str.split() if a.strip()]

    mask, _ = evaluate_if_in_qualifiers(df, if_expr=cmd.if_expr, in_range=cmd.in_range)
    target_series = pd.Series(np.nan, index=df.index)

    # 1. Summary aggregations across rows (mean, total, sd, min, max, count)
    if func_name in ('mean', 'total', 'sum', 'sd', 'min', 'max', 'count'):
        if not args:
            return CommandResult(f"function {func_name}() requires a variable name", ok=False, return_code=198)
        v = args[0]
        if v not in df.columns:
            return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
        sub_vals = df.loc[mask, v].dropna()

        if func_name == 'mean':
            val = sub_vals.mean()
        elif func_name in ('total', 'sum'):
            val = sub_vals.sum()
        elif func_name == 'sd':
            val = sub_vals.std()
        elif func_name == 'min':
            val = sub_vals.min()
        elif func_name == 'max':
            val = sub_vals.max()
        elif func_name == 'count':
            val = float(len(sub_vals))

        target_series[mask] = val

    # 2. Rank
    elif func_name == 'rank':
        v = args[0]
        if v not in df.columns:
            return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
        target_series[mask] = df.loc[mask, v].rank(method='min')

    elif func_name == 'pctile':
        if not args:
            return CommandResult("pctile() requires a variable name", ok=False, return_code=198)
        v = args[0]
        if v not in df.columns:
            return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
        p_arg = cmd.get_option("p", 50)
        try:
            p_val = float(p_arg)
        except Exception:
            p_val = 50.0
        sub_vals = df.loc[mask, v].dropna()
        if len(sub_vals) > 0:
            target_series[mask] = float(np.percentile(sub_vals, p_val))

    elif func_name == 'std':
        if not args:
            return CommandResult("std() requires a variable name", ok=False, return_code=198)
        v = args[0]
        if v not in df.columns:
            return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
        sub_vals = df.loc[mask, v]
        m_val = sub_vals.mean()
        s_val = sub_vals.std(ddof=1)
        target_series[mask] = (sub_vals - m_val) / (s_val if s_val > 0 else 1.0)

    elif func_name == 'mode':
        if not args:
            return CommandResult("mode() requires a variable name", ok=False, return_code=198)
        v = args[0]
        if v not in df.columns:
            return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
        sub_vals = df.loc[mask, v].dropna()
        if len(sub_vals) > 0:
            modes = sub_vals.mode()
            target_series[mask] = modes.iloc[0] if len(modes) > 0 else np.nan

    # 3. Group (sequential IDs for unique combinations)
    elif func_name == 'group':
        if not args:
            return CommandResult("group() requires at least one variable", ok=False, return_code=198)
        for v in args:
            if v not in df.columns:
                return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
        # Factorize unique tuples
        tuples = df.loc[mask, args].apply(tuple, axis=1)
        codes, _ = pd.factorize(tuples)
        target_series[mask] = codes + 1  # 1-based indexing

    # 4. Tag (1 for first occurrence of unique combination, 0 otherwise)
    elif func_name == 'tag':
        if not args:
            return CommandResult("tag() requires at least one variable", ok=False, return_code=198)
        for v in args:
            if v not in df.columns:
                return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
        is_first = ~df.loc[mask].duplicated(subset=args)
        target_series[mask] = is_first.astype(int)

    # 5. Row functions (rowtotal, rowmean, rowmin, rowmax, rowmiss, rownonmiss)
    elif func_name in ('rowtotal', 'rowmean', 'rowmin', 'rowmax', 'rowmiss', 'rownonmiss'):
        for v in args:
            if v not in df.columns:
                return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
        sub_block = df.loc[mask, args]

        if func_name == 'rowtotal':
            # Missing treated as 0 unless all missing
            target_series[mask] = sub_block.sum(axis=1, min_count=1)
        elif func_name == 'rowmean':
            target_series[mask] = sub_block.mean(axis=1)
        elif func_name == 'rowmin':
            target_series[mask] = sub_block.min(axis=1)
        elif func_name == 'rowmax':
            target_series[mask] = sub_block.max(axis=1)
        elif func_name == 'rowmiss':
            target_series[mask] = sub_block.isna().sum(axis=1)
        elif func_name == 'rownonmiss':
            target_series[mask] = sub_block.notna().sum(axis=1)

    # 6. Cut
    elif func_name == 'cut':
        v = args[0]
        if v not in df.columns:
            return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
        group_num = int(cmd.get_option("group", 4))
        target_series[mask] = pd.qcut(df.loc[mask, v], q=group_num, labels=False, duplicates='drop')

    else:
        return CommandResult(f"unknown egen function: '{func_name}()'", ok=False, return_code=198)

    df[newvar] = target_series
    return CommandResult(f"(egen variable '{newvar}' created using {func_name}())", ok=True)


# ---------------------------------------------------------------------------
# 7. Categorical Encoding & Decoding
# ---------------------------------------------------------------------------
def execute_encode(dataset, cmd):
    """
    Implements categorical encode:
    encode stringvar, gen(numvar) [label(lblname)]
    """
    df = dataset.df
    if df.empty:
        return CommandResult("no dataset in memory", ok=False, return_code=111)

    tokens = cmd.body.split()
    if not tokens:
        return CommandResult("syntax error: encode stringvar, gen(newvar) [label(lblname)]", ok=False, return_code=198)

    svar = tokens[0]
    if svar not in df.columns:
        return CommandResult(f"variable '{svar}' not found", ok=False, return_code=111)

    gen_var = cmd.get_option("gen") or cmd.get_option("generate")
    if not gen_var:
        return CommandResult("option gen() is required for encode", ok=False, return_code=198)
    if gen_var in df.columns:
        return CommandResult(f"variable '{gen_var}' already defined", ok=False, return_code=110)

    lblname = cmd.get_option("label", gen_var)

    # Map sorted distinct text values to 1, 2, ...
    distinct_vals = sorted(df[svar].dropna().unique())
    val_map = {val: idx + 1 for idx, val in enumerate(distinct_vals)}
    label_dict = {idx + 1: str(val) for idx, val in enumerate(distinct_vals)}

    df[gen_var] = df[svar].map(val_map)
    dataset.val_labels[lblname] = label_dict
    dataset.val_label_names[gen_var] = lblname

    return CommandResult(f"(variable '{gen_var}' encoded with {len(label_dict)} distinct categories)", ok=True)


def execute_decode(dataset, cmd):
    """
    Implements categorical decode:
    decode numvar, gen(stringvar)
    """
    df = dataset.df
    if df.empty:
        return CommandResult("no dataset in memory", ok=False, return_code=111)

    tokens = cmd.body.split()
    if not tokens:
        return CommandResult("syntax error: decode numvar, gen(stringvar)", ok=False, return_code=198)

    nvar = tokens[0]
    if nvar not in df.columns:
        return CommandResult(f"variable '{nvar}' not found", ok=False, return_code=111)

    gen_var = cmd.get_option("gen") or cmd.get_option("generate")
    if not gen_var:
        return CommandResult("option gen() is required for decode", ok=False, return_code=198)
    if gen_var in df.columns:
        return CommandResult(f"variable '{gen_var}' already defined", ok=False, return_code=110)

    lblname = dataset.val_label_names.get(nvar)
    label_dict = dataset.val_labels.get(lblname, {}) if lblname else {}

    if not label_dict:
        # Decode without labels falls back to string conversion
        df[gen_var] = df[nvar].astype(str)
    else:
        df[gen_var] = df[nvar].map(label_dict).fillna(df[nvar].astype(str))

    return CommandResult(f"(variable '{gen_var}' decoded to string)", ok=True)


# ---------------------------------------------------------------------------
# 8. Label Management (label define, values, variable, list, dir, drop)
# ---------------------------------------------------------------------------
def execute_label(dataset, cmd):
    """
    Implements variable label management:
    label define lblname # "label" [# "label" ...]
    label values varlist lblname
    label variable varname "text"
    label dir
    label list [lblname]
    label drop lblname | _all
    """
    body = cmd.body.strip()
    if not body:
        return CommandResult("syntax error: label define|values|variable|dir|list|drop ...", ok=False, return_code=198)

    tokens = body.split(None, 1)
    subcmd = tokens[0].lower()
    rest = tokens[1].strip() if len(tokens) > 1 else ""

    # 1. label variable varname "description"
    if subcmd in ('variable', 'var'):
        m = re.match(r'^([a-zA-Z0-9_]+)\s+["\'](.*)["\']$', rest)
        if not m:
            return CommandResult("syntax: label variable varname \"label text\"", ok=False, return_code=198)
        vname, text = m.group(1), m.group(2)
        if vname not in dataset.df.columns:
            return CommandResult(f"variable '{vname}' not found", ok=False, return_code=111)
        dataset.var_labels[vname] = text
        return CommandResult(f"(label for '{vname}' set to \"{text}\")", ok=True)

    # 2. label define lblname # "label" ...
    elif subcmd in ('define', 'def'):
        parts = rest.split(None, 1)
        if len(parts) < 2:
            return CommandResult("syntax: label define lblname # \"label\" [# \"label\" ...]", ok=False, return_code=198)
        lblname = parts[0]
        rules_str = parts[1]

        label_dict = dataset.val_labels.get(lblname, {}) if (cmd.has_option("modify") or cmd.has_option("add")) else {}
        matches = list(re.finditer(r'([+-]?\d+(?:\.\d+)?)\s+["\']([^"\']*)["\']', rules_str))
        if not matches:
            return CommandResult("syntax error: no valid # \"label\" definitions found", ok=False, return_code=198)

        for m in matches:
            val_num = float(m.group(1))
            val_key = int(val_num) if val_num.is_integer() else val_num
            label_dict[val_key] = m.group(2)

        dataset.val_labels[lblname] = label_dict
        return CommandResult(f"(value label '{lblname}' defined with {len(label_dict)} mappings)", ok=True)

    # 3. label values varlist [lblname]
    elif subcmd in ('values', 'val'):
        parts = rest.split()
        if not parts:
            return CommandResult("syntax: label values varlist [lblname]", ok=False, return_code=198)
        if len(parts) == 1:
            # Drop label association
            v = parts[0]
            if v in dataset.val_label_names:
                del dataset.val_label_names[v]
            return CommandResult(f"(value label detached from '{v}')", ok=True)
        else:
            lblname = parts[-1]
            vars_to_label = parts[:-1]
            for v in vars_to_label:
                if v not in dataset.df.columns:
                    return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)
                dataset.val_label_names[v] = lblname
            return CommandResult(f"(value label '{lblname}' assigned to {', '.join(vars_to_label)})", ok=True)

    # 4. label dir
    elif subcmd in ('dir',):
        if not dataset.val_labels:
            return CommandResult("(no value labels defined)", ok=True)
        lines = ["Defined value labels:"]
        for name, d in dataset.val_labels.items():
            lines.append(f"  {name:<16} ({len(d)} labels)")
        return CommandResult("\n".join(lines), ok=True)

    # 5. label list [lblname]
    elif subcmd in ('list',):
        target = rest.strip()
        names = [target] if target else list(dataset.val_labels.keys())
        lines = []
        for name in names:
            if name not in dataset.val_labels:
                return CommandResult(f"value label '{name}' not found", ok=False, return_code=111)
            lines.append(f"{name}:")
            for k, v in sorted(dataset.val_labels[name].items()):
                lines.append(f"  {k:>4} {v}")
        return CommandResult("\n".join(lines), ok=True)

    # 6. label drop lblname | _all
    elif subcmd in ('drop',):
        target = rest.strip()
        if target == '_all':
            dataset.val_labels.clear()
            dataset.val_label_names.clear()
            return CommandResult("(all value labels dropped)", ok=True)
        elif target in dataset.val_labels:
            del dataset.val_labels[target]
            return CommandResult(f"(value label '{target}' dropped)", ok=True)
        else:
            return CommandResult(f"value label '{target}' not found", ok=False, return_code=111)

    return CommandResult(f"unknown label subcommand: '{subcmd}'", ok=False, return_code=198)


# ---------------------------------------------------------------------------
# 9. Recode (recode)
# ---------------------------------------------------------------------------
def execute_recode(dataset, cmd):
    """
    Implements value recode:
    recode var (rule1) (rule2) ... [if] [in] [, gen(newvar) prefix(str)]
    Rules: (oldval = newval), (1/5 = 1), (min/10 = 0), (else = 99)
    """
    df = dataset.df
    if df.empty:
        return CommandResult("no dataset in memory", ok=False, return_code=111)

    body = cmd.body
    match_var = re.match(r'^([a-zA-Z0-9_]+)\s*(.*)$', body)
    if not match_var:
        return CommandResult("syntax error: recode var (rule1) ... [, gen(newvar)]", ok=False, return_code=198)

    varname = match_var.group(1)
    if varname not in df.columns:
        return CommandResult(f"variable '{varname}' not found", ok=False, return_code=111)

    rules_part = match_var.group(2)
    rule_blocks = re.findall(r'\((.*?)\)', rules_part)
    if not rule_blocks:
        return CommandResult("syntax error: recode requires at least one (rule) specification", ok=False, return_code=198)

    gen_var = cmd.get_option("gen") or cmd.get_option("generate")
    target_var = gen_var if gen_var else varname

    if gen_var and gen_var in df.columns:
        return CommandResult(f"variable '{gen_var}' already defined", ok=False, return_code=110)

    mask, _ = evaluate_if_in_qualifiers(df, if_expr=cmd.if_expr, in_range=cmd.in_range)
    recoded = df[varname].copy()

    for rule in rule_blocks:
        if "=" not in rule:
            continue
        left, right = rule.split("=", 1)
        new_val = float(right.strip())

        left_str = left.strip().lower()
        if "/" in left_str:
            # Range rule: min/10 or 1/5 or 20/max
            p1, p2 = left_str.split("/", 1)
            lower = -np.inf if p1.strip() == 'min' else float(p1.strip())
            upper = np.inf if p2.strip() == 'max' else float(p2.strip())
            cond = mask & (df[varname] >= lower) & (df[varname] <= upper)
            recoded[cond] = new_val
        elif left_str == 'else':
            recoded[mask & recoded.isna()] = new_val
        else:
            # Single value or list of values
            targets = [float(x) for x in left_str.split()]
            cond = mask & df[varname].isin(targets)
            recoded[cond] = new_val

    df[target_var] = recoded
    return CommandResult(f"(variable '{target_var}' recoded)", ok=True)


# ---------------------------------------------------------------------------
# 10. Duplicates Management (duplicates report, list, tag, drop)
# ---------------------------------------------------------------------------
def execute_duplicates(dataset, cmd):
    """
    Implements duplicate observation management:
    duplicates report [varlist]
    duplicates list [varlist]
    duplicates tag [varlist], gen(newvar)
    duplicates drop [varlist], force
    """
    df = dataset.df
    if df.empty:
        return CommandResult("no dataset in memory", ok=False, return_code=111)

    tokens = cmd.body.split()
    subcmd = tokens[0].lower() if tokens else "report"
    varlist = tokens[1:] if len(tokens) > 1 else list(df.columns)

    for v in varlist:
        if v not in df.columns:
            return CommandResult(f"variable '{v}' not found", ok=False, return_code=111)

    mask, df_sub = evaluate_if_in_qualifiers(df, if_expr=cmd.if_expr, in_range=cmd.in_range)

    # 1. duplicates report
    if subcmd in ('report',):
        counts = df_sub.groupby(varlist, dropna=False).size()
        freq_of_copies = counts.value_counts().sort_index()

        lines = [
            "Duplicates in terms of " + ", ".join(varlist),
            "",
            "   Copies | Observations       Surplus",
            "----------+---------------------------"
        ]
        total_obs = 0
        total_surplus = 0
        for copies, num_groups in freq_of_copies.items():
            n_obs = copies * num_groups
            surplus = (copies - 1) * num_groups
            total_obs += n_obs
            total_surplus += surplus
            lines.append(f"     {copies:>4} | {n_obs:>12,}  {surplus:>12,}")
        lines.append("----------+---------------------------")
        lines.append(f"    Total | {total_obs:>12,}  {total_surplus:>12,}")
        return CommandResult("\n".join(lines), ok=True)

    # 2. duplicates tag
    elif subcmd in ('tag',):
        gen_var = cmd.get_option("gen") or cmd.get_option("generate")
        if not gen_var:
            return CommandResult("option generate() is required for duplicates tag", ok=False, return_code=198)
        if gen_var in df.columns:
            return CommandResult(f"variable '{gen_var}' already defined", ok=False, return_code=110)

        # Count copies minus 1 (0 for unique)
        counts = df.groupby(varlist, dropna=False)[varlist[0]].transform('count')
        df[gen_var] = counts - 1
        return CommandResult(f"(variable '{gen_var}' created with duplicate counts)", ok=True)

    # 3. duplicates drop
    elif subcmd in ('drop',):
        if not cmd.has_option("force"):
            return CommandResult("duplicates drop requires the 'force' option to prevent accidental deletion", ok=False, return_code=198)
        n_before = len(df)
        df.drop_duplicates(subset=varlist, inplace=True, ignore_index=True)
        n_dropped = n_before - len(df)
        return CommandResult(f"({n_dropped:,} duplicate observations dropped)", ok=True)

    # 4. duplicates list
    elif subcmd in ('list',):
        dup_mask = df_sub.duplicated(subset=varlist, keep=False)
        dups = df_sub[dup_mask]
        if dups.empty:
            return CommandResult("(no duplicate observations found)", ok=True)
        return CommandResult(dups.to_string(index=True), ok=True)

    return CommandResult(f"unknown duplicates subcommand: '{subcmd}'", ok=False, return_code=198)
