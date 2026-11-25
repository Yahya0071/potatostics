"""
plugins/example_plugin.py - Sample User Plugin for Potatostics
Adds a 'zscore <varname>' command to compute standardized (z-score) variables.
"""

def handle_zscore(interpreter, args, options):
    if interpreter.dataset.is_empty:
        return "no dataset in memory"
    if not args:
        return "syntax: zscore <varname>"
    var = args[0]
    if var not in interpreter.dataset.df.columns:
        return f"variable '{var}' not found"
        
    s = interpreter.dataset.df[var].dropna()
    z_name = f"z_{var}"
    interpreter.dataset.df[z_name] = (interpreter.dataset.df[var] - s.mean()) / s.std(ddof=1)
    interpreter.dataset.var_labels[z_name] = f"Standardized z-score of {var}"
    interpreter.dataset.var_formats[z_name] = "%9.4f"
    return f"(standardized variable '{z_name}' created)"

def register_plugin(interpreter):
    # Register command into interpreter custom verbs dictionary
    interpreter.custom_commands["zscore"] = lambda args, options, line: handle_zscore(interpreter, args, options)
