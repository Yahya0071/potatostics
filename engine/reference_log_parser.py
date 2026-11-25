"""
engine/reference_log_parser.py - Parser for Reference Log Files and Benchmark Validation
Extracts estimation tables from benchmark logs (.log / text) and cross-checks
estimated coefficients and standard errors against Potatostics.
"""

import re
from typing import Dict, Any, List


def parse_reference_log(log_text: str) -> Dict[str, Any]:
    """
    Parses a plain-text benchmark log and returns a dictionary of detected models.
    Supports regress, logit, and xtreg estimation tables.
    """
    models = {}
    lines = log_text.splitlines()
    
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        
        # Check command invocation line (e.g. ". regress price mpg weight")
        cmd_match = re.match(r"^\.\s*(regress|logit|probit|poisson|xtreg|reghdfe|ivregress)\b", line, re.IGNORECASE)
        if cmd_match:
            cmd_type = cmd_match.group(1).lower()
            cmd_line = line
            i += 1
            
            # Find the coefficient table header line (contains 'Coefficient' or 'Coef.')
            while i < n:
                if re.search(r"\b(Coefficient|Coef\.)\b", lines[i], re.IGNORECASE):
                    break
                # If another command starts, abort
                if re.match(r"^\.\s*[a-z]", lines[i].strip(), re.IGNORECASE):
                    break
                i += 1
            
            if i >= n or re.match(r"^\.\s*[a-z]", lines[i].strip(), re.IGNORECASE):
                continue
                
            # Skip past the separator line right below the header
            i += 1
            while i < n and ("----" in lines[i] or "====" in lines[i]):
                i += 1
                
            table = {}
            # Parse table rows until end-of-table dashed line or next command/blank section
            while i < n:
                row_line = lines[i].strip()
                if row_line.startswith("----") or row_line.startswith("===="):
                    break
                if not row_line or row_line.startswith("."):
                    break
                    
                # Standard benchmark table row: "varname |  coef   std.err   t/z   p ..."
                if "|" in row_line:
                    parts = row_line.split("|", 1)
                    var_name = parts[0].strip()
                    nums_str = parts[1].strip()
                    
                    if var_name:
                        # Extract numerical tokens (handles leading dots like .6413538, negative signs)
                        num_tokens = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", nums_str)
                        if len(num_tokens) >= 2:
                            try:
                                coef_str = num_tokens[0]
                                se_str = num_tokens[1]
                                coef = float(coef_str)
                                se = float(se_str)
                                stat = float(num_tokens[2]) if len(num_tokens) >= 3 else None
                                pval = float(num_tokens[3]) if len(num_tokens) >= 4 else None
                                
                                # Count printed decimal places
                                dec_coef = len(coef_str.split(".")[1]) if "." in coef_str else 0
                                dec_se = len(se_str.split(".")[1]) if "." in se_str else 0
                                
                                table[var_name] = {
                                    "coef": coef,
                                    "se": se,
                                    "stat": stat,
                                    "p": pval,
                                    "dec_coef": dec_coef,
                                    "dec_se": dec_se,
                                }
                            except ValueError:
                                pass
                i += 1
                
            if table:
                models[cmd_type] = {
                    "command": cmd_line,
                    "table": table
                }
        else:
            i += 1

    return models


def compare_models_to_reference(
    potatostics_model: Any,
    reference_table: Dict[str, Dict[str, float]],
    tolerance: float = 1e-4
) -> List[str]:
    """
    Compares coefficients and standard errors of a Potatostics estimated model
    against an extracted benchmark table.
    
    Accepts an Interpreter instance, EstimatesStore, ModelEstimates object, or dict.
    Returns a list of discrepancy description strings; empty if all within tolerance.
    """
    discrepancies = []
    
    # Unwrap Interpreter / EstimatesStore if passed directly
    if hasattr(potatostics_model, "estimates"):
        potatostics_model = getattr(potatostics_model.estimates, "last_model", potatostics_model)
    elif hasattr(potatostics_model, "last_model"):
        potatostics_model = potatostics_model.last_model

    pot_params = {}
    pot_se = {}

    if isinstance(potatostics_model, dict):
        if "coefs" in potatostics_model:
            for item in potatostics_model["coefs"]:
                pot_params[item["name"]] = item.get("coef", item.get("b", 0.0))
                pot_se[item["name"]] = item.get("se", 0.0)
        elif "b" in potatostics_model and "se" in potatostics_model:
            pot_params = dict(potatostics_model["b"])
            pot_se = dict(potatostics_model["se"])
    elif hasattr(potatostics_model, "registers"):
        pot_params = dict(potatostics_model.registers.e_b)
        pot_se = dict(potatostics_model.registers.e_se)
    elif hasattr(potatostics_model, "params") and hasattr(potatostics_model, "b_se"):
        pot_params = dict(potatostics_model.params)
        pot_se = dict(potatostics_model.b_se)
    elif hasattr(potatostics_model, "b") and hasattr(potatostics_model, "se"):
        pot_params = dict(potatostics_model.b)
        pot_se = dict(potatostics_model.se)
    else:
        return [f"Unknown model estimates structure: {type(potatostics_model)}"]

    for var, ref_vals in reference_table.items():
        # Match var name (handle _cons vs const)
        pot_var = var
        if pot_var not in pot_params:
            if var == "_cons" and "const" in pot_params:
                pot_var = "const"
            elif var == "const" and "_cons" in pot_params:
                pot_var = "_cons"
            else:
                discrepancies.append(f"Variable '{var}' present in benchmark table but absent in Potatostics model")
                continue
                
        pot_coef = float(pot_params[pot_var])
        pot_stderr = float(pot_se[pot_var])
        ref_coef = float(ref_vals["coef"])
        ref_stderr = float(ref_vals["se"])
        
        # Effective tolerance: 4 decimals (1e-4), or printed precision if fewer decimals exist
        dec_c = ref_vals.get("dec_coef", 4)
        dec_s = ref_vals.get("dec_se", 4)
        tol_c = max(tolerance, 10 ** (-dec_c)) if dec_c < 4 else tolerance
        tol_s = max(tolerance, 10 ** (-dec_s)) if dec_s < 4 else tolerance

        # Check coefficient difference
        coef_diff = abs(pot_coef - ref_coef)
        if coef_diff > tol_c:
            discrepancies.append(
                f"Variable '{var}' coef discrepancy: Potatostics={pot_coef:.6f}, Reference={ref_coef:.6f}, diff={coef_diff:.6f} > {tol_c}"
            )
            
        # Check standard error difference
        se_diff = abs(pot_stderr - ref_stderr)
        if se_diff > tol_s:
            discrepancies.append(
                f"Variable '{var}' se discrepancy: Potatostics={pot_stderr:.6f}, Reference={ref_stderr:.6f}, diff={se_diff:.6f} > {tol_s}"
            )

    return discrepancies
