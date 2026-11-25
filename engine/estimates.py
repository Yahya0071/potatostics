"""
engine/estimates.py - Estimates Store & Academic Comparison Table Generator
Powers 'estimates store', 'estimates table', and 'esttab' publication-ready exports (LaTeX, HTML, Excel).
"""


class EstimatesStore:
    def __init__(self):
        self.store = {}
        self.last_model = None

    def save_last_model(self, model_dict):
        self.last_model = model_dict

    def store_estimate(self, name):
        if not self.last_model:
            raise ValueError("No active estimation model in memory to store.")
        self.store[name] = self.last_model
        return f"(model stored as '{name}')"

    def list_estimates(self):
        if not self.store:
            return "no stored estimation models in memory"
        lines = [f"{'Name':<15} {'Model Type':<12} {'Depvar':<12} {'N Obs':>8} {'R2 / Pseudo R2':>15}"]
        lines.append("-" * 65)
        for name, m in self.store.items():
            r2_val = m.get('r2', m.get('pseudo_r2', m.get('r2_within', 0.0)))
            lines.append(f"{name:<15} {m['model_type']:<12} {m['depvar']:<12} {m['n']:>8,} {r2_val:>15.4f}")
        return "\n".join(lines)

    def generate_comparison_table(self, model_names=None):
        if not model_names:
            model_names = list(self.store.keys())
        if not model_names:
            if self.last_model:
                model_names = ['(active)']
                models = {'(active)': self.last_model}
            else:
                return "no estimation models available to compare"
        else:
            models = {name: self.store[name] for name in model_names if name in self.store}

        # Collect all unique regressors across models
        all_vars = []
        for m in models.values():
            for row in m['coefs']:
                if row['name'] not in all_vars and row['name'] != '_cons':
                    all_vars.append(row['name'])
        all_vars.append('_cons')

        col_w = 16
        lines = []
        header = f"{'Variable':<14} | " + " ".join(f"{name:^{col_w}}" for name in models.keys())
        lines.append(header)
        lines.append("-" * 15 + "+" + "-" * (col_w * len(models) + len(models)))

        for v in all_vars:
            coef_strs = []
            se_strs = []
            for m in models.values():
                coef_dict = {r['name']: r for r in m['coefs']}
                if v in coef_dict:
                    row = coef_dict[v]
                    b = row['coef']
                    p = row['p']
                    se = row['se']
                    # Stars
                    stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
                    coef_strs.append(f"{b:10.4f}{stars:<3}")
                    se_strs.append(f"({se:10.4f})   ")
                else:
                    coef_strs.append(f"{'':^{col_w}}")
                    se_strs.append(f"{'':^{col_w}}")
            lines.append(f"{v:<14} | " + " ".join(f"{s:^{col_w}}" for s in coef_strs))
            lines.append(f"{'':<14} | " + " ".join(f"{s:^{col_w}}" for s in se_strs))

        lines.append("-" * 15 + "+" + "-" * (col_w * len(models) + len(models)))
        
        # Summary statistics rows
        obs_strs = [f"{m['n']:^{col_w},}" for m in models.values()]
        lines.append(f"{'N Obs':<14} | " + " ".join(obs_strs))
        
        r2_strs = []
        for m in models.values():
            r2_val = m.get('r2', m.get('pseudo_r2', m.get('r2_within', 0.0)))
            r2_strs.append(f"{r2_val:^{col_w}.4f}")
        lines.append(f"{'R-squared':<14} | " + " ".join(r2_strs))
        lines.append("-" * 15 + "+" + "-" * (col_w * len(models) + len(models)))
        lines.append("* p<0.10, ** p<0.05, *** p<0.01; standard errors in parentheses")
        return "\n".join(lines)

