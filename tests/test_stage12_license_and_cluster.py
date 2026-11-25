import os
import unittest
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

class TestLicenseAgreement(unittest.TestCase):
    def test_license_consistency_across_files(self):
        """LICENSE, README.md, and pyproject.toml must declare identical licenses."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        # 1. pyproject.toml
        pyproject_path = os.path.join(repo_root, "pyproject.toml")
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        pyproj_license = data.get("project", {}).get("license", {}).get("text", "")

        # 2. LICENSE file
        license_path = os.path.join(repo_root, "LICENSE")
        with open(license_path, "r", encoding="utf-8") as f:
            license_text = f.read()

        # 3. README.md
        readme_path = os.path.join(repo_root, "README.md")
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_text = f.read()

        # Assert consistent LGPL-3.0-or-later declarations
        self.assertEqual(pyproj_license, "LGPL-3.0-or-later")
        self.assertIn("GNU LESSER GENERAL PUBLIC LICENSE", license_text)
        self.assertIn("Version 3", license_text)
        self.assertIn("LGPL-3.0-or-later", readme_text)

        # 4. gui/branding.py must declare LGPL-3.0-or-later, not MIT License
        branding_path = os.path.join(repo_root, "gui", "branding.py")
        with open(branding_path, "r", encoding="utf-8") as f:
            branding_text = f.read()
        self.assertNotIn("MIT License", branding_text)
        self.assertIn("LGPL-3.0-or-later License", branding_text)

import numpy as np
import pandas as pd
from scipy import stats
from engine.estimation import compute_logit
from engine.formatter import format_logit

class TestLogitClusterInference(unittest.TestCase):
    def test_logit_cluster_independent_sandwich_and_format(self):
        """
        Logit with vce(cluster) must match an independent sandwich implementation using
        G/(G-1) small-sample factor and standard normal z inference, and format_logit
        must print 'Std. err.', 'z', 'P>|z|', and '(Std. err. adjusted for N clusters in ...)'.
        """
        np.random.seed(42)
        n = 150
        n_groups = 15
        groups = np.repeat(np.arange(n_groups), n // n_groups)
        x = np.random.randn(n)
        # Latent index with group effects
        group_effects = np.random.randn(n_groups) * 0.5
        z_latent = 0.5 + 1.2 * x + group_effects[groups]
        p_true = 1.0 / (1.0 + np.exp(-z_latent))
        y = (np.random.rand(n) < p_true).astype(float)

        df = pd.DataFrame({'y': y, 'x': x, 'cid': groups})
        res = compute_logit(df, 'y', ['x'], vce='cluster', cluster_var='cid')

        # --- Independent manual sandwich verification ---
        beta = res['beta']
        X = np.column_stack([df['x'].to_numpy(dtype=float), np.ones(n)])
        p_hat = 1.0 / (1.0 + np.exp(-X @ beta))
        
        # Hessian: - X' diag(p*(1-p)) X
        w = p_hat * (1.0 - p_hat)
        H = - (X.T * w) @ X
        inv_H = np.linalg.inv(-H)

        # Cluster scores
        scores = X * (df['y'].to_numpy() - p_hat)[:, None]
        unique_c = np.unique(groups)
        G = len(unique_c)
        meat = np.zeros_like(H)
        for g in unique_c:
            s_g = np.sum(scores[groups == g], axis=0)
            meat += np.outer(s_g, s_g)

        c_factor = G / (G - 1.0)
        vcov_manual = c_factor * (inv_H @ meat @ inv_H)
        se_manual = np.sqrt(np.diag(vcov_manual))
        z_manual = beta / se_manual
        p_manual = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_manual)))

        # Assert vcov, SE, z, and p match
        np.testing.assert_allclose(res['vcov'], vcov_manual, rtol=1e-6)
        for i, col_name in enumerate(['x', '_cons']):
            np.testing.assert_allclose(res['coefs'][i]['se'], se_manual[i], rtol=1e-6)
            np.testing.assert_allclose(res['coefs'][i]['z'], z_manual[i], rtol=1e-6)
            np.testing.assert_allclose(res['coefs'][i]['p'], p_manual[i], rtol=1e-6)

        # --- Verify formatting matches Standard convention ---
        formatted = format_logit(res)
        self.assertIn("Std. err.", formatted)
        self.assertNotIn("Robust std. err.", formatted)
        self.assertIn("z", formatted)
        self.assertIn("P>|z|", formatted)
        self.assertNotIn("P>|t|", formatted)
        self.assertIn(f"Std. err. adjusted for {G} clusters in cid", formatted)

if __name__ == '__main__':
    unittest.main()
