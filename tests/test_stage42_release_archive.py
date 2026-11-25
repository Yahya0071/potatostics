"""
Stage 42 — Clean Release Archive Verification Test
Validates:
1. create_release_archive script creates a valid zip archive without errors.
2. Required source, asset, and documentation files are present.
3. Package data assets in engine/assets/data/ are present.
4. Excluded artifacts (.git, __pycache__, *.pyc, .pytest_cache) are completely absent.
5. Archive file size is within professional bounds (< 20MB).
"""

import os
import tempfile
import zipfile

from scripts.create_clean_release_archive import create_release_archive


def test_clean_release_archive_creation_and_contents():
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_zip = os.path.join(tmp_dir, "potatostics-test-release.zip")
        created_path = create_release_archive(".", target_zip)

        assert os.path.exists(created_path)
        archive_size_mb = os.path.getsize(created_path) / (1024 * 1024)
        assert archive_size_mb < 20.0, f"Archive too large: {archive_size_mb:.2f} MB"

        with zipfile.ZipFile(created_path, "r") as zf:
            namelist = zf.namelist()

            # Required files must be included
            assert "main.py" in namelist
            assert "pyproject.toml" in namelist
            assert "README.md" in namelist
            assert "engine/benchmarks.py" in namelist
            assert "engine/causal_modern.py" in namelist

            # Package data assets must be present
            assert "engine/assets/data/auto.dta" in namelist
            assert "engine/assets/data/basque.csv" in namelist
            assert "engine/assets/data/mpdta.csv" in namelist

            # Strictly excluded directories and files
            for entry in namelist:
                parts = entry.split("/")
                assert ".git" not in parts, f".git entry leaked: {entry}"
                assert "__pycache__" not in parts, f"__pycache__ entry leaked: {entry}"
                assert ".pytest_cache" not in parts, f".pytest_cache entry leaked: {entry}"
                assert not entry.endswith(".pyc"), f".pyc leaked: {entry}"
                assert not entry.endswith(".pyo"), f".pyo leaked: {entry}"
                assert not entry.endswith(".DS_Store"), f".DS_Store leaked: {entry}"
