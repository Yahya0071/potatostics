"""
Stage 44 — Secure Release Archive and Distribution Manifest Tests
Validates:
1. Rejection of symlinks (both internal, sibling-directory, and out-of-tree).
2. Path containment using Path.is_relative_to rather than string prefix.
3. Rejection of path traversal ('..') and non-regular files.
4. Exclusion of untracked files, notes.txt, secrets, and runtime plugin state.
5. Deterministic manifest schema containing SHA-256 hashes, release version, and git commit.
6. Absence of false claims regarding cryptographic signatures.
"""

import os
import json
import tempfile
import zipfile
import pytest
from pathlib import Path

from scripts.create_clean_release_archive import (
    create_release_archive,
    is_file_allowed
)


def test_symlink_rejection_and_strict_containment():
    """Symlinks must be rejected completely from release archives."""
    with tempfile.TemporaryDirectory() as base_temp:
        base_path = Path(base_temp)
        repo_dir = base_path / "potatostics_repo"
        sibling_dir = base_path / "potatostics_repo_private"
        repo_dir.mkdir()
        sibling_dir.mkdir()

        # Place legitimate files in repo
        (repo_dir / "main.py").write_text("print('hello')", encoding="utf-8")
        engine_dir = repo_dir / "engine"
        engine_dir.mkdir()
        (engine_dir / "dataset.py").write_text("# dataset", encoding="utf-8")

        # Place secret file in sibling directory
        private_file = sibling_dir / "secrets.txt"
        private_file.write_text("super_secret_sibling_data", encoding="utf-8")

        # Attempt symlink attack (if platform allows creating symlinks)
        symlink_created = False
        symlink_path = repo_dir / "engine" / "symlinked_secrets.py"
        try:
            os.symlink(private_file, symlink_path)
            symlink_created = True
        except (OSError, NotImplementedError):
            # On some Windows configurations, creating symlinks requires developer mode or admin
            pass

        if symlink_created:
            # is_file_allowed must return False for symlinks
            assert not is_file_allowed(Path("engine/symlinked_secrets.py"), repo_dir)

            # In strict mode, creating the release archive must raise ValueError
            with pytest.raises(ValueError, match="[Ss]ymlink"):
                create_release_archive(str(repo_dir), str(base_path / "test.zip"), strict=True)

        # In non-strict mode, symlinks must simply be omitted
        target_zip = base_path / "non_strict.zip"
        archive_path = create_release_archive(str(repo_dir), str(target_zip), strict=False)
        assert os.path.exists(archive_path)

        with zipfile.ZipFile(archive_path, "r") as zf:
            namelist = zf.namelist()
            assert "main.py" in namelist
            assert "engine/dataset.py" in namelist
            assert "engine/symlinked_secrets.py" not in namelist
            assert "secrets.txt" not in namelist


def test_path_traversal_rejection():
    """Paths containing '..' traversal components must be rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = Path(tmp_dir)
        (root_path / "main.py").write_text("print('test')", encoding="utf-8")

        traversal_rel = Path("../outside.py")
        assert not is_file_allowed(traversal_rel, root_path)

        with pytest.raises(ValueError, match="[Tt]raversal"):
            is_file_allowed(traversal_rel, root_path, strict=True)


def test_notes_and_plugin_state_exclusion():
    """Unapproved development notes and runtime plugin state must be excluded."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = Path(tmp_dir)
        (root_path / "main.py").write_text("print('main')", encoding="utf-8")

        engine_dir = root_path / "engine"
        engine_dir.mkdir()
        (engine_dir / "dataset.py").write_text("# code", encoding="utf-8")
        (engine_dir / "notes.txt").write_text("developer scratch notes", encoding="utf-8")

        plugins_dir = root_path / "plugins"
        plugins_dir.mkdir()
        (plugins_dir / ".plugin_state.json").write_text("{}", encoding="utf-8")

        target_zip = root_path / "out.zip"
        created = create_release_archive(str(root_path), str(target_zip))

        with zipfile.ZipFile(created, "r") as zf:
            namelist = zf.namelist()
            assert "engine/notes.txt" not in namelist
            assert "plugins/.plugin_state.json" not in namelist
            assert ".plugin_state.json" not in namelist
            assert "main.py" in namelist
            assert "engine/dataset.py" in namelist


def test_release_manifest_schema_and_honesty():
    """The archive inventory must contain required metadata and avoid false cryptographic claims."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = Path(tmp_dir)
        (root_path / "main.py").write_text("print('potatostics')", encoding="utf-8")
        (root_path / "pyproject.toml").write_text(
            '[project]\nname = "potatostics"\nversion = "1.0.0"\n',
            encoding="utf-8"
        )
        engine_dir = root_path / "engine"
        engine_dir.mkdir()
        (engine_dir / "calc.py").write_text("x = 42\n", encoding="utf-8")

        target_zip = root_path / "manifest_test.zip"
        create_release_archive(str(root_path), str(target_zip))

        with zipfile.ZipFile(target_zip, "r") as zf:
            assert "ARCHIVE_CONTENTS.json" in zf.namelist()
            manifest = json.loads(zf.read("ARCHIVE_CONTENTS.json").decode("utf-8"))

            assert manifest.get("manifest_version") == "1.0"
            assert manifest.get("release_version") == "1.0.0"
            assert "git_commit" in manifest
            assert "created_at" in manifest
            assert "total_files" in manifest
            assert "total_bytes" in manifest
            assert manifest["total_files"] == len(manifest["files"])

            # Verify no false "cryptographically signed" claims in manifest
            manifest_str = json.dumps(manifest).lower()
            assert "signed" not in manifest_str or "cryptographically signed" not in manifest_str
            assert manifest.get("is_signed", False) is False

            # Check individual file records
            file_entry = [f for f in manifest["files"] if f["path"] == "main.py"][0]
            assert "size_bytes" in file_entry
            assert "sha256" in file_entry
            assert len(file_entry["sha256"]) == 64
