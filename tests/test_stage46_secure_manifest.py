"""
Stage 46 — Release Manifest Integrity, Adversarial File Rejection,
and Hardened Post-Build Verification Test Suite.
Verifies all 12 mandatory negative verification cases and fail-closed manifest rules
with and without Git metadata.
"""

import os
import json
import zipfile
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.create_clean_release_archive import (
    create_release_archive,
    verify_release_archive,
    compare_release_archives,
    MANDATORY_RELEASE_FILES,
    load_approved_manifest
)


class TestStage46SecureManifest(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.tmp_dir.name).resolve()

        # Create minimal valid repository matching mandatory files
        for rel in MANDATORY_RELEASE_FILES:
            target = self.root_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"# Content for {rel}\n", encoding="utf-8")

        # Create valid release_manifest.json
        manifest_data = {
            "manifest_version": "1.0",
            "mandatory_files": sorted(list(MANDATORY_RELEASE_FILES)),
            "approved_files": sorted(list(MANDATORY_RELEASE_FILES))
        }
        (self.root_path / "release_manifest.json").write_text(
            json.dumps(manifest_data, indent=2), encoding="utf-8"
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    # =========================================================================
    # Phase 2: Adversarial File Rejection
    # =========================================================================

    def test_01_adversarial_unauthorized_files_rejected_with_and_without_git(self):
        """Verifies synthetic unauthorized files under all subsystems are strictly rejected."""
        unauthorized_rel_paths = [
            "engine/temp_hack.py",
            "gui/assets/unauth.svg",
            "gui_qt/unauth.json",
            "tests/unauth.py",
            "scripts/unauth.bat",
            "engine/assets/data/unauth.csv"
        ]
        for rel in unauthorized_rel_paths:
            p = self.root_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("synthetic unauthorized test content", encoding="utf-8")

        for has_git in (False, True):
            git_dir = self.root_path / ".git"
            if has_git:
                git_dir.mkdir(exist_ok=True)
            elif git_dir.exists():
                git_dir.rmdir()

            zip_out = self.root_path / f"test_release_{'git' if has_git else 'nogit'}.zip"
            created_zip = create_release_archive(
                root_dir=str(self.root_path),
                output_zip=str(zip_out),
                strict=True
            )

            # Inspect zip contents
            with zipfile.ZipFile(created_zip, "r") as zf:
                namelist = zf.namelist()
                for unauth in unauthorized_rel_paths:
                    self.assertNotIn(
                        unauth, namelist,
                        f"Unauthorized file '{unauth}' was packaged (has_git={has_git})"
                    )

            # Verification against approved manifest must succeed and confirm 0 unauth
            res = verify_release_archive(created_zip, root_path=self.root_path, test_extraction=True)
            self.assertTrue(res["verified"], f"Verification failed: {res.get('mismatches')}")

    def test_02_missing_mandatory_file_fails_creation(self):
        """Verifies that creation aborts if any mandatory file is missing on disk."""
        # Delete a mandatory file
        (self.root_path / "main.py").unlink()
        zip_out = self.root_path / "should_fail.zip"
        with self.assertRaises(FileNotFoundError):
            create_release_archive(
                root_dir=str(self.root_path),
                output_zip=str(zip_out),
                strict=True
            )

    # =========================================================================
    # Phase 3: Mandatory Negative Verification Cases (1 to 12)
    # =========================================================================

    def _create_synthetic_zip(self, members: dict, manifest_dict: dict = None) -> Path:
        """Helper to create synthetic ZIP with specified members and embedded manifest."""
        zip_path = self.root_path / f"synthetic_{os.urandom(4).hex()}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if manifest_dict is not None:
                zf.writestr("ARCHIVE_CONTENTS.json", json.dumps(manifest_dict, indent=2))
            for name, content in members.items():
                if isinstance(content, str):
                    content = content.encode("utf-8")
                zf.writestr(name, content)
        return zip_path

    def test_03_case1_only_readme_and_manifest(self):
        """Case 1: Archive containing only README.md and a matching manifest."""
        members = {"README.md": b"Only Readme"}
        manifest = {
            "files": [{"path": "README.md", "size_bytes": 11, "sha256": hashlib.sha256(b"Only Readme").hexdigest()}]
        }
        zip_p = self._create_synthetic_zip(members, manifest)
        res = verify_release_archive(str(zip_p), root_path=self.root_path)
        self.assertFalse(res["verified"])
        self.assertTrue(any("Mandatory release file missing" in m for m in res["mismatches"]))

    def test_04_case2_missing_application_entrypoint(self):
        """Case 2: Archive missing main.py."""
        approved, mandatory = load_approved_manifest(self.root_path)
        # Archive has all except main.py
        members = {m: b"content" for m in mandatory if m != "main.py"}
        manifest_files = [
            {"path": m, "size_bytes": 7, "sha256": hashlib.sha256(b"content").hexdigest()}
            for m in members
        ]
        zip_p = self._create_synthetic_zip(members, {"files": manifest_files})
        res = verify_release_archive(str(zip_p), root_path=self.root_path)
        self.assertFalse(res["verified"])
        self.assertIn("Mandatory release file missing from archive: main.py", res["mismatches"])

    def test_05_case3_missing_core_engine_module(self):
        """Case 3: Archive missing engine/interpreter.py."""
        approved, mandatory = load_approved_manifest(self.root_path)
        members = {m: b"content" for m in mandatory if m != "engine/interpreter.py"}
        manifest_files = [
            {"path": m, "size_bytes": 7, "sha256": hashlib.sha256(b"content").hexdigest()}
            for m in members
        ]
        zip_p = self._create_synthetic_zip(members, {"files": manifest_files})
        res = verify_release_archive(str(zip_p), root_path=self.root_path)
        self.assertFalse(res["verified"])
        self.assertIn("Mandatory release file missing from archive: engine/interpreter.py", res["mismatches"])

    def test_06_case4_missing_required_asset(self):
        """Case 4: Archive missing engine/assets/data/auto.dta."""
        approved, mandatory = load_approved_manifest(self.root_path)
        members = {m: b"content" for m in mandatory if m != "engine/assets/data/auto.dta"}
        manifest_files = [
            {"path": m, "size_bytes": 7, "sha256": hashlib.sha256(b"content").hexdigest()}
            for m in members
        ]
        zip_p = self._create_synthetic_zip(members, {"files": manifest_files})
        res = verify_release_archive(str(zip_p), root_path=self.root_path)
        self.assertFalse(res["verified"])
        self.assertIn("Mandatory release file missing from archive: engine/assets/data/auto.dta", res["mismatches"])

    def test_07_case5_unauthorized_csv(self):
        """Case 5: Archive containing unauthorized CSV (not in approved manifest)."""
        approved, mandatory = load_approved_manifest(self.root_path)
        members = {m: b"content" for m in mandatory}
        members["engine/assets/data/unauth.csv"] = b"col1,col2\n1,2\n"
        manifest_files = [
            {"path": k, "size_bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()}
            for k, v in members.items()
        ]
        zip_p = self._create_synthetic_zip(members, {"files": manifest_files})
        res = verify_release_archive(str(zip_p), root_path=self.root_path)
        self.assertFalse(res["verified"])
        self.assertTrue(any("Unauthorized file in zip archive" in m for m in res["mismatches"]))

    def test_08_case6_unauthorized_python_file(self):
        """Case 6: Archive containing unauthorized python file."""
        approved, mandatory = load_approved_manifest(self.root_path)
        members = {m: b"content" for m in mandatory}
        members["engine/temp_hack.py"] = b"print('hack')\n"
        manifest_files = [
            {"path": k, "size_bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()}
            for k, v in members.items()
        ]
        zip_p = self._create_synthetic_zip(members, {"files": manifest_files})
        res = verify_release_archive(str(zip_p), root_path=self.root_path)
        self.assertFalse(res["verified"])
        self.assertTrue(any("Unauthorized file in zip archive" in m for m in res["mismatches"]))

    def test_09_case7_duplicate_zip_member(self):
        """Case 7: Archive containing duplicate zip members."""
        zip_path = self.root_path / "dup.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("ARCHIVE_CONTENTS.json", json.dumps({"files": []}))
            zf.writestr("main.py", b"# content 1")
            zf.writestr("main.py", b"# content 2")
        res = verify_release_archive(str(zip_path), root_path=self.root_path)
        self.assertFalse(res["verified"])
        self.assertIn("Duplicate paths detected in zip archive", res["mismatches"])

    def test_case_insensitive_member_collision(self):
        # Both spellings can coexist in ZIP, but extract to the same Windows file.
        zip_path = self.root_path / "case-conflict.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("ARCHIVE_CONTENTS.json", json.dumps({"files": []}))
            zf.writestr("release_manifest.json", b"{}")
            zf.writestr("RELEASE_MANIFEST.json", b"{}")
        result = verify_release_archive(str(zip_path), root_path=self.root_path, test_extraction=True)
        self.assertFalse(result["verified"])
        self.assertIn("Case-insensitive path collision in zip archive", result["mismatches"])

    def test_10_case8_incorrect_checksum(self):
        """Case 8: Archive member with corrupted/tampered bytes differing from manifest."""
        approved, mandatory = load_approved_manifest(self.root_path)
        members = {m: b"content" for m in mandatory}
        manifest_files = [
            {"path": m, "size_bytes": 7, "sha256": "0" * 64}  # Bad SHA-256!
            for m in members
        ]
        zip_p = self._create_synthetic_zip(members, {"files": manifest_files})
        res = verify_release_archive(str(zip_p), root_path=self.root_path)
        self.assertFalse(res["verified"])
        self.assertTrue(any("SHA-256 hash mismatch" in m for m in res["mismatches"]))

    def test_11_case9_incorrect_file_size(self):
        """Case 9: Archive member with byte size differing from manifest."""
        approved, mandatory = load_approved_manifest(self.root_path)
        members = {m: b"content" for m in mandatory}
        manifest_files = [
            {"path": m, "size_bytes": 99999, "sha256": hashlib.sha256(b"content").hexdigest()}  # Bad size!
            for m in members
        ]
        zip_p = self._create_synthetic_zip(members, {"files": manifest_files})
        res = verify_release_archive(str(zip_p), root_path=self.root_path)
        self.assertFalse(res["verified"])
        self.assertTrue(any("Size mismatch" in m for m in res["mismatches"]))

    def test_12_case10_traversal_path(self):
        """Case 10: Archive containing path traversal member."""
        members = {"../outside.txt": b"escaped"}
        manifest = {"files": [{"path": "../outside.txt", "size_bytes": 7, "sha256": hashlib.sha256(b"escaped").hexdigest()}]}
        zip_p = self._create_synthetic_zip(members, manifest)
        res = verify_release_archive(str(zip_p), root_path=self.root_path)
        self.assertFalse(res["verified"])
        self.assertTrue(any("traversal detected" in m for m in res["mismatches"]))

    def test_13_case11_malformed_manifest(self):
        """Case 11: Archive containing malformed JSON in manifest."""
        zip_path = self.root_path / "bad_json.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("ARCHIVE_CONTENTS.json", b"{ malformed json string !!!")
            zf.writestr("main.py", b"# content")
        res = verify_release_archive(str(zip_path), root_path=self.root_path)
        self.assertFalse(res["verified"])
        self.assertTrue(any("Malformed manifest" in m for m in res["mismatches"]))

    def test_14_case12_valid_manifest_with_unexpected_extra_file(self):
        """Case 12: Archive with valid manifest but unlisted extra file in zip."""
        approved, mandatory = load_approved_manifest(self.root_path)
        members = {m: b"content" for m in mandatory}
        members["unexpected_file.txt"] = b"unexpected"
        # Manifest does not list unexpected_file.txt
        manifest_files = [
            {"path": m, "size_bytes": 7, "sha256": hashlib.sha256(b"content").hexdigest()}
            for m in mandatory
        ]
        zip_p = self._create_synthetic_zip(members, {"files": manifest_files})
        res = verify_release_archive(str(zip_p), root_path=self.root_path)
        self.assertFalse(res["verified"])
        self.assertTrue(any("Unauthorized file in zip" in m for m in res["mismatches"]))

    # =========================================================================
    # Stage 47: Standalone Verification & Repackaging Self-Containment
    # =========================================================================

    def test_15_standalone_verification_requires_explicit_policy(self):
        """Verifies that standalone verification without trusted manifest policy returns NOT_VERIFIED."""
        zip_out = self.root_path / "test_standalone.zip"
        created_zip = create_release_archive(
            root_dir=str(self.root_path),
            output_zip=str(zip_out),
            strict=True
        )
        # Calling standalone without root_path, trusted_manifest_path, or manifest sets
        res = verify_release_archive(created_zip)
        self.assertFalse(res["verified"])
        self.assertEqual(res.get("status"), "NOT_VERIFIED")
        self.assertIn("Missing trusted release manifest policy", res["mismatches"])

    def test_16_standalone_verification_with_trusted_manifest_path(self):
        """Verifies that standalone verification succeeds when an explicit trusted manifest path is provided."""
        zip_out = self.root_path / "test_standalone_trusted.zip"
        created_zip = create_release_archive(
            root_dir=str(self.root_path),
            output_zip=str(zip_out),
            strict=True
        )
        trusted_manifest = self.root_path / "release_manifest.json"
        res = verify_release_archive(created_zip, trusted_manifest_path=str(trusted_manifest), test_extraction=True)
        self.assertTrue(res["verified"])
        self.assertEqual(res.get("status"), "VERIFIED")

    def test_17_release_provenance_metadata(self):
        """Verifies the archive inventory contains complete provenance and environment metadata."""
        zip_out = self.root_path / "test_provenance.zip"
        created_zip = create_release_archive(
            root_dir=str(self.root_path),
            output_zip=str(zip_out),
            strict=True
        )
        with zipfile.ZipFile(created_zip, "r") as zf:
            manifest_raw = zf.read("ARCHIVE_CONTENTS.json")
            manifest = json.loads(manifest_raw.decode("utf-8"))

        self.assertIn("build_environment", manifest)
        self.assertIn("source_manifest_sha256", manifest)
        self.assertIn("git_dirty", manifest)
        self.assertIn("build_type", manifest)
        self.assertIn("os", manifest["build_environment"])
        self.assertIn("python_version", manifest["build_environment"])
        self.assertIn("untracked_files", manifest)
        self.assertIn("modified_files", manifest)

    def test_18_extracted_archive_repackaging_self_containment(self):
        """Verifies extracting an archive into a clean directory preserves release_manifest.json and repackages identically."""
        zip_out1 = self.root_path / "initial_release.zip"
        created_zip1 = create_release_archive(
            root_dir=str(self.root_path),
            output_zip=str(zip_out1),
            strict=True
        )

        with tempfile.TemporaryDirectory() as extract_dir:
            extract_path = Path(extract_dir).resolve()
            with zipfile.ZipFile(created_zip1, "r") as zf:
                self.assertEqual(len(zf.namelist()), len({p.casefold() for p in zf.namelist()}))
                inventory = zf.read("ARCHIVE_CONTENTS.json")
                zf.extractall(extract_path)

            self.assertEqual((extract_path / "release_manifest.json").read_bytes(),
                             (self.root_path / "release_manifest.json").read_bytes())
            self.assertEqual((extract_path / "ARCHIVE_CONTENTS.json").read_bytes(), inventory)

            # Repackage from extracted directory
            zip_out2 = extract_path / "repackaged_release.zip"
            created_zip2 = create_release_archive(
                root_dir=str(extract_path),
                output_zip=str(zip_out2),
                strict=True
            )

            # Inspect member count and names of both
            with zipfile.ZipFile(created_zip1, "r") as zf1, zipfile.ZipFile(created_zip2, "r") as zf2:
                names1 = sorted(zf1.namelist())
                names2 = sorted(zf2.namelist())
                self.assertEqual(names1, names2)
                self.assertEqual(len(names1), len(names2))

    def test_19_byte_identical_reproducibility_with_controlled_metadata(self):
        """Verifies byte-for-byte identical ZIP archive SHA-256 with identical inputs and controlled metadata."""
        zip_out1 = self.root_path / "repro1.zip"
        zip_out2 = self.root_path / "repro2.zip"

        meta_override = {
            "created_at": "2026-01-01T00:00:00Z",
            "git_commit": "0123456789abcdef",
            "git_dirty": False,
            "build_type": "clean_git_release",
            "provenance_source": "git_repository"
        }

        created1 = create_release_archive(
            root_dir=str(self.root_path),
            output_zip=str(zip_out1),
            strict=True,
            normalize_timestamps=True,
            metadata_override=meta_override
        )
        created2 = create_release_archive(
            root_dir=str(self.root_path),
            output_zip=str(zip_out2),
            strict=True,
            normalize_timestamps=True,
            metadata_override=meta_override
        )

        comparison = compare_release_archives(created1, created2)
        self.assertTrue(comparison["byte_identical"])
        self.assertTrue(comparison["files_identical"])
        self.assertEqual(comparison["archive_a_sha256"], comparison["archive_b_sha256"])
        self.assertEqual(len(comparison["file_mismatches"]), 0)
        self.assertEqual(len(comparison["provenance_diffs"]), 0)

    def test_20_rebuild_from_extracted_archive_provenance_distinction(self):
        """Verifies that rebuilding from extracted archive preserves all 177 files identically while accurately reporting provenance differences."""
        # 1. Build original archive in simulated git repo
        git_dir = self.root_path / ".git"
        git_dir.mkdir(exist_ok=True)
        zip_orig = self.root_path / "orig_release.zip"
        created_orig = create_release_archive(
            root_dir=str(self.root_path),
            output_zip=str(zip_orig),
            strict=True,
            normalize_timestamps=True
        )

        # 2. Extract into clean directory (no git metadata)
        with tempfile.TemporaryDirectory() as extract_dir:
            extract_path = Path(extract_dir).resolve()
            with zipfile.ZipFile(created_orig, "r") as zf:
                zf.extractall(extract_path)

            # 3. Rebuild from extracted directory
            zip_rebuilt = extract_path / "rebuilt_from_extract.zip"
            created_rebuilt = create_release_archive(
                root_dir=str(extract_path),
                output_zip=str(zip_rebuilt),
                strict=True,
                normalize_timestamps=True
            )

            comparison = compare_release_archives(created_orig, created_rebuilt)
            # Source files are 100% identical across both archives
            self.assertTrue(comparison["files_identical"])
            self.assertEqual(len(comparison["file_mismatches"]), 0)
            self.assertTrue(comparison["names_match"])

            # Provenance differences are honestly tracked (not falsely claiming identical git commit or status)
            self.assertFalse(comparison["byte_identical"])
            self.assertIn("provenance_source", comparison["provenance_diffs"])
            self.assertEqual(comparison["provenance_diffs"]["provenance_source"]["archive_a"], "git_repository")
            self.assertEqual(comparison["provenance_diffs"]["provenance_source"]["archive_b"], "extracted_release")
            self.assertEqual(comparison["provenance_diffs"]["build_type"]["archive_b"], "extracted_source_build")


if __name__ == "__main__":
    unittest.main()
