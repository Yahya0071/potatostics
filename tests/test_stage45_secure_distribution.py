"""
Stage 45 — Secure Release Distribution and Fail-Closed Verification Tests
Validates:
1. Rejection of unapproved files (e.g. synthetic CSV under engine/) when Git is unavailable.
2. Elimination of untracked asset inclusion under engine/assets/data/ (only approved benchmark assets allowed).
3. Post-build release verification (verify_release_archive): checking hashes, file sizes, duplicate paths, and clean extraction.
4. Reproducible zip archive generation via timestamp normalization.
"""

import zipfile
import tempfile
from pathlib import Path

from scripts.create_clean_release_archive import (
    create_release_archive,
    verify_release_archive,
    is_file_allowed
)


def test_01_synthetic_csv_under_engine_excluded_without_git():
    """Unapproved CSV files under engine/ must be excluded when Git is not available."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = Path(tmp_dir)
        # Setup legitimate files
        (root_path / "main.py").write_text("print('test')", encoding="utf-8")
        engine_dir = root_path / "engine"
        engine_dir.mkdir()
        (engine_dir / "dataset.py").write_text("# legitimate engine file", encoding="utf-8")

        # Plant unauthorized synthetic CSV under engine/
        unintended_csv = engine_dir / "synthetic_unintended.csv"
        unintended_csv.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

        assert not is_file_allowed(Path("engine/synthetic_unintended.csv"), root_path)

        target_zip = root_path / "out_nogit.zip"
        archive_path = create_release_archive(str(root_path), str(target_zip))

        with zipfile.ZipFile(archive_path, "r") as zf:
            namelist = zf.namelist()
            assert "main.py" in namelist
            assert "engine/dataset.py" in namelist
            assert "engine/synthetic_unintended.csv" not in namelist


def test_02_unapproved_data_in_benchmark_dir_excluded():
    """Only explicit benchmark assets (auto.dta, basque.csv, mpdta.csv) may be included."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = Path(tmp_dir)
        (root_path / "main.py").write_text("print('main')", encoding="utf-8")
        data_dir = root_path / "engine" / "assets" / "data"
        data_dir.mkdir(parents=True)

        # Approved benchmark assets
        (data_dir / "auto.dta").write_text("fake dta", encoding="utf-8")
        (data_dir / "basque.csv").write_text("fake basque", encoding="utf-8")
        (data_dir / "mpdta.csv").write_text("fake mpdta", encoding="utf-8")

        # Unapproved extra data file
        (data_dir / "untracked_leak.csv").write_text("should,not,enter\n", encoding="utf-8")

        assert is_file_allowed(Path("engine/assets/data/auto.dta"), root_path)
        assert is_file_allowed(Path("engine/assets/data/basque.csv"), root_path)
        assert is_file_allowed(Path("engine/assets/data/mpdta.csv"), root_path)
        assert not is_file_allowed(Path("engine/assets/data/untracked_leak.csv"), root_path)

        target_zip = root_path / "out_data.zip"
        archive_path = create_release_archive(str(root_path), str(target_zip))

        with zipfile.ZipFile(archive_path, "r") as zf:
            namelist = zf.namelist()
            assert "engine/assets/data/auto.dta" in namelist
            assert "engine/assets/data/basque.csv" in namelist
            assert "engine/assets/data/mpdta.csv" in namelist
            assert "engine/assets/data/untracked_leak.csv" not in namelist


def test_03_verify_release_archive_success_and_tamper_detection():
    """verify_release_archive must pass on valid archive and fail if contents are tampered."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = Path(tmp_dir)
        (root_path / "main.py").write_text("print('clean')", encoding="utf-8")
        (root_path / "pyproject.toml").write_text('[project]\nname="potatostics"\nversion="1.0.0"\n', encoding="utf-8")

        target_zip = root_path / "release.zip"
        created_zip = create_release_archive(str(root_path), str(target_zip))

        # Verification must pass cleanly with test_extraction=True
        report = verify_release_archive(created_zip, root_path=root_path, test_extraction=True)
        assert report["verified"] is True
        assert report["total_files"] >= 2
        assert report["mismatches"] == []

        # Tampering test: corrupt a file in the zip
        corrupted_zip = root_path / "corrupted.zip"
        with zipfile.ZipFile(created_zip, "r") as zf_in, zipfile.ZipFile(corrupted_zip, "w") as zf_out:
            for item in zf_in.infolist():
                data = zf_in.read(item.filename)
                if item.filename == "main.py":
                    data = b"tampered content"
                zf_out.writestr(item, data)

        report_corrupt = verify_release_archive(str(corrupted_zip), root_path=root_path, test_extraction=False)
        assert report_corrupt["verified"] is False
        assert len(report_corrupt["mismatches"]) > 0


def test_04_reproducible_zip_archive_deterministic_hashes():
    """Consecutive archive builds of identical content must produce bit-for-bit identical ZIP files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = Path(tmp_dir)
        (root_path / "main.py").write_text("print('deterministic')", encoding="utf-8")
        (root_path / "pyproject.toml").write_text('[project]\nname="potatostics"\nversion="1.0.0"\n', encoding="utf-8")

        zip1 = root_path / "rel1.zip"
        zip2 = root_path / "rel2.zip"

        create_release_archive(str(root_path), str(zip1), normalize_timestamps=True)
        create_release_archive(str(root_path), str(zip2), normalize_timestamps=True)

        assert zip1.read_bytes() == zip2.read_bytes(), "Archives must be bit-for-bit identical when timestamps are normalized"
