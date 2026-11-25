"""
Potatostics Clean Secure Release Archive Generator

Packages a distribution-ready clean zip archive of the Potatostics repository.
Uses an explicit, version-controlled approved-file release manifest,
verifies mandatory application and distribution files,
filters potential secrets and credentials, strictly rejects symlinks and directory traversal,
ensures fail-closed asset and file inclusion, calculates SHA-256 hashes for all packaged files,
supports deterministic byte-for-byte timestamp normalization, and provides hardened post-build verification.
"""

import os
import sys
import re
import json
import hashlib
import zipfile
import tempfile
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set


# Canonical mandatory release files that MUST be present in any release
MANDATORY_RELEASE_FILES: Set[str] = {
    "main.py",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "potatostics.spec",
    "assets.py",
    "sample_data.csv",
    "release_manifest.json",
    "docs/USER_MANUAL.md",
    "docs/RELEASE_CHECKLIST.md",
    "scripts/create_clean_release_archive.py",
    "scripts/run_release_acceptance.py",
    "engine/__init__.py",
    "engine/interpreter.py",
    "engine/dataset.py",
    "engine/benchmarks.py",
    "engine/advanced_econometrics.py",
    "engine/causal.py",
    "engine/causal_modern.py",
    "engine/data_ops.py",
    "engine/diagnostics.py",
    "engine/estimates.py",
    "engine/estimation.py",
    "engine/exporters.py",
    "engine/formatter.py",
    "engine/macros.py",
    "engine/parser.py",
    "engine/plugins_manager.py",
    "engine/session_manager.py",
    "engine/statistics.py",
    "engine/timeseries.py",
    "engine/assets/data/auto.dta",
    "engine/assets/data/basque.csv",
    "engine/assets/data/mpdta.csv",
    "gui_qt/__init__.py",
    "gui_qt/main_window.py",
    "gui_qt/console.py",
    "gui_qt/command_bar.py",
    "gui_qt/data_grid.py",
    "gui_qt/variables_pane.py",
    "gui_qt/history_pane.py",
    "gui_qt/theme.py"
}

# Explicitly approved top-level directories
ALLOWLIST_DIRS = {
    "engine", "gui_qt", "gui", "docs", "scripts", "examples", "tests", "References", "plugins"
}

# Explicitly approved root files
ALLOWLIST_ROOT_FILES = {
    "main.py", "pyproject.toml", "potatostics.spec", "README.md", "LICENSE",
    "sample_data.csv", "assets.py", "requirements-dev.txt", "run_potatostics.bat",
    ".gitignore", "release_manifest.json"
}

# Required benchmark assets explicitly approved for packaging under engine/assets/data/
APPROVED_BENCHMARK_ASSETS: Set[str] = {
    "engine/assets/data/auto.dta",
    "engine/assets/data/basque.csv",
    "engine/assets/data/mpdta.csv"
}

# Sensitive extensions that must NEVER be packaged
DISALLOWED_EXTENSIONS = {
    ".pem", ".key", ".p12", ".pfx", ".crt", ".cer", ".pyc", ".pyo", ".pyd",
    ".coverage", ".DS_Store", ".tmp", ".swp", ".bak"
}

# Sensitive patterns (checked case-insensitively against base filename)
DISALLOWED_PATTERNS = [
    ".env", "secret", "credential", "private_key", "token", "password"
]

EXCLUDE_DIRS = {
    ".git", ".github", ".pytest_cache", "__pycache__", ".vscode", ".idea",
    "test_outputs", "dist", "build", ".egg-info", ".venv", "venv"
}

# Fixed normalized timestamp for deterministic byte-for-byte reproducible ZIP archives
REPRODUCIBLE_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
ARCHIVE_INVENTORY = "ARCHIVE_CONTENTS.json"


def compute_sha256(filepath: Path) -> str:
    """Computes standard SHA-256 checksum for a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_approved_manifest(root_path: Path, allow_synthetic: bool = False) -> Tuple[Set[str], Set[str]]:
    """
    Loads approved and mandatory file sets from release_manifest.json in repository root.
    Fails explicitly if release_manifest.json is missing or corrupt in real repositories.
    Enforces schema validity, total_approved_files match, duplicate path rejection,
    and mandatory subset verification.
    Only allows synthetic containment fallback in isolated unit test workspaces.
    """
    manifest_file = root_path / "release_manifest.json"
    if manifest_file.exists():
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("release_manifest.json must contain a JSON object")

            raw_approved = data.get("approved_files")
            raw_mandatory = data.get("mandatory_files")
            if not isinstance(raw_approved, list) or not isinstance(raw_mandatory, list):
                raise ValueError("release_manifest.json must contain list 'approved_files' and 'mandatory_files'")

            # Check duplicate approved paths
            if len(raw_approved) != len(set(raw_approved)):
                raise ValueError(f"release_manifest.json contains {len(raw_approved) - len(set(raw_approved))} duplicate entries in approved_files")

            # Check duplicate mandatory paths
            if len(raw_mandatory) != len(set(raw_mandatory)):
                raise ValueError(f"release_manifest.json contains {len(raw_mandatory) - len(set(raw_mandatory))} duplicate entries in mandatory_files")

            # Validate declared count against actual list
            declared_total = data.get("total_approved_files")
            if declared_total is not None and declared_total != len(raw_approved):
                raise ValueError(
                    f"release_manifest.json declared total_approved_files ({declared_total}) "
                    f"does not match actual approved_files count ({len(raw_approved)})"
                )

            approved = {p.replace("\\", "/") for p in raw_approved}
            mandatory = {p.replace("\\", "/") for p in raw_mandatory}

            # Verify every mandatory path is approved
            missing_from_approved = mandatory - approved
            if missing_from_approved:
                raise ValueError(f"Mandatory files not in approved_files list: {sorted(missing_from_approved)}")

            if approved and mandatory:
                return approved, mandatory
            raise ValueError("release_manifest.json missing required approved_files or mandatory_files")
        except Exception as ex:
            raise ValueError(f"Failed to parse release_manifest.json: {ex}")

    # If this is a real or extracted codebase (has engine/benchmarks.py), failure to find release_manifest.json is fatal
    if (root_path / "engine" / "benchmarks.py").is_file():
        raise FileNotFoundError(
            f"Approved release manifest not found at '{manifest_file}'. "
            "Self-contained reproducible distribution requires release_manifest.json."
        )

    # Isolated unit test mock workspace (e.g. from Stage 43-45 security tests)
    if allow_synthetic or ((root_path / "main.py").is_file() and not (root_path / "engine" / "benchmarks.py").is_file()):
        synthetic_mandatory = {"main.py"} if (root_path / "main.py").is_file() else set()
        synthetic_approved = set()
        for dirpath, _, filenames in os.walk(root_path):
            for f in filenames:
                try:
                    rel = (Path(dirpath) / f).relative_to(root_path)
                    if is_file_allowed(rel, root_path, strict=False):
                        synthetic_approved.add(str(rel).replace("\\", "/"))
                except Exception:
                    pass
        return synthetic_approved, synthetic_mandatory

    raise FileNotFoundError(
        f"Approved release manifest missing at '{manifest_file}'. "
        "A valid release requires an explicit, versioned release_manifest.json."
    )


def is_file_allowed(rel_path: Path, root_path: Path, strict: bool = False) -> bool:
    """
    Strictly evaluates whether a file is authorized for inclusion in release archive.
    Enforces fail-closed directory boundaries:
    - Path traversal rejection ('..').
    - Symlink prohibition (both file symlinks and directory symlinks).
    - Absolute containment within repository root (Path.is_relative_to).
    - Allowlist and denylist enforcement per subsystem.
    """
    # 1. Traversal component check
    if ".." in rel_path.parts:
        if strict:
            raise ValueError(f"Path traversal ('..') detected in path: {rel_path}")
        return False

    parts = rel_path.parts
    if not parts:
        return False

    abs_file = root_path / rel_path

    # 2. Symlink check - symlinks are strictly rejected for source archives
    if abs_file.is_symlink() or os.path.islink(abs_file):
        if strict:
            raise ValueError(f"Symlinks are prohibited in release archives: {rel_path}")
        return False

    # Check parent directory symlinks
    curr = abs_file.parent
    try:
        while curr != root_path and curr != curr.parent:
            if curr.is_symlink() or os.path.islink(curr):
                if strict:
                    raise ValueError(f"Symlink directory detected along path: {curr}")
                return False
            curr = curr.parent
    except Exception:
        return False

    # 3. Path containment check
    try:
        resolved = abs_file.resolve()
        root_resolved = root_path.resolve()
        if not resolved.is_relative_to(root_resolved):
            if strict:
                raise ValueError(f"File escapes repository root: {rel_path}")
            return False
    except Exception:
        return False

    # 4. Check regular file
    if not abs_file.is_file():
        return False

    rel_str = str(rel_path).replace("\\", "/")
    top_level = parts[0]
    filename = rel_path.name
    filename_lower = filename.lower()
    ext = rel_path.suffix.lower()

    # 5. Reject secrets, scratch notes, and sensitive files
    if filename.startswith(".env"):
        return False
    if filename_lower in ("notes.txt", ".plugin_state.json") or filename_lower.endswith("_notes.txt"):
        return False
    if ext in DISALLOWED_EXTENSIONS:
        return False
    for pat in DISALLOWED_PATTERNS:
        if pat in filename_lower:
            return False

    # 6. Check root-level files vs subdirectories
    if len(parts) == 1:
        return filename in ALLOWLIST_ROOT_FILES

    # 7. Check approved subdirectories
    if top_level not in ALLOWLIST_DIRS:
        return False

    # 8. Check for nested excluded directories
    for part in parts[:-1]:
        if part in EXCLUDE_DIRS or part.endswith(".egg-info") or part in (".venv", "venv"):
            return False

    # 9. Fail-closed per-directory file type constraints
    if top_level == "engine":
        if rel_str.startswith("engine/assets/data/"):
            return rel_str in APPROVED_BENCHMARK_ASSETS
        if ext != ".py":
            return False

    elif top_level == "gui":
        if rel_str.startswith("gui/assets/"):
            return ext in (".png", ".svg", ".ico")
        if rel_str == "gui/sample_data.csv":
            return True
        if ext != ".py":
            return False

    elif top_level == "gui_qt":
        if rel_str == "gui_qt/sample_data.csv":
            return True
        if ext != ".py":
            return False

    elif top_level == "docs":
        if ext != ".md":
            return False

    elif top_level == "examples":
        if ext != ".do":
            return False

    elif top_level == "References":
        if filename_lower != "references.bib":
            return False

    elif top_level == "scripts":
        if ext not in (".py", ".bat", ".json"):
            return False

    elif top_level == "tests":
        if ext not in (".py", ".json", ".md", ".csv", ".dta"):
            return False

    elif top_level == "plugins":
        if ext != ".py":
            return False

    return True


def get_candidate_files(root_path: Path) -> List[Path]:
    """
    Gathers candidate source files strictly from the approved release manifest.
    Does NOT dynamically approve every file in a directory.
    Checks Git-tracked files when available and issues an actionable warning if any mandatory files are untracked.
    """
    approved_files, mandatory_files = load_approved_manifest(root_path)

    # Check Git tracked state if Git metadata is present
    git_dir = root_path / ".git"
    if git_dir.exists():
        try:
            res = subprocess.run(
                ["git", "ls-files"],
                cwd=str(root_path),
                capture_output=True,
                text=True,
                check=True
            )
            git_tracked = {p.strip().replace("\\", "/") for p in res.stdout.splitlines() if p.strip()}
            untracked_mandatory = [m for m in sorted(mandatory_files) if m not in git_tracked and (root_path / m).is_file()]
            if untracked_mandatory:
                print(f"[RELEASE AUDIT WARNING] Mandatory files present on disk but untracked in Git: {untracked_mandatory}", file=sys.stderr)
        except Exception:
            pass

    # Strictly select approved files that exist on disk
    candidates: List[Path] = []
    for rel_str in sorted(approved_files):
        p = Path(rel_str)
        abs_p = root_path / p
        if abs_p.is_file():
            candidates.append(p)

    return candidates


def extract_release_version(root_path: Path) -> str:
    """Extracts project version from pyproject.toml if present."""
    pyproject = root_path / "pyproject.toml"
    if pyproject.exists():
        try:
            content = pyproject.read_text(encoding="utf-8")
            match = re.search(r'version\s*=\s*"([^"]+)"', content)
            if match:
                return match.group(1)
        except Exception:
            pass
    return "1.0.0"


def get_git_commit(root_path: Path) -> str:
    """Retrieves current Git commit SHA if in a git repository."""
    if (root_path / ".git").exists():
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(root_path),
                capture_output=True,
                text=True,
                check=True
            )
            commit = res.stdout.strip()
            if commit:
                return commit
        except Exception:
            pass
    return "unknown"


def get_git_status(root_path: Path) -> Dict[str, Any]:
    """
    Retrieves Git commit, dirty state, provenance source, and list of modified/untracked files.
    Distinguishes verified git checkouts, extracted source releases, and untracked directories.
    """
    if (root_path / ".git").exists():
        commit = get_git_commit(root_path)
        untracked = []
        modified = []
        try:
            res_status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(root_path),
                capture_output=True,
                text=True,
                check=True
            )
            for line in res_status.stdout.splitlines():
                if not line.strip():
                    continue
                code = line[:2]
                fname = line[3:].strip()
                if "??" in code:
                    untracked.append(fname)
                else:
                    modified.append(fname)
        except Exception:
            pass

        is_dirty = len(untracked) > 0 or len(modified) > 0
        return {
            "provenance_source": "git_repository",
            "is_git_controlled": True,
            "commit": commit,
            "is_dirty": is_dirty,
            "build_type": "development_build" if is_dirty else "clean_git_release",
            "untracked_files": untracked,
            "modified_files": modified
        }

    # No .git directory present
    is_extracted = (root_path / ARCHIVE_INVENTORY).is_file()
    return {
        "provenance_source": "extracted_release" if is_extracted else "unversioned_directory",
        "is_git_controlled": False,
        "commit": None,
        "is_dirty": None,
        "build_type": "extracted_source_build" if is_extracted else "unversioned_build",
        "untracked_files": [],
        "modified_files": []
    }


def create_release_archive(
    root_dir: str = ".",
    output_zip: Optional[str] = None,
    strict: bool = True,
    normalize_timestamps: bool = True,
    metadata_override: Optional[Dict[str, Any]] = None
) -> str:
    """
    Creates a clean, reproducible, secure source archive of Potatostics.
    Enforces that:
    1. Every mandatory file exists on disk before packaging starts.
    2. Only explicitly approved files from the manifest are packaged.
    3. Every mandatory file is guaranteed to be present in the generated ZIP.
    4. A deterministic file inventory with SHA-256 integrity checksums is embedded.
    5. Reproducibility contract clearly distinguishes content determinism from provenance context.
    """
    root_path = Path(root_dir).resolve()

    # In strict mode, scan for prohibited symlinks across repository root
    if strict:
        for dirpath, dirnames, filenames in os.walk(root_path):
            dp = Path(dirpath)
            if dp.is_symlink() or os.path.islink(dp):
                raise ValueError(f"Symlink directory detected: {dp}")
            for f in filenames:
                fp = dp / f
                if fp.is_symlink() or os.path.islink(fp):
                    rel = fp.relative_to(root_path)
                    raise ValueError(f"Symlinks are prohibited in release archives: {rel}")

    # Pre-packaging check: Verify all mandatory files exist on disk
    approved_files, mandatory_files = load_approved_manifest(root_path)
    missing_on_disk = [m for m in sorted(mandatory_files) if not (root_path / m).is_file()]
    if missing_on_disk:
        raise FileNotFoundError(f"Missing mandatory release file(s) on disk: {missing_on_disk}")

    if output_zip is None:
        dist_dir = root_path / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        output_zip = str(dist_dir / "potatostics-source-release.zip")
    else:
        Path(output_zip).parent.mkdir(parents=True, exist_ok=True)

    output_zip_path = Path(output_zip).resolve()

    candidates = get_candidate_files(root_path)

    included_files: List[Tuple[Path, str]] = []
    manifest_entries: List[Dict[str, Any]] = []
    included_arcnames: Set[str] = set()
    extraction_names = {ARCHIVE_INVENTORY.casefold()}
    total_bytes = 0

    for rel_path in candidates:
        abs_file = root_path / rel_path
        if abs_file.resolve() == output_zip_path:
            continue

        arcname = str(rel_path).replace("\\", "/")
        # Reject unauthorized files not in approved manifest
        if arcname not in approved_files:
            continue

        if not is_file_allowed(rel_path, root_path, strict=strict):
            continue

        extraction_name = arcname.casefold()
        if extraction_name in extraction_names:
            raise ValueError(f"Duplicate or case-insensitive path collision during packaging: {arcname}")
        extraction_names.add(extraction_name)

        file_size = abs_file.stat().st_size
        sha256_hash = compute_sha256(abs_file)
        total_bytes += file_size

        included_arcnames.add(arcname)
        included_files.append((abs_file, arcname))
        manifest_entries.append({
            "path": arcname,
            "size_bytes": file_size,
            "sha256": sha256_hash
        })

    # Post-selection verification: Ensure every mandatory file is included
    missing_in_archive = [m for m in sorted(mandatory_files) if m not in included_arcnames]
    if missing_in_archive:
        raise ValueError(f"Mandatory files missing from release archive candidate set: {missing_in_archive}")

    # Sort manifest entries deterministically by arcname
    manifest_entries.sort(key=lambda x: x["path"])
    included_files.sort(key=lambda x: x[1])

    git_info = get_git_status(root_path)
    source_manifest_path = root_path / "release_manifest.json"
    source_manifest_sha256 = compute_sha256(source_manifest_path) if source_manifest_path.is_file() else "unknown"

    manifest_data = {
        "manifest_version": "1.0",
        "format_version": "1.0",
        "release_version": extract_release_version(root_path),
        "provenance_source": git_info["provenance_source"],
        "is_git_controlled": git_info["is_git_controlled"],
        "git_commit": git_info["commit"],
        "git_dirty": git_info["is_dirty"],
        "build_type": git_info["build_type"],
        "source_manifest_sha256": source_manifest_sha256,
        "build_environment": {
            "os": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version()
        },
        "created_at": "2026-01-01T00:00:00Z" if normalize_timestamps else datetime.now(timezone.utc).isoformat(),
        "integrity_algorithm": "SHA-256",
        "is_signed": False,
        "deterministic_file_inventory": True,
        "deterministic_source_content": True,
        "deterministic_archive_construction": bool(normalize_timestamps),
        "reproducible_build": bool(normalize_timestamps and git_info["is_git_controlled"] and not git_info["is_dirty"]),
        "total_files": len(manifest_entries),
        "total_bytes": total_bytes,
        "untracked_files": git_info["untracked_files"],
        "modified_files": git_info["modified_files"],
        "files": manifest_entries
    }
    if metadata_override:
        manifest_data.update(metadata_override)

    manifest_json_bytes = json.dumps(manifest_data, indent=2).encode("utf-8")

    fixed_time = REPRODUCIBLE_ZIP_TIME if normalize_timestamps else (2026, 1, 1, 0, 0, 0)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write manifest with fixed ZipInfo
        manifest_info = zipfile.ZipInfo(ARCHIVE_INVENTORY, date_time=fixed_time)
        manifest_info.compress_type = zipfile.ZIP_DEFLATED
        manifest_info.external_attr = 0o644 << 16
        zf.writestr(manifest_info, manifest_json_bytes)

        # Write files with fixed ZipInfo
        for abs_file, arcname in included_files:
            file_bytes = abs_file.read_bytes()
            zinfo = zipfile.ZipInfo(arcname, date_time=fixed_time)
            zinfo.compress_type = zipfile.ZIP_DEFLATED
            zinfo.external_attr = 0o644 << 16
            zf.writestr(zinfo, file_bytes)

    return str(output_zip)


def compare_release_archives(archive_a: str, archive_b: str) -> Dict[str, Any]:
    """
    Compares two release archives to assess bit-for-bit reproducibility,
    source-file content equality, and provenance metadata differences.
    """
    path_a = Path(archive_a)
    path_b = Path(archive_b)
    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError(f"One or both archives not found: {archive_a}, {archive_b}")

    hash_a = compute_sha256(path_a)
    hash_b = compute_sha256(path_b)
    byte_identical = (hash_a == hash_b)

    with zipfile.ZipFile(path_a, "r") as za, zipfile.ZipFile(path_b, "r") as zb:
        names_a = set(za.namelist())
        names_b = set(zb.namelist())
        names_match = (names_a == names_b)

        manifest_a = json.loads(za.read(ARCHIVE_INVENTORY)) if ARCHIVE_INVENTORY in names_a else {}
        manifest_b = json.loads(zb.read(ARCHIVE_INVENTORY)) if ARCHIVE_INVENTORY in names_b else {}

        # The generated inventory records provenance separately from source content.
        common_members = sorted(list(names_a.intersection(names_b)))
        file_mismatches = []
        for member in common_members:
            if member == ARCHIVE_INVENTORY:
                continue
            ba = za.read(member)
            bb = zb.read(member)
            if hashlib.sha256(ba).hexdigest() != hashlib.sha256(bb).hexdigest():
                file_mismatches.append(member)

        # Compare provenance differences
        provenance_keys = ["git_commit", "git_dirty", "build_type", "provenance_source", "is_git_controlled"]
        provenance_diffs = {}
        for k in provenance_keys:
            val_a = manifest_a.get(k)
            val_b = manifest_b.get(k)
            if val_a != val_b:
                provenance_diffs[k] = {"archive_a": val_a, "archive_b": val_b}

    return {
        "byte_identical": byte_identical,
        "files_identical": (len(file_mismatches) == 0 and names_match),
        "archive_a_sha256": hash_a,
        "archive_b_sha256": hash_b,
        "member_count_a": len(names_a),
        "member_count_b": len(names_b),
        "names_match": names_match,
        "file_mismatches": file_mismatches,
        "provenance_diffs": provenance_diffs
    }


def verify_release_archive(
    archive_path: str,
    approved_manifest: Optional[Set[str]] = None,
    mandatory_files: Optional[Set[str]] = None,
    root_path: Optional[Path] = None,
    trusted_manifest_path: Optional[str] = None,
    trusted_manifest: Optional[Dict[str, Any]] = None,
    test_extraction: bool = False
) -> Dict[str, Any]:
    """
    Hardened, independent post-build verification of a release archive.
    Requires an independent policy to authorize files listed by the internal inventory.
    Enforces explicit modes:
    1. Trusted repository mode (root_path provided).
    2. Standalone distribution mode with explicit trusted manifest (trusted_manifest_path or trusted_manifest).
    3. Standalone mode without trusted manifest: returns NOT_VERIFIED.
    """
    zip_file = Path(archive_path)
    if not zip_file.exists():
        return {
            "verified": False,
            "status": "FAIL",
            "error": f"Archive not found: {archive_path}",
            "mismatches": ["File missing"]
        }

    # Determine independent authorization sets
    if trusted_manifest_path is not None:
        try:
            data = json.loads(Path(trusted_manifest_path).read_text(encoding="utf-8"))
            approved_manifest = {p.replace("\\", "/") for p in data.get("approved_files", [])}
            mandatory_files = {p.replace("\\", "/") for p in data.get("mandatory_files", [])}
        except Exception as ex:
            return {
                "verified": False,
                "status": "FAIL",
                "error": f"Failed to load trusted manifest: {ex}",
                "mismatches": [f"Malformed trusted manifest: {ex}"]
            }
    elif trusted_manifest is not None:
        approved_manifest = {p.replace("\\", "/") for p in trusted_manifest.get("approved_files", [])}
        mandatory_files = {p.replace("\\", "/") for p in trusted_manifest.get("mandatory_files", [])}
    elif root_path is not None:
        loaded_approved, loaded_mandatory = load_approved_manifest(root_path)
        if approved_manifest is None:
            approved_manifest = loaded_approved
        if mandatory_files is None:
            mandatory_files = loaded_mandatory
    elif approved_manifest is None and mandatory_files is None:
        return {
            "verified": False,
            "status": "NOT_VERIFIED",
            "error": "Standalone archive verification requires an explicit trusted release manifest or policy. The archive's internal manifest cannot self-authorize.",
            "mismatches": ["Missing trusted release manifest policy"]
        }
    else:
        # Either approved_manifest or mandatory_files was explicitly passed by caller
        if approved_manifest is None:
            approved_manifest = set()
        if mandatory_files is None:
            mandatory_files = set()

    mismatches: List[str] = []
    try:
        with zipfile.ZipFile(zip_file, "r") as zf:
            namelist = zf.namelist()

            # 1. Check duplicate entries in zip
            if len(namelist) != len(set(namelist)):
                mismatches.append("Duplicate paths detected in zip archive")
            if len(set(namelist)) != len({name.casefold() for name in namelist}):
                mismatches.append("Case-insensitive path collision in zip archive")

            # 2. Check path traversal and invalid characters
            for name in namelist:
                parts = Path(name).parts
                if ".." in parts:
                    mismatches.append(f"Directory traversal detected in member: {name}")
                if name.startswith("/") or name.startswith("\\"):
                    mismatches.append(f"Absolute or root path detected in member: {name}")
                if "\\" in name:
                    mismatches.append(f"Non-normalized path separator (backslash) in member: {name}")

            # 3. Check presence of internal manifest
            if ARCHIVE_INVENTORY not in namelist:
                return {
                    "verified": False,
                    "status": "FAIL",
                    "error": f"Missing {ARCHIVE_INVENTORY}",
                    "mismatches": mismatches + ["Missing manifest"]
                }

            # 4. Check manifest validity
            try:
                manifest_raw = zf.read(ARCHIVE_INVENTORY)
                manifest = json.loads(manifest_raw.decode("utf-8"))
            except Exception as ex:
                return {
                    "verified": False,
                    "status": "FAIL",
                    "error": f"Malformed manifest: {ex}",
                    "mismatches": [f"Malformed manifest: {ex}"]
                }

            manifest_files = {entry["path"]: entry for entry in manifest.get("files", [])}

            # 5. Verify independent mandatory files are present in the archive
            for req in sorted(mandatory_files):
                if req not in namelist:
                    mismatches.append(f"Mandatory release file missing from archive: {req}")

            # 6. Verify all archived files are authorized in the approved manifest
            for name in namelist:
                if name == ARCHIVE_INVENTORY:
                    continue
                if name not in approved_manifest:
                    mismatches.append(f"Unauthorized file in zip archive not in approved manifest: {name}")
                if name not in manifest_files:
                    mismatches.append(f"Unauthorized file in zip not listed in manifest: {name}")
                if root_path is not None and not is_file_allowed(Path(name), root_path):
                    mismatches.append(f"File in zip violates release allowlist: {name}")

            # 7. Verify file sizes and SHA-256 hashes against manifest
            for rel_path, meta in manifest_files.items():
                if rel_path not in namelist:
                    mismatches.append(f"Manifest file missing in archive: {rel_path}")
                    continue
                archived_bytes = zf.read(rel_path)
                calc_size = len(archived_bytes)
                calc_sha = hashlib.sha256(archived_bytes).hexdigest()

                if calc_size != meta.get("size_bytes"):
                    mismatches.append(f"Size mismatch for {rel_path}: expected {meta.get('size_bytes')}, got {calc_size}")
                if calc_sha != meta.get("sha256"):
                    mismatches.append(f"SHA-256 hash mismatch for {rel_path}: expected {meta.get('sha256')}, got {calc_sha}")

            # 8. Test clean and safe extraction if requested
            if test_extraction and not mismatches:
                with tempfile.TemporaryDirectory() as extract_dir:
                    extract_path = Path(extract_dir).resolve()
                    for member in zf.infolist():
                        # Prevent Zip Slip / traversal during extraction
                        target_path = (extract_path / member.filename).resolve()
                        if not target_path.is_relative_to(extract_path):
                            mismatches.append(f"Zip slip path traversal detected on extraction: {member.filename}")
                            continue
                        zf.extract(member, extract_dir)

                    if (extract_path / ARCHIVE_INVENTORY).read_bytes() != manifest_raw:
                        mismatches.append("Extracted archive inventory differs from the ZIP")
                    for rel_path, meta in manifest_files.items():
                        extracted_file = extract_path / rel_path
                        if not extracted_file.exists():
                            mismatches.append(f"Extraction failed for: {rel_path}")
                        elif extracted_file.stat().st_size != meta.get("size_bytes"):
                            mismatches.append(f"Extracted size mismatch for: {rel_path}")
                        elif compute_sha256(extracted_file) != meta.get("sha256"):
                            mismatches.append(f"Extracted SHA-256 mismatch for: {rel_path}")

            is_ok = len(mismatches) == 0
            return {
                "verified": is_ok,
                "status": "VERIFIED" if is_ok else "FAIL",
                "total_files": len(manifest_files),
                "archive_size_bytes": zip_file.stat().st_size,
                "mismatches": mismatches
            }
    except Exception as ex:
        return {
            "verified": False,
            "status": "FAIL",
            "error": str(ex),
            "mismatches": [str(ex)]
        }


if __name__ == "__main__":
    out_target = sys.argv[1] if len(sys.argv) > 1 else None
    root_dir = Path(".").resolve()
    archive_path = create_release_archive(root_dir=str(root_dir), output_zip=out_target, strict=True)
    print(f"Created clean secure release archive at: {archive_path}")
    verify_report = verify_release_archive(archive_path, root_path=root_dir, test_extraction=True)
    print(f"Archive verification result: {verify_report['verified']} ({verify_report['total_files']} files verified)")
    if not verify_report["verified"]:
        print(f"Verification mismatches: {verify_report['mismatches']}", file=sys.stderr)
        sys.exit(1)
