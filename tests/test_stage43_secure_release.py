import os
import tempfile
import zipfile
import json

from scripts.create_clean_release_archive import create_release_archive


def test_secure_release_archive_excludes_secrets_and_generates_manifest():
    """
    Stage 43: Verify that create_release_archive uses allowlists,
    strictly rejects secrets (.env, *.pem, credentials, tokens),
    and generates a valid archive inventory with SHA-256 hashes.
    """
    with tempfile.TemporaryDirectory() as tmp_root:
        # Construct synthetic workspace
        main_py = os.path.join(tmp_root, "main.py")
        with open(main_py, "w", encoding="utf-8") as f:
            f.write("print('potatostics')\n")

        engine_dir = os.path.join(tmp_root, "engine")
        os.makedirs(engine_dir, exist_ok=True)
        with open(os.path.join(engine_dir, "dataset.py"), "w", encoding="utf-8") as f:
            f.write("# dataset code\n")

        # Plant secrets that must NEVER enter release archive
        with open(os.path.join(tmp_root, ".env"), "w", encoding="utf-8") as f:
            f.write("SECRET_KEY=supersecret123\n")
        with open(os.path.join(tmp_root, ".env.production"), "w", encoding="utf-8") as f:
            f.write("DATABASE_URL=postgres://...\n")
        with open(os.path.join(tmp_root, "private_key.pem"), "w", encoding="utf-8") as f:
            f.write("-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n")
        with open(os.path.join(engine_dir, "credentials.json"), "w", encoding="utf-8") as f:
            f.write('{"api_token": "xyz987"}\n')
        with open(os.path.join(tmp_root, "auth_token.txt"), "w", encoding="utf-8") as f:
            f.write("bearer_token_12345\n")

        # Plant unauthorized directory
        unauth_dir = os.path.join(tmp_root, "internal_admin_tools")
        os.makedirs(unauth_dir, exist_ok=True)
        with open(os.path.join(unauth_dir, "tool.py"), "w", encoding="utf-8") as f:
            f.write("# admin\n")

        target_zip = os.path.join(tmp_root, "test_release.zip")
        created_path = create_release_archive(tmp_root, target_zip)

        assert os.path.exists(created_path)

        with zipfile.ZipFile(created_path, "r") as zf:
            namelist = zf.namelist()

            # Legitimate files must be present
            assert "main.py" in namelist
            assert "engine/dataset.py" in namelist
            assert "ARCHIVE_CONTENTS.json" in namelist

            # Secrets and unauthorized dirs must NOT be present
            for entry in namelist:
                assert ".env" not in entry, f"Leaked env file: {entry}"
                assert not entry.endswith(".pem"), f"Leaked pem file: {entry}"
                assert not entry.endswith(".key"), f"Leaked key file: {entry}"
                assert "credentials.json" not in entry, f"Leaked credentials: {entry}"
                assert "auth_token" not in entry, f"Leaked token: {entry}"
                assert "internal_admin_tools" not in entry, f"Leaked unapproved directory: {entry}"

            # Verify manifest integrity
            manifest_data = json.loads(zf.read("ARCHIVE_CONTENTS.json").decode("utf-8"))
            assert "files" in manifest_data
            assert "total_files" in manifest_data
            assert "sha256" in manifest_data["files"][0]
            assert manifest_data["total_files"] == len(manifest_data["files"])
