"""
scripts/smoke_test_windows.py - Automated Release Smoke Test for Windows PyInstaller Executable
Builds the PyInstaller executable, runs it in headless offscreen mode for 10 seconds,
and verifies that it launches cleanly without exiting with an error or writing to potatostics_startup.log.
"""

import os
import sys
import time
import subprocess

STARTUP_LOG = "potatostics_startup.log"
SPEC_FILE = "potatostics.spec"
EXE_DIR = os.path.join("dist", "Potatostics")
EXE_PATH = os.path.join(EXE_DIR, "Potatostics.exe")


def build_executable():
    print(f"[*] Building PyInstaller executable using {SPEC_FILE}...")
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", SPEC_FILE]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("[-] PyInstaller build failed!")
        print("STDOUT:\n", res.stdout[-1500:])
        print("STDERR:\n", res.stderr[-1500:])
        sys.exit(1)
    
    if not os.path.exists(EXE_PATH):
        print(f"[-] Expected executable not found at: {EXE_PATH}")
        sys.exit(1)
    print(f"[+] Executable successfully built: {EXE_PATH} ({os.path.getsize(EXE_PATH):,} bytes)")


def run_smoke_test(duration=10):
    # Remove existing startup log if present
    if os.path.exists(STARTUP_LOG):
        try:
            os.remove(STARTUP_LOG)
        except Exception as e:
            print(f"[!] Warning: could not remove existing {STARTUP_LOG}: {e}")

    print(f"[*] Launching executable headless for {duration} seconds (QT_QPA_PLATFORM=offscreen)...")
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["POTATOSTICS_TESTING"] = "1"

    proc = subprocess.Popen(
        [os.path.abspath(EXE_PATH)],
        env=env,
        cwd=os.path.abspath("."),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    start_time = time.time()
    while time.time() - start_time < duration:
        ret = proc.poll()
        if ret is not None:
            stdout, stderr = proc.communicate()
            if ret != 0:
                print(f"[-] SMOKE TEST FAILED: Executable exited prematurely with exit code {ret}!")
                print("STDERR:\n", stderr)
                print("STDOUT:\n", stdout)
                sys.exit(1)
            else:
                print(f"[-] Executable exited with code 0 before {duration}s duration elapsed.")
                break
        time.sleep(1)

    # Check startup log
    if os.path.exists(STARTUP_LOG):
        with open(STARTUP_LOG, "r", encoding="utf-8", errors="ignore") as f:
            log_contents = f.read().strip()
        if log_contents:
            print(f"[-] SMOKE TEST FAILED: Errors written to {STARTUP_LOG}:")
            print(log_contents)
            proc.terminate()
            sys.exit(1)

    if proc.poll() is None:
        print(f"[+] Executable running stably for {duration} seconds with 0 errors.")
        print("[*] Cleanly terminating test instance...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    # Re-verify startup log after termination
    if os.path.exists(STARTUP_LOG):
        with open(STARTUP_LOG, "r", encoding="utf-8", errors="ignore") as f:
            log_contents = f.read().strip()
        if log_contents:
            print(f"[-] SMOKE TEST FAILED: Teardown errors logged in {STARTUP_LOG}:")
            print(log_contents)
            sys.exit(1)

    print("[+] WINDOWS SMOKE TEST PASSED: Executable built and executed cleanly.")


if __name__ == "__main__":
    build_executable()
    run_smoke_test(duration=10)
