import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_cli_doctor():
    r = subprocess.run(
        [sys.executable, "-m", "mmforge.cli", "--doctor"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "numpy" in r.stdout.lower()


def test_cli_run_smoke():
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "mmforge.cli",
            "--seeds",
            "1",
            "--epochs",
            "5",
            "--regimes",
            "linear",
            "--out",
            "/tmp/mmforge_cli_smoke.json",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "summary" in (r.stdout + r.stderr) or os.path.exists("/tmp/mmforge_cli_smoke.json")
