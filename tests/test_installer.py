"""install.sh: the system report, the plan, and the rules that protect the user's files.

These tests never install anything: they use ``--detect`` and ``--dry-run``, or a copy of the script in an empty
folder. The real installation is run by the continuous-integration workflow on Linux, macOS and Windows.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT

INSTALLER = ROOT / "install.sh"


def find_bash():
    if sys.platform != "win32":
        return shutil.which("bash")
    # On Windows, "bash" on the PATH can be WSL; the installer is meant for the bash of Git for Windows.
    candidates = [Path(os.environ.get(v, "")) / "Git" / "bin" / "bash.exe" for v in ("ProgramFiles", "ProgramFiles(x86)")]
    git = shutil.which("git")
    if git:
        candidates.append(Path(git).resolve().parent.parent / "bin" / "bash.exe")
    return next((str(c) for c in candidates if c.is_file()), None)


BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not found")


def run(*args, cwd=ROOT, script="install.sh"):
    return subprocess.run([BASH, script, *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", stdin=subprocess.DEVNULL, timeout=120)


def test_script_has_valid_syntax_and_unix_line_endings():
    assert b"\r\n" not in INSTALLER.read_bytes()
    assert subprocess.run([BASH, "-n", "install.sh"], cwd=ROOT).returncode == 0


def test_help_lists_every_option():
    r = run("--help")
    assert r.returncode == 0
    for option in ("--yes", "--detect", "--dry-run", "--env venv|conda|docker", "--python", "--venv", "--with",
                   "--without", "--full", "--minimal", "--install-python"):
        assert option in r.stdout, option
    assert "set -euo" not in r.stdout  # the help text stops before the code


def test_detect_reports_the_system_and_changes_nothing():
    r = run("--detect")
    assert r.returncode == 0, r.stderr
    for heading in ("Operating system", "Processor", "Free disk space", "Python (memtune needs 3.10 to 3.12)",
                    "conda", "Docker", "Recommended for this system"):
        assert heading in r.stdout, heading
    assert "Plan" not in r.stdout


def test_dry_run_shows_the_plan_without_installing(tmp_path):
    shutil.copy(INSTALLER, tmp_path / "install.sh")
    r = run("--yes", "--dry-run", "--full", cwd=tmp_path)
    if "No supported Python was found" in r.stdout:
        pytest.skip("no supported Python visible to bash on this machine")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Plan" in r.stdout and "Dry run: nothing was changed." in r.stdout
    for step in ("requirements-notebooks.txt", "requirements-dev.txt", "Precompute the demo data",
                 "Check the installation (memtune doctor)", "Run the complete test suite"):
        assert step in r.stdout, step
    assert sorted(p.name for p in tmp_path.iterdir()) == ["install.sh"]  # nothing was created


def test_minimal_plan_has_no_extras(tmp_path):
    shutil.copy(INSTALLER, tmp_path / "install.sh")
    r = run("--yes", "--dry-run", "--minimal", cwd=tmp_path)
    if "No supported Python was found" in r.stdout:
        pytest.skip("no supported Python visible to bash on this machine")
    assert r.returncode == 0
    assert "requirements-lock.txt" in r.stdout
    for absent in ("demo data", "notebooks", "requirements-dev.txt", "test suite"):
        assert absent not in r.stdout.split("Plan")[1], absent


def test_an_unusable_environment_is_never_deleted_without_asking(tmp_path):
    """A .venv made by another system (Windows next to WSL, say) must survive ``--yes``."""
    shutil.copy(INSTALLER, tmp_path / "install.sh")
    keep = tmp_path / ".venv" / "keep.txt"
    keep.parent.mkdir()
    keep.write_text("mine", encoding="utf-8")
    r = run("--yes", "--env", "venv", cwd=tmp_path)
    if "No supported Python was found" in r.stdout or "Install Python 3.12 in one of these ways" in r.stdout:
        pytest.skip("no supported Python visible to bash on this machine")
    assert r.returncode == 1
    assert "Nothing was changed." in r.stdout and "--venv .venv-" in r.stdout
    assert keep.read_text(encoding="utf-8") == "mine"


@pytest.mark.parametrize("args, message", [
    (["--frobnicate"], "unknown option"),
    (["--with", "everything"], "unknown extra"),
    (["--env", "vagrant"], "--env must be venv, conda or docker"),
    (["--python", "no-such-python-anywhere", "--env", "venv", "--yes", "--dry-run"], "does not run"),
])
def test_wrong_arguments_give_a_clear_error(args, message):
    r = run(*args)
    assert r.returncode == 1
    assert message in r.stderr


def test_old_setup_script_calls_the_installer():
    r = run("--dry-run", script="scripts/setup.sh")
    assert "memtune installer" in r.stdout
    text = (ROOT / "scripts" / "setup.sh").read_text(encoding="utf-8")
    assert "install.sh" in text and "--yes --env venv" in text


def test_powershell_entry_points_exist():
    ps1 = (ROOT / "install.ps1").read_text(encoding="utf-8")
    assert "install.sh" in ps1 and "scripts\\setup.ps1" in ps1
    assert "memtune doctor" in (ROOT / "scripts" / "setup.ps1").read_text(encoding="utf-8")
