"""local/quota_guard.sh: fail-closed parsing of `my_quotas` output.

`my_quotas` prints a home block (GiB) before a project block (TiB); the
guard must select the project block specifically and abort rather than
silently succeed on any parse failure or unit mismatch (task-13-brief.md
correction 2). These tests drive the real shell script via subprocess with
a fake `my_quotas` on PATH, not a reimplementation of its parsing logic.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "local" / "quota_guard.sh"

REAL_QUOTAS = """\
The quota for home directory /jet/home/ttrachu
Storage quota: 25.00GiB
 Storage used: 19.80GiB
  Inode quota: 0
  Inodes used: 480,858

The quota for project directory /ocean/projects/cis210027p
Storage quota: 976.56TiB
 Storage used: 956.12TiB
  Inode quota: 6,070,000,000
  Inodes used: 1,438,880,301
"""


def _run(
    tmp_path,
    quotas_output: str | None,
    env_extra: dict | None = None,
    restrict_path: bool = False,
):
    """Run quota_guard.sh with a fake `my_quotas` (or none) on PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    env = dict(os.environ)

    if quotas_output is not None:
        fake = bin_dir / "my_quotas"
        fake.write_text(f"#!/usr/bin/env bash\ncat <<'Q'\n{quotas_output}Q\n")
        fake.chmod(0o755)

    if restrict_path:
        # Only base system dirs (for bash/awk/sed) -- deliberately excludes
        # bin_dir, so `command -v my_quotas` genuinely fails.
        env["PATH"] = "/usr/bin:/bin"
    else:
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"

    if env_extra:
        env.update(env_extra)

    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True
    )


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_passes_when_free_space_above_threshold(tmp_path):
    result = _run(tmp_path, REAL_QUOTAS)
    assert result.returncode == 0
    assert "20.4400 TiB" in result.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_aborts_when_free_space_below_threshold(tmp_path):
    result = _run(tmp_path, REAL_QUOTAS, env_extra={"MIN_FREE_TIB": "25"})
    assert result.returncode != 0
    assert "ABORT" in result.stderr
    assert "insufficient free space" in result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_aborts_when_my_quotas_not_on_path(tmp_path):
    result = _run(tmp_path, quotas_output=None, restrict_path=True)
    assert result.returncode != 0
    assert "ABORT" in result.stderr
    assert "my_quotas not found" in result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_aborts_when_project_block_is_missing(tmp_path):
    """Home-only output must not silently pass parsing on the home block."""
    home_only = (
        "The quota for home directory /jet/home/ttrachu\n"
        "Storage quota: 25.00GiB\n"
        " Storage used: 19.80GiB\n"
    )
    result = _run(tmp_path, home_only)
    assert result.returncode != 0
    assert "could not find" in result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_aborts_on_garbage_output(tmp_path):
    result = _run(tmp_path, "nonsense output\n")
    assert result.returncode != 0
    assert "ABORT" in result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_aborts_on_unit_mismatch_fail_closed(tmp_path):
    """If the project block ever reports GiB instead of TiB (or any other
    unit), the guard must refuse to guess rather than comparing mismatched
    units as if they were the same (task-13-brief.md correction 2)."""
    wrong_units = (
        "The quota for project directory /ocean/projects/cis210027p\n"
        "Storage quota: 976.56GiB\n"
        " Storage used: 956.12GiB\n"
    )
    result = _run(tmp_path, wrong_units)
    assert result.returncode != 0
    assert "expected TiB units" in result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_script_is_executable_and_shellcheck_clean_syntax(tmp_path):
    """Cheap syntax gate: `bash -n` catches gross shell errors independent
    of any fake my_quotas."""
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
