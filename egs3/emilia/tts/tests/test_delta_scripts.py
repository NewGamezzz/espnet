"""Job scripts for NCSA Delta under local/delta/.

They live in their own directory so the PSC scripts in local/ (and the tests
that pin PSC-specific details of every local/*.sbatch) are left alone.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1]
REPO = RECIPE.parents[2]
DELTA = RECIPE / "local" / "delta"
ENV = DELTA / "env.sh"
SBATCH = sorted(DELTA.glob("*.sbatch")) if DELTA.is_dir() else []


def _source_env(extra_env: dict, script: str = "") -> subprocess.CompletedProcess:
    """Source env.sh in a clean shell and print what it exported."""
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], **extra_env}
    return subprocess.run(
        [
            "bash",
            "-c",
            f'source "{ENV}" && {script or "true"} && '
            'echo "PY=$PY" && echo "PYTHONPATH=$PYTHONPATH" && '
            'echo "ROOT=$EMILIA_CORPUS_ROOT" && echo "LAYOUT=$EMILIA_CORPUS_LAYOUT" && '
            'echo "NOUSER=$PYTHONNOUSERSITE"',
        ],
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.fixture
def site(tmp_path):
    """Site settings supplied through the environment, no site_env.sh."""
    return {
        "EMILIA_PY": "/site/python",
        "EMILIA_CORPUS_ROOT": "/site/corpus",
        "EMILIA_SITE_ENV": str(tmp_path / "absent_site_env.sh"),
    }


def test_delta_scripts_exist():
    names = {p.name for p in SBATCH}
    assert {
        "prepare_data.sbatch",
        "probe_memory.sbatch",
        "stage_emilia.sbatch",
        "train_trial.sbatch",
    } <= names


def _sbatch_option(text: str, name: str) -> str:
    match = re.search(rf"^#SBATCH --{name}=(\S+)$", text, re.MULTILINE)
    assert match, f"no #SBATCH --{name}"
    return match.group(1)


def _code(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def test_trial_job_runs_the_trial_config_once_on_the_interactive_partition():
    """The trial is one job: it resumes nothing and queues no successor."""
    text = (DELTA / "train_trial.sbatch").read_text(encoding="utf-8")
    code = _code(text)
    assert _sbatch_option(text, "partition") == "gpuA100x4-interactive"
    assert "CONF=${1:-conf/training_f5_tts_base_delta_trial.yaml}" in code
    assert 'run.py --stages train --training_config "$CONF"' in code
    assert "--ckpt_path" not in code
    assert not re.search(r"\bsbatch\b", code)
    # one dead rank must end the job instead of idling the others in NCCL
    assert "srun --kill-on-bad-exit=1 " in code


def test_trial_job_has_one_task_per_device_of_its_config():
    """Lightning takes the rank count from Slurm; it must match the config."""
    from omegaconf import OmegaConf

    text = (DELTA / "train_trial.sbatch").read_text(encoding="utf-8")
    cfg = OmegaConf.load(RECIPE / "conf" / "training_f5_tts_base_delta_trial.yaml")
    assert int(_sbatch_option(text, "ntasks-per-node")) == cfg.num_device
    assert int(_sbatch_option(text, "gpus-per-node")) == cfg.num_device
    assert int(_sbatch_option(text, "nodes")) == cfg.num_nodes


def test_trial_job_gives_every_rank_a_core_per_loader_worker():
    """Loader workers beyond the task's cores only contend with each other."""
    from omegaconf import OmegaConf

    text = (DELTA / "train_trial.sbatch").read_text(encoding="utf-8")
    cfg = OmegaConf.load(RECIPE / "conf" / "training_f5_tts_base_delta_trial.yaml")
    workers = cfg.dataloader.train.iter_factory.num_workers
    assert workers < int(_sbatch_option(text, "cpus-per-task"))


def test_stage_job_refuses_to_run_without_a_destination():
    """Staging writes about a terabyte; where to must be said, never guessed."""
    text = (DELTA / "stage_emilia.sbatch").read_text(encoding="utf-8")
    guard = text.index('"${EMILIA_STAGE_DEST:?')
    assert guard < text.index("$PY local/stage_emilia.py")
    assert '--dest "$EMILIA_STAGE_DEST"' in text


@pytest.mark.parametrize("path", [ENV, *SBATCH], ids=lambda p: p.name)
def test_delta_script_is_valid_bash(path):
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_env_puts_this_tree_first_on_pythonpath(site):
    """The shared env has another espnet checkout installed.

    Without this tree first, espnet3 and egs3 resolve from that checkout
    while src/ and dataset/ load from here.
    """
    result = _source_env(site)
    assert result.returncode == 0, result.stderr
    pythonpath = re.search(r"^PYTHONPATH=(.*)$", result.stdout, re.MULTILINE).group(1)
    assert pythonpath.split(":")[:2] == [str(REPO), str(RECIPE)]


def test_env_points_the_recipe_at_the_speaker_tree_corpus(site):
    result = _source_env(site)
    assert "PY=/site/python\n" in result.stdout
    assert "ROOT=/site/corpus\n" in result.stdout
    assert "LAYOUT=speaker_tree\n" in result.stdout


def test_env_keeps_user_site_packages_out(site):
    """pip --user packages would shadow the env's own and differ per user."""
    assert "NOUSER=1\n" in _source_env(site).stdout


@pytest.mark.parametrize("missing", ["EMILIA_PY", "EMILIA_CORPUS_ROOT"])
def test_env_fails_loudly_without_site_settings(site, missing):
    del site[missing]
    result = _source_env(site)
    assert result.returncode != 0
    assert missing in result.stderr


def test_env_reads_site_settings_from_a_site_file(tmp_path):
    site_file = tmp_path / "site_env.sh"
    site_file.write_text(
        "EMILIA_PY=/from/file/python\nEMILIA_CORPUS_ROOT=/from/file/corpus\n", "utf-8"
    )
    result = _source_env({"EMILIA_SITE_ENV": str(site_file)})
    assert result.returncode == 0, result.stderr
    assert "PY=/from/file/python\n" in result.stdout
    assert "ROOT=/from/file/corpus\n" in result.stdout


@pytest.mark.parametrize("path", [ENV, *SBATCH], ids=lambda p: p.name)
def test_delta_scripts_carry_no_site_paths_or_accounts(path):
    """Site paths and allocation names stay out of version control.

    They go in the untracked local/delta/site_env.sh and on the sbatch
    command line (--account).
    """
    text = path.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("# ")
    )
    assert not re.search(r"(?<![\w$}])/(work|u|projects|scratch|ocean)/", code)
    assert "--account" not in code and "#SBATCH -A" not in code


@pytest.mark.parametrize("path", SBATCH, ids=lambda p: p.name)
def test_delta_sbatch_sources_env_before_running_python(path):
    text = path.read_text(encoding="utf-8")
    source = text.index('source "$SLURM_SUBMIT_DIR/local/delta/env.sh"')
    first_python = re.search(r"^\s*(srun\s[^\n]*)?\$PY\b", text, re.MULTILINE)
    assert first_python, f"{path.name}: no $PY invocation"
    assert source < first_python.start()


@pytest.mark.parametrize("path", SBATCH, ids=lambda p: p.name)
def test_delta_sbatch_checks_which_tree_python_imports(path):
    """A wrong import origin must stop the job, not surface hours later."""
    assert "local/delta/check_imports.py" in path.read_text(encoding="utf-8")


def test_site_env_file_is_ignored_by_git():
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(DELTA / "site_env.sh")], cwd=RECIPE
    )
    assert result.returncode == 0


def test_check_imports_accepts_this_tree(tmp_path):
    from egs3.emilia.tts.local.delta.check_imports import foreign_modules

    assert foreign_modules(REPO) == []


def test_check_imports_reports_modules_from_another_tree(tmp_path):
    from egs3.emilia.tts.local.delta.check_imports import foreign_modules

    foreign = foreign_modules(tmp_path)
    assert {name for name, _ in foreign} == {
        "espnet2",
        "espnet3",
        "egs3.emilia.tts",
        "espnet2.tasks.tts",
    }


def test_check_imports_fails_when_the_training_task_cannot_be_imported(monkeypatch):
    """The task's imports need packages the data stages never touch.

    On Delta the env lacked pyworld: the CPU job passed and the GPU job died
    in its first minute. The check runs at the top of every job instead.
    """
    import sys

    from egs3.emilia.tts.local.delta.check_imports import foreign_modules

    monkeypatch.setitem(sys.modules, "pyworld", None)  # import pyworld -> ImportError
    for name in ("espnet2.tasks.tts", "espnet2.tts.feats_extract.dio"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    with pytest.raises(ImportError, match="pyworld"):
        foreign_modules(REPO)
