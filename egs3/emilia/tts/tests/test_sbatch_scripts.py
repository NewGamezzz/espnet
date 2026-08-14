"""Cheap, environment-independent checks on the PSC sbatch scripts.

These cannot exercise the real Slurm/PSC-path behavior (that only runs on
the cluster, out of scope here -- see task-13-brief.md's scope boundary),
but a `bash -n` syntax gate plus invariant checks on the safety mechanisms
this task added (quota guard first, chain-depth cap, kill-on-bad-exit) are
cheap and catch a broken script before it ever reaches PSC.
"""

import subprocess
from pathlib import Path

LOCAL = Path(__file__).resolve().parents[1] / "local"
SMOKE = LOCAL / "submit_smoke.sbatch"
TRAIN = LOCAL / "submit_train.sbatch"


def _bash_syntax_ok(path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)


def test_smoke_sbatch_is_valid_bash():
    result = _bash_syntax_ok(SMOKE)
    assert result.returncode == 0, result.stderr


def test_train_sbatch_is_valid_bash():
    result = _bash_syntax_ok(TRAIN)
    assert result.returncode == 0, result.stderr


def test_smoke_sbatch_requests_full_8gpu_node():
    text = SMOKE.read_text()
    assert "--gres=gpu:v100-32:8" in text
    assert "--ntasks-per-node=8" in text


def test_smoke_sbatch_uses_smoke_config():
    text = SMOKE.read_text()
    assert "training_f5_tts_base_smoke.yaml" in text


def test_smoke_sbatch_documents_corrected_success_criterion():
    """The plan's original Task 11 step 6 wording ("len(batches) unchanged")
    is unsatisfiable once max_samples=64 (Task 12) is active; the sbatch
    header must carry the corrected form so whoever runs the smoke doesn't
    read a correct result as a regression."""
    text = SMOKE.read_text()
    assert "RSS fell" in text
    assert "batch count increased only at the short end" in text
    assert "no batch" in text and "exceeds 64" in text


def test_train_sbatch_runs_quota_guard_before_training():
    text = TRAIN.read_text()
    guard_idx = text.index("local/quota_guard.sh")
    train_idx = text.index("run.py --stages train")
    assert guard_idx < train_idx


def test_train_sbatch_has_chain_depth_cap():
    text = TRAIN.read_text()
    assert "MAX_CHAIN_DEPTH" in text
    assert "CHAIN_DEPTH" in text
    assert "--dependency=afterany" in text


def test_train_sbatch_forwards_chain_depth_to_resubmission():
    """--export=ALL must be present on the resubmit, or CHAIN_DEPTH resets
    to 0 on every hop and the cap above never binds."""
    text = TRAIN.read_text()
    assert "--export=ALL,CHAIN_DEPTH=" in text


def test_train_sbatch_kills_siblings_on_rank_crash():
    """Addresses the rank-0-OOM-masquerades-as-NCCL-timeout failure mode
    ([[psc-conversational-f5-oom-hang]]): one dead rank must tear down the
    step instead of leaving the others in _sync2skip's barrier."""
    text = TRAIN.read_text()
    assert "--kill-on-bad-exit=1" in text
    assert "TORCH_NCCL_ASYNC_ERROR_HANDLING" in text


def test_train_sbatch_checks_completion_before_training():
    text = TRAIN.read_text()
    assert "STOP" in text
    assert "MAX_STEPS" in text
