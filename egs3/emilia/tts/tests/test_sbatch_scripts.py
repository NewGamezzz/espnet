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


def _executable_body(text: str) -> str:
    """Return the script text after the header comment block.

    submit_train.sbatch's header prose mentions several command names
    (srun, --dependency=afterany, scancel, ...) while explaining the fix,
    so a plain `text.index(...)` for those substrings can match the
    prose instead of the real, executable line. `set -euo pipefail` is
    the first line of actual code in both sbatch scripts.
    """
    return text[text.index("set -euo pipefail") :]


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


def test_smoke_sbatch_documents_sampler_construction_cost():
    """Sampler construction is paid on every restart of a chained run.

    Replaces an earlier assertion about a max_samples=64 success criterion,
    which went away with the custom sampler. What still matters is that
    whoever runs the smoke knows to measure how long the stock `numel`
    sampler takes to build and how much host RSS it holds -- roughly 12 GB
    and ~4 minutes per rank extrapolated to 37M utterances.
    """
    text = SMOKE.read_text()
    assert "PEAK HOST RSS" in text
    assert "stock `numel` sampler" in text
    assert "EVERY restart" in text


def test_train_sbatch_runs_quota_guard_before_training():
    body = _executable_body(TRAIN.read_text())
    guard_idx = body.index("local/quota_guard.sh")
    train_idx = body.index("run.py --stages train")
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


def test_train_sbatch_queues_resubmission_before_training_runs():
    """Fix round 1 regression guard: the original script queued the next
    hop AFTER `srun` returned, which never runs when Slurm delivers
    SIGTERM at the walltime limit (the primary scenario chaining exists
    for) -- the chain silently stopped at every walltime boundary. The
    resubmission must be queued before training starts, so it is already
    in the queue regardless of how this job later dies."""
    body = _executable_body(TRAIN.read_text())
    queue_idx = body.index("--dependency=afterany:")
    srun_idx = body.index("srun --kill-on-bad-exit=1")
    assert queue_idx < srun_idx, (
        "resubmission must be queued before srun, not after -- a job "
        "killed by SIGTERM (walltime) never reaches a trailing resubmit"
    )


def test_train_sbatch_completion_check_precedes_resubmission_queue():
    """The completion check (STOP sentinel / max_steps reached) must run
    BEFORE the next hop is queued -- otherwise a finished run queues one
    more job that immediately exits, forever."""
    body = _executable_body(TRAIN.read_text())
    queue_idx = body.index("--dependency=afterany:")
    stop_check_idx = body.index('[[ -f "$EXP_DIR/STOP" ]]')
    max_steps_check_idx = body.index("STEP >= MAX_STEPS")
    assert stop_check_idx < queue_idx
    assert max_steps_check_idx < queue_idx


def test_train_sbatch_chain_depth_cap_precedes_resubmission_queue():
    """If this job's own chain depth already reached the cap, it must not
    queue a further hop."""
    body = _executable_body(TRAIN.read_text())
    queue_idx = body.index("--dependency=afterany:")
    depth_check_idx = body.index("CHAIN_DEPTH >= MAX_CHAIN_DEPTH")
    assert depth_check_idx < queue_idx


def test_train_sbatch_documents_scancel_behavior():
    """--dependency=afterany fires on cancellation too, so scancel-ing a
    running job does not cancel its already-queued successor; this must be
    documented so an operator isn't surprised by a chain that keeps going
    after a manual scancel."""
    text = TRAIN.read_text()
    assert "scancel" in text.lower()
    assert "STOP" in text


def test_train_sbatch_passes_config_via_env_not_string_interpolation():
    """A $CONFIG path containing a single quote would break a python -c
    string built by direct interpolation; pass it through the environment
    instead."""
    text = TRAIN.read_text()
    assert "os.environ['CONFIG']" in text
    assert "Path('$CONFIG')" not in text


def test_quota_guard_documents_awk_vs_bc_deviation():
    text = (LOCAL / "quota_guard.sh").read_text()
    assert "DEVIATION" in text
    assert "bc" in text


def test_train_sbatch_uses_template_config_name_not_recipe_filename():
    """load_and_merge_config's `config_name` selects the near-empty
    TEMPLATE stub (egs3/TEMPLATE/tts/conf/training.yaml), independent of
    whichever concrete config $CONFIG points at -- it is NOT supposed to
    match this recipe's own filename. Passing
    config_name='training_f5_tts_base.yaml' (a file that only exists under
    this recipe's own conf/, not the template's) made load_and_merge_config
    raise FileNotFoundError on every invocation; verified directly against
    run.py's identical bug (see tests/test_run_default_configs.py)."""
    body = _executable_body(TRAIN.read_text())
    assert "config_name='training.yaml'" in body
    assert "config_name='training_f5_tts_base.yaml'" not in body


def test_train_sbatch_preflight_checks_artifacts_before_resubmission_queue():
    """IMPORTANT 1: a missing vocab_file/feats_shape/manifest must abort a
    single job, not queue up to MAX_CHAIN_DEPTH=500 queued-and-failing
    successors -- the pre-flight check must run, and be visible in the
    script, before the queuing step."""
    body = _executable_body(TRAIN.read_text())
    queue_idx = body.index("--dependency=afterany:")
    preflight_idx = body.index("PRE-FLIGHT FAILED")
    assert preflight_idx < queue_idx
    assert "VOCAB_FILE" in body
    assert "TRAIN_SHAPE" in body
    assert "VALID_SHAPE" in body
    assert "TRAIN_MANIFEST" in body
    assert "VALID_MANIFEST" in body


def test_train_sbatch_passes_ckpt_path_conditionally():
    """conf/training_f5_tts_base.yaml ships `fit: {}` (IMPORTANT 1): resume
    must come from here instead, and only when $LAST_CKPT already exists on
    disk, or a fresh run's first hop FileNotFoundErrors in Lightning."""
    body = _executable_body(TRAIN.read_text())
    ckpt_check_idx = body.index('if [[ -e "$LAST_CKPT" ]]; then\n    CKPT_ARGS')
    srun_idx = body.index("srun --kill-on-bad-exit=1")
    assert ckpt_check_idx < srun_idx
    assert '--ckpt_path "$LAST_CKPT"' in body
    assert '"${CKPT_ARGS[@]}"' in body


def test_every_job_script_exports_pythonpath():
    """Every sbatch that runs run.py must put THIS tree on PYTHONPATH first.

    The pixi env's installed espnet lives in a different tree
    (espnet_f5_libritts), so without an explicit PYTHONPATH `espnet3` and
    `egs3` resolve from THERE while src/ and dataset/ load from here. The
    result is a half-mixed import graph that dies with
    `ModuleNotFoundError: No module named 'egs3.emilia'` -- which is exactly
    how the first real training job failed (43825659), on every rank, after
    passing its own pre-flight artifact checks. Those checks test that files
    exist on disk, not that the code is importable.

    It also carries pylibs/, which holds rjieba: absent from the shared env
    and imported lazily by F5PinyinPreprocessor, so omitting it fails
    mid-training rather than at startup.
    """
    import re

    local_dir = Path(__file__).resolve().parents[1] / "local"
    scripts = sorted(local_dir.glob("*.sbatch"))
    assert scripts, "no sbatch scripts found"

    checked = 0
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        if "run.py" not in text and "parity/" not in text:
            continue
        checked += 1
        m = re.search(r'^export PYTHONPATH=.*$', text, re.MULTILINE)
        assert m, f"{script.name} runs a recipe entry point but never exports PYTHONPATH"
        line = m.group(0)
        assert "$ROOT/pylibs" in line, f"{script.name}: pylibs missing from PYTHONPATH"
        assert "$ROOT:" in line or '"$ROOT"' in line, (
            f"{script.name}: repo root missing from PYTHONPATH -- espnet3/egs3 "
            "would resolve from the installed espnet in another tree"
        )
        # ...and before the first real invocation. Match an executable line,
        # not a mention: these headers discuss run.py in prose, and an earlier
        # version of this test compared against a comment.
        invocation = re.search(
            r'^\s*(srun[^\n]*\s)?python[^\n]*\b(run\.py|parity/)',
            text, re.MULTILINE,
        )
        assert invocation, f"{script.name}: no python invocation found"
        assert text.index(line) < invocation.start(), (
            f"{script.name}: PYTHONPATH is exported after the first python "
            "invocation"
        )
    assert checked >= 3, f"expected several job scripts, checked {checked}"
