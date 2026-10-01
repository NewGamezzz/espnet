"""The chain must stop by itself when its links stop advancing the checkpoint.

The first production run queued 134 one-hour links that each restarted from
scratch and crashed at the same batch; nothing looked at the checkpoint.
"""

import os
from pathlib import Path

import pytest

from local.chain_guard import STOP_EXIT, check


def _link(exp, step):
    (exp / f"step{step}.ckpt").write_bytes(b"x")
    last = exp / "last.ckpt"
    if last.is_symlink():
        last.unlink()
    os.symlink(f"step{step}.ckpt", last)


def test_first_link_runs_without_a_checkpoint(tmp_path):
    exp = tmp_path / "exp" / "tag"  # not created yet
    assert check(exp, max_stalls=2) == 0
    assert not (exp / "STOP").exists()


def test_chain_stops_after_two_links_without_a_checkpoint(tmp_path):
    exp = tmp_path / "tag"
    assert [check(exp, max_stalls=2) for _ in range(3)] == [0, 0, STOP_EXIT]
    assert "no progress" in (exp / "STOP").read_text()


def test_chain_stops_when_the_checkpoint_stops_advancing(tmp_path):
    exp = tmp_path / "tag"
    exp.mkdir()
    codes = []
    for step in (None, 900, 1800, 1800, 1800):
        if step is not None:
            _link(exp, step)
        codes.append(check(exp, max_stalls=2))
    assert codes == [0, 0, 0, 0, STOP_EXIT]


def test_one_lost_link_does_not_stop_the_chain(tmp_path):
    # a node failure or an NCCL error costs a link now and then
    exp = tmp_path / "tag"
    exp.mkdir()
    codes = []
    for step in (900, 900, 1800, 1800, 2700):
        _link(exp, step)
        codes.append(check(exp, max_stalls=2))
    assert codes == [0] * 5 and not (exp / "STOP").exists()


def test_a_plain_file_checkpoint_is_tracked_by_size_and_mtime(tmp_path):
    exp = tmp_path / "tag"
    exp.mkdir()
    last = exp / "last.ckpt"
    last.write_bytes(b"a")
    assert check(exp, max_stalls=1) == 0
    last.write_bytes(b"ab")
    assert check(exp, max_stalls=1) == 0
    assert check(exp, max_stalls=1) == STOP_EXIT


def test_chain_link_guards_kills_and_keeps_cores_out():
    s = Path("local/train_link.sh").read_text()
    guard = s.index("$PY local/chain_guard.py")
    queue = s.index("sbatch --dependency=afterany")
    run = s.index("srun --kill-on-bad-exit=1 ")  # one dead rank ends the job
    assert guard < queue < run  # a stopped chain queues no successor
    assert "ulimit -c 0" in s
    assert "every validation" not in s  # last.ckpt is written at epoch ends
    # a last.ckpt link whose target is gone must not restart from scratch
    assert '[ -L "$CKPT" ] && [ ! -e "$CKPT" ]' in s
    # the successor is the sbatch file that sourced this body, on the same cluster
    assert '"$SLURM_SUBMIT_DIR/local/$SELF"' in s


@pytest.mark.parametrize(
    "name, env, suffix",
    [
        ("submit_train", "delta_env.sh", ""),
        ("submit_train_dtai", "delta_ai_env.sh", "_dtai"),
    ],
)
def test_submit_scripts_are_headers_plus_env_plus_the_shared_link(name, env, suffix):
    s = Path(f"local/{name}.sbatch").read_text()
    assert f"SELF={name}.sbatch" in s and f'CHAIN_SUFFIX="{suffix}"' in s
    assert f'source "$SLURM_SUBMIT_DIR/local/{env}"' in s
    assert s.rstrip().endswith('source "$SLURM_SUBMIT_DIR/local/train_link.sh"')
    assert "srun" not in s and "chain_guard" not in s  # the body owns these
    assert f"--output=logs/train{suffix}_%j.out" in s  # Delta and DeltaAI ids collide


def test_dtai_script_and_env_follow_the_aarch64_rules():
    s = Path("local/submit_train_dtai.sbatch").read_text()
    assert "--partition=ghx4\n" in s and "--account=bbjs-dtai-gh" in s
    assert "#SBATCH --partition=ghx4-interactive" not in s  # 2x billing, 2 h cap
    assert "--gpus-per-node=2" in s and "--ntasks-per-node=2" in s
    assert "CONF_DEFAULT=conf/training_f5_base_dualprompt_gh200.yaml" in s
    env = Path("local/delta_ai_env.sh").read_text()
    assert "export OMP_NUM_THREADS=1" in env  # forked loader workers livelock otherwise
    assert "pixi_env/conversational_f5_stage2" in env  # the aarch64 CUDA torch env
    assert "export PYTHONPATH=$ROOT:$RECIPE:$PYLIBS" in env
    for f in (s, env):
        assert "module load cudnn" not in f
