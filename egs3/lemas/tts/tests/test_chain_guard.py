"""The chain must stop by itself when its links stop advancing the checkpoint.

The first production run queued 134 one-hour links that each restarted from
scratch and crashed at the same batch; nothing looked at the checkpoint.
"""

import os
from pathlib import Path

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


def test_submit_script_guards_kills_and_keeps_cores_out():
    s = Path("local/submit_train.sbatch").read_text()
    guard = s.index("$PY local/chain_guard.py")
    queue = s.index("sbatch --dependency=afterany")
    run = s.index("srun --kill-on-bad-exit=1 ")  # one dead rank ends the job
    assert guard < queue < run  # a stopped chain queues no successor
    assert "ulimit -c 0" in s
    assert "every validation" not in s  # last.ckpt is written at epoch ends
    # a last.ckpt link whose target is gone must not restart from scratch
    assert '[ -L "$CKPT" ] && [ ! -e "$CKPT" ]' in s
