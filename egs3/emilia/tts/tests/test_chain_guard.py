"""The chain must stop by itself when its links stop advancing the checkpoint.

A chain whose links each restart from scratch, or each die at the same batch,
looks healthy from the queue: jobs start, run and end. Nothing but the
checkpoint shows it.
"""

import os
import zipfile

from egs3.emilia.tts.local.chain_guard import STOP_EXIT, check, repair_last


def _link(exp, step):
    (exp / f"step{step}.ckpt").write_bytes(b"x")
    last = exp / "last.ckpt"
    if last.is_symlink():
        last.unlink()
    os.symlink(f"step{step}.ckpt", last)


def _zip_ckpt(path, complete=True):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", b"x" * 100)
    if not complete:  # a kill mid-write leaves no central directory at the end
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])


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


def test_a_link_left_dangling_by_a_kill_mid_save_is_repaired(tmp_path):
    # the walltime kill can land after Lightning wrote the new step file and
    # deleted the old one but before it repointed last.ckpt
    exp = tmp_path / "tag"
    exp.mkdir()
    _zip_ckpt(exp / "step78000.ckpt")
    os.symlink("step77900.ckpt", exp / "last.ckpt")
    assert repair_last(exp) == "step78000.ckpt"
    assert os.readlink(exp / "last.ckpt") == "step78000.ckpt"


def test_repair_refuses_an_incomplete_or_ambiguous_checkpoint(tmp_path):
    exp = tmp_path / "tag"
    exp.mkdir()
    _zip_ckpt(exp / "step78000.ckpt", complete=False)
    os.symlink("step77900.ckpt", exp / "last.ckpt")
    assert repair_last(exp) is None  # left for a human
    assert os.readlink(exp / "last.ckpt") == "step77900.ckpt"
    (exp / "step78000.ckpt").unlink()
    _zip_ckpt(exp / "step78000.ckpt")
    _zip_ckpt(exp / "step78100.ckpt")
    assert repair_last(exp) is None  # two candidates: not the known race


def test_a_healthy_link_is_left_alone(tmp_path):
    exp = tmp_path / "tag"
    exp.mkdir()
    _zip_ckpt(exp / "step78000.ckpt")
    os.symlink("step78000.ckpt", exp / "last.ckpt")
    assert repair_last(exp) is None
    assert repair_last(tmp_path / "missing") is None
