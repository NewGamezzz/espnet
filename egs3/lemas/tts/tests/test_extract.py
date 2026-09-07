import io
import json
import tarfile

import numpy as np
import pytest
import soundfile as sf
from dataset.extract import extract_shard, read_pack_index, read_shard_members


def _tar_with(tmp_path, names, sr=16000, sec=0.1):
    tar = tmp_path / "de000.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        for k, name in enumerate(names):
            buf = io.BytesIO()
            wav = np.full(int(sr * sec), 0.01 * (k + 1), dtype=np.float32)
            sf.write(buf, wav, sr, format="WAV")
            data = buf.getvalue()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return tar


def test_packs_only_listed_members_in_tar_order(tmp_path):
    tar = _tar_with(tmp_path, ["de000/a.mp3", "de000/b.mp3", "de000/c.mp3"])
    out = tmp_path / "pcm"
    cov = extract_shard(tar, {"de000/c.mp3", "de000/a.mp3"}, out)
    index = read_pack_index(out / "de000.index.tsv")
    assert list(index) == ["de000/a.mp3", "de000/c.mp3"]
    assert index == {"de000/a.mp3": (0, 1600), "de000/c.mp3": (1600, 1600)}
    pcm = np.frombuffer((out / "de000.pcm").read_bytes(), dtype="<i2")
    assert len(pcm) == 3200
    assert abs(pcm[0] / 32768 - 0.01) < 1e-3 and abs(pcm[1600] / 32768 - 0.03) < 1e-3
    assert cov == {
        "manifest_rows": 2,
        "members_extracted": 2,
        "resampled": 0,
        "samples": 3200,
        "missing": [],
    }
    assert (out / "de000.complete").is_file()
    assert not (out / "de000.pcm.tmp").exists()
    assert json.loads((out / "de000.coverage.json").read_text())["samples"] == 3200


def test_other_rate_members_are_resampled_to_16k(tmp_path):
    # the Emilia portions of en/zh ship at 24/32 kHz inside the LEMAS tars
    tar = _tar_with(tmp_path, ["de000/a.mp3"], sr=32000)
    out = tmp_path / "pcm"
    cov = extract_shard(tar, {"de000/a.mp3"}, out)
    assert read_pack_index(out / "de000.index.tsv")["de000/a.mp3"] == (0, 1600)
    assert cov["resampled"] == 1 and cov["members_extracted"] == 1


def test_rerun_skips_completed_shard(tmp_path):
    tar = _tar_with(tmp_path, ["de000/a.mp3"])
    out = tmp_path / "pcm"
    extract_shard(tar, {"de000/a.mp3"}, out)
    tar.unlink()  # a second run must not need the tar at all
    assert extract_shard(tar, {"de000/a.mp3"}, out)["members_extracted"] == 1


def test_missing_member_fails_loudly(tmp_path):
    tar = _tar_with(tmp_path, ["de000/a.mp3"])
    with pytest.raises(RuntimeError, match="de000/zzz.mp3"):
        extract_shard(tar, {"de000/a.mp3", "de000/zzz.mp3"}, tmp_path / "pcm")
    assert not (tmp_path / "pcm" / "de000.complete").exists()


def test_read_shard_members(tmp_path):
    tsv = tmp_path / "de000.tsv"
    tsv.write_text("k1\tde000/a.mp3\t1.0\tyodas\t0\nk2\tde000/b.mp3\t2.0\tmls\t5\n")
    assert read_shard_members(tsv) == {"de000/a.mp3", "de000/b.mp3"}
