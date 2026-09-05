import io
import json
import tarfile

import numpy as np
import pytest
import soundfile as sf

from dataset.extract import extract_shard, read_shard_members


def _tar_with(tmp_path, names):
    tar = tmp_path / "de000.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        for name in names:
            buf = io.BytesIO()
            sf.write(buf, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
            data = buf.getvalue()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return tar


def test_extracts_only_listed_members_and_writes_coverage(tmp_path):
    tar = _tar_with(tmp_path, ["de000/a.mp3", "de000/b.mp3", "de000/c.mp3"])
    out = tmp_path / "flac"
    cov = extract_shard(tar, {"de000/a.mp3", "de000/c.mp3"}, out)
    assert (out / "de000" / "a.flac").is_file()
    assert not (out / "de000" / "b.flac").exists()
    info = sf.info(str(out / "de000" / "c.flac"))
    assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
    assert cov == {"manifest_rows": 2, "members_extracted": 2, "missing": []}
    assert (out / "de000.complete").is_file()
    cov_file = json.loads((out / "de000.coverage.json").read_text())
    assert cov_file["members_extracted"] == 2


def test_rerun_skips_completed_shard(tmp_path):
    tar = _tar_with(tmp_path, ["de000/a.mp3"])
    out = tmp_path / "flac"
    extract_shard(tar, {"de000/a.mp3"}, out)
    tar.unlink()  # a second run must not need the tar at all
    assert extract_shard(tar, {"de000/a.mp3"}, out)["members_extracted"] == 1


def test_missing_member_fails_loudly(tmp_path):
    tar = _tar_with(tmp_path, ["de000/a.mp3"])
    with pytest.raises(RuntimeError, match="de000/zzz.mp3"):
        extract_shard(tar, {"de000/a.mp3", "de000/zzz.mp3"}, tmp_path / "flac")
    assert not (tmp_path / "flac" / "de000.complete").exists()


def test_read_shard_members(tmp_path):
    tsv = tmp_path / "de000.tsv"
    tsv.write_text("k1\tde000/a.mp3\t1.0\tyodas\t0\nk2\tde000/b.mp3\t2.0\tmls\t5\n")
    assert read_shard_members(tsv) == {"de000/a.mp3", "de000/b.mp3"}
