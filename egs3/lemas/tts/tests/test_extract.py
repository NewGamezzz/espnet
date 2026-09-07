import io
import json
import tarfile

import numpy as np
import pytest
import soundfile as sf
from dataset.extract import (
    bucket_of,
    chunk_key,
    extract_shard,
    read_pack_index,
    read_shard_members,
    regroup_language,
)


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
    # 16 kHz members are upsampled to the 24 kHz pack rate: 1600 -> 2400
    assert index == {"de000/a.mp3": (0, 2400), "de000/c.mp3": (2400, 2400)}
    pcm = np.frombuffer((out / "de000.pcm").read_bytes(), dtype="<i2")
    assert len(pcm) == 4800
    assert abs(pcm[100] / 32768 - 0.01) < 1e-3 and abs(pcm[2500] / 32768 - 0.03) < 1e-3
    assert cov == {
        "manifest_rows": 2,
        "members_extracted": 2,
        "resampled": 2,
        "samples": 4800,
        "missing": [],
    }
    assert (out / "de000.complete").is_file()
    assert not (out / "de000.pcm.tmp").exists()
    assert json.loads((out / "de000.coverage.json").read_text())["samples"] == 4800


def test_emilia_rates_land_at_24k_without_touching_native_24k(tmp_path):
    # the Emilia portions of en/zh ship at 24/32 kHz inside the LEMAS tars
    tar = _tar_with(tmp_path, ["de000/a.mp3", "de000/b.mp3"], sr=32000)
    out = tmp_path / "pcm"
    cov = extract_shard(tar, {"de000/a.mp3"}, out)
    assert read_pack_index(out / "de000.index.tsv")["de000/a.mp3"] == (0, 2400)
    assert cov["resampled"] == 1 and cov["members_extracted"] == 1
    tar = (
        _tar_with(tmp_path / "n", ["de000/a.mp3"], sr=24000)
        if (tmp_path / "n").mkdir() is None
        else None
    )
    cov = extract_shard(tar, {"de000/a.mp3"}, tmp_path / "n" / "pcm")
    assert cov["resampled"] == 0 and cov["samples"] == 2400


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


def _shard_tar(tmp_path, shard, members, amp):
    tar = tmp_path / f"{shard}.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        for name in members:
            buf = io.BytesIO()
            sf.write(
                buf, np.full(1600, amp[name], dtype=np.float32), 16000, format="WAV"
            )
            data = buf.getvalue()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return tar


def test_chunk_key_cuts_recording_groups_by_segment_and_speakers_by_hash():
    assert chunk_key("vidA", "k", seg=7, group_size=20, chunk_rows=3) == "vidA\x012"
    assert chunk_key("", "k", seg=None, group_size=1) == "k"
    subs = {chunk_key("spk", f"k{i}", None, 100, 32) for i in range(100)}
    assert 2 <= len(subs) <= 4  # ceil(100/32) = 4 sub-chunks, hashed
    assert 0 <= bucket_of("spk\x011", 8) < 8


def test_regroup_language_makes_chunks_contiguous_and_shuffled(tmp_path):
    # group A: 8 segments split over two shards; B: 2 rows; C, D: singletons
    a = [f"de00{s}/vidA-{i:05d}.mp3" for i, s in enumerate([0] * 5 + [1] * 3)]
    others = [
        "de000/vidB-00000.mp3",
        "de001/vidB-00001.mp3",
        "de000/vidC-00000.mp3",
        "de001/vidD-00000.mp3",
    ]
    members = a + others
    amp = {m: 0.001 * (i + 1) for i, m in enumerate(members)}
    tar0 = _shard_tar(
        tmp_path, "de000", [m for m in members if m.startswith("de000/")], amp
    )
    tar1 = _shard_tar(
        tmp_path, "de001", [m for m in members if m.startswith("de001/")], amp
    )
    out = tmp_path / "pcm"
    for tar in (tar0, tar1):
        extract_shard(
            tar, {m for m in members if m.startswith(tar.name[:5] + "/")}, out / "de"
        )

    def row(m):
        vid, seg = m.split("/")[1][:-4].split("-")
        group = vid if vid in ("vidA", "vidB") else ""
        size = {"vidA": 8, "vidB": 2}.get(vid, 1)
        ck = chunk_key(group, m, int(seg), size, chunk_rows=3)
        return m, (bucket_of(ck, 4), ck)

    shards = [
        (
            out / "de" / f"{s}.pcm",
            out / "de" / f"{s}.index.tsv",
            dict(row(m) for m in members if m.startswith(s + "/")),
        )
        for s in ("de000", "de001")
    ]
    stats = regroup_language(
        "de", shards, out, seed=1, chunk_rows=3, n_buckets=4, n_workers=1
    )
    assert stats == {"rows": 12, "chunks": 6, "samples": 12 * 2400}
    index = read_pack_index(out / "de" / "de.index.tsv")
    assert set(index) == set(members)
    pcm = np.frombuffer((out / "de" / "de.pcm").read_bytes(), dtype="<i2")
    assert len(pcm) == 12 * 2400
    for m in members:  # content survives the two passes
        s0, n = index[m]
        assert abs(pcm[s0] / 32768 - amp[m]) < 1e-3 and n == 2400
    # chunks of vidA are contiguous and in segment order: seg 0-2, 3-5, 6-7
    by_start = sorted(index.items(), key=lambda kv: kv[1][0])
    order = [m for m, _ in by_start]
    for chunk in ([a[0], a[1], a[2]], [a[3], a[4], a[5]], [a[6], a[7]]):
        i = order.index(chunk[0])
        assert order[i : i + len(chunk)] == chunk
    assert not (out / "de" / "_parts").exists()
    assert (out / "de" / "de.regrouped").is_file()
    # deterministic per seed, different for another seed
    order1 = order
    (out / "de" / "de.regrouped").unlink()
    for tar in (tar0, tar1):
        pass
    regroup_language("de", shards, out, seed=2, chunk_rows=3, n_buckets=4, n_workers=1)
    order2 = [
        m
        for m, _ in sorted(
            read_pack_index(out / "de" / "de.index.tsv").items(),
            key=lambda kv: kv[1][0],
        )
    ]
    assert order2 != order1 and set(order2) == set(order1)
    # a completed language is not redone
    assert regroup_language("de", shards, out, seed=3)["rows"] == 12
