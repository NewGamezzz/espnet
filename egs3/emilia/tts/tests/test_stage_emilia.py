"""local/stage_emilia.py: Hugging Face archives -> the ``speaker_tree`` layout.

Each archive of the dataset repo is a flat tar of ``<utt_id>.json`` and
``<utt_id>.mp3`` members. Staging turns a language's archives into the tree
the recipe reads with ``corpus_layout: speaker_tree``, so a language can be
staged privately next to a shared copy of the others.
"""

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.emilia.tts.dataset.builder import EmiliaBuilder
from egs3.emilia.tts.local.stage_emilia import extract_archive, merge_parts, stage

_CONFIG = """
builder:
  corpus_root: {corpus}
  corpus_layout: speaker_tree
  langs: [ZH]
  data_path: data
  val_ratio: 0.2
  seed: 42
  min_duration: 0.3
  max_duration: 30.0
  strict_text_filters: false
  audio_suffix: .mp3
  manifest_paths:
    train: manifest/train.tsv
    valid: manifest/valid.tsv
  shard_table_path: manifest/shards.txt
dataset:
  split_manifest_paths:
    train: manifest/train.tsv
    valid: manifest/valid.tsv
"""


def _record(utt_id: str, text: str = "你好", duration: float = 3.0) -> dict:
    batch, speaker = utt_id.rsplit("_", 2)[0], utt_id.rsplit("_", 1)[0]
    return {
        "id": utt_id,
        "wav": f"{batch}/{speaker}/mp3/{utt_id}.mp3",
        "text": text,
        "duration": duration,
        "speaker": speaker,
        "language": "zh",
        "dnsmos": 3.0,
    }


def _add(tf: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tf.addfile(info, io.BytesIO(payload))


def _write_archive(path: Path, utt_ids, extra=()) -> Path:
    """A flat archive like the dataset's: a json and an mp3 per utterance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tf:
        for utt_id in utt_ids:
            record = json.dumps(_record(utt_id), ensure_ascii=False).encode("utf-8")
            _add(tf, f"{utt_id}.json", record)
            _add(tf, f"{utt_id}.mp3", b"audio-of-" + utt_id.encode())
        for name, payload in extra:
            _add(tf, name, payload)
    return path


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def test_extract_puts_audio_where_the_speaker_tree_layout_looks(tmp_path):
    archive = _write_archive(
        tmp_path / "ZH-B000000.tar",
        ["ZH_B00000_S00001_W000000", "ZH_B00000_S00002_W000003"],
    )
    root = tmp_path / "corpus" / "emilia"

    counts = extract_archive(archive, root, "ZH", tmp_path / "parts" / "a.jsonl")

    audio = root / "ZH/ZH_B00000/ZH_B00000_S00002/mp3/ZH_B00000_S00002_W000003.mp3"
    assert audio.read_bytes() == b"audio-of-ZH_B00000_S00002_W000003"
    assert counts == {"audio": 2, "records": 2, "skipped": 0}


def test_extract_writes_the_archive_records_as_its_completion_marker(tmp_path):
    archive = _write_archive(
        tmp_path / "ZH-B000000.tar",
        ["ZH_B00000_S00002_W000003", "ZH_B00000_S00001_W000000"],
    )
    part = tmp_path / "parts" / "ZH-B000000.jsonl"

    extract_archive(archive, tmp_path / "emilia", "ZH", part)

    assert [r["id"] for r in _lines(part)] == [
        "ZH_B00000_S00001_W000000",
        "ZH_B00000_S00002_W000003",
    ]
    assert _lines(part)[0]["text"] == "你好"
    assert list(part.parent.iterdir()) == [part]  # no temp file left behind


@pytest.mark.parametrize(
    "name",
    [
        "../ZH_B00000_S00009_W000000.mp3",
        "/abs/ZH_B00000_S00009_W000000.mp3",
        "nested/ZH_B00000_S00009_W000000.mp3",
        "EN_B00000_S00009_W000000.mp3",  # another language's tree
        "ZH_B00000_S00009_W000000.exe",
        "notes.txt",
    ],
)
def test_extract_skips_members_it_does_not_recognise(tmp_path, name):
    """An archive is downloaded data: only well-formed members are written."""
    archive = _write_archive(
        tmp_path / "ZH-B000000.tar",
        ["ZH_B00000_S00001_W000000"],
        extra=[(name, b"x")],
    )
    dest = tmp_path / "dest"

    counts = extract_archive(archive, dest / "emilia", "ZH", dest / "parts" / "a.jsonl")

    assert counts == {"audio": 1, "records": 1, "skipped": 1}
    written = sorted(p for p in tmp_path.rglob("*") if p.is_file())
    assert written == sorted(
        [
            archive,
            dest / "parts" / "a.jsonl",
            dest
            / "emilia/ZH/ZH_B00000/ZH_B00000_S00001/mp3/ZH_B00000_S00001_W000000.mp3",
        ]
    )


def test_extract_skips_links(tmp_path):
    archive = tmp_path / "ZH-B000000.tar"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("ZH_B00000_S00001_W000000.mp3")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)

    counts = extract_archive(archive, tmp_path / "emilia", "ZH", tmp_path / "p.jsonl")

    assert counts == {"audio": 0, "records": 0, "skipped": 1}
    assert not (tmp_path / "emilia").exists()


def test_extract_skips_a_record_filed_under_another_id(tmp_path):
    archive = tmp_path / "ZH-B000000.tar"
    with tarfile.open(archive, "w") as tf:
        wrong = json.dumps(_record("ZH_B00000_S00001_W000009")).encode("utf-8")
        _add(tf, "ZH_B00000_S00001_W000000.json", wrong)
        _add(tf, "ZH_B00000_S00001_W000001.json", b"{not json")
    part = tmp_path / "p.jsonl"

    counts = extract_archive(archive, tmp_path / "emilia", "ZH", part)

    assert counts == {"audio": 0, "records": 0, "skipped": 2}
    assert part.read_text("utf-8") == ""


def _fetcher(tmp_path, contents: dict, calls: list):
    """A stand-in for the download: builds the named archive on demand."""

    def fetch(name: str) -> Path:
        calls.append(name)
        return _write_archive(tmp_path / "downloads" / f"{name}.tar", contents[name])

    return fetch


_CONTENTS = {
    "ZH-B000000": ["ZH_B00000_S00001_W000000", "ZH_B00000_S00002_W000000"],
    "ZH-B000001": ["ZH_B00000_S00003_W000000"],
    "ZH-B000010": ["ZH_B00001_S00001_W000000"],
}


def test_stage_extracts_every_archive_and_removes_the_downloads(tmp_path):
    calls: list = []
    dest = tmp_path / "corpus"

    done = stage("ZH", dest, sorted(_CONTENTS), _fetcher(tmp_path, _CONTENTS, calls))

    assert sorted(calls) == sorted(_CONTENTS)
    assert done == {name: True for name in _CONTENTS}
    assert not list((tmp_path / "downloads").glob("*.tar"))
    assert len(list((dest / "emilia" / "ZH").rglob("*.mp3"))) == 4


def test_stage_resumes_without_fetching_finished_archives(tmp_path):
    calls: list = []
    dest = tmp_path / "corpus"
    stage("ZH", dest, ["ZH-B000000"], _fetcher(tmp_path, _CONTENTS, calls))

    stage("ZH", dest, sorted(_CONTENTS), _fetcher(tmp_path, _CONTENTS, calls))

    assert calls == ["ZH-B000000", "ZH-B000001", "ZH-B000010"]


def test_stage_reports_an_archive_that_failed_and_goes_on(tmp_path):
    calls: list = []
    good = _fetcher(tmp_path, _CONTENTS, calls)

    def fetch(name: str) -> Path:
        if name == "ZH-B000001":
            raise OSError("connection reset")
        return good(name)

    done = stage("ZH", tmp_path / "corpus", sorted(_CONTENTS), fetch)

    assert done == {"ZH-B000000": True, "ZH-B000001": False, "ZH-B000010": True}


def test_merge_writes_one_sorted_jsonl_per_batch(tmp_path):
    dest = tmp_path / "corpus"
    stage("ZH", dest, sorted(_CONTENTS), _fetcher(tmp_path, _CONTENTS, []))

    rows = merge_parts("ZH", dest, sorted(_CONTENTS))

    zh = dest / "emilia" / "ZH"
    assert rows == {"ZH_B00000": 3, "ZH_B00001": 1}
    assert sorted(p.name for p in zh.glob("*.jsonl")) == [
        "ZH_B00000.jsonl",
        "ZH_B00001.jsonl",
    ]
    assert [r["id"] for r in _lines(zh / "ZH_B00000.jsonl")] == [
        "ZH_B00000_S00001_W000000",
        "ZH_B00000_S00002_W000000",
        "ZH_B00000_S00003_W000000",
    ]


def test_merge_refuses_to_publish_metadata_for_a_partly_staged_language(tmp_path):
    """A batch jsonl is what the builder trusts, so it must be complete."""
    dest = tmp_path / "corpus"
    stage("ZH", dest, ["ZH-B000000"], _fetcher(tmp_path, _CONTENTS, []))

    with pytest.raises(RuntimeError, match="ZH-B000001"):
        merge_parts("ZH", dest, sorted(_CONTENTS))
    assert not list((dest / "emilia" / "ZH").glob("*.jsonl"))


def test_a_staged_language_builds_with_the_speaker_tree_layout(tmp_path):
    """End to end: what staging writes is what the manifest builder reads."""
    names = ["ZH-B000000"]
    utts = _CONTENTS["ZH-B000000"]

    def fetch(name: str) -> Path:
        path = tmp_path / "downloads" / f"{name}.tar"
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "w") as tf:
            for utt_id in utts:
                record = json.dumps(_record(utt_id), ensure_ascii=False)
                _add(tf, f"{utt_id}.json", record.encode("utf-8"))
                audio = io.BytesIO()
                sf.write(audio, np.zeros(48000, dtype=np.float32), 16000, format="MP3")
                _add(tf, f"{utt_id}.mp3", audio.getvalue())
        return path

    corpus = tmp_path / "corpus"
    stage("ZH", corpus, names, fetch)
    merge_parts("ZH", corpus, names)

    recipe = tmp_path / "recipe"
    (recipe / "dataset").mkdir(parents=True)
    (recipe / "dataset" / "config.yaml").write_text(
        _CONFIG.format(corpus=corpus), encoding="utf-8"
    )
    EmiliaBuilder().build(recipe_dir=recipe)

    report = json.loads((recipe / "data/manifest/report.json").read_text("utf-8"))
    assert report["kept"] == 2
    assert report["dropped"] == {}
