"""EmiliaBuilder manifest generation."""

import json
from pathlib import Path

import pytest

from egs3.emilia.tts.dataset.builder import EmiliaBuilder


def _write_utt(
    shard_dir: Path,
    utt_id: str,
    speaker: str,
    text: str,
    duration: float,
    language: str,
):
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / f"{utt_id}.mp3").write_bytes(b"\x00")
    (shard_dir / f"{utt_id}.json").write_text(
        json.dumps(
            {
                "id": utt_id,
                "wav": f"ignored/{utt_id}.mp3",
                "text": text,
                "duration": duration,
                "speaker": speaker,
                "language": language,
                "dnsmos": 3.0,
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def corpus(tmp_path):
    """Two EN sub-shards of the same logical batch, plus one ZH shard."""
    root = tmp_path / "raw"
    # Sub-shard 0 and sub-shard 1 of logical batch 12 -> same B00012 prefix.
    _write_utt(
        root / "emilia/EN/EN-B000120",
        "EN_B00012_S00001_W000000",
        "EN_B00012_S00001",
        "the first english utterance",
        4.0,
        "en",
    )
    _write_utt(
        root / "emilia/EN/EN-B000121",
        "EN_B00012_S00002_W000000",
        "EN_B00012_S00002",
        "the second english utterance",
        6.0,
        "en",
    )
    # Blocklisted speaker.
    _write_utt(
        root / "emilia/EN/EN-B000130",
        "EN_B00013_S00913_W000000",
        "EN_B00013_S00913",
        "blocklisted speaker text",
        5.0,
        "en",
    )
    # Too short.
    _write_utt(
        root / "emilia/EN/EN-B000130",
        "EN_B00013_S00001_W000000",
        "EN_B00013_S00001",
        "tiny",
        0.2,
        "en",
    )
    _write_utt(
        root / "emilia/ZH/ZH-B000000",
        "ZH_B00000_S00001_W000000",
        "ZH_B00000_S00001",
        "你好,世界",
        3.0,
        "zh",
    )
    return root


@pytest.fixture
def recipe(tmp_path, corpus):
    recipe_dir = tmp_path / "recipe"
    (recipe_dir / "dataset").mkdir(parents=True)
    (recipe_dir / "dataset" / "config.yaml").write_text(
        f"""
builder:
  corpus_root: {corpus}
  langs: [EN, ZH]
  data_path: data
  val_ratio: 0.2
  seed: 42
  min_duration: 0.3
  max_duration: 30.0
  strict_text_filters: false
  manifest_paths:
    train: manifest/train.tsv
    valid: manifest/valid.tsv
  shard_table_path: manifest/shards.txt
dataset:
  split_manifest_paths:
    train: manifest/train.tsv
    valid: manifest/valid.tsv
""",
        encoding="utf-8",
    )
    return recipe_dir


def _read_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        utt_id, shard_idx, lang, duration, text = line.split("\t", 4)
        rows.append((utt_id, int(shard_idx), lang, float(duration), text))
    return rows


def test_build_writes_manifests_and_shard_table(recipe):
    builder = EmiliaBuilder()
    assert builder.is_built(recipe_dir=recipe) is False
    builder.build(recipe_dir=recipe)
    assert builder.is_built(recipe_dir=recipe) is True

    data = recipe / "data" / "manifest"
    train = _read_rows(data / "train.tsv")
    valid = _read_rows(data / "valid.tsv")
    # 5 written, 2 dropped (blocklist + duration) -> 3 kept.
    assert len(train) + len(valid) == 3


def test_blocklisted_and_short_utterances_are_dropped(recipe):
    EmiliaBuilder().build(recipe_dir=recipe)
    data = recipe / "data" / "manifest"
    ids = {r[0] for r in _read_rows(data / "train.tsv")}
    ids |= {r[0] for r in _read_rows(data / "valid.tsv")}
    assert "EN_B00013_S00913_W000000" not in ids
    assert "EN_B00013_S00001_W000000" not in ids


def test_shard_table_disambiguates_sub_shards(recipe):
    """Two utterances with the same B00012 prefix live in different dirs."""
    EmiliaBuilder().build(recipe_dir=recipe)
    data = recipe / "data" / "manifest"
    shards = (data / "shards.txt").read_text(encoding="utf-8").splitlines()
    rows = _read_rows(data / "train.tsv") + _read_rows(data / "valid.tsv")
    by_id = {r[0]: r for r in rows}
    assert shards[by_id["EN_B00012_S00001_W000000"][1]] == "EN/EN-B000120"
    assert shards[by_id["EN_B00012_S00002_W000000"][1]] == "EN/EN-B000121"


def test_zh_text_is_punctuation_normalized(recipe):
    EmiliaBuilder().build(recipe_dir=recipe)
    data = recipe / "data" / "manifest"
    rows = _read_rows(data / "train.tsv") + _read_rows(data / "valid.tsv")
    zh = [r for r in rows if r[2] == "ZH"]
    assert zh and zh[0][4] == "你好，世界"


def test_report_records_drop_reasons_and_histogram(recipe):
    EmiliaBuilder().build(recipe_dir=recipe)
    report = json.loads(
        (recipe / "data" / "manifest" / "report.json").read_text("utf-8")
    )
    assert report["kept"] == 3
    assert report["dropped"]["blocklist"] == 1
    assert report["dropped"]["duration"] == 1
    assert len(report["duration_histogram"]) == 50
    assert report["total_hours"] == pytest.approx(13.0 / 3600, rel=1e-6)


def test_build_is_deterministic(recipe):
    EmiliaBuilder().build(recipe_dir=recipe)
    first = (recipe / "data" / "manifest" / "train.tsv").read_text("utf-8")
    EmiliaBuilder().build(recipe_dir=recipe)
    assert (recipe / "data" / "manifest" / "train.tsv").read_text("utf-8") == first


def test_is_source_prepared_false_when_corpus_missing(tmp_path, recipe):
    (recipe / "dataset" / "config.yaml").write_text(
        (recipe / "dataset" / "config.yaml")
        .read_text("utf-8")
        .replace(str(tmp_path / "raw"), str(tmp_path / "nope")),
        encoding="utf-8",
    )
    assert EmiliaBuilder().is_source_prepared(recipe_dir=recipe) is False


def test_prepare_source_refuses_to_download(recipe, tmp_path):
    """The corpus is staged and read-only; prepare_source must never write."""
    cfg = recipe / "dataset" / "config.yaml"
    cfg.write_text(
        cfg.read_text("utf-8").replace(str(tmp_path / "raw"), str(tmp_path / "nope")),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="staged"):
        EmiliaBuilder().prepare_source(recipe_dir=recipe)


def test_single_surviving_utterance_goes_to_train_not_valid(tmp_path):
    """With exactly one kept utterance, it must land in train, not valid.

    n_total == 1 is the edge case where a naive val_ratio-based rounding
    (``max(1, int(n * ratio))``) rounds up to consume the *entire* corpus
    into valid.tsv, leaving train.tsv empty and a training job with zero
    rows but no error from this stage.
    """
    root = tmp_path / "raw"
    _write_utt(
        root / "emilia/EN/EN-B000010",
        "EN_B00001_S00001_W000000",
        "EN_B00001_S00001",
        "the only surviving utterance",
        4.0,
        "en",
    )
    # Blocklisted speaker: filtered out, so exactly one utterance survives.
    _write_utt(
        root / "emilia/EN/EN-B000010",
        "EN_B00013_S00913_W000000",
        "EN_B00013_S00913",
        "blocklisted speaker text",
        5.0,
        "en",
    )

    recipe_dir = tmp_path / "recipe"
    (recipe_dir / "dataset").mkdir(parents=True)
    (recipe_dir / "dataset" / "config.yaml").write_text(
        f"""
builder:
  corpus_root: {root}
  langs: [EN]
  data_path: data
  val_ratio: 0.2
  seed: 42
  min_duration: 0.3
  max_duration: 30.0
  strict_text_filters: false
  manifest_paths:
    train: manifest/train.tsv
    valid: manifest/valid.tsv
  shard_table_path: manifest/shards.txt
dataset:
  split_manifest_paths:
    train: manifest/train.tsv
    valid: manifest/valid.tsv
""",
        encoding="utf-8",
    )

    EmiliaBuilder().build(recipe_dir=recipe_dir)
    data = recipe_dir / "data" / "manifest"
    train = _read_rows(data / "train.tsv")
    valid = _read_rows(data / "valid.tsv")
    assert len(train) == 1
    assert len(valid) == 0


def test_audio_suffix_config_is_honored_by_the_scan(tmp_path):
    """builder.py must not hardcode '.mp3': _scan_shard's missing_audio
    check has to key off builder.audio_suffix, or any corpus staged with a
    non-mp3 suffix (e.g. the .wav fixtures test recipes use to avoid
    needing an mp3 encoder) silently drops every row as missing_audio and
    ships an empty manifest.
    """
    root = tmp_path / "raw"
    shard_dir = root / "emilia" / "EN" / "EN-B000000"
    shard_dir.mkdir(parents=True)
    (shard_dir / "EN_B00000_S00000_W000000.wav").write_bytes(b"\x00")
    (shard_dir / "EN_B00000_S00000_W000000.json").write_text(
        json.dumps(
            {
                "id": "EN_B00000_S00000_W000000",
                "wav": "ignored.wav",
                "text": "a wav-suffixed utterance",
                "duration": 4.0,
                "speaker": "EN_B00000_S00000",
                "language": "en",
                "dnsmos": 3.0,
            }
        ),
        encoding="utf-8",
    )

    recipe_dir = tmp_path / "recipe"
    (recipe_dir / "dataset").mkdir(parents=True)
    (recipe_dir / "dataset" / "config.yaml").write_text(
        f"""
builder:
  corpus_root: {root}
  langs: [EN]
  data_path: data
  val_ratio: 0.2
  seed: 42
  min_duration: 0.3
  max_duration: 30.0
  strict_text_filters: false
  audio_suffix: .wav
  manifest_paths:
    train: manifest/train.tsv
    valid: manifest/valid.tsv
  shard_table_path: manifest/shards.txt
dataset:
  split_manifest_paths:
    train: manifest/train.tsv
    valid: manifest/valid.tsv
""",
        encoding="utf-8",
    )

    EmiliaBuilder().build(recipe_dir=recipe_dir)
    data = recipe_dir / "data" / "manifest"
    rows = _read_rows(data / "train.tsv") + _read_rows(data / "valid.tsv")
    assert len(rows) == 1
    report = json.loads((data / "report.json").read_text("utf-8"))
    assert report["dropped"].get("missing_audio", 0) == 0


def test_embedded_newline_in_text_does_not_corrupt_manifest_rows(tmp_path):
    """A JSON `text` field containing a literal newline or tab must not
    split or shift the TSV row it's written into -- normalize_text strips
    it (see test_filters.py), and this is the end-to-end check that the
    manifest still has exactly one row per utterance with 5 fields."""
    root = tmp_path / "raw"
    _write_utt(
        root / "emilia/EN/EN-B000000",
        "EN_B00000_S00000_W000000",
        "EN_B00000_S00000",
        "an utterance\nwith an embedded\tnewline and tab",
        4.0,
        "en",
    )

    recipe_dir = tmp_path / "recipe"
    (recipe_dir / "dataset").mkdir(parents=True)
    (recipe_dir / "dataset" / "config.yaml").write_text(
        f"""
builder:
  corpus_root: {root}
  langs: [EN]
  data_path: data
  val_ratio: 0.2
  seed: 42
  min_duration: 0.3
  max_duration: 30.0
  strict_text_filters: false
  manifest_paths:
    train: manifest/train.tsv
    valid: manifest/valid.tsv
  shard_table_path: manifest/shards.txt
dataset:
  split_manifest_paths:
    train: manifest/train.tsv
    valid: manifest/valid.tsv
""",
        encoding="utf-8",
    )

    EmiliaBuilder().build(recipe_dir=recipe_dir)
    data = recipe_dir / "data" / "manifest"
    lines = (data / "train.tsv").read_text("utf-8").splitlines()
    assert len(lines) == 1
    fields = lines[0].split("\t")
    assert len(fields) == 5
    assert "\n" not in fields[4] and "\t" not in fields[4]


def test_multi_worker_build_matches_single_worker(recipe, monkeypatch):
    """The ProcessPoolExecutor path must run and match the sequential path.

    Production builds set EMILIA_BUILD_WORKERS > 1 to scan ~2,060 shard
    directories in parallel. This exercises that path directly instead of
    only reading the code, proving the job-tuple arguments are picklable,
    _scan_shard is importable in a worker process, and the result is
    byte-identical to the sequential (n_workers=1) build over the same
    fixture -- the actual determinism guarantee under the pool.
    """
    EmiliaBuilder().build(recipe_dir=recipe)
    data = recipe / "data" / "manifest"
    sequential_train = (data / "train.tsv").read_text("utf-8")
    sequential_valid = (data / "valid.tsv").read_text("utf-8")

    monkeypatch.setenv("EMILIA_BUILD_WORKERS", "2")
    EmiliaBuilder().build(recipe_dir=recipe)
    assert (data / "train.tsv").read_text("utf-8") == sequential_train
    assert (data / "valid.tsv").read_text("utf-8") == sequential_valid


# --- resumability -----------------------------------------------------------


def _cache_dir(recipe: Path) -> Path:
    return recipe / "data" / "manifest" / ".shard_cache"


def test_build_checkpoints_every_shard(recipe):
    """Each shard leaves a .tsv and a .json behind, so work is never redone."""
    EmiliaBuilder().build(recipe_dir=recipe)
    cache = _cache_dir(recipe)
    tsvs = sorted(cache.glob("*.tsv"))
    metas = sorted(cache.glob("*.json"))
    metas = [m for m in metas if m.name != "fingerprint.json"]
    assert tsvs, "no shard checkpoints written"
    assert len(tsvs) == len(metas)
    assert (cache / "fingerprint.json").is_file()
    # no temp files survive a clean run
    assert not list(cache.glob("*.tmp"))


def test_resume_reuses_cached_shards_and_gives_identical_output(recipe, monkeypatch):
    """A resumed build must produce byte-identical manifests.

    Simulates the walltime kill this exists for: build once, delete the final
    manifests but KEEP the shard cache, then rebuild. The second run must scan
    nothing and still reproduce the same train/valid split, since the split is
    a seeded shuffle over rows read back in shard-index order.
    """
    import egs3.emilia.tts.dataset.builder as builder_mod

    EmiliaBuilder().build(recipe_dir=recipe)
    train = (recipe / "data" / "manifest" / "train.tsv").read_text("utf-8")
    valid = (recipe / "data" / "manifest" / "valid.tsv").read_text("utf-8")

    (recipe / "data" / "manifest" / "train.tsv").unlink()
    (recipe / "data" / "manifest" / "valid.tsv").unlink()

    calls = []
    real = builder_mod._scan_shard

    def counting(args):
        calls.append(args[1])
        return real(args)

    monkeypatch.setattr(builder_mod, "_scan_shard", counting)
    EmiliaBuilder().build(recipe_dir=recipe)

    assert calls == [], f"resumed run rescanned shards {calls}"
    assert (recipe / "data" / "manifest" / "train.tsv").read_text("utf-8") == train
    assert (recipe / "data" / "manifest" / "valid.tsv").read_text("utf-8") == valid


def test_partial_cache_only_rescans_missing_shards(recipe, monkeypatch):
    """Deleting one shard's checkpoint rescans exactly that shard."""
    import egs3.emilia.tts.dataset.builder as builder_mod

    EmiliaBuilder().build(recipe_dir=recipe)
    expected = (recipe / "data" / "manifest" / "train.tsv").read_text("utf-8")

    cache = _cache_dir(recipe)
    victim = sorted(cache.glob("*.tsv"))[0]
    idx = int(victim.stem)
    victim.unlink()
    (cache / f"{idx:06d}.json").unlink()

    calls = []
    real = builder_mod._scan_shard

    def counting(args):
        calls.append(args[1])
        return real(args)

    monkeypatch.setattr(builder_mod, "_scan_shard", counting)
    EmiliaBuilder().build(recipe_dir=recipe)

    assert calls == [idx], f"expected only shard {idx} rescanned, got {calls}"
    assert (recipe / "data" / "manifest" / "train.tsv").read_text("utf-8") == expected


def test_stale_cache_is_refused_not_silently_reused(recipe):
    """Changing a filter-relevant setting must invalidate the cache loudly.

    Silently reusing shards built under different duration bounds or filter
    semantics would mix incompatible rows across a 38M-row corpus, and nothing
    downstream could detect it.
    """
    import yaml

    EmiliaBuilder().build(recipe_dir=recipe)

    cfg_path = recipe / "dataset" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text("utf-8"))
    cfg["builder"]["min_duration"] = 1.5  # was 0.3
    cfg_path.write_text(yaml.safe_dump(cfg), "utf-8")

    with pytest.raises(RuntimeError, match="different settings"):
        EmiliaBuilder().build(recipe_dir=recipe)
