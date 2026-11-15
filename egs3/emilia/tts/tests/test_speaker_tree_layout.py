"""The ``speaker_tree`` corpus layout (the Emilia copy shared on NCSA Delta).

That copy stores ``<LANG>/<LANG>_Bxxxxx/<speaker>/mp3/<utt_id>.mp3`` with one
``<LANG>/<LANG>_Bxxxxx.jsonl`` of records per batch, instead of the flat
per-utterance ``.json`` + ``.mp3`` sub-shards the default layout reads.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.emilia.tts.dataset.builder import EmiliaBuilder
from egs3.emilia.tts.dataset.dataset import EmiliaDataset

RECIPE_ROOT = Path(__file__).resolve().parents[1]

_CONFIG = """
builder:
  corpus_root: {corpus}
  corpus_layout: speaker_tree
  langs: [EN, ZH]
  data_path: data
  val_ratio: 0.2
  seed: 42
  min_duration: 0.3
  max_duration: 30.0
  strict_text_filters: false
  audio_suffix: {suffix}
  manifest_paths:
    train: manifest/train.tsv
    valid: manifest/valid.tsv
  shard_table_path: manifest/shards.txt
dataset:
  split_manifest_paths:
    train: manifest/train.tsv
    valid: manifest/valid.tsv
"""

# (utt_id, speaker, text, duration, language, audio_on_disk)
_RECORDS = [
    ("EN_B00012_S00001_W000000", "EN_B00012_S00001", " first english", 4.0, "en", True),
    (
        "EN_B00012_S00002_W000000",
        "EN_B00012_S00002",
        " second english",
        6.0,
        "en",
        True,
    ),
    # Blocklisted speaker.
    ("EN_B00013_S00913_W000000", "EN_B00013_S00913", " blocklisted", 5.0, "en", True),
    # Too short.
    ("EN_B00013_S00001_W000000", "EN_B00013_S00001", " tiny", 0.2, "en", True),
    # Listed in the jsonl but the audio file is absent.
    ("EN_B00013_S00002_W000000", "EN_B00013_S00002", " no audio", 5.0, "en", False),
    ("ZH_B00000_S00001_W000000", "ZH_B00000_S00001", "你好,世界", 3.0, "zh", True),
]


def _write_recipe(tmp_path: Path, corpus: Path, suffix: str) -> Path:
    recipe = tmp_path / "recipe"
    (recipe / "dataset").mkdir(parents=True)
    (recipe / "dataset" / "config.yaml").write_text(
        _CONFIG.format(corpus=corpus, suffix=suffix), encoding="utf-8"
    )
    return recipe


@pytest.fixture
def corpus(tmp_path):
    """A speaker-tree corpus: batch jsonl files plus real audio."""
    root = tmp_path / "shared"
    lines: dict[Path, list[str]] = {}
    for utt_id, speaker, text, duration, language, on_disk in _RECORDS:
        lang, batch = utt_id.split("_")[0], "_".join(utt_id.split("_")[:2])
        wav_rel = f"{batch}/{speaker}/mp3/{utt_id}.wav"
        if on_disk:
            wav = root / "emilia" / lang / wav_rel
            wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(wav, np.zeros(int(duration * 16000), dtype=np.float32), 16000)
        lines.setdefault(root / "emilia" / lang / f"{batch}.jsonl", []).append(
            json.dumps(
                {
                    "id": utt_id,
                    "wav": wav_rel,
                    "text": text,
                    "duration": duration,
                    "speaker": speaker,
                    "language": language,
                    "dnsmos": 3.0,
                },
                ensure_ascii=False,
            )
        )
    for path, rows in lines.items():
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def recipe(tmp_path, corpus):
    return _write_recipe(tmp_path, corpus, ".wav")


def _read_rows(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        utt_id, shard_idx, lang, duration, text = line.split("\t", 4)
        rows.append((utt_id, int(shard_idx), lang, float(duration), text))
    return rows


def _all_rows(recipe: Path):
    data = recipe / "data" / "manifest"
    return _read_rows(data / "train.tsv") + _read_rows(data / "valid.tsv")


def test_builder_reads_records_from_the_batch_jsonl(recipe):
    EmiliaBuilder().build(recipe_dir=recipe)
    by_id = {r[0]: r for r in _all_rows(recipe)}
    assert set(by_id) == {
        "EN_B00012_S00001_W000000",
        "EN_B00012_S00002_W000000",
        "ZH_B00000_S00001_W000000",
    }
    # Same row semantics as the default layout: text normalised per language,
    # the leading space Emilia ships on EN records preserved.
    assert by_id["EN_B00012_S00001_W000000"][2:] == ("EN", 4.0, " first english")
    assert by_id["ZH_B00000_S00001_W000000"][4] == "你好，世界"


def test_builder_shard_table_lists_one_entry_per_batch_jsonl(recipe):
    EmiliaBuilder().build(recipe_dir=recipe)
    shards = (recipe / "data" / "manifest" / "shards.txt").read_text("utf-8")
    assert shards.splitlines() == ["EN/EN_B00012", "EN/EN_B00013", "ZH/ZH_B00000"]
    by_id = {r[0]: r for r in _all_rows(recipe)}
    assert by_id["ZH_B00000_S00001_W000000"][1] == 2


def test_builder_drops_records_whose_audio_is_missing(recipe):
    """A row without audio would kill a training rank weeks into the run."""
    EmiliaBuilder().build(recipe_dir=recipe)
    report = json.loads(
        (recipe / "data" / "manifest" / "report.json").read_text("utf-8")
    )
    assert report["kept"] == 3
    assert report["dropped"] == {"blocklist": 1, "duration": 1, "missing_audio": 1}


def test_builder_stops_on_an_audio_directory_it_cannot_read(recipe, corpus):
    """Permission denied is a fault of the site, not a property of the data.

    Counted as missing audio it silently halves the corpus (on Delta every ZH
    batch directory was owner-only and the build "succeeded" on EN alone).
    """
    if os.geteuid() == 0:
        pytest.skip("root can list a directory whatever its mode")
    locked = corpus / "emilia" / "ZH" / "ZH_B00000" / "ZH_B00000_S00001" / "mp3"
    locked.chmod(0o000)
    try:
        with pytest.raises(OSError, match="ZH_B00000_S00001"):
            EmiliaBuilder().build(recipe_dir=recipe)
    finally:
        locked.chmod(0o755)
    # The shard is not checkpointed, so a rerun after the fix scans it again.
    EmiliaBuilder().build(recipe_dir=recipe)
    assert "ZH_B00000_S00001_W000000" in {r[0] for r in _all_rows(recipe)}


def test_builder_is_source_prepared_in_speaker_tree_layout(recipe):
    assert EmiliaBuilder().is_source_prepared(recipe_dir=recipe) is True


def test_dataset_reads_the_audio_of_a_built_speaker_tree_manifest(recipe):
    EmiliaBuilder().build(recipe_dir=recipe)
    data = recipe / "data" / "manifest"
    split = "train" if _read_rows(data / "train.tsv") else "valid"
    ds = EmiliaDataset(split=split, recipe_dir=recipe, fs=24000, inference=True)
    sample = ds[0]
    utt_id = str(sample["utt_id"])
    duration = {r[0]: r[3] for r in _RECORDS}[utt_id]
    assert abs(len(sample["speech"]) - int(duration * 24000)) < 100


def test_dataset_resolves_audio_from_the_utt_id_not_the_shard_table(tmp_path, corpus):
    """A manifest built on the sub-shard layout must load on the speaker tree.

    That is what lets a run move between clusters without rebuilding the
    manifest: the rows (and so the train/valid split and every batch) stay
    byte-identical, only the audio location changes.
    """
    recipe = _write_recipe(tmp_path, corpus, ".wav")
    manifest = recipe / "data" / "manifest"
    manifest.mkdir(parents=True)
    # Sub-shard names from the other layout; no such directories exist here.
    (manifest / "shards.txt").write_text("EN/EN-B000120\nEN/EN-B000121\n", "utf-8")
    (manifest / "train.tsv").write_text(
        "EN_B00012_S00001_W000000\t0\tEN\t4.0\t first english\n"
        "EN_B00012_S00002_W000000\t1\tEN\t6.0\t second english\n",
        "utf-8",
    )

    ds = EmiliaDataset(split="train", recipe_dir=recipe, inference=True)
    assert ds[1]["wav_path"] == str(
        corpus / "emilia/EN/EN_B00012/EN_B00012_S00002/mp3/EN_B00012_S00002_W000000.wav"
    )
    assert len(ds[1]["speech"]) == 6 * 16000


def test_unknown_corpus_layout_is_rejected(tmp_path, corpus):
    recipe = _write_recipe(tmp_path, corpus, ".wav")
    cfg = recipe / "dataset" / "config.yaml"
    cfg.write_text(
        cfg.read_text("utf-8").replace("speaker_tree", "speaker_trees"), "utf-8"
    )
    with pytest.raises(ValueError, match="corpus_layout"):
        EmiliaBuilder().build(recipe_dir=recipe)


def test_cache_built_under_another_layout_is_not_reused(tmp_path, corpus):
    recipe = _write_recipe(tmp_path, corpus, ".wav")
    EmiliaBuilder().build(recipe_dir=recipe)
    cfg = recipe / "dataset" / "config.yaml"
    cfg.write_text(
        cfg.read_text("utf-8").replace("corpus_layout: speaker_tree", ""), "utf-8"
    )
    with pytest.raises(RuntimeError, match="different settings"):
        EmiliaBuilder().build(recipe_dir=recipe)


def test_shipped_config_defaults_to_the_staged_psc_corpus(monkeypatch):
    monkeypatch.delenv("EMILIA_CORPUS_ROOT", raising=False)
    monkeypatch.delenv("EMILIA_CORPUS_LAYOUT", raising=False)
    from egs3.emilia.tts.dataset.builder import _load_cfg

    cfg = _load_cfg(RECIPE_ROOT)
    assert cfg["corpus_root"] == "/ocean/projects/cis210027p/ttrachu/emilia_dataset/raw"
    assert cfg["corpus_layout"] == "sub_shards"


def test_shipped_config_takes_corpus_location_from_the_environment(monkeypatch):
    """Another cluster points the recipe at its copy without editing files."""
    monkeypatch.setenv("EMILIA_CORPUS_ROOT", "/work/hdd/bbjs/shared")
    monkeypatch.setenv("EMILIA_CORPUS_LAYOUT", "speaker_tree")
    from egs3.emilia.tts.dataset.builder import _load_cfg

    cfg = _load_cfg(RECIPE_ROOT)
    assert cfg["corpus_root"] == "/work/hdd/bbjs/shared"
    assert cfg["corpus_layout"] == "speaker_tree"
