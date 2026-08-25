"""EmiliaDataset columnar manifest and path reconstruction."""

import numpy as np
import pytest
import soundfile as sf

from egs3.emilia.tts.dataset.dataset import EmiliaDataset


@pytest.fixture
def built(tmp_path):
    """A recipe dir with manifests and real audio on disk."""
    recipe = tmp_path / "recipe"
    manifest = recipe / "data" / "manifest"
    manifest.mkdir(parents=True)
    (recipe / "dataset").mkdir(parents=True)
    corpus = tmp_path / "raw"

    rows = [
        ("EN_B00012_S00001_W000000", 0, "EN", 1.0, "first utterance"),
        ("EN_B00012_S00002_W000000", 1, "EN", 2.0, "second utterance"),
        ("ZH_B00000_S00001_W000000", 2, "ZH", 1.5, "你好，世界"),
    ]
    shards = ["EN/EN-B000120", "EN/EN-B000121", "ZH/ZH-B000000"]
    (manifest / "shards.txt").write_text("\n".join(shards) + "\n", "utf-8")
    with (manifest / "train.tsv").open("w", encoding="utf-8") as fh:
        for utt_id, shard, lang, dur, text in rows:
            fh.write(f"{utt_id}\t{shard}\t{lang}\t{dur!r}\t{text}\n")

    for utt_id, shard_idx, _lang, dur, _text in rows:
        d = corpus / "emilia" / shards[shard_idx]
        d.mkdir(parents=True, exist_ok=True)
        sf.write(
            d / f"{utt_id}.wav", np.zeros(int(dur * 16000), dtype=np.float32), 16000
        )

    (recipe / "dataset" / "config.yaml").write_text(
        f"""
builder:
  corpus_root: {corpus}
  langs: [EN, ZH]
  data_path: data
  val_ratio: 0.01
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
    return recipe


def test_len_and_text(built):
    ds = EmiliaDataset(split="train", recipe_dir=built, load_speech=False)
    assert len(ds) == 3
    assert ds[0]["text"] == "first utterance"
    assert ds[2]["text"] == "你好，世界"


def test_sub_shard_paths_resolve(built):
    """Both utterances share the B00012 prefix but live in different dirs."""
    ds = EmiliaDataset(
        split="train", recipe_dir=built, inference=True, load_speech=False
    )
    assert ds[0]["wav_path"].endswith("EN/EN-B000120/EN_B00012_S00001_W000000.wav")
    assert ds[1]["wav_path"].endswith("EN/EN-B000121/EN_B00012_S00002_W000000.wav")


def test_speech_is_loaded_and_resampled(built):
    ds = EmiliaDataset(split="train", recipe_dir=built, fs=24000)
    sample = ds[1]
    assert sample["speech"].dtype == np.float32
    # 2.0 s at 24 kHz, allowing resampler edge effects.
    assert abs(len(sample["speech"]) - 48000) < 100


def test_columns_are_numpy_not_python_objects(built):
    """Spec 5.3: fork sharing requires numpy columns, not per-row objects."""
    ds = EmiliaDataset(split="train", recipe_dir=built, load_speech=False)
    assert isinstance(ds.durations, np.ndarray)
    assert ds.durations.dtype == np.float32
    assert isinstance(ds._shard_idx, np.ndarray)
    assert isinstance(ds._text_offsets, np.ndarray)
    assert isinstance(ds._text_buffer, (bytes, bytearray, np.ndarray))


def test_n_frames_is_analytic(built):
    ds = EmiliaDataset(split="train", recipe_dir=built, load_speech=False)
    frames = ds.n_frames(hop_length=256, sample_rate=24000)
    assert frames.dtype == np.int32
    # 1.0 s -> 24000/256 = 93.75 -> 1 + 24000//256 = 94
    assert frames[0] == 94
    assert frames[1] == 1 + int(2.0 * 24000) // 256


def test_missing_manifest_raises(tmp_path, built):
    with pytest.raises(FileNotFoundError):
        EmiliaDataset(split="test", recipe_dir=built, load_speech=False)


def test_leading_space_survives_the_tsv_round_trip(tmp_path):
    """A leading space must reach the model, not just the manifest.

    normalize_text deliberately stops stripping so our text matches
    upstream's. That is only worth anything if the space survives being
    written into a tab-separated row and read back out: builder.py writes the
    text as the last unquoted field, and dataset.py parses with
    line.rstrip("\n").split("\t", 4) -- which preserves leading and trailing
    spaces but would be broken by a naive .strip() on either side.
    """
    from egs3.emilia.tts.dataset.dataset import EmiliaDataset

    recipe = tmp_path / "recipe"
    manifest = recipe / "data" / "manifest"
    manifest.mkdir(parents=True)
    (recipe / "dataset").mkdir(parents=True)

    text = " You can help my mother and you- No. "
    (manifest / "shards.txt").write_text("EN/EN-B000120\n", "utf-8")
    (manifest / "train.tsv").write_text(
        f"EN_B00012_S00001_W000000\t0\tEN\t1.0\t{text}\n", "utf-8"
    )
    (recipe / "dataset" / "config.yaml").write_text(
        """
builder:
  corpus_root: /nonexistent
  langs: [EN, ZH]
  data_path: data
  val_ratio: 0.01
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
        "utf-8",
    )

    ds = EmiliaDataset(split="train", recipe_dir=str(recipe), load_speech=False)
    assert ds[0]["text"] == text, repr(ds[0]["text"])
    assert ds[0]["text"].startswith(" ")
    assert ds[0]["text"].endswith(" ")
