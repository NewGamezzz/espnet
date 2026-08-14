"""create_dataset -> create_shape compose on a synthetic mini-corpus."""

import json

import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

from egs3.emilia.tts.dataset.builder import EmiliaBuilder
from egs3.emilia.tts.dataset.dataset import EmiliaDataset
from egs3.emilia.tts.src.shape import write_shape_file
from egs3.emilia.tts.src.system import TTSSystem
from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.samplers.num_elements_batch_sampler import NumElementsBatchSampler


def test_builder_to_shape_to_sampler(tmp_path):
    corpus = tmp_path / "raw"
    recipe = tmp_path / "recipe"
    (recipe / "dataset").mkdir(parents=True)
    shard = corpus / "emilia" / "EN" / "EN-B000000"
    shard.mkdir(parents=True)

    for i in range(40):
        utt = f"EN_B00000_S{i:05d}_W000000"
        dur = 1.0 + (i % 8) * 0.5
        sf.write(
            shard / f"{utt}.wav", np.zeros(int(dur * 24000), dtype=np.float32), 24000
        )
        (shard / f"{utt}.json").write_text(
            json.dumps(
                {
                    "id": utt,
                    "wav": "x",
                    "text": f"utterance number {i}",
                    "duration": dur,
                    "speaker": f"EN_B00000_S{i:05d}",
                    "language": "en",
                    "dnsmos": 3.0,
                }
            ),
            encoding="utf-8",
        )

    (recipe / "dataset" / "config.yaml").write_text(
        f"""
builder:
  corpus_root: {corpus}
  langs: [EN]
  data_path: data
  val_ratio: 0.1
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

    EmiliaBuilder().build(recipe_dir=recipe)
    ds = EmiliaDataset(split="train", recipe_dir=recipe, load_speech=False)
    shape_path = recipe / "stats" / "train" / "feats_shape"
    n = write_shape_file(ds, shape_path, hop_length=256, sample_rate=24000, n_mels=100)
    assert n == len(ds)

    sampler = NumElementsBatchSampler(
        batch_bins=48000,
        shape_files=[str(shape_path)],
        min_batch_size=2,
    )
    batches = list(sampler)
    assert batches
    covered = sorted(int(k) for batch in batches for k in batch)
    assert covered == list(range(len(ds)))

    # The DDP path in dataloader.py:281 rejects any batch smaller than
    # world_size. NumElementsBatchSampler's trailing branch appends the
    # remainder regardless of min_batch_size when drop_last is False, so
    # the production config sets drop_last: true. Pin that here.
    dropping = NumElementsBatchSampler(
        batch_bins=48000,
        shape_files=[str(shape_path)],
        min_batch_size=8,
        drop_last=True,
    )
    assert all(len(b) >= 8 for b in dropping)

    # Every index the sampler emits must be loadable.
    loading = EmiliaDataset(split="train", recipe_dir=recipe, fs=24000)
    sample = loading[int(batches[0][0])]
    assert sample["speech"].dtype == np.float32
    assert isinstance(sample["text"], str)


def _write_synthetic_corpus(tmp_path, n_utts=30):
    """A synthetic Emilia-shaped corpus plus a recipe dataset/config.yaml.

    Returns the recipe dir. Does not run the builder.
    """
    corpus = tmp_path / "raw"
    recipe = tmp_path / "recipe"
    (recipe / "dataset").mkdir(parents=True)
    shard = corpus / "emilia" / "EN" / "EN-B000000"
    shard.mkdir(parents=True)

    for i in range(n_utts):
        utt = f"EN_B00000_S{i:05d}_W000000"
        dur = 1.0 + (i % 6) * 0.5
        sf.write(
            shard / f"{utt}.wav", np.zeros(int(dur * 24000), dtype=np.float32), 24000
        )
        (shard / f"{utt}.json").write_text(
            json.dumps(
                {
                    "id": utt,
                    "wav": "x",
                    "text": f"stage chain utterance {i}",
                    "duration": dur,
                    "speaker": f"EN_B00000_S{i:05d}",
                    "language": "en",
                    "dnsmos": 3.0,
                }
            ),
            encoding="utf-8",
        )

    (recipe / "dataset" / "config.yaml").write_text(
        f"""
builder:
  corpus_root: {corpus}
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
    return recipe


def test_ttssystem_create_shape_wires_config_to_shape_files(tmp_path):
    """No test anywhere else constructs a TTSSystem and calls
    .create_shape(): this is the stage's own glue (cfg['splits'],
    cfg['manifest_paths'][split], training_config.recipe_dir, the
    EmiliaDataset construction and the output path), not just the pure
    write_shape_file helper. A typo in a config key here would otherwise
    surface for the first time on PSC.
    """
    # val_ratio=0.2 over 30 utterances gives an asymmetric 24/6 train/valid
    # split (seed=42). That asymmetry is load-bearing: it is what would
    # catch a swapped cfg["manifest_paths"][split] lookup in create_shape,
    # which two equal-sized splits would let through silently.
    recipe = _write_synthetic_corpus(tmp_path)
    EmiliaBuilder().build(recipe_dir=recipe)

    manifest_dir = recipe / "data" / "manifest"
    exp_dir = tmp_path / "exp"
    stats_dir = exp_dir / "stats"

    training_config = OmegaConf.create(
        {
            "recipe_dir": str(recipe),
            "exp_dir": str(exp_dir),
            "stats_dir": str(stats_dir),
            "create_shape": {
                "splits": ["train", "valid"],
                "save_path": str(stats_dir),
                "hop_length": 256,
                "sample_rate": 24000,
                "n_mels": 100,
                "manifest_paths": {
                    "train": str(manifest_dir / "train.tsv"),
                    "valid": str(manifest_dir / "valid.tsv"),
                },
            },
        }
    )

    system = TTSSystem(training_config=training_config)
    system.create_shape()

    for split in ("train", "valid"):
        manifest_lines = (manifest_dir / f"{split}.tsv").read_text("utf-8").splitlines()
        shape_path = stats_dir / split / "feats_shape"
        assert shape_path.is_file()

        loaded = load_num_sequence_text(str(shape_path), loader_type="csv_int")
        assert len(loaded) == len(manifest_lines)
        for i in range(len(manifest_lines)):
            t, d = loaded[str(i)]
            assert t > 0
            assert d == 100
