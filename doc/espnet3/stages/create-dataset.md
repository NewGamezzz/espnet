---
title: ESPnet3 Create Dataset Stage
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Create Dataset Stage

The `create_dataset` stage prepares raw data for training. It is the place to
download archives, extract corpora, and build manifests or directory layouts
that later stages consume.

## How to run

```bash
python run.py --stages create_dataset --train_config conf/train.yaml
```

The log directory for this stage is set to `create_dataset.dataset_dir` (or
`dataset_dir` / `data_dir` as a fallback). The dataset is also created under the
same directory, so logs and generated data live together in the dataset output
path.

In ESPnet3, `create_dataset` is driven by a callable defined in your
`train.yaml` config (see the [train config reference](../core/config/training.md)).
The system resolves the function and passes the remaining config keys as keyword
arguments.

## Where it is configured

In `conf/train*.yaml`, set `create_dataset.func` and any required arguments:

```yaml
create_dataset:
  func: src.create_dataset.create_dataset
  dataset_dir: ${dataset_dir}
  # optional args specific to the recipe
  archive_path: ${recipe_dir}/../../egs2/mini_an4/asr1/downloads.tar.gz
```

Config keys map to function arguments. For example, if your recipe defines:

```python
def create_dataset(dataset_dir: Path, *, archive_path: Path | None = None) -> None:
    dataset_dir = Path(dataset_dir)
    archive = Path(archive_path) if archive_path else None

    an4_root = ensure_extracted(archive, dataset_dir)
    sph2pipe = shutil.which("sph2pipe")
    train = prepare_split(an4_root, dataset_dir, "train", sph2pipe)
    test = prepare_split(an4_root, dataset_dir, "test", sph2pipe)

    manifest_dir = dataset_dir / "manifest"
    write_manifest(manifest_dir / "train_dev.tsv", train[:1])
    write_manifest(manifest_dir / "train_nodev.tsv", train[1:])
    write_manifest(manifest_dir / "test.tsv", test)
```

then the stage will call it with the values from your config block.

## Where the code lives

Typical recipe structure:

```
egs3/<recipe>/<system>/
  conf/
    train.yaml
  src/
    create_dataset.py
    dataset.py
```

`create_dataset.py` prepares files and manifests. `dataset.py` defines the
Torch dataset class consumed by `train.yaml`.

## Example: Mini AN4 (manifest-based)

`egs3/mini_an4/asr/src/create_dataset.py`:

- Extracts the archive.
- Converts SPH to WAV using `sph2pipe`.
- Writes tab-separated manifests under `dataset_dir/manifest/`.

Resulting layout:

```
${dataset_dir}/
  wav/
    train/
    test/
  manifest/
    train_dev.tsv
    train_nodev.tsv
    test.tsv
```

## Example: LibriSpeech 100h (download + extract)

`egs3/librispeech_100/asr/src/create_dataset.py` downloads and extracts splits
from OpenSLR into `dataset_dir/LibriSpeech/` without extra preprocessing.

### Minimal conceptual code (from egs3/librispeech_100/asr/src/create_dataset.py)

Below is the smallest conceptual snippet that shows the intent. The real
implementation adds split checks and re-run safety.

```python
dataset_dir = Path(dataset_dir)
librispeech_root = dataset_dir / "LibriSpeech"

for split in requested_splits:
    filename = SPLITS[split]
    url = f"{OPENSLR_BASE_URL}/{filename}"
    archive_path = dataset_dir / filename
    extracted_dir = librispeech_root / filename.replace(".tar.gz", "")

    if not extracted_dir.exists():
        download_url(url=url, dst_path=archive_path, logger=logger)
        extract_targz(archive_path, dataset_dir, logger)
```

### Planned usage (from egs3/librispeech_100/asr/src/dataset.py)

The dataset loader is expected to read the on-disk tree under
`dataset_dir/LibriSpeech/...` directly. A minimal usage looks like:

```python
dataset = LibriSpeechDataset(
    data_dir=dataset_dir,         # or data_dir=dataset_dir / "LibriSpeech"
    split="train-clean-100",
)

item = dataset[0]
speech = item["speech"]  # np.float32 waveform
text = item["text"]      # transcript string
```

Below is a tiny conceptual snippet of how the dataset is built internally:

```python
class LibriSpeechDataset:
    def __init__(self, data_dir, split):
        root = resolve_librispeech_root(data_dir)
        split_dir = root / split

        self.examples = []
        for transcript_path in walk("*.trans.txt", split_dir):
            for line in transcript_path.read_text().splitlines():
                utt_id, *words = line.split()
                flac_path = transcript_path.parent / f"{utt_id}.flac"
                if flac_path.is_file():
                    self.examples.append((flac_path, " ".join(words)))

    def __getitem__(self, idx):
        flac_path, text = self.examples[idx]
        speech = soundfile.read(flac_path, dtype="float32")
        return {"speech": speech, "text": text}
```

Example logs when the download package runs:

```
INFO:espnet3.systems.asr.system:ASRSystem.create_dataset(): starting dataset creation process
INFO:espnet3.systems.asr.system:Creating dataset with function src.create_dataset.create_dataset
2026-01-21 01:50:52 | INFO | create_dataset | Start processing split: train.clean.100
2026-01-21 01:50:52 | INFO | create_dataset | Start download: train-clean-100.tar.gz
2026-01-21 01:50:52 | INFO | create_dataset | Target directory: /data/user_data/msomeki/espnet3/egs3/librispeech_100/asr/download/LibriSpeech
2026-01-21 01:50:53 | INFO | create_dataset | Downloading train-clean-100.tar.gz: 0% (0.0MB / 6091.4MB)
2026-01-21 01:51:04 | INFO | create_dataset | Downloading train-clean-100.tar.gz: 5% (304.6MB / 6091.4MB)
2026-01-21 01:51:14 | INFO | create_dataset | Downloading train-clean-100.tar.gz: 10% (609.1MB / 6091.4MB)
2026-01-21 01:51:24 | INFO | create_dataset | Downloading train-clean-100.tar.gz: 15% (913.7MB / 6091.4MB)
2026-01-21 01:51:34 | INFO | create_dataset | Downloading train-clean-100.tar.gz: 20% (1218.3MB / 6091.4MB)
```

### Notes

- The `create_dataset` stage should be deterministic and safe to re-run.
- Keep outputs in `dataset_dir` so later stages can reuse them without
  rebuilding.
