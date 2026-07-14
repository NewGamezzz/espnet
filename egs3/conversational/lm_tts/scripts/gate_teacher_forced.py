"""Teacher-forced forward gate for BagPiper.

Loads BagPiper via ``load_bagpiper`` and runs the model's real training forward
(``model(**batch)`` returning a cross-entropy loss) on ONE ``dev_multi_talker``
sample, printing the loss.

Batch construction (documented per Task 5 amendment 5): we drive the *real*
synced ESPnet preprocessing pipeline -- ``SpeechLMJobTemplate.build_preprocessor()``
+ ``SpeechLMPreprocessor.collate_fn`` -- rather than the DataIteratorFactory,
which would require registered datasets / tokenizer infrastructure that does not
exist locally. We hand-assemble the single sample into the preprocessor's
``{"dialogue": [[role, io_name, content], ...]}`` input form (mapping the SFT
schema's ``audio`` modality to the ``discrete_audio`` output IO and loading the
referenced wav), then let the genuine preprocessor tokenize text, place the audio
placeholder streams, build the delay-interleaved multi-stream sequence, and set
the assistant-region loss mask. The Xcodec tokenization of the wav happens inside
the model forward (``_embed`` -> ``encode_batch``), exactly as in training.

``--build-only`` builds and shape-checks the batch WITHOUT loading the 16.9 GB
model (RAM-free); the default also runs the forward.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from load_bagpiper import load_bagpiper  # noqa: E402

CONF = os.path.join(HERE, "..", "conf", "bagpiper_train_config.yaml")
CKPT_DIR = os.path.join(HERE, "..", "downloads", "bagpiper", "speechlm-qwen3-8b")
DEV = os.path.join(HERE, "..", "downloads", "bagpiper_sft", "dev_multi_talker")
DIALOGUES = os.path.join(DEV, "stages", "v1", "stage5_dialogues", "dialogues.jsonl")
AUDIO_ROOT = os.path.join(DEV, "audio")

# SFT-schema modality -> preprocessor IO name.
_IO_FOR_MODALITY = {"text": "text", "audio": "discrete_audio"}


def _resolve_wav(stale_path):
    """The recorded wav path points at the original training host; resolve by
    basename against the shipped audio tree."""
    base = os.path.basename(stale_path)
    for root, _dirs, files in os.walk(AUDIO_ROOT):
        if base in files:
            return os.path.join(root, base)
    raise FileNotFoundError(f"wav {base!r} not found under {AUDIO_ROOT}")


def _load_wav(path):
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)  # [samples, channels]
    return wav.T, sr  # -> [channels, samples]


def build_dialogue(record):
    """Map an SFT record's messages to the preprocessor dialogue form."""
    dialogue = []
    for role, modality, content in record["messages"]:
        io_name = _IO_FOR_MODALITY[modality]
        if modality == "audio":
            content = _load_wav(_resolve_wav(content))
        dialogue.append([role, io_name, content])
    return dialogue


def build_batch(config):
    from espnet2.speechlm.model.speechlm.speechlm_job import SpeechLMJobTemplate

    # is_train=True so assistant turns are kept and get a loss mask.
    job = SpeechLMJobTemplate(config, is_train=True)
    preproc = job.build_preprocessor()

    record = json.loads(open(DIALOGUES).readline())
    dialogue = build_dialogue(record)
    key = (None, None, None)  # ignored when "dialogue" is present
    data_dict = {"dialogue": dialogue}
    batch = preproc.collate_fn([(key, data_dict)])
    return batch, record["example_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    with open(CONF) as f:
        config = yaml.safe_load(f)

    batch, example_id = build_batch(config)
    print(f"sample: {example_id}")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:26s} {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k:26s} {type(v).__name__}={v}")

    if args.build_only:
        print("\n--build-only: batch built OK (model forward skipped).")
        return 0

    model = load_bagpiper(CONF, CKPT_DIR, device=args.device, dtype=torch.bfloat16)
    # Dtype landmines (docs/bagpiper-findings.md, Task 5 + Delta gate run):
    # the model (incl. the Xcodec codec) is bf16, but the collated batch
    # carries float32 wav (discrete_audio_feats) and float64 loss_masks --
    # both clash inside the forward (codec conv, _loss masked_scatter_).
    # The reference pipeline (espnet2/speechlm/bin/inference.py) solves this
    # with to_device(..., dtype=...), which casts float tensors to the model
    # dtype and leaves int tensors alone. Use the same utility here -- the
    # fix lives in the gate script, never in espnet2/.
    from espnet2.speechlm.utils.data import to_device

    batch = to_device(batch, args.device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model(**batch)
    loss = out["loss"]
    print(f"\nTEACHER-FORCED LOSS: {float(loss):.6f}")
    print("stats:", {k: float(v) for k, v in out.get("stats", {}).items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
