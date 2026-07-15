"""Single-channel generation gate for BagPiper (Task 6 / deferred PSC item 2).

Loads BagPiper via ``load_bagpiper``, calls ``model.prepare_inference()`` and
``model.inference(...)`` (following ``espnet2/speechlm/bin/inference.py``) on the
PROMPT portion (system + user text only) of one ``dev_multi_talker`` sample, and
writes every generated audio segment to wav (plus any generated text, e.g. the
<think> block, to a sidecar .txt).

Audio sampling parameters per the plan (and the checkpoint config.json defaults):
temperature 0.8, topk 20, CFG 3.0, max_step 1024. Text: temperature 0.6, topk 20.

The multi-talker format is ONE token stream covering all speakers (see
docs/bagpiper-findings.md), so a single output wav with all speaker turns is the
expected artifact.
"""

import argparse
import json
import os
import sys

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from load_bagpiper import load_bagpiper  # noqa: E402
from gate_teacher_forced import (  # noqa: E402
    CONF,
    CKPT_DIR,
    DIALOGUES,
    _IO_FOR_MODALITY,
)

# Sampling parameters: plan Task 6 == checkpoint config.json baked defaults
# (audio_temperature 0.8, audio_topk 20, text_temperature 0.6, text_topk 20;
# client_all.py sends cfg 3.0 for audio generation).
INFERENCE_CONFIG = {
    "num_hypo": 1,
    "text": {"temperature": 0.6, "topk": 20, "max_step": 1024, "cfg": 1},
    "audio": {"temperature": 0.8, "topk": 20, "max_step": 1024, "cfg": 3.0},
}


def build_prompt_batch(config, sample_index=0):
    """Collate the prompt-only (system + user) part of a dev_multi_talker record.

    The preprocessor ends the user turn with <|eos|>; ``inference_segment``
    itself appends the <|assistant|> token, completing the trained prompt format
    ``<|bos|>...<|user|><|text|>text<|eos|><|assistant|>`` (chat_template.jinja).
    """
    from espnet2.speechlm.model.speechlm.speechlm_job import SpeechLMJobTemplate

    job = SpeechLMJobTemplate(config, is_train=False)
    preproc = job.build_preprocessor()

    with open(DIALOGUES) as f:
        for _ in range(sample_index):
            f.readline()
        record = json.loads(f.readline())

    dialogue = []
    for role, modality, content in record["messages"]:
        if role == "assistant":
            continue  # generation: prompt only, no teacher forcing
        dialogue.append([role, _IO_FOR_MODALITY[modality], content])

    batch = preproc.collate_fn([((None, None, None), {"dialogue": dialogue})])
    return batch, record


def main():
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-dir", default=os.path.join(HERE, "..", "exp", "gate_generate")
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    with open(CONF) as f:
        config = yaml.safe_load(f)

    batch, record = build_prompt_batch(config, args.sample_index)
    example_id = record["example_id"]
    print(f"sample: {example_id}")
    seqs_shape = tuple(batch["seqs"].shape)
    print(f"prompt seqs: {seqs_shape}")

    model = load_bagpiper(CONF, CKPT_DIR, device="cpu", dtype=torch.bfloat16)
    model.prepare_inference()
    model = model.to(args.device).eval()

    batch.pop("keys", None)
    batch.pop("loss_masks", None)  # not used by inference
    batch = {
        k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }

    with torch.no_grad():
        messages, _cache = model.inference(INFERENCE_CONFIG, **batch)

    os.makedirs(args.out_dir, exist_ok=True)
    n_wav = 0
    for idx, (role, modality, content) in enumerate(messages):
        if modality == "audio":
            audio, lengths, sample_rate = content
            wav = audio[0, 0, : lengths[0]].float().cpu().numpy()
            path = os.path.join(args.out_dir, f"{example_id}_seg{idx}.wav")
            sf.write(path, wav, sample_rate)
            n_wav += 1
            dur = len(wav) / sample_rate
            print(f"  [{idx}] {role}/audio -> {path} ({dur:.2f}s @ {sample_rate} Hz)")
        else:
            text = content[0] if isinstance(content, (list, tuple)) else content
            path = os.path.join(args.out_dir, f"{example_id}_seg{idx}.txt")
            with open(path, "w") as f:
                f.write(str(text))
            print(f"  [{idx}] {role}/{modality} -> {path}")
            print(f"      {str(text)[:300]}")

    if n_wav == 0:
        print("WARNING: no audio segment was generated")
        return 1
    print(f"\nGENERATION OK: {n_wav} wav segment(s) written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
