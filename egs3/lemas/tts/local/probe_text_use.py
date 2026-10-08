"""Does a checkpoint read its text lane? Teacher-forced loss under text ablations.

Runs validation batches through the training loss with identical noise and
time draws under four text conditions: the true phones, the phones shuffled
within the row, random phone ids of the same length, and no phones (filler).
A model that uses the text must score the true phones lowest; a model that
ignores them scores all four alike. Role tokens, the language tag and the
prompt regions are kept in every condition, so only the target phones vary.

Usage (recipe dir, GPU):
    python local/probe_text_use.py --train_config conf/training_...yaml \\
        --ckpt exp/<tag>/backup_step199901.ckpt --n_batches 100
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from dataset.dataset import LEMASDataset
from src.layout import TokenTable
from src.sampler import BlockBatchSampler
from src.text.lemas_phonemizer import LANGS, special_tokens

CONDITIONS = ("true", "shuffled", "random", "none")


def load_model(train_config: str, ckpt: str, device: str):
    """Build the model from the training config and load the EMA weights."""
    cfg = OmegaConf.load(train_config)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model)
    state = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    prefix = "ema_model."
    ema = {
        k[len(prefix) :]: v
        for k, v in state["ema_model_state_dict"].items()
        if k.startswith(prefix)
    }
    missing, unexpected = model.load_state_dict(ema, strict=False)
    print(
        f"loaded EMA of step {state['global_step']}: "
        f"{len(missing)} missing, {len(unexpected)} unexpected keys"
    )
    model.cfm.audio_drop_prob = 0.0  # the probe wants every row conditioned
    model.cfm.cond_drop_prob = 0.0
    return cfg, model.to(device).eval()


def ablate(text: np.ndarray, cond: str, table: TokenTable, rng) -> np.ndarray:
    """Return the text lane with the target phones changed per ``cond``."""
    tags = set()
    for lang in LANGS:  # a token table may hold a subset of the languages
        try:
            tags.add(table.tag(lang))
        except KeyError:
            pass
    head = next(i for i, t in enumerate(text.tolist()) if t in tags) + 1
    phones = text[head:].copy()
    if cond == "shuffled":
        rng.shuffle(phones)
    elif cond == "random":
        specials = {table.id(t) for t in special_tokens()} | {table.unk, 0}
        pool = np.array([i for i in range(table.size) if i not in specials])
        phones = rng.choice(pool, size=len(phones))
    elif cond == "none":
        return text[:head]
    return np.concatenate([text[:head], phones])


def collate(samples, condition, table, rng):
    texts = [ablate(s["text"], condition, table, rng) for s in samples]
    speech = [s["speech"] for s in samples]
    b = len(samples)
    text = torch.zeros(b, max(len(t) for t in texts), dtype=torch.long)
    wav = torch.zeros(b, max(len(w) for w in speech))
    for i, (t, w) in enumerate(zip(texts, speech)):
        text[i, : len(t)] = torch.from_numpy(np.asarray(t))
        wav[i, : len(w)] = torch.from_numpy(w)
    return dict(
        text=text,
        text_lengths=torch.tensor([len(t) for t in texts]),
        speech=wav,
        speech_lengths=torch.tensor([len(w) for w in speech]),
        cond_frames=torch.tensor([int(s["cond_frames"][0]) for s in samples]),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train_config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_batches", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg, model = load_model(args.train_config, args.ckpt, args.device)
    ds = LEMASDataset(
        split="valid",
        recipe_dir=".",
        token_list=cfg.token_list,
        prompt_config=OmegaConf.to_container(cfg.prompt_config),
        seed=int(cfg.seed),
    )
    table = TokenTable(cfg.token_list)
    sampler = BlockBatchSampler(
        ds, batch_bins=int(cfg.batch_sampler.batch_bins), n_mels=100, seed=0
    )
    batches = list(sampler)[: args.n_batches]
    rng = np.random.default_rng(0)
    losses = {c: [] for c in CONDITIONS}
    with torch.no_grad():
        for bi, idx in enumerate(batches):
            samples = [ds[i] for i in idx]
            for c in CONDITIONS:
                batch = {
                    k: v.to(args.device)
                    for k, v in collate(samples, c, table, rng).items()
                }
                torch.manual_seed(1000 + bi)  # same x0 and t for every condition
                random.seed(1000 + bi)
                with torch.autocast(
                    args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"
                ):
                    loss, _, _ = model(**batch)
                losses[c].append(float(loss))
            if bi % 20 == 0:
                print(
                    f"batch {bi}: "
                    + "  ".join(f"{c}={losses[c][-1]:.4f}" for c in CONDITIONS),
                    flush=True,
                )
    arr = {c: np.array(v) for c, v in losses.items()}
    print(
        "\n== teacher-forced loss, %d batches (mean +- sem), paired vs true"
        % len(batches)
    )
    out = {}
    for c in CONDITIONS:
        d = arr[c] - arr["true"]
        out[c] = dict(
            mean=float(arr[c].mean()),
            sem=float(arr[c].std(ddof=1) / np.sqrt(len(d))),
            delta_vs_true=float(d.mean()),
            delta_sem=float(d.std(ddof=1) / np.sqrt(len(d))),
            frac_batches_worse_than_true=float((d > 0).mean()),
        )
        print(
            "  %-9s %.4f +- %.4f   delta %+.4f +- %.4f   worse than true in %3.0f%% of batches"
            % (
                c,
                out[c]["mean"],
                out[c]["sem"],
                out[c]["delta_vs_true"],
                out[c]["delta_sem"],
                100 * out[c]["frac_batches_worse_than_true"],
            )
        )
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                dict(ckpt=args.ckpt, n_batches=len(batches), results=out), indent=1
            )
        )


if __name__ == "__main__":
    main()
