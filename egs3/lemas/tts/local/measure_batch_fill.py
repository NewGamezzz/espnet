"""How full are the training batches? Real frames against the sampler's bound.

``BlockBatchSampler`` cuts batches on ``LEMASDataset.frames_bound``, which
assumes both prompts at their longest, so a dual-prompt batch carries fewer
real frames than ``batch_bins`` allows; a stock-layout batch has no prompts
and fills to its bound. This walks the configured sampler over a few epochs
and adds up ``drawn_frames`` (the served sample lengths, no audio read).

Usage (recipe dir, cpu node):
    python local/measure_batch_fill.py conf/training_....yaml [...] --epochs 3
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Sequence

from dataset.dataset import LEMASDataset
from omegaconf import OmegaConf
from src.sampler import BlockBatchSampler


def batch_fill(ds: LEMASDataset, sampler, epochs: Sequence[int]) -> Dict[str, float]:
    """Mean real and bound frames per batch over ``epochs``.

    Args:
        ds: The dataset the sampler batches.
        sampler: A ``BlockBatchSampler`` over ``ds``.
        epochs: Epochs to walk (each is the sampler's slice for that epoch).

    Returns:
        ``n_batches``, ``rows_per_batch``, ``real_frames_per_batch``,
        ``bound_frames_per_batch`` and ``fill`` (real over bound).

    Example:
        >>> batch_fill(ds, sampler, epochs=[0, 1])["fill"] <= 1
        True
    """
    bound_of = ds.frames_bound()
    n = rows = real = bound = 0
    for epoch in epochs:
        ds.set_epoch(epoch)
        for batch in sampler:
            n += 1
            rows += len(batch)
            real += sum(ds.drawn_frames(i) for i in batch)
            bound += int(bound_of[batch].sum())
    return dict(
        n_batches=n,
        rows_per_batch=rows / n,
        real_frames_per_batch=real / n,
        bound_frames_per_batch=bound / n,
        fill=real / bound,
    )


def measure(train_config: str, epochs: int) -> Dict[str, float]:
    """``batch_fill`` of a training config's train set and sampler.

    Args:
        train_config: Training YAML (``dataset.train``, ``batch_sampler``,
            ``trainer.accumulate_grad_batches``, ``num_device``).
        epochs: Epochs to walk, from 0.

    Returns:
        ``batch_fill``'s fields plus ``real_frames_per_update``.

    Example:
        >>> measure("conf/training_f5_small_en_stock.yaml", 3)["fill"] > 0.99
        True
    """
    cfg = OmegaConf.load(train_config)
    src = OmegaConf.to_container(cfg.dataset.train[0].data_src_args, resolve=True)
    ds = LEMASDataset(recipe_dir=".", load_speech=False, **src)
    kw = OmegaConf.to_container(cfg.batch_sampler, resolve=True)
    kw.pop("_target_")
    # one rank's view: fill is a property of a batch, not of the sharding
    sampler = BlockBatchSampler(ds, rank=0, world_size=1, **kw)
    out = batch_fill(ds, sampler, epochs=range(epochs))
    out["real_frames_per_update"] = (
        out["real_frames_per_batch"]
        * int(cfg.trainer.accumulate_grad_batches)
        * int(cfg.num_device)
    )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("train_configs", nargs="+")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = {}
    for path in args.train_configs:
        report[path] = measure(path, args.epochs)
        print(path, json.dumps(report[path], indent=1), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
