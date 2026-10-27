r"""Loss of one checkpoint on validation batches by length, against training.

Training logs a valid loss well above the train loss (S0c smoke: 8.7 against
5.4 at step 600; the D2 bf16 run: 4.3 against 1.7 at epoch 22). Validation
batches come length-sorted and unshuffled from ``BlockBatchSampler`` and
Lightning scores the first ``limit_val_batches`` of each rank, so the valid
loss is measured on the shortest rows. This script scores one checkpoint on
every validation batch and on training batches of epoch 0 with one model
state (eval mode, CFG coins off, a fixed seed per batch) and reports the
means by split and by length tertile, with the share of rows that carry a
prompt (dual-prompt layouts). A valid loss that matches the train loss at
matched length and prompt share makes the gap a property of the metric, not
of the model or the data.

Usage (recipe dir, one GPU):
    python local/loss_by_split.py --training_config conf/training_...yaml \\
        --ckpt exp/<tag>/last.ckpt --out exp/loss_by_split/<tag>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_dataloader import _EpochModule  # noqa: E402

from espnet3.systems.base.training import _instantiate_model  # noqa: E402
from espnet3.utils.config_utils import load_and_merge_config  # noqa: E402
from espnet3.utils.run_utils import (  # noqa: E402
    apply_training_experiment_context,
    resolve_loaded_configs,
)
from src.layout import HOP  # noqa: E402


def _load_cfg(path: str):
    cfg = load_and_merge_config(Path(path), config_name="training.yaml", resolve=False)
    apply_training_experiment_context(
        training_config=cfg,
        inference_config=None,
        metrics_config=None,
        publication_config=None,
        log=logging.getLogger(__name__),
    )
    resolve_loaded_configs(cfg, None)
    return cfg


def _load_weights(model, ckpt: str) -> int:
    """Load the live weights of a Lightning checkpoint; return its global step.

    espnet3 saves the inner model's ``state_dict`` without a prefix
    (``cfm.transformer...``); the EMA copy sits under
    ``ema_model_state_dict`` and is not what the logged loss measures.
    """
    state = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    sd = {k.removeprefix("model."): v for k, v in state["state_dict"].items()}
    model.load_state_dict(sd, strict=True)
    return int(state["global_step"])


def _batch_loss(model, data, device: str, seed: int) -> float:
    data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}
    torch.manual_seed(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _stats, _w = model(**data)
    return float(loss)


def _summarise(rows, head: int):
    """Mean, frame-weighted mean and tertiles by length of per-batch losses."""
    if not rows:
        return {}
    loss = np.array([r["loss"] for r in rows])
    frames = np.array([r["rows"] * r["mean_frames"] for r in rows])
    order = np.argsort([r["mean_frames"] for r in rows])
    tertiles = []
    for part in np.array_split(order, 3):
        tertiles.append(
            dict(
                batches=int(len(part)),
                mean_frames_per_row=float(
                    np.mean([rows[i]["mean_frames"] for i in part])
                ),
                mean_loss=float(loss[part].mean()),
            )
        )
    return dict(
        batches=len(rows),
        prompt_rows=float(np.mean([r["prompt_rows"] for r in rows])),
        mean_cond_frames=float(np.mean([r["mean_cond_frames"] for r in rows])),
        first_batches_mean_loss=float(loss[:head].mean()),
        mean_loss=float(loss.mean()),
        frame_weighted_loss=float((loss * frames).sum() / frames.sum()),
        mean_frames_per_row=float(np.mean([r["mean_frames"] for r in rows])),
        by_length_tertile=tertiles,
    )


def main() -> None:
    """Score the checkpoint on both loaders and write the summary."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--training_config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--valid_batches", type=int, default=10**9, help="all by default")
    p.add_argument("--train_batches", type=int, default=100)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cfg = _load_cfg(args.training_config)
    model = _instantiate_model(cfg)
    step = _load_weights(model, args.ckpt)
    print(f"loaded live weights of step {step}", flush=True)
    device = "cuda"
    model.eval().to(device)
    for m in model.modules():
        if hasattr(m, "audio_drop_prob"):
            m.audio_drop_prob = 0.0
            m.cond_drop_prob = 0.0
    lit = _EpochModule(model, cfg)
    lit.probe_epoch = 0
    head = int(cfg.trainer.get("limit_val_batches", 50))

    rows = []
    for split, loader, limit in (
        ("valid", lit.val_dataloader(), args.valid_batches),
        ("train", lit.train_dataloader(), args.train_batches),
    ):
        for b, (uids, data) in enumerate(loader):
            if b >= limit:
                break
            cf = torch.as_tensor(data["cond_frames"]).view(-1)
            rows.append(
                dict(
                    split=split,
                    batch=b,
                    rows=len(uids),
                    mean_frames=float(data["speech_lengths"].float().mean()) / HOP,
                    # dual-prompt rows: cond_frames > 0 when a prompt is present
                    # (for D1 that is the speaker prompt); stock rows carry -1
                    prompt_rows=float((cf > 0).float().mean()),
                    mean_cond_frames=float(cf.clamp(min=0).float().mean()),
                    loss=_batch_loss(model, data, device, 1000 + b),
                )
            )
            if b % 25 == 0:
                print(split, b, rows[-1], flush=True)

    summary = dict(
        ckpt=args.ckpt,
        limit_val_batches=head,
        global_step=step,
        valid=_summarise([r for r in rows if r["split"] == "valid"], head),
        train=_summarise([r for r in rows if r["split"] == "train"], head),
    )
    print(json.dumps(summary, indent=1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, batches=rows), indent=1))


if __name__ == "__main__":
    main()
