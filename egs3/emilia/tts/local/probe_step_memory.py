"""Measure the per-GPU memory and time of one full training step.

``batch_bins`` has to be chosen per GPU type, and a short training run is a
poor instrument for it: batches are shuffled, so a few hundred steps can miss
the batch that later runs the job out of memory. This probe instead replays
the production ``numel`` sampler over the corpus's real frame counts, takes
the worst batch shapes it produces, and runs complete steps (forward,
backward, gradient clipping, AdamW, EMA shadow update) at exactly those
shapes. Audio is synthetic: memory depends on shapes, not on sample values.

Peak memory is a per-rank quantity (espnet3 hands each rank whole batches), so
one GPU is enough. DDP adds its gradient buckets and NCCL buffers on top;
leave headroom for them when reading the table.

Example:
    python local/probe_step_memory.py \
        --training_config conf/training_f5_tts_base.yaml \
        --batch_bins 960000 1920000 --precisions 32-true bf16-mixed \
        --out exp/probe/step_memory.json
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet3.utils.task_utils import get_espnet_model

logger = logging.getLogger(__name__)

PRECISIONS = ("32-true", "bf16-mixed")
_GIB = float(1 << 30)


def load_frame_lengths(shape_file: str | Path) -> np.ndarray:
    """Read the frame counts of a ``'<uid> <T>,<D>'`` shape file."""
    with open(shape_file, "r", encoding="utf-8") as fh:
        return np.fromiter(
            (
                int(line.split(" ", 1)[1].split(",", 1)[0])
                for line in fh
                if line.strip()
            ),
            dtype=np.int64,
        )


def replay_numel_batches(
    lengths: np.ndarray, n_mels: int, batch_bins: int, drop_last: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the batch shapes the ``numel`` sampler produces.

    Reproduces ``espnet2.samplers.NumElementsBatchSampler`` with
    ``min_batch_size=1`` and padding: utterances are taken shortest first and a
    batch closes on the first count whose padded size exceeds ``batch_bins``.
    Only the shapes are computed, so 37M utterances cost seconds and no
    per-utterance Python objects.

    Returns:
        ``(sizes, max_frames)``, one entry per batch in sampler order.
    """
    ordered = np.sort(np.asarray(lengths, dtype=np.int64)).tolist()
    sizes: list[int] = []
    max_frames: list[int] = []
    count = 0
    for length in ordered:
        count += 1
        if count * length * n_mels > batch_bins:
            sizes.append(count)
            max_frames.append(length)
            count = 0
    if count and (not drop_last or not sizes):
        sizes.append(count)
        max_frames.append(ordered[-1])
    return np.asarray(sizes, dtype=np.int64), np.asarray(max_frames, dtype=np.int64)


def peak_batches(
    sizes: np.ndarray, max_frames: np.ndarray
) -> dict[str, tuple[int, int]]:
    """Pick the ``(utterances, frames)`` shapes likely to set peak memory.

    A batch can be the worst by holding the most utterances, the most padded
    frames, or the longest utterance; which one binds depends on the model, so
    all three are measured.
    """
    padded = sizes * max_frames
    order = {
        "most_utterances": np.lexsort((padded, sizes)),
        "most_padded_frames": np.lexsort((max_frames, padded)),
        "longest_utterance": np.lexsort((sizes, max_frames)),
    }
    return {
        name: (int(sizes[idx[-1]]), int(max_frames[idx[-1]]))
        for name, idx in order.items()
    }


def _autocast(precision: str, device: torch.device):
    if precision == "bf16-mixed":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _synthetic_batch(
    n_utts: int, n_frames: int, hop_length: int, vocab_size: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Build a batch whose every row is ``n_frames`` mel frames long.

    Real batches are length-sorted, so their rows sit just below the longest
    one; padding all rows to it is the upper bound the sampler budgets for.
    """
    n_samples = (n_frames - 1) * hop_length
    n_tokens = max(1, n_frames // 8)
    return {
        "speech": 0.1 * torch.randn(n_utts, n_samples, device=device),
        "speech_lengths": torch.full((n_utts,), n_samples, device=device),
        "text": torch.randint(1, vocab_size, (n_utts, n_tokens), device=device),
        "text_lengths": torch.full((n_utts,), n_tokens, device=device),
    }


def _run_steps(
    model,
    params: list,
    optimizer,
    ema: list,
    ema_decay: float,
    clip: float,
    precision: str,
    device: torch.device,
    make_batch: Callable[[], dict],
    steps: int,
) -> tuple[float, float]:
    """Run ``steps`` full training steps; return ``(loss, seconds_per_step)``.

    Every tensor of a step is a local of this function, so nothing of a step
    that runs out of memory (its loss holds the whole autograd graph)
    outlives the call.
    """
    timings = []
    loss_value = float("nan")
    for _ in range(steps):
        batch = make_batch()
        start = time.perf_counter()
        with _autocast(precision, device):
            loss, _stats, _weight = model(**batch)
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(params, clip)
        optimizer.step()
        with torch.no_grad():
            for shadow, param in zip(ema, params):
                shadow.lerp_(param, 1.0 - ema_decay)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
        loss_value = float(loss.detach())
    # the first step pays cuDNN autotuning and allocator growth
    return loss_value, float(np.mean(timings[1:] or timings))


def run_probe(
    config: DictConfig,
    shapes: dict[str, tuple[int, int]],
    precisions: list[str],
    device: str = "cuda",
    steps: int = 3,
    on_model_built: Callable | None = None,
) -> list[dict]:
    """Run full training steps at each batch shape under each precision.

    Args:
        config: Resolved-on-access training config of the recipe.
        shapes: ``{name: (utterances, frames)}`` batch shapes to measure.
        precisions: Lightning precision names, a subset of ``PRECISIONS``.
        device: Torch device; memory columns are ``None`` off CUDA.
        steps: Steps per shape; the time reported excludes the first.
        on_model_built: Called once with the model before any step.

    Returns:
        One row per ``(precision, shape)``, in that order.
    """
    unknown = [p for p in precisions if p not in PRECISIONS]
    if unknown:
        raise ValueError(f"Unsupported precision {unknown}; expected {PRECISIONS}")

    dev = torch.device(device)
    on_cuda = dev.type == "cuda"
    model_config = OmegaConf.to_container(config.model, resolve=True)
    with open(model_config["token_list"], "r", encoding="utf-8") as fh:
        vocab_size = sum(1 for _ in fh)
    model = get_espnet_model(config.task, model_config).to(dev).train()
    if on_model_built is not None:
        on_model_built(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = instantiate(config.optimizer, params=params)
    ema_decay = 0.9999
    ema = [p.detach().clone() for p in params]
    clip = float(config.trainer.get("gradient_clip_val", 0.0) or 0.0)
    hop_length = int(config.hop_length)

    rows = []
    for precision in precisions:
        for name, (n_utts, n_frames) in shapes.items():
            row = {
                "precision": precision,
                "batch": name,
                "utterances": int(n_utts),
                "frames": int(n_frames),
                "padded_frames": int(n_utts * n_frames),
                "status": "ok",
                "loss": None,
                "seconds_per_step": None,
                "peak_allocated_gib": None,
                "allocated_before_gib": None,
                "peak_reserved_gib": None,
            }
            if on_cuda:
                torch.cuda.reset_peak_memory_stats(dev)
                row["allocated_before_gib"] = torch.cuda.memory_allocated(dev) / _GIB
            try:
                row["loss"], row["seconds_per_step"] = _run_steps(
                    model,
                    params,
                    optimizer,
                    ema,
                    ema_decay,
                    clip,
                    precision,
                    dev,
                    lambda: _synthetic_batch(
                        n_utts, n_frames, hop_length, vocab_size, dev
                    ),
                    steps,
                )
            except torch.OutOfMemoryError:
                row["status"] = "oom"
            if on_cuda:
                row["peak_allocated_gib"] = torch.cuda.max_memory_allocated(dev) / _GIB
                row["peak_reserved_gib"] = torch.cuda.max_memory_reserved(dev) / _GIB
            # A failed step leaves gradients behind, and its traceback kept the
            # graph alive until the except clause ended: release both, or every
            # later row is measured on a GPU that is already full.
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            if on_cuda:
                torch.cuda.empty_cache()
            logger.info("probe: %s", row)
            rows.append(row)
    return rows


def _format(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def main(argv=None) -> None:
    """Probe every requested ``batch_bins`` and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--batch_bins", type=int, nargs="+", required=True)
    parser.add_argument("--precisions", nargs="+", default=list(PRECISIONS))
    parser.add_argument("--shape_file", default=None)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = OmegaConf.load(args.training_config)
    batches_cfg = config.dataloader.train.iter_factory.batches
    shape_file = args.shape_file or batches_cfg.shape_files[0]
    n_mels = int(config.n_mel_channels)
    lengths = load_frame_lengths(shape_file)
    logger.info("probe: %d utterances from %s", len(lengths), shape_file)

    torch.set_float32_matmul_precision("high")
    report = {"shape_file": str(shape_file), "n_utterances": int(len(lengths))}
    report["batch_bins"] = {}
    shapes: dict[str, tuple[int, int]] = {}
    for bins in args.batch_bins:
        sizes, max_frames = replay_numel_batches(
            lengths, n_mels, bins, drop_last=bool(batches_cfg.get("drop_last", False))
        )
        peaks = peak_batches(sizes, max_frames)
        report["batch_bins"][str(bins)] = {"n_batches": int(len(sizes)), "peaks": peaks}
        for name, shape in peaks.items():
            shapes[f"{bins}/{name}"] = shape
    report["rows"] = run_probe(
        config, shapes, args.precisions, device=args.device, steps=args.steps
    )
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report["gpu"] = {"name": props.name, "total_gib": props.total_memory / _GIB}

    print(
        "precision   batch_bins/batch                 B     L  padded  status"
        "  before GiB  alloc GiB  reserved GiB  s/step"
    )
    for row in report["rows"]:
        print(
            f"{row['precision']:<11} {row['batch']:<30} {row['utterances']:>3} "
            f"{row['frames']:>5} {row['padded_frames']:>7}  {row['status']:<6} "
            f"{_format(row['allocated_before_gib'], '10.2f')} "
            f"{_format(row['peak_allocated_gib'], '10.2f')} "
            f"{_format(row['peak_reserved_gib'], '13.2f')} "
            f"{_format(row['seconds_per_step'], '7.3f')}"
        )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("probe: report written to %s", out)


if __name__ == "__main__":
    main()
