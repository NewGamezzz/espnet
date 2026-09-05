"""Analytic feats_shape synthesis, replacing collect_stats.

collect_stats produces feats_shape by decoding every utterance and running
mel extraction. LEMAS's manifests carry duration, and mel framing is analytic,
so 30 M decodes (three per row with the two prompts) are avoidable. The file
format is exactly what espnet3/components/data/collect_stats.py writes:
'<uid> <T>,<D>' with uid being str(index).

F5 uses no feature normalization, so no mean/variance statistics are needed
either, and nothing else is lost by dropping collect_stats.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def write_shape_file(
    dataset,
    out_path: str | Path,
    hop_length: int,
    sample_rate: int,
    n_mels: int,
) -> int:
    """Write a feats_shape file for ``dataset``.

    Args:
        dataset: Object with ``__len__`` and
            ``n_frames(hop_length, sample_rate) -> np.ndarray``.
        out_path: Destination file.
        hop_length: Mel hop in samples.
        sample_rate: Sample rate the frame rule is evaluated at.
        n_mels: Feature dimension written after the frame count.

    Returns:
        The number of rows written.

    Raises:
        RuntimeError: If ``dataset`` is empty.

    Example:
        .. code-block:: python

            >>> n = write_shape_file(dataset, "stats/train/feats_shape", 256, 24000, 100)

    Note:
        The row key is the dataset index as a string, which is what the numel
        batch sampler and ``CombinedDataset`` agree on.
    """
    out_path = Path(out_path)
    if len(dataset) == 0:
        raise RuntimeError(f"Refusing to write an empty shape file: {out_path}")

    frames = dataset.n_frames(hop_length=hop_length, sample_rate=sample_rate)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for idx, n_frame in enumerate(frames.tolist()):
            fh.write(f"{idx} {n_frame},{n_mels}\n")
    logger.info("write_shape_file: %d rows -> %s", len(frames), out_path)
    return int(len(frames))
