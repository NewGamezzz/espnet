"""Analytic feats_shape synthesis."""

import numpy as np
import pytest

from egs3.emilia.tts.src.shape import write_shape_file
from espnet2.fileio.read_text import load_num_sequence_text


class FakeDataset:
    def __init__(self, durations):
        self.durations = np.asarray(durations, dtype=np.float32)

    def __len__(self):
        return len(self.durations)

    def n_frames(self, hop_length, sample_rate):
        n = (self.durations.astype(np.float64) * sample_rate).astype(np.int64)
        return (1 + n // hop_length).astype(np.int32)


def test_shape_file_format_matches_collect_stats(tmp_path):
    """collect_stats writes '<uid> <T>,<D>' with uid = str(index)."""
    out = tmp_path / "feats_shape"
    n = write_shape_file(
        FakeDataset([1.0, 2.0]), out, hop_length=256, sample_rate=24000, n_mels=100
    )
    assert n == 2
    assert out.read_text("utf-8").splitlines() == ["0 94,100", "1 188,100"]


def test_shape_file_is_loadable_by_the_sampler(tmp_path):
    """The espnet2 numel sampler must be able to parse it."""
    out = tmp_path / "feats_shape"
    write_shape_file(
        FakeDataset([1.0, 2.0, 3.0]), out, hop_length=256, sample_rate=24000, n_mels=100
    )
    loaded = load_num_sequence_text(str(out), loader_type="csv_int")
    assert loaded["0"] == [94, 100]
    assert len(loaded) == 3


def test_zero_length_dataset_raises(tmp_path):
    with pytest.raises(RuntimeError, match="empty"):
        write_shape_file(
            FakeDataset([]),
            tmp_path / "feats_shape",
            hop_length=256,
            sample_rate=24000,
            n_mels=100,
        )
