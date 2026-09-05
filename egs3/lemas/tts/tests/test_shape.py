import numpy as np
from espnet2.fileio.read_text import load_num_sequence_text

from src.shape import write_shape_file


class FakeDataset:
    def __init__(self, durations):
        self.durations = np.asarray(durations, dtype=np.float32)

    def __len__(self):
        return len(self.durations)

    def n_frames(self, hop_length, sample_rate):
        n = (self.durations.astype(np.float64) * sample_rate).astype(np.int64)
        return (1 + n // hop_length).astype(np.int32)


def test_shape_file_format(tmp_path):
    out = tmp_path / "feats_shape"
    n = write_shape_file(
        FakeDataset([1.0, 2.0]), out, hop_length=256, sample_rate=24000, n_mels=100
    )
    assert n == 2
    assert out.read_text("utf-8").splitlines() == ["0 94,100", "1 188,100"]
    loaded = load_num_sequence_text(str(out), loader_type="csv_int")
    assert loaded["1"] == [188, 100]
