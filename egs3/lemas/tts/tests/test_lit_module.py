from src.checkpoint_io import MmapCheckpointIO
from src.lit_module import LEMASLightningModule, _set_epoch


def test_set_epoch_reaches_inner_datasets():
    class Inner:
        epoch = None

        def set_epoch(self, e):
            self.epoch = e

    class Wrapped:
        def __init__(self, d):
            self.dataset = d

    class Combined:
        def __init__(self):
            self.datasets = [Inner(), Wrapped(Inner())]

    c = Combined()
    _set_epoch(c, 7)
    assert c.datasets[0].epoch == 7 and c.datasets[1].dataset.epoch == 7


def test_module_and_plugin_importable():
    assert LEMASLightningModule.__mro__[1].__name__ == "ESPnetLightningModule"
    assert MmapCheckpointIO.__name__ == "MmapCheckpointIO"
