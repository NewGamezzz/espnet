"""Dataset side of the LEMAS recipe: keys, manifests, extraction, datasets.

``Dataset`` and ``DatasetBuilder`` are what espnet3 looks up when a training
config has no ``data_src`` (local ``recipe_dir/dataset`` module).
"""

from dataset.builder import LEMASBuilder as DatasetBuilder
from dataset.dataset import LEMASDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
