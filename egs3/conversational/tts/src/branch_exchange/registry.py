"""Backbone registry: where a model keeps its transformer blocks and how to
adapt each block's output to the exchange contract.

A ``BlockSpec`` describes one backbone family:

- ``path``: dotted attribute path from the model to the ``nn.ModuleList`` of
  transformer blocks.
- ``unpack``: maps a block's raw output to the hidden-state tensor of shape
  ``(batch, seq, dim)``.
- ``repack``: maps ``(original block output, new hidden tensor)`` back to the
  block's output structure.
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class BlockSpec:
    """How to locate a backbone's block list and (un)pack block outputs."""

    path: str
    unpack: Callable[[Any], Any] = lambda out: out
    repack: Callable[[Any, Any], Any] = lambda out, h: h


REGISTRY = {
    "f5_dit": BlockSpec(path="transformer_blocks"),
    "hf_decoder": BlockSpec(
        path="model.layers",
        unpack=lambda o: o[0],
        repack=lambda o, h: (h,) + tuple(o[1:]),
    ),
}
