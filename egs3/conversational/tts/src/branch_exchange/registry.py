"""Backbone registry: where a model keeps its transformer blocks and how to
adapt each block's output to the exchange contract.

A ``BlockSpec`` describes one backbone family: ``target`` locates the blocks
(PEFT-style), ``unpack``/``repack`` adapt each block's output.
"""

from dataclasses import dataclass
from typing import Any, Callable, Sequence, Union


@dataclass(frozen=True)
class BlockSpec:
    """How to locate a backbone's transformer blocks and (un)pack their outputs.

    ``target`` takes one of three forms, all resolving to the same ordered
    list of blocks whose position IS the depth used by ``ExchangeSchedule``
    (depth placement stays index-based and reproducible):

    - dotted attribute path to the ``nn.ModuleList`` of blocks
      (``"transformer_blocks"``, ``"model.layers"``) - the fast path;
    - regex whose first capture group is each block's integer depth,
      ``re.fullmatch``-ed against ``model.named_modules()`` names
      (``r"(?:.*\\.)?layers\\.(\\d+)"`` is robust to wrapper nesting and to
      blocks living in an ``nn.Sequential``); a string is treated as a regex
      whenever it is not a plain dotted identifier path;
    - explicit ordered list of module names (depth = list position) - the
      escape hatch for models with no usable name pattern, e.g. multiple
      block stacks or blocks as plain attributes.

    ``unpack`` maps a block's raw output to the hidden-state tensor of shape
    ``(batch, seq, dim)``; ``repack`` maps ``(original block output, new
    hidden tensor)`` back to the block's output structure.
    """

    target: Union[str, Sequence[str]]
    unpack: Callable[[Any], Any] = lambda out: out
    repack: Callable[[Any, Any], Any] = lambda out, h: h


REGISTRY = {
    "f5_dit": BlockSpec(target="transformer_blocks"),
    "hf_decoder": BlockSpec(
        target=r"(?:.*\.)?layers\.(\d+)",
        unpack=lambda o: o[0],
        repack=lambda o, h: (h,) + tuple(o[1:]),
    ),
}
