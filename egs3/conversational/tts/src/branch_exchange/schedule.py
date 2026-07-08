"""Depth schedule: which blocks get an exchange module, and which mode they run.

Modes:

- ``P``: plain parallel branches, no communication at this depth.
- ``P_TAC``: parallel branches with an exchange module after the block
  (each such depth gets its OWN exchange instance with independent weights,
  shared across branches within that depth).
- ``M``: reserved for a future merged mode; the enum member exists so config
  files stay stable, but constructing a schedule that uses it raises
  ``NotImplementedError``.
"""

from __future__ import annotations

import enum
from typing import Callable, Dict

from torch import nn


class Mode(enum.Enum):
    P = "P"
    P_TAC = "P+TAC"
    M = "M"


_MODE_ALIASES = {
    "P": Mode.P,
    "P+TAC": Mode.P_TAC,
    "P_TAC": Mode.P_TAC,
    "M": Mode.M,
}


def _parse_mode(value) -> Mode:
    if isinstance(value, Mode):
        return value
    key = str(value).strip()
    if key not in _MODE_ALIASES:
        raise ValueError(
            f"unknown exchange mode {value!r}; expected one of {sorted(_MODE_ALIASES)}"
        )
    return _MODE_ALIASES[key]


def _parse_range(key: str) -> tuple[int, int]:
    """Parse a 1-indexed inclusive range key like ``"1-6"`` or an index ``"7"``."""
    key = key.strip()
    if "-" in key:
        lo_s, hi_s = key.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
    else:
        lo = hi = int(key)
    if lo < 1 or hi < lo:
        raise ValueError(
            f"invalid block range {key!r}: must be 1-indexed with start <= end"
        )
    return lo, hi


class ExchangeSchedule:
    """Per-depth exchange plan built from a spec of 1-indexed inclusive ranges.

    ``spec`` example: ``{"1-6": "P", "7-22": "P+TAC"}``.
    Every block index ``1..depth`` must be covered exactly once.
    """

    def __init__(self, modes: list[Mode], exchanges: Dict[int, nn.Module]):
        self._modes = list(modes)
        self._exchanges = dict(exchanges)

    @classmethod
    def from_spec(
        cls,
        spec: Dict[str, str],
        depth: int,
        factory: Callable[[], nn.Module],
    ) -> "ExchangeSchedule":
        modes: list[Mode | None] = [None] * depth
        for key, mode_value in spec.items():
            mode = _parse_mode(mode_value)
            lo, hi = _parse_range(key)
            if hi > depth:
                raise ValueError(f"range {key!r} exceeds depth {depth}")
            for i in range(lo - 1, hi):
                if modes[i] is not None:
                    raise ValueError(
                        f"block {i + 1} covered more than once (by range {key!r})"
                    )
                modes[i] = mode
        missing = [i + 1 for i, m in enumerate(modes) if m is None]
        if missing:
            raise ValueError(f"blocks {missing} not covered by spec {spec!r}")
        if Mode.M in modes:
            raise NotImplementedError("Mode.M is reserved and not implemented yet")
        exchanges = {i: factory() for i, m in enumerate(modes) if m is Mode.P_TAC}
        return cls(modes, exchanges)  # type: ignore[arg-type]

    @property
    def depth(self) -> int:
        return len(self._modes)

    def mode(self, i: int) -> Mode:
        """Mode of the 0-indexed block ``i``."""
        return self._modes[i]

    def exchange_for(self, i: int) -> nn.Module | None:
        """Exchange module for 0-indexed block ``i`` (``None`` for ``P`` blocks)."""
        return self._exchanges.get(i)
