"""branch_exchange: run N weight-shared copies of a transformer stack with
communication (exchange) modules injected between blocks.

Branches are folded into the batch dimension as a packed layout
``(sum(N_i), T, d)`` (``ctx.branches(counts=[...])``): conversations with
different speaker counts are stacked with no padding rows, and exchanges
group rows by per-row conversation ids. Exchanges are
permutation-equivariant on the branch axis with NO positional encoding, so
branches are interchangeable. Backbone-agnostic: only ``torch``, ``einops``,
and the standard library are imported.
"""

from .exchange import BranchMHAExchange, IdentityExchange, TACExchange
from .inject import BranchContext, ExchangedBlock, inject_exchange, remove_exchange
from .registry import REGISTRY, BlockSpec
from .schedule import ExchangeSchedule, Mode

__all__ = [
    "TACExchange",
    "BranchMHAExchange",
    "IdentityExchange",
    "Mode",
    "ExchangeSchedule",
    "BranchContext",
    "ExchangedBlock",
    "inject_exchange",
    "remove_exchange",
    "BlockSpec",
    "REGISTRY",
]
