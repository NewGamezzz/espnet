"""branch_exchange: run N weight-shared copies of a transformer stack with
communication (exchange) modules injected between blocks.

Branches are folded into the batch dimension as ``(B*N, T, d)``; exchanges
operate on the unfolded ``(B, N, T, d)`` view and are permutation-equivariant
in ``N`` with NO positional encoding on the branch axis, so branches are
interchangeable. Backbone-agnostic: only ``torch``, ``einops``, and the
standard library are imported.
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
