"""BlockSpec targeting tests: the three target forms (dotted path, regex
with a depth capture group, explicit name list) resolve to the same ordered
blocks, and mis-scoped targets fail loudly. Tiny fabricated models only; no
espnet imports here."""

import pytest
import torch
from torch import nn

from branch_exchange import (
    REGISTRY,
    BlockSpec,
    BranchContext,
    ExchangedBlock,
    ExchangeSchedule,
    IdentityExchange,
    TACExchange,
    inject_exchange,
    remove_exchange,
)

DIM = 8
COUNTS = (2, 1)

REGEX_SPEC = BlockSpec(target=r"(?:.*\.)?layers\.(\d+)")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(DIM, DIM)

    def forward(self, x):
        return x + self.self_attn(x)


class TupleBlock(Block):
    """HF-style block: returns a tuple with the hidden states first."""

    def forward(self, x):
        return (x + self.self_attn(x),)


class Stack(nn.Module):
    """Blocks in ``self.layers``, HF-inner-model style."""

    def __init__(self, depth, block=Block):
        super().__init__()
        self.layers = nn.ModuleList(block() for _ in range(depth))

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out
        return x


class Wrapped(nn.Module):
    """One wrapper level: blocks live at ``model.layers.<i>``."""

    def __init__(self, depth, block=Block):
        super().__init__()
        self.model = Stack(depth, block)

    def forward(self, x):
        return self.model(x)


class DoubleWrapped(nn.Module):
    """Two wrapper levels: blocks live at ``model.model.layers.<i>``."""

    def __init__(self, depth):
        super().__init__()
        self.model = Wrapped(depth)

    def forward(self, x):
        return self.model(x)


class AttrBlocks(nn.Module):
    """Blocks as plain attributes: no ModuleList, no usable name pattern."""

    def __init__(self):
        super().__init__()
        self.first = Block()
        self.second = Block()

    def forward(self, x):
        return self.second(self.first(x))


def inject(model, spec, depth, factory=IdentityExchange):
    ctx = BranchContext()
    schedule = ExchangeSchedule.from_spec({f"1-{depth}": "P+TAC"}, depth=depth, factory=factory)
    inject_exchange(model, spec, schedule, ctx)
    return ctx


def make_x(seed=0):
    return torch.randn(sum(COUNTS), 4, DIM, generator=torch.Generator().manual_seed(seed))


# ---- the three target forms resolve to the same blocks ----


def test_path_target():
    model = Wrapped(3)
    orig = list(model.model.layers)
    spec = BlockSpec(target="model.layers")
    inject(model, spec, depth=3)
    for i in range(3):
        assert isinstance(model.model.layers[i], ExchangedBlock)
        assert model.model.layers[i].base_block is orig[i]
    remove_exchange(model, spec)
    assert all(model.model.layers[i] is orig[i] for i in range(3))


@pytest.mark.parametrize("model_cls", [Wrapped, DoubleWrapped])
def test_regex_target_survives_wrapper_nesting(model_cls):
    """One regex registry entry serves the same model at any wrapper depth."""
    model = model_cls(3)
    stack = model.model.model if model_cls is DoubleWrapped else model.model
    orig = list(stack.layers)
    inject(model, REGEX_SPEC, depth=3)
    for i in range(3):
        assert isinstance(stack.layers[i], ExchangedBlock)
        assert stack.layers[i].base_block is orig[i]
    remove_exchange(model, REGEX_SPEC)
    assert all(stack.layers[i] is orig[i] for i in range(3))


def test_name_list_target_plain_attributes():
    model = AttrBlocks()
    orig = (model.first, model.second)
    x = make_x()
    with torch.no_grad():
        ref = model(x)
    spec = BlockSpec(target=["first", "second"])
    ctx = inject(model, spec, depth=2, factory=lambda: TACExchange(DIM))
    assert isinstance(model.first, ExchangedBlock)
    assert model.first.base_block is orig[0]
    with torch.no_grad(), ctx.branches(counts=COUNTS):
        out = model(x)
    assert torch.equal(out, ref)  # zero-init gates keep bit-equal identity
    remove_exchange(model, spec)
    assert model.first is orig[0] and model.second is orig[1]


def test_hf_decoder_registry_entry_end_to_end():
    """The shipped hf_decoder spec: regex target plus tuple unpack/repack."""
    model = Wrapped(2, block=TupleBlock)
    x = make_x()
    with torch.no_grad():
        ref = model(x)
    ctx = inject(model, REGISTRY["hf_decoder"], depth=2, factory=lambda: TACExchange(DIM))
    with torch.no_grad(), ctx.branches(counts=COUNTS):
        out = model(x)
    assert torch.equal(out, ref)


# ---- validation: mis-scoped targets fail loudly ----


def test_regex_requires_capture_group():
    with pytest.raises(ValueError, match="capture group"):
        inject(Wrapped(2), BlockSpec(target=r"(?:.*\.)?layers\.\d+"), depth=2)


def test_regex_zero_matches_lists_near_misses():
    with pytest.raises(ValueError, match="matched no module names") as exc:
        inject(Wrapped(2), BlockSpec(target=r"(?:.*\.)?layer\.(\d+)"), depth=2)
    assert "model.layers" in str(exc.value)


def test_regex_descendant_match_is_error():
    class NestedBlock(nn.Module):
        """A block whose own submodules also match a loose pattern."""

        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(DIM, DIM)])

    with pytest.raises(ValueError, match="inside another matched block"):
        inject(Wrapped(2, block=NestedBlock), REGEX_SPEC, depth=2)


def test_regex_duplicate_depth_is_error():
    class EncDec(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = Stack(2)
            self.dec = Stack(2)

    with pytest.raises(ValueError, match="explicit list"):
        inject(EncDec(), REGEX_SPEC, depth=4)


def test_regex_depth_gap_is_error():
    class Gappy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = Block()
            self.layer_2 = Block()

    with pytest.raises(ValueError, match="no gaps"):
        inject(Gappy(), BlockSpec(target=r"layer_(\d+)"), depth=2)


def test_name_list_missing_name_is_error():
    with pytest.raises(ValueError, match="not found") as exc:
        inject(AttrBlocks(), BlockSpec(target=["first", "seconde"]), depth=2)
    assert "'second'" in str(exc.value)


def test_name_list_nested_names_is_error():
    spec = BlockSpec(target=["model.layers.0", "model.layers.0.self_attn"])
    with pytest.raises(ValueError, match="inside another matched block"):
        inject(Wrapped(2), spec, depth=2)


def test_resolved_count_must_match_schedule_depth():
    with pytest.raises(ValueError, match="depth"):
        inject(Wrapped(3), REGEX_SPEC, depth=2)
