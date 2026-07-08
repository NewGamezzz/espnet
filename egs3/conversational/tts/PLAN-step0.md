---
date: 2026-07-08
tags:
  - speech-ai
  - tts
  - implementation
---

Part of [[00-ESPnet3 F5-TTS]].
Derived from [[Implementation Plan - Branch Exchange in ESPnet3]]; that note stays the source of truth for design decisions.
Copy everything below the horizontal rule into the espnet repo (suggested: `egs3/conversational/tts/PLAN-step0.md`) and point a Claude Code session at it.

---

# Step 0: `branch_exchange` package + test suite

## Goal

Build a self-contained, backbone-agnostic package that runs N weight-shared copies of a transformer stack with communication modules exchanged between blocks, plus a test suite proving its correctness properties.
This is step 0 of a larger plan; do NOT build the recipe, data pipeline, or trainer in this task.

## Context

- Repo branch: `espnet3/recipe/f5_tts`.
- The F5-TTS port lives in `espnet2/tts/f5/`; its backbone is `espnet2/tts/f5/backbones/dit.py`, class `DiT`, whose blocks live in `self.transformer_blocks` (an `nn.ModuleList` of `DiTBlock`), iterated in a plain loop in `DiT.forward`.
- Each `DiTBlock` is called as `block(x, t, mask=mask, rope=rope)` and returns a bare hidden-state tensor of shape `(batch, seq, dim)`.
- `DiT.forward` supports CFG inference by concatenating cond and uncond along the batch dim (`b n d -> 2b n d`) and optional per-block activation checkpointing with `use_reentrant=False`.
- Research context (background only): N branches, one per speaker, all sharing one set of pretrained weights; branches are folded into the batch dim as `(B*N, T, d)`; communication modules exchange information across the N branches at chosen depths.

## Hard constraints

1. Do not modify ANY existing file. `git diff` must be empty outside the new directory. In particular `espnet2/tts/f5/` stays untouched.
2. The package `egs3/conversational/tts/src/branch_exchange/` may import only `torch`, `einops`, and the Python standard library. Tests may additionally import `espnet2.tts.f5` and `pytest`.
3. No positional encoding, index embedding, or any branch-order-dependent computation anywhere on the branch axis. Branches must be interchangeable.
4. No pretrained checkpoint is needed or should be downloaded. All tests use small random-init models (e.g., `DiT(dim=64, depth=4, heads=2, dim_head=32, mel_dim=20, text_num_embeds=50)`).

## Package layout

```
egs3/conversational/tts/src/branch_exchange/
  __init__.py     # public API re-exports
  exchange.py     # TACExchange, BranchMHAExchange, IdentityExchange
  schedule.py     # Mode enum, ExchangeSchedule
  inject.py       # BranchContext, ExchangedBlock, inject_exchange, remove_exchange
  registry.py     # BlockSpec + REGISTRY
  tests/
    test_exchange.py
    test_inject.py
```

## Module specs

### Exchange contract (all classes in `exchange.py`)

Input `(B, N, T, d)` -> output `(B, N, T, d)`.
Permutation-equivariant in N, works for any N >= 1, all weights shared across branches.
Optional `pad_mask: (B, N) bool` marking padded ghost branches (True = padded); padded branches must not influence real ones.
Every exchange has a scalar gate parameter `g` initialized to 0 so the module is exactly the identity at init (output bit-equal to input).

### `TACExchange(dim, hidden=None)`

Transform-average-concatenate (Luo et al., ICASSP 2020) plus our zero-init gate.
Per time frame:

1. Transform: shared `Linear(dim, hidden) + PReLU` applied to every branch -> `z_i`; `hidden` defaults to `dim`.
2. Average: mean of `z_i` over the branch axis (excluding padded branches; divide by the real count), then shared `Linear(hidden, hidden) + PReLU` -> `z_bar`.
3. Concatenate: per branch, `Linear(2*hidden, dim) + PReLU` on `[z_i ; z_bar]` -> `u_i`.
4. Output: `h_i + g * u_i`.

### `BranchMHAExchange(dim, n_heads=8, d_c=None)`

1. Pre-norm: shared `LayerNorm(dim)`.
2. Fold time into batch: `(B, N, T, d) -> (B*T, N, d)` so the branch axis is the attention sequence axis.
3. Multi-head attention over the N branch tokens: shared projections `W_q, W_k, W_v: dim -> d_c` and `W_o: d_c -> dim`; `d_c` defaults to `dim`; NO positional encoding; self-attention includes self; `pad_mask` becomes the key padding mask.
4. Output: `h_i + g * attn_out_i` with scalar `g` init 0.

### `IdentityExchange()`

Returns its input unchanged; exists as the no-communication baseline and for injection-guard tests.

### `schedule.py`

- `Mode = Enum("P", "P_TAC", "M")` (accept `"P+TAC"` as a string alias when parsing).
- `ExchangeSchedule.from_spec(spec: dict[str, str], depth: int, factory: Callable[[], nn.Module])` where spec keys are 1-indexed inclusive ranges like `{"1-6": "P", "7-22": "P+TAC"}`; every block index 1..depth must be covered exactly once.
- `schedule.mode(i)` returns the mode for 0-indexed block `i`; `Mode.M` raises `NotImplementedError` at construction time for now (the enum member must exist so config files stay stable).
- Blocks in `P` mode get no exchange module at all; `P_TAC` blocks each get their OWN exchange instance from the factory (independent weights per depth, shared across branches within a depth).

### `inject.py`

```python
class BranchContext:
    # plain object, NOT an nn.Module; holds n_branch (int | None) and pad_mask
    # active property: n_branch is not None
    # context manager: with ctx.branches(n, pad_mask=None): ...
    #   (sets on enter, restores None on exit, exception-safe)

class ExchangedBlock(nn.Module):
    # holds base_block (submodule, name "base_block") and exchange (submodule, name "exchange")
    # holds ctx and spec as PLAIN attributes (use object.__setattr__) so they are
    # never registered as submodules and never appear in the state dict
    def forward(self, *args, **kwargs):
        out = self.base_block(*args, **kwargs)
        if not self.ctx.active:
            return out
        h = self.spec.unpack(out)                    # (B*N, T, d)
        h = h.unflatten(0, (-1, self.ctx.n_branch))  # (B, N, T, d)
        h = self.exchange(h, pad_mask=self.ctx.pad_mask)
        return self.spec.repack(out, h.flatten(0, 1))

def inject_exchange(model, spec, schedule, ctx) -> model
    # resolves spec.path on model to get the ModuleList, replaces blocks[i] in place
    # with ExchangedBlock ONLY for blocks whose schedule mode is P_TAC;
    # P-mode blocks stay untouched (original object, original state-dict keys)

def remove_exchange(model, spec) -> model
    # restores every ExchangedBlock back to its base_block
```

Note on CFG safety: `unflatten(0, (-1, n_branch))` is correct even when the caller has concatenated cond and uncond along batch, because each segment's length is a multiple of `n_branch`; groups never straddle the cond/uncond boundary.
Add a test for this anyway (see test 5c).

### `registry.py`

```python
@dataclass(frozen=True)
class BlockSpec:
    path: str                                  # attribute path to the ModuleList
    unpack: Callable = lambda out: out         # block output -> hidden tensor
    repack: Callable = lambda out, h: h        # (orig output, new hidden) -> block output

REGISTRY = {
    "f5_dit": BlockSpec(path="transformer_blocks"),
    "hf_decoder": BlockSpec(path="model.layers",
                            unpack=lambda o: o[0],
                            repack=lambda o, h: (h,) + tuple(o[1:])),
}
```

## Test suite (acceptance criteria)

Run with `pytest egs3/conversational/tts/src/branch_exchange/tests/ -q`; all must pass on CPU in under ~2 minutes.
Use a fixed seed and a small random-init `DiT` as described in Hard constraints #4.
"Independent passes" below means: run the unmodified original `DiT` separately on each branch's inputs and stack the outputs.

1. **Zero-init identity**: inject `P+TAC` at every block (test both `TACExchange` and `BranchMHAExchange`), activate `ctx` with N=3; output must be bit-equal (`torch.equal`) to independent passes, because `g=0` makes each exchange exactly the identity.
2. **Permutation equivariance**: set all gates to a nonzero value (e.g., `g=0.5`) and randomize exchange weights; for a random permutation `perm` of branches, `f(x[:, perm])` must equal `f(x)[:, perm]` (allclose, atol 1e-5). Test both exchange types, including with a `pad_mask` (permute it consistently).
3. **Count generalization**: the same injected model (same weights) must run at N=2, N=3, and N=4 without error and with correct output shapes; additionally, with nonzero gates, an N=3 batch where branch 3 is fully padded via `pad_mask` must match the corresponding N=2 run on the real branches (allclose) - padding must be equivalent to absence.
4. **Gradient flow**: with `ctx` active and a scalar loss on the output, every gate `g` receives a nonzero gradient at init; after setting `g=0.5`, all exchange weights receive gradients.
5. **Injection guard**:
   a. With `IdentityExchange` and active `ctx`, and separately with any exchange but inactive `ctx`, output is bit-equal to the original model's output.
   b. `remove_exchange(inject_exchange(model, ...))` restores a state dict with exactly the original keys and bit-equal values.
   c. CFG-style call: concatenate two `(B*N, T, d)` segments along batch (simulating cond/uncond) with `IdentityExchange` active; output equals original. Then with `TACExchange`, `g=0.5`: verify by construction that no exchange group mixes rows from different segments (e.g., make the second segment all zeros and check the first segment's output matches a run without the second segment).
   d. Repeat test 1 with `checkpoint_activations=True` on the DiT; must still pass, and test 4 must pass with checkpointing enabled.
6. **Import purity**: a test asserting that importing `branch_exchange` does not import any `espnet2`/`espnet3` module (`sys.modules` check).

## Definition of done

- All tests above implemented and passing.
- `git status` shows only the new `egs3/conversational/tts/src/branch_exchange/` directory (plus this plan file).
- Public API exported from `__init__.py`: `TACExchange`, `BranchMHAExchange`, `IdentityExchange`, `Mode`, `ExchangeSchedule`, `BranchContext`, `ExchangedBlock`, `inject_exchange`, `remove_exchange`, `BlockSpec`, `REGISTRY`.
- Docstrings state the `(B, N, T, d)` contract and the no-positional-encoding rule.
