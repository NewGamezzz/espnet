import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler
from espnet2.schedulers.linear_warmup_decay import (
    LinearWarmupDecayLR,
    linear_warmup_decay,
)

BASE_LR = 7.5e-5
WARMUP_STEPS = 20000
TOTAL_STEPS = 600000
# Sweep past total_steps: the clamp there is a load-bearing property.
SWEEP_STEPS = 650000


def legacy_linear_warmup_decay(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    start_factor: float = 1e-8,
    end_factor: float = 1e-8,
):
    """The pre-class implementation, vendored so the reference outlives it.

    This is a verbatim copy of the ``SequentialLR``-of-two-``LinearLR``s
    factory that ``LinearWarmupDecayLR`` replaced. The F5-TTS schedule it
    encodes reproduces arXiv 2410.06885 Table 9, so the replacement must emit
    bit-identical learning rates, not merely close ones.
    """
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)
    decay_steps = max(total_steps - warmup_steps, 1)

    warmup = LinearLR(
        optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps
    )
    decay = LinearLR(
        optimizer, start_factor=1.0, end_factor=end_factor, total_iters=decay_steps
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]
    )


def _build_optimizer(lr=BASE_LR):
    torch.manual_seed(0)
    linear = torch.nn.Linear(2, 2)
    return torch.optim.SGD(linear.parameters(), lr=lr)


def _collect_lrs(scheduler, optimizer, num_steps):
    """Return the lr before the first step plus the lr after each step."""
    # One optimizer step up front keeps torch from warning about ordering.
    optimizer.step()
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(num_steps):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    return lrs


def test_bit_identical_to_legacy_factory():
    """The full-range sweep. Kept to a single test; the rest run coarse."""
    legacy_opt = _build_optimizer()
    legacy = legacy_linear_warmup_decay(legacy_opt, WARMUP_STEPS, TOTAL_STEPS)
    legacy_lrs = _collect_lrs(legacy, legacy_opt, SWEEP_STEPS)

    new_opt = _build_optimizer()
    new = LinearWarmupDecayLR(new_opt, WARMUP_STEPS, TOTAL_STEPS)
    new_lrs = _collect_lrs(new, new_opt, SWEEP_STEPS)

    # Exact float equality on purpose: not approx, not allclose.
    assert new_lrs == legacy_lrs

    # Anchor the curve so a matching-but-wrong pair of implementations
    # cannot pass this test.
    assert new_lrs[0] == BASE_LR * 1e-8
    assert new_lrs[WARMUP_STEPS] == BASE_LR
    assert new_lrs[TOTAL_STEPS] == new_lrs[SWEEP_STEPS]
    assert max(new_lrs) == BASE_LR


def test_is_abs_batch_step_scheduler():
    optimizer = _build_optimizer()
    scheduler = LinearWarmupDecayLR(optimizer, warmup_steps=50, total_steps=500)
    assert isinstance(scheduler, AbsBatchStepScheduler)


def test_repr():
    optimizer = _build_optimizer()
    scheduler = LinearWarmupDecayLR(optimizer, warmup_steps=50, total_steps=500)
    assert "LinearWarmupDecayLR" in repr(scheduler)
    assert "warmup_steps=50" in repr(scheduler)
    assert "total_steps=500" in repr(scheduler)


def test_state_dict_round_trip():
    opt = _build_optimizer()
    scheduler = LinearWarmupDecayLR(opt, warmup_steps=50, total_steps=500)
    _collect_lrs(scheduler, opt, 120)

    optimizer_state = opt.state_dict()
    scheduler_state = scheduler.state_dict()

    # The recurrence reads the live param_group lr, exactly as the legacy
    # SequentialLR did, so resuming restores the optimizer too. The scheduler
    # must be constructed before either load: construction performs an initial
    # step that would otherwise overwrite the restored lr.
    resumed_opt = _build_optimizer()
    resumed = LinearWarmupDecayLR(resumed_opt, warmup_steps=50, total_steps=500)
    resumed_opt.load_state_dict(optimizer_state)
    resumed.load_state_dict(scheduler_state)

    assert resumed.last_epoch == scheduler.last_epoch
    assert resumed_opt.param_groups[0]["lr"] == opt.param_groups[0]["lr"]

    for _ in range(200):
        scheduler.step()
        resumed.step()
        assert resumed_opt.param_groups[0]["lr"] == opt.param_groups[0]["lr"]


def test_compat_factory_returns_class_and_matches_curve():
    factory_opt = _build_optimizer()
    factory = linear_warmup_decay(factory_opt, warmup_steps=50, total_steps=500)
    assert isinstance(factory, LinearWarmupDecayLR)
    factory_lrs = _collect_lrs(factory, factory_opt, 600)

    legacy_opt = _build_optimizer()
    legacy = legacy_linear_warmup_decay(legacy_opt, 50, 500)
    legacy_lrs = _collect_lrs(legacy, legacy_opt, 600)

    assert factory_lrs == legacy_lrs


def test_matches_legacy_for_non_default_factors():
    kwargs = dict(warmup_steps=37, total_steps=411, start_factor=1e-3, end_factor=1e-2)

    legacy_opt = _build_optimizer(lr=1e-3)
    legacy = legacy_linear_warmup_decay(legacy_opt, **kwargs)
    legacy_lrs = _collect_lrs(legacy, legacy_opt, 500)

    new_opt = _build_optimizer(lr=1e-3)
    new = LinearWarmupDecayLR(new_opt, **kwargs)
    new_lrs = _collect_lrs(new, new_opt, 500)

    assert new_lrs == legacy_lrs
