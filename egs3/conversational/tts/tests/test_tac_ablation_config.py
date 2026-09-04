"""Table 4 row c (severed TAC) config: the existing all-on run's config with
exactly two changes, so row d needs no new training.

Design note: "Design - Table 4 Communication-Block Ablation" (vault).
The row-d config is vendored beside it (training_mixed_allon_h100.yaml,
copied from the Delta checkout espnet_conv_allon, md5 verified both sides),
which is what makes this pairing auditable from git at all.
Read as plain YAML; no ${} resolution needed.
"""

from pathlib import Path

import yaml

RECIPE = Path(__file__).resolve().parents[1]


def load(name):
    return yaml.safe_load((RECIPE / "conf" / name).read_text(encoding="utf-8"))


ALLON = load("training_mixed_allon_h100.yaml")
SEVERED = load("training_mixed_allon_severed_h100.yaml")


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def test_severed_row_is_allon_plus_exactly_two_changes():
    fa, fb = flatten(ALLON), flatten(SEVERED)
    changed = {k for k in fa.keys() | fb.keys() if fa.get(k, "<absent>") != fb.get(k, "<absent>")}
    assert changed == {"exp_tag", "model.exchange.severed"}
    assert SEVERED["model"]["exchange"]["severed"] is True
    assert "severed" not in ALLON["model"]["exchange"]
    assert SEVERED["exp_tag"] == "train_mixed4_allon_severed_h100"


def test_neither_arm_caps_steps_so_the_lr_curves_are_identical():
    # No step cap in either arm: row c is stopped by not chaining another job,
    # and the matched step is chosen at EVALUATION time from the quarter-epoch
    # backups (every 1978 steps), not baked into training.
    assert "max_steps" not in SEVERED["trainer"]
    assert "max_steps" not in ALLON["trainer"]
    assert SEVERED["trainer"]["max_epochs"] == ALLON["trainer"]["max_epochs"]
    assert SEVERED["trainer"]["val_check_interval"] == ALLON["trainer"]["val_check_interval"]
    assert (
        SEVERED["trainer"]["check_val_every_n_epoch"]
        == ALLON["trainer"]["check_val_every_n_epoch"]
    )
    # Same LR at the same step in both arms: the schedule horizon is untouched.
    assert SEVERED["scheduler"]["total_steps"] == ALLON["scheduler"]["total_steps"] == 200000
    assert SEVERED["scheduler"]["warmup_steps"] == ALLON["scheduler"]["warmup_steps"]


def test_pairing_invariants_that_must_not_drift():
    for key in ("seed", "num_device", "num_nodes"):
        assert SEVERED[key] == ALLON[key], key
    assert SEVERED["model"]["pretrained_ckpt"] == ALLON["model"]["pretrained_ckpt"]
    assert SEVERED["dataloader"]["train"]["weights"] == ALLON["dataloader"]["train"]["weights"]
    assert SEVERED["dataloader"]["train"]["batch_bins"] == ALLON["dataloader"]["train"]["batch_bins"]
    assert (
        SEVERED["trainer"]["accumulate_grad_batches"]
        == ALLON["trainer"]["accumulate_grad_batches"]
    )
    assert SEVERED["optim"]["lr_exchange"] == ALLON["optim"]["lr_exchange"]
    assert SEVERED["optim"]["lr_backbone"] == ALLON["optim"]["lr_backbone"]


# --- row e: co-attention (BranchMHAExchange), the mechanism arm

COATTN = load("training_mixed_allon_coattn_h100.yaml")


def test_coattn_row_is_allon_with_the_exchange_swapped():
    """Row e differs from the all-on run only in the tag and the exchange
    module. `hidden` must be GONE: it is a TACExchange kwarg and the factory
    forwards leftover keys straight to BranchMHAExchange, which would raise.
    """
    fa, fb = flatten(ALLON), flatten(COATTN)
    changed = {k for k in fa.keys() | fb.keys() if fa.get(k, "<absent>") != fb.get(k, "<absent>")}
    assert changed == {
        "exp_tag",
        "model.exchange.type",
        "model.exchange.hidden",
        "model.exchange.n_heads",
    }
    ex = COATTN["model"]["exchange"]
    assert ex["type"] == "branch_mha"
    assert "hidden" not in ex
    assert ex["n_heads"] == 8
    # d_c omitted so it defaults to the model width, which is what makes the
    # block parameter-matched to TAC (4 d^2 either way).
    assert "d_c" not in ex
    assert ex["schedule"] == ALLON["model"]["exchange"]["schedule"]
    assert COATTN["exp_tag"] == "train_mixed4_allon_coattn_h100"


def test_coattn_shares_every_pairing_invariant_with_the_other_arms():
    for key in ("seed", "num_device", "num_nodes"):
        assert COATTN[key] == ALLON[key] == SEVERED[key], key
    assert COATTN["model"]["pretrained_ckpt"] == ALLON["model"]["pretrained_ckpt"]
    assert COATTN["dataloader"]["train"]["weights"] == ALLON["dataloader"]["train"]["weights"]
    assert COATTN["dataloader"]["train"]["batch_bins"] == ALLON["dataloader"]["train"]["batch_bins"]
    assert (
        COATTN["trainer"]["accumulate_grad_batches"]
        == ALLON["trainer"]["accumulate_grad_batches"]
    )
    assert COATTN["scheduler"]["total_steps"] == ALLON["scheduler"]["total_steps"] == 200000
    assert "max_steps" not in COATTN["trainer"]
    assert COATTN["optim"]["lr_exchange"] == ALLON["optim"]["lr_exchange"]
    assert COATTN["optim"]["lr_backbone"] == ALLON["optim"]["lr_backbone"]
