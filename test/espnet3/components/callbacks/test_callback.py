import shutil
import warnings
from pathlib import Path
from unittest import mock

import pytest
import torch
from hydra.utils import instantiate
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from espnet3.components.callbacks.default_callbacks import (
    AverageCheckpointsCallback,
    _metric_to_float,
    get_default_callbacks,
)

# ===============================================================
# Test Case Summary for AverageCheckpointsCallback
# ===============================================================
#
# Normal Cases
# | Test Name                                      | Description                       |
# |-----------------------------------------------|------------------------------------|
# | test_average_checkpoints_callback_on_validation_end  | Verifies that checkpoint    |
# |                        | averaging and saving works correctly with dummy weights.  |
# | test_get_default_callbacks_structure          | Checks structure and types of      |
# |                                   | callbacks returned by get_default_callbacks(). |
# | test_average_checkpoints_with_multiple_metrics| Confirms correct averaging for     |
# |                |  multiple ModelCheckpoint instances with different monitor names. |
# | test_output_filename_format                 | Ensures output filename is formatted |
# |                                         | using monitor name and checkpoint count. |
# | test_duplicate_learning_rate_monitor_from_config | Confirms that if                |
# | |LearningRateMonitor is defined both by default and in the config, duplicates occur|
# | | (no deduplication or warning yet).                                       |
#
# Edge/Error Cases
# | Test Name                                      | Description                       |
# |-----------------------------------------------|------------------------------------|
# | test_average_checkpoint_on_non_global_zero    | Ensures callback is skipped when   |
# |                       | trainer.is_global_zero is False (e.g., non-main DDP rank). |
# | test_average_checkpoint_with_inconsistent_keys| Raises KeyError if state_dict keys |
# |                                               | differ across checkpoints. |
# | test_average_checkpoint_with_int_and_float_mix| Confirms floats are averaged and   |
# |                          | ints are accumulated properly during checkpoint merging.|
# | test_average_checkpoint_skips_rotated_away_checkpoint | best_k_models names a     |
# | | checkpoint save_top_k already deleted: it is skipped, the survivors are     |
# | | averaged, and the denominator/filename use the surviving count.             |
# | test_average_checkpoint_all_tracked_checkpoints_missing | Every tracked path is   |
# | | gone: no crash and no file written (the Emilia resume crash loop).          |
# | test_average_checkpoint_unprefixed_state_dict | state_dict keys carry no "model." |
# | | prefix (what ESPnetLightningModule returns): averaged file must not be empty.|


@pytest.fixture
def dummy_state_dict():
    return {
        "state_dict": {
            "model.layer.weight": torch.tensor([1.0, 2.0]),
            "model.layer.bias": torch.tensor([0.5]),
            "model.bn.num_batches_tracked": torch.tensor(100, dtype=torch.int64),
        }
    }


def test_average_checkpoints_callback_on_validation_end(tmp_path, dummy_state_dict):
    """Test average checkpoints.

    Ensure AverageCheckpointsCallback correctly averages and saves model.
    """
    ckpt_paths = [tmp_path / f"ckpt_{i}.ckpt" for i in range(2)]

    with (
        mock.patch("torch.load", return_value=dummy_state_dict),
        mock.patch("torch.save") as mock_save,
    ):

        callback = AverageCheckpointsCallback(
            output_dir=str(tmp_path),
            best_ckpt_callbacks=[
                mock.Mock(
                    best_k_models={str(p): 0.0 for p in ckpt_paths},
                    monitor="valid/loss",
                )
            ],
        )
        trainer = mock.Mock()
        trainer.is_global_zero = True

        callback.on_validation_end(trainer, pl_module=mock.Mock())

        mock_save.assert_called_once()

        save_path = mock_save.call_args[0][1]
        assert Path(save_path).name.startswith("valid.loss.ave_2best.pth")

        averaged_state = mock_save.call_args[0][0]
        assert torch.allclose(averaged_state["layer.weight"], torch.tensor([1.0, 2.0]))
        assert torch.allclose(averaged_state["layer.bias"], torch.tensor([0.5]))
        assert "bn.num_batches_tracked" in averaged_state


def test_get_default_callbacks_structure():
    """Test Get default callbacks.

    Verify the structure and types of callbacks returned.
    """
    callbacks = get_default_callbacks(
        exp_dir="test_utils/espnet3_dummy/",
        best_model_criterion=[("valid/loss", 2, "min"), ("valid/wer", 2, "min")],
    )

    assert len(callbacks) == 7

    monitor_names = [None, "valid/loss", "valid/wer"]  # None for last checkpoint
    ckpt_callbacks = [cb for cb in callbacks if isinstance(cb, ModelCheckpoint)]
    for cb, expected_monitor in zip(ckpt_callbacks, monitor_names):
        assert cb.monitor == expected_monitor

    has_ave = any(isinstance(cb, AverageCheckpointsCallback) for cb in callbacks)
    assert has_ave


def test_average_checkpoints_with_multiple_metrics(tmp_path, dummy_state_dict):
    """Test averaging for multiple ModelCheckpoints with different monitor names."""
    ckpt_paths_1 = [tmp_path / f"ckpt_loss_{i}.ckpt" for i in range(2)]
    ckpt_paths_2 = [tmp_path / f"ckpt_acc_{i}.ckpt" for i in range(2)]

    with (
        mock.patch("torch.load", return_value=dummy_state_dict),
        mock.patch("torch.save") as mock_save,
    ):
        callback = AverageCheckpointsCallback(
            output_dir=str(tmp_path),
            best_ckpt_callbacks=[
                mock.Mock(
                    best_k_models={str(p): 0.0 for p in ckpt_paths_1},
                    monitor="valid/loss",
                ),
                mock.Mock(
                    best_k_models={str(p): 0.0 for p in ckpt_paths_2},
                    monitor="valid/acc",
                ),
            ],
        )
        trainer = mock.Mock(is_global_zero=True)
        callback.on_validation_end(trainer, pl_module=mock.Mock())

        assert mock_save.call_count == 2
        filenames = [Path(call.args[1]).name for call in mock_save.call_args_list]
        assert "valid.loss.ave_2best.pth" in filenames
        assert "valid.acc.ave_2best.pth" in filenames


def test_output_filename_format(tmp_path, dummy_state_dict):
    """Ensure output filename is formatted properly."""
    ckpt_paths = [tmp_path / f"ckpt_{i}.ckpt" for i in range(3)]

    with (
        mock.patch("torch.load", return_value=dummy_state_dict),
        mock.patch("torch.save") as mock_save,
    ):
        callback = AverageCheckpointsCallback(
            output_dir=str(tmp_path),
            best_ckpt_callbacks=[
                mock.Mock(
                    best_k_models={str(p): 0.0 for p in ckpt_paths},
                    monitor="some/metric",
                )
            ],
        )
        trainer = mock.Mock(is_global_zero=True)
        callback.on_validation_end(trainer, pl_module=mock.Mock())

        filename = Path(mock_save.call_args[0][1]).name
        assert filename == "some.metric.ave_3best.pth"


def test_average_checkpoint_on_non_global_zero(tmp_path, dummy_state_dict):
    """Ensure averaging does nothing when not global rank 0."""
    with (
        mock.patch("torch.load", return_value=dummy_state_dict),
        mock.patch("torch.save") as mock_save,
    ):
        callback = AverageCheckpointsCallback(
            output_dir=str(tmp_path),
            best_ckpt_callbacks=[
                mock.Mock(best_k_models={"dummy.ckpt": 0.0}, monitor="valid/loss")
            ],
        )
        trainer = mock.Mock(is_global_zero=False)
        callback.on_validation_end(trainer, pl_module=mock.Mock())

        mock_save.assert_not_called()


def test_average_checkpoint_with_inconsistent_keys(tmp_path):
    """Raise error when checkpoints have inconsistent keys."""
    ckpt_path1 = tmp_path / "ckpt_1.ckpt"
    ckpt_path2 = tmp_path / "ckpt_2.ckpt"

    inconsistent_state_dicts = [
        {"state_dict": {"model.layer.weight": torch.tensor([1.0])}},  # 1 key
        {
            "state_dict": {
                "model.layer.weight": torch.tensor([1.0]),
                "model.layer.bias": torch.tensor([0.5]),
            }
        },
    ]

    def load_side_effect(path, *args, **kwargs):
        return inconsistent_state_dicts.pop(0)

    with (
        mock.patch("torch.load", side_effect=load_side_effect),
        pytest.raises(KeyError),
    ):
        callback = AverageCheckpointsCallback(
            output_dir=str(tmp_path),
            best_ckpt_callbacks=[
                mock.Mock(
                    best_k_models={str(ckpt_path1): 0.0, str(ckpt_path2): 0.0},
                    monitor="valid/loss",
                )
            ],
        )
        trainer = mock.Mock(is_global_zero=True)
        callback.on_validation_end(trainer, pl_module=mock.Mock())


def test_average_checkpoint_with_int_and_float_mix(tmp_path):
    """Ensure float params are averaged, int params are accumulated."""
    ckpt_path1 = tmp_path / "ckpt_1.ckpt"
    ckpt_path2 = tmp_path / "ckpt_2.ckpt"

    mock_state_dicts = [
        {
            "state_dict": {
                "model.weight": torch.tensor([2.0, 4.0]),
                "model.counter": torch.tensor(10, dtype=torch.int64),
            }
        },
        {
            "state_dict": {
                "model.weight": torch.tensor([6.0, 2.0]),
                "model.counter": torch.tensor(30, dtype=torch.int64),
            }
        },
    ]

    def load_side_effect(path, *args, **kwargs):
        return mock_state_dicts.pop(0)

    with (
        mock.patch("torch.load", side_effect=load_side_effect),
        mock.patch("torch.save") as mock_save,
    ):
        callback = AverageCheckpointsCallback(
            output_dir=str(tmp_path),
            best_ckpt_callbacks=[
                mock.Mock(
                    best_k_models={str(ckpt_path1): 0.0, str(ckpt_path2): 0.0},
                    monitor="valid/loss",
                )
            ],
        )
        trainer = mock.Mock(is_global_zero=True)
        callback.on_validation_end(trainer, pl_module=mock.Mock())

        saved = mock_save.call_args[0][0]
        # Float averaged
        assert torch.allclose(saved["weight"], torch.tensor([4.0, 3.0]))
        # Int not averaged
        assert saved["counter"] == 40


def test_average_checkpoint_with_no_checkpoints(tmp_path):
    """Ensure averaging does nothing when there are no checkpoints."""
    with mock.patch("torch.save") as mock_save:
        callback = AverageCheckpointsCallback(
            output_dir=str(tmp_path),
            best_ckpt_callbacks=[mock.Mock(best_k_models={}, monitor="valid/loss")],
        )
        trainer = mock.Mock(is_global_zero=True)
        # This should not raise an exception
        callback.on_validation_end(trainer, pl_module=mock.Mock())

        mock_save.assert_not_called()


def test_duplicate_learning_rate_monitor_from_config():
    """Test duplicate LearningRateMonitor creation.

    Verify that when a LearningRateMonitor is provided both by default and in the user
    configuration, two separate instances are created. The current behavior does not
    emit a warning or perform any deduplication of these callbacks.
    """
    # First, get the default callbacks (contains exactly one LearningRateMonitor)
    callbacks = get_default_callbacks(
        exp_dir="test_utils/espnet3_dummy/",
        best_model_criterion=[("valid/loss", 2, "min")],
    )
    # Ensure only one LearningRateMonitor is included by default
    assert sum(isinstance(cb, LearningRateMonitor) for cb in callbacks) == 1

    # Simulate specifying LearningRateMonitor again via config (Hydra-style)
    cfg = OmegaConf.create(
        {"callbacks": [{"_target_": "lightning.pytorch.callbacks.LearningRateMonitor"}]}
    )
    # Append the instantiated callback to mimic trainer logic
    for cb_conf in cfg.callbacks:
        callbacks.append(instantiate(cb_conf))

    # Now we should have duplicates (2 LearningRateMonitor instances)
    # because no deduplication or warning is implemented yet
    assert sum(isinstance(cb, LearningRateMonitor) for cb in callbacks) == 2

    # AverageCheckpointsCallback should still be exactly one (unaffected by duplicates)
    assert sum(isinstance(cb, AverageCheckpointsCallback) for cb in callbacks) == 1


def test_metric_to_float_rejects_non_scalar_tensor():
    with pytest.raises(AssertionError, match="supports only scalar metric values"):
        _metric_to_float(torch.tensor([1.0, 2.0]))


def test_metric_to_float_rejects_unsupported_type():
    with pytest.raises(
        AssertionError, match="does not support metric values of type dict"
    ):
        _metric_to_float({"loss": 1.0})


def _write_lightning_ckpt(path, value):
    """Write a Lightning-layout checkpoint with UNPREFIXED keys.

    ESPnetLightningModule.state_dict() returns the inner model's state dict
    (`return self.model.state_dict(...)`), so real checkpoints from this
    framework carry no "model." prefix.
    """
    torch.save({"state_dict": {"layer.weight": torch.tensor([value, value])}}, path)


def _run_average(tmp_path, ckpt_paths):
    callback = AverageCheckpointsCallback(
        output_dir=str(tmp_path),
        best_ckpt_callbacks=[
            mock.Mock(
                best_k_models={str(p): 0.0 for p in ckpt_paths},
                monitor="valid/loss",
            )
        ],
    )
    trainer = mock.Mock()
    trainer.is_global_zero = True
    callback.on_validation_end(trainer, pl_module=mock.Mock())
    return sorted(tmp_path.glob("valid.loss.ave_*best.pth"))


def test_average_checkpoint_skips_rotated_away_checkpoint(tmp_path):
    """A checkpoint save_top_k already deleted must be skipped, not fatal.

    best_k_models is restored from the resuming checkpoint's callback state and
    routinely names files that save_top_k has since rotated off disk.
    """
    present = [tmp_path / "real_0.ckpt", tmp_path / "real_1.ckpt"]
    _write_lightning_ckpt(present[0], 1.0)
    _write_lightning_ckpt(present[1], 3.0)
    missing = tmp_path / "rotated_away.ckpt"  # deliberately never created

    written = _run_average(tmp_path, present + [missing])

    assert len(written) == 1
    # Denominator and filename both use the SURVIVING count, not the tracked
    # count -- dividing by 3 here would silently shrink the weights.
    assert written[0].name == "valid.loss.ave_2best.pth"
    averaged = torch.load(written[0], map_location="cpu", weights_only=False)
    assert torch.allclose(averaged["layer.weight"], torch.tensor([2.0, 2.0]))


def test_average_checkpoint_all_tracked_checkpoints_missing(tmp_path):
    """Every tracked checkpoint gone: no crash, nothing written.

    The `if not checkpoints` guard sits before the load loop and does not cover
    this -- the list is non-empty, every load just fails. Regression test for
    the Emilia resume crash loop (102 consecutive failed chain links).
    """
    gone = [tmp_path / "gone_0.ckpt", tmp_path / "gone_1.ckpt"]
    written = _run_average(tmp_path, gone)

    assert written == []


def test_average_checkpoint_unprefixed_state_dict(tmp_path):
    """Unprefixed keys must survive averaging instead of writing an empty file.

    Filtering on `k.startswith("model.")` alone empties the averaged dict for
    this framework's checkpoints; the shipped Emilia valid.loss.ave_1best.pth
    was 916 bytes and loaded to zero keys.
    """
    present = [tmp_path / "real_0.ckpt", tmp_path / "real_1.ckpt"]
    _write_lightning_ckpt(present[0], 2.0)
    _write_lightning_ckpt(present[1], 4.0)

    written = _run_average(tmp_path, present)

    assert len(written) == 1
    averaged = torch.load(written[0], map_location="cpu", weights_only=False)
    assert averaged, "averaged state dict must not be empty"
    assert torch.allclose(averaged["layer.weight"], torch.tensor([3.0, 3.0]))


# ===============================================================
# Resume-chain regression tests for the last-checkpoint callback
# ===============================================================
#
# | Test Name                                     | Description                     |
# |-----------------------------------------------|--------------------------------|
# | test_last_ckpt_relinks_across_repeated_resumes | last.ckpt tracks the newest    |
# | | step across two consecutive resumes; no last-v*.ckpt is ever created.      |
# | test_last_ckpt_relinks_after_exp_dir_is_moved  | THE 2026-08-25 regression:     |
# | | exp_dir moved, so the absolute dirpath in the checkpoint no longer matches  |
# | | and Lightning declines to restore last_model_path -- last.ckpt must still   |
# | | be relinked rather than superseded by last-v1.ckpt.                        |
# | test_resume_chain_advances_global_step        | global_step actually advances   |
# | | across a resume (the property four consecutive Emilia jobs violated).      |
# | test_resume_when_stored_last_model_path_no_longer_exists | Restored callback   |
# | | state names a deleted last-vN.ckpt (what the repair left behind): the       |
# | | removal of the missing previous checkpoint must be a no-op, not a crash.   |
#
# These drive a real Trainer instead of mocking, because the regression lived
# entirely in the interaction between Lightning's callback-state restore and its
# filename version counter -- every unit-level assertion about the callback in
# isolation passed while the chain made zero progress for four 8h jobs.
#
# NOTE: only test_last_ckpt_relinks_after_exp_dir_is_moved fails against the
# pre-fix code. The other three pass both before and after, by design: when
# dirpath is unchanged the version counter correctly stands down on its own, so
# they are invariants guarding normal resume, not reproductions of the bug.


class _Data(Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, i):
        return torch.full((4,), float(i) / 32.0)


class _TinyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        return self(batch).abs().mean()

    def validation_step(self, batch, batch_idx):
        loss = self(batch).abs().mean()
        self.log("valid/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _fit(exp_dir, max_steps, ckpt_path=None):
    callbacks = get_default_callbacks(
        exp_dir=str(exp_dir),
        log_interval=1000,
        best_model_criterion=[("valid/loss", 2, "min")],
        save_every_n_train_steps=2,
    )
    trainer = Trainer(
        default_root_dir=str(exp_dir),
        max_steps=max_steps,
        callbacks=callbacks,
        accelerator="cpu",
        devices=1,
        logger=CSVLogger(save_dir=str(exp_dir)),
        num_sanity_val_steps=0,
        val_check_interval=2,
        limit_val_batches=2,
        enable_model_summary=False,
    )
    loader = DataLoader(_Data(), batch_size=4)
    trainer.fit(_TinyModel(), loader, loader, ckpt_path=ckpt_path)
    return trainer


def _versioned(exp_dir):
    return sorted(p.name for p in Path(exp_dir).glob("last-v*.ckpt"))


def _link_target(exp_dir):
    link = Path(exp_dir) / "last.ckpt"
    assert link.is_symlink(), f"last.ckpt is not a symlink: {link}"
    return Path(str(link.resolve())).name


def test_last_ckpt_relinks_across_repeated_resumes(tmp_path):
    """last.ckpt must track the newest step across a resume chain.

    Without enable_version_counter=False this writes last-v1/last-v2 and leaves
    last.ckpt frozen, which is what pinned four consecutive Emilia jobs to the
    same starting batch.
    """
    exp = tmp_path / "exp"
    exp.mkdir()

    _fit(exp, max_steps=4)
    assert _link_target(exp) == "step4.ckpt"
    assert _versioned(exp) == []

    _fit(exp, max_steps=8, ckpt_path=str(exp / "last.ckpt"))
    assert _link_target(exp) == "step8.ckpt", "last.ckpt did not advance on resume #1"
    assert _versioned(exp) == [], f"version counter fired: {_versioned(exp)}"

    _fit(exp, max_steps=12, ckpt_path=str(exp / "last.ckpt"))
    assert _link_target(exp) == "step12.ckpt", "last.ckpt did not advance on resume #2"
    assert _versioned(exp) == [], f"version counter fired: {_versioned(exp)}"


def test_last_ckpt_relinks_after_exp_dir_is_moved(tmp_path):
    """The exact 2026-08-25 failure: exp_dir moved, so the absolute dirpath
    stored in the checkpoint no longer matches and Lightning declines to restore
    last_model_path (model_checkpoint.py:559). The last-checkpoint name must
    still be pinned to last.ckpt regardless.
    """
    old = tmp_path / "espnet_emilia" / "exp"
    old.mkdir(parents=True)
    _fit(old, max_steps=4)
    assert _link_target(old) == "step4.ckpt"

    new = tmp_path / "espnet_emilia_f5" / "exp"
    new.parent.mkdir(parents=True)
    shutil.move(str(old), str(new))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit(new, max_steps=8, ckpt_path=str(new / "last.ckpt"))

    # Precondition: the restore really was declined. If this stops holding, the
    # test is no longer exercising the regression.
    assert any("dirpath has changed" in str(w.message) for w in caught), (
        "expected Lightning's dirpath-changed warning; the test is not "
        "reproducing the original condition"
    )
    assert _link_target(new) == "step8.ckpt", "last.ckpt froze after exp_dir move"
    assert _versioned(new) == [], f"version counter fired: {_versioned(new)}"


def test_resume_chain_advances_global_step(tmp_path):
    """Resuming from last.ckpt must actually advance global_step.

    The original symptom was that it did not: every job restarted from the same
    checkpoint, so this asserts the property the train logs disproved.
    """
    exp = tmp_path / "exp"
    exp.mkdir()

    _fit(exp, max_steps=4)
    first = torch.load(exp / "last.ckpt", map_location="cpu", weights_only=False)
    assert first["global_step"] == 4

    _fit(exp, max_steps=8, ckpt_path=str(exp / "last.ckpt"))
    second = torch.load(exp / "last.ckpt", map_location="cpu", weights_only=False)
    assert second["global_step"] == 8, (
        f"resume did not advance: {first['global_step']} -> {second['global_step']}"
    )


def test_resume_when_stored_last_model_path_no_longer_exists(tmp_path):
    """Callback state restores cleanly but names a last-vN.ckpt that is gone.

    This is the exact state left behind by repairing the 2026-08-25 failure:
    step13000-v4.ckpt stores last_model_path=last-v5.ckpt, and the repair
    deleted the stale last-v* symlinks. On the next save Lightning computes
    `previous`=last-v5.ckpt and calls _remove_checkpoint on it, so this asserts
    the missing-file path is a no-op (TorchCheckpointIO guards with fs.exists)
    rather than an exception that would kill the first save of every job.
    """
    exp = tmp_path / "exp"
    exp.mkdir()
    _fit(exp, max_steps=4)

    # Rewrite the checkpoint's callback state to reference a checkpoint that
    # does not exist, mirroring the repaired directory on PSC.
    real = exp / "step4.ckpt"
    ckpt = torch.load(real, map_location="cpu", weights_only=False)
    ghost = str(exp / "last-v5.ckpt")
    patched = 0
    for key, state in ckpt["callbacks"].items():
        if isinstance(state, dict) and "last_model_path" in state and state["last_model_path"]:
            state["last_model_path"] = ghost
            patched += 1
    assert patched == 1, f"expected exactly one last_model_path to patch, got {patched}"
    torch.save(ckpt, real)
    assert not Path(ghost).exists()

    # Must not raise, and must still take ownership of the last.ckpt name.
    _fit(exp, max_steps=8, ckpt_path=str(exp / "last.ckpt"))
    assert _link_target(exp) == "step8.ckpt"
    assert _versioned(exp) == [], f"version counter fired: {_versioned(exp)}"
