"""End-to-end smoke: 30 optimizer steps on synthetic data + EMA/deepcopy."""

import copy

import lightning
import lightning.pytorch as pl
import pytest
import torch
from omegaconf import OmegaConf
from .conftest import (
    MEL,
    T,
    AJ_TEXTS,
    make_packed_mels,
    randomize_params,
    write_conversation_flac,
)
from .test_build_model import build_tiny  # noqa: F401  (fixture reuse)

from egs3.conversational.tts.dataset.dataset import ConversationDataset
from egs3.conversational.tts.dataset.preprocessing.sessions import (
    SessionRecord,
    write_session_manifest,
)
from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.dataset.preprocessor import ConversationalTextPreprocessor
from egs3.conversational.tts.src.branch_exchange import ExchangedBlock, get_context
from egs3.conversational.tts.src.build_model import exchange_param_groups
from egs3.conversational.tts.src.lit_module import (
    ConversationalLightningModule,
    PackedConversationCollator,
    PlannedWindowView,
)
from egs3.conversational.tts.src.sampler import ConversationBatchSampler
from espnet3.components.data.dataset import CombinedDataset


def _fake_samples(step: int, n: int = 2):
    """Fabricated ConversationDataset+preprocessor output (post-transform)."""
    gen = torch.Generator().manual_seed(1000 + step)
    samples = []
    for i, t_wav in enumerate((6144, 5120)):
        samples.append(
            {
                "window_id": f"w{step}_{i}",
                "num_channels": n,
                "speech": 0.1 * torch.randn(n, t_wav, generator=gen),
                "text": [torch.randint(0, 12, (30,), generator=gen) for _ in range(n)],
            }
        )
    return samples


def test_training_smoke(ext_vocab_file):
    torch.manual_seed(0)
    model = build_tiny(ext_vocab_file)
    # DiT zero-inits proj_out/AdaLN (no gradient reaches the blocks until
    # proj_out moves); the real model loads pretrained weights there, so
    # give the tiny stand-in non-zero weights too (gates stay zero).
    randomize_params(model, seed=42)
    model.train()
    optimizer = torch.optim.AdamW(
        exchange_param_groups(model, lr_exchange=1e-2, lr_backbone=1e-4)
    )
    collator = PackedConversationCollator()

    gates = [m.exchange.g for m in model.modules() if isinstance(m, ExchangedBlock)]
    for step in range(30):
        window_ids, batch = collator(_fake_samples(step))
        assert len(window_ids) == 2
        loss, stats, weight = model(**batch)
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        assert torch.isfinite(stats["loss_ch0"]) and torch.isfinite(stats["loss_ch1"])
        assert int(weight) == 2  # conversations, not rows
        optimizer.zero_grad()
        loss.backward()
        if step == 0:
            grads = [g.grad for g in gates]
            assert all(grad is not None for grad in grads)
            assert any(grad.abs() > 0 for grad in grads)
        optimizer.step()

    assert any(g.detach().abs() > 0 for g in gates), "no gate moved off zero"


def test_ema_deepcopy_safety(ext_vocab_file):
    """copy.deepcopy of the assembled model succeeds and the copy's forward
    matches the original's (same inputs, eval mode)."""
    model = build_tiny(ext_vocab_file).eval()
    clone = copy.deepcopy(model).eval()

    # The copy's blocks share one NEW context, still consistent internally.
    assert get_context(clone.cfm.transformer) is clone.cfm.ctx
    assert clone.cfm.ctx is not model.cfm.ctx

    mel, text, lens = make_packed_mels([2], seed=5)
    gen = torch.Generator().manual_seed(6)
    kwargs = dict(
        counts=[2],
        lens=lens,
        frac_lengths=torch.tensor([0.8]),
        time=torch.tensor([0.5]),
        x0=torch.randn(2, T, MEL, generator=gen),
    )

    torch.manual_seed(7)  # span start draw
    loss1, _, extras1 = model.cfm(mel, text, **kwargs)
    torch.manual_seed(7)
    loss2, _, extras2 = clone.cfm(mel, text, **kwargs)

    assert torch.equal(loss1, loss2)
    assert torch.equal(extras1["pred"], extras2["pred"])


def test_training_smoke_mixed_counts(ext_vocab_file):
    """One N=1 (LibriTTS-style) and one N=2 conversation in the same packed
    batch: forward, loss, and gradients through TAC at branch count 1."""
    torch.manual_seed(0)
    model = build_tiny(ext_vocab_file)
    randomize_params(model, seed=43)
    model.train()
    collator = PackedConversationCollator()
    gen = torch.Generator().manual_seed(7)
    samples = [
        {
            "window_id": "libritts_utt",
            "num_channels": 1,
            "speech": 0.1 * torch.randn(1, 6144, generator=gen),
            "text": [torch.randint(0, 12, (30,), generator=gen)],
        },
        {
            "window_id": "sssd_win",
            "num_channels": 2,
            "speech": 0.1 * torch.randn(2, 5120, generator=gen),
            "text": [torch.randint(0, 12, (30,), generator=gen) for _ in range(2)],
        },
    ]
    window_ids, batch = collator(samples)
    assert batch["counts"] == [1, 2]
    loss, stats, weight = model(**batch)
    assert torch.isfinite(loss)
    assert int(weight) == 2  # conversations, not rows
    loss.backward()
    gates = [m.exchange.g for m in model.modules() if isinstance(m, ExchangedBlock)]
    assert all(g.grad is not None for g in gates)


def test_training_smoke_cond_frames_reaches_cfm(ext_vocab_file):
    """Task 8's collator key is dead unless it survives the MultiBranchF5
    wrapper hop: ``ConversationalLightningModule`` calls ``model(**batch)``,
    which lands on ``MultiBranchF5.forward`` (src/model.py), NOT directly on
    ``MultiBranchCFM.forward``.  This exercises the real path end to end and
    confirms cond_frames reaches the CFM's forward unmodified."""
    torch.manual_seed(0)
    model = build_tiny(ext_vocab_file)
    randomize_params(model, seed=42)
    model.train()
    collator = PackedConversationCollator()

    samples = _fake_samples(0)
    samples[0]["cond_frames"] = 5  # well within the ~24-frame assembled mel
    window_ids, batch = collator(samples)
    assert batch["cond_frames"].tolist() == [5, -1]

    seen = {}
    orig_forward = model.cfm.forward

    def spy(*args, **kwargs):
        seen["cond_frames"] = kwargs.get("cond_frames")
        return orig_forward(*args, **kwargs)

    model.cfm.forward = spy
    loss, stats, weight = model(**batch)

    assert seen["cond_frames"] is not None
    assert seen["cond_frames"].tolist() == [5, -1]
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------
# Task 10 plan-level verification item (from Task 9's review): the sampler
# docstring claims Lightning tolerates per-epoch batch-count variance because
# it re-queries len(batch_sampler) at the start of every epoch. That is
# unverified there and, per lightning/pytorch/loops/fit_loop.py, doubtful:
# FitLoop.setup_data() only recomputes max_batches/combined_loader.limits
# when trainer.reload_dataloaders_every_n_epochs makes _should_reload_train_dl
# True, and this recipe's trainer config keeps it at 0 (a documented fix for
# a resume bug - see test_lit_module.py's
# test_training_config_has_no_per_epoch_reload_and_keeps_sanity_probe). This
# harness settles the question empirically: a real lightning.pytorch.Trainer
# runs 2 epochs through the REAL online path (ConversationalLightningModule.
# _packed_dataloader -> PlannedWindowView -> ConversationBatchSampler(
# online=True) -> a real ConversationDataset with real audio -> the tiny F5
# model), with window_seed chosen so epoch 1 packs to FEWER batches than
# epoch 0 - the direction that risks the fit ending early if a short epoch's
# StopIteration propagated past FitLoop instead of being caught locally by
# TrainingEpochLoop.run()'s own `except StopIteration: break`.
# --------------------------------------------------------------------------


def _online_smoke_turns(session_id, num_channels, duration, utt_len=1.6, gap=1.2):
    turns = []
    t, i = 0.5, 0
    while t + utt_len + 0.5 < duration:
        channel = i % num_channels
        turns.append(
            Turn(
                channel=channel,
                speaker=f"{session_id}_spk{channel}",
                text=AJ_TEXTS[i % len(AJ_TEXTS)],
                start=round(t, 3),
                end=round(t + utt_len, 3),
            )
        )
        t += utt_len + gap
        i += 1
    return turns


ONLINE_SMOKE_FS = 24000
ONLINE_SMOKE_BINS = 2 * round(ONLINE_SMOKE_FS * 10.0)
ONLINE_SMOKE_WINDOW_PARAMS = {"window_min": 4.0, "window_max": 10.0, "tail_min": 2.0}
# window_seed=3 with this session shape (4 x 45 s, N=2, the turn cadence
# above) packs epoch 0 to 20 batches and epoch 1 to 18 - found by scanning
# window_seed 0..19 for a seed where epoch 1 < epoch 0 (see task-10-report.md
# for the search script); the exact counts are re-derived below from the
# real dataset/sampler, not hardcoded, so this test does not silently go
# vacuous if planner internals change.
ONLINE_SMOKE_WINDOW_SEED = 3


@pytest.fixture
def online_train_combined(tmp_path, ext_vocab_file):
    sessions = []
    for k in range(4):
        session_id = f"sess{k}"
        write_conversation_flac(
            tmp_path / "original" / f"{session_id}_mixed.flac", 2, 45.0
        )
        sessions.append(
            SessionRecord(
                session_id=session_id,
                audio_relpath=f"original/{session_id}_mixed.flac",
                num_channels=2,
                sample_rate=48000,
                duration=45.0,
                turns=tuple(_online_smoke_turns(session_id, 2, 45.0)),
            )
        )
    manifest = tmp_path / "sessions_train.jsonl"
    write_session_manifest(manifest, sessions)
    dataset = ConversationDataset(
        split="train",
        manifest_path=manifest,
        dataset_root=tmp_path,
        fs=ONLINE_SMOKE_FS,
        permute_channels=False,
        window_params=ONLINE_SMOKE_WINDOW_PARAMS,
        window_seed=ONLINE_SMOKE_WINDOW_SEED,
    )
    preprocessor = ConversationalTextPreprocessor(token_list=ext_vocab_file)
    return CombinedDataset(
        datasets=[dataset],
        transforms=[(None, preprocessor)],
        use_espnet_preprocessor=True,
    )


def _sampler_len(dataset, epoch):
    return len(
        ConversationBatchSampler(
            dataset, batch_bins=ONLINE_SMOKE_BINS, online=True, seed=0, epoch=epoch
        )
    )


class _EpochRecorder(pl.Callback):
    """Per-epoch batch count and the set of window_ids actually reaching
    training_step, so the test can check both "how many batches" and "did
    the fresh online plan really reach the model" (a stale-iterator bug
    would keep replaying epoch 0's window_ids under a new epoch number)."""

    def __init__(self):
        self.batches_per_epoch: dict[int, int] = {}
        self.window_ids_per_epoch: dict[int, set[str]] = {}

    def on_train_epoch_start(self, trainer, pl_module):
        e = trainer.current_epoch
        self.batches_per_epoch[e] = 0
        self.window_ids_per_epoch[e] = set()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        e = trainer.current_epoch
        window_ids, _ = batch
        self.batches_per_epoch[e] += 1
        self.window_ids_per_epoch[e].update(window_ids)


def _online_smoke_module(combined, ext_vocab_file) -> ConversationalLightningModule:
    """Builds a REAL ConversationalLightningModule (not a hand-rolled
    LightningModule stand-in) so _packed_dataloader/train_dataloader/
    configure_optimizers/_initial_epoch all run as in production; only the
    ESPnetLightningModule.__init__ DataOrganizer/hydra-instantiate step is
    skipped (out of this task's scope) by calling lightning.LightningModule.
    __init__ directly and assigning the remaining attributes it would have
    set, exactly mirroring what ESPnetLightningModule.__init__ does."""
    model = build_tiny(ext_vocab_file)
    randomize_params(model, seed=42)
    module = ConversationalLightningModule.__new__(ConversationalLightningModule)
    lightning.LightningModule.__init__(module)
    module.config = OmegaConf.create(
        {
            "seed": 0,
            "num_device": 1,
            "optim": {
                "_target_": "torch.optim.AdamW",
                "lr_exchange": 1.0e-2,
                "lr_backbone": 1.0e-4,
                "betas": [0.9, 0.999],
                "weight_decay": 0.0,
            },
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.ConstantLR",
                "factor": 1.0,
                "total_iters": 1,
            },
            "scheduler_interval": "step",
            "dataloader": {
                "train": {
                    "batch_bins": ONLINE_SMOKE_BINS,
                    "min_batch_size": 1,
                    "shuffle": True,
                    "num_workers": 0,
                },
                "valid": {
                    "batch_bins": ONLINE_SMOKE_BINS,
                    "min_batch_size": 1,
                    "shuffle": False,
                    "num_workers": 0,
                },
            },
        }
    )
    module.model = model
    module.train_dataset = combined
    module.valid_dataset = combined
    module.nan_countdown = 0
    module.is_espnet_sampler = True
    module.collate_fn = PackedConversationCollator()
    return module


def test_online_sampler_survives_two_epochs_with_varying_batch_counts(
    online_train_combined, ext_vocab_file
):
    dataset = online_train_combined.datasets[0]
    expected = {e: _sampler_len(dataset, e) for e in (0, 1)}
    assert expected[1] < expected[0], (
        "fixture must exercise the risky fewer-batches-next-epoch direction "
        f"(got {expected}); re-search ONLINE_SMOKE_WINDOW_SEED if planner "
        "internals changed"
    )

    module = _online_smoke_module(online_train_combined, ext_vocab_file)
    recorder = _EpochRecorder()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        reload_dataloaders_every_n_epochs=0,  # matches conf/training_*.yaml
        use_distributed_sampler=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        accumulate_grad_batches=1,
        callbacks=[recorder],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    trainer.fit(module)

    # The fit must reach both epochs: a StopIteration from the shorter epoch
    # 1 escaping past TrainingEpochLoop.run()'s own handler would truncate
    # the whole fit at epoch 1 instead of just that epoch.
    assert trainer.current_epoch == 2
    assert set(recorder.batches_per_epoch) == {0, 1}
    assert recorder.batches_per_epoch[0] > 0
    assert recorder.batches_per_epoch[1] > 0
    # The fresh per-epoch plan actually reached the model: epoch 1 trained on
    # a different window set than epoch 0, not a replay under a new epoch
    # number.
    assert recorder.window_ids_per_epoch[0] != recorder.window_ids_per_epoch[1]


# window_seed=20 with 6 sessions (otherwise identical shape to the fixture
# above) packs epoch 0 to 26 batches and epoch 1 to 29 - found by scanning
# window_seed 0..59 for the largest epoch1-over-epoch0 gap; re-derived below
# from the real dataset/sampler, not hardcoded.
ONLINE_SMOKE_MORE_WINDOW_SEED = 20


@pytest.fixture
def online_train_combined_more_sessions(tmp_path, ext_vocab_file):
    sessions = []
    for k in range(6):
        session_id = f"sess{k}"
        write_conversation_flac(
            tmp_path / "original" / f"{session_id}_mixed.flac", 2, 45.0
        )
        sessions.append(
            SessionRecord(
                session_id=session_id,
                audio_relpath=f"original/{session_id}_mixed.flac",
                num_channels=2,
                sample_rate=48000,
                duration=45.0,
                turns=tuple(_online_smoke_turns(session_id, 2, 45.0)),
            )
        )
    manifest = tmp_path / "sessions_train.jsonl"
    write_session_manifest(manifest, sessions)
    dataset = ConversationDataset(
        split="train",
        manifest_path=manifest,
        dataset_root=tmp_path,
        fs=ONLINE_SMOKE_FS,
        permute_channels=False,
        window_params=ONLINE_SMOKE_WINDOW_PARAMS,
        window_seed=ONLINE_SMOKE_MORE_WINDOW_SEED,
    )
    preprocessor = ConversationalTextPreprocessor(token_list=ext_vocab_file)
    return CombinedDataset(
        datasets=[dataset],
        transforms=[(None, preprocessor)],
        use_espnet_preprocessor=True,
    )


def test_online_sampler_epoch1_more_batches_is_capped_at_epoch0_count(
    online_train_combined_more_sessions, ext_vocab_file
):
    """Discriminates the retired sampler docstring's claim ("Lightning
    re-queries len(batch_sampler) at the start of every epoch") from the
    corrected one ("epoch 0's count is a hard cap for the rest of that fit
    segment"). Direction 1 above (fewer batches in epoch 1) is observably
    identical under both hypotheses: a short epoch just ends early either
    way. This is the direction that actually distinguishes them: epoch 1's
    real plan packs to MORE batches than epoch 0, so if Lightning re-queried
    the length every epoch (the retired claim), the trainer would consume
    all of epoch 1's real batch count; if it caches max_batches from epoch 0
    (the corrected claim), epoch 1 is silently truncated down to epoch 0's
    count. The assertion below is the exact number that falls out only under
    the corrected hypothesis."""
    dataset = online_train_combined_more_sessions.datasets[0]
    planned = {e: _sampler_len(dataset, e) for e in (0, 1)}
    assert planned[1] > planned[0], (
        "fixture must exercise the more-batches-next-epoch direction "
        f"(got {planned}); re-search ONLINE_SMOKE_MORE_WINDOW_SEED if "
        "planner internals changed"
    )

    module = _online_smoke_module(online_train_combined_more_sessions, ext_vocab_file)
    recorder = _EpochRecorder()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        reload_dataloaders_every_n_epochs=0,  # matches conf/training_*.yaml
        use_distributed_sampler=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        accumulate_grad_batches=1,
        callbacks=[recorder],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    trainer.fit(module)

    assert trainer.current_epoch == 2
    assert recorder.batches_per_epoch[0] == planned[0]
    # The discriminating assertion: epoch 1 observed a batch count equal to
    # epoch 0's frozen cap, NOT its own (larger) planned count. Under the
    # retired "re-queries every epoch" claim this would equal planned[1]
    # instead, and the assertion below would fail.
    assert recorder.batches_per_epoch[1] == planned[0]
    assert recorder.batches_per_epoch[1] < planned[1]
