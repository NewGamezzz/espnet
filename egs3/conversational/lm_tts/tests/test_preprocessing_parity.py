"""Preprocessing parity + audio-offset helper (Task 6).

Two tiers, in one file, following this recipe's existing test layout
(``tests/test_pretrained_real.py`` is the asset-gated precedent):

1. Local (always runs, no assets): ``compute_audio_offsets`` /
   ``compute_audio_offsets_from_batch`` (``src/offsets.py``) unit-tested
   against hand-collated tensors that byte-match the ``tiny_parallel_llm``
   fixture's vocab layout (``tests/conftest.py``: specials [0,16), text
   [16,80), 4 audio streams [80,112) in blocks of 8).
2. Asset-gated (skips locally unless ``BAGPIPER_CKPT`` is set - same guard as
   ``test_pretrained_real.py``): feeds one real ``dataset.emit.emit_tac_records``
   record and one real ``dataset.emit.emit_mono_record`` record (built from a
   synthetic 16 kHz wav, not a hand-rolled dict) through the REAL
   ``SpeechLMJobTemplate(config).build_preprocessor().collate_fn``, exactly
   the harness ``scripts/gate_teacher_forced.py`` uses. This class only calls
   ``build_preprocessor()`` (tokenizer/codec config), never
   ``load_bagpiper()``'s ~16.9 GB bf16 model weights, so - unlike
   ``test_pretrained_real.py`` - it is safe to run on a box too small to hold
   the full model (confirmed locally: ``scripts/gate_teacher_forced.py
   --build-only`` completes in seconds on this 16 GB dev machine).

   KEY FINDING (see ``TestRealPreprocessorParity`` docstrings below): on the
   REAL collate_fn output, ``compute_audio_offsets``'s stream-0
   vocab-interval scan (the literally-specified token-scanning algorithm)
   finds NOTHING and raises ``ValueError`` on every row, because BagPiper's
   preprocessor defers audio tokenization to the model forward pass - the
   collated ``seqs`` tensor's audio region is the ``<|pad|>`` placeholder
   (id 0) until ``ParallelHFModel._embed`` mutates it in place using
   ``discrete_audio_indices`` (see ``espnet2/speechlm/model/speechlm/lm/
   parallel.py``). This is exactly the class of bug this task exists to
   catch (same class as the earlier F5 tokenizer-parity issue): a helper
   that is correct by its literal spec but would be silently useless
   against the real pipeline's actual behavior. ``src/offsets.py`` resolves
   this with two functions: ``compute_audio_offsets`` stays the literal
   token-scanning primitive (correct for tensors that already carry real
   codec ids), while ``compute_audio_offsets_from_batch`` - the function
   step-3 training actually calls - reads
   ``batch["discrete_audio_indices"][:, 1]`` directly, which IS available
   pre-forward and IS what this test class proves matches the real
   preprocessor's own structural offsets.

How to run locally (tier 1 always, tier 2 skips without assets):
    PYTHONPATH=<espnet_bagpiper worktree>:<this recipe dir> \\
        python -m pytest tests/test_preprocessing_parity.py -v

How to run tier 2 for real (Delta, or any box holding the checkpoint/SFT
assets under /work/nvme/bbjs/ttrachu/bagpiper-gate per the deferred-test
pattern in docs/bagpiper-findings.md):
    BAGPIPER_CKPT=/work/nvme/bbjs/ttrachu/bagpiper-gate/downloads/bagpiper/speechlm-qwen3-8b \\
    BAGPIPER_TRAIN_CONFIG=/work/nvme/bbjs/ttrachu/bagpiper-gate/egs3/conversational/lm_tts/conf/bagpiper_train_config.yaml \\
    PYTHONPATH=<espnet_bagpiper worktree>:<this recipe dir> \\
        python -m pytest tests/test_preprocessing_parity.py -v
(``BAGPIPER_TRAIN_CONFIG`` is optional; it defaults to the committed
``conf/bagpiper_train_config.yaml`` when unset, same as
``test_pretrained_real.py``.)
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from src.offsets import compute_audio_offsets, compute_audio_offsets_from_batch

RECIPE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CKPT = os.environ.get("BAGPIPER_CKPT")
CFG = os.environ.get("BAGPIPER_TRAIN_CONFIG") or os.path.join(
    RECIPE_DIR, "conf", "bagpiper_train_config.yaml"
)

_REAL_ASSET_SKIP = pytest.mark.skipif(
    not (CKPT and os.path.exists(CKPT)),
    reason=(
        "set BAGPIPER_CKPT to the BagPiper safetensors shard directory to run "
        "the real-preprocessor parity test (BAGPIPER_TRAIN_CONFIG defaults to "
        "conf/bagpiper_train_config.yaml)"
    ),
)


# ---------------------------------------------------------------------------
# Tier 1: local, no assets - compute_audio_offsets on hand-collated tensors.
# ---------------------------------------------------------------------------


class TestComputeAudioOffsets:
    """Hand-collated tensors byte-matching the tiny_parallel_llm vocab layout
    (tests/conftest.py: specials [0,16), text [16,80), 4 audio streams
    [80,112) in 8-token blocks). Rows use stream-0 columns already carrying
    real per-stream codec ids, exactly the state ``compute_audio_offsets``'s
    stream-0 scan is documented to require (see src/offsets.py)."""

    def test_single_row_offset(self, tiny_parallel_llm):
        vi = tiny_parallel_llm.vocab_intervals
        audio_start = vi["discrete_audio"][0][0]
        seq = torch.tensor(
            [
                [
                    [1, 0, 0, 0],
                    [20, 0, 0, 0],
                    [25, 0, 0, 0],
                    [audio_start + 2, 0, 0, 0],
                    [audio_start + 3, 0, 0, 0],
                ]
            ]
        )
        offsets = compute_audio_offsets(seq, vi)
        assert offsets.shape == (1,)
        assert offsets.dtype == torch.long
        assert offsets.tolist() == [3]

    def test_batch_with_different_offsets(self, tiny_parallel_llm):
        vi = tiny_parallel_llm.vocab_intervals
        audio_start = vi["discrete_audio"][0][0]
        row0 = [[1, 0, 0, 0], [audio_start, 0, 0, 0], [20, 0, 0, 0], [0, 0, 0, 0]]
        row1 = [[1, 0, 0, 0], [20, 0, 0, 0], [21, 0, 0, 0], [audio_start + 1, 0, 0, 0]]
        seq = torch.tensor([row0, row1])
        offsets = compute_audio_offsets(seq, vi)
        assert offsets.tolist() == [1, 3]

    def test_no_audio_row_raises(self, tiny_parallel_llm):
        vi = tiny_parallel_llm.vocab_intervals
        seq = torch.tensor([[[1, 0, 0, 0], [20, 0, 0, 0], [21, 0, 0, 0]]])
        with pytest.raises(ValueError, match=r"row"):
            compute_audio_offsets(seq, vi)

    def test_one_audio_row_among_several_raises_naming_the_row(self, tiny_parallel_llm):
        vi = tiny_parallel_llm.vocab_intervals
        audio_start = vi["discrete_audio"][0][0]
        row0 = [[1, 0, 0, 0], [audio_start, 0, 0, 0]]
        row1 = [[1, 0, 0, 0], [20, 0, 0, 0]]
        seq = torch.tensor([row0, row1])
        with pytest.raises(ValueError, match=r"\[1\]"):
            compute_audio_offsets(seq, vi)

    def test_2d_shape_supported(self, tiny_parallel_llm):
        vi = tiny_parallel_llm.vocab_intervals
        audio_start = vi["discrete_audio"][0][0]
        seq = torch.tensor([[1, 20, audio_start, audio_start + 1]])
        offsets = compute_audio_offsets(seq, vi)
        assert offsets.tolist() == [2]

    def test_matches_any_stream_interval_not_just_first(self, tiny_parallel_llm):
        """A stream-0 value landing in a LATER stream's interval still counts
        as audio - membership is checked against ANY interval in
        vocab_intervals['discrete_audio'], not just interval 0."""
        vi = tiny_parallel_llm.vocab_intervals
        last_start = vi["discrete_audio"][-1][0]
        seq = torch.tensor([[[1, 0, 0, 0], [20, 0, 0, 0], [last_start, 0, 0, 0]]])
        offsets = compute_audio_offsets(seq, vi)
        assert offsets.tolist() == [2]

    def test_invalid_ndim_raises(self, tiny_parallel_llm):
        vi = tiny_parallel_llm.vocab_intervals
        seq = torch.zeros(2, 3, 4, 5, dtype=torch.long)
        with pytest.raises(ValueError, match=r"\(B, T\)"):
            compute_audio_offsets(seq, vi)

    def test_device_preserved_cpu(self, tiny_parallel_llm):
        vi = tiny_parallel_llm.vocab_intervals
        audio_start = vi["discrete_audio"][0][0]
        seq = torch.tensor([[[1, 0, 0, 0], [audio_start, 0, 0, 0]]], device="cpu")
        offsets = compute_audio_offsets(seq, vi)
        assert offsets.device == seq.device


class TestComputeAudioOffsetsFromBatch:
    """Unlike compute_audio_offsets, this reads discrete_audio_indices (the
    preprocessor's structural start-position record), not token values in
    seqs - so seqs can be all-zero placeholder here, exactly like a real
    pre-forward collate_fn batch (see src/offsets.py's module docstring and
    TestRealPreprocessorParity below)."""

    def test_single_row(self):
        seqs = torch.zeros(1, 5, 4, dtype=torch.long)
        batch = {"seqs": seqs, "discrete_audio_indices": torch.tensor([[0, 2, 3]])}
        offsets = compute_audio_offsets_from_batch(batch)
        assert offsets.shape == (1,)
        assert offsets.dtype == torch.long
        assert offsets.tolist() == [2]

    def test_batch_with_different_offsets(self):
        seqs = torch.zeros(2, 6, 4, dtype=torch.long)
        batch = {
            "seqs": seqs,
            "discrete_audio_indices": torch.tensor([[0, 1, 2], [1, 3, 2]]),
        }
        offsets = compute_audio_offsets_from_batch(batch)
        assert offsets.tolist() == [1, 3]

    def test_row_with_no_indices_entry_raises_naming_the_row(self):
        seqs = torch.zeros(2, 6, 4, dtype=torch.long)
        batch = {"seqs": seqs, "discrete_audio_indices": torch.tensor([[0, 1, 2]])}
        with pytest.raises(ValueError, match=r"\[1\]"):
            compute_audio_offsets_from_batch(batch)

    def test_takes_min_start_for_duplicate_row_index(self):
        """General-infrastructure correctness: if a row somehow has more
        than one discrete_audio_indices entry, the offset is the earliest
        (minimum) start, not the last one seen. BagPiper's real records
        emit exactly one audio segment per row, so this is inert on real
        data today but the helper is meant to be general."""
        seqs = torch.zeros(1, 10, 4, dtype=torch.long)
        batch = {
            "seqs": seqs,
            "discrete_audio_indices": torch.tensor([[0, 5, 2], [0, 2, 2]]),
        }
        offsets = compute_audio_offsets_from_batch(batch)
        assert offsets.tolist() == [2]

    def test_missing_discrete_audio_indices_key_raises_keyerror(self):
        with pytest.raises(KeyError):
            compute_audio_offsets_from_batch({"seqs": torch.zeros(1, 3, 4)})

    def test_missing_seqs_key_raises_keyerror(self):
        with pytest.raises(KeyError):
            compute_audio_offsets_from_batch(
                {"discrete_audio_indices": torch.tensor([[0, 1, 1]])}
            )

    def test_device_preserved_cpu(self):
        seqs = torch.zeros(1, 5, 4, dtype=torch.long, device="cpu")
        batch = {
            "seqs": seqs,
            "discrete_audio_indices": torch.tensor([[0, 2, 3]], device="cpu"),
        }
        offsets = compute_audio_offsets_from_batch(batch)
        assert offsets.device == seqs.device


# ---------------------------------------------------------------------------
# Tier 2: asset-gated real preprocessor parity (skips without BAGPIPER_CKPT).
# ---------------------------------------------------------------------------


def _write_sine_wav(path, seconds=1.5, sr=16000, freq=220.0):
    import soundfile as sf

    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    wav = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), wav, sr)


def _build_emitted_records(tmp_path):
    """One real emit_tac_records record and one real emit_mono_record record
    (dataset.emit.py's actual emitters, not hand-rolled dicts), each pointing
    at a real synthetic 16 kHz wav under tmp_path."""
    from dataset.emit import emit_mono_record, emit_tac_records
    from dataset.preprocessing.attributes import SpeakerAttrs
    from dataset.preprocessing.audio import WindowAudio
    from dataset.preprocessing.sssd import Turn
    from dataset.preprocessing.windows import WindowRecord

    def attrs(gender):
        return SpeakerAttrs(
            median_f0=180.0,
            f0_iqr=20.0,
            words_per_sec=3.0,
            pitch_band="medium",
            variability_band="flat",
            rate_band="moderate",
            gender=gender,
            gender_source="metadata",
        )

    turns = [
        Turn(channel=0, speaker="spk0", text="hello there", start=0.5, end=2.0),
        Turn(channel=1, speaker="spk1", text="hi how are you", start=2.5, end=4.0),
        Turn(channel=0, speaker="spk0", text="doing well thanks", start=4.5, end=6.0),
    ]
    win = WindowRecord(
        window_id="parity_w00000",
        session_id="parity_sess",
        audio_relpath="original/parity_sess_mixed.wav",
        num_channels=2,
        sample_rate=48000,
        t0=0.0,
        t1=6.0,
        turns=tuple(turns),
    )
    ch0_wav = tmp_path / f"{win.window_id}_ch0.wav"
    ch1_wav = tmp_path / f"{win.window_id}_ch1.wav"
    mix_wav = tmp_path / f"{win.window_id}_mix.wav"
    _write_sine_wav(ch0_wav, freq=220.0)
    _write_sine_wav(ch1_wav, freq=330.0)
    _write_sine_wav(mix_wav, freq=180.0)
    wa = WindowAudio(
        window_id=win.window_id,
        channel_paths=(ch0_wav, ch1_wav),
        mix_path=mix_wav,
        channel_durations=(win.duration, win.duration),
        mix_duration=win.duration,
    )
    attrs_by_speaker = {"spk0": attrs("male"), "spk1": attrs("female")}

    tac_records = emit_tac_records(win, attrs_by_speaker, wa)
    assert len(tac_records) == 2
    tac_record = tac_records[0]  # channel 0
    mono_record = emit_mono_record(win, attrs_by_speaker, wa)
    return tac_record, mono_record


def _dialogue_from_record(record):
    """Map an SFT record's ``messages`` to the preprocessor's dialogue form,
    same modality mapping gate_teacher_forced.py uses. Unlike that script's
    build_dialogue, our records' audio paths are already real local tmp_path
    files (freshly written by this test), so no AUDIO_ROOT basename
    resolution is needed - just read them directly."""
    import soundfile as sf

    _IO_FOR_MODALITY = {"text": "text", "audio": "discrete_audio"}
    dialogue = []
    for role, modality, content in record["messages"]:
        io_name = _IO_FOR_MODALITY[modality]
        if modality == "audio":
            wav, sr = sf.read(content, dtype="float32", always_2d=True)
            content = (wav.T, sr)  # -> ([channels, samples], sr), matches _load_wav
        dialogue.append([role, io_name, content])
    return dialogue


@_REAL_ASSET_SKIP
class TestRealPreprocessorParity:
    def _build_batch(self, tmp_path):
        import yaml

        from espnet2.speechlm.model.speechlm.speechlm_job import SpeechLMJobTemplate

        with open(CFG) as f:
            config = yaml.safe_load(f)
        job = SpeechLMJobTemplate(config, is_train=True)
        preproc = job.build_preprocessor()

        tac_record, mono_record = _build_emitted_records(tmp_path)
        data_lst = [
            ((None, None, None), {"dialogue": _dialogue_from_record(tac_record)}),
            ((None, None, None), {"dialogue": _dialogue_from_record(mono_record)}),
        ]
        batch = preproc.collate_fn(data_lst)
        return job, batch

    def test_collate_succeeds_and_message_order_preserved(self, tmp_path):
        job, batch = self._build_batch(tmp_path)

        assert "seqs" in batch and "loss_masks" in batch
        seqs = batch["seqs"]
        assert seqs.dim() == 3
        B, T, n_stream = seqs.shape
        assert B == 2

        vocab = job.vocab
        sys_id = vocab.index("<|system|>")
        user_id = vocab.index("<|user|>")
        assistant_id = vocab.index("<|assistant|>")
        text_id = vocab.index("<|text|>")
        audio_id = vocab.index("<|audio|>")

        def first_idx(row, token_id, after=0):
            positions = (seqs[row, :, 0] == token_id).nonzero(as_tuple=True)[0]
            positions = positions[positions >= after]
            assert len(positions) > 0, f"token {token_id} not found in row {row} after {after}"
            return int(positions[0])

        for row in range(B):
            sys_pos = first_idx(row, sys_id)
            user_pos = first_idx(row, user_id, after=sys_pos + 1)
            # "assistant" role appears twice (text CoT, then audio); the
            # first occurrence must be the text turn.
            asst_text_pos = first_idx(row, assistant_id, after=user_pos + 1)
            asst_text_modality_pos = first_idx(row, text_id, after=asst_text_pos + 1)
            # the audio modality marker is the SECOND assistant turn.
            asst_audio_pos = first_idx(row, assistant_id, after=asst_text_modality_pos + 1)
            asst_audio_modality_pos = first_idx(row, audio_id, after=asst_audio_pos + 1)

            # system text < user text < assistant text < assistant audio, in
            # token-stream order.
            assert sys_pos < user_pos < asst_text_pos < asst_audio_pos < asst_audio_modality_pos

    def test_stream0_scan_finds_no_audio_on_raw_collate_output(self, tmp_path):
        """KEY FINDING: on the REAL (pre-forward) collate_fn batch,
        compute_audio_offsets's literally-specified stream-0 vocab-interval
        scan correctly raises ValueError on every row - the audio region is
        still the <|pad|> placeholder; real codec ids only appear once
        ParallelHFModel._embed mutates seqs in place during the forward pass
        (espnet2/speechlm/model/speechlm/lm/parallel.py). This is not a bug
        in compute_audio_offsets: it is doing exactly what its docstring
        says, and the result proves the placeholder-audio hypothesis (see
        module docstring above and src/offsets.py's module docstring)."""
        job, batch = self._build_batch(tmp_path)
        with pytest.raises(ValueError, match=r"row"):
            compute_audio_offsets(batch["seqs"], job.vocab_intervals)

        # Confirm directly: the audio region really is all-zero placeholder.
        dai = batch["discrete_audio_indices"]
        for bidx, start, length in dai.tolist():
            region = batch["seqs"][bidx, start : start + length]
            assert bool((region == 0).all()), (
                "expected the pre-forward audio region to be the <|pad|> "
                "placeholder; found a nonzero token, which would mean the "
                "placeholder hypothesis above is wrong"
            )

    def test_compute_audio_offsets_from_batch_matches_indices_ground_truth(self, tmp_path):
        """compute_audio_offsets_from_batch(batch) IS the real pre-forward
        offset helper (task-6-brief.md line 3: "audio_start ... equals the
        value our offset helper compute_audio_offsets(batch) returns"). It
        reads discrete_audio_indices' start column (col 1), which the
        preprocessor computes structurally from role/modality/content token
        COUNTS (speechlm_job.py `preprocessing`, step 3.4's `accum_length`),
        independent of the (not-yet-tokenized) audio content itself - see
        the previous test for why token-value scanning (compute_audio_offsets)
        cannot do this pre-forward. Also re-verifies the identity-region
        guarantee independently of compute_audio_offsets_from_batch's own
        internal logic: no stream, at any position before the offset,
        already carries a token id landing in a discrete_audio interval."""
        job, batch = self._build_batch(tmp_path)
        seqs = batch["seqs"]
        B, T, _n_stream = seqs.shape
        dai = batch["discrete_audio_indices"]
        offsets_by_row = {int(b): int(s) for b, s, _l in dai.tolist()}
        assert set(offsets_by_row) == set(range(B))

        offsets = compute_audio_offsets_from_batch(batch)
        assert offsets.shape == (B,)
        assert offsets.dtype == torch.long
        expected = torch.tensor([offsets_by_row[row] for row in range(B)], dtype=torch.long)
        assert torch.equal(offsets, expected)

        intervals = job.vocab_intervals["discrete_audio"]
        for row in range(B):
            offset = int(offsets[row])
            assert 0 < offset < T
            prefix = seqs[row, :offset]  # (offset, n_stream) - ALL streams
            for start, end in intervals:
                in_interval = (prefix >= start) & (prefix < end)
                assert not bool(in_interval.any()), (
                    f"row {row}: found a discrete_audio-interval token before "
                    f"the recorded offset {offset} - identity-region guarantee "
                    "violated"
                )
