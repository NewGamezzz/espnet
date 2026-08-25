"""``TurnTakingJudgeMetric``: the Talking Turns judge (Arora et al., ICLR
2025, arXiv 2503.01174; ESPnet PR #5948, recipe ``egs2/swbd/slu1``) applied
to the mixdown of each window.

Why: the dGSLM statistics in :mod:`.interaction` say how OFTEN events happen
and how LONG they last; they never ask whether an event happened at a
plausible moment given what was just said. The judge is a causal classifier
(frozen Whisper-medium encoder + linear head) that predicts, every 40 ms from
up to 30 s of mono context, which of five events comes next:

    C  continuation   NA silence   I interruption   BC backchannel   T turn change

Scoring a system's REALIZED events against those predictions measures
contextual timing. Decisions (vault note "Design - Talking Turns Judge
Turn-Taking Evaluation"):

* realized per-channel activity = the same Silero IPUs ``InteractionMetric``
  uses, rasterised onto the judge's grid with the upstream inclusion rule;
* the upstream Switchboard label state machine
  (``local/create_switchboard_data_2channels_mono.py``) is ported VERBATIM,
  state resetting per window;
* backchannels have no annotation on generated audio, so a proxy relabels a
  span as ``BC`` when it is <= 1.08 s (Switchboard annotated-backchannel
  duration p95), starts while the other channel holds the floor, and the
  floor does not pass to it afterwards - applied identically to every
  system, ground-truth anchor included;
* layer 1 = the paper's judge-validation protocol (``human_human=True`` in
  ``compute_turn_take_metrics``): per-class macro-F1 and ROC-AUC of judge
  vs realized, pooled over the run; layer 2 = the paper's role-conditioned
  accuracies computed under BOTH speaker-role assignments and pooled, since
  our dialogues have no AI/human role;
* likelihoods are cached per window in the upstream text format, so a new
  label policy never re-runs the encoder.

Only two-channel windows are defined (one-channel windows are padded with a
silent partner; 3+ channels are skipped and counted). Mono-ingested
baselines (``ingest.output: mixture``) have no per-channel activity and are
NOT scorable until a diarization-based labeler exists; scores from this
Silero labeler live in the channel-output table only.

Read the numbers COMPARATIVELY: the judge is Switchboard-telephony trained
and its human agreement is 57-81 % in the paper; the ``gt`` anchor on the
same set gives the ceiling for this audio and label pipeline.

Nothing here is imported by the training path.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from espnet3.components.metrics.base_metric import BaseMetric

from ._common import load_wav, summary_value
from .quality import SileroVADSegmenter, VADBackend

Span = Tuple[float, float]

MIN_START = 0.2  # ModelParam.min_start_time
CHUNK = 0.04  # ModelParam.chunk_length
JUDGE_SR = 16000
START_SAMPLES = 3200  # slu_inference.py run_chunk: start_chunk
HOP_SAMPLES = 640  # slu_inference.py run_chunk: sim_chunk_length
CONTEXT_SAMPLES = 480000  # slu_inference.py run_chunk: 30 s causal window
CLASSES = ("C", "NA", "IN", "BC", "T")  # LabelIndex order (I spelt IN)
HEAD_TOKENS = ["C", "NA", "I", "BC", "T"]  # checkpoint token list minus specials
BC_MAX_SEC = 1.08  # Switchboard annotated-backchannel duration p95 (n=60917)


def _round2(x: float) -> float:
    return float(f"{x:.2f}")


# --------------------------------------------------------------------------- #
# grid + rasteriser (pure)
# --------------------------------------------------------------------------- #
def chunk_ends(duration_sec: float) -> List[float]:
    """Chunk END times on the judge's grid, rounded exactly as upstream."""
    end_time = _round2(duration_sec)
    total = int((end_time - _round2(MIN_START)) / CHUNK)
    return [_round2(MIN_START + (i + 1) * CHUNK) for i in range(max(total, 0))]


def rasterise_channel(spans: Sequence[Span], ends: Sequence[float]) -> List[str]:
    """Per-chunk ``IPU``/``NA`` from speech spans with the upstream
    ``get_label`` rule: a chunk ending at ``e`` is active iff some span
    ``(s, t)`` satisfies ``s - CHUNK <= e < t`` (one chunk of look-ahead at
    the onset, the span's last partial chunk excluded)."""
    ordered = sorted((float(s), float(t)) for s, t in spans)
    out = []
    for e in ends:
        active = any(s - CHUNK <= e + 1e-9 and e < t - 1e-9 for s, t in ordered)
        out.append("IPU" if active else "NA")
    return out


# --------------------------------------------------------------------------- #
# backchannel proxy (pure)
# --------------------------------------------------------------------------- #
def _other_holds_floor(t: float, mine: Sequence[Span], other: Sequence[Span]) -> bool:
    """True if the OTHER channel holds the floor at time ``t``: it is active
    at ``t``, or its last IPU to end before ``t`` ended after mine did."""
    if any(s <= t < e for s, e in other):
        return True
    other_last = max((e for _, e in other if e <= t), default=None)
    mine_last = max((e for _, e in mine if e <= t), default=None)
    return other_last is not None and (mine_last is None or other_last > mine_last)


def apply_backchannel_proxy(
    channel_spans: Sequence[Sequence[Span]],
    duration_sec: float,
    bc_max_sec: float = BC_MAX_SEC,
) -> List[List[Tuple[Span, str]]]:
    """Relabel a span as ``BC`` when it is short (<= ``bc_max_sec``), starts
    while the other channel holds the floor, and the floor does not pass to
    this channel afterwards (the other channel is active again before this
    channel's next IPU). Defined for two channels; any other count is
    returned as all-``IPU``."""
    chans = [sorted((float(s), float(e)) for s, e in ch) for ch in channel_spans]
    if len(chans) != 2:
        return [[(sp, "IPU") for sp in ch] for ch in chans]
    out: List[List[Tuple[Span, str]]] = []
    for idx, mine in enumerate(chans):
        other = chans[1 - idx]
        labelled: List[Tuple[Span, str]] = []
        for k, (s, e) in enumerate(mine):
            kind = "IPU"
            if e - s <= bc_max_sec + 1e-9 and _other_holds_floor(s, mine, other):
                next_start = mine[k + 1][0] if k + 1 < len(mine) else duration_sec
                if any(oe > e and os_ < next_start for os_, oe in other):
                    kind = "BC"
            labelled.append(((s, e), kind))
        out.append(labelled)
    return out


# --------------------------------------------------------------------------- #
# upstream label state machine (verbatim port)
# --------------------------------------------------------------------------- #
def _mono_labels(
    label_a: Sequence[str], label_b: Sequence[str]
) -> Tuple[List[str], List[str]]:
    """Port of ``local/create_switchboard_data_2channels_mono.py`` (state
    resets per window). Returns ``(labels, floor_state_before_chunk)``."""
    prev = "NA"
    labels: List[str] = []
    turns: List[str] = []
    for la, lb in zip(label_a, label_b):
        turns.append(prev)
        if la == "NA" and lb == "NA":
            labels.append("NA")
        elif la == "IPU" and lb == "NA":
            if prev == "A":
                labels.append("C")
            else:
                labels.append("C" if prev == "AB" else "T")
                prev = "A"
        elif la == "NA" and lb == "IPU":
            if prev == "B":
                labels.append("C")
            else:
                labels.append("C" if prev == "BA" else "T")
                prev = "B"
        elif la == "IPU" and lb == "IPU":
            labels.append("I")
            if prev not in ("AB", "BA"):
                prev = "AB" if prev == "A" else "BA"
        elif la == "BC" and lb == "BC":
            labels.append("BC_2")
        elif la == "BC":
            labels.append("BC" if prev in ("B", "BA") else "BC_1")
        elif lb == "BC":
            labels.append("BC" if prev in ("A", "AB") else "BC_1")
        else:
            raise ValueError(f"unexpected channel labels {la!r}/{lb!r}")
    return labels, turns


def _rasterise_with_bc(
    labelled: Sequence[Tuple[Span, str]], ends: Sequence[float]
) -> List[str]:
    """Upstream ``get_chunk_dict`` precedence: BC wins over IPU per chunk."""
    ipu = rasterise_channel([sp for sp, k in labelled if k == "IPU"], ends)
    bc = rasterise_channel([sp for sp, k in labelled if k == "BC"], ends)
    return ["BC" if b == "IPU" else i for i, b in zip(ipu, bc)]


def label_rows(
    window_id: str,
    channel_spans: Sequence[Sequence[Span]],
    duration_sec: float,
    bc_max_sec: float = BC_MAX_SEC,
) -> List[str]:
    """Upstream ``*_Two_Channel_Label_Mono.csv`` rows for one window:
    ``wid,start,end,label,turn`` per chunk."""
    if len(channel_spans) == 1:
        channel_spans = [channel_spans[0], []]
    if len(channel_spans) != 2:
        raise ValueError(
            f"judge labels are defined for 2 channels, got {len(channel_spans)}"
        )
    ends = chunk_ends(duration_sec)
    labelled = apply_backchannel_proxy(channel_spans, duration_sec, bc_max_sec)
    la = _rasterise_with_bc(labelled[0], ends)
    lb = _rasterise_with_bc(labelled[1], ends)
    labels, turns = _mono_labels(la, lb)
    return [
        f"{window_id},{_round2(e - CHUNK)},{e},{lab},{turn}"
        for e, lab, turn in zip(ends, labels, turns)
    ]


_ROLE_SWAP = {"A": "B", "B": "A", "AB": "BA", "BA": "AB", "NA": "NA"}


def swap_roles(rows: Sequence[str]) -> List[str]:
    """Swap the speaker-turn column (A<->B, AB<->BA) so the role-conditioned
    upstream metrics can be asked from the other speaker's side."""
    out = []
    for r in rows:
        head, turn = r.rsplit(",", 1)
        out.append(f"{head},{_ROLE_SWAP[turn]}")
    return out


# --------------------------------------------------------------------------- #
# judge wrapper (run_chunk port)
# --------------------------------------------------------------------------- #
class TurnTakingJudge:
    """Ports the ``run_chunk`` loop of ``espnet2/bin/slu_inference.py``: one
    causal <= 30 s window per 40 ms hop, prediction read from the LAST
    encoder frame (``use_only_last_correct``), softmax over the 5 classes.

    ``encode_fn`` (window float32 ``(T,)`` -> 5 probabilities) is the test
    seam; the real one is built lazily from the Hugging Face checkpoint
    (``espnet/Turn_taking_prediction_SWBD``, CC-BY-4.0) or a local snapshot
    directory holding ``config.yaml`` + ``valid.loss.ave.pth``.
    """

    def __init__(
        self,
        checkpoint: str = "espnet/Turn_taking_prediction_SWBD",
        device: str = "cpu",
        encode_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self._encode_fn = encode_fn

    def _load(self) -> None:
        if self._encode_fn is not None:
            return
        import torch
        from espnet2.tasks.slu import SLUTask

        root = Path(self.checkpoint)
        if not root.exists():
            from huggingface_hub import snapshot_download

            root = Path(snapshot_download(self.checkpoint))
        cfgs = sorted(root.rglob("config.yaml"))
        pths = sorted(root.rglob("valid.loss.ave.pth"))
        if len(cfgs) != 1 or len(pths) != 1:
            raise FileNotFoundError(
                f"expected one config.yaml and one valid.loss.ave.pth under {root}"
            )
        model, _args = SLUTask.build_model_from_file(
            str(cfgs[0]), str(pths[0]), self.device
        )
        model.eval()
        head = [
            t for t in model.token_list if t not in ("<blank>", "<unk>", "<sos/eos>")
        ]
        if head != HEAD_TOKENS:
            raise RuntimeError(
                f"unexpected judge token order {head}; scorer assumes {HEAD_TOKENS}"
            )
        if not getattr(model, "use_only_last_correct", False):
            raise RuntimeError("judge checkpoint must have use_only_last_correct=True")

        def encode(window: np.ndarray) -> np.ndarray:
            speech = torch.as_tensor(
                window, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
            with torch.no_grad():
                enc, enc_olens = model.encode(speech, lengths)
                feats = model.transform_mean(model.act_fn(enc))
                last = feats[0, enc_olens[0] - 1]
                logits = model.transform_linear(last)
                return torch.softmax(logits, dim=-1).float().cpu().numpy()

        self._encode_fn = encode

    def predict(self, wav16k: np.ndarray) -> np.ndarray:
        """``(n_chunks, 5)`` likelihoods, ``n_chunks = (T - 3200) // 640``."""
        self._load()
        wav = np.asarray(wav16k, dtype=np.float32)
        n = (len(wav) - START_SAMPLES) // HOP_SAMPLES
        if n <= 0:
            return np.zeros((0, 5), dtype=np.float32)
        out = np.zeros((n, 5), dtype=np.float32)
        for i in range(n):
            end = (i + 1) * HOP_SAMPLES + START_SAMPLES
            start = max(0, end - CONTEXT_SAMPLES)
            out[i] = self._encode_fn(wav[start:end])
        return out

    @staticmethod
    def likelihood_line(window_id: str, probs: np.ndarray) -> str:
        """Upstream likelihood text: ``wid p,p,p,p,p p,p,p,p,p ...``."""
        cells = [",".join(f"{float(p):g}" for p in row) for row in probs]
        return " ".join([window_id, *cells])
