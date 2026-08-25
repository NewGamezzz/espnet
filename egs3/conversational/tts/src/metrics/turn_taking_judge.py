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

    ``encode_fn`` (batch float32 ``(B, T)`` of EQUAL-length windows -> ``(B, 5)``
    probabilities) is the test seam; the real one is built lazily from the
    Hugging Face checkpoint
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

        def encode(batch: np.ndarray) -> np.ndarray:
            # (B, T) windows of ONE length: no padding, so the batch is
            # sample-for-sample what the upstream batch-1 loop encodes
            # (the Whisper encoder has no attention mask, so padded batches
            # would NOT be faithful - see predict_many).
            speech = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
            lengths = speech.new_full(
                [speech.size(0)], dtype=torch.long, fill_value=speech.size(1)
            )
            with torch.no_grad():
                enc, enc_olens = model.encode(speech, lengths)
                feats = model.transform_mean(model.act_fn(enc))
                last = torch.stack(
                    [feats[k, enc_olens[k] - 1] for k in range(feats.size(0))]
                )
                logits = model.transform_linear(last)
                return torch.softmax(logits, dim=-1).float().cpu().numpy()

        self._encode_fn = encode

    @staticmethod
    def n_chunks(wav: np.ndarray) -> int:
        return max((len(wav) - START_SAMPLES) // HOP_SAMPLES, 0)

    def predict(self, wav16k: np.ndarray) -> np.ndarray:
        """``(n_chunks, 5)`` likelihoods, ``n_chunks = (T - 3200) // 640``."""
        return self.predict_many([wav16k])[0]

    def predict_many(
        self, wavs: Sequence[np.ndarray], batch_size: int = 32
    ) -> List[np.ndarray]:
        """Likelihoods for several windows, batching chunk ``i`` ACROSS
        windows: chunk ``i`` of every window covers the same sample span
        ``[max(0, end - 30 s), end)`` with ``end = 3200 + (i + 1) * 640``,
        so the stacked batch has one length and needs no padding - the
        inputs are identical to the sequential loop, only stacked. With a
        deterministic encoder the output is independent of ``batch_size``;
        on a GPU, batched kernels differ from the batch-1 path at the
        ~1e-3 level (measured 2.4e-3 max abs diff over 16k chunks, A100)
        and buy only ~1.5x (the encoder is compute-bound), so the metric
        defaults to ``window_batch=1`` (exactly the upstream loop) and the
        knob exists for callers who accept the deviation."""
        self._load()
        arrs = [np.asarray(w, dtype=np.float32) for w in wavs]
        counts = [self.n_chunks(w) for w in arrs]
        outs = [np.zeros((n, 5), dtype=np.float32) for n in counts]
        for i in range(max(counts, default=0)):
            end = (i + 1) * HOP_SAMPLES + START_SAMPLES
            start = max(0, end - CONTEXT_SAMPLES)
            members = [k for k, n in enumerate(counts) if i < n]
            for b in range(0, len(members), batch_size):
                idx = members[b : b + batch_size]
                batch = np.stack([arrs[k][start:end] for k in idx])
                probs = np.asarray(self._encode_fn(batch), dtype=np.float32)
                if probs.shape != (len(idx), 5):
                    raise RuntimeError(
                        f"encoder returned {probs.shape}, expected {(len(idx), 5)}"
                    )
                for row, k in zip(probs, idx):
                    outs[k][i] = row
        return outs

    @staticmethod
    def likelihood_line(window_id: str, probs: np.ndarray) -> str:
        """Upstream likelihood text: ``wid p,p,p,p,p p,p,p,p,p ...``."""
        cells = [",".join(f"{float(p):g}" for p in row) for row in probs]
        return " ".join([window_id, *cells])


# --------------------------------------------------------------------------- #
# upstream scoring library (imported by path, never vendored)
# --------------------------------------------------------------------------- #
_TEMPLATE_DIR = Path(__file__).resolve().parents[5] / "egs2" / "TEMPLATE" / "asr1"


def _upstream():
    """``pyscripts.utils.compute_turn_take_metrics`` from the ESPnet tree
    (a namespace package under ``egs2/TEMPLATE/asr1``)."""
    os.environ.setdefault("MPLBACKEND", "Agg")  # it imports pyplot at top
    if str(_TEMPLATE_DIR) not in sys.path:
        sys.path.insert(0, str(_TEMPLATE_DIR))
    from pyscripts.utils import compute_turn_take_metrics as lib

    return lib


ROLE_KEYS = (
    "judge_acc_pause",
    "judge_acc_turn_change",
    "judge_acc_bc",
    "judge_acc_no_bc",
    "judge_acc_interrupt",
    "judge_acc_no_interrupt",
    "judge_acc_willing_pause",
    "judge_acc_willing_turn",
    "judge_acc_interrupt_unsuccess",
    "judge_acc_interrupt_success",
)


# --------------------------------------------------------------------------- #
# the metric
# --------------------------------------------------------------------------- #
class TurnTakingJudgeMetric(BaseMetric):
    """Talking Turns judge agreement on each window's mixdown. See the module
    docstring for what is measured and how to read it."""

    def __init__(
        self,
        judge: Optional[TurnTakingJudge] = None,
        vad_backend: Optional[VADBackend] = None,
        bc_max_sec: float = BC_MAX_SEC,
        cache_likelihoods: bool = True,
        report_role_metrics: bool = False,
        window_batch: int = 1,
    ) -> None:
        self.judge = judge if judge is not None else TurnTakingJudge()
        self.vad_backend = (
            vad_backend if vad_backend is not None else SileroVADSegmenter()
        )
        self.bc_max_sec = bc_max_sec
        self.cache_likelihoods = cache_likelihoods
        self.report_role_metrics = report_role_metrics
        self.window_batch = window_batch

    # -- BaseMetric entrypoint ------------------------------------------- #
    def __call__(
        self, data: Dict[str, Path], test_name: str, output_dir: Path
    ) -> Dict[str, Optional[float]]:
        test_dir = Path(data["meta"]).parent
        out_dir = Path(output_dir) / test_name / "scoring" / "turn_taking_judge"
        (out_dir / "likelihoods").mkdir(parents=True, exist_ok=True)
        (out_dir / "labels").mkdir(parents=True, exist_ok=True)

        lik_lines: List[str] = []
        label_lines: List[str] = []
        per_window: List[Dict[str, Any]] = []
        skipped = 0
        metas = [
            json.loads((test_dir / row["meta"]).read_text("utf-8"))
            for _wid, row in self.iter_inputs(data, "meta")
        ]
        self._fill_likelihood_cache(
            [m for m in metas if len(m["channels"]) <= 2], test_dir, out_dir
        )
        with (out_dir / "windows.jsonl").open("w", encoding="utf-8") as fout:
            for meta in metas:
                if len(meta["channels"]) > 2:
                    skipped += 1
                    fout.write(
                        json.dumps(
                            {"window_id": meta["window_id"], "skipped": "channels>2"}
                        )
                        + "\n"
                    )
                    continue
                rec, lik, rows = self._score_window(meta, test_dir, out_dir)
                lik_lines.append(lik)
                label_lines.extend(rows)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                per_window.append(rec)

        summary, confusion = self._summarize(lik_lines, label_lines, per_window, skipped)
        (out_dir / "confusion.json").write_text(
            json.dumps(confusion, indent=2), encoding="utf-8"
        )
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    # -- likelihoods, batched across windows -------------------------------- #
    def _fill_likelihood_cache(self, metas, test_dir: Path, out_dir: Path) -> None:
        """Run the judge on every window whose likelihood file is missing,
        ``window_batch`` windows at a time (chunk-index batching inside
        :meth:`TurnTakingJudge.predict_many`), and write the cache files
        ``_score_window`` then reads."""
        todo = [
            m
            for m in metas
            if not (
                self.cache_likelihoods
                and (out_dir / "likelihoods" / f"{m['window_id']}.txt").exists()
            )
        ]
        for b in range(0, len(todo), self.window_batch):
            group = todo[b : b + self.window_batch]
            wavs = [
                load_wav(test_dir / m["mix_wav"], target_sr=JUDGE_SR)[0] for m in group
            ]
            for m, probs in zip(group, self.judge.predict_many(wavs)):
                line = self.judge.likelihood_line(m["window_id"], probs)
                (out_dir / "likelihoods" / f"{m['window_id']}.txt").write_text(
                    line + "\n", encoding="utf-8"
                )

    # -- per window ------------------------------------------------------- #
    def _score_window(self, meta: Dict[str, Any], test_dir: Path, out_dir: Path):
        wid = meta["window_id"]
        if "sw0" in wid:
            # compute_turn_likelihoods strips "sw0" from ids (Switchboard
            # convention); such a window would fail the join silently.
            raise ValueError(f"window id {wid!r} contains 'sw0'; rename before scoring")
        dur = float(meta["window_duration_sec"])
        lik_path = out_dir / "likelihoods" / f"{wid}.txt"
        # written by _fill_likelihood_cache (always, cache on or off)
        lik_line = lik_path.read_text("utf-8").strip()

        spans = []
        for ch in meta["channels"]:
            wav, sr = load_wav(test_dir / ch["gen_wav"], target_sr=JUDGE_SR)
            spans.append(self.vad_backend(wav, sr))
        rows = label_rows(wid, spans, dur, self.bc_max_sec)
        (out_dir / "labels" / f"{wid}.txt").write_text(
            "".join(r + "\n" for r in rows), encoding="utf-8"
        )

        labels = [r.split(",")[3] for r in rows]
        bc_spans = sum(
            1
            for ch in apply_backchannel_proxy(spans, dur, self.bc_max_sec)
            for _, kind in ch
            if kind == "BC"
        )
        return (
            {
                "window_id": wid,
                "duration_sec": dur,
                "expected_chunks": len(rows),
                "judge_chunks": len(lik_line.split()) - 1,
                "matched_chunks": min(len(lik_line.split()) - 1, len(rows)),
                "bc_proxy_spans": bc_spans,
                "label_counts": {
                    c: labels.count(c)
                    for c in ("C", "NA", "I", "BC", "T", "BC_1", "BC_2")
                },
                "confusion": self._confusion([lik_line], rows),
            },
            lik_line,
            rows,
        )

    # -- run level -------------------------------------------------------- #
    @staticmethod
    def _scorer(lib, lik_lines: Sequence[str], rows: Sequence[str]):
        lik = lib.compute_turn_likelihoods(
            list(lik_lines),
            lib.ModelParam.min_start_time.value,
            lib.ModelParam.chunk_length.value,
        )
        dec, turn = lib.compute_turn_decisions(list(rows))
        # human_human=True: the realized labels are the truth and the judge is
        # what gets scored. Upstream's parameter names are inverted relative
        # to what they hold; mimic score_turn_take.py's call order exactly
        # (decisions first, likelihoods second).
        return lik, lib.ScoreResult(dec, lik, turn, list(CLASSES), human_human=True)

    def _confusion(self, lik_lines: Sequence[str], rows: Sequence[str]):
        lib = _upstream()
        _lik, scorer = self._scorer(lib, lik_lines, rows)
        true = np.asarray(scorer.true_arr)
        pred = np.asarray(scorer.pred_arr_hard_label)
        return {
            t: {p: int(((true == t) & (pred == p)).sum()) for p in CLASSES}
            for t in CLASSES
        }

    def _summarize(self, lik_lines, label_lines, per_window, skipped):
        from sklearn.metrics import f1_score, roc_auc_score

        name = type(self).__name__
        lib = _upstream()
        summary: Dict[str, Optional[float]] = {
            "judge_skipped_windows": skipped,
            "judge_expected_chunks": sum(w["expected_chunks"] for w in per_window),
            "judge_bc_proxy_count": sum(w["bc_proxy_spans"] for w in per_window),
        }
        empty = {t: {p: 0 for p in CLASSES} for t in CLASSES}
        f1s: Dict[str, Optional[float]] = {c: None for c in CLASSES}
        aucs: Dict[str, Optional[float]] = {c: None for c in CLASSES}
        role: Dict[str, Optional[float]] = {k: None for k in ROLE_KEYS}
        confusion = empty
        matched: Optional[int] = None

        if lik_lines:
            lik, scorer = self._scorer(lib, lik_lines, label_lines)
            true = np.asarray(scorer.true_arr)
            hard = np.asarray(scorer.pred_arr_hard_label)
            soft = np.asarray(scorer.pred_arr_soft_label)
            matched = int(len(true))
            expected = summary["judge_expected_chunks"]
            if expected and matched / expected < 0.99:
                raise RuntimeError(
                    f"judge grid drift: matched {matched} of {expected} chunks"
                )
            for c in CLASSES:
                pos = true == c
                if not pos.any():
                    continue  # class absent from the realized labels
                f1s[c] = float(f1_score(pos, hard == c, average="macro"))
                aucs[c] = float(roc_auc_score(pos, soft[:, lib.LabelIndex[c].value]))
            confusion = {
                t: {p: int(((true == t) & (hard == p)).sum()) for p in CLASSES}
                for t in CLASSES
            }
            if self.report_role_metrics:
                role = self._role_metrics(lib.ScoreResult, lik, label_lines)

        summary["judge_matched_chunks"] = summary_value(
            matched, "judge_matched_chunks", metric_name=name
        )
        for c in CLASSES:
            summary[f"judge_f1_{c}"] = summary_value(
                f1s[c], f"judge_f1_{c}", metric_name=name
            )
            summary[f"judge_auc_{c}"] = summary_value(
                aucs[c], f"judge_auc_{c}", metric_name=name
            )
        f1_vals = [v for v in f1s.values() if v is not None]
        auc_vals = [v for v in aucs.values() if v is not None]
        summary["judge_f1_macro"] = summary_value(
            sum(f1_vals) / len(f1_vals) if f1_vals else None,
            "judge_f1_macro",
            metric_name=name,
        )
        summary["judge_auc_mean"] = summary_value(
            sum(auc_vals) / len(auc_vals) if auc_vals else None,
            "judge_auc_mean",
            metric_name=name,
        )
        for k in ROLE_KEYS:
            summary[k] = summary_value(role[k], k, metric_name=name)
        return summary, confusion

    # -- layer 2 ---------------------------------------------------------- #
    def _role_metrics(self, score_cls, lik_dict, rows) -> Dict[str, Optional[float]]:
        """The paper's role-conditioned accuracies, asked from BOTH sides
        (``only_AI`` on the original rows = "channel B holds the floor, what
        does A do"; then on the role-swapped rows), decision arrays pooled
        before the accuracy is taken."""
        lib = _upstream()
        dec_a, turn_a = lib.compute_turn_decisions(list(rows))
        dec_b, turn_b = lib.compute_turn_decisions(swap_roles(rows))
        s_a = score_cls(lik_dict, dec_a, turn_a, list(CLASSES), only_AI=True)
        s_b = score_cls(lik_dict, dec_b, turn_b, list(CLASSES), only_AI=True)
        for attr in ("pred_arr", "turn_arr", "true_arr_soft_label", "true_arr_hard_label"):
            setattr(
                s_a, attr, np.concatenate([getattr(s_a, attr), getattr(s_b, attr)])
            )

        def safe(fn):
            try:
                return fn()
            except (ValueError, ZeroDivisionError, IndexError):
                return (None, None)

        out: Dict[str, Optional[float]] = {}
        out["judge_acc_pause"], out["judge_acc_turn_change"] = safe(
            s_a.turn_change_metric
        )
        out["judge_acc_bc"], out["judge_acc_no_bc"] = safe(s_a.make_backchannel_metric)
        out["judge_acc_interrupt"], out["judge_acc_no_interrupt"] = safe(
            s_a.make_interruption_metric
        )
        out["judge_acc_willing_pause"], out["judge_acc_willing_turn"] = safe(
            s_a.turn_willingness_metric
        )
        out["judge_acc_interrupt_unsuccess"], out["judge_acc_interrupt_success"] = safe(
            s_a.handle_interruption_metric
        )
        # upstream accuracy_score on an empty selection yields NaN (warning,
        # not an exception): undefined stays None, never NaN in the summary.
        return {
            k: (None if v is None or np.isnan(float(v)) else float(v))
            for k, v in out.items()
        }
