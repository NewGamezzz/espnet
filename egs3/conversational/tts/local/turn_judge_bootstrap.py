"""Paired bootstrap over windows for the Talking Turns judge summaries.

Reads the cached likelihood/label files of several runs on the SAME test set,
builds per-window arrays once (upstream ScoreResult per window, both
human_human and the two role assignments), then resamples window ids and
recomputes every layer-1 / layer-2 number from concatenated arrays. Because
all runs of a set share window ids, the same resample is applied to every
run, so differences between runs are paired.

Also reports the decision COUNT behind every layer-2 accuracy (point
estimate) by wrapping sklearn's accuracy_score.

Usage::

    python local/turn_judge_bootstrap.py [--labels DIR] <n_reps> <seed> <label>=<test_dir> ...

``<test_dir>`` is ``<inference_dir>/<test_name>`` of an already-scored run
(its ``scoring/turn_taking_judge/{likelihoods,labels}`` are read). Writes
``judge_bootstrap_<labels>.json`` next to the inference dirs and prints a
markdown table (point [95% CI] (N) per run, paired differences).
``--labels labels_lex`` reads a tagged label policy instead of ``labels``
(output name gets the same suffix); likelihoods are always the shared cache.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from egs3.conversational.tts.src.metrics.turn_taking_judge import (
    CLASSES,
    ROLE_KEYS,
    _upstream,
    swap_roles,
)

lib = _upstream()

L1 = [f"judge_f1_{c}" for c in CLASSES] + ["judge_f1_macro"] + [f"judge_auc_{c}" for c in CLASSES] + ["judge_auc_mean"]
ROLE_ORDER = [  # accuracy_score call order inside the five upstream metrics
    "judge_acc_pause", "judge_acc_turn_change", "judge_acc_bc", "judge_acc_no_bc",
    "judge_acc_interrupt", "judge_acc_no_interrupt", "judge_acc_willing_pause",
    "judge_acc_willing_turn", "judge_acc_interrupt_unsuccess", "judge_acc_interrupt_success",
]


class _Arrays:
    """Bag of the four arrays the upstream metric methods read."""

    def __init__(self, pred_arr, turn_arr, soft, hard):
        self.pred_arr, self.turn_arr = pred_arr, turn_arr
        self.true_arr_soft_label, self.true_arr_hard_label = soft, hard


LABELS_DIR = "labels"


def per_window(test_dir: Path, labels_dir: str = None):
    d = test_dir / "scoring" / "turn_taking_judge"
    labels_dir = labels_dir or LABELS_DIR
    wins = {}
    for lp in sorted((d / "likelihoods").glob("*.txt")):
        wid = lp.stem
        lab = d / labels_dir / f"{wid}.txt"
        if not lab.exists():
            continue
        lik_line = lp.read_text().strip()
        rows = lab.read_text().splitlines()
        lik = lib.compute_turn_likelihoods([lik_line], lib.ModelParam.min_start_time.value, lib.ModelParam.chunk_length.value)
        dec, turn = lib.compute_turn_decisions(rows)
        hh = lib.ScoreResult(dec, lik, turn, list(CLASSES), human_human=True)
        dec_b, turn_b = lib.compute_turn_decisions(swap_roles(rows))
        ra = lib.ScoreResult(lik, dec, turn, list(CLASSES))
        rb = lib.ScoreResult(lik, dec_b, turn_b, list(CLASSES))
        wins[wid] = {
            "true": np.asarray(hh.true_arr), "hard": np.asarray(hh.pred_arr_hard_label), "soft": np.asarray(hh.pred_arr_soft_label),
            "role": [(np.asarray(r.pred_arr), np.asarray(r.turn_arr), np.asarray(r.true_arr_soft_label), np.asarray(r.true_arr_hard_label)) for r in (ra, rb)],
        }
    return wins


def summarize(wins, ids, counts=None):
    true = np.concatenate([wins[w]["true"] for w in ids])
    hard = np.concatenate([wins[w]["hard"] for w in ids])
    soft = np.concatenate([wins[w]["soft"] for w in ids])
    out = {}
    f1s, aucs = [], []
    for c in CLASSES:
        pos = true == c
        if pos.any() and not pos.all():
            out[f"judge_f1_{c}"] = float(f1_score(pos, hard == c, average="macro"))
            out[f"judge_auc_{c}"] = float(roc_auc_score(pos, soft[:, lib.LabelIndex[c].value]))
            f1s.append(out[f"judge_f1_{c}"]); aucs.append(out[f"judge_auc_{c}"])
        else:
            out[f"judge_f1_{c}"] = out[f"judge_auc_{c}"] = np.nan
    out["judge_f1_macro"] = float(np.mean(f1s)) if f1s else np.nan
    out["judge_auc_mean"] = float(np.mean(aucs)) if aucs else np.nan
    # layer 2: both role assignments, then pooled (same as the metric)
    parts = [[], [], [], []]
    for w in ids:
        for r in wins[w]["role"]:
            for k in range(4):
                parts[k].append(r[k])
    s = _Arrays(*[np.concatenate(p) if p else np.array([]) for p in parts])
    if s.true_arr_soft_label.ndim == 1:
        s.true_arr_soft_label = s.true_arr_soft_label.reshape(0, 5)
    import io, contextlib
    sizes = []
    if counts is not None:
        real = lib.accuracy_score

        def rec(y_true, y_pred, *a, **k):
            sizes.append(len(y_true)); return real(y_true, y_pred, *a, **k)
        lib.accuracy_score = rec
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            vals = []
            for fn in (lib.ScoreResult.turn_change_metric, lib.ScoreResult.make_backchannel_metric,
                       lib.ScoreResult.make_interruption_metric, lib.ScoreResult.turn_willingness_metric,
                       lib.ScoreResult.handle_interruption_metric):
                try:
                    vals.extend(fn(s))
                except (ValueError, ZeroDivisionError, IndexError):
                    vals.extend([None, None])
    finally:
        if counts is not None:
            lib.accuracy_score = real
    for k, v in zip(ROLE_ORDER, vals):
        out[k] = np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
    if counts is not None:
        # sizes align with ROLE_ORDER except that handle_interruption skips the
        # second call when no successful interruption exists
        for k, n in zip(ROLE_ORDER, sizes):
            counts[k] = n
    return out


_G = {}  # inherited by forked workers (no per-task pickling of the arrays)


def one_rep(seed):
    rng = np.random.default_rng(seed)
    ids_all, runs = _G["ids"], _G["runs"]
    ids = list(rng.choice(ids_all, size=len(ids_all), replace=True))
    return {name: summarize(w, ids) for name, w in runs.items()}


def main():
    global LABELS_DIR
    argv = list(sys.argv[1:])
    if argv and argv[0] == "--labels":
        LABELS_DIR = argv[1]
        argv = argv[2:]
    n_reps, seed = int(argv[0]), int(argv[1])
    specs = argv[2:]
    runs = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        runs[name] = per_window(Path(path), LABELS_DIR)
        print(f"loaded {name}: {len(runs[name])} windows", flush=True)
    ids_all = sorted(set.intersection(*[set(w) for w in runs.values()]))
    print(f"common windows: {len(ids_all)}", flush=True)
    point, counts = {}, {}
    for name, w in runs.items():
        counts[name] = {}
        point[name] = summarize(w, ids_all, counts[name])
    _G["ids"], _G["runs"] = ids_all, runs
    seeds = [seed + r for r in range(n_reps)]
    if sys.platform.startswith("linux"):  # fork: workers inherit _G for free
        with ProcessPoolExecutor() as ex:
            reps = list(ex.map(one_rep, seeds, chunksize=4))
    else:  # spawn cannot pickle a path-loaded module; serial is fine for tests
        reps = [one_rep(s) for s in seeds]
    keys = L1 + list(ROLE_KEYS)
    names = list(runs)
    result = {"n_windows": len(ids_all), "n_reps": n_reps, "point": point, "counts": counts, "ci": {}, "diff": {}}
    for name in names:
        result["ci"][name] = {}
        for k in keys:
            arr = np.array([r[name][k] for r in reps], dtype=float)
            arr = arr[~np.isnan(arr)]
            result["ci"][name][k] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))] if len(arr) else None
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = f"{a} - {b}"
            result["diff"][key] = {}
            for k in keys:
                arr = np.array([r[a][k] - r[b][k] for r in reps], dtype=float)
                arr = arr[~np.isnan(arr)]
                pa, pb = point[a][k], point[b][k]
                result["diff"][key][k] = {
                    "point": None if np.isnan(pa) or np.isnan(pb) else float(pa - pb),
                    "ci": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))] if len(arr) else None,
                }
    suffix = "" if LABELS_DIR == "labels" else f"_{LABELS_DIR}"
    out = Path(specs[0].split("=", 1)[1]).parent.parent / f"judge_bootstrap_{'_'.join(names)}{suffix}.json"
    out.write_text(json.dumps(result, indent=1))
    print("wrote", out)
    # human table
    def fmt(v):
        return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}" if abs(v) < 5 else f"{v:.1f}"
    print("\n| key | " + " | ".join(f"{n} [95% CI] (N)" for n in names) + " | " + " | ".join(f"diff {d} [CI]" for d in result["diff"]) + " |")
    for k in keys:
        cells = []
        for n in names:
            ci = result["ci"][n][k]; c = counts[n].get(k)
            cells.append(f"{fmt(point[n][k])} [{fmt(ci[0])}, {fmt(ci[1])}]" + (f" (N={c})" if c is not None else "") if ci else "-")
        dcells = []
        for d in result["diff"].values():
            v = d[k]
            dcells.append(f"{fmt(v['point'])} [{fmt(v['ci'][0])}, {fmt(v['ci'][1])}]" if v["ci"] else "-")
        print(f"| {k.replace('judge_', '')} | " + " | ".join(cells) + " | " + " | ".join(dcells) + " |")


if __name__ == "__main__":
    main()
