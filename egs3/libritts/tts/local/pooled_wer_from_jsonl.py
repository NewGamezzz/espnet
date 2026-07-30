#!/usr/bin/env python
"""Pooled-error-count WER supplement for a scorer's ``_wer_results.jsonl``.

The official F5-TTS scorer (`f5_tts/eval/eval_librispeech_test_clean.py`,
via `run_asr_wer` in `f5_tts/eval/utils_eval.py`) writes one JSON record per
utterance with only `wav`, `truth` (raw reference text), `hypo` (raw ASR
hypothesis text), and `wer` (that utterance's own WER ratio). It has no
raw substitution/deletion/insertion counts, and its headline number
(printed as `WER: <mean>` and appended as the file's last summary line) is
the unweighted MEAN of these per-utterance WER ratios -- every utterance
contributes equally regardless of reference length.

This script recomputes a POOLED WER instead: it sums substitutions +
deletions + insertions and reference word counts across every utterance,
then divides once, so long utterances contribute proportionally more error
budget than short ones. This is the standard corpus-level WER used
elsewhere (see the "WER: pool error counts" rule in project memory) and is
NOT the same number as the scorer's own per-utterance mean; both are
reported, labeled, for comparison.

Per-utterance S/D/I counts aren't in the jsonl, so they are recomputed here
via `jiwer.process_words`, using the EXACT normalization the official
scorer applies before its own `process_words` call (confirmed by reading
`run_asr_wer`): strip all punctuation (zhon.hanzi punctuation +
string.punctuation), collapse doubled spaces, then lowercase (English
only; this script does not implement the `zh` character-split path).
Re-running `jiwer.process_words` on the same normalized strings must
therefore reproduce each record's own stored `wer` field -- this script
verifies that on every record and raises if any mismatch exceeds a small
float tolerance, so a normalization drift is caught rather than silently
producing a wrong pooled number.

Usage:
    python local/pooled_wer_from_jsonl.py <path/to/_wer_results.jsonl> [...]

Prints, per file: official mean-of-per-utterance WER (recomputed directly
from the stored `wer` fields, not just trusting the file's own trailing
summary line), the recomputed pooled WER, and the total utterance/word
counts pooling was computed over.
"""

import argparse
import json
import string
import sys

from jiwer import process_words
from zhon.hanzi import punctuation as zh_punctuation

PUNCTUATION_ALL = zh_punctuation + string.punctuation

# Matches run_asr_wer's own tolerance-free re-derivation; per-utterance wer
# values are float64 ratios from jiwer, so allow a small epsilon for
# roundtrip float noise only, not for any real normalization mismatch.
WER_MATCH_TOL = 1e-6


def normalize_en(text: str) -> str:
    """Exact port of run_asr_wer's English normalization path."""
    for ch in PUNCTUATION_ALL:
        text = text.replace(ch, "")
    text = text.replace("  ", " ")
    return text.lower()


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue  # skip blank line / trailing "WER: 0.xxxxx" summary
            records.append(json.loads(line))
    return records


def pooled_wer(records: list[dict]) -> dict:
    total_errors = 0
    total_ref_words = 0
    per_utt_wers = []
    mismatches = []

    for rec in records:
        truth_norm = normalize_en(rec["truth"])
        hypo_norm = normalize_en(rec["hypo"])
        measures = process_words(truth_norm, hypo_norm)

        errors = measures.substitutions + measures.deletions + measures.insertions
        ref_words = measures.substitutions + measures.deletions + measures.hits

        recomputed_wer = errors / ref_words if ref_words > 0 else 0.0
        if abs(recomputed_wer - rec["wer"]) > WER_MATCH_TOL:
            mismatches.append((rec["wav"], rec["wer"], recomputed_wer))

        total_errors += errors
        total_ref_words += ref_words
        per_utt_wers.append(rec["wer"])

    if mismatches:
        lines = "\n".join(f"  {w}: stored={s} recomputed={r}" for w, s, r in mismatches[:10])
        raise RuntimeError(
            f"{len(mismatches)} of {len(records)} utterances' recomputed WER "
            f"diverged from the stored `wer` field by more than {WER_MATCH_TOL} "
            f"-- normalization mismatch, not safe to report a pooled number. "
            f"First mismatches:\n{lines}"
        )

    return {
        "n_utt": len(records),
        "total_ref_words": total_ref_words,
        "total_errors": total_errors,
        "official_mean_wer": sum(per_utt_wers) / len(per_utt_wers) if per_utt_wers else 0.0,
        "pooled_wer": total_errors / total_ref_words if total_ref_words > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_paths", nargs="+", help="One or more _wer_results.jsonl files")
    args = parser.parse_args()

    for path in args.jsonl_paths:
        records = load_records(path)
        result = pooled_wer(records)
        print(f"\n{path}")
        print(f"  n_utt              = {result['n_utt']}")
        print(f"  total_ref_words    = {result['total_ref_words']}")
        print(f"  total_errors (S+D+I) = {result['total_errors']}")
        print(f"  official mean-of-per-utterance WER = {result['official_mean_wer']:.5f}")
        print(f"  pooled (corpus-level) WER          = {result['pooled_wer']:.5f}")


if __name__ == "__main__":
    sys.exit(main())
