r"""No-GPU pre-flight for the MoonCast input file.

Two properties of their pipeline are knowable before a GPU-second is spent,
and both belong in the writeup rather than in a surprise:

* ``Model._clean_text`` REWRITES our script.  It runs on every turn and on
  both reference texts (``_process_text``), stripping curly quotes and
  asterisks, mapping ``...`` and the ellipsis character to a space, and
  mapping ``:`` to ``,``.  MoonCast is the first re-run baseline whose input
  text differs from the reference transcript, so the rows it alters are
  counted here.
* The context is one flat sequence - both reference texts, every turn's
  text, both reference audio token sequences, and then every generated turn
  appended and never dropped - against the model's
  ``pretraining_sequence_length``.  At 50 Hz that grows four times as fast
  per second of audio as FireRedTTS-2's 12.5 Hz did, which is why it is
  checked rather than assumed.

Run it in THEIR environment, from their repo, so the tokenizer and the
cleaner checked are the ones that will run::

    cd /work/hdd/bbjs/ttrachu/development/MoonCast
    .pixi/envs/default/bin/python <recipe>/local/mooncast_preflight.py \
        --input input_v2.jsonl --set-dir <built set>

The text side of the count is EXACT - it uses their own sentencepiece
tokenizer and their own frame layout.  The audio side is an estimate: the
prompt frames are real, but the generated frames come from the reference
duration, and their generation will differ.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: ``PrefixStreamingFlowMatchingDetokenizer.frame_size`` is 480 samples at
#: 24 kHz, so one semantic token is 20 ms.
FRAMES_PER_SEC = 50
#: ``user_msg_start + "user" + speaker + name_end`` is 4 tokens, plus the
#: trailing ``msg_end`` (``infer_with_prompt`` lines 125-132).
TOKENS_PER_TEXT_TURN = 5
#: ``assistant_msg_start + "assistant" + speaker + name_end`` is 4 tokens,
#: ``media_begin + "audio" + media_content`` is 3, and ``media_end +
#: msg_end`` is 2 (lines 96-97, 136-137, 147).
TOKENS_PER_AUDIO_SEGMENT = 9


def estimate_context_tokens(
    prompt_secs: list[float],
    target_sec: float,
    text_token_counts: list[int],
    num_turns: int,
) -> dict:
    """Tokens the whole dialogue would occupy in their flat sequence.

    Reference text, turn text, reference audio and generated audio are all
    rows of the same sequence, so all four count.
    """
    prompt_audio = int(sum(prompt_secs) * FRAMES_PER_SEC)
    target_audio = int(target_sec * FRAMES_PER_SEC)
    text = sum(count + TOKENS_PER_TEXT_TURN for count in text_token_counts)
    framing = TOKENS_PER_AUDIO_SEGMENT * (len(prompt_secs) + num_turns)
    return {
        "prompt_audio_tokens": prompt_audio,
        "target_audio_tokens": target_audio,
        "text_tokens": text,
        "framing_tokens": framing,
        "tokens": prompt_audio + target_audio + text + framing,
    }


def check_rows(
    rows: list[dict],
    cleaner,
    encoder,
    duration_of,
    ceiling: int,
    target_secs: dict[str, float] | None = None,
) -> dict:
    """Clean, tokenize and size every row.

    ``cleaner`` is their ``_clean_text``; ``encoder`` their tokenizer's
    ``encode``; ``duration_of`` maps a wav path to seconds; ``target_secs``
    maps a dialogue id to its reference duration (absent = 0, i.e. text and
    prompts only).
    """
    target_secs = target_secs or {}
    reports: list[dict] = []
    cleaned_rows: list[str] = []
    over: list[str] = []
    for row in rows:
        wid = row["window_id"]
        texts = [turn["text"] for turn in row["dialogue"]]
        refs = [row["role_mapping"][role]["ref_text"] for role in ("0", "1")]
        cleaned = [cleaner(text) for text in texts + refs]
        if cleaned != [text.strip() for text in texts + refs]:
            cleaned_rows.append(wid)
        counts = [len(encoder(text)) for text in cleaned]
        prompts = [row["role_mapping"][role]["ref_audio"] for role in ("0", "1")]
        estimate = estimate_context_tokens(
            [duration_of(path) for path in prompts],
            target_secs.get(wid, 0.0),
            counts,
            len(texts),
        )
        if estimate["tokens"] > ceiling:
            over.append(wid)
        reports.append({"window_id": wid, "num_turns": len(texts), **estimate})
    return {"rows": reports, "cleaned_rows": cleaned_rows, "over_ceiling": over}


def target_durations(set_dir: Path) -> dict[str, float]:
    """Return the reference duration per id, from the built set's GT wavs."""
    import soundfile

    lines = (set_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    out: dict[str, float] = {}
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        best = 0.0
        for channel in record["channels"]:
            rel = channel.get("gt_wav")
            if rel is None:
                continue
            info = soundfile.info(str(set_dir / rel))
            best = max(best, info.frames / info.samplerate)
        out[record["window_id"]] = best
    return out


def main() -> None:
    """CLI entry point - run from their repo, in their environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--set-dir",
        type=Path,
        default=None,
        help="the built set, for reference durations (recommended)",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("."),
        help="their repo root; every resource path is relative to it",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("resources/text2semantic/config.json"),
        help="their model config, for the sequence-length ceiling",
    )
    args = parser.parse_args()

    args.input = args.input.resolve()
    if args.set_dir is not None:
        args.set_dir = args.set_dir.resolve()
    # Their checkpoint paths are relative to the repo root, and running a
    # script by path puts the SCRIPT's directory on sys.path rather than
    # the working directory - so both have to be arranged explicitly.
    repo_dir = args.repo_dir.resolve()
    os.chdir(repo_dir)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    import soundfile

    import inference
    from modules.tokenizer.tokenizer import get_tokenizer_and_extra_tokens

    tokenizer, _ = get_tokenizer_and_extra_tokens()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    ceiling = int(config["pretraining_sequence_length"])

    lines = args.input.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]

    def duration_of(path):
        info = soundfile.info(str(path))
        return info.frames / info.samplerate

    targets = target_durations(args.set_dir) if args.set_dir else None
    report = check_rows(
        rows,
        # _clean_text never touches self, so it needs no live model.
        lambda text: inference.Model._clean_text(None, text),
        tokenizer.encode,
        duration_of,
        ceiling,
        target_secs=targets,
    )

    tokens = [row["tokens"] for row in report["rows"]]
    print(f"{len(rows)} rows checked against a {ceiling}-token ceiling")
    print(
        f"rows their _clean_text alters: {len(report['cleaned_rows'])}"
        + (
            f" ({', '.join(report['cleaned_rows'][:20])})"
            if report["cleaned_rows"]
            else ""
        )
    )
    print(
        f"estimated context tokens: max {max(tokens)}, "
        f"mean {sum(tokens) // len(tokens)}"
    )
    if report["over_ceiling"]:
        print(f"OVER THE CEILING: {', '.join(report['over_ceiling'])}")
    else:
        print("no row is estimated over the ceiling")


if __name__ == "__main__":
    main()
