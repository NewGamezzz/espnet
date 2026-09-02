"""Export one AMI stratum's frozen manifest as baseline inputs.

Every baseline gets exactly the prompts Chorus used (the eval manifest pins
them as session-absolute spans on source headset channels), cut from the
4-channel FLAC at its native rate, mono, normalized to ``--normalize-db``
active RMS (the v2 convention), plus the window script in turn order with
speaker tags S1..SK assigned by row order.  Three files:

* ``manifest.jsonl`` - the built-set shape ``make_zipvoice_baseline_tsv.py``
  reads (``num_channels``, ``channels[i].prompt_wav/prompt_text``,
  ``turns[j].speaker/text``);
* ``moss.jsonl``     - MOSS-TTSD ``voice_clone_and_continuation`` rows
  (``text`` with ``[S<n>]`` tags, ``prompt_audio_speaker<n>`` /
  ``prompt_text_speaker<n>``, ``base_path``);
* ``firered.jsonl``  - FireRedTTS-2 ``generate_dialogue`` lists
  (``text_list``, ``prompt_wav_list``, ``prompt_text_list``; the tag is the
  first four characters of every prompt text, as their code reads it).

Usage:
    python local/export_ami_baseline_inputs.py --eval-manifest data/eval/ami_test_k3_v1.jsonl \\
        --window-manifest data/manifest/ami_test.jsonl \\
        --dataset-root /work/hdd/bbjs/ttrachu/dataset/ami --out-dir exp/ami/baseline_inputs/k3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from egs3.conversational.tts.dataset.dataset import read_window_manifest  # noqa: E402
from egs3.conversational.tts.local.build_zipvoice_dialog_testset import (  # noqa: E402
    _normalized,
)
from egs3.conversational.tts.src.eval_manifest import load_eval_manifest  # noqa: E402
from egs3.conversational.tts.src.inference import (  # noqa: E402
    _build_turn_pools,
    _resolve_pinned_turns,
)


def export(
    *,
    eval_manifest,
    window_manifest,
    dataset_root,
    out_dir,
    normalize_db: float | None,
) -> dict:
    out_dir = Path(out_dir).resolve()
    (out_dir / "prompt").mkdir(parents=True, exist_ok=True)
    records = {r.window_id: r for r in read_window_manifest(window_manifest)}
    pools = _build_turn_pools(list(records.values()))
    _header, rows = load_eval_manifest(eval_manifest)
    generic, moss, fire, limited = [], [], [], []
    for row in rows:
        rec = records[row["window_id"]]
        src_rows = rec.row_channels
        row_of = {c: i for i, c in enumerate(src_rows)}
        selected = _resolve_pinned_turns(pools.get(rec.session_id, []), row["prompts"], rec)
        path = Path(dataset_root) / rec.audio_relpath
        chans = []
        with sf.SoundFile(str(path)) as audio:
            sr = audio.samplerate
            for i, turn in enumerate(selected):
                audio.seek(int(round(turn.start * sr)))
                block = audio.read(
                    int(round((turn.end - turn.start) * sr)), dtype="float32", always_2d=True
                )
                mono = block[:, turn.channel]
                if normalize_db is not None:
                    mono, _gain, lim = _normalized(mono, sr, normalize_db)
                    if lim:
                        limited.append(f"{rec.window_id}_ch{i}")
                rel = f"prompt/{rec.window_id}_ch{i}.wav"
                sf.write(str(out_dir / rel), mono.astype(np.float32), sr, subtype="PCM_16")
                chans.append({"prompt_wav": rel, "prompt_text": turn.text, "speaker": f"S{i + 1}"})
        turns = [
            {"speaker": f"S{row_of[t.channel] + 1}", "channel": row_of[t.channel], "text": t.text}
            for t in sorted(rec.turns, key=lambda t: (t.start, t.channel))
        ]
        generic.append(
            {
                "window_id": rec.window_id,
                "session_id": rec.session_id,
                "num_channels": rec.num_rows,
                "source_channels": list(src_rows),
                "duration_sec": round(rec.duration, 6),
                "channels": chans,
                "turns": turns,
            }
        )
        script = " ".join(f"[{t['speaker']}] {t['text']}" for t in turns)
        m = {"window_id": rec.window_id, "base_path": str(out_dir), "text": script}
        for i, c in enumerate(chans, start=1):
            m[f"prompt_audio_speaker{i}"] = c["prompt_wav"]
            m[f"prompt_text_speaker{i}"] = c["prompt_text"]
        moss.append(m)
        fire.append(
            {
                "window_id": rec.window_id,
                "text_list": [f"[{t['speaker']}] {t['text']}" for t in turns],
                "prompt_wav_list": [str(out_dir / c["prompt_wav"]) for c in chans],
                "prompt_text_list": [f"[{c['speaker']}] {c['prompt_text']}" for c in chans],
            }
        )
    for name, data in (("manifest.jsonl", generic), ("moss.jsonl", moss), ("firered.jsonl", fire)):
        (out_dir / name).write_text(
            "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in data), "utf-8"
        )
    summary = {
        "n_windows": len(generic),
        "normalize_db": normalize_db,
        "peak_limited": limited,
        "eval_manifest": str(eval_manifest),
        "window_manifest": str(window_manifest),
    }
    (out_dir / "export_meta.json").write_text(json.dumps(summary, indent=2), "utf-8")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--eval-manifest", type=Path, required=True)
    ap.add_argument("--window-manifest", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--normalize-db", type=float, default=-23.0)
    a = ap.parse_args(argv)
    print(
        json.dumps(
            export(
                eval_manifest=a.eval_manifest,
                window_manifest=a.window_manifest,
                dataset_root=a.dataset_root,
                out_dir=a.out_dir,
                normalize_db=a.normalize_db,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
