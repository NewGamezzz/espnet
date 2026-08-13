"""Generate golden window-manifest fixtures from the CURRENT (offline) SSSD /
LibriTTS / CANDOR / Fisher builders.

These fixtures freeze today's offline window-manifest output on synthetic
corpora so later tasks in the online-window-sampling refactor can prove the
new pipeline is bit-identical to this one. This script (a) fabricates tiny
synthetic corpora under ``inputs/<corpus>/``, (b) runs each CURRENT builder's
``build()`` against them into a shared scratch recipe dir -- SSSD first,
since it writes the extended vocab the other three builders read -- and (c)
copies the resulting ``data/manifest/*.jsonl`` window manifests to
``<corpus>_<split>.jsonl`` next to this file.

Usage (one-shot CLI, from the recipe root ``egs3/conversational/tts``)::

    PYTHONPATH=<repo_root>:$(pwd) python -m dataset.tests.golden.generate_goldens

Pass ``--out-dir`` to write into a scratch directory instead of overwriting
the committed fixtures (used to verify determinism: run twice into two
different ``--out-dir``s and diff).

Also exposes ``load_golden(name)``, the parity tests' loader for the
committed ``.jsonl`` fixtures (Tasks 5-8).

Determinism
-----------
Every builder call is pinned to ``seed=0`` (already each corpus's
``config.yaml`` default; pinned explicitly here regardless). The synthetic
corpora below reuse the existing end-to-end test helpers (imported, never
copied) and introduce no randomness of their own: session/utterance layout
is fully enumerated in Python. The only randomness anywhere in the pipeline
is inside the builders themselves (``split_sessions``, per-session window
placement, and ``LibriTTSBuilder``'s ``subsample_to_hours`` for the
dev-clean valid split) -- all seeded from the same ``seed=0`` argument.
Re-running this script twice into different ``--out-dir``s therefore
produces byte-identical ``<corpus>_<split>.jsonl`` golden files (verified in
task-1-report.md). The committed ``inputs/candor`` and ``inputs/fisher``
``*.jsonl.gz`` manifests -- written by the REUSED, unmodified
``test_candor.write_candor_manifests`` / ``test_fisher.write_fisher_*``
helpers via plain ``gzip.open(..., "wt")`` -- embed a wall-clock ``mtime`` in
their gzip header and so are NOT byte-identical across repeated runs; their
decompressed content is. This is a property of the reused test helpers
(which this task must not modify or copy), not of the golden manifests
themselves, so it does not affect parity-test correctness. The SSSD corpus
manifests fabricated directly in this file use ``mtime=0`` and so ARE
byte-identical.

Scaled-up SSSD/CANDOR/Fisher corpora
-------------------------------------
The end-to-end tests these fabricators are lifted from use 3-4 sessions.
With ``split_ratios`` train/valid/test = 0.96/0.02/0.02, ``round(3 * 0.02)
== round(4 * 0.02) == 0``, so those unit-test-sized fixtures collapse
valid/test to empty manifests -- fine for narrow unit tests (see
``test_candor.py``/``test_fisher.py``'s "tiny fixture: a split may
legitimately be empty" comments, and ``dataset.read_window_manifest``, which
tolerates this by design), but useless for goldens whose whole purpose is to
let later tasks prove byte-parity for EVERY split. ``split_sessions`` (see
``builder.py``) empirically needs ``n = 26`` sessions to be the smallest
count that puts exactly one session in each of valid/test (24 in train; a
quick sweep of n=25..30 confirmed 26 is the threshold, n=25 still rounds
both splits to 0) -- so SSSD/CANDOR/Fisher are fabricated with 26 sessions
each: a deliberate, minimal, documented departure from "reuse the fixture
verbatim" for session COUNT only. The per-session fabrication logic
(``write_flac``, ``_alternating_sups``, ``two_speaker_session``, ...) is
unchanged and imported, not copied; only the session COUNT and, for SSSD
only, per-session DURATION and audio sample rate are chosen freshly here
(see below). LibriTTS needs no such departure: its train/valid split is
subset-based, not session-ratio-based, and its existing fixture already
yields non-empty output for both.

Every session's audio must still be long enough to survive windowing
(``tail_min=5.0s`` after ``trim_to_turns``) with real turns inside it, which
sets each corpus's minimum safe duration: 8.0s for SSSD's
``_alternating_sups`` (matches the original fixture's own "sess_short",
empirically the shortest duration that keeps 2 utterances after trimming)
and 10.5s for CANDOR/Fisher's ``two_speaker_session`` (that helper places
its second turn only once ``duration > 10.0``; 10.5s keeps a safety margin).
Sizes are held at this minimum, not the original fixtures' longer sessions
(60s/40s), because 26 real sessions at FLAC-compressed pure-tone rates adds
up fast (a naive first pass at 30 sessions of 12-20s came to ~70 MB
committed); minimal safe durations bring the total under it.

SSSD's fabricated audio additionally uses an 8 kHz sample rate (passed
explicitly to ``write_flac``'s existing ``sr`` parameter -- ordinary use of
its documented API, not a modification of it) instead of the original
fixture's 48 kHz default. This is safe because ``SSSDBuilder.build()`` never
reads the audio bytes at all (``preprocessing/sssd.py`` only parses the
gzipped JSON recording/supervision manifests; window placement is a pure
function of the declared turn timestamps, never of the actual waveform) --
unlike CANDOR/Fisher, whose ``measured_durations()`` step reads real FLAC
headers and enforces sample rate/channel count against the corpus's
hardcoded manifest fields inside the reused (unmodified) test helpers, so
their rate cannot be lowered without copying those helpers. The recording
manifest's declared ``sampling_rate`` and the actual FLAC's rate are kept
consistent with each other (both 8 kHz) for SSSD, so nothing about the
frozen window-manifest output changes -- only disk footprint.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import shutil
import string
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
# egs3/conversational/tts/dataset/tests/golden/generate_goldens.py -> repo root
REPO_ROOT = _HERE.parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egs3.conversational.tts.dataset.builder import SSSDBuilder  # noqa: E402
from egs3.conversational.tts.dataset.candor_builder import CandorBuilder  # noqa: E402
from egs3.conversational.tts.dataset.fisher_builder import FisherBuilder  # noqa: E402
from egs3.conversational.tts.dataset.libritts_builder import (  # noqa: E402
    LibriTTSBuilder,
)
from egs3.conversational.tts.dataset.tests import (  # noqa: E402
    test_candor,
    test_fisher,
    test_libritts,
)
from egs3.conversational.tts.dataset.tests.conftest import (  # noqa: E402
    _alternating_sups,
    write_flac,
)

GOLDEN_DIR = Path(__file__).resolve().parent
SEED = 0

# `base_vocab` in conftest.py is a pytest fixture and cannot be called
# directly outside pytest; its body is reconstructed verbatim here.
BASE_VOCAB = (
    ["<blank>", "<unk>", "<space>"]
    + list(string.ascii_lowercase)
    + [".", ",", "?", "!", "'", "<sos/eos>"]
)

# 26 sessions: the minimal count for which split_sessions puts exactly one
# session in each of valid/test (24 in train). See "Scaled-up ..." in the
# module docstring. Durations are held at the minimum safe value (8.0s);
# channel count alternates 2/3 (one 3-channel session in six) to keep the
# original fixture's N=3 coverage without inflating size.
_SSSD_SAMPLE_RATE = 8000
_SSSD_SESSIONS: list[tuple[str, int, float]] = [
    (f"sess_{i:03d}", 3 if i % 6 == 0 else 2, 8.0) for i in range(26)
]

_CANDOR_DURATIONS: dict[str, float] = {f"conv-{i:03d}": 10.5 for i in range(26)}

_FISHER_SESSIONS: dict[str, tuple[float, list[dict]]] = {
    f"fe_03_{i:05d}": (10.5, test_fisher.two_speaker_session(10.5)) for i in range(26)
}


def _gzip_text_writer(path: Path) -> io.TextIOWrapper:
    """Deterministic gzip text writer (mtime=0), for corpus manifests
    fabricated directly in this file (not the reused test-module helpers,
    which use plain ``gzip.open`` and so are exempt -- see module
    docstring)."""
    raw = gzip.GzipFile(filename=str(path), mode="wb", mtime=0)
    return io.TextIOWrapper(raw, encoding="utf-8")


def _make_sssd_corpus(tmp: Path) -> dict:
    """Fabricate a miniature SSSD corpus tree + base vocab file. Lifted from
    conftest.py's ``fake_corpus`` fixture body (module-level helpers
    ``write_flac``/``_alternating_sups`` imported, not copied), scaled to 30
    sessions -- see module docstring."""
    root = tmp / "corpus"
    recordings, supervisions = [], []
    for session_id, num_channels, duration in _SSSD_SESSIONS:
        write_flac(
            root / "original" / f"{session_id}_mixed.flac",
            num_channels,
            duration,
            sr=_SSSD_SAMPLE_RATE,
        )
        recordings.append(
            {
                "id": session_id,
                "sources": [
                    {
                        "type": "file",
                        "channels": list(range(num_channels)),
                        "source": (
                            f"/scratch/elsewhere/original/{session_id}_mixed.flac"
                        ),
                    }
                ],
                "sampling_rate": _SSSD_SAMPLE_RATE,
                "num_samples": int(duration * _SSSD_SAMPLE_RATE),
                "duration": duration,
                "channel_ids": list(range(num_channels)),
            }
        )
        supervisions.extend(_alternating_sups(session_id, num_channels, duration))

    manifests = root / "lhotse_manifests_48"
    manifests.mkdir(parents=True)
    with _gzip_text_writer(manifests / "recordings.jsonl.gz") as f:
        for rec in recordings:
            f.write(json.dumps(rec) + "\n")
    with _gzip_text_writer(manifests / "supervisions.jsonl.gz") as f:
        for sup in supervisions:
            f.write(json.dumps(sup) + "\n")

    vocab_path = tmp / "base_vocab.txt"
    vocab_path.write_text("\n".join(BASE_VOCAB) + "\n", encoding="utf-8")
    return {"root": root, "base_vocab_path": vocab_path}


def _make_libritts_corpus(tmp: Path) -> Path:
    """LibriTTS corpus tree, unmodified from the end-to-end test (subset
    split is not session-ratio-based, so no scale-up is needed)."""
    root = tmp / "LibriTTS"
    test_libritts.fabricate_corpus(root)
    return root


def _make_candor_corpus(tmp: Path) -> dict:
    """CANDOR corpus manifests + pre-transcoded FLACs, 30 sessions (see
    module docstring)."""
    root, flac_dir = test_candor.fabricate_candor(tmp, _CANDOR_DURATIONS)
    return {"root": root, "flac_dir": flac_dir}


def _make_fisher_corpus(tmp: Path) -> dict:
    """Fisher corpus manifests + pre-merged stereo FLACs, 30 sessions (see
    module docstring). ``fabricate_fisher`` writes merged FLACs directly, so
    ``prepare_source``'s ffmpeg step is never invoked here."""
    root, flac_dir = test_fisher.fabricate_fisher(tmp, _FISHER_SESSIONS)
    return {"root": root, "flac_dir": flac_dir}


def generate(out_dir: Path) -> None:
    """Fabricate all four synthetic corpora under ``out_dir/inputs/<corpus>/``,
    run each CURRENT builder's ``build()`` (SSSD first) into a shared scratch
    recipe dir, and copy the resulting window manifests to
    ``out_dir/<corpus>_<split>.jsonl``."""
    out_dir = Path(out_dir)
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    sssd_tmp = inputs_dir / "sssd"
    sssd_tmp.mkdir(parents=True, exist_ok=True)
    sssd = _make_sssd_corpus(sssd_tmp)

    libritts_tmp = inputs_dir / "libritts"
    libritts_tmp.mkdir(parents=True, exist_ok=True)
    libritts_root = _make_libritts_corpus(libritts_tmp)

    candor_tmp = inputs_dir / "candor"
    candor_tmp.mkdir(parents=True, exist_ok=True)
    candor = _make_candor_corpus(candor_tmp)

    fisher_tmp = inputs_dir / "fisher"
    fisher_tmp.mkdir(parents=True, exist_ok=True)
    fisher = _make_fisher_corpus(fisher_tmp)

    # The recipe dir is pure build OUTPUT (window manifests + extended
    # vocab), not fabricated corpus input, so it is scratch-only and never
    # committed; only the corpus trees above and the golden manifests copied
    # out below are committed.
    with tempfile.TemporaryDirectory(prefix="golden_build_") as scratch:
        recipe_dir = Path(scratch) / "recipe"

        # SSSD MUST run first: it writes the extended vocab (recipe_dir/
        # data/tokens/vocab.txt) that LibriTTS/CANDOR/Fisher normalize
        # transcripts against.
        SSSDBuilder().build(
            recipe_dir=recipe_dir,
            dataset_root=sssd["root"],
            seed=SEED,
            base_vocab_path=sssd["base_vocab_path"],
        )
        LibriTTSBuilder().build(
            recipe_dir=recipe_dir, dataset_root=libritts_root, seed=SEED
        )
        CandorBuilder().build(
            recipe_dir=recipe_dir,
            dataset_root=candor["root"],
            flac_dir=candor["flac_dir"],
            seed=SEED,
        )
        FisherBuilder().build(
            recipe_dir=recipe_dir,
            dataset_root=fisher["root"],
            fisher_flac_dir=fisher["flac_dir"],
            seed=SEED,
        )

        manifest_dir = recipe_dir / "data" / "manifest"
        copies = {
            "sssd_train.jsonl": "train.jsonl",
            "sssd_valid.jsonl": "valid.jsonl",
            "sssd_test.jsonl": "test.jsonl",
            "libritts_train.jsonl": "libritts_train.jsonl",
            "libritts_valid.jsonl": "libritts_valid.jsonl",
            "candor_train.jsonl": "candor_train.jsonl",
            "candor_valid.jsonl": "candor_valid.jsonl",
            "candor_test.jsonl": "candor_test.jsonl",
            "fisher_train.jsonl": "fisher_train.jsonl",
            "fisher_valid.jsonl": "fisher_valid.jsonl",
            "fisher_test.jsonl": "fisher_test.jsonl",
        }
        for golden_name, src_name in copies.items():
            shutil.copyfile(manifest_dir / src_name, out_dir / golden_name)


def load_golden(name: str) -> list[dict]:
    """Parsed JSON objects of a committed golden manifest, in file order."""
    path = Path(__file__).parent / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=GOLDEN_DIR,
        help=(
            "Directory to write golden fixtures + inputs/ into (default: "
            "this package's own directory, i.e. the committed fixtures). "
            "Pass a scratch directory to check determinism without "
            "touching the committed files."
        ),
    )
    args = parser.parse_args()
    generate(args.out_dir)
    print(f"Golden fixtures written under {args.out_dir}")


if __name__ == "__main__":
    main()
