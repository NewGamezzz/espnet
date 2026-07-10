"""Optional integration test over the real SSSD corpus.

Skipped unless ``SSSD_ROOT`` points at the corpus root (e.g. on Delta:
/work/hdd/bbjs/ttrachu/dataset/ScalableSpontaneousSpeechDataset).
"""

import json
import os
import random

import pytest

from egs3.conversational.tts.dataset.dataset import ConversationDataset
from egs3.conversational.tts.dataset.sssd import (
    load_recordings,
    load_supervisions,
    merge_turns,
)
from egs3.conversational.tts.dataset.text import extend_vocab
from egs3.conversational.tts.dataset.windows import build_windows, to_json

pytestmark = pytest.mark.skipif(
    not os.environ.get("SSSD_ROOT"), reason="SSSD_ROOT not set"
)

WINDOW_KW = dict(window_min=10.0, window_max=60.0, boundary_guard=0.0, tail_min=5.0)


@pytest.fixture(scope="module")
def corpus_root():
    from pathlib import Path

    return Path(os.environ["SSSD_ROOT"])


def test_real_corpus_windows_and_item(corpus_root, tmp_path, base_vocab):
    manifests = corpus_root / "lhotse_manifests_48"
    recordings = load_recordings(manifests / "recordings.jsonl.gz")
    supervisions = load_supervisions(manifests / "supervisions.jsonl.gz", recordings)
    session_ids = sorted(set(recordings) & set(supervisions))
    assert len(session_ids) > 1000, "expected ~1587 sessions"

    all_records = []
    for sid in session_ids[:3]:
        turns = merge_turns(supervisions[sid], merge_gap=1.0)
        records, _ = build_windows(
            sid,
            recordings[sid],
            turns,
            rng=random.Random(f"0:window:{sid}"),
            **WINDOW_KW,
        )
        for w in records:
            assert WINDOW_KW["tail_min"] <= w.duration <= WINDOW_KW["window_max"] + 1e-6
            for t in w.turns:
                assert w.t0 <= t.start and t.end <= w.t1
        all_records.extend(records)
    assert all_records

    manifest = tmp_path / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for w in all_records:
            f.write(json.dumps(to_json(w)) + "\n")
    vocab_path = tmp_path / "vocab.txt"
    # The turn texts are not normalized here, so tolerate OOV by widening the
    # charset: this test checks audio plumbing, not the vocab pipeline.
    charset = sorted({c for w in all_records for t in w.turns for c in t.text})
    vocab_path.write_text(
        "\n".join(
            extend_vocab(
                base_vocab + [c for c in charset if c not in base_vocab and c != " "]
            )
        )
        + "\n",
        encoding="utf-8",
    )
    ds = ConversationDataset(
        split="valid",
        manifest_path=manifest,
        vocab_path=vocab_path,
        dataset_root=corpus_root,
        permute_channels=False,
    )
    item = ds[0]
    w = all_records[0]
    assert item["speech"].shape[0] == w.num_channels
    assert abs(item["speech"].shape[1] - round(24000 * (w.t1 - w.t0))) <= 1
    assert item["speech"].abs().max() > 0, "audio should not be silent"
