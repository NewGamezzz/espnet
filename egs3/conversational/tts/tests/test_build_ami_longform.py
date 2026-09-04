"""local/build_ami_longform.py: one AMI meeting -> one external-manifest
dialogue (first in-band turn per participant as the prompt, the rest as the
script, masked full-length ground truth), on a fabricated 4-channel meeting."""
import json
import string
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.dataset.preprocessing.sssd import Turn
from egs3.conversational.tts.local.build_ami_longform import (
    build_longform,
    pick_prompt_turn,
)
from egs3.conversational.tts.local.build_zipvoice_dialog_testset import active_rms_db
from egs3.conversational.tts.src.external_testset import load_external_manifest

SR = 24000


def _turns():
    # A: "yeah" (sub-lexical), then a 4 s lexical turn (prompt), then more
    # B: an 0.8 s lexical fragment (too short), then a 3 s turn (prompt)
    # C: a 12 s turn (too long), then a 3 s turn overlapped by D (not solo),
    #    then a 2.5 s solo turn (prompt)
    # D: one 3 s turn (prompt) - D's only turn, so D has no script left
    return [
        Turn(0, "MEO015", "yeah", 0.5, 0.9),
        Turn(0, "MEO015", "one two three four", 2.0, 6.0),
        Turn(1, "FEE013", "hm okay right", 7.0, 7.8),
        Turn(1, "FEE013", "five six seven", 9.0, 12.0),
        Turn(2, "MEE014", "a long turn " * 5, 13.0, 25.0),
        Turn(2, "MEE014", "eight nine ten", 26.0, 29.0),
        Turn(3, "FEE016", "eleven twelve thirteen", 28.5, 31.5),
        Turn(2, "MEE014", "fourteen fifteen sixteen", 33.0, 35.5),
        Turn(0, "MEO015", "seventeen eighteen nineteen", 37.0, 40.0),
        Turn(1, "FEE013", "twenty twenty-one twenty-two", 41.0, 44.0),
        Turn(2, "MEE014", "twenty-three twenty-four twenty-five", 45.0, 48.0),
    ]


class TestPickPrompt:
    def test_first_lexical_in_band_solo_turn(self):
        turns = _turns()
        kw = dict(min_words=3, band=(2.0, 10.0), solo_guard=0.3)
        assert pick_prompt_turn(turns, 0, **kw) == (turns[1], "gated_solo")
        assert pick_prompt_turn(turns, 1, **kw) == (turns[3], "gated_solo")
        assert pick_prompt_turn(turns, 2, **kw) == (turns[7], "gated_solo")
        # D's only turn overlaps C: no solo tier, so the in-band tier keeps D
        assert pick_prompt_turn(turns, 3, **kw) == (turns[6], "in_band")

    def test_excluded_span_falls_to_the_next_gated_turn_then_to_solo(self):
        turns = _turns()
        kw = dict(min_words=3, band=(2.0, 10.0), solo_guard=0.3)
        assert pick_prompt_turn(turns, 0, excluded=frozenset({(0, 2.0, 6.0)}), **kw) == (turns[8], "gated_solo")
        both = frozenset({(0, 2.0, 6.0), (0, 37.0, 40.0)})
        assert pick_prompt_turn(turns, 0, excluded=both, **kw) == (turns[1], "solo")

    def test_no_lexical_in_band_turn_drops_the_participant(self):
        turns = [Turn(0, "a", "yeah", 0.5, 0.9), Turn(0, "a", "a very long turn " * 6, 2.0, 20.0)]
        assert pick_prompt_turn(turns, 0, min_words=3, band=(2.0, 10.0), solo_guard=0.3) is None


def _corpus(tmp_path: Path, seconds: float = 50.0):
    root = tmp_path / "ami"
    (root / "ami_flac").mkdir(parents=True)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = np.stack([0.2 * np.sin(2 * np.pi * 200 * (c + 1) * t) for c in range(4)], axis=1).astype("float32")
    sf.write(str(root / "ami_flac" / "ES2004a.flac"), x, SR, subtype="PCM_16", format="FLAC")
    sessions = tmp_path / "sessions.jsonl"
    line = {
        "session_id": "ES2004a", "audio_relpath": "ami_flac/ES2004a.flac", "sample_rate": SR,
        "num_channels": 4, "duration": seconds,
        "speakers": {"0": "MEO015", "1": "FEE013", "2": "MEE014", "3": "FEE016"},
        "turns": [{"channel": tr.channel, "speaker": tr.speaker, "text": tr.text,
                   "start": tr.start, "end": tr.end} for tr in _turns()],
    }
    sessions.write_text(json.dumps(line) + "\n")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["<blank>", "<unk>", "<space>"] + list(string.ascii_lowercase)
                               + [".", ",", "?", "!", "'", "-", "<sos/eos>", "<turn>", "<OTHER>"]) + "\n")
    return root, sessions, vocab


class TestBuild:
    def test_build_one_meeting(self, tmp_path):
        root, sessions, vocab = _corpus(tmp_path)
        out = tmp_path / "ami-longform"
        meta = build_longform(sessions=sessions, dataset_root=root, out_dir=out, exclude_spans=None,
                              min_words=3, band=(2.0, 10.0), solo_guard=0.3, mask_guard=0.15,
                              normalize_db=-23.0, meetings=None)
        rows = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
        assert len(rows) == 1
        r = rows[0]
        # D's only turn is its prompt (in_band tier) -> nothing left to say -> D dropped; K = 3
        assert r["window_id"] == "ES2004a" and r["num_channels"] == 3
        assert r["source_channels"] == [0, 1, 2]
        assert [c["prompt_text"] for c in r["channels"]] == [
            "one two three four", "five six seven", "fourteen fifteen sixteen"]
        assert [c["prompt_tier"] for c in r["channels"]] == ["gated_solo"] * 3
        texts = [t["text"] for t in r["turns"]]
        assert "one two three four" not in texts and "five six seven" not in texts
        assert "fourteen fifteen sixteen" not in texts
        assert "eleven twelve thirteen" not in texts  # D's turn is gone with D
        assert texts[0] == "yeah" and texts[-1] == "twenty-three twenty-four twenty-five"
        assert [t["channel"] for t in r["turns"]][:3] == [0, 1, 2]
        # prompt wav: mono, native rate, normalized, right length
        wav, sr = sf.read(str(out / r["channels"][0]["prompt_wav"]))
        assert wav.ndim == 1 and sr == SR and abs(len(wav) / sr - 4.0) < 0.01
        assert abs(active_rms_db(wav, sr) - (-23.0)) < 0.5
        # gt: full length, masked outside the channel's script turns (prompt turn masked too)
        gt, sr = sf.read(str(out / r["channels"][0]["gt_wav"]))
        assert gt.ndim == 1 and abs(len(gt) / sr - 50.0) < 0.01
        assert float(np.abs(gt[int(3.0 * SR):int(5.0 * SR)]).max()) == 0.0   # prompt turn masked
        assert float(np.abs(gt[int(0.6 * SR):int(0.8 * SR)]).max()) > 0.0    # "yeah" kept
        assert float(np.abs(gt[int(38.0 * SR):int(39.0 * SR)]).max()) > 0.0  # later turn kept
        assert float(np.abs(gt[int(20.0 * SR):int(21.0 * SR)]).max()) == 0.0  # C speaking, A silent
        assert meta["n_meetings"] == 1 and meta["meetings"]["ES2004a"]["num_channels"] == 3
        assert meta["meetings"]["ES2004a"]["dropped_channels"] == [3]
        assert meta["meetings"]["ES2004a"]["prompt_tiers"] == {"0": "gated_solo", "1": "gated_solo", "2": "gated_solo"}
        # loads through the real external-manifest loader
        recs = load_external_manifest(out / "manifest.jsonl", vocab)
        assert recs[0].num_channels == 3 and recs[0].gt_paths is not None
        assert abs(recs[0].gt_duration_sec - 50.0) < 0.01
        assert len(recs[0].turns) == len(r["turns"])

    def test_meeting_subset_and_determinism(self, tmp_path):
        root, sessions, vocab = _corpus(tmp_path)
        a = build_longform(sessions=sessions, dataset_root=root, out_dir=tmp_path / "a", exclude_spans=None,
                           min_words=3, band=(2.0, 10.0), solo_guard=0.3, mask_guard=0.15,
                           normalize_db=-23.0, meetings=["ES2004a"])
        b = build_longform(sessions=sessions, dataset_root=root, out_dir=tmp_path / "b", exclude_spans=None,
                           min_words=3, band=(2.0, 10.0), solo_guard=0.3, mask_guard=0.15,
                           normalize_db=-23.0, meetings=["ES2004a"])
        assert (tmp_path / "a" / "manifest.jsonl").read_bytes() == (tmp_path / "b" / "manifest.jsonl").read_bytes()
        none = build_longform(sessions=sessions, dataset_root=root, out_dir=tmp_path / "c", exclude_spans=None,
                              min_words=3, band=(2.0, 10.0), solo_guard=0.3, mask_guard=0.15,
                              normalize_db=-23.0, meetings=["ES2014a"])
        assert none["n_meetings"] == 0


# ---------------------------------------------------------------------------
# Record-level exclusion spans + pinned id lists (the Fisher long-form arm,
# "Design - Long-Form Two-Speaker Evaluation on Fisher", 2026-09-04).
# ---------------------------------------------------------------------------


def _fisher_turns():
    # exclusion span [20.0, 23.5] (unintelligible speech, no channel):
    # ch0: "yeah" (sub-lexical), a 4 s lexical solo turn (prompt), a turn
    #      overlapping the span (dropped from the script), a kept turn
    # ch1: first lexical solo turn overlaps the span (NOT a prompt, dropped),
    #      then the 3 s prompt, then a kept turn
    return [
        Turn(0, "4123", "yeah", 0.5, 0.9),
        Turn(0, "4123", "one two three four", 2.0, 6.0),
        Turn(1, "4775", "alpha beta gamma", 19.5, 22.5),
        Turn(0, "4123", "seventeen eighteen nineteen", 23.0, 24.5),
        Turn(1, "4775", "five six seven", 25.0, 28.0),
        Turn(0, "4123", "a b c d", 30.0, 33.0),
        Turn(1, "4775", "delta epsilon zeta", 40.0, 43.0),
    ]


def _fisher_corpus(tmp_path: Path, seconds: float = 50.0):
    root = tmp_path / "fisher_flac"
    (root / "000").mkdir(parents=True)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = np.stack([0.2 * np.sin(2 * np.pi * 200 * (c + 1) * t) for c in range(2)], axis=1).astype("float32")
    sf.write(str(root / "000" / "fe_03_00027.flac"), x, SR, subtype="PCM_16", format="FLAC")
    sessions = tmp_path / "sessions_fisher_test.jsonl"
    rows = []
    for sid, spans in (("fe_03_00027", [[20.0, 23.5]]), ("fe_03_00126", [])):
        rows.append({
            "session_id": sid, "window_id": sid, "audio_relpath": "000/fe_03_00027.flac",
            "sample_rate": SR, "num_channels": 2, "duration": seconds, "atomic": False,
            "exclusion_spans": spans,
            "turns": [{"channel": tr.channel, "speaker": tr.speaker, "text": tr.text,
                       "start": tr.start, "end": tr.end} for tr in _fisher_turns()],
        })
    sessions.write_text("".join(json.dumps(r) + "\n" for r in rows))
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["<blank>", "<unk>", "<space>"] + list(string.ascii_lowercase)
                               + [".", ",", "?", "!", "'", "-", "<sos/eos>", "<turn>", "<OTHER>"]) + "\n")
    return root, sessions, vocab


class TestExclusionSpans:
    def test_span_overlap_disqualifies_a_prompt_candidate_at_every_tier(self):
        turns = _fisher_turns()
        kw = dict(min_words=3, band=(2.0, 10.0), solo_guard=0.3)
        assert pick_prompt_turn(turns, 1, **kw) == (turns[2], "gated_solo")
        assert pick_prompt_turn(turns, 1, exclusion_spans=[(20.0, 23.5)], **kw) == (turns[4], "gated_solo")
        # a span is never relaxed: with every ch1 turn covered there is no prompt
        assert pick_prompt_turn(turns, 1, exclusion_spans=[(19.0, 44.0)], **kw) is None

    def test_build_drops_span_turns_on_both_channels_and_masks_them(self, tmp_path):
        root, sessions, vocab = _fisher_corpus(tmp_path)
        out = tmp_path / "fisher-longform"
        meta = build_longform(sessions=sessions, dataset_root=root, out_dir=out, exclude_spans=None,
                              min_words=3, band=(2.0, 10.0), solo_guard=0.3, mask_guard=0.15,
                              normalize_db=-23.0, meetings=None)
        rows = {json.loads(l)["window_id"]: json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()}
        assert set(rows) == {"fe_03_00027", "fe_03_00126"}
        r = rows["fe_03_00027"]
        assert r["num_channels"] == 2
        assert [c["prompt_text"] for c in r["channels"]] == ["one two three four", "five six seven"]
        texts = [t["text"] for t in r["turns"]]
        assert "alpha beta gamma" not in texts and "seventeen eighteen nineteen" not in texts
        assert texts == ["yeah", "a b c d", "delta epsilon zeta"]
        gt0, sr = sf.read(str(out / r["channels"][0]["gt_wav"]))
        assert float(np.abs(gt0[int(23.2 * SR):int(24.3 * SR)]).max()) == 0.0   # span turn masked
        assert float(np.abs(gt0[int(31.0 * SR):int(32.0 * SR)]).max()) > 0.0    # kept turn audible
        gt1, _ = sf.read(str(out / r["channels"][1]["gt_wav"]))
        assert float(np.abs(gt1[int(20.0 * SR):int(21.0 * SR)]).max()) == 0.0   # ch1 span turn masked
        assert float(np.abs(gt1[int(41.0 * SR):int(42.0 * SR)]).max()) > 0.0
        m = meta["meetings"]["fe_03_00027"]
        assert m["n_turns_excluded"] == 2 and m["exclusion_sec"] == 3.5
        # the record without spans keeps every non-prompt turn
        r2 = rows["fe_03_00126"]
        assert [c["prompt_text"] for c in r2["channels"]] == ["one two three four", "alpha beta gamma"]
        assert len(r2["turns"]) == 5 and meta["meetings"]["fe_03_00126"]["n_turns_excluded"] == 0
        assert meta["exclusion"] == {"n_records_with_spans": 1, "n_turns_excluded": 2}
        recs = load_external_manifest(out / "manifest.jsonl", vocab)
        assert {x.num_channels for x in recs} == {2} and all(x.gt_paths is not None for x in recs)

    def test_ids_file_pins_the_subset(self, tmp_path):
        from egs3.conversational.tts.local.build_ami_longform import main

        root, sessions, vocab = _fisher_corpus(tmp_path)
        ids = tmp_path / "ids.txt"
        ids.write_text("# pinned\nfe_03_00126\n\n")
        out = tmp_path / "sub"
        assert main(["--sessions", str(sessions), "--dataset-root", str(root), "--out", str(out),
                     "--ids-file", str(ids)]) == 0
        rows = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
        assert [r["window_id"] for r in rows] == ["fe_03_00126"]
        meta = json.loads((out / "build_meta.json").read_text())
        assert meta["params"]["ids_file"] == str(ids)
        assert meta["sessions_md5"] and len(meta["sessions_md5"]) == 32
        with pytest.raises(SystemExit):
            main(["--sessions", str(sessions), "--dataset-root", str(root), "--out", str(tmp_path / "x"),
                  "--ids-file", str(ids), "--meetings", "fe_03_00027"])


# ---------------------------------------------------------------------------
# External prompt pool (LibriTTS test-clean prompts, the Fisher long-form
# PRIMARY construction: CoVoMix2 protocol, no corpus turn spent as a prompt).
# ---------------------------------------------------------------------------


def _pool(tmp_path: Path):
    from egs3.conversational.tts.local.build_ami_longform import load_prompt_pool

    root = tmp_path / "pool"
    root.mkdir()
    rows = []
    # 4 speakers x 2 utterances; speaker 900 is out of band (12 s)
    for spk, sex in (("100", "M"), ("200", "F"), ("300", "M"), ("400", "F")):
        for k, dur in ((1, 4.0), (2, 6.0)):
            n = int(dur * SR)
            x = 0.3 * np.sin(2 * np.pi * (100 + 50 * int(spk[0])) * np.arange(n) / SR)
            wav = root / f"{spk}_{k}.wav"
            sf.write(str(wav), x.astype("float32"), SR, subtype="PCM_16")
            rows.append({"window_id": f"libritts_{spk}_{k}", "session_id": f"libritts_{spk}", "num_channels": 1,
                         "turns": [{"channel": 0, "speaker": spk, "text": f"pool text {spk} {k}"}],
                         "channels": [{"prompt_wav": f"{spk}_{k}.wav", "prompt_text": "x",
                                       "gt_wav": f"{spk}_{k}.wav", "speaker": spk}]})
    n = int(12.0 * SR)
    sf.write(str(root / "900_1.wav"), (0.3 * np.sin(np.arange(n) / 50)).astype("float32"), SR, subtype="PCM_16")
    rows.append({"window_id": "libritts_900_1", "session_id": "libritts_900", "num_channels": 1,
                 "turns": [{"channel": 0, "speaker": "900", "text": "too long"}],
                 "channels": [{"prompt_wav": "900_1.wav", "prompt_text": "x", "gt_wav": "900_1.wav", "speaker": "900"}]})
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    speakers = tmp_path / "SPEAKERS.txt"
    speakers.write_text(";header\n100 | M | test-clean | 8 | a\n200 | F | test-clean | 8 | b\n"
                        "300 | M | test-clean | 8 | c\n400 | F | test-clean | 8 | d\n900 | M | test-clean | 8 | e\n")
    return root / "manifest.jsonl", speakers, load_prompt_pool


class TestPromptPool:
    def test_pool_loads_in_band_only(self, tmp_path):
        manifest, _, load_prompt_pool = _pool(tmp_path)
        pool = load_prompt_pool(manifest, (3.0, 10.0))
        assert sorted({it.speaker for it in pool}) == ["100", "200", "300", "400"] and len(pool) == 8
        assert all(3.0 <= it.duration <= 10.0 for it in pool)
        with pytest.raises(ValueError):
            load_prompt_pool(manifest, (20.0, 30.0))

    def test_draw_is_seeded_distinct_and_gender_matched(self, tmp_path):
        from egs3.conversational.tts.local.build_ami_longform import draw_prompts, load_pool_genders

        manifest, speakers, load_prompt_pool = _pool(tmp_path)
        pool = load_prompt_pool(manifest, (3.0, 10.0))
        genders = load_pool_genders(speakers)
        assert genders == {"100": "M", "200": "F", "300": "M", "400": "F", "900": "M"}
        a = draw_prompts(pool, "fe_03_00027", 2, 0)
        b = draw_prompts(pool, "fe_03_00027", 2, 0)
        assert [x.source_id for x in a] == [x.source_id for x in b]
        assert a[0].speaker != a[1].speaker
        assert draw_prompts(pool, "fe_03_00027", 2, 1) != a or draw_prompts(pool, "fe_03_00126", 2, 0) != a
        for sid in ("fe_03_00027", "fe_03_00126", "fe_03_00202"):
            d = draw_prompts(pool, sid, 2, 0, channel_genders=["F", "M"], pool_genders=genders)
            assert (genders[d[0].speaker], genders[d[1].speaker]) == ("F", "M")
        with pytest.raises(ValueError):
            draw_prompts(pool, "x", 5, 0)

    def test_build_with_pool_keeps_the_whole_script(self, tmp_path):
        root, sessions, vocab = _fisher_corpus(tmp_path)
        manifest, speakers, _ = _pool(tmp_path)
        cg = tmp_path / "genders.json"
        cg.write_text(json.dumps({"fe_03_00027": ["F", "M"], "fe_03_00126": ["M", "M"]}))
        out = tmp_path / "fisher-longform-pool"
        meta = build_longform(sessions=sessions, dataset_root=root, out_dir=out, exclude_spans=None,
                              min_words=3, band=(2.0, 10.0), solo_guard=0.3, mask_guard=0.15,
                              normalize_db=-23.0, meetings=None, prompt_pool=manifest,
                              pool_band=(3.0, 10.0), prompt_seed=0, pool_genders=speakers, channel_genders=cg)
        rows = {json.loads(l)["window_id"]: json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()}
        r = rows["fe_03_00027"]
        # no corpus turn spent: script = every turn minus the exclusion-span turns
        assert [t["text"] for t in r["turns"]] == ["yeah", "one two three four", "five six seven",
                                                   "a b c d", "delta epsilon zeta"]
        assert [c["prompt_tier"] for c in r["channels"]] == ["external", "external"]
        assert [c["speaker"] for c in r["channels"]] == [meta["meetings"]["fe_03_00027"]["prompt_speakers"][k] for k in ("0", "1")]
        assert r["channels"][0]["speaker"] in ("200", "400") and r["channels"][1]["speaker"] in ("100", "300")
        assert r["channels"][0]["prompt_source"].startswith("libritts_") and r["channels"][0]["prompt_span"] is None
        assert r["channels"][0]["prompt_text"].startswith("pool text")
        wav, sr = sf.read(str(out / r["channels"][0]["prompt_wav"]))
        assert sr == SR and wav.ndim == 1 and (abs(len(wav) / sr - 4.0) < 0.01 or abs(len(wav) / sr - 6.0) < 0.01)
        assert abs(active_rms_db(wav, sr) - (-23.0)) < 0.5
        gt0, _ = sf.read(str(out / r["channels"][0]["gt_wav"]))
        assert float(np.abs(gt0[int(3.0 * SR):int(5.0 * SR)]).max()) > 0.0   # the ex-prompt turn is now GT
        assert float(np.abs(gt0[int(23.2 * SR):int(24.3 * SR)]).max()) == 0.0  # span turn still masked
        r2 = rows["fe_03_00126"]
        assert len(r2["turns"]) == 7 and {c["speaker"] for c in r2["channels"]} <= {"100", "300"}
        assert meta["pool_size"] == 8 and meta["pool_speakers"] == 4
        assert meta["params"]["prompt_pool"] == str(manifest) and meta["params"]["prompt_seed"] == 0
        assert meta["exclusion"]["n_turns_excluded"] == 2
        recs = load_external_manifest(out / "manifest.jsonl", vocab)
        assert all(x.num_channels == 2 and x.gt_paths is not None for x in recs)
        # determinism across builds
        build_longform(sessions=sessions, dataset_root=root, out_dir=tmp_path / "again", exclude_spans=None,
                       min_words=3, band=(2.0, 10.0), solo_guard=0.3, mask_guard=0.15,
                       normalize_db=-23.0, meetings=None, prompt_pool=manifest,
                       pool_band=(3.0, 10.0), prompt_seed=0, pool_genders=speakers, channel_genders=cg)
        assert (out / "manifest.jsonl").read_bytes() == (tmp_path / "again" / "manifest.jsonl").read_bytes()

    def test_channel_genders_missing_session_is_an_error(self, tmp_path):
        root, sessions, vocab = _fisher_corpus(tmp_path)
        manifest, speakers, _ = _pool(tmp_path)
        cg = tmp_path / "genders.json"
        cg.write_text(json.dumps({"fe_03_00027": ["F", "M"]}))
        with pytest.raises(ValueError):
            build_longform(sessions=sessions, dataset_root=root, out_dir=tmp_path / "x", exclude_spans=None,
                           min_words=3, band=(2.0, 10.0), solo_guard=0.3, mask_guard=0.15,
                           normalize_db=-23.0, meetings=None, prompt_pool=manifest,
                           pool_genders=speakers, channel_genders=cg)
