"""src/prompt_pool.py: pool loading (band filter), SPEAKERS.txt, AMI gender
ids, the seeded distinct-speaker draw, and manifest entry round trip."""
import json

import numpy as np
import pytest
import soundfile as sf

from egs3.conversational.tts.src.prompt_pool import (
    PoolItem,
    ami_gender,
    draw_prompts,
    load_pool_genders,
    load_prompt_pool,
    pool_entry,
    pool_turn_from_entry,
    read_pool_prompt,
)

SR = 24000


def _pool_dir(tmp_path, durs=(2.0, 3.5, 4.4, 6.0)):
    root = tmp_path / "pool"; root.mkdir()
    lines = []
    for i, d in enumerate(durs):
        name = f"u{i}"
        sf.write(str(root / f"{name}.wav"), 0.1 * np.sin(np.arange(int(d * SR)) / 20).astype("float32"), SR, subtype="PCM_16")
        lines.append(json.dumps({"window_id": name, "session_id": f"s{i}", "num_channels": 1,
                                 "turns": [{"channel": 0, "speaker": f"spk{i}", "text": f"t{i}"}],
                                 "channels": [{"gt_wav": f"{name}.wav", "prompt_wav": f"{name}.wav", "speaker": f"spk{i}"}]}))
    (root / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    return root


class TestLoad:
    def test_band_filter_and_fields(self, tmp_path):
        root = _pool_dir(tmp_path)
        items = load_prompt_pool(root / "manifest.jsonl", (3.0, 4.5))
        assert [it.source_id for it in items] == ["u1", "u2"]
        assert items[0].wav.is_absolute() and items[0].text == "t1" and items[0].speaker == "spk1"
        assert items[0].duration == pytest.approx(3.5, abs=1e-3)

    def test_empty_band_is_an_error(self, tmp_path):
        root = _pool_dir(tmp_path)
        with pytest.raises(ValueError, match="no pool utterance"):
            load_prompt_pool(root / "manifest.jsonl", (10.0, 12.0))

    def test_multichannel_row_rejected(self, tmp_path):
        root = _pool_dir(tmp_path)
        bad = root / "bad.jsonl"
        bad.write_text(json.dumps({"window_id": "b", "num_channels": 2, "turns": [{}], "channels": [{}]}) + "\n")
        with pytest.raises(ValueError, match="one-channel"):
            load_prompt_pool(bad, (0.0, 100.0))

    def test_speakers_txt(self, tmp_path):
        p = tmp_path / "SPEAKERS.txt"
        p.write_text(";ID  |SEX| SUBSET\n14   | F | train-clean-360 | 25.03 | X\n1089 | M | test-clean | 1 | Y\n")
        assert load_pool_genders(p) == {"14": "F", "1089": "M"}


class TestGender:
    def test_ami_ids(self):
        assert ami_gender("FEE013") == "F" and ami_gender("MEO015") == "M"
        assert ami_gender("spk_a") is None and ami_gender(None) is None and ami_gender("FEE13") is None


def _items(n_spk=4, per=2):
    return [PoolItem(f"s{s}", tmp := __import__("pathlib").Path(f"/x/s{s}_{u}.wav"), f"text {s} {u}", f"s{s}_{u}", 3.5)
            for s in range(n_spk) for u in range(per)]


class TestDraw:
    def test_distinct_speakers_and_seed(self):
        items = _items()
        a = draw_prompts(items, "w0", 3, 0); b = draw_prompts(items, "w0", 3, 0)
        assert a == b and len({it.speaker for it in a}) == 3
        assert draw_prompts(items, "w0", 3, 1) != a or draw_prompts(items, "w1", 3, 0) != a

    def test_gender_match_when_possible(self):
        items = _items()
        genders = {"s0": "F", "s1": "M", "s2": "F", "s3": "M"}
        for key in ("a", "b", "c", "d"):
            out = draw_prompts(items, key, 2, 0, channel_genders=["F", "M"], pool_genders=genders)
            assert genders[out[0].speaker] == "F" and genders[out[1].speaker] == "M"
        # a channel with unknown gender is unconstrained
        out = draw_prompts(items, "a", 2, 0, channel_genders=[None, "M"], pool_genders=genders)
        assert genders[out[1].speaker] == "M"

    def test_too_few_speakers(self):
        with pytest.raises(ValueError, match="need 5"):
            draw_prompts(_items(), "w", 5, 0)


class TestEntries:
    def test_round_trip_and_read(self, tmp_path):
        root = _pool_dir(tmp_path)
        items = load_prompt_pool(root / "manifest.jsonl", (3.0, 4.5))
        from egs3.conversational.tts.src.prompt_pool import PoolTurn
        t = PoolTurn(channel=2, speaker=items[0].speaker, text=items[0].text, start=0.0,
                     end=items[0].duration, wav=str(items[0].wav), pool_id=items[0].source_id, gender="F")
        e = pool_entry(t)
        assert set(e) == {"channel", "pool_id", "wav", "text", "speaker", "gender"} and "start" not in e
        back = pool_turn_from_entry(e, 2)
        assert back.channel == 2 and back.wav == t.wav and back.end == pytest.approx(t.end, abs=1e-6)
        x = read_pool_prompt(t.wav, SR)
        assert x.ndim == 1 and abs(x.shape[0] / SR - 3.5) < 1e-3
        assert read_pool_prompt(t.wav, 16000).shape[0] == pytest.approx(3.5 * 16000, abs=2)
