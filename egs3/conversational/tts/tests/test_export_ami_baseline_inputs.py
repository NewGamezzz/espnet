"""local/export_ami_baseline_inputs.py: one AMI stratum's frozen manifest ->
normalized prompt wavs plus generic / MOSS-TTSD / FireRedTTS-2 input files."""
import json

import soundfile as sf

from egs3.conversational.tts.local.export_ami_baseline_inputs import export
from egs3.conversational.tts.src.eval_manifest import write_eval_manifest

from .test_inference import EXT_TOKENS, _write_four_fixture


def _manifest(tmp_path):
    man = tmp_path / "eval.jsonl"
    write_eval_manifest(
        man,
        {"record_type": "header", "manifest_version": 1, "split": "valid"},
        [
            {
                "record_type": "window",
                "window_id": "four_w00000",
                "session_id": "four",
                "t0": 5.0,
                "t1": 13.0,
                "prompts": [
                    {"channel": 1, "start": 20.0, "end": 23.0},
                    {"channel": 3, "start": 25.0, "end": 28.0},
                ],
            }
        ],
    )
    return man


def test_export_writes_prompts_and_three_input_files(tmp_path):
    fx = _write_four_fixture(tmp_path)
    out = tmp_path / "export"
    summary = export(
        eval_manifest=_manifest(tmp_path),
        window_manifest=fx["manifest"],
        dataset_root=fx["dataset_root"],
        out_dir=out,
        normalize_db=-23.0,
    )
    assert summary["n_windows"] == 1
    rows = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    r = rows[0]
    assert r["num_channels"] == 2 and r["source_channels"] == [1, 3]
    assert [c["speaker"] for c in r["channels"]] == ["S1", "S2"]
    assert [t["speaker"] for t in r["turns"]] == ["S1", "S2"]
    assert [t["channel"] for t in r["turns"]] == [0, 1]
    assert r["turns"][0]["text"] == "abc def ghi"
    wav, sr = sf.read(str(out / r["channels"][0]["prompt_wav"]))
    assert wav.ndim == 1 and sr == 48000 and abs(len(wav) / sr - 3.0) < 0.01
    moss = json.loads((out / "moss.jsonl").read_text().splitlines()[0])
    assert moss["text"].startswith("[S1] abc def ghi [S2] bead cab fed")
    assert set(moss) >= {
        "prompt_audio_speaker1", "prompt_text_speaker1",
        "prompt_audio_speaker2", "prompt_text_speaker2", "base_path",
    }
    assert moss["prompt_text_speaker1"] == "cage jade"
    fire = json.loads((out / "firered.jsonl").read_text().splitlines()[0])
    assert fire["text_list"] == ["[S1] abc def ghi", "[S2] bead cab fed"]
    assert [p[:4] for p in fire["prompt_text_list"]] == ["[S1]", "[S2]"]
    assert len(fire["prompt_wav_list"]) == 2
    meta = json.loads((out / "export_meta.json").read_text())
    assert meta["normalize_db"] == -23.0 and meta["n_windows"] == 1


def test_export_normalizes_prompt_level(tmp_path):
    from egs3.conversational.tts.local.build_zipvoice_dialog_testset import active_rms_db

    fx = _write_four_fixture(tmp_path)
    export(
        eval_manifest=_manifest(tmp_path),
        window_manifest=fx["manifest"],
        dataset_root=fx["dataset_root"],
        out_dir=tmp_path / "e",
        normalize_db=-23.0,
    )
    wav, sr = sf.read(str(tmp_path / "e" / "prompt" / "four_w00000_ch0.wav"))
    assert abs(active_rms_db(wav, sr) - (-23.0)) < 0.5


def test_export_gt_writes_masked_normalized_anchor_loadable_as_external_testset(tmp_path):
    import numpy as np
    from egs3.conversational.tts.local.build_zipvoice_dialog_testset import active_rms_db
    from egs3.conversational.tts.src.external_testset import load_external_manifest

    fx = _write_four_fixture(tmp_path)
    out = tmp_path / "out_gt"
    summary = export(
        eval_manifest=_manifest(tmp_path),
        window_manifest=fx["manifest"],
        dataset_root=fx["dataset_root"],
        out_dir=out,
        normalize_db=-23.0,
        gt=True,
        gt_mask_guard=0.15,
    )
    assert summary["gt"] is True and summary["gt_mask_guard"] == 0.15
    r = json.loads((out / "manifest.jsonl").read_text().splitlines()[0])
    assert [c["gt_wav"] for c in r["channels"]] == ["gt/four_w00000_ch0.wav", "gt/four_w00000_ch1.wav"]
    assert r["t0"] == 5.0 and r["t1"] == 13.0
    for i, c in enumerate(r["channels"]):
        wav, sr = sf.read(str(out / c["gt_wav"]))
        assert wav.ndim == 1 and abs(len(wav) / sr - 8.0) < 0.01
        # masked: some silence, and the active part sits at -23 dBFS
        assert float(np.abs(wav).min()) == 0.0 and (np.abs(wav) == 0).mean() > 0.2
        assert abs(active_rms_db(wav, sr) - (-23.0)) < 0.5
        # every window turn of this row lies inside the un-masked region
        for t in r["turns"]:
            if t["channel"] != i:
                continue
    vocab = tmp_path / "vocab_ext.txt"
    vocab.write_text("\n".join(EXT_TOKENS) + "\n", encoding="utf-8")
    recs = load_external_manifest(out / "manifest.jsonl", vocab)
    assert recs[0].num_channels == 2 and recs[0].gt_paths is not None
    assert abs(recs[0].gt_duration_sec - 8.0) < 0.01
