import json

import numpy as np
import soundfile as sf
from dataset.lemas_eval import LEMASEvalDataset
from local.prepare_lemas_eval import build_eval_manifest


def _meta(tmp_path):
    root = tmp_path / "eval"
    rows = []
    for lang, vids in (("de", ["v1", "v2", "v3"]), ("zh", ["1", "2", "3"])):
        for v in vids:
            key = (
                f"de_{v}AAAAAAAAA-00001-00000000-00000500"
                if lang == "de"
                else f"zh_emilia_zh_000000000{v}"
            )
            rel = f"{lang}/{v}.flac"
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                p,
                np.zeros(16000 * 5, np.float32),
                16000,
                format="FLAC",
                subtype="PCM_16",
            )
            words = [
                {
                    "word": f"w{i}",
                    "start": i * 1.0 + 0.1,
                    "end": i * 1.0 + 0.9,
                    "score": 1.0,
                }
                for i in range(5)
            ]
            rows.append(
                {
                    "key": key,
                    "file_name": rel,
                    "dur": 5.0,
                    "txt": " ".join(w["word"] for w in words),
                    "align": {"words": words},
                }
            )
    meta = root / "metadata.jsonl"
    meta.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return meta, root


def test_manifest_and_clips(tmp_path):
    meta, root = _meta(tmp_path)
    out = build_eval_manifest(meta, root, tmp_path / "out")
    lines = out.read_text().splitlines()
    assert len(lines) == 6
    utt, lang, text, spk, lp, gt = lines[0].split("\t")
    # boundary nearest 30% of 5 s = end of w1 (1.9 s);
    # target = start of w2 (2.1 s) to end
    assert text == "w2 w3 w4"
    assert (
        abs(sf.info(spk).duration - 1.9) < 1e-6
        and abs(sf.info(gt).duration - 2.9) < 1e-6
    )
    assert lp != spk and lp.endswith("_spk.flac")
    assert all(line.split("\t")[1] == line.split("\t")[0][:2] for line in lines)
    langs_of_partner = {line.split("\t")[4].split("/")[-1][:2] for line in lines}
    assert langs_of_partner == {"de", "zh"}


def test_eval_dataset_keys(tmp_path):
    meta, root = _meta(tmp_path)
    out = build_eval_manifest(meta, root, tmp_path / "out")
    ds = LEMASEvalDataset(out)
    s = ds[0]
    assert s["spk_prompt_speech"].dtype == np.float32
    assert len(s["spk_prompt_speech"]) == int(1.9 * 24000)
    assert set(s) >= {
        "utt_id",
        "text",
        "lang",
        "raw_text",
        "spk_prompt_speech",
        "lang_prompt_speech",
        "ref_wav_path",
        "lang_ref_wav_path",
        "gt_wav_path",
    }
    assert "lang_prompt_speech" not in LEMASEvalDataset(out, use_lang_prompt=False)[0]
    assert len(LEMASEvalDataset(out, lang="zh")) == 3
