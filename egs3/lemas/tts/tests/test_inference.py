import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.inference import DualPromptInference, build_output
from src.layout import HOP

TOKENS = ["<blank>", "<unk>", "a", "b", "<space>", "<spk>", "<lang>", "<de>", "<zh>", "<sos/eos>"]


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    tokens = tmp_path / "tokens.txt"
    tokens.write_text("\n".join(TOKENS) + "\n")
    cfg = OmegaConf.load("conf/training_f5_base_dualprompt.yaml")
    cfg.token_list = str(tokens)
    cfg.model.hidden_size, cfg.model.depth, cfg.model.attention_heads = 32, 1, 2
    cfg.model.text_embedding_size, cfg.model.convolution_layers = 16, 1
    tc = tmp_path / "train.yaml"
    OmegaConf.save(cfg, tc)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model)
    ck = tmp_path / "last.ckpt"
    torch.save({"state_dict": model.state_dict()}, ck)
    stats = tmp_path / "lang_stats.json"
    stats.write_text('{"de": {"tokens_per_sec": 12.0}, "zh": {"tokens_per_sec": 6.0}}')
    monkeypatch.setattr(DualPromptInference, "_load_vocoder", lambda self, n, p: None)
    monkeypatch.setattr(DualPromptInference, "_vocode", lambda self, mel: torch.zeros(mel.shape[-1] * HOP))

    class FakePhon:
        def phonemize(self, text, lang):
            return [c for c in text if c != " "]

    monkeypatch.setattr("src.inference.LEMASPhonemizer", lambda: FakePhon())
    return DualPromptInference(str(tc), str(ck), str(tokens), str(stats), device="cpu", lowpass_hz=None)


def test_build_inputs_matches_dataset_layout(tiny):
    spk = (np.random.randn(24000 * 2 + 100) * 0.05).astype(np.float32)  # not hop aligned
    lp = (np.random.randn(24000) * 0.05).astype(np.float32)
    cond, ids, cf, n_tgt = tiny.build_inputs("ab ba", "de", spk, lp)
    spk_frames, lang_frames = (len(spk) // 768) * 3, (len(lp) // 768) * 3
    assert cond.shape[1] == cf == spk_frames + lang_frames
    assert ids[0, :spk_frames].tolist() == [5] * spk_frames
    assert ids[0, spk_frames:cf].tolist() == [6] * lang_frames
    assert ids[0, cf].item() == 7 and ids[0, cf + 1 :].tolist() == [2, 3, 3, 2]
    assert n_tgt == int(round(4 / 12.0 * 24000 / HOP))


def test_no_lang_prompt_omits_region(tiny):
    spk = (np.random.randn(24000) * 0.05).astype(np.float32)
    cond, ids, cf, _ = tiny.build_inputs("a", "de", spk, None)
    assert set(ids[0, :cf].tolist()) == {5} and cond.shape[1] == cf


def test_call_returns_target_length_wav(tiny):
    spk = (np.random.randn(24000) * 0.05).astype(np.float32)
    out = tiny(text="ab", lang="de", spk_prompt_speech=spk, lang_prompt_speech=spk)
    assert out["wav"].dtype == np.float32 and len(out["wav"]) > 0


def test_build_output_keys():
    d = {"utt_id": "u", "raw_text": "t", "ref_wav_path": "s.flac", "lang_ref_wav_path": "l.flac",
         "gt_wav_path": "g.flac"}
    o = build_output(d, {"wav": np.zeros(3, np.float32)}, 0)
    assert {k: v for k, v in o.items() if k != "wav"} == {
        "utt_id": "u", "text": "t", "ref": "s.flac", "lang_ref": "l.flac", "gt": "g.flac"}


def test_inference_configs_load():
    for name in ("inference_lemas_eval.yaml", "inference_lemas_eval_spk_only.yaml"):
        cfg = OmegaConf.load(f"conf/{name}")
        assert cfg.model._target_ == "src.inference.DualPromptInference"
        assert len(cfg.dataset.test) == 10
        assert cfg.output_fn == "src.inference.build_output"
    a = OmegaConf.load("conf/inference_lemas_eval.yaml")
    b = OmegaConf.load("conf/inference_lemas_eval_spk_only.yaml")
    assert "lang_prompt_speech" in a.input_key and "lang_prompt_speech" not in b.input_key
    assert all(t.data_src_args.use_lang_prompt is False for t in b.dataset.test)
