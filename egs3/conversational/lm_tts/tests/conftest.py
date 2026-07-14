import pytest
import torch
from torch import nn


def _tiny_qwen3_config():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    return Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=512,
        architectures=["Qwen3ForCausalLM"],
    )


@pytest.fixture()
def tiny_qwen3():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    torch.manual_seed(0)
    return Qwen3ForCausalLM(_tiny_qwen3_config()).eval()


# --- tiny ParallelLLM (BagPiper speechlm class around the tiny Qwen3) ------
#
# Scaled-down mirror of the real vocab layout:
#   [0, 16)    specials (pad=0, bos=1, eos=2, eot=3, system=4, user=5,
#              assistant=6, text=7, audio=8, image=9, video=10, toolcall=11)
#   [16, 80)   text tokens
#   [80, 112)  audio codec tokens: 4 streams x 8 (like 152192 + s*1025)
_NUM_STREAM = 4
_SPECIALS = [
    "<|pad|>", "<|bos|>", "<|eos|>", "<|eot|>", "<|system|>", "<|user|>",
    "<|assistant|>", "<|text|>", "<|audio|>", "<|image|>", "<|video|>",
    "<|toolcall|>", "<|unused12|>", "<|unused13|>", "<|unused14|>",
    "<|unused15|>",
]
TINY_TEXT_START, TINY_TEXT_END = 16, 80
TINY_AUDIO_START, TINY_AUDIO_LAYER = 80, 8
TINY_VOCAB_SIZE = TINY_AUDIO_START + _NUM_STREAM * TINY_AUDIO_LAYER


class _MockDiscreteAudioIO(nn.Module):
    is_discrete = True

    def num_stream(self):
        return _NUM_STREAM

    def decode_batch(self, seq, lengths):
        return seq

    def dummy_forward(self, ref_tensor):
        # DDP unused-parameter keep-alive hook (see abs_io.py:216); this
        # mock has no parameters, so contribute nothing.
        return torch.zeros((), device=ref_tensor.device, dtype=ref_tensor.dtype)


@pytest.fixture()
def tiny_parallel_llm(monkeypatch):
    """The real ParallelLLM class built around the tiny Qwen3, wired the way
    from_pretrained wires the 8B model - including the checkpoint condition
    that embed_tokens row 0 (the pad embedding) is nonzero noise
    (from_pretrained re-inits the rebuilt embedding with nn.init.normal_,
    and padding_idx only zeroes the gradient)."""
    import transformers

    from espnet2.speechlm.model.speechlm.lm.parallel import build_parallel_hf_class

    cfg = _tiny_qwen3_config()
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: cfg
    )
    cls = build_parallel_hf_class("tiny-qwen3")
    torch.manual_seed(0)
    model = cls(cfg)

    hidden = cfg.hidden_size
    embed = nn.Embedding(TINY_VOCAB_SIZE, hidden, padding_idx=0)
    nn.init.normal_(embed.weight, mean=0.0, std=0.02)  # row 0 stays NONZERO
    model.model.embed_tokens = embed
    model.lm_head = nn.Linear(hidden, TINY_VOCAB_SIZE, bias=False)
    nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)

    model.num_stream = _NUM_STREAM
    model.stream_emb = nn.Embedding(_NUM_STREAM, hidden)
    nn.init.zeros_(model.stream_emb.weight)
    model.multimodal_io_dict = nn.ModuleDict(
        {"discrete_audio": _MockDiscreteAudioIO()}
    )
    model.adaptor = nn.ModuleDict()

    vocab = list(_SPECIALS)
    vocab += [f"<text{i}>" for i in range(TINY_TEXT_END - TINY_TEXT_START)]
    vocab += [
        f"<audio{s}_{i}>"
        for s in range(_NUM_STREAM)
        for i in range(TINY_AUDIO_LAYER)
    ]
    assert len(vocab) == TINY_VOCAB_SIZE
    model.vocab = vocab
    model.vocab_intervals = {
        "special_token": [(0, TINY_TEXT_START)],
        "text": [(TINY_TEXT_START, TINY_TEXT_END)],
        "discrete_audio": [
            (
                TINY_AUDIO_START + s * TINY_AUDIO_LAYER,
                TINY_AUDIO_START + (s + 1) * TINY_AUDIO_LAYER,
            )
            for s in range(_NUM_STREAM)
        ],
    }
    model.loss_intervals = list(model.vocab_intervals["discrete_audio"])

    model.eval()
    model.prepare_inference()
    return model
