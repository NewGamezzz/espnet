import pytest
import torch


@pytest.fixture()
def tiny_qwen3():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=512,
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).eval()
