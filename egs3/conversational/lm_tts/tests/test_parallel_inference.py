"""Regression tests for the two BagPiper generation bugs found on Delta.

Both bugs live in espnet2/speechlm/model/speechlm/lm/parallel.py and only
manifest during autoregressive inference (teacher-forced forwards were
always healthy), which is why the rest of the suite never caught them:

1. `_step` embedded fed-back tokens with a bare
   `embed_tokens(input_ids).sum(dim=2)`, without the pad-zeroing `_embed`
   applies - and the checkpoint's row-0 (pad) embedding is nonzero noise,
   so every text decode step gained (num_stream - 1) copies of that noise
   and generation degenerated deterministically.
2. `inference` re-invoked `inference_segment` with the same kwargs, and the
   segment prefill unconditionally re-embedded the whole prompt onto the
   accumulated KV cache - corrupting every segment after the first.
"""

import copy

import torch

from .conftest import TINY_TEXT_START


def test_step_embedding_zeroes_stream_pad(tiny_parallel_llm):
    model = tiny_parallel_llm
    text_id = TINY_TEXT_START + 5
    token = torch.zeros((1, 1, model.num_stream), dtype=torch.long)
    token[0, 0, 0] = text_id

    row0 = model.model.embed_tokens.weight[0]
    assert row0.abs().sum() > 0  # the checkpoint condition: pad row is noise

    # The summed embedding for [id, 0, ..., 0] must be embed(id) alone.
    summed = model._embed_and_sum_streams(token)
    expected = model.model.embed_tokens.weight[text_id]
    assert torch.equal(summed[0, 0], expected)

    # And _step's input_ids path must match feeding the correctly-zeroed
    # embedding explicitly (pre-fix these differed by (num_stream-1)*row0).
    with torch.no_grad():
        logits_ids, _ = model._step(input_ids=token)
        logits_emb, _ = model._step(input_embeds=model._embed(token, {}))
    assert torch.equal(logits_ids, logits_emb)


def test_continuation_cache_growth_independent_of_prompt_length(tiny_parallel_llm):
    model = tiny_parallel_llm
    config = {"text": {"temperature": 0, "topk": 1, "max_step": 3}}

    def _prompt(length):
        seqs = torch.zeros((1, length, model.num_stream), dtype=torch.long)
        seqs[0, :, 0] = torch.arange(length) % 8 + TINY_TEXT_START
        return seqs

    with torch.no_grad():
        _, cache = model.inference_segment(
            config, cache=None, enforce_modality="text",
            first_segment=True, seqs=_prompt(6),
        )
    base_len = cache.get_seq_length()

    deltas, hypo_lens = [], []
    for cont_prompt_len in (4, 11):  # continuation must IGNORE the prompt
        with torch.no_grad():
            hypos, cont_cache = model.inference_segment(
                config, cache=copy.deepcopy(cache), enforce_modality="text",
                first_segment=False, seqs=_prompt(cont_prompt_len),
            )
        deltas.append(cont_cache.get_seq_length() - base_len)
        hypo_lens.append(hypos[0][0].shape[0])

    # greedy + identical starting cache => identical continuation both times
    assert hypo_lens[0] == hypo_lens[1]
    assert deltas[0] == deltas[1]
    # 1 (<|assistant|>) + decode steps + 1 (finalize/extend token)
    assert deltas[0] == 1 + hypo_lens[0] + 1
