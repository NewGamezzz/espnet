"""The text-use probe must read the validation set the training config names."""

import numpy as np
from local.probe_text_use import ablate, lane_head, valid_dataset
from omegaconf import OmegaConf
from src.layout import TokenTable


def _cfg(corpus, layout, **prompt):
    return OmegaConf.create(
        dict(
            dataset=dict(
                valid=[
                    dict(
                        data_src_args=dict(
                            split="valid",
                            manifest_path=str(corpus["manifest"]),
                            token_list=str(corpus["tokens"]),
                            prompt_config=dict(layout=layout, **prompt),
                            seed=0,
                        )
                    )
                ]
            )
        )
    )


def test_valid_dataset_follows_the_config_manifest_and_layout(corpus):
    ds = valid_dataset(_cfg(corpus, "stock"), audio_root=corpus["audio"])
    assert ds.stock and not ds.train
    assert int(ds[0]["cond_frames"][0]) == -1


def test_ablation_keeps_the_tag_of_a_stock_row(corpus):
    ds = valid_dataset(_cfg(corpus, "stock"), audio_root=corpus["audio"])
    table = TokenTable(corpus["tokens"])
    text = ds[0]["text"]
    head = lane_head(text, int(ds[0]["cond_frames"][0]), table, text_tag=True)
    assert head == 1
    assert ablate(text, "none", table, None, head).tolist() == text[:1].tolist()


def test_lane_head_without_tag_is_the_first_target_position(corpus):
    # S0a's lane has no language tag to locate the phones by
    ds = valid_dataset(
        _cfg(corpus, "stock", text_tag=False), audio_root=corpus["audio"]
    )
    table = TokenTable(corpus["tokens"])
    text, cf = ds[0]["text"], int(ds[0]["cond_frames"][0])
    assert lane_head(text, cf, table, text_tag=False) == 0
    assert ablate(text, "none", table, None, head=0).tolist() == []
    shuffled = ablate(text, "shuffled", table, np.random.default_rng(0), head=0)
    assert sorted(shuffled.tolist()) == sorted(text.tolist())


def test_lane_head_with_tag_matches_the_old_behaviour(corpus):
    ds = valid_dataset(_cfg(corpus, "dualprompt"), audio_root=corpus["audio"])
    table = TokenTable(corpus["tokens"])
    text, cf = ds[0]["text"], int(ds[0]["cond_frames"][0])
    assert lane_head(text, cf, table, text_tag=True) == cf + 1
    assert int(text[cf]) == table.tag("de")
