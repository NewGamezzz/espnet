"""The text-use probe must read the validation set the training config names."""

from local.probe_text_use import ablate, valid_dataset
from omegaconf import OmegaConf
from src.layout import TokenTable


def _cfg(corpus, layout):
    return OmegaConf.create(
        dict(
            dataset=dict(
                valid=[
                    dict(
                        data_src_args=dict(
                            split="valid",
                            manifest_path=str(corpus["manifest"]),
                            token_list=str(corpus["tokens"]),
                            prompt_config=dict(layout=layout),
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
    assert ablate(text, "none", table, None).tolist() == text[:1].tolist()
