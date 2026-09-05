"""The exact path the train stage takes: config -> DataOrganizer -> collate."""

import shutil
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet2.train.collate_fn import CommonCollateFn

RECIPE_DIR = Path(__file__).resolve().parents[1]


def test_training_config_resolves_local_dataset_and_collates(corpus, tmp_path):
    cfg = OmegaConf.load(RECIPE_DIR / "conf" / "training_f5_base_dualprompt.yaml")
    cfg.recipe_dir = str(RECIPE_DIR)
    cfg.data_dir = str(tmp_path)
    cfg.token_list = str(corpus["tokens"])
    (tmp_path / "manifest").mkdir()
    for split in ("train", "valid"):
        shutil.copy(corpus["manifest"], tmp_path / "manifest" / f"{split}.tsv")
        cfg.dataset[split][0].data_src_args.audio_root = str(corpus["audio"])
    # no data_src in the config: espnet3 must find dataset.Dataset by itself
    organizer = instantiate(cfg.dataset)
    assert len(organizer.train) == 9 and len(organizer.valid) == 9
    collate = instantiate(cfg.dataloader.collate_fn)
    assert isinstance(collate, CommonCollateFn)
    # the lightning module flips this flag for CommonCollateFn (uid, dict) items
    organizer.train.use_espnet_collator = True
    ids, batch = collate([organizer.train[0], organizer.train[1]])
    assert len(ids) == 2
    assert batch["cond_frames"].shape == (2, 1)
    assert batch["cond_frames"].dtype == torch.int64
    assert "cond_frames_lengths" not in batch
    assert batch["text"].shape[0] == 2 and batch["text_lengths"].shape == (2,)
    assert batch["speech"].ndim == 2 and batch["speech_lengths"].shape == (2,)
    assert (batch["speech_lengths"] <= batch["speech"].shape[1]).all()
