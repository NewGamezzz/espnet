"""The exact path the train stage takes: config -> DataOrganizer -> collate."""

import shutil
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.sampler import BlockBatchSampler

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


def test_training_config_uses_the_plain_loader_with_the_block_sampler(corpus, tmp_path):
    cfg = OmegaConf.load(RECIPE_DIR / "conf" / "training_f5_base_dualprompt.yaml")
    assert (
        cfg.dataloader.train.iter_factory is None
        and cfg.dataloader.valid.iter_factory is None
    )
    assert cfg.trainer.use_distributed_sampler is False
    assert cfg.trainer.reload_dataloaders_every_n_epochs == 1
    cfg.recipe_dir = str(RECIPE_DIR)
    cfg.data_dir = str(tmp_path)
    cfg.token_list = str(corpus["tokens"])
    (tmp_path / "manifest").mkdir()
    for split in ("train", "valid"):
        shutil.copy(corpus["manifest"], tmp_path / "manifest" / f"{split}.tsv")
        cfg.dataset[split][0].data_src_args.audio_root = str(corpus["audio"])
    organizer = instantiate(cfg.dataset)
    organizer.train.use_espnet_collator = True
    sampler = instantiate(cfg.batch_sampler, organizer.train)
    assert isinstance(sampler, BlockBatchSampler)
    loader = torch.utils.data.DataLoader(
        organizer.train,
        batch_sampler=sampler,
        collate_fn=instantiate(cfg.dataloader.collate_fn),
        num_workers=0,
    )
    ids, batch = next(iter(loader))
    assert batch["cond_frames"].shape == (len(ids), 1) and batch["speech"].ndim == 2
