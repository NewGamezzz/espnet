"""Conversational TTS system: dataset preparation + multi-branch F5 training.

Mirrors the libritts ``TTSSystem`` training wiring, minus the stages this
recipe does not use (x-vectors, token lists, stats: the SSSD builder already
produces the extended vocab, and no feature normalization is collected for
F5 fine-tuning).  The model comes from ``instantiate(config.model)``
(``src.build_model.build_multibranch_f5``); the LightningModule is the
recipe-local subclass with the two-group optimizer and the conversation
batch sampler.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from egs3.conversational.tts.src.lit_module import ConversationalLightningModule
from espnet3.components.trainers.trainer import ESPnet3LightningTrainer
from espnet3.parallel.parallel import set_parallel
from espnet3.systems.base.system import BaseSystem

logger = logging.getLogger(__name__)


class ConversationalTTSSystem(BaseSystem):
    """System with ``create_dataset`` (inherited: SSSD builder) and ``train``."""

    def _ensure_directories(self) -> None:
        config = self.training_config
        Path(config.exp_dir).mkdir(parents=True, exist_ok=True)

    def _prepare_training_runtime(self) -> None:
        config = self.training_config
        self._ensure_directories()

        if config.get("parallel"):
            set_parallel(config.parallel)

        if config.get("seed") is not None:
            L.seed_everything(int(config.seed), workers=True)

        torch.set_float32_matmul_precision("high")

    def _build_trainer(self) -> ESPnet3LightningTrainer:
        config = self.training_config
        model = instantiate(config.model)
        lit_model = ConversationalLightningModule(model, config)
        return ESPnet3LightningTrainer(
            model=lit_model,
            exp_dir=config.exp_dir,
            config=config.trainer,
            best_model_criterion=config.best_model_criterion,
        )

    def train(self, *args, **kwargs):
        """Run the training stage (``trainer.fit`` with ``config.fit`` kwargs)."""
        self._reject_stage_args("train", args, kwargs)
        start = time.perf_counter()
        self._prepare_training_runtime()

        trainer = self._build_trainer()

        fit_kwargs: Dict[str, Any] = {}
        if hasattr(self.training_config, "fit") and self.training_config.fit:
            fit_kwargs = OmegaConf.to_container(self.training_config.fit, resolve=True)

        trainer.fit(**fit_kwargs)
        logger.info(
            "Training finished in %.2fs | exp_dir=%s model=%s",
            time.perf_counter() - start,
            self.training_config.exp_dir,
            (
                self.training_config.model.get("_target_", None)
                if isinstance(self.training_config.model, DictConfig)
                else None
            ),
        )
