"""LEMAS recipe TTSSystem.

The stock TTS system with a Lightning module that tells the dataset the epoch.

``create_token_list`` and ``remove_long_short`` are inherited from the stock
``espnet3.systems.tts.system.TTSSystem`` (the recipe uses the first and skips
the second). ``collect_stats`` is inherited too but is not in this recipe's
default stage list: batches come from ``src/sampler.py``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from omegaconf import OmegaConf

from espnet3.components.trainers.trainer import ESPnet3LightningTrainer
from espnet3.systems.base.training import _instantiate_model
from espnet3.systems.tts.system import TTSSystem as _StockTTSSystem

logger = logging.getLogger(__name__)


class TTSSystem(_StockTTSSystem):
    """TTS system for the LEMAS dual-prompt recipe.

    ``train`` builds ``src.lit_module.LEMASLightningModule`` so the dataset
    learns the epoch before every dataloader build (online prompt draws).
    """

    def _build_trainer(self) -> ESPnet3LightningTrainer:
        from src.lit_module import LEMASLightningModule

        config = self.training_config
        model = _instantiate_model(config)
        lit_model = LEMASLightningModule(model, config)
        return ESPnet3LightningTrainer(
            model=lit_model,
            exp_dir=config.exp_dir,
            config=config.trainer,
            best_model_criterion=config.best_model_criterion,
        )

    def train(self, *args, **kwargs):
        """Run the training stage with the recipe's Lightning module."""
        self._reject_stage_args("train", args, kwargs)
        start = time.perf_counter()
        self._prepare_training_runtime()
        trainer = self._build_trainer()
        fit_kwargs: Dict[str, Any] = {}
        if hasattr(self.training_config, "fit") and self.training_config.fit:
            fit_kwargs = OmegaConf.to_container(self.training_config.fit, resolve=True)
        trainer.fit(**fit_kwargs)
        logger.info(
            "Training finished in %.2fs | exp_dir=%s",
            time.perf_counter() - start,
            self.training_config.exp_dir,
        )
