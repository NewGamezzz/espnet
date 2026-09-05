"""LEMAS recipe TTSSystem: the stock TTS system plus an analytic create_shape
stage and a Lightning module that tells the dataset the epoch.

``create_token_list`` and ``remove_long_short`` are inherited from the stock
``espnet3.systems.tts.system.TTSSystem`` (the recipe uses the first and skips
the second). ``collect_stats`` is inherited too but is not in this recipe's
default stage list; ``create_shape`` replaces it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf

from espnet3.components.trainers.trainer import ESPnet3LightningTrainer
from espnet3.systems.base.training import _instantiate_model
from espnet3.systems.tts.system import TTSSystem as _StockTTSSystem

logger = logging.getLogger(__name__)


class TTSSystem(_StockTTSSystem):
    """TTS system for the LEMAS dual-prompt recipe.

    Adds two things to the stock TTS system:

    - ``create_shape``: analytic ``feats_shape`` synthesis from manifest
      durations at the longest prompt layout (replaces ``collect_stats``).
    - ``train`` builds ``src.lit_module.LEMASLightningModule`` so the dataset
      learns the epoch before every dataloader build (online prompt draws).

    Additional stage log path:
        | Stage        | Path reference                          |
        |---           |---                                      |
        | create_shape | training_config.create_shape.save_path  |
    """

    def __init__(
        self,
        training_config=None,
        inference_config=None,
        metrics_config=None,
        **kwargs,
    ) -> None:
        super().__init__(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=metrics_config,
            **kwargs,
        )
        mapping = getattr(self, "stage_log_mapping", None)
        if isinstance(mapping, dict):
            mapping["create_shape"] = "training_config.create_shape.save_path"

    def create_shape(self, *args, **kwargs):
        """Synthesize feats_shape analytically from manifest durations.

        Config (``training_config.create_shape``):
            - ``splits``: list of split names
            - ``save_path``: stats dir root (``<save_path>/<split>/feats_shape``)
            - ``manifest_paths``: split -> manifest tsv
            - ``hop_length`` / ``sample_rate`` / ``n_mels``
            - ``prompt_config``: the prompt length ranges; the shape is an upper
              bound computed at the longest layout.
        """
        self._reject_stage_args("create_shape", args, kwargs)
        cfg = self.training_config.get("create_shape", None)
        if cfg is None:
            raise RuntimeError(
                "training_config.create_shape must be set for the create_shape stage."
            )
        from dataset.dataset import LEMASDataset
        from src.shape import write_shape_file

        prompt_config = cfg.get("prompt_config")
        if prompt_config is not None:
            prompt_config = OmegaConf.to_container(prompt_config, resolve=True)
        save_root = Path(cfg["save_path"])
        for split in cfg["splits"]:
            dataset = LEMASDataset(
                split=split,
                recipe_dir=self.training_config.recipe_dir,
                manifest_path=cfg["manifest_paths"][split],
                load_speech=False,
                prompt_config=prompt_config,
            )
            out_path = save_root / split / "feats_shape"
            n = write_shape_file(
                dataset,
                out_path,
                hop_length=int(cfg["hop_length"]),
                sample_rate=int(cfg["sample_rate"]),
                n_mels=int(cfg["n_mels"]),
            )
            logger.info("create_shape: %s -> %d rows", out_path, n)

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
