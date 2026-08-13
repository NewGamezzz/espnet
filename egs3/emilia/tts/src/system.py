"""Emilia recipe TTSSystem: the LibriTTS system plus a create_shape stage.

Copied from egs3/libritts/tts/src/system.py with remove_long_short,
create_token_list and compute_xvectors dropped (Emilia's builder and F5's
downloaded vocab.txt handle those roles instead). collect_stats is
kept even though it never runs as part of this recipe's DEFAULT_STAGES:
Task 10 uses it to validate the analytic create_shape stage against the
real collect_stats over a small Emilia subset, and keeping it here avoids a
cross-recipe dependency on the libritts system class for that comparison.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from espnet2.train.abs_gan_espnet_model import AbsGANESPnetModel
from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.components.trainers.trainer import ESPnet3LightningTrainer
from espnet3.parallel.parallel import set_parallel
from espnet3.systems.base.system import BaseSystem
from espnet3.utils.task_utils import get_espnet_model, save_espnet_config

logger = logging.getLogger(__name__)


def _instantiate_model(config: DictConfig) -> Any:
    task = config.get("task")
    if task:
        model_config = OmegaConf.to_container(config.model, resolve=True)
        return get_espnet_model(task, model_config)
    return instantiate(config.model)


class TTSSystem(BaseSystem):
    """TTS-specific system.

    This system adds:
      - Analytic feats_shape synthesis (create_shape), replacing collect_stats
        in the default stage list.

    Additional stage log paths:
        | Stage         | Path reference                     |
        |---            |---                                 |
        | create_shape  | training_config.create_shape.save_path |
    """

    def __init__(
        self,
        training_config=None,
        inference_config=None,
        metrics_config=None,
        **kwargs,
    ) -> None:
        """Initialize the TTS system with TTS-specific stage mappings."""
        super().__init__(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=metrics_config,
            stage_log_mapping={
                "create_shape": "training_config.create_shape.save_path",
            },
            **kwargs,
        )

    def create_shape(self, *args, **kwargs):
        """Synthesize feats_shape analytically from manifest durations.

        Replaces collect_stats. See the spec, section 5.4.

        Config:
            training_config.create_shape.splits: list of split names
            training_config.create_shape.save_path: stats dir root
            training_config.create_shape.hop_length / sample_rate / n_mels
        """
        self._reject_stage_args("create_shape", args, kwargs)
        cfg = self.training_config.get("create_shape", None)
        if cfg is None:
            raise RuntimeError(
                "training_config.create_shape must be set for the "
                "create_shape stage."
            )
        from egs3.emilia.tts.dataset.dataset import EmiliaDataset
        from egs3.emilia.tts.src.shape import write_shape_file

        save_root = Path(cfg["save_path"])
        for split in cfg["splits"]:
            # Manifest paths come from this stage's own config block, not
            # from the DataOrganizer entries. The stage stays independently
            # testable and does not depend on how DataOrganizer wraps sources.
            dataset = EmiliaDataset(
                split=split,
                recipe_dir=self.training_config.recipe_dir,
                manifest_path=cfg["manifest_paths"][split],
                load_speech=False,
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

    def _ensure_directories(self) -> None:
        config = self.training_config
        Path(config.exp_dir).mkdir(parents=True, exist_ok=True)
        if hasattr(config, "stats_dir"):
            Path(config.stats_dir).mkdir(parents=True, exist_ok=True)

    def _build_trainer(self) -> ESPnet3LightningTrainer:
        config = self.training_config
        model = _instantiate_model(config)
        if isinstance(model, AbsGANESPnetModel):
            # Lazy: egs3/emilia/tts/src/gan_trainer.py does not exist (F5-TTS
            # is not a GAN model, so this branch is currently unreachable in
            # this recipe). A module-level import would break every import
            # of src.system, since libritts's gan_trainer.py pulls in a
            # models/gan_model.py that has no reason to exist here.
            from src.gan_trainer import build_gan_trainer

            return build_gan_trainer(config, model)

        lit_model = ESPnetLightningModule(model, config)
        return ESPnet3LightningTrainer(
            model=lit_model,
            exp_dir=config.exp_dir,
            config=config.trainer,
            best_model_criterion=config.best_model_criterion,
        )

    def _prepare_training_runtime(self) -> None:
        config = self.training_config
        self._ensure_directories()

        if config.get("parallel"):
            set_parallel(config.parallel)

        if config.get("seed") is not None:
            L.seed_everything(int(config.seed), workers=True)

        torch.set_float32_matmul_precision("high")

    def collect_stats(self, *args, **kwargs):
        """Run the collect_stats stage using the configured trainer.

        Prepares the training runtime (directories, parallelism, seed), then
        delegates to the trainer's ``collect_stats`` method.  Positional and
        keyword stage arguments are rejected to avoid silent misconfiguration.

        Args:
            *args: Must be empty.  Passing any positional argument raises
                ``ValueError`` via ``_reject_stage_args``.
            **kwargs: Must be empty.  Passing any keyword argument raises
                ``ValueError`` via ``_reject_stage_args``.

        Returns:
            None

        Raises:
            ValueError: If any positional or keyword arguments are passed.

        Notes:
            The ``normalize: null`` pattern from recipe configs is intentionally
            preserved — no normalization is applied during stats collection.

        Examples:
            >>> from omegaconf import OmegaConf
            >>> cfg = OmegaConf.create({"exp_dir": "/tmp/exp"})
            >>> system = TTSSystem(training_config=cfg)
            >>> system.collect_stats()  # runs stats collection end-to-end
        """
        self._reject_stage_args("collect_stats", args, kwargs)
        start = time.perf_counter()
        self._prepare_training_runtime()

        # Preserve `normalize: null` from recipe configs.
        trainer = self._build_trainer()
        trainer.collect_stats()
        logger.info(
            "Collect stats finished in %.2fs | exp_dir=%s stats_dir=%s",
            time.perf_counter() - start,
            self.training_config.exp_dir,
            getattr(self.training_config, "stats_dir", None),
        )

    def train(self, *args, **kwargs):
        """Run the training stage using the configured trainer.

        Prepares the runtime, optionally saves the ESPnet config, then calls
        ``trainer.fit`` with any keyword arguments drawn from
        ``training_config.fit``.  For GAN models, a ``GANTTSLightningTrainer``
        is used automatically; for all other models, ``ESPnet3LightningTrainer``
        is used.

        Args:
            *args: Must be empty.  Passing any positional argument raises
                ``ValueError`` via ``_reject_stage_args``.
            **kwargs: Must be empty.  Passing any keyword argument raises
                ``ValueError`` via ``_reject_stage_args``.

        Returns:
            None

        Raises:
            ValueError: If any positional or keyword arguments are passed.

        Notes:
            ``training_config.fit`` is forwarded verbatim to ``trainer.fit``.
            Common keys include ``max_epochs``, ``ckpt_path``, etc.

        Examples:
            >>> from omegaconf import OmegaConf
            >>> cfg = OmegaConf.create({
            ...     "exp_dir": "/tmp/exp",
            ...     "task": "tts",
            ...     "model": {"_target_": "my.Model"},
            ...     "fit": {"max_epochs": 10},
            ... })
            >>> system = TTSSystem(training_config=cfg)
            >>> system.train()  # trains the model for 10 epochs
        """
        self._reject_stage_args("train", args, kwargs)
        start = time.perf_counter()
        self._prepare_training_runtime()

        task = self.training_config.get("task")
        if task:
            save_espnet_config(task, self.training_config, self.training_config.exp_dir)

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
