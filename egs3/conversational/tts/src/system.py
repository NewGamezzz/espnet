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

# ``src.external_inference.MODE``, duplicated as a literal so dispatching on
# it never imports that module during an SSSD run. Pinned equal by
# ``tests/test_external_testset.py::test_system_dispatch_literal_matches_mode``.
EXTERNAL_MODE = "generate_external"

# ``src.chunked_inference.MODE``, duplicated as a literal for the same
# reason; pinned by ``tests/test_chunked_inference.py``.
CHUNKED_MODE = "generate_external_chunked"

# ``src.concat_baseline.MODE``, duplicated as a literal for the same reason;
# pinned by ``tests/test_concat_baseline.py``.
BASELINE_MODE = "generate_concat_baseline"

# ``src.external_anchor.MODE``, duplicated as a literal for the same reason;
# pinned by ``tests/test_external_manifest.py``.
GT_ANCHOR_MODE = "generate_external_gt"

# ``src.external_system_ingest.MODE``, duplicated as a literal for the same
# reason; pinned by ``tests/test_external_system_ingest.py``.
INGEST_MODE = "ingest_external_system"


class ConversationalTTSSystem(BaseSystem):
    """System with ``create_dataset`` (inherited SSSD builder), ``train``, a
    recipe-local ``infer`` (multi-channel generate / gt / resynth), and
    ``measure`` (inherited from ``BaseSystem`` unmodified: it just calls
    ``espnet3.systems.base.metric.measure(self.metrics_config)`` over
    whatever ``infer`` wrote, so no recipe-local override is needed here)."""

    def infer(self, *args, **kwargs):
        """Run the multi-channel infer stage.

        ``mode`` selects the implementation: the SSSD modes (generate / gt /
        resynth) go to ``src/inference.py`` exactly as before, the audio-free
        external test set goes to ``src/external_inference.py``, and its
        chunked variant to ``src/chunked_inference.py``.  The dispatch is
        additive and mode-gated - the SSSD path's behaviour is unchanged for
        every pre-existing config.
        """
        self._reject_stage_args("infer", args, kwargs)
        mode = getattr(self.inference_config, "mode", None)
        logger.info(
            "Inference start | inference_dir=%s mode=%s",
            getattr(self.inference_config, "inference_dir", None),
            mode,
        )
        # Compared as a literal, NOT imported from external_inference: an
        # import here would pull that module (and its dependencies) into
        # every SSSD run too. tests/test_external_testset.py pins the
        # literal against external_inference.MODE so the two cannot drift.
        if mode == EXTERNAL_MODE:
            from egs3.conversational.tts.src.external_inference import (
                run_external_inference,
            )

            return run_external_inference(self.inference_config)

        if mode == CHUNKED_MODE:
            from egs3.conversational.tts.src.chunked_inference import (
                run_chunked_inference,
            )

            return run_chunked_inference(self.inference_config)

        if mode == BASELINE_MODE:
            from egs3.conversational.tts.src.concat_baseline import (
                run_concat_baseline,
            )

            return run_concat_baseline(self.inference_config)

        if mode == GT_ANCHOR_MODE:
            from egs3.conversational.tts.src.external_anchor import run_external_gt

            return run_external_gt(self.inference_config)

        if mode == INGEST_MODE:
            from egs3.conversational.tts.src.external_system_ingest import (
                run_external_system_ingest,
            )

            return run_external_system_ingest(self.inference_config)

        from egs3.conversational.tts.src.inference import run_inference

        return run_inference(self.inference_config)

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
