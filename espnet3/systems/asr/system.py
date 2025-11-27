# system_asr.py

import logging
import os
from importlib import import_module
from pathlib import Path

from espnet3.systems.asr.tokenizer.utils import (
    gather_training_text,
    train_sentencepiece,
)
from espnet3.systems.base.system import BaseSystem

logger = logging.getLogger(__name__)


def load_function(path):
    """
    Import a callable from its dotted Python path.

    Args:
        path (str): Dotted path like ``"package.module.function"``.

    Returns:
        Callable: The referenced attribute, typically a function.

    Example:
        >>> sqrt = load_function("math.sqrt")
        >>> sqrt(9)
        3.0
    """
    module_path, func_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, func_name)


class ASRSystem(BaseSystem):
    """
    ASR-specific system that optionally trains a tokenizer before training.

    This class wires together dataset preparation, tokenizer training, and
    the base training/inference hooks defined in :class:`BaseSystem`.

    Args:
        train_config (DictConfig): Training configuration consumed by the
            underlying ESPnet components.
        eval_config (DictConfig | None): Optional evaluation configuration.

    Example:
        >>> system = ASRSystem(train_config, eval_config)  # doctest: +SKIP
        >>> system.train(dataset_dir="/data/asr")  # doctest: +SKIP
        >>> scores = system.evaluate()  # doctest: +SKIP
    """

    def create_dataset(self, func: str, **kwargs):
        """
        Run a user-provided dataset creation function.

        Args:
            func (str): Dotted path to a callable that prepares datasets.
            **kwargs: Keyword arguments forwarded to the callable.

        Returns:
            Any: Whatever the dataset creation function returns.

        Example:
            >>> system = ASRSystem(train_config, eval_config)  # doctest: +SKIP
            >>> system.create_dataset("my.module.prepare", root="/data")  # doctest: +SKIP
        """
        logger.info("ASRSystem.create_dataset(): starting dataset creation process")
        logger.info(f"Creating dataset with function {func}")
        fn = load_function(func)
        return fn(**kwargs)

    def train(self, dataset_dir: str = None):
        """
        Train tokenizer if necessary, then delegate to the base trainer.

        Args:
            dataset_dir (str | None): Root path of the training corpus. Required
                when the tokenizer model has not been generated yet.

        Returns:
            Any: The result of :meth:`BaseSystem.train`, typically the Lightning
            trainer's fit return value.

        Example:
            >>> system = ASRSystem(train_config, eval_config)  # doctest: +SKIP
            >>> system.train(dataset_dir="/data/LibriSpeech")  # doctest: +SKIP
        """
        logger.info("ASRSystem.train(): starting training process")

        # Train tokenizer if not trained previously
        tokenizer_path = (
            Path(self.train_config.tokenizer.save_path)
            / f"{self.train_config.tokenizer.model_type}.model"
        )
        if not tokenizer_path.exists():
            self.train_tokenizer(dataset_dir=dataset_dir)

        # Proceed with standard training
        return super().train()

    def train_tokenizer(self, dataset_dir: str = None):
        """
        Train a sentencepiece tokenizer from the LibriSpeech-style dataset.

        Args:
            dataset_dir (str): Root directory containing the training split.

        Returns:
            None

        Note:
            The training text is gathered from the ``train-clean-100`` subset and
            saved under ``train.txt`` in ``tokenizer.save_path``.

        Example:
            >>> system = ASRSystem(train_config, eval_config)  # doctest: +SKIP
            >>> system.train_tokenizer(dataset_dir="/data/LibriSpeech")  # doctest: +SKIP
        """
        assert (
            dataset_dir is not None
        ), "dataset_dir must be provided for tokenizer training"

        split_path = os.path.join(dataset_dir, "train-clean-100")
        output_path = Path(self.train_config.tokenizer.save_path)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Gathering examples from: {split_path}")

        texts = gather_training_text(split_path)
        with open(output_path / "train.txt", "w") as f:
            f.write("\n".join(texts))

        logger.info(f"Training tokenizer: {self.train_config.tokenizer.model_type}")
        logger.info(f"Tokenizer output: {self.train_config.tokenizer.save_path}")

        # Example placeholder:
        train_sentencepiece(
            output_path / "train.txt",
            output_path,
            self.train_config.tokenizer.vocab_size,
            model_type=self.train_config.tokenizer.model_type,
        )
        logger.info("Tokenizer training completed.")
