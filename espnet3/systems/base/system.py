# system_base.py

import logging
from pathlib import Path

from omegaconf import DictConfig

from espnet3.systems.base.inference import inference
from espnet3.systems.base.score import score
from espnet3.systems.base.train import collect_stats, train

logger = logging.getLogger(__name__)


class BaseSystem:
    """Base class for all ESPnet3 systems.

    Each system should implement the following:
      - create_dataset()
      - train()
      - decode()
      - score()
      - publish()

    This class intentionally does NOT implement:
      - DAG
      - dependency checks
      - caching

    All behavior is config-driven.
    """

    def __init__(
        self, train_config: DictConfig, eval_config: DictConfig = None
    ) -> None:
        self.train_config = train_config
        self.eval_config = eval_config
        self.exp_dir = Path(train_config.exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Stage stubs (override in subclasses if needed)
    # ---------------------------------------------------------
    def create_dataset(self):
        """
        Prepare datasets for the current recipe.

        Returns:
            None: Base implementation is a stub and only logs.

        Example:
            >>> system = BaseSystem(train_config, eval_config)  # doctest: +SKIP
            >>> system.create_dataset()  # doctest: +SKIP
        """
        logger.info("Running prepare() (BaseSystem stub). Nothing done.")

    def collect_stats(self):
        """
        Run statistics collection for the training/validation splits.

        Returns:
            Any: Output of :func:`espnet3.systems.base.train.collect_stats`.

        Example:
            >>> system = BaseSystem(train_config, eval_config)  # doctest: +SKIP
            >>> system.collect_stats()  # doctest: +SKIP
        """
        return collect_stats(self.train_config)

    def train(self):
        """
        Launch model training.

        Returns:
            Any: Output of :func:`espnet3.systems.base.train.train`, typically
            the Lightning trainer fit result.

        Example:
            >>> system = BaseSystem(train_config, eval_config)  # doctest: +SKIP
            >>> system.train()  # doctest: +SKIP
        """
        return train(self.train_config)

    def evaluate(self):
        """
        Decode and score using the evaluation config.

        Returns:
            Any: Metrics produced by :meth:`score`.

        Example:
            >>> system = BaseSystem(train_config, eval_config)  # doctest: +SKIP
            >>> system.evaluate()  # doctest: +SKIP
        """
        self.decode()
        return self.score()

    def decode(self):
        """
        Run model inference on the evaluation datasets.

        Returns:
            Any: Result of :func:`espnet3.systems.base.inference.inference`.

        Example:
            >>> system = BaseSystem(train_config, eval_config)  # doctest: +SKIP
            >>> system.decode()  # doctest: +SKIP
        """
        return inference(self.eval_config)

    def score(self):
        """
        Compute metrics for decoded outputs.

        Returns:
            Any: Metrics dictionary produced by
            :func:`espnet3.systems.base.score.score`.

        Example:
            >>> system = BaseSystem(train_config, eval_config)  # doctest: +SKIP
            >>> scores = system.score()  # doctest: +SKIP
        """
        result = score(self.eval_config)
        logger.info("Scoring results: %s", result)
        return result

    def publish(self):
        """
        Placeholder for publishing artifacts.

        Returns:
            None: Base implementation only logs and performs no action.

        Example:
            >>> system = BaseSystem(train_config, eval_config)  # doctest: +SKIP
            >>> system.publish()  # doctest: +SKIP
        """
        logger.info("Running publish() (BaseSystem stub). Nothing done.")
