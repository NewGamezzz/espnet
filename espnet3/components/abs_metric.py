from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List


class AbsMetrics(ABC):
    """
    Abstract base class for metrics used in inference evaluation.

    Subclasses must implement ``__call__`` to compute metric values from decoded
    hypotheses and references.

    Example:
        >>> class DummyMetric(AbsMetrics):
        ...     def __call__(self, data, test_name, decode_dir):
        ...         return {\"acc\": 1.0}
    """

    @abstractmethod
    def __call__(
        self, data: Dict[str, List[str]], test_name: str, decode_dir: Path
    ) -> Dict[str, float]:
        """
        Compute and return metric values.

        Args:
            data (Dict[str, List[str]]): Mapping of field name to list of strings
                (e.g., hypotheses, references) for a single test split.
            test_name (str): Name of the test dataset (e.g., ``\"test-other\"``).
            decode_dir (Path): Directory containing decode outputs; useful when
                metrics need to read auxiliary files.

        Returns:
            Dict[str, float]: Computed metric values keyed by metric name.

        Note:
            The return value must be JSON-serializable because it will be written
            into ``scores.json`` by :func:`espnet3.systems.base.score`.
        """
        raise NotImplementedError
