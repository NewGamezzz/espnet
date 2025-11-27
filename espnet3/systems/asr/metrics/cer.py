"""
Character error rate metric utilities.

Provides a callable :class:`CER` metric that normalizes text, computes CER, and
writes alignment visualizations for later inspection.

Example:
    >>> from pathlib import Path
    >>> metric = CER()
    >>> scores = metric({"ref": ["a b"], "hyp": ["a c"]}, "test", Path("decode"))  # doctest: +SKIP
    >>> scores["CER"]  # doctest: +SKIP
    50.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import jiwer

from espnet2.text.cleaner import TextCleaner
from espnet3.components.abs_metric import AbsMetrics


class CER(AbsMetrics):
    """
    Compute character error rate (CER) for decoded hypotheses.

    Args:
        ref_key (str): Dictionary key that stores reference strings.
        hyp_key (str): Dictionary key that stores hypothesis strings.
        clean_types (Iterable[str] | None): Text normalizers passed to
            :class:`espnet2.text.cleaner.TextCleaner`.

    Example:
        >>> from pathlib import Path
        >>> metric = CER(ref_key="ref", hyp_key="hyp")
        >>> data = {"ref": ["hello"], "hyp": ["hullo"]}
        >>> metric(data, "demo", Path("decode"))  # doctest: +SKIP
        {'CER': 20.0}
    """

    def __init__(
        self,
        ref_key: str = "ref",
        hyp_key: str = "hyp",
        clean_types: Iterable[str] | None = None,
    ) -> None:
        self.cleaner = TextCleaner(clean_types)
        self.ref_key = ref_key
        self.hyp_key = hyp_key

    def _clean(self, text: str) -> str:
        cleaned = self.cleaner(text).strip()
        return cleaned if cleaned else "."

    def __call__(
        self,
        data: Dict[str, List[str]],
        test_name: str,
        decode_dir: Path,
    ) -> Dict[str, float]:
        """
        Calculate CER for a test split and write alignment details to disk.

        Args:
            data (Dict[str, List[str]]): Mapping containing lists of references
                and hypotheses keyed by ``ref_key``/``hyp_key``.
            test_name (str): Name of the evaluation split (used for output dir).
            decode_dir (Path): Directory where alignment files are written.

        Returns:
            Dict[str, float]: Dictionary with a single ``"CER"`` percentage.

        Example:
            >>> metric = CER()
            >>> metric({"ref": ["abc"], "hyp": ["axc"]}, "set1", Path("decode"))  # doctest: +SKIP
            {'CER': 33.33}
        """
        refs = [self._clean(x) for x in data[self.ref_key]]
        hyps = [self._clean(x) for x in data[self.hyp_key]]

        score = jiwer.cer(refs, hyps) * 100
        details = jiwer.process_characters(refs, hyps)

        test_dir = Path(decode_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        with (test_dir / "cer_alignment").open("w", encoding="utf-8") as f:
            f.write(jiwer.visualize_alignment(details))

        return {"CER": round(score, 2)}
