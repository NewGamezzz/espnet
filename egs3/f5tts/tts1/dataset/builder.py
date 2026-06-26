"""
ESPnet3 DatasetBuilder for F5-TTS.

Responsibilities:
  prepare_source(): validate that raw corpora exist
  build():          run prepare_emilia.py and prepare_eval.py to produce CSVs
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from espnet3.components.data.dataset_builder import DatasetBuilder


class F5TTSBuilder(DatasetBuilder):

    # ------------------------------------------------------------------
    # Source checks
    # ------------------------------------------------------------------
    def is_source_prepared(
        self,
        recipe_dir: str,
        emilia_dir: str,
        seed_tts_dir: str,
        librispeech_dir: str,
        **kwargs,
    ) -> bool:
        return (
            (Path(emilia_dir) / "EN").is_dir()
            and (Path(emilia_dir) / "ZH").is_dir()
            and (Path(seed_tts_dir) / "seedtts_testset").is_dir()
            and (Path(librispeech_dir) / "test-clean").is_dir()
        )

    def prepare_source(
        self,
        recipe_dir: str,
        emilia_dir: str,
        seed_tts_dir: str,
        librispeech_dir: str,
        **kwargs,
    ) -> None:
        missing = []
        if not (Path(emilia_dir) / "EN").is_dir():
            missing.append(str(Path(emilia_dir) / "EN"))
        if not (Path(emilia_dir) / "ZH").is_dir():
            missing.append(str(Path(emilia_dir) / "ZH"))
        if not (Path(seed_tts_dir) / "seedtts_testset").is_dir():
            missing.append(str(Path(seed_tts_dir) / "seedtts_testset"))
        if not (Path(librispeech_dir) / "test-clean").is_dir():
            missing.append(str(Path(librispeech_dir) / "test-clean"))

        raise RuntimeError(
            "Raw dataset sources are missing. "
            "Run data_load_prep/data_download.sh first.\n"
            "Missing:\n" + "\n".join(f"  {p}" for p in missing)
        )

    # ------------------------------------------------------------------
    # Build checks
    # ------------------------------------------------------------------
    def is_built(
        self,
        recipe_dir: str,
        dataset_dir: str,
        **kwargs,
    ) -> bool:
        d = Path(dataset_dir)
        return (
            (d / "EN" / "train.csv").exists()
            and (d / "EN" / "val.csv").exists()
            and (d / "EN" / "duration.json").exists()
            and (d / "ZH" / "train.csv").exists()
            and (d / "ZH" / "val.csv").exists()
            and (d / "ZH" / "duration.json").exists()
            and (d / "eval_en" / "meta.csv").exists()
            and (d / "eval_zh" / "meta.csv").exists()
            and (d / "librispeech_test_clean_eval" / "meta.csv").exists()
            and (d / "token_list" / "tokens.txt").exists()
        )

    def build(
        self,
        recipe_dir: str,
        dataset_dir: str,
        emilia_dir: str,
        seed_tts_dir: str,
        librispeech_dir: str,
        nj: int = 8,
        val_ratio: float = 0.01,
        **kwargs,
    ) -> None:
        scripts = Path(recipe_dir) / "data_load_prep"

        subprocess.run(
            [
                sys.executable, str(scripts / "prepare_emilia.py"),
                "--nj",        str(nj),
                "--val_ratio", str(val_ratio),
                "--langs",     "EN", "ZH",
                emilia_dir,
                dataset_dir,
            ],
            check=True,
        )

        subprocess.run(
            [
                sys.executable, str(scripts / "prepare_eval.py"),
                "--eval_set", "eval",
                seed_tts_dir,
                librispeech_dir,
                dataset_dir,
            ],
            check=True,
        )