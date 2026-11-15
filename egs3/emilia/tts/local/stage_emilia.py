"""Stage one Emilia language from Hugging Face into the ``speaker_tree`` layout.

The dataset repo holds flat tar archives (``Emilia/<LANG>/<LANG>-Bxxxxxx.tar``)
of ``<utt_id>.json`` + ``<utt_id>.mp3`` members. This script downloads them
and writes the tree ``dataset/layout.py`` calls ``speaker_tree``::

    <dest>/emilia/<LANG>/<LANG>_Bxxxxx/<speaker>/mp3/<utt_id>.mp3
    <dest>/emilia/<LANG>/<LANG>_Bxxxxx.jsonl

so a language can be staged privately next to a shared copy of the others
(link the shared ``emilia/<OTHER>`` into ``<dest>/emilia/``).

The run is resumable: an archive is finished when its record file exists under
``<dest>/.staging/parts/``, and that file is written last. The batch jsonl
files, which are what the manifest builder trusts, are only written once every
archive of the language is finished.

The dataset is gated: log in first (``hf auth login``) with an account that
accepted its terms.

Example:
    python local/stage_emilia.py --lang ZH --dest /path/to/corpus --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

from egs3.emilia.tts.dataset.layout import speaker_tree_audio_dir

logger = logging.getLogger(__name__)

REPO_ID = "amphion/Emilia-Dataset"
_UTT_ID = re.compile(r"^[A-Z]{2}_B\d{5}_S\d{5}_W\d{6}$")


def _parts_dir(dest: Path, lang: str) -> Path:
    return Path(dest) / ".staging" / "parts" / lang


def _write_atomic(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.writelines(lines)
    os.replace(tmp, path)


def extract_archive(
    tar_path: str | Path, emilia_root: str | Path, lang: str, part_path: str | Path
) -> dict[str, int]:
    """Extract one archive into the speaker tree and record what it held.

    Only regular members named ``<utt_id>.mp3`` or ``<utt_id>.json`` with an
    utterance id of ``lang`` are used; the target path is derived from the
    validated id, never from the member name. ``part_path`` receives the
    archive's records, one JSON object per line sorted by id, and is written
    last so that its presence marks the archive as finished.

    Args:
        tar_path: Downloaded archive.
        emilia_root: The ``<dest>/emilia`` directory.
        lang: Language of the archive, e.g. ``"ZH"``.
        part_path: Where to write the archive's records.

    Returns:
        ``{"audio": n, "records": n, "skipped": n}``.
    """
    emilia_root = Path(emilia_root)
    counts = {"audio": 0, "records": 0, "skipped": 0}
    records: dict[str, str] = {}
    made: set[str] = set()
    with tarfile.open(tar_path, "r:") as tf:
        for member in tf:
            utt_id, dot, ext = member.name.rpartition(".")
            if (
                not member.isfile()
                or ext not in ("mp3", "json")
                or not _UTT_ID.match(utt_id)
                or not utt_id.startswith(lang + "_")
            ):
                counts["skipped"] += 1
                continue
            source = tf.extractfile(member)
            if ext == "json":
                try:
                    record = json.loads(source.read().decode("utf-8"))
                    if record["id"] != utt_id:
                        raise ValueError("record id does not match its file name")
                except (ValueError, KeyError, TypeError):
                    counts["skipped"] += 1
                    continue
                records[utt_id] = json.dumps(record, ensure_ascii=False) + "\n"
                counts["records"] += 1
                continue
            audio_dir = speaker_tree_audio_dir(utt_id)
            if audio_dir not in made:
                (emilia_root / audio_dir).mkdir(parents=True, exist_ok=True)
                made.add(audio_dir)
            with open(emilia_root / audio_dir / f"{utt_id}.mp3", "wb") as out:
                shutil.copyfileobj(source, out, 1 << 20)
            counts["audio"] += 1
    _write_atomic(Path(part_path), (records[k] for k in sorted(records)))
    return counts


def stage(
    lang: str,
    dest: str | Path,
    names: list[str],
    fetch: Callable[[str], Path],
    workers: int = 1,
) -> dict[str, bool]:
    """Download and extract the archives that are not finished yet.

    A failing archive is logged and left unfinished; the others go on, and a
    rerun picks the failed ones up again.

    Args:
        lang: Language, e.g. ``"ZH"``.
        dest: Corpus root; audio goes under ``<dest>/emilia/<lang>/``.
        names: Archive names without ``.tar``, e.g. ``"ZH-B000000"``.
        fetch: Downloads the named archive and returns its local path. The
            file is deleted once it has been extracted.
        workers: Archives handled at a time.

    Returns:
        ``{name: finished}`` for every name.
    """
    dest = Path(dest)
    parts = _parts_dir(dest, lang)
    todo = [n for n in names if not (parts / f"{n}.jsonl").is_file()]
    logger.info(
        "stage: %d of %d %s archives already finished",
        len(names) - len(todo),
        len(names),
        lang,
    )
    started = time.time()
    n_done = 0

    def one(name: str) -> bool:
        nonlocal n_done
        try:
            tar_path = Path(fetch(name))
            try:
                counts = extract_archive(
                    tar_path, dest / "emilia", lang, parts / f"{name}.jsonl"
                )
            finally:
                tar_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("stage: %s failed, left for a rerun", name)
            return False
        n_done += 1
        logger.info(
            "stage: %s %s | %d/%d this run | %.2f h elapsed",
            name,
            counts,
            n_done,
            len(todo),
            (time.time() - started) / 3600.0,
        )
        return True

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        results = dict(zip(todo, pool.map(one, todo)))
    return {name: results.get(name, True) for name in names}


def merge_parts(lang: str, dest: str | Path, names: list[str]) -> dict[str, int]:
    """Write the per-batch jsonl files from the finished archives' records.

    Args:
        lang: Language, e.g. ``"ZH"``.
        dest: Corpus root given to :func:`stage`.
        names: Every archive name of the language.

    Returns:
        ``{batch: rows}``, e.g. ``{"ZH_B00000": 336057}``.

    Raises:
        RuntimeError: If an archive is not finished; nothing is written then.
    """
    dest = Path(dest)
    parts = _parts_dir(dest, lang)
    missing = [n for n in names if not (parts / f"{n}.jsonl").is_file()]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(names)} {lang} archives are not staged yet "
            f"(first: {missing[0]}); rerun the staging before merging"
        )
    by_batch: dict[str, list[str]] = {}
    for name in names:
        with open(parts / f"{name}.jsonl", "r", encoding="utf-8") as fh:
            for line in fh:
                # '"id": "ZH_B00000_S00001_W000000"' -> batch "ZH_B00000"
                utt_id = json.loads(line)["id"]
                by_batch.setdefault(utt_id.rsplit("_", 2)[0], []).append((utt_id, line))
    rows = {}
    for batch in sorted(by_batch):
        lines = sorted(by_batch[batch])
        _write_atomic(
            dest / "emilia" / lang / f"{batch}.jsonl", (line for _, line in lines)
        )
        rows[batch] = len(lines)
    return rows


def archive_names(lang: str) -> list[str]:
    """List the language's archive names in the dataset repo, sorted."""
    from huggingface_hub import HfApi

    tree = HfApi().list_repo_tree(REPO_ID, f"Emilia/{lang}", repo_type="dataset")
    return sorted(
        Path(entry.path).stem for entry in tree if entry.path.endswith(".tar")
    )


def main(argv=None) -> int:
    """Stage a language; exit 0 when complete, 3 when archives remain."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lang", required=True)
    parser.add_argument("--dest", required=True, help="corpus root to write into")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="archives this run")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from huggingface_hub import hf_hub_download

    dest = Path(args.dest)
    tars = dest / ".staging" / "tars"

    def fetch(name: str) -> Path:
        return Path(
            hf_hub_download(
                REPO_ID,
                f"Emilia/{args.lang}/{name}.tar",
                repo_type="dataset",
                local_dir=tars,
            )
        )

    names = archive_names(args.lang)
    parts = _parts_dir(dest, args.lang)
    todo = [n for n in names if not (parts / f"{n}.jsonl").is_file()]
    this_run = todo if args.limit is None else todo[: args.limit]
    finished = [n for n in names if n not in todo]
    stage(args.lang, dest, finished + this_run, fetch, workers=args.workers)

    left = [n for n in names if not (parts / f"{n}.jsonl").is_file()]
    if left:
        print(f"STAGE_INCOMPLETE {len(left)} of {len(names)} archives left")
        return 3
    rows = merge_parts(args.lang, dest, names)
    print(f"STAGE_COMPLETE {len(rows)} batches, {sum(rows.values())} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
