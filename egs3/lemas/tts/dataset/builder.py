"""LEMAS poc3k dataset builder: extraction, phonemization, groups, manifests.

``prepare_source`` packs the shard tars at 24 kHz (``extract.py``).
``build`` reads the poc3k row lists, seeks each row's transcript and word
alignments in the mirrored jsonl by ``byte_offset``, phonemizes on a process
pool, derives the group id and the speaker-prompt mode per row, holds out
validation groups, and writes the manifests plus ``lang_stats.json``.
Prompt partners are NOT chosen here; the dataset draws them online.
"""

from __future__ import annotations

import functools
import json
import logging
import random
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from importlib import resources
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from dataset.extract import (
    bucket_of,
    chunk_key,
    extract_all,
    read_pack_index,
    regroup_all,
)
from dataset.keys import classify_key, group_id, segment_index
from dataset.manifest import ManifestRow
from src.layout import PACK_SR, text_fits

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)

PoolItem = Tuple[str, str, float, str, str, int, str, List[Tuple[str, float, float]]]


def load_builder_config() -> dict:
    """Return the ``builder`` block of ``dataset/config.yaml``."""
    res = resources.files("dataset").joinpath("config.yaml")
    with resources.as_file(res) as p:
        return load_config_with_defaults(str(p), resolve=False)["builder"]


def decide_spk_mode(group_size: int, dur: float, cfg: dict) -> str:
    """Return the speaker-prompt mode of a row.

    ``group`` if the group has 2+ rows, else ``split`` when the row is long
    enough to split at a word boundary, else ``none`` (spec 3.4).
    """
    if group_size >= 2:
        return "group"
    if dur >= float(cfg["split_min_sec"]):
        return "split"
    return "none"


def split_candidates(
    word_bounds: Sequence[Tuple[float, float]],
    dur: float,
    cfg: dict,
    targets: Optional[Dict[int, Sequence[str]]] = None,
    n_samples: Optional[int] = None,
) -> List[int]:
    """Word indexes ``k`` such that ``words[:k]`` is a legal prompt.

    Args:
        word_bounds: ``(start, end)`` seconds per word.
        dur: Row duration in seconds.
        cfg: Needs ``split_frac`` ``[lo, hi]`` and ``split_min_prompt_sec``.
        targets: Target phones per split point (see :func:`split_points`).
            With ``n_samples``, a split point is kept only when it has a
            target and that target fits the frames it is left with
            (``src.layout.text_fits``).
        n_samples: Row length in samples at 24 kHz.

    Returns:
        Candidate ``k`` values (prompt ends at ``word_bounds[k-1][1]``).

    Example:
        >>> split_candidates([(0, .5), (.6, 1.4), (1.5, 2.2)], 5.0, CFG)
        [2]
    """
    lo, hi = cfg["split_frac"]
    floor = float(cfg["split_min_prompt_sec"])
    ks = []
    for k in range(1, len(word_bounds)):
        end = word_bounds[k - 1][1]
        if end >= floor and lo * dur <= end <= hi * dur:
            ks.append(k)
    if targets is None or n_samples is None:
        return ks
    return [
        k
        for k in ks
        if k in targets
        and text_fits(len(targets[k]), n_samples - int(word_bounds[k][0] * PACK_SR))
    ]


_WORD_PIECE = re.compile(r"[^\W_]+")
_HANZI = re.compile(r"[\u4e00-\u9fff]")
_RESYNC_TXT, _RESYNC_ALIGN = 4, 12  # how far a mismatch may be skipped


def _pieces(text: str) -> List[str]:
    """Lower-cased letter/digit runs: what the aligner's words are made of."""
    return _WORD_PIECE.findall(unicodedata.normalize("NFKC", text).lower())


def split_points(txt: str, words: Sequence[str], lang: str) -> Dict[int, int]:
    """Where the transcript resumes after the first ``k`` aligned words.

    The aligner's words are a normalized transcript: punctuation stripped,
    apostrophes and hyphens split (``I'm`` -> ``I``, ``m``), numbers spelled
    out, and pinyin syllables for zh. A split point ``k`` is kept only when
    aligned word ``k`` starts a transcript word (a character for zh), so
    the target text is a suffix of the transcript and is phonemized exactly
    as inference phonemizes a target. Words that do not match one to one
    (numbers, abbreviations) are skipped up to the next matching word.

    Args:
        txt: Transcript (manifest column 3 is phonemized from it).
        words: Aligned words, in order.
        lang: Language code.

    Returns:
        ``k -> character offset in txt`` for ``1 <= k < len(words)``; zh
        rows whose character and syllable counts differ get ``{}``.

    Example:
        >>> split_points("I'm not sure.", ["I", "m", "not", "sure"], "en")
        {2: 4, 3: 8}
    """
    if lang == "zh":
        offs = [m.start() for m in _HANZI.finditer(txt)]
        if len(offs) != len(words):
            return {}
        return {k: offs[k] for k in range(1, len(words))}
    tokens = [(m.start(), _pieces(m.group())) for m in re.finditer(r"\S+", txt)]
    tokens = [(off, p) for off, p in tokens if p]
    aligned = ["".join(_pieces(w)) for w in words]

    def matches(j: int, a: int) -> bool:
        p = tokens[j][1]
        return aligned[a : a + len(p)] == p

    points: Dict[int, int] = {}
    j = a = 0
    while j < len(tokens) and a < len(aligned):
        if not matches(j, a):
            jump = next(
                (
                    (j + dj, a + da)
                    for dj in range(_RESYNC_TXT + 1)
                    for da in range(_RESYNC_ALIGN + 1)
                    if (dj or da)
                    and j + dj < len(tokens)
                    and a + da < len(aligned)
                    and matches(j + dj, a + da)
                ),
                None,
            )
            if jump is None:
                break
            j, a = jump
        if 0 < a < len(aligned):
            points[a] = tokens[j][0]
        a += len(tokens[j][1])
        j += 1
    return points


def _fmt_bounds(words) -> str:
    return ",".join(f"{s:g}:{e:g}" for _w, s, e in words)


def group_sizes_of(pool: Iterable[PoolItem]) -> Counter:
    """Count rows per ``(lang, group)`` over pool items."""
    sizes: Counter = Counter()
    for key, _a, _d, source, *_rest in pool:
        g = group_id(key, source)
        if g:
            sizes[(key[:2], g)] += 1
    return sizes


def build_rows(
    pool: Sequence[PoolItem],
    cfg: dict,
    phonemizer,
    group_sizes: Optional[Counter] = None,
) -> List[ManifestRow]:
    """Turn pool items into manifest rows.

    Args:
        pool: Items ``(key, audio_rel, dur, source, jsonl_path, byte_offset,
            txt, words)`` with ``words`` as ``(word, start, end)`` tuples.
        cfg: Builder config block.
        phonemizer: Object with ``phonemize(text, lang)``.
        group_sizes: ``(lang, group) -> count`` over the WHOLE language; when
            ``None`` it is computed from ``pool`` (fine for tests, wrong for a
            chunk of a language).

    Returns:
        Rows that pass the duration filter, whose text does not match the
        language's ``drop_text_regex`` (e.g. Latin letters in zh, which the
        pinyin front-end would pass through as raw words), and whose phones
        are non-empty and fit the row's frames. The last rule removes rows
        whose transcript is not what the audio says: 71 of the 29.6 M poc3k
        rows, e.g. a 2.8 s ru row with 485 phones because eSpeak-NG spells
        out every character of a transcript in another script. A split row
        with no split point whose remaining phones fit is kept as ``none``.

    Example:
        >>> rows = build_rows(pool, cfg, LEMASPhonemizer(["de"]), sizes)
    """
    sizes = group_sizes if group_sizes is not None else group_sizes_of(pool)
    rows: List[ManifestRow] = []
    lo, hi = float(cfg["min_target_sec"]), float(cfg["max_target_sec"])
    drop_re = {
        lang: re.compile(pat)
        for lang, pat in (cfg.get("drop_text_regex") or {}).items()
    }
    for key, audio, dur, source, jsonl_path, byte_offset, txt, words in pool:
        lang = key[:2]
        if not (lo <= dur <= hi):
            continue
        if lang in drop_re and drop_re[lang].search(txt):
            continue
        g = group_id(key, source) or ""
        mode = decide_spk_mode(sizes[(lang, g)] if g else 0, dur, cfg)
        n_samples = int(round(dur * PACK_SR))
        wb, st = "", ""
        if mode == "split":
            bounds = [(s, e) for _w, s, e in words]
            offsets = split_points(txt, [w for w, _s, _e in words], lang)
            targets = {}
            for k in split_candidates(bounds, dur, cfg):
                if k in offsets:
                    tail = phonemizer.phonemize(txt[offsets[k] :], lang)
                    # "|" and " " delimit the column; a token holding one
                    # could not be read back
                    if tail and not any("|" in t or " " in t for t in tail):
                        targets[k] = tail
            ks = split_candidates(bounds, dur, cfg, targets, n_samples)
            if not ks:
                mode = "none"
            else:
                wb = _fmt_bounds(words)
                st = "|".join(f"{k}={' '.join(targets[k])}" for k in ks)
        phones = " ".join(phonemizer.phonemize(txt, lang))
        # counted as the dataset counts them: the space-separated column
        if not phones or not text_fits(len(phones.split(" ")), n_samples):
            continue
        rows.append(
            ManifestRow(
                key,
                audio,
                phones,
                lang,
                source,
                g,
                float(dur),
                jsonl_path,
                int(byte_offset),
                mode,
                wb,
                st,
            )
        )
    return rows


def lang_stats(rows: Sequence[ManifestRow]) -> Dict[str, dict]:
    """Median-free rate: total column-3 tokens over total seconds per language."""
    tok: Counter = Counter()
    sec: Counter = Counter()
    for r in rows:
        tok[r.lang] += len(r.phones.split(" "))
        sec[r.lang] += r.dur
    return {lang: {"tokens_per_sec": tok[lang] / sec[lang]} for lang in tok}


# ----------------------------------------------------------------------------
# worker side
# ----------------------------------------------------------------------------
_WORKER_PHON = None
_WORKER_FACTORY: Optional[Callable] = None


def _worker_init(factory):
    from dataset.extract import _tune_worker

    _tune_worker()  # freeze the inherited heap, no transparent huge pages
    global _WORKER_PHON, _WORKER_FACTORY
    _WORKER_FACTORY = factory
    _WORKER_PHON = None


def _get_phon():
    global _WORKER_PHON
    if _WORKER_PHON is None:
        if _WORKER_FACTORY is not None:
            _WORKER_PHON = _WORKER_FACTORY()
        else:
            from src.text.lemas_phonemizer import LEMASPhonemizer

            _WORKER_PHON = LEMASPhonemizer()
    return _WORKER_PHON


def _read_jsonl_rows(jsonl_abs: Path, offsets: Sequence[int]) -> List[dict]:
    """Read one json object per byte offset (offsets sorted ascending)."""
    out = []
    with jsonl_abs.open("rb") as f:
        for off in offsets:
            f.seek(off)
            out.append(json.loads(f.readline()))
    return out


def _chunk_job(args):
    """Phonemize one chunk of rows of one shard; returns manifest lines."""
    chunk, jsonl_abs, jsonl_rel, cfg, sizes = args
    chunk = sorted(chunk, key=lambda r: r[4])  # by byte_offset
    objs = _read_jsonl_rows(Path(jsonl_abs), [r[4] for r in chunk])
    pool: List[PoolItem] = []
    for (key, audio, dur, source, off), obj in zip(chunk, objs):
        if obj.get("key") != key:
            raise RuntimeError(f"byte_offset mismatch for {key}: got {obj.get('key')}")
        # ms precision so that the split-candidate check made here and the one
        # the dataset repeats from the manifest string see identical numbers
        words = [
            (w["word"], round(float(w["start"]), 3), round(float(w["end"]), 3))
            for w in (obj.get("align") or {}).get("words", [])
            if "start" in w and "end" in w
        ]
        pool.append(
            (
                key,
                audio,
                float(dur),
                source,
                jsonl_rel,
                int(off),
                obj.get("txt", ""),
                words,
            )
        )
    rows = build_rows(pool, cfg, _get_phon(), Counter(sizes))
    return [
        (
            r.utt_id,
            r.lang,
            r.group,
            r.spk_mode,
            len(r.phones.split(" ")),
            r.dur,
            r.to_line(),
        )
        for r in rows
    ]


class LEMASBuilder(DatasetBuilder):
    """Prepare LEMAS poc3k audio and manifests for the dual-prompt recipe."""

    def __init__(self, cfg: Optional[dict] = None):
        """Take an explicit builder config, or read ``dataset/config.yaml``."""
        self.cfg = dict(cfg) if cfg is not None else load_builder_config()

    # ---- paths ------------------------------------------------------------
    def _mirror(self) -> Path:
        return Path(self.cfg["mirror_root"])

    def _manifest_tsvs(self, lang: str) -> List[Path]:
        return sorted((self._mirror() / self.cfg["manifest_dir"] / lang).glob("*.tsv"))

    def _data_dir(self, recipe_dir, data_path=None) -> Path:
        return Path(recipe_dir).resolve() / (data_path or self.cfg["data_path"])

    # ---- source -------------------------------------------------------------
    def is_source_prepared(self, recipe_dir=None, **_kwargs) -> bool:
        """Return True when every language has its chunk-contiguous pack."""
        audio_root = Path(self.cfg["audio_root"])
        for lang in self.cfg["langs"]:
            if not self._manifest_tsvs(lang):
                return False
            if not (audio_root / lang / f"{lang}.regrouped").is_file():
                return False
        return True

    def _shard_rows(self, lang: str):
        """Per shard: ``(pack, index, member -> (bucket, chunk))`` for the regroup."""
        chunk_rows = int(self.cfg.get("chunk_rows", 32))
        n_buckets = int(self.cfg.get("n_buckets", 128))
        audio_root = Path(self.cfg["audio_root"])
        per_shard: List[Tuple[str, str, str, Optional[int], str]] = []
        sizes: Counter = Counter()
        for tsv in self._manifest_tsvs(lang):
            with tsv.open(encoding="utf-8") as f:
                for line in f:
                    key, member, _dur, source, _off = line.rstrip("\n").split("\t")
                    if source == "unknown":
                        source = classify_key(key)
                    g = group_id(key, source) or ""
                    per_shard.append(
                        (tsv.stem, key, member, segment_index(key, source), g)
                    )
                    if g:
                        sizes[g] += 1
        shards = []
        for tsv in self._manifest_tsvs(lang):
            rows = {}
            for shard, key, member, seg, g in per_shard:
                if shard != tsv.stem:
                    continue
                ck = chunk_key(g, key, seg, sizes[g] if g else 1, chunk_rows)
                rows[member] = (bucket_of(ck, n_buckets), ck)
            shards.append(
                (
                    str(audio_root / lang / f"{tsv.stem}.pcm"),
                    str(audio_root / lang / f"{tsv.stem}.index.tsv"),
                    rows,
                )
            )
        return shards

    def prepare_source(self, recipe_dir=None, **_kwargs) -> None:
        """Pass 1: shard packs from the tars; pass 2: chunk-contiguous packs."""
        extract_all(
            self._mirror(),
            self.cfg["manifest_dir"],
            self.cfg["langs"],
            self.cfg["audio_root"],
            int(self.cfg["n_workers"]),
            int(self.cfg["sample_rate"]),
        )
        regroup_all(
            {lang: self._shard_rows(lang) for lang in self.cfg["langs"]},
            self.cfg["audio_root"],
            seed=int(self.cfg["seed"]),
            chunk_rows=int(self.cfg.get("chunk_rows", 32)),
            n_buckets=int(self.cfg.get("n_buckets", 128)),
            n_workers=int(self.cfg["n_workers"]),
        )

    # ---- build --------------------------------------------------------------
    def is_built(self, recipe_dir=None, data_path=None, **_kwargs) -> bool:
        """Return True when both manifests and ``lang_stats.json`` exist.

        Args:
            recipe_dir: Recipe root.
            data_path: Data tree under it (default ``cfg["data_path"]``); the
                character build of the stock F5-TTS baseline uses its own.
        """
        d = self._data_dir(recipe_dir, data_path)
        return all(
            (d / p).is_file()
            for p in list(self.cfg["manifest_paths"].values())
            + [self.cfg["lang_stats_path"]]
        )

    def _read_pool(
        self, lang: str
    ) -> Dict[str, List[Tuple[str, str, float, str, int]]]:
        """Return shard -> rows ``(key, audio_spec, dur, source, byte_offset)``.

        ``audio_spec`` is ``<lang>/<lang>.pcm:<start>:<n>`` (samples at 24 kHz)
        from the language's chunk-contiguous pack index, and ``dur`` is
        ``n / sample_rate``: the packed length, not the jsonl duration (which
        overstates the audio by up to 64 samples).
        """
        by_shard: Dict[str, list] = defaultdict(list)
        audio_root = Path(self.cfg["audio_root"])
        sr = int(self.cfg["sample_rate"])
        index_path = audio_root / lang / f"{lang}.index.tsv"
        index = read_pack_index(index_path)
        pack_rel = f"{lang}/{lang}.pcm"
        for tsv in self._manifest_tsvs(lang):
            with tsv.open(encoding="utf-8") as f:
                for line in f:
                    key, audio, _dur, source, off = line.rstrip("\n").split("\t")
                    if source == "unknown":
                        source = classify_key(key)
                    if audio not in index:
                        raise RuntimeError(f"{audio} is not in {index_path}")
                    start, n = index[audio]
                    by_shard[tsv.stem].append(
                        (key, f"{pack_rel}:{start}:{n}", n / sr, source, int(off))
                    )
        return by_shard

    def _choose_valid(self, lang: str, by_shard, sizes: Counter, rng: random.Random):
        """Hold out whole groups (and single rows without a group)."""
        target = int(self.cfg["valid_rows_per_lang"])
        groups = sorted(g for (lg, g), _n in sizes.items() if lg == lang)
        rng.shuffle(groups)
        valid_groups, n = set(), 0
        for g in groups:
            if n >= target:
                break
            valid_groups.add(g)
            n += sizes[(lang, g)]
        valid_keys = set()
        if n < target:
            loose = [
                r[0]
                for rows in by_shard.values()
                for r in rows
                if not group_id(r[0], r[3])
            ]
            rng.shuffle(loose)
            valid_keys.update(loose[: target - n])
        return valid_groups, valid_keys

    def build(
        self,
        recipe_dir=None,
        phonemizer_factory: Optional[Callable] = None,
        text_units: str = "phones",
        data_path: Optional[str] = None,
        **_kwargs,
    ) -> None:
        """Write ``manifest/train.tsv``, ``manifest/valid.tsv`` and ``lang_stats.json``.

        Args:
            recipe_dir: Recipe root; outputs go under ``<recipe_dir>/<data_path>``.
            phonemizer_factory: Zero-arg callable returning the text front-end;
                used by tests. Defaults to the front-end of ``text_units``.
            text_units: ``phones`` (eSpeak IPA and pinyin initials/finals, the
                recipe's front-end) or ``chars`` (stock F5-TTS: characters and
                pinyin syllables, for the baseline row S0a).
            data_path: Data tree under ``recipe_dir`` (default
                ``cfg["data_path"]``), so a second front-end gets its own
                manifests, token list and ``lang_stats.json``.

        Raises:
            ValueError: If ``text_units`` is unknown.
        """
        from src.text.char_tokenizer import TEXT_UNITS, text_frontend

        if text_units not in TEXT_UNITS:
            raise ValueError(f"text_units must be one of {TEXT_UNITS}: {text_units!r}")
        if phonemizer_factory is None:
            phonemizer_factory = functools.partial(text_frontend, text_units)
        cfg = self.cfg
        data_dir = self._data_dir(recipe_dir, data_path)
        train_path = data_dir / cfg["manifest_paths"]["train"]
        valid_path = data_dir / cfg["manifest_paths"]["valid"]
        train_path.parent.mkdir(parents=True, exist_ok=True)
        rng = random.Random(int(cfg["seed"]))
        n_workers = int(cfg["n_workers"])
        chunk_size = int(cfg.get("chunk_size", 20000))
        tok: Counter = Counter()
        sec: Counter = Counter()
        mode_counts: Dict[str, Counter] = defaultdict(Counter)
        counts: Dict[str, Dict[str, int]] = {}
        jsonl_root = self._mirror() / "LEMAS-train" / "train"
        with (
            train_path.open("w", encoding="utf-8") as ftrain,
            valid_path.open("w", encoding="utf-8") as fvalid,
        ):
            for lang in cfg["langs"]:
                by_shard = self._read_pool(lang)
                if not by_shard:
                    logger.warning("build: no rows for %s", lang)
                    continue
                sizes = group_sizes_of(
                    (r[0], r[1], r[2], r[3], "", r[4], "", [])
                    for rows in by_shard.values()
                    for r in rows
                )
                valid_groups, valid_keys = self._choose_valid(
                    lang, by_shard, sizes, rng
                )
                jobs = []
                for shard, rows in by_shard.items():
                    jsonl_rel = f"{lang}/{shard}.jsonl"
                    rows = sorted(rows, key=lambda r: r[4])
                    for i in range(0, len(rows), chunk_size):
                        chunk = rows[i : i + chunk_size]
                        needed = {(lang, group_id(r[0], r[3])) for r in chunk}
                        sub = {k: sizes[k] for k in needed if k[1]}
                        jobs.append(
                            (
                                chunk,
                                str(jsonl_root / jsonl_rel),
                                jsonl_rel,
                                dict(cfg),
                                sub,
                            )
                        )
                if n_workers <= 1:
                    _worker_init(phonemizer_factory)
                    results = map(_chunk_job, jobs)
                else:
                    pool = ProcessPoolExecutor(
                        max_workers=n_workers,
                        initializer=_worker_init,
                        initargs=(phonemizer_factory,),
                    )
                    results = pool.map(_chunk_job, jobs, chunksize=1)
                n_pool = sum(len(rows) for rows in by_shard.values())
                n_train = n_valid = 0
                for lines in results:
                    for utt, lg, g, mode, n_tok, dur, line in lines:
                        is_valid = (g and g in valid_groups) or (
                            not g and utt in valid_keys
                        )
                        if is_valid:
                            fvalid.write(line + "\n")
                            n_valid += 1
                        else:
                            ftrain.write(line + "\n")
                            n_train += 1
                            tok[lg] += n_tok
                            sec[lg] += dur
                        mode_counts[lg][mode] += 1
                if n_workers > 1:
                    pool.shutdown()
                counts[lang] = {
                    "n_pool": n_pool,
                    "n_train": n_train,
                    "n_valid": n_valid,
                    "n_dropped": n_pool - n_train - n_valid,
                }
                logger.info(
                    "build %s: pool %d train %d valid %d dropped %d modes %s",
                    lang,
                    n_pool,
                    n_train,
                    n_valid,
                    n_pool - n_train - n_valid,
                    dict(mode_counts[lang]),
                )
        stats = {
            lang: {"tokens_per_sec": tok[lang] / sec[lang], **counts[lang]}
            for lang in tok
            if sec[lang] > 0
        }
        (data_dir / cfg["lang_stats_path"]).write_text(json.dumps(stats, indent=1))
        (data_dir / "spk_mode_counts.json").write_text(
            json.dumps({k: dict(v) for k, v in mode_counts.items()}, indent=1)
        )
