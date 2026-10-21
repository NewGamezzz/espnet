r"""Check the batches the training dataloader hands to the model.

Builds ``LEMASLightningModule`` from the training config exactly as the
``train`` stage does and iterates the loaders its ``train_dataloader`` /
``val_dataloader`` return (batch sampler, worker processes, per-epoch
reseeding, collate). Only ``current_epoch`` is overridden. Every row of
every inspected batch is checked for:

- structure: lengths within the padded tensors, zero padding, one role
  token per prompt frame then the language tag, text no longer than the
  mel the model extracts, a non-empty target span;
- consistency: the row equals ``dataset[index]`` drawn at the same epoch,
  and its target phones equal the manifest's (column 3, or the split
  target of the drawn split point);
- content: Whisper large-v3 on the target region against the row's
  transcript (the transcript suffix for split rows), and speaker
  similarity of the speaker prompt and of the language prompt to the
  target.

Usage (recipe dir, one GPU):
    python local/check_dataloader.py --training_config conf/training_...yaml \\
        --epochs 0 1 1723 --batches 2 --out exp/dataloader_check
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch
from dataset.builder import split_points
from src.layout import HOP, SR, TokenTable
from src.lit_module import LEMASLightningModule

from espnet3.systems.base.training import _instantiate_model
from espnet3.utils.config_utils import load_and_merge_config
from espnet3.utils.run_utils import (
    apply_training_experiment_context,
    resolve_loaded_configs,
)


class _EpochModule(LEMASLightningModule):
    """The training module with a settable epoch and nothing else changed."""

    probe_epoch = 0

    @property
    def current_epoch(self) -> int:
        return self.probe_epoch


def _norm(text: str):
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^\w\s']", " ", text).split()


def _source_text(jsonl_root: Path, ref, split_k):
    """Transcript of the row, or its suffix after split point ``split_k``."""
    jsonl_rel, offset, lang = ref
    with (jsonl_root / jsonl_rel).open("rb") as f:
        f.seek(offset)
        obj = json.loads(f.readline())
    txt = obj.get("txt", "")
    if split_k is None:
        return txt
    words = [
        w["word"]
        for w in (obj.get("align") or {}).get("words", [])
        if "start" in w and "end" in w
    ]
    return txt[split_points(txt, words, lang)[split_k] :]


def _manifest_refs(path: Path, rows):
    """``row -> (jsonl_rel, byte_offset, lang)`` for the requested rows."""
    want, out = set(rows), {}
    with path.open("rb") as f:
        for n, line in enumerate(f):
            if n in want:
                p = line.rstrip(b"\n").split(b"\t")
                out[n] = (p[7].decode(), int(p[8]), p[3].decode())
    return out


def row_problems(r, uids, data, f_len, dataset, wrapper, table, lang):
    """Structural and consistency problems of row ``r`` of a collated batch.

    Args:
        r: Row position in the batch.
        uids: Batch utterance ids (the wrapper's ``str(index)``).
        data: Collated batch dict (``speech``, ``speech_lengths``, ``text``,
            ``text_lengths``, ``cond_frames``).
        f_len: ``[B]`` mel frame counts the model extracts from the batch.
        dataset: The ``LEMASDataset`` behind the wrapper.
        wrapper: The espnet3 dataset the loader reads (``(uid, sample)``).
        table: Token table.
        lang: The row's language code from the manifest.

    Returns:
        ``(problem kinds, info)`` with ``info`` holding the unpadded ``wav``,
        ``ids``, ``cf``, ``n_spk``, the epoch's ``draw`` and the target
        tokens ``got``.

    Example:
        >>> kinds, info = row_problems(0, uids, data, f_len, ds, wrapper, t, "de")
        >>> kinds
        []
    """
    inv = {v: k for k, v in table._id.items()}
    speech, s_len = data["speech"], data["speech_lengths"]
    text, t_len = data["text"], data["text_lengths"]
    n, nt = int(s_len[r]), int(t_len[r])
    cf = int(torch.as_tensor(data["cond_frames"]).view(-1)[r])
    row = int(uids[r])
    kinds = []
    if n > speech.shape[1] or nt > text.shape[1]:
        kinds.append("length beyond tensor")
    if speech[r, n:].abs().sum() != 0 or text[r, nt:].abs().sum() != 0:
        kinds.append("non-zero padding")
    wav = speech[r, :n].numpy()
    ids = text[r, :nt].numpy()
    if int(f_len[r]) != n // HOP + 1:
        kinds.append("mel frames != samples // hop + 1")
    if nt > int(f_len[r]):
        kinds.append("text longer than mel")
    if not 0 <= cf < int(f_len[r]) - 1:
        kinds.append("empty or negative target span")
    roles = ids[:cf]
    if not np.all((roles == table.spk) | (roles == table.lang)):
        kinds.append("prompt frames without a role token")
    n_spk = int((roles == table.spk).sum())
    if np.any(roles[n_spk:] != table.lang):
        kinds.append("role tokens out of order")
    if cf >= len(ids) or int(ids[cf]) != table.tag(lang):
        kinds.append("language tag missing or wrong")
    # the same row through the wrapper, in this process, at this epoch (the
    # loader's workers forked after the epoch was set)
    draw = dataset.draw(row)
    uid, item = wrapper[row]
    if uid != str(row):
        kinds.append("uid is not the manifest row")
    if not (
        np.array_equal(item["speech"], wav)
        and np.array_equal(item["text"], ids)
        and int(item["cond_frames"][0]) == cf
    ):
        kinds.append("batch row != dataset[index]")
    cols = dataset.cols
    want = (
        cols.phones(row).split(" ")
        if draw.split_k is None
        else cols.split_targets(row)[draw.split_k]
    )
    got = [inv[int(x)] for x in ids[cf + 1 :]]
    if got != [t if t in table._id else "<unk>" for t in want]:
        kinds.append("target phones != manifest")
    info = dict(wav=wav, ids=ids, cf=cf, n_spk=n_spk, draw=draw, got=got)
    return kinds, info


def main() -> None:
    """Inspect the configured epochs' batches and print a summary."""
    import Levenshtein
    import soxr

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--training_config", required=True)
    p.add_argument("--epochs", type=int, nargs="+", default=[0, 1])
    p.add_argument("--batches", type=int, default=2, help="per epoch and split")
    p.add_argument("--asr_rows", type=int, default=12, help="per batch")
    p.add_argument("--jsonl_root", default=None)
    p.add_argument("--asr_cache", default=None, help="faster-whisper model dir")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cfg = load_and_merge_config(
        Path(args.training_config), config_name="training.yaml", resolve=False
    )
    apply_training_experiment_context(
        training_config=cfg,
        inference_config=None,
        metrics_config=None,
        publication_config=None,
        log=logging.getLogger(__name__),
    )
    resolve_loaded_configs(cfg, None)
    model = _instantiate_model(cfg)
    lit = _EpochModule(model, cfg)
    table = TokenTable(cfg.token_list)
    from dataset.builder import load_builder_config

    jsonl_root = Path(
        args.jsonl_root
        or Path(load_builder_config()["mirror_root"]) / "LEMAS-train" / "train"
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from faster_whisper import WhisperModel

    from espnet2.bin.spk_inference import Speech2Embedding

    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr = WhisperModel(
        "large-v3",
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
        download_root=args.asr_cache,
    )
    spk = Speech2Embedding.from_pretrained(
        model_tag="espnet/voxcelebs12_ecapa_wavlm_joint", device=device
    )

    def emb(w):
        e = spk(torch.tensor(soxr.resample(w, SR, 16000))).squeeze().float()
        return e / e.norm()

    problems = collections.Counter()
    examples = collections.defaultdict(list)
    wer = collections.defaultdict(lambda: [0, 0])
    sims = collections.defaultdict(list)
    n_rows = 0
    records = []

    def flag(kind, detail):
        problems[kind] += 1
        if len(examples[kind]) < 5:
            examples[kind].append(detail)

    from dataset.dataset import LEMASDataset

    for split in ("train", "valid"):
        wrapper = lit.train_dataset if split == "train" else lit.valid_dataset
        (dataset,) = wrapper.datasets  # one LEMASDataset per split
        assert isinstance(dataset, LEMASDataset), type(dataset)
        manifest = Path(cfg.dataset[split][0].data_src_args.manifest_path)
        for epoch in args.epochs if split == "train" else args.epochs[:1]:
            lit.probe_epoch = epoch
            loader = (
                lit.train_dataloader() if split == "train" else lit.val_dataloader()
            )
            for b, batch in enumerate(loader):
                if b >= args.batches:
                    break
                uids, data = batch
                with torch.no_grad():
                    _feats, f_len = model._extract_feats(
                        data["speech"], data["speech_lengths"]
                    )
                rows = [int(u) for u in uids]
                refs = _manifest_refs(manifest, rows)
                for r, row in enumerate(rows):
                    n_rows += 1
                    tag = f"{split} e{epoch} b{b} row {row}"
                    lang = refs[row][2]
                    kinds, row_info = row_problems(
                        r, uids, data, f_len, dataset, wrapper, table, lang
                    )
                    for kind in kinds:
                        flag(kind, tag)
                    wav, cf, n_spk = row_info["wav"], row_info["cf"], row_info["n_spk"]
                    draw, got = row_info["draw"], row_info["got"]
                    roles = row_info["ids"][:cf]
                    problems["unk_tokens"] += got.count("<unk>")
                    problems["target_tokens"] += len(got)
                    if r >= args.asr_rows:
                        continue
                    tgt = wav[cf * HOP :]
                    ref = _source_text(jsonl_root, refs[row], draw.split_k)
                    segs, _ = asr.transcribe(tgt, language=lang, beam_size=5)
                    hyp = " ".join(s.text for s in segs).strip()
                    if lang == "zh":
                        a, h = list("".join(_norm(ref))), list("".join(_norm(hyp)))
                    else:
                        a, h = _norm(ref), _norm(hyp)
                    e = Levenshtein.distance(a, h)
                    key = lang + ("_split" if draw.split_k is not None else "")
                    wer[key][0] += e
                    wer[key][1] += len(a)
                    rec = dict(
                        tag=tag, lang=lang, split_k=draw.split_k, ref=ref, hyp=hyp
                    )
                    if len(tgt) > 8000:
                        et = emb(tgt)
                        n_lang = int((roles == table.lang).sum())
                        if n_spk * HOP > 8000:
                            rec["spk_sim"] = float(emb(wav[: n_spk * HOP]) @ et)
                            sims[f"spk_{lang}"].append(rec["spk_sim"])
                        if n_lang * HOP > 8000:
                            seg = wav[n_spk * HOP : (n_spk + n_lang) * HOP]
                            rec["lang_sim"] = float(emb(seg) @ et)
                            sims[f"lang_{lang}"].append(rec["lang_sim"])
                    records.append(rec)
                print(f"{split} epoch {epoch} batch {b}: {len(rows)} rows", flush=True)

    nz = [v for k, v in wer.items() if not k.startswith("zh")]
    summary = dict(
        rows_checked=n_rows,
        problems={
            k: v
            for k, v in problems.items()
            if k not in ("unk_tokens", "target_tokens")
        },
        unk_rate=problems["unk_tokens"] / max(problems["target_tokens"], 1),
        examples=examples,
        pooled_wer_no_zh=sum(v[0] for v in nz) / max(sum(v[1] for v in nz), 1),
        per_lang_err={
            k: (round(v[0] / max(v[1], 1), 3), v[1]) for k, v in sorted(wer.items())
        },
        sim={k: (round(float(np.mean(v)), 3), len(v)) for k, v in sorted(sims.items())},
    )
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
