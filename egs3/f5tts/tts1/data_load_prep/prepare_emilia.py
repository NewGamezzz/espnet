"""
Prepare Emilia dataset (EN + ZH) for F5-TTS training in ESPnet.

Outputs per language:
  <out_dir>/EN/train.csv        pipe-delimited: audio_file|text
  <out_dir>/EN/val.csv
  <out_dir>/ZH/train.csv
  <out_dir>/ZH/val.csv
  <out_dir>/token_list/tokens.txt
  <out_dir>/duration.json
"""

import argparse
import json
import random
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Curated blocklists
# ---------------------------------------------------------------------------
OUT_ZH = {
    "ZH_B00041_S06226", "ZH_B00042_S09204", "ZH_B00065_S09430",
    "ZH_B00065_S09431", "ZH_B00066_S09327", "ZH_B00066_S09328",
}

OUT_EN = {
    "EN_B00013_S00913", "EN_B00042_S00120", "EN_B00055_S04111",
    "EN_B00061_S00693", "EN_B00061_S01494", "EN_B00061_S03375",
    "EN_B00059_S00092", "EN_B00111_S04300", "EN_B00100_S03759",
    "EN_B00087_S03811", "EN_B00059_S00950", "EN_B00089_S00946",
    "EN_B00078_S05127", "EN_B00070_S04089", "EN_B00074_S09659",
    "EN_B00061_S06983", "EN_B00061_S07060", "EN_B00059_S08397",
    "EN_B00082_S06192", "EN_B00091_S01238", "EN_B00089_S07349",
    "EN_B00070_S04343", "EN_B00061_S02400", "EN_B00076_S01262",
    "EN_B00068_S06467", "EN_B00076_S02943", "EN_B00064_S05954",
    "EN_B00061_S05386", "EN_B00066_S06544", "EN_B00076_S06944",
    "EN_B00072_S08620", "EN_B00076_S07135", "EN_B00076_S09127",
    "EN_B00065_S00497", "EN_B00059_S06227", "EN_B00063_S02859",
    "EN_B00075_S01547", "EN_B00061_S08286", "EN_B00079_S02901",
    "EN_B00092_S03643", "EN_B00096_S08653", "EN_B00063_S04297",
    "EN_B00063_S04614", "EN_B00079_S04698", "EN_B00104_S01666",
    "EN_B00061_S09504", "EN_B00061_S09694", "EN_B00065_S05444",
    "EN_B00063_S06860", "EN_B00065_S05725", "EN_B00069_S07628",
    "EN_B00083_S03875", "EN_B00071_S07665", "EN_B00062_S04187",
    "EN_B00065_S09873", "EN_B00065_S09922", "EN_B00084_S02463",
    "EN_B00067_S05066", "EN_B00106_S08060", "EN_B00073_S06399",
    "EN_B00073_S09236", "EN_B00087_S00432", "EN_B00085_S05618",
    "EN_B00064_S01262", "EN_B00072_S01739", "EN_B00059_S03913",
    "EN_B00069_S04036", "EN_B00067_S05623", "EN_B00060_S05389",
    "EN_B00060_S07290", "EN_B00062_S08995",
}

ZH_CHAR_FILTERS = ["い", "て"]
EN_CHAR_FILTERS = ["ا", "い", "て"]

ZH_PUNCT_TABLE = str.maketrans({",": "，", "!": "！", "?": "？"})


# ---------------------------------------------------------------------------
# Repetition detection
# ---------------------------------------------------------------------------
def repetition_found(text: str, length: int = 2) -> bool:
    words = text.split()
    if len(words) < length:
        return False
    seen: set = set()
    for i in range(len(words) - length + 1):
        ng = tuple(words[i: i + length])
        if ng in seen:
            return True
        seen.add(ng)
    return False


# ---------------------------------------------------------------------------
# Per-batch-directory worker
# ---------------------------------------------------------------------------
def process_batch_dir(args):
    batch_dir: Path
    db_root: Path
    batch_dir, db_root = args

    lang      = batch_dir.parent.name.upper()
    out_set   = OUT_EN if lang == "EN" else OUT_ZH
    ch_filter = EN_CHAR_FILTERS if lang == "EN" else ZH_CHAR_FILTERS
    rep_len   = 4 if lang == "EN" else 2

    rows, durations, vocab_set = [], [], set()
    bad_zh = bad_en = 0

    try:
        # single listdir call — much cheaper than glob/scandir on Lustre
        all_files = os.listdir(batch_dir)
    except OSError:
        return [], [], set(), 0, 0

    # build a set of mp3s for fast existence check — avoids per-file os.path.exists
    mp3_set = {f for f in all_files if f.endswith(".mp3")}
    json_files = [f for f in all_files if f.endswith(".json")]

    if not json_files:
        return [], [], set(), 0, 0

    for fname in json_files:
        json_path = batch_dir / fname
        mp3_name  = fname.replace(".json", ".mp3")

        if mp3_name not in mp3_set:
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        utt_id = obj["id"]
        text   = obj["text"].strip()
        dur    = obj["duration"]
        wav    = str(batch_dir / mp3_name)

        if utt_id in out_set:
            if lang == "ZH": bad_zh += 1
            else:            bad_en += 1
            continue

        if any(ch in text for ch in ch_filter):
            if lang == "ZH": bad_zh += 1
            else:            bad_en += 1
            continue

        if repetition_found(text, length=rep_len):
            if lang == "ZH": bad_zh += 1
            else:            bad_en += 1
            continue

        if lang == "ZH":
            text = text.translate(ZH_PUNCT_TABLE)

        rows.append((wav, text, dur))
        durations.append(dur)
        vocab_set.update(list(text))

    return rows, durations, vocab_set, bad_zh, bad_en
# ---------------------------------------------------------------------------
# Write F5-TTS style pipe-delimited CSV
# ---------------------------------------------------------------------------
def write_csv(rows: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("audio_file|text\n")
        for wav, text, _dur in rows:
            f.write(f"{wav}|{text}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare Emilia EN+ZH for F5-TTS training in ESPnet"
    )
    parser.add_argument("db_emilia", help="Root of Emilia corpus (contains EN/ and ZH/ subdirs)")
    parser.add_argument("out_dir",   help="Output directory for CSVs and token list")
    parser.add_argument("--langs",     nargs="+", default=["EN", "ZH"])
    parser.add_argument("--nj",        type=int,   default=32)
    parser.add_argument("--val_ratio", type=float, default=0.01,
                        help="Fraction of each language to hold out as val (default: 0.01)")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    assert 0.0 < args.val_ratio < 1.0, "--val_ratio must be between 0 and 1"

    db_root  = Path(args.db_emilia)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_vocab:     set  = set()
    all_durations: list = []

    for lang in args.langs:
        lang_dir = db_root / lang
        if not lang_dir.is_dir():
            print(f"Warning: {lang_dir} not found, skipping.", file=sys.stderr)
            continue

        batch_dirs  = sorted([d for d in lang_dir.iterdir() if d.is_dir()])
        worker_args = [(bd, db_root) for bd in batch_dirs]
        lang_rows: list = []
        total_bad_zh = total_bad_en = 0

        print(f"\n[{lang}] Found {len(batch_dirs)} batch directories.")

        with ProcessPoolExecutor(max_workers=args.nj) as ex:
            for rows, durs, vocab, bad_zh, bad_en in tqdm(
                ex.map(process_batch_dir, worker_args),
                total=len(worker_args),
                desc=f"[{lang}] Processing",
            ):
                lang_rows.extend(rows)
                all_durations.extend(durs)
                all_vocab.update(vocab)
                total_bad_zh += bad_zh
                total_bad_en += bad_en

        random.seed(args.seed)
        random.shuffle(lang_rows)
        n_val   = max(1, int(len(lang_rows) * args.val_ratio))
        n_train = len(lang_rows) - n_val

        lang_out = out_root / lang
        write_csv(lang_rows[:n_train], lang_out / "train.csv")
        write_csv(lang_rows[n_train:], lang_out / "val.csv")

        print(f"[{lang}] Kept {len(lang_rows)}  "
              f"(filtered: ZH={total_bad_zh}, EN={total_bad_en})")
        print(f"[{lang}] Train: {n_train}  Val: {n_val}  "
              f"({(1-args.val_ratio)*100:.1f}/{args.val_ratio*100:.1f} split)")
        print(f"[{lang}] Duration: {sum(d for _, _, d in lang_rows)/3600:.2f} h")

    # Token list
    token_dir  = out_root / "token_list"
    token_dir.mkdir(parents=True, exist_ok=True)
    token_path = token_dir / "tokens.txt"
    with open(token_path, "w", encoding="utf-8") as f:
        f.write("<blank>\n")
        f.write("<unk>\n")
        for tok in sorted(all_vocab):
            f.write(tok + "\n")
        f.write("<sos/eos>\n")
    print(f"\nToken list → {token_path}  ({len(all_vocab)+3} tokens incl. specials)")

    # duration.json
    dur_path = out_root / "duration.json"
    with open(dur_path, "w", encoding="utf-8") as f:
        json.dump({"duration": all_durations}, f, ensure_ascii=False)
    print(f"Duration manifest → {dur_path}")
    print(f"\nTotal: {sum(all_durations)/3600:.2f} h  |  Vocab: {len(all_vocab)} chars")


if __name__ == "__main__":
    main()
