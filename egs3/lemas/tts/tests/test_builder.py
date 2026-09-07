from dataset.builder import (
    LEMASBuilder,
    build_rows,
    decide_spk_mode,
    lang_stats,
    split_candidates,
)
from dataset.manifest import ManifestRow

CFG = dict(
    split_min_sec=4.0,
    split_min_prompt_sec=1.0,
    split_frac=[0.2, 0.4],
    min_target_sec=1.0,
    max_target_sec=20.0,
    valid_rows_per_lang=1,
    seed=1,
)


class FakePhon:
    def phonemize(self, text, lang):
        return [c for c in text.replace(" ", "_")]

    def phonemize_words(self, words, lang):
        return [[c for c in w] for w in words]


def test_decide_spk_mode():
    assert decide_spk_mode(group_size=3, dur=2.0, cfg=CFG) == "group"
    assert decide_spk_mode(group_size=1, dur=5.0, cfg=CFG) == "split"
    assert decide_spk_mode(group_size=1, dur=3.0, cfg=CFG) == "none"
    assert decide_spk_mode(group_size=0, dur=5.0, cfg=CFG) == "split"


def test_split_candidates_respects_fraction_and_floor():
    wb = [(0.0, 0.5), (0.6, 1.4), (1.5, 2.2), (2.3, 3.1), (3.2, 5.0)]
    ks = split_candidates(wb, dur=5.0, cfg=CFG)  # prompt = words[:k], k >= 1
    # k=2 -> prompt ends 1.4 s (28%); k=3 -> 2.2 s (44%, too long); k=1 -> 0.5 s (< 1 s)
    assert ks == [2]


def test_build_rows_assigns_modes_groups_and_split_columns():
    pool = [
        (
            "de_vidAAAAAAAA-00001-00000000-00000200",
            "de/de000/a.flac",
            2.0,
            "yodas",
            "de/de000.jsonl",
            0,
            "hallo welt",
            [("hallo", 0.1, 0.9), ("welt", 1.0, 1.9)],
        ),
        (
            "de_vidAAAAAAAA-00002-00000200-00000400",
            "de/de000/b.flac",
            2.0,
            "yodas",
            "de/de000.jsonl",
            100,
            "guten tag",
            [("guten", 0.1, 0.9), ("tag", 1.0, 1.9)],
        ),
        (
            "zh_emilia_zh_0000000001",
            "zh/zh000/c.flac",
            6.0,
            "emilia",
            "zh/zh000.jsonl",
            0,
            "你 好 世 界 朋 友",
            [
                ("你", 0.2, 0.9),
                ("好", 1.0, 1.7),
                ("世", 1.8, 2.5),
                ("界", 2.6, 3.4),
                ("朋", 3.5, 4.6),
                ("友", 4.7, 5.8),
            ],
        ),
        (
            "zh_emilia_zh_0000000002",
            "zh/zh000/d.flac",
            0.5,
            "emilia",
            "zh/zh000.jsonl",
            90,
            "短",
            [("短", 0.1, 0.4)],
        ),
    ]
    rows = build_rows(pool, CFG, FakePhon())
    by = {r.utt_id: r for r in rows}
    assert len(rows) == 3  # the 0.5 s row is filtered
    a = by["de_vidAAAAAAAA-00001-00000000-00000200"]
    assert a.spk_mode == "group" and a.group == "vidAAAAAAAA"
    zh = by["zh_emilia_zh_0000000001"]
    assert zh.spk_mode == "split" and zh.group == ""
    assert zh.word_bounds.startswith("0.2:0.9,1:1.7")
    assert zh.phones_by_word.count("|") == 5
    assert zh.phones == "你 _ 好 _ 世 _ 界 _ 朋 _ 友"


def test_lang_stats_counts_column3_tokens_per_second():
    r = ManifestRow(
        "de_x", "a", "a b <space> c", "de", "yodas", "g", 2.0, "j", 0, "group", "", ""
    )
    assert lang_stats([r]) == {"de": {"tokens_per_sec": 2.0}}


def test_builder_is_a_dataset_builder():
    from espnet3.components.data.dataset_builder import DatasetBuilder

    assert issubclass(LEMASBuilder, DatasetBuilder)


def _fake_mirror(tmp_path):
    """A two-language mirror: shard tars, jsonl and poc3k tsv."""
    import json as _json

    mirror = tmp_path / "mirror"
    rows = {
        "de": [
            (
                "de_vidAAAAAAAA-0000%d-00000000-00000300" % i,
                "de000/a%d.mp3" % i,
                3.0,
                "yodas",
                "hallo welt",
                [("hallo", 0.1, 0.9), ("welt", 1.0, 2.9)],
            )
            for i in range(6)
        ]
        + [
            (
                "de_vidBBBBBBBB-0000%d-00000000-00000300" % i,
                "de000/b%d.mp3" % i,
                3.0,
                "yodas",
                "guten tag",
                [("guten", 0.1, 0.9), ("tag", 1.0, 2.9)],
            )
            for i in range(4)
        ],
        "zh": [
            (
                "zh_emilia_zh_000000000%d" % i,
                "zh000/c%d.mp3" % i,
                6.0,
                "emilia",
                "你 好 世 界 朋 友",
                [
                    ("你", 0.2, 0.9),
                    ("好", 1.0, 1.7),
                    ("世", 1.8, 2.5),
                    ("界", 2.6, 3.4),
                    ("朋", 3.5, 4.6),
                    ("友", 4.7, 5.8),
                ],
            )
            for i in range(4)
        ],
    }
    for lang, rs in rows.items():
        jl = mirror / "LEMAS-train" / "train" / lang / f"{lang}000.jsonl"
        jl.parent.mkdir(parents=True, exist_ok=True)
        tsv = mirror / "manifests_poc3k" / lang / f"{lang}000.tsv"
        tsv.parent.mkdir(parents=True, exist_ok=True)
        # the shard tar: WAV members of the stated duration (48 samples short,
        # like the real jsonl durations)
        import io as _io
        import tarfile as _tarfile

        import numpy as _np
        import soundfile as _sf

        with _tarfile.open(jl.with_suffix(".tar.gz"), "w:gz") as tf:
            for key, audio, dur, source, txt, words in rs:
                buf = _io.BytesIO()
                _sf.write(
                    buf,
                    _np.zeros(int(dur * 16000) - 48, _np.float32),
                    16000,
                    format="WAV",
                )
                data = buf.getvalue()
                info = _tarfile.TarInfo(audio)
                info.size = len(data)
                tf.addfile(info, _io.BytesIO(data))
        off = 0
        with jl.open("wb") as fj, tsv.open("w") as ft:
            for key, audio, dur, source, txt, words in rs:
                obj = {
                    "key": key,
                    "audio": audio,
                    "dur": dur,
                    "txt": txt,
                    "align": {
                        "words": [
                            {"word": w, "start": s, "end": e, "score": 1.0}
                            for w, s, e in words
                        ]
                    },
                }
                line = (_json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
                fj.write(line)
                ft.write(f"{key}\t{audio}\t{dur}\t{source}\t{off}\n")
                off += len(line)
    return mirror


def test_build_end_to_end_fake_mirror(tmp_path):
    import json as _json

    mirror = _fake_mirror(tmp_path)
    cfg = dict(
        CFG,
        mirror_root=str(mirror),
        manifest_dir="manifests_poc3k",
        audio_root=str(tmp_path / "pcm"),
        data_path="data",
        langs=["de", "zh"],
        source_sample_rate=16000,
        valid_rows_per_lang=2,
        n_workers=1,
        manifest_paths={"train": "manifest/train.tsv", "valid": "manifest/valid.tsv"},
        lang_stats_path="lang_stats.json",
        chunk_size=3,
    )
    b = LEMASBuilder(cfg)
    recipe = tmp_path / "recipe"
    assert not b.is_source_prepared()
    b.prepare_source()
    assert b.is_source_prepared()
    assert (tmp_path / "pcm" / "de" / "de000.pcm").is_file()
    assert not b.is_built(recipe_dir=recipe)
    b.build(recipe_dir=recipe, phonemizer_factory=FakePhon)
    assert b.is_built(recipe_dir=recipe)
    train = (recipe / "data/manifest/train.tsv").read_text().splitlines()
    valid = (recipe / "data/manifest/valid.tsv").read_text().splitlines()
    assert len(train) + len(valid) == 14
    # whole groups are held out: valid de rows share one video, none of it in train
    de_valid_groups = {ln.split("\t")[5] for ln in valid if ln.startswith("de_")}
    assert len(de_valid_groups) == 1 and not any(
        ln.split("\t")[5] in de_valid_groups for ln in train
    )
    de_train = [ln.split("\t") for ln in train if ln.startswith("de_")]
    assert all(p[1].startswith("de/de000.pcm:") for p in de_train)
    # dur comes from the packed length (48 samples short of the jsonl's 3.0 s)
    assert all(abs(float(p[6]) - (48000 - 48) / 16000) < 1e-6 for p in de_train)
    stats = _json.loads((recipe / "data/lang_stats.json").read_text())
    assert set(stats) == {"de", "zh"} and stats["zh"]["tokens_per_sec"] > 0
    modes = _json.loads((recipe / "data/spk_mode_counts.json").read_text())
    assert modes["de"] == {"group": 10} and modes["zh"] == {"split": 4}


def test_drop_text_regex_removes_matching_rows_before_phonemizing():
    cfg = dict(CFG, drop_text_regex={"zh": "[A-Za-z]"})
    pool = [
        (
            "zh_emilia_zh_0000000001",
            "zh/a.flac",
            3.0,
            "emilia",
            "zh/z.jsonl",
            0,
            "你好",
            [],
        ),
        (
            "zh_emilia_zh_0000000002",
            "zh/b.flac",
            3.0,
            "emilia",
            "zh/z.jsonl",
            1,
            "And he was raised in Nigeria",
            [],
        ),
        (
            "de_vidAAAAAAAA-00001-00000000-00000300",
            "de/c.flac",
            3.0,
            "yodas",
            "de/d.jsonl",
            2,
            "Hello there",
            [],
        ),
    ]
    rows = build_rows(pool, cfg, FakePhon())
    assert [r.utt_id for r in rows] == [
        "zh_emilia_zh_0000000001",
        "de_vidAAAAAAAA-00001-00000000-00000300",
    ]


def test_word_bounds_are_ms_rounded_so_manifest_and_build_agree(tmp_path):
    # a boundary exactly at 40% of the duration: {:g} formatting of the raw
    # float would round it past the window and the dataset would find no split
    import json as _json

    from dataset.builder import _chunk_job, split_candidates

    dur = 12.3456888  # 0.4 * dur = 4.93827552; {:g} of the raw end gives 4.93828
    words = [("a", 0.0, 0.4 * dur - 0.0000001), ("b", 0.4 * dur, dur)]
    jl = tmp_path / "zh000.jsonl"
    obj = {
        "key": "zh_emilia_zh_0000000001",
        "audio": "zh000/x.mp3",
        "dur": dur,
        "txt": "a b",
        "align": {"words": [{"word": w, "start": s, "end": e} for w, s, e in words]},
    }
    jl.write_bytes((_json.dumps(obj) + "\n").encode())
    chunk = [("zh_emilia_zh_0000000001", "zh/zh000.pcm:0:197531", dur, "emilia", 0)]
    cfg = dict(CFG, n_workers=1)
    from dataset import builder as _b

    _b._worker_init(FakePhon)
    (line,) = _chunk_job((chunk, str(jl), "zh/zh000.jsonl", cfg, {}))
    parts = line[-1].split("\t")
    assert parts[9] == "split"
    wb = [tuple(float(v) for v in x.split(":")) for x in parts[10].split(",")]
    assert split_candidates(wb, float(parts[6]), cfg) == [1]
