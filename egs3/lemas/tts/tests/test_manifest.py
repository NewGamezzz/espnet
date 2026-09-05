from dataset.manifest import COLUMNS, ManifestColumns, ManifestRow, write_manifest


def _row(i, **kw):
    base = dict(
        utt_id=f"de_vidAAAAAAAA-0000{i}-00000000-00000300",
        audio=f"de/de000/u{i}.flac",
        phones="a b <space> c",
        lang="de",
        source="yodas",
        group="vidAAAAAAAA",
        dur=2.5,
        jsonl_path="de/de000.jsonl",
        byte_offset=0,
        spk_mode="group",
        word_bounds="",
        phones_by_word="",
    )
    base.update(kw)
    return ManifestRow(**base)


def test_roundtrip(tmp_path):
    p = tmp_path / "m.tsv"
    rows = [
        _row(0),
        _row(
            1,
            utt_id="ru_" + "a" * 32,
            group="",
            spk_mode="split",
            word_bounds="0.1:0.5,0.6:1.2",
            phones_by_word="a b|c d",
            dur=1.3,
            source="golos",
            lang="ru",
        ),
    ]
    write_manifest(rows, p)
    assert p.read_text().splitlines()[0].count("\t") == len(COLUMNS) - 1
    cols = ManifestColumns.load(p)
    assert cols.n_rows == 2
    assert cols.audio(1) == "de/de000/u1.flac" and cols.phones(0) == "a b <space> c"
    assert cols.group[0] >= 0 and cols.group[1] == -1
    assert cols.spk_mode.tolist() == [1, 2]
    assert cols.seg.tolist() == [0, -1]
    assert cols.word_bounds(1) == [(0.1, 0.5), (0.6, 1.2)]
    assert cols.phones_by_word(1) == [["a", "b"], ["c", "d"]]
    assert cols.word_bounds(0) == []
    assert abs(float(cols.dur[1]) - 1.3) < 1e-6
    assert cols.group_names == ["vidAAAAAAAA"]


def test_manifest_column_three_is_phones(tmp_path):
    p = tmp_path / "m.tsv"
    write_manifest([_row(0)], p)
    assert p.read_text().split("\t")[2] == "a b <space> c"
