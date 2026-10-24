"""The dataloader check must pass a correct batch and flag broken ones."""

import torch
from local.check_dataloader import row_problems
from src.layout import HOP, TokenTable

from espnet2.train.collate_fn import CommonCollateFn

from .test_dataset import _ds


class _Wrapper:
    """What espnet3's CombinedDataset hands the loader: ``(str(idx), sample)``."""

    def __init__(self, ds):
        self.datasets = [ds]

    def __getitem__(self, idx):
        return str(idx), self.datasets[0][idx]


def _batch(corpus, rows, **kw):
    ds = _ds(corpus, **kw)
    ds.set_epoch(4)
    wrapper = _Wrapper(ds)
    collate = CommonCollateFn(
        int_pad_value=0, float_pad_value=0.0, not_sequence=["cond_frames"]
    )
    uids, data = collate([wrapper[i] for i in rows])
    f_len = data["speech_lengths"] // HOP + 1  # vocos framing, centre-padded
    table = TokenTable(corpus["tokens"])
    langs = {i: ("zh" if ds.cols.lang[i] == 9 else "de") for i in rows}
    return ds, wrapper, uids, data, f_len, table, langs


def _check(corpus, rows, mutate=None, **kw):
    ds, wrapper, uids, data, f_len, table, langs = _batch(corpus, rows, **kw)
    if mutate:
        mutate(uids, data)
    return [
        row_problems(r, uids, data, f_len, ds, wrapper, table, langs[int(u)])[0]
        for r, u in enumerate(uids)
    ]


def _rows(corpus):
    ds = _ds(corpus)
    # a group row, the 8 s mls row and a zh split row: different lengths
    split = next(i for i in range(len(ds)) if ds.cols.spk_mode[i] == 2)
    return [0, 5, split]


def test_a_correct_batch_has_no_problems(corpus):
    assert _check(corpus, _rows(corpus)) == [[], [], []]


def test_audio_of_another_time_is_caught(corpus):
    def shift(uids, data):  # the regroup bug: the row holds earlier audio
        n = int(data["speech_lengths"][0])
        data["speech"][0, :n] = torch.roll(data["speech"][0, :n], 4000)

    assert "batch row != dataset[index]" in _check(corpus, _rows(corpus), shift)[0]


def test_rows_swapped_between_uids_are_caught(corpus):
    def swap(uids, data):
        uids[0], uids[1] = uids[1], uids[0]

    problems = _check(corpus, _rows(corpus), swap)
    assert "batch row != dataset[index]" in problems[0]
    assert "batch row != dataset[index]" in problems[1]


def test_padding_leak_is_caught(corpus):
    def leak(uids, data):
        r = int(torch.argmin(data["speech_lengths"]))
        data["speech"][r, -1] = 0.5

    problems = _check(corpus, _rows(corpus), leak)
    assert any("non-zero padding" in p for p in problems)


def test_wrong_text_is_caught(corpus):
    def retag(uids, data):  # the target phones of another row
        cf = int(data["cond_frames"][0])
        data["text"][0, cf + 1] = 3 if int(data["text"][0, cf + 1]) == 2 else 2

    problems = _check(corpus, _rows(corpus), retag)[0]
    assert "target phones != manifest" in problems
    assert "batch row != dataset[index]" in problems


def test_missing_role_tokens_are_caught(corpus):
    def unrole(uids, data):
        data["text"][0, 0] = 2  # a phone where the speaker prompt is

    assert (
        "prompt frames without a role token" in _check(corpus, _rows(corpus), unrole)[0]
    )


def test_a_correct_stock_batch_has_no_problems(corpus):
    assert _check(corpus, _rows(corpus), layout="stock") == [[], [], []]


def test_stock_audio_of_another_time_is_caught(corpus):
    def shift(uids, data):
        n = int(data["speech_lengths"][0])
        data["speech"][0, :n] = torch.roll(data["speech"][0, :n], 4000)

    problems = _check(corpus, _rows(corpus), shift, layout="stock")[0]
    assert "speech != the row's own audio" in problems


def test_stock_wrong_phone_is_caught(corpus):
    def retag(uids, data):
        data["text"][0, 1] = 3 if int(data["text"][0, 1]) == 2 else 2

    problems = _check(corpus, _rows(corpus), retag, layout="stock")[0]
    assert "target phones != manifest" in problems


def test_stock_row_with_a_prompt_boundary_is_caught(corpus):
    def bound(uids, data):
        data["cond_frames"][0] = 3

    problems = _check(corpus, _rows(corpus), bound, layout="stock")[0]
    assert "stock row without the random-span sentinel" in problems


def test_a_stock_batch_without_tag_passes(corpus):
    # S0a (stock F5-TTS): no language tag in the lane, phones == manifest
    assert _check(corpus, [0, 5], layout="stock", text_tag=False) == [[], []]


def test_a_tag_in_a_no_tag_lane_is_caught(corpus):
    def add_tag(uids, data):  # a <de> slipped in although text_tag is false
        n = int(data["text_lengths"][0])
        text = torch.nn.functional.pad(data["text"], (0, 1))
        text[0, 1 : n + 1] = data["text"][0, :n].clone()
        text[0, 0] = 7  # <de>
        data["text"] = text
        data["text_lengths"][0] = n + 1

    kinds = _check(corpus, [0], mutate=add_tag, layout="stock", text_tag=False)
    assert "language tag present but text_tag is false" in kinds[0]
