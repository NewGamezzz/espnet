"""Numpy-backed NumElementsBatchSampler with an upstream-matching max_samples
cap.

Upstream F5-TTS caps every batch at ``max_samples=64``;
espnet2's ``NumElementsBatchSampler`` has no such parameter, and at the short
end of Emilia's length-sorted order an uncapped batch exceeds 300 samples
(0.3 s floor = 28 frames, batch_bins=480000 / 100 mel channels = 4800
frames/batch). This module provides a drop-in replacement that adds the cap.

The packing loop below is a structural port of
``espnet2.samplers.num_elements_batch_sampler.NumElementsBatchSampler``:
same accumulate-until-batch_bins rule, same min_batch_size interaction, same
ascending sort order, same trailing-remainder/drop_last handling, same
sort_in_batch/sort_batch semantics. With ``max_samples`` unset the extra
cap condition is always false, so the two samplers produce exactly the same
batch partition (see ``egs3/emilia/tts/tests/test_sampler.py``). The dict
of Python lists that the stock sampler builds from the shape file is
replaced with flat numpy arrays, which is the "array" half of this class's
name and what keeps the recipe's startup RSS down for a 37M-utterance
corpus.

Keys are assumed to be non-negative integers, matching this recipe's
``create_shape`` stage (``src/shape.py``), which writes ``str(index)`` as
the utterance id. Arbitrary string utterance ids are not supported.
"""

from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
from typeguard import typechecked

from espnet2.samplers.abs_sampler import AbsSampler


def _load_shape_file(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a shape file ("<key> <length>,<d1>,<d2>,...") into arrays.

    Returns (keys, lengths, dims): int64 keys, int32 lengths, and int64
    per-row products of the trailing dimensions (1 when there are none).
    Row order matches file line order.

    Two passes (count, then fill preallocated arrays) rather than building
    Python lists and converting: at Emilia's 37M-utterance scale, three
    37M-element Python int lists would themselves be a multi-GB transient
    RSS spike, defeating the point of the numpy backing.

    Raises RuntimeError on a non-integer key (this sampler requires
    integer keys; see the class docstring) or a duplicate key (stock's
    ``read_2columns_text`` raises on duplicates too; a hand-rolled parser
    that silently kept both rows would emit the same utterance in two
    different batches).
    """
    with open(path, "r", encoding="utf-8") as f:
        n_lines = sum(1 for line in f if line.strip())

    keys = np.empty(n_lines, dtype=np.int64)
    lengths = np.empty(n_lines, dtype=np.int32)
    dims = np.empty(n_lines, dtype=np.int64)

    with open(path, "r", encoding="utf-8") as f:
        i = 0
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            key_str, shape_str = line.split(maxsplit=1)
            parts = shape_str.split(",")
            try:
                key_val = int(key_str)
            except ValueError as e:
                raise RuntimeError(
                    f"{path}:{lineno}: numel_array requires integer keys, "
                    f"got {key_str!r}"
                ) from e
            keys[i] = key_val
            lengths[i] = int(parts[0])
            dims[i] = (
                int(np.prod([int(x) for x in parts[1:]])) if len(parts) > 1 else 1
            )
            i += 1

    # O(n log n) duplicate check (mirrors read_2columns_text's duplicate
    # guard, which this hand-rolled parser otherwise bypasses).
    sorted_keys = np.sort(keys)
    dup_mask = sorted_keys[1:] == sorted_keys[:-1]
    if dup_mask.any():
        dup_key = int(sorted_keys[1:][dup_mask][0])
        raise RuntimeError(f"{path}: duplicate key {dup_key}")

    return keys, lengths, dims


class NumElementsArraySampler(AbsSampler):
    """Drop-in, numpy-backed replacement for NumElementsBatchSampler.

    Accepts the same arguments as
    ``espnet2.samplers.num_elements_batch_sampler.NumElementsBatchSampler``,
    plus ``max_samples``: an optional hard cap on the number of samples per
    batch, matching upstream F5-TTS's ``max_samples=64``. When
    ``max_samples`` is None (the default) this sampler produces exactly the
    same batch partition as the stock sampler.

    Shape file keys must be non-negative integers (this recipe's
    ``create_shape`` stage writes ``str(index)`` as the utterance id;
    arbitrary string utterance ids are not supported and raise
    ``RuntimeError`` with the offending file, line, and value).
    """

    @typechecked
    def __init__(
        self,
        batch_bins: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        min_batch_size: int = 1,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
        padding: bool = True,
        max_samples: Optional[int] = None,
    ):
        assert batch_bins > 0
        if max_samples is not None:
            assert max_samples > 0
            # close_by_cap (unlike close_by_bins) ignores min_batch_size by
            # design, since the cap is a hard ceiling: if max_samples were
            # allowed below min_batch_size, every batch would close at
            # max_samples, the trailing-remainder redistribution above
            # would stall with every batch already at the cap, and every
            # realized batch would end up under min_batch_size with no
            # error (the espnet3 dataloader only validates the *configured*
            # min_batch_size, not realized batch sizes, so an under-sized
            # batch on an 8-GPU run would surface much later as a DDP
            # deadlock, not here).
            assert max_samples >= min_batch_size, (
                f"max_samples ({max_samples}) must be >= min_batch_size "
                f"({min_batch_size})"
            )
        if sort_batch != "ascending" and sort_batch != "descending":
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )
        if sort_in_batch != "descending" and sort_in_batch != "ascending":
            raise ValueError(
                "sort_in_batch must be ascending"
                f" or descending: {sort_in_batch}"
            )

        self.batch_bins = batch_bins
        self.shape_files = shape_files
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last
        self.max_samples = max_samples

        per_file = [_load_shape_file(s) for s in shape_files]

        primary_keys, primary_lengths, primary_dims = per_file[0]
        if len(primary_keys) == 0:
            raise RuntimeError(f"0 lines found: {shape_files[0]}")

        # Skip entirely for the common single-shape-file case (this
        # recipe's only real usage): building a 37M-element Python set just
        # to compare it against itself would be a multi-GB no-op. For the
        # multi-file case, compare sorted arrays instead of Python sets to
        # keep the check numpy-backed too.
        if len(shape_files) > 1:
            primary_sorted = np.sort(primary_keys)
            for s, (k, _, _) in zip(shape_files[1:], per_file[1:]):
                if not np.array_equal(np.sort(k), primary_sorted):
                    raise RuntimeError(
                        f"keys are mismatched between {s} != {shape_files[0]}"
                    )

        # Sort ascending by the *first* shape file's length, exactly as
        # NumElementsBatchSampler does (keys = sorted(first_utt2shape,
        # key=lambda k: first_utt2shape[k][0])). argsort with a stable sort
        # preserves file-line order on ties, matching Python's sorted().
        order = np.argsort(primary_lengths, kind="stable")
        sorted_keys = primary_keys[order]

        lengths_by_file = [primary_lengths[order]]
        dims_by_file = [primary_dims[order]]
        for keys_i, lengths_i, dims_i in per_file[1:]:
            pos = {int(k): idx for idx, k in enumerate(keys_i.tolist())}
            idx_map = np.array(
                [pos[int(k)] for k in sorted_keys.tolist()], dtype=np.int64
            )
            lengths_by_file.append(lengths_i[idx_map])
            dims_by_file.append(dims_i[idx_map])

        n = len(sorted_keys)
        n_files = len(shape_files)

        if padding:
            # If padding case, the feat-dim must be same over the whole
            # corpus, so the first sample (after sorting) is referred.
            feat_dims = []
            for i, dims_i in enumerate(dims_by_file):
                if not np.all(dims_i == dims_i[0]):
                    raise RuntimeError(
                        "If padding=True, the feature dimension must be "
                        f"unified: {shape_files[i]}"
                    )
                feat_dims.append(int(dims_i[0]))
        else:
            feat_dims = None

        # Decide batch-sizes: structurally the same accumulate-and-close
        # loop as the stock sampler, plus the max_samples cap. current_sum
        # is an incremental version of the stock non-padding branch's full
        # recompute (sum(np.prod(d[k]) for k in current_batch_keys ...)):
        # since it's a plain sum over the open batch, accumulating equals
        # recomputing.
        batch_sizes = []
        current_count = 0
        current_sum = 0
        for i in range(n):
            current_count += 1
            if padding:
                bins = sum(
                    current_count * int(lengths_by_file[j][i]) * feat_dims[j]
                    for j in range(n_files)
                )
            else:
                current_sum += sum(
                    int(lengths_by_file[j][i]) * int(dims_by_file[j][i])
                    for j in range(n_files)
                )
                bins = current_sum

            close_by_bins = bins > batch_bins and current_count >= min_batch_size
            close_by_cap = (
                max_samples is not None and current_count >= max_samples
            )
            if close_by_bins or close_by_cap:
                batch_sizes.append(current_count)
                current_count = 0
                current_sum = 0

        if current_count != 0 and (not self.drop_last or len(batch_sizes) == 0):
            batch_sizes.append(current_count)

        if len(batch_sizes) == 0:
            # Maybe we can't reach here
            raise RuntimeError("0 batches")

        # If the last batch-size is smaller than minimum batch_size, the
        # samples are redistributed to the other mini-batches.
        if len(batch_sizes) > 1 and batch_sizes[-1] < min_batch_size:
            leftover = batch_sizes.pop(-1)
            if max_samples is None:
                for i in range(leftover):
                    batch_sizes[-(i % len(batch_sizes)) - 1] += 1
            else:
                # Same round-robin redistribution, but max_samples is a hard
                # cap (unlike min_batch_size, which is only a soft target
                # here): skip batches already at the cap, and if a full
                # cycle finds no batch with room, keep whatever is left as
                # its own (possibly undersized) batch rather than push any
                # batch over max_samples. Unreachable in this recipe's
                # configs, which set drop_last: true and so never populate
                # a trailing remainder to redistribute in the first place;
                # guarded here because drop_last=False is still a supported
                # constructor argument.
                i = 0
                stalled = 0
                while leftover > 0 and stalled < len(batch_sizes):
                    target = -(i % len(batch_sizes)) - 1
                    if batch_sizes[target] < max_samples:
                        batch_sizes[target] += 1
                        leftover -= 1
                        stalled = 0
                    else:
                        stalled += 1
                    i += 1
                if leftover > 0:
                    batch_sizes.append(leftover)

        if not self.drop_last:
            # Bug check
            assert sum(batch_sizes) == n, f"{sum(batch_sizes)} != {n}"

        # Set mini-batch
        self.batch_list = []
        idx = 0
        for bs in batch_sizes:
            minibatch_keys = [str(int(k)) for k in sorted_keys[idx : idx + bs]]
            idx += bs

            if sort_in_batch == "descending":
                minibatch_keys.reverse()
            elif sort_in_batch == "ascending":
                # Keys are already sorted in ascending order.
                pass
            else:
                raise ValueError(
                    "sort_in_batch must be ascending"
                    f" or descending: {sort_in_batch}"
                )

            self.batch_list.append(tuple(minibatch_keys))

        if sort_batch == "ascending":
            pass
        elif sort_batch == "descending":
            self.batch_list.reverse()
        else:
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_bins={self.batch_bins}, "
            f"max_samples={self.max_samples}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
