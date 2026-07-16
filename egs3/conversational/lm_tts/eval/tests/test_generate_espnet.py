"""Tests for the espnet-path generation CLI (Task 7): the engine-equivalence
anchor generator that drives BagPiper through the real espnet decode path
(``model.inference(...)``, following ``scripts/gate_generate.py``) rather
than the vLLM server (Task 6).

Two tiers, mirroring ``tests/test_preprocessing_parity.py``'s gating and
Task 6's ``test_generate_vllm.py`` import-hygiene check:

1. Ungated (always runs, no assets): the CLI arg parser rejects a manifest
   invocation missing ``--ids`` - argparse's own ``required=True`` behavior,
   exercised without importing anything heavy - plus an import-hygiene
   check that ``eval.generate_espnet`` never pulls in ``torch``/``espnet``
   merely by being imported (the binding lazy-import constraint: the module
   must be importable, and its parser testable, on a box with no torch/espnet
   installed at all).
2. Asset-gated (skips locally unless ``BAGPIPER_CKPT`` is set, same guard as
   ``test_preprocessing_parity.py``): builds the prompt-only batch for a
   synthetic manifest entry via the real
   ``SpeechLMJobTemplate(config).build_preprocessor().collate_fn`` and checks
   the resulting tensor keys/shape - this only calls ``build_prompt_batch``
   (config + preprocessor), never ``load_bagpiper``'s ~16.9 GB model weights,
   so it is safe and fast on a machine too small to hold the full model.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

RECIPE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

CKPT = os.environ.get("BAGPIPER_CKPT")
CFG = os.environ.get("BAGPIPER_TRAIN_CONFIG") or os.path.join(
    RECIPE_DIR, "conf", "bagpiper_train_config.yaml"
)

_REAL_ASSET_SKIP = pytest.mark.skipif(
    not (CKPT and os.path.exists(CKPT)),
    reason=(
        "set BAGPIPER_CKPT to the BagPiper safetensors shard directory to run "
        "the real prompt-batch construction test (BAGPIPER_TRAIN_CONFIG "
        "defaults to conf/bagpiper_train_config.yaml)"
    ),
)


# ---------------------------------------------------------------------------
# Tier 1: ungated - CLI arg parsing and import hygiene, no heavy deps.
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_missing_ids_raises_system_exit(self):
        from eval.generate_espnet import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--manifest", "manifest.json", "--out-dir", "out"])

    def test_all_required_args_present_parses_ok(self):
        from eval.generate_espnet import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--manifest",
                "manifest.json",
                "--out-dir",
                "out",
                "--ids",
                "ex1,ex2",
            ]
        )
        assert args.manifest == "manifest.json"
        assert args.out_dir == "out"
        assert args.ids == "ex1,ex2"


class TestParseIds:
    def test_splits_on_comma_and_strips_whitespace(self):
        from eval.generate_espnet import parse_ids

        assert parse_ids(" ex1, ex2 ,ex3") == ["ex1", "ex2", "ex3"]

    def test_single_id(self):
        from eval.generate_espnet import parse_ids

        assert parse_ids("ex1") == ["ex1"]


class TestSelectEntries:
    def test_selects_in_requested_order(self):
        from eval.generate_espnet import select_entries

        entries = [
            {"example_id": "a"},
            {"example_id": "b"},
            {"example_id": "c"},
        ]
        selected = select_entries(entries, ["c", "a"])
        assert [e["example_id"] for e in selected] == ["c", "a"]

    def test_missing_id_raises_value_error_naming_it(self):
        from eval.generate_espnet import select_entries

        entries = [{"example_id": "a"}]
        with pytest.raises(ValueError, match="nope"):
            select_entries(entries, ["a", "nope"])


def test_importing_eval_generate_espnet_does_not_load_heavy_deps():
    """Runs the import check in a FRESH subprocess rather than mutating this
    process's ``sys.modules``.

    An earlier version of this test deleted ``torch``/``espnet2``/``numpy``
    from ``sys.modules`` in-process before re-importing - the same pattern
    ``eval/generate_vllm.py``'s equivalent hygiene test uses. That is unsafe
    here: deleting an already-loaded native extension module out from under a
    live process and then letting anything re-import it corrupts the
    process (observed firsthand: a later gated test's ``import torch``
    crashed with "module functions cannot set METH_CLASS or METH_STATIC").
    It is also order-dependent even when made non-destructive (e.g. a
    before/after ``sys.modules`` diff) - ``tests/test_preprocessing_parity.py``
    imports ``torch`` at module scope, and pytest imports every test module
    during collection, before any test body runs; by the time this test's
    body executes in the full suite, torch is already in ``sys.modules`` and
    a before/after diff would find nothing new regardless of what
    ``generate_espnet`` itself imports.

    A subprocess sidesteps both problems: it starts with a clean
    ``sys.modules``, so the check is meaningful regardless of what already
    ran in this process, and there is nothing to corrupt when it exits.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, eval.generate_espnet; "
            "heavy = [m for m in sys.modules "
            "if m.split('.')[0] in ('torch', 'espnet2', 'numpy')]; "
            "assert not heavy, heavy",
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import eval.generate_espnet pulled in heavy deps "
        f"(stdout={result.stdout!r}, stderr={result.stderr!r})"
    )


# ---------------------------------------------------------------------------
# Tier 2: asset-gated - real prompt-batch construction (skips w/o BAGPIPER_CKPT).
# ---------------------------------------------------------------------------


@_REAL_ASSET_SKIP
class TestBuildPromptBatch:
    def test_builds_batch_with_expected_tensor_keys_for_synthetic_entry(self):
        import yaml

        from eval.generate_espnet import build_prompt_batch

        with open(CFG) as f:
            config = yaml.safe_load(f)

        entry = {
            "example_id": "synthetic_1",
            "set": "sft",
            "system": "You are a multi-talker text-to-speech system.",
            "caption": "“hello there” “hi how are you”",
        }

        batch = build_prompt_batch(config, entry)

        assert "seqs" in batch
        seqs = batch["seqs"]
        assert seqs.dim() == 3
        assert seqs.shape[0] == 1
