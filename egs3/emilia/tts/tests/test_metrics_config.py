"""Pin the metrics config to what `measure` actually needs.

CRITICAL 2 in the final whole-branch review: `metrics.yaml` targeted
`src.metrics.versa.VersaMetric`, but `egs3/emilia/tts/src/metrics/` did not
exist, and `metrics.yaml` had no test at all -- this file closes that gap
(IMPORTANT 5).

The second half of CRITICAL 2: `metrics.yaml` used to carry a `dataset:
{test: [{name: valid}, {name: test}]}` block copied verbatim from
LibriTTS. `espnet3/systems/base/metric.py::_resolve_test_sets` returns
those names as-is when the block is present, then looks for
`inference_dir/valid/` and `inference_dir/test/` -- but the `infer` stage
(`conf/inference_f5_seedtts.yaml`) writes `test_en` / `test_zh` /
`test_hard`, so nothing would ever match. Verified directly: with the
`dataset:` block absent, `_resolve_test_sets` scans
`metrics_config.inference_dir` and treats every non-hidden subdirectory as
a test set -- exactly what `infer` produces -- so the fix is to delete the
block, not rename its entries. The test below pins that deletion so the
two configs can never drift apart again.
"""

from pathlib import Path

from omegaconf import OmegaConf

from egs3.emilia.tts.tests._dotted_paths import iter_targets, resolve_dotted_path

CONF = Path(__file__).resolve().parents[1] / "conf" / "metrics.yaml"


def _cfg():
    return OmegaConf.load(CONF)


def test_no_hardcoded_dataset_block():
    """A `dataset:` block here can only drift from the `infer` stage's real
    output directories (test_en/test_zh/test_hard); its absence is what
    makes `_resolve_test_sets` auto-scan `inference_dir` instead."""
    cfg = _cfg()
    assert "dataset" not in cfg


def test_every_metric_target_resolves():
    cfg = _cfg()
    raw = OmegaConf.to_container(cfg, resolve=False)
    targets = list(iter_targets(raw))
    assert targets == ["src.metrics.versa.VersaMetric"]
    for target in targets:
        obj = resolve_dotted_path(target)
        assert callable(obj)


def test_versa_metric_keys_match_inference_output():
    """`src/inference.py::build_output` returns
    {"utt_id", "text", "ref", "wav"}; the metric's wav/ref/text keys and
    `inputs` mapping must line up with those, or `measure` KeyErrors on the
    first scp lookup."""
    cfg = _cfg()
    metric_cfg = cfg.metrics[0].metric
    assert metric_cfg.wav_key == "wav"
    assert metric_cfg.ref_key == "ref"
    assert metric_cfg.text_key == "text"
    assert dict(cfg.metrics[0].inputs) == {"wav": "wav", "ref": "ref", "text": "text"}
