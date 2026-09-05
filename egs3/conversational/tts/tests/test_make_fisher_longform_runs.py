"""local/make_fisher_longform_runs.py: per-arm configs + 1-GPU sbatch files
for the Fisher long-form arm (gt anchor / Chorus stage-2 chunked / Concat-F5)."""
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from egs3.conversational.tts.local import make_fisher_longform_runs as mk

RECIPE = Path(__file__).resolve().parents[1]


def _gen(tmp_path, **kw):
    return mk.generate(
        recipe=RECIPE, out_conf=tmp_path / "conf", out_jobs=tmp_path / "jobs",
        ckpt="/ckpts/backup_step14199.ckpt", tag="s14199", **kw,
    )


class TestArms:
    def test_three_arms_subset(self, tmp_path):
        ids = tmp_path / "ids.txt"
        ids.write_text("fe_03_00330\nfe_03_00349\n")
        names = _gen(tmp_path, arms=["gt", "chorus", "concat"], ids_file=ids, shards=1)
        assert names == ["fisher_longform_sub_gt_s14199", "fisher_longform_sub_chorus_s14199",
                         "fisher_longform_sub_concat_s14199"]
        confs = {n: OmegaConf.load(tmp_path / "conf" / f"inference_{n}.yaml") for n in names}
        gt, ch, cc = (confs[n] for n in names)
        assert gt.mode == "generate_external_gt" and gt.ckpt is None
        assert ch.mode == "generate_external_chunked" and ch.ckpt == "/ckpts/backup_step14199.ckpt"
        assert ch.chunk.cond_format == "special_tokens" and ch.chunk.target_sec == 45.0
        assert cc.mode == "generate_concat_baseline" and cc.source == "external" and cc.ckpt is None
        assert cc.training_config == "conf/generated/training_covomix2_eval.yaml"
        assert ch.training_config == gt.training_config == "conf/generated/training_allon_eval.yaml"
        assert cc.duration.source == ch.duration.source == "predicted"
        assert cc.duration.rate_prior_chars == 100.0 and cc.duration.scale == 1.048
        assert "chunk" not in cc or cc.chunk.get("cond_format") is None
        for c in (gt, ch, cc):
            assert c.testset.manifest.endswith("fisher-longform-v1/manifest.jsonl")
            assert Path(c.selection.dialogue_ids).read_text().split() == ["fe_03_00330", "fe_03_00349"]
            assert c.selection.shard_count == 1 and c.batching.max_batch_dialogues == 1
        met_cc = OmegaConf.load(tmp_path / "conf" / f"metrics_{names[2]}.yaml")
        met_ch = OmegaConf.load(tmp_path / "conf" / f"metrics_{names[1]}.yaml")
        targets = lambda m: [x.metric._target_.rsplit(".", 1)[1] for x in m.metrics]
        assert "InteractionMetric" in targets(met_ch)
        assert "InteractionMetric" not in targets(met_cc)  # no gt_wav on the concat path
        assert targets(met_cc) == [x for x in targets(met_ch) if x != "InteractionMetric"]
        for n in names:
            raw_i = OmegaConf.to_container(OmegaConf.load(tmp_path / "conf" / f"inference_{n}.yaml"), resolve=False)
            raw_m = OmegaConf.to_container(OmegaConf.load(tmp_path / "conf" / f"metrics_{n}.yaml"), resolve=False)
            assert raw_i["inference_dir"] == raw_m["inference_dir"] == f"${{exp_dir}}/{n}"
            assert raw_m["mode"] == raw_i["mode"]
            sb = (tmp_path / "jobs" / f"run_{n}.sbatch").read_text()
            assert "--gpus-per-node=1" in sb and "PYTHONUNBUFFERED=1" in sb and mk.PY in sb
            assert f"inference_{n}.yaml" in sb and f"metrics_{n}.yaml" in sb
            assert raw_i["training_config"] in sb

    def test_shards_and_walltime(self, tmp_path):
        names = _gen(tmp_path, arms=["chorus"], ids_file=None, shards=3, walltime="03:30:00")
        assert names == [f"fisher_longform_sh{i}of3_chorus_s14199" for i in range(3)]
        for i, n in enumerate(names):
            c = OmegaConf.load(tmp_path / "conf" / f"inference_{n}.yaml")
            assert (c.selection.shard_index, c.selection.shard_count) == (i, 3)
            assert c.selection.dialogue_ids is None
            assert OmegaConf.to_container(c, resolve=False)["inference_dir"] == f"${{exp_dir}}/fisher_longform_chorus_s14199/shard{i}"
            assert "--time=03:30:00" in (tmp_path / "jobs" / f"run_{n}.sbatch").read_text()

    def test_unknown_arm_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            _gen(tmp_path, arms=["reanchor"], ids_file=None, shards=1)


class TestVariants:
    def test_chorus_overrides_and_suffix(self, tmp_path):
        ids = tmp_path / "ids.txt"
        ids.write_text("fe_03_00330\n")
        names = _gen(tmp_path, arms=["chorus"], ids_file=ids, shards=1,
                     overrides={"source": "ground_truth", "scale": "1.6", "cond_prev_sec": "0.0",
                                "cond_prompt_sec": "4.0"}, arm_suffix="_gtdur")
        assert names == ["fisher_longform_sub_chorus_gtdur_s14199"]
        c = OmegaConf.load(tmp_path / "conf" / f"inference_{names[0]}.yaml")
        assert c.duration.source == "ground_truth" and c.duration.scale == 1.6
        assert c.chunk.cond_prev_sec == 0.0 and c.chunk.cond_prompt_sec == 4.0
        assert c.chunk.target_sec == 45.0 and c.sampling.cfg_strength == 3.5  # untouched
        raw = OmegaConf.to_container(c, resolve=False)
        assert raw["inference_dir"] == "${exp_dir}/fisher_longform_sub_chorus_gtdur_s14199"
        assert (tmp_path / "jobs" / f"run_{names[0]}.sbatch").exists()

    def test_overrides_rejected_on_other_arms_and_protected_keys(self, tmp_path):
        with pytest.raises(ValueError):
            _gen(tmp_path, arms=["gt"], ids_file=None, shards=1, overrides={"scale": "1.6"})
        with pytest.raises(ValueError):
            _gen(tmp_path, arms=["chorus"], ids_file=None, shards=1, overrides={"seed": "1"})
        with pytest.raises(KeyError):
            _gen(tmp_path, arms=["chorus"], ids_file=None, shards=1, overrides={"no_such_key": "1"})
