"""``local/eval_report.py`` tests: fabricated ``metrics.json`` files (no infer
stage, no real metrics) driven through the report's own functions and its
``main()`` CLI entry point.

Covers: the condition-column / metric-summary-key-row table shape, one
section per metric class, missing keys/conditions rendering as ``-`` rather
than raising, an entirely-missing ``metrics.json`` for one condition, and the
``--label``/``-o`` CLI surface.
"""

from __future__ import annotations

import json
from pathlib import Path

from egs3.conversational.tts.local.eval_report import (
    build_sections,
    load_metrics_json,
    main,
    render_markdown,
)


_ASR_CLASS = "egs3.conversational.tts.src.metrics.asr.ConversationASRMetric"


def _write_metrics_json(inference_dir: Path, payload: dict) -> None:
    inference_dir.mkdir(parents=True, exist_ok=True)
    (inference_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def _asr_payload(**overrides) -> dict:
    summary = {
        "wer_ch_mean": 0.1,
        "wer_ch_worst": 0.2,
        "cpwer": 0.15,
        "swap_rate": 0.0,
        "turn_order_acc": 0.9,
        "kendall_tau": 0.8,
        "turn_count_ratio": 1.0,
    }
    summary.update(overrides)
    return {_ASR_CLASS: {"valid": summary}}


def _quality_payload(utmos: float) -> dict:
    return {
        "egs3.conversational.tts.src.metrics.quality.ChannelQualityMetric": {
            "valid": {"utmos_mean": utmos}
        }
    }


# --------------------------------------------------------------------------- #
# load_metrics_json
# --------------------------------------------------------------------------- #
class TestLoadMetricsJson:
    def test_loads_a_real_file(self, tmp_path):
        _write_metrics_json(tmp_path, _asr_payload())
        loaded = load_metrics_json(tmp_path)
        assert _ASR_CLASS in loaded

    def test_missing_file_returns_empty_dict_not_an_exception(self, tmp_path):
        assert load_metrics_json(tmp_path / "does_not_exist") == {}


# --------------------------------------------------------------------------- #
# build_sections: the condition x metric-class x summary-key pivot
# --------------------------------------------------------------------------- #
class TestBuildSections:
    def test_one_section_per_metric_class_conditions_as_columns(self):
        conditions = [
            ("gt", {**_asr_payload(), **_quality_payload(4.0)}),
            ("generate", {**_asr_payload(wer_ch_mean=0.5), **_quality_payload(2.5)}),
        ]
        sections = build_sections(conditions, test_name="valid")

        assert set(sections) == {"ConversationASRMetric", "ChannelQualityMetric"}
        asr = sections["ConversationASRMetric"]
        assert asr["values"]["gt"]["wer_ch_mean"] == 0.1
        assert asr["values"]["generate"]["wer_ch_mean"] == 0.5
        assert "wer_ch_mean" in asr["keys"]
        assert "cpwer" in asr["keys"]

    def test_missing_condition_for_a_metric_class_leaves_it_out_of_values(self):
        # "resynth" never ran ChannelQualityMetric at all.
        conditions = [
            ("gt", {**_asr_payload(), **_quality_payload(4.0)}),
            ("resynth", _asr_payload()),
        ]
        sections = build_sections(conditions, test_name="valid")
        quality = sections["ChannelQualityMetric"]
        assert "gt" in quality["values"]
        assert "resynth" not in quality["values"]

    def test_missing_test_name_for_a_condition_leaves_it_out_of_values(self):
        payload = _asr_payload()
        # This condition only measured a different test split.
        payload[_ASR_CLASS]["test-other"] = payload[_ASR_CLASS].pop("valid")
        conditions = [("gt", _asr_payload()), ("odd_split", payload)]
        sections = build_sections(conditions, test_name="valid")
        asr = sections["ConversationASRMetric"]
        assert "gt" in asr["values"]
        assert "odd_split" not in asr["values"]

    def test_key_union_preserves_first_seen_order_then_appends_new_keys(self):
        conditions = [
            ("gt", _asr_payload()),
            (
                "generate",
                {
                    "egs3.conversational.tts.src.metrics.asr.ConversationASRMetric": {
                        "valid": {"wer_ch_mean": 0.4, "brand_new_key": 1.0}
                    }
                },
            ),
        ]
        sections = build_sections(conditions, test_name="valid")
        keys = sections["ConversationASRMetric"]["keys"]
        assert keys[0] == "wer_ch_mean"  # first-seen order from "gt"
        assert keys[-1] == "brand_new_key"  # appended from "generate"

    def test_empty_conditions_yields_no_sections(self):
        assert build_sections([], test_name="valid") == {}


# --------------------------------------------------------------------------- #
# render_markdown
# --------------------------------------------------------------------------- #
class TestRenderMarkdown:
    def test_missing_condition_renders_as_dash_not_a_crash(self):
        conditions = [
            ("gt", {**_asr_payload(), **_quality_payload(4.0)}),
            ("resynth", _asr_payload()),  # no ChannelQualityMetric here
        ]
        sections = build_sections(conditions, test_name="valid")
        md = render_markdown(["gt", "resynth"], sections, test_name="valid")

        assert "## ChannelQualityMetric" in md
        # the resynth column exists but its utmos_mean cell is a dash.
        lines = [ln for ln in md.splitlines() if ln.startswith("| utmos_mean")]
        assert len(lines) == 1
        assert "-" in [cell.strip() for cell in lines[0].split("|")]

    def test_conditions_are_columns_in_the_given_order(self):
        conditions = [("pretrained", _asr_payload()), ("finetuned", _asr_payload())]
        sections = build_sections(conditions, test_name="valid")
        md = render_markdown(["pretrained", "finetuned"], sections, test_name="valid")
        header_line = next(ln for ln in md.splitlines() if ln.startswith("| metric"))
        assert header_line.index("pretrained") < header_line.index("finetuned")

    def test_metric_class_gets_its_own_section_heading(self):
        conditions = [("gt", {**_asr_payload(), **_quality_payload(4.0)})]
        sections = build_sections(conditions, test_name="valid")
        md = render_markdown(["gt"], sections, test_name="valid")
        assert "## ConversationASRMetric" in md
        assert "## ChannelQualityMetric" in md

    def test_no_sections_still_produces_a_valid_report_with_no_metrics_note(self):
        md = render_markdown(["gt", "resynth"], {}, test_name="valid")
        assert "no metrics" in md.lower()

    def test_float_values_are_formatted_not_full_precision_floats(self):
        conditions = [("gt", _asr_payload(wer_ch_mean=0.123456789))]
        sections = build_sections(conditions, test_name="valid")
        md = render_markdown(["gt"], sections, test_name="valid")
        assert "0.123456789" not in md
        assert "0.1235" in md


# --------------------------------------------------------------------------- #
# main(): CLI surface, fabricated metrics.json fixtures on disk
# --------------------------------------------------------------------------- #
class TestMainCli:
    def _build_conditions(self, tmp_path):
        gt_dir = tmp_path / "infer_gt"
        gen_dir = tmp_path / "infer_generate"
        _write_metrics_json(gt_dir, {**_asr_payload(), **_quality_payload(4.0)})
        _write_metrics_json(gen_dir, _asr_payload(wer_ch_mean=0.6))
        return gt_dir, gen_dir

    def test_writes_report_to_the_given_output_path(self, tmp_path, capsys):
        gt_dir, gen_dir = self._build_conditions(tmp_path)
        out_path = tmp_path / "report.md"

        code = main(
            [
                "--label",
                "gt",
                str(gt_dir),
                "--label",
                "generate",
                str(gen_dir),
                "-o",
                str(out_path),
            ]
        )

        assert code == 0
        report = out_path.read_text("utf-8")
        assert "## ConversationASRMetric" in report
        assert "gt" in report and "generate" in report

    def test_defaults_to_stdout_when_no_output_path_given(self, tmp_path, capsys):
        gt_dir, gen_dir = self._build_conditions(tmp_path)

        code = main(
            ["--label", "gt", str(gt_dir), "--label", "generate", str(gen_dir)]
        )

        assert code == 0
        captured = capsys.readouterr()
        assert "## ConversationASRMetric" in captured.out

    def test_missing_metrics_json_for_one_condition_does_not_crash(
        self, tmp_path, capsys
    ):
        gt_dir, _gen_dir = self._build_conditions(tmp_path)
        missing_dir = tmp_path / "infer_finetuned"  # never ran measure

        code = main(
            [
                "--label",
                "gt",
                str(gt_dir),
                "--label",
                "finetuned",
                str(missing_dir),
            ]
        )

        assert code == 0
        out = capsys.readouterr().out
        assert "finetuned" in out

    def test_no_labels_is_a_usage_error_not_a_crash(self, capsys):
        code = main([])
        assert code == 2
        captured = capsys.readouterr()
        assert "--label" in captured.out or "--label" in captured.err

    def test_duplicate_label_names_are_a_usage_error(self, tmp_path, capsys):
        # Silently collapsing two same-named conditions into one column would
        # drop the first condition's data with no indication; fail loudly.
        gt_dir, gen_dir = self._build_conditions(tmp_path)
        code = main(
            [
                "--label",
                "pretrained",
                str(gt_dir),
                "--label",
                "pretrained",
                str(gen_dir),
            ]
        )
        assert code == 2
        captured = capsys.readouterr()
        assert "pretrained" in captured.out

    def test_custom_test_name_selects_that_split(self, tmp_path, capsys):
        gt_dir = tmp_path / "infer_gt"
        payload = _asr_payload()
        payload[_ASR_CLASS]["test-clean"] = payload[_ASR_CLASS].pop("valid")
        _write_metrics_json(gt_dir, payload)

        code = main(["--label", "gt", str(gt_dir), "--test-name", "test-clean"])
        assert code == 0
        out = capsys.readouterr().out
        assert "wer_ch_mean" in out
