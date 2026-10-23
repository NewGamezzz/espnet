"""Every new backup checkpoint gets its evaluation arm submitted, once.

Thanapat (2026-10-05): "please always submit the job so that I don't need
to ask you again or disconnect the server when I want".
"""

from pathlib import Path

from local.eval_watch import PROBE_RUNS, RUNS, pending_arms, submit_pending


def _run(tmp_path, tag, backups):
    exp = tmp_path / "exp" / tag
    exp.mkdir(parents=True)
    for b in backups:
        (exp / f"backup_step{b}.ckpt").write_bytes(b"x")
    return exp


def test_pending_arms_lists_unevaluated_backups_oldest_first(tmp_path):
    _run(tmp_path, "tag", [50000, 25000])
    arms = pending_arms(tmp_path, [("tag", "conf/train.yaml")])
    assert [(a.ckpt, a.train_config) for a in arms] == [
        ("backup_step25000", "conf/train.yaml"),
        ("backup_step50000", "conf/train.yaml"),
    ]


def test_submit_marks_each_checkpoint_and_never_resubmits(tmp_path):
    exp = _run(tmp_path, "tag", [25000])
    calls = []

    def sbatch(cmd):
        calls.append(cmd)
        return "12345"

    submit_pending(tmp_path, [("tag", "conf/train.yaml")], sbatch)
    assert len(calls) == 1 and "LEMAS_CKPT=backup_step25000" in calls[0]
    assert "conf/inference_lemas_eval_sub50.yaml" in calls[0]
    assert (exp / "eval_submitted" / "backup_step25000.job").read_text() == "12345"
    submit_pending(tmp_path, [("tag", "conf/train.yaml")], sbatch)
    assert len(calls) == 1  # marked, not submitted again
    (exp / "backup_step50000.ckpt").write_bytes(b"x")
    submit_pending(tmp_path, [("tag", "conf/train.yaml")], sbatch)
    assert len(calls) == 2 and "LEMAS_CKPT=backup_step50000" in calls[1]


def test_a_versioned_duplicate_backup_is_ignored(tmp_path):
    # a resume re-fires the epoch end and Lightning writes backup_stepN-v1.ckpt
    exp = _run(tmp_path, "tag", [25000])
    (exp / "backup_step25000-v1.ckpt").write_bytes(b"x")
    assert [a.ckpt for a in pending_arms(tmp_path, [("tag", "c.yaml")])] == [
        "backup_step25000"
    ]


def test_a_run_without_backups_or_without_exp_dir_is_fine(tmp_path):
    assert pending_arms(tmp_path, [("missing", "conf/train.yaml")]) == []


def test_runs_table_watches_the_live_run_under_its_config_tag():
    assert RUNS == [
        (
            "train_f5_base_lemas_dualprompt_gh200_lr1.5e-4_r3",
            "conf/training_f5_base_dualprompt_gh200.yaml",
        )
    ]
    for tag, conf in RUNS:
        assert f"exp_tag: {tag}\n" in Path(conf).read_text()


def test_watch_script_rechains_hourly_and_is_stoppable():
    s = Path("local/eval_watch.sbatch").read_text()
    assert "--partition=cpu" in s and "--time=00:10:00" in s
    assert "--begin=now+1hour" in s and "exp/eval_watch/STOP" in s
    assert "$PY local/eval_watch.py" in s


def test_probes_are_submitted_once_per_backup_beside_the_eval_arms(tmp_path):
    exp = _run(tmp_path, "tag", [5000])
    (exp / "eval_submitted").mkdir()
    (exp / "eval_submitted" / "backup_step5000.job").write_text("1")
    calls = []

    def sbatch(cmd):
        calls.append(cmd)
        return "777"

    runs = [("tag", "conf/diag.yaml")]
    submit_pending(tmp_path, runs, sbatch, kind="probe")
    assert calls == [
        "sbatch --parsable -J probe_tag_step5000 local/probe_text_1gpu.sbatch "
        "conf/diag.yaml backup_step5000"
    ]
    assert (exp / "probe_submitted" / "backup_step5000.job").read_text() == "777"
    submit_pending(tmp_path, runs, sbatch, kind="probe")
    assert len(calls) == 1


def test_probe_runs_cover_the_live_run_and_both_layout_diagnostics():
    assert [conf for _, conf in PROBE_RUNS] == [
        "conf/training_f5_base_dualprompt_gh200.yaml",
        "conf/training_f5_small_en_dualprompt.yaml",
        "conf/training_f5_small_en_stock.yaml",
    ]
    for tag, conf in PROBE_RUNS:
        assert f"exp_tag: {tag}\n" in Path(conf).read_text()


def test_probe_script_probes_the_named_checkpoint_on_one_a100():
    s = Path("local/probe_text_1gpu.sbatch").read_text()
    assert "#SBATCH --partition=gpuA100x4" in s and "#SBATCH --gpus=1" in s
    assert "$PY local/probe_text_use.py --train_config" in s
    assert "--out exp/$TAG/probe_text_$CKPT.json" in s
