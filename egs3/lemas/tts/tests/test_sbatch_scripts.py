from pathlib import Path


def test_sbatch_scripts_use_batch_partitions_and_pythonpath():
    for name in ("submit_create_dataset", "submit_train", "run_arm_1gpu"):
        s = Path(f"local/{name}.sbatch").read_text()
        assert "--account=bbjs-delta-" in s
        assert 'source "$SLURM_SUBMIT_DIR/local/delta_env.sh"' in s
        assert "interactive" not in s
        # under sbatch $0 is the spool copy, so the cwd must come from Slurm
        body = s if name != "submit_train" else Path("local/train_link.sh").read_text()
        assert 'cd "$SLURM_SUBMIT_DIR"' in body and 'dirname "$0"' not in s
    assert "--partition=cpu" in Path("local/submit_create_dataset.sbatch").read_text()
    train = Path("local/submit_train.sbatch").read_text()
    assert "--partition=gpuA100x4" in train and '"$0"' not in train
    assert '"$0"' not in Path("local/train_link.sh").read_text()
    assert "--time=01:00:00" in Path("local/run_arm_1gpu.sbatch").read_text()


def test_readme_documents_stages_and_knobs():
    s = Path("README.md").read_text()
    for key in (
        "create_dataset",
        "create_token_list",
        "p_drop_spk",
        "spk_prompt_sec",
        "inference_lemas_eval_spk_only.yaml",
        "lowpass",
    ):
        assert key in s, key


def test_delta_env_sets_phonemizer_and_python_path():
    s = Path("local/delta_env.sh").read_text()
    for key in (
        "export PYTHONPATH=",
        "PHONEMIZER_ESPEAK_LIBRARY=",
        "ESPEAK_DATA_PATH=",
        "PY=/work/nvme/bbjs/ttrachu/pixi_x86",
        "PYLIBS=",
    ):
        assert key in s, key


def test_train_chain_is_bounded_and_stoppable():
    assert "#SBATCH --time=01:00:00" in Path("local/submit_train.sbatch").read_text()
    s = Path("local/train_link.sh").read_text()
    assert "CHAIN_LEFT" in s and "exp/$TAG/STOP" in s


def test_arm_script_takes_the_training_config_and_passes_the_checkpoint():
    # the GH200 run has its own training config (exp_tag); LEMAS_CKPT selects
    # the frozen checkpoint through the environment sbatch exports
    s = Path("local/run_arm_1gpu.sbatch").read_text()
    assert "TRAIN=${3:-conf/training_f5_base_dualprompt.yaml}" in s
    assert '--training_config "$TRAIN"' in s
    assert "LEMAS_CKPT" in s and "versa" in Path("local/delta_env.sh").read_text()


def test_arm_script_can_rerun_only_the_measure_stage():
    # scoring fails independently of synthesis (a missing scorer dependency,
    # a wrong test-set name); the wavs are kept and only measure reruns
    s = Path("local/run_arm_1gpu.sbatch").read_text()
    assert 'STAGES=${STAGES:-"infer measure"}' in s and "--stages $STAGES" in s


def test_one_gpu_chain_for_the_layout_diagnostics():
    s = Path("local/submit_train_1gpu.sbatch").read_text()
    for line in (
        "#SBATCH --partition=gpuA100x4",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --time=01:00:00",
        "SELF=submit_train_1gpu.sbatch",
        'source "$SLURM_SUBMIT_DIR/local/train_link.sh"',
    ):
        assert line in s
    # no default config: each diagnostic names its own
    assert 'CONF_DEFAULT=""' in s


def test_scripts_only_name_stages_that_run_py_defines():
    import ast
    import re

    tree = ast.parse(Path("run.py").read_text())
    stages = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and n.targets[0].id == "DEFAULT_STAGES"
    )
    for p in sorted(Path("local").glob("*.s*")):
        s = p.read_text()
        named = re.findall(r"--stages ([^-\n]+)", s) + re.findall(
            r'STAGES:-"([^"]+)"', s
        )
        for words in named:
            for w in words.split():
                assert w.startswith("$") or w in stages, f"{p}: unknown stage {w}"


def test_batch_fill_job_measures_both_diagnostics_and_both_main_runs_on_cpu():
    s = Path("local/measure_batch_fill.sbatch").read_text()
    assert "--partition=cpu" in s and "--account=bbjs-delta-cpu" in s
    assert 'source "$SLURM_SUBMIT_DIR/local/delta_env.sh"' in s
    for conf in (
        "conf/training_f5_small_en_dualprompt.yaml",
        "conf/training_f5_small_en_stock.yaml",
        "conf/training_f5_base_dualprompt.yaml",
        "conf/training_f5_base_dualprompt_gh200.yaml",
    ):
        assert conf in s and Path(conf).is_file()


def test_chars_manifest_build_script():
    # S0a's character manifests: the create_dataset + create_token_list stages
    # driven by the S0a config, into data_chars/, on a cpu node
    s = Path("local/build_chars_manifest.sbatch").read_text()
    assert "--partition=cpu" in s and 'cd "$SLURM_SUBMIT_DIR"' in s
    assert 'source "$SLURM_SUBMIT_DIR/local/delta_env.sh"' in s
    assert "conf/training_f5_base_s0a_stock_chars.yaml" in s
    assert "create_dataset create_token_list" in s and "data_chars" in s


def test_fill_measurement_covers_the_rows_and_the_base():
    # the base config stays in the list as the regression reference (3,261.45)
    s = Path("local/measure_batch_fill.sbatch").read_text()
    for c in (
        "training_f5_base_s0c_stock_phones.yaml",
        "training_f5_base_s0a_stock_chars.yaml",
        "training_f5_base_d1_spkonly.yaml",
        "training_f5_base_dualprompt.yaml",
    ):
        assert c in s, c


def test_readme_documents_the_baseline_rows():
    s = Path("README.md").read_text()
    for key in (
        "training_f5_base_s0c_stock_phones.yaml",
        "training_f5_base_s0a_stock_chars.yaml",
        "training_f5_base_d1_spkonly.yaml",
        "WANDB_MODE=disabled",
        "build_chars_manifest.sbatch",
    ):
        assert key in s, key


def test_chain_links_and_watcher_requeues_inherit_the_job_name():
    # the baseline rows run from their own worktree next to the production
    # chain; submitted with -J lemas_rows_<row> they must keep that name on
    # every successor so squeue tells them apart from lemas_f5_base and
    # lemas_eval_watch (jobs that must never be touched)
    link = Path("local/train_link.sh").read_text()
    assert '-J "$SLURM_JOB_NAME"' in link
    watch = Path("local/eval_watch.sbatch").read_text()
    assert '-J "$SLURM_JOB_NAME"' in watch


def test_chars_manifest_verify_script_compares_common_rows(tmp_path):
    # job 23191988 built the manifests but exited 141: the acceptance block
    # piped cut into head under pipefail, and its column comparison could
    # only say "differ" once the chars side held its extra rows. The checks
    # live in their own script now, run after the build and on their own.
    import subprocess

    script = Path("local/verify_chars_manifest.sh").resolve()
    ph = tmp_path / "data" / "manifest"
    ch = tmp_path / "data_chars" / "manifest"
    ph.mkdir(parents=True)
    ch.mkdir(parents=True)
    row = "{k}\t{k}.flac\t{t}\t{l}\tsrc\tg\t1.5\tp.jsonl\t7\tgroup\t\t\n"
    ph.joinpath("train.tsv").write_text(
        row.format(k="de_a", t="a b", l="de") + row.format(k="zh_b", t="x y", l="zh")
    )
    ch.joinpath("train.tsv").write_text(
        row.format(k="de_a", t="a <space> b", l="de")
        + row.format(k="zh_b", t="x <space> | y", l="zh")
        + row.format(k="ru_c", t="c", l="ru")
    )
    for d in (ph, ch):
        d.joinpath("valid.tsv").write_text(row.format(k="de_v", t="v", l="de"))
    (tmp_path / "data_chars" / "tokens").mkdir()
    (tmp_path / "data_chars" / "tokens" / "tokens.txt").write_text(
        "<blank>\n<unk>\n<space>\n|\n<spk>\n<lang>\n<de>\n<sos/eos>\n"
    )
    (tmp_path / "data_chars" / "lang_stats.json").write_text("{}\n")
    r = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "train: phones 2  chars 3" in r.stdout
    assert "rows only in phones (train): 0" in r.stdout
    assert "1 ru" in r.stdout and "1 zh" in r.stdout
    assert "identical" in r.stdout
    assert "x <space> | y" in r.stdout
    build = Path("local/build_chars_manifest.sbatch").read_text()
    assert "bash local/verify_chars_manifest.sh" in build
