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
