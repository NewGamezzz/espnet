from pathlib import Path


def test_sbatch_scripts_use_batch_partitions_and_pythonpath():
    for name in ("submit_create_dataset", "submit_train", "run_arm_1gpu"):
        s = Path(f"local/{name}.sbatch").read_text()
        assert "PYTHONPATH=" in s and "--account=bbjs-delta-" in s
        assert "interactive" not in s
    assert "--partition=cpu" in Path("local/submit_create_dataset.sbatch").read_text()
    assert "--partition=gpuA100x4" in Path("local/submit_train.sbatch").read_text()
    assert "--time=01:00:00" in Path("local/run_arm_1gpu.sbatch").read_text()


def test_readme_documents_stages_and_knobs():
    s = Path("README.md").read_text()
    for key in (
        "create_dataset",
        "create_token_list",
        "create_shape",
        "p_drop_spk",
        "spk_prompt_sec",
        "inference_lemas_eval_spk_only.yaml",
        "lowpass",
    ):
        assert key in s, key
