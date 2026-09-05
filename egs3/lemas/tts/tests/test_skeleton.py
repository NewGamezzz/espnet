import importlib


def test_packages_import():
    for name in [
        "egs3.lemas.tts.dataset",
        "egs3.lemas.tts.src",
        "src.system",
        "src.shape",
    ]:
        importlib.import_module(name)


def test_run_stages():
    run = importlib.import_module("run")
    assert run.DEFAULT_STAGES == [
        "create_dataset",
        "create_token_list",
        "create_shape",
        "train",
        "infer",
        "measure",
    ]
