"""A tiny, offline F5-TTS recipe for the template tests.

The recipe is laid out under ``egs3/<corpus_name>/f5tts/`` inside a
temporary directory, the way a real one is, with a freshly initialised model
saved as its checkpoint. Nothing is trained and nothing is downloaded: the
vocoder is replaced by a stand-in.
"""

import shutil
import sys
from importlib import resources

import pytest
import torch
from omegaconf import OmegaConf

import espnet3.parallel.parallel as parallel_module
from espnet3.systems.f5tts.f5tts import F5TTS
from espnet3.systems.f5tts.inference import F5TTSInference

PACKAGE = "egs3.TEMPLATE.f5tts"
TOKENS = ["<blank>", "<unk>", "a", "b", "c", "<space>", "<sos/eos>"]
MODEL_OVERRIDES = {
    "hidden_size": 32,
    "depth": 1,
    "attention_heads": 2,
    "attention_head_size": 16,
    "feed_forward_multiplier": 1,
    "text_embedding_size": 16,
    "convolution_layers": 1,
}


class StubVocos:
    """Stands in for Vocos: exposes ``decode``, upsamples by the hop length."""

    def decode(self, mel):
        return torch.zeros(1, mel.shape[-1] * 256)


@pytest.fixture(autouse=True)
def isolated_process_state(monkeypatch):
    """Keep process-global state from leaking into or out of a test.

    A packed bundle ships a top-level ``src`` package and is imported by
    putting the bundle on ``sys.path``; a ``src`` already imported from
    another directory would be found instead. The parallel config is
    module-global too, and a multi-worker one left by another test would
    start a Dask cluster for the data stages run here.
    """
    monkeypatch.setattr(parallel_module, "parallel_config", None)
    monkeypatch.setattr(sys, "path", list(sys.path))

    def forget_bundled_modules():
        for name in list(sys.modules):
            if name.split(".")[0] in ("src", "dataset"):
                del sys.modules[name]

    forget_bundled_modules()
    yield
    forget_bundled_modules()


@pytest.fixture
def stub_vocoder(monkeypatch):
    monkeypatch.setattr(F5TTSInference, "_load_vocoder", lambda self, path: StubVocos())


def template_path(*parts):
    """Return the path of a file shipped in the template package."""
    return resources.files(PACKAGE).joinpath(*parts)


@pytest.fixture
def recipe_dir(tmp_path, monkeypatch):
    """A trained-looking recipe directory; the working directory is set to it."""
    recipe = tmp_path / "egs3" / "minicorpus" / "f5tts"
    (recipe / "conf").mkdir(parents=True)
    (recipe / "dataset").mkdir()
    (recipe / "dataset" / "__init__.py").write_text("", encoding="utf-8")

    # src/: the template's helpers, copied as its README tells a recipe to.
    (recipe / "src").mkdir()
    for name in ("__init__.py", "inference.py", "app.py"):
        shutil.copy(template_path("src", name), recipe / "src" / name)

    # conf/training.yaml: the template config, kept self-contained, with a
    # model small enough to build in milliseconds.
    training = OmegaConf.load(template_path("conf", "training.yaml"))
    training.model.update(MODEL_OVERRIDES)
    training.trainer.max_steps = 40000
    # Run the data stages in-process: a Dask cluster, which the template's
    # `parallel` block would start, takes longer than a unit test may.
    training.parallel = None
    OmegaConf.save(training, recipe / "conf" / "training.yaml")
    for name, content in {
        "inference.yaml": {"model": {"ode_solver_steps": 2, "seed": 0}},
        "publication.yaml": {},
        "demo.yaml": {},
    }.items():
        OmegaConf.save(OmegaConf.create(content), recipe / "conf" / name)

    # What create_token_list and train would have written.
    token_list = recipe / "data" / "token_list" / "tokens.txt"
    token_list.parent.mkdir(parents=True)
    token_list.write_text("\n".join(TOKENS) + "\n", encoding="utf-8")
    exp_dir = recipe / "exp" / "training"
    exp_dir.mkdir(parents=True)
    model = F5TTS(
        token_list=str(token_list),
        feats_extract_config={
            "fs": 24000,
            "n_fft": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mels": 100,
        },
        **MODEL_OVERRIDES,
    )
    torch.save({"state_dict": model.state_dict()}, exp_dir / "last.ckpt")
    # Files the template's exclude patterns must keep out of the bundle.
    (exp_dir / "step40000.ckpt").write_bytes(b"periodic checkpoint")
    (exp_dir / "train.log").write_text("log", encoding="utf-8")
    (exp_dir / "stats").mkdir()
    (exp_dir / "stats" / "feats_shape").write_text("utt 10,100\n", encoding="utf-8")

    monkeypatch.chdir(recipe)
    return recipe
