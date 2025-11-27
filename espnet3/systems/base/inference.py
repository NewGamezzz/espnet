import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from espnet3.parallel.base_runner import BaseRunner
from espnet3.parallel.parallel import set_parallel
from espnet3.systems.base.inference_provider import (
    InferenceProvider as BaseInferenceProvider,
)


class InferenceProvider(BaseInferenceProvider):
    """
    Provider that builds dataset/model for simple single-GPU inference.

    Args:
        config (DictConfig): Hydra configuration containing ``dataset`` organizer,
            ``model`` factory, and ``test_set`` selection.

    Example:
        >>> provider = InferenceProvider(cfg)  # doctest: +SKIP
        >>> env = provider.build_env_local()  # doctest: +SKIP
        >>> {"dataset", "model"} <= set(env.keys())  # doctest: +SKIP
        True
    """

    @staticmethod
    def build_dataset(config):
        """
        Instantiate the requested test split from the dataset organizer.

        Args:
            config (DictConfig): Configuration with a ``dataset`` organizer and
                ``test_set`` name.

        Returns:
            Any: Dataset object corresponding to ``config.test_set``.

        Example:
            >>> ds = InferenceProvider.build_dataset(cfg)  # doctest: +SKIP
            >>> len(ds)  # doctest: +SKIP
            100
        """
        # config includes test dataset
        organizer = instantiate(config.dataset)
        test_set = config.test_set
        return organizer.test[test_set]

    @staticmethod
    def build_model(config):
        """
        Load the speech-to-text model and place it on the available device.

        Args:
            config (DictConfig): Configuration with a ``model`` instantiation
                target that accepts a ``device`` keyword.

        Returns:
            Any: Inference-ready model instance.

        Example:
            >>> model = InferenceProvider.build_model(cfg)  # doctest: +SKIP
            >>> callable(model)  # doctest: +SKIP
            True
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            device_id = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
            device = f"cuda:{device_id}"

        # config includes model
        model = instantiate(
            config.model, device=device
        )  # In this recipe we assume this to be espnet2.bin.asr_inference.Speech2Text
        return model


class InferenceRunner(BaseRunner):
    """
    Runner that produces hypotheses and references for each sample.

    Example:
        >>> runner = InferenceRunner(provider)  # doctest: +SKIP
        >>> outputs = runner([0, 1])  # doctest: +SKIP
    """

    @staticmethod
    def forward(idx, dataset=None, model=None, **kwargs):
        """
        Run inference for a single dataset index.

        Args:
            idx (int): Sample index to decode.
            dataset: Dataset instance providing ``speech`` and ``text`` fields.
            model: Inference model supporting callable interface and tokenizer.
            **kwargs: Unused additional arguments.

        Returns:
            Dict[str, Any]: Mapping with ``idx``, ``hyp`` (decoded text),
            and ``ref`` (reference text).

        Example:
            >>> InferenceRunner.forward(0, dataset=ds, model=asr_model)  # doctest: +SKIP
            {'idx': 0, 'hyp': '...', 'ref': '...'}
        """
        data = dataset[idx]
        speech = data["speech"]
        hyp = model(speech)[0][0]
        ref = model.tokenizer.tokens2text(model.converter.ids2tokens(data["text"]))
        return {"idx": idx, "hyp": hyp, "ref": ref}


def inference(config: DictConfig):
    """
    Decode all configured test sets and write hyp/ref SCP files.

    Args:
        config (DictConfig): Configuration containing dataset/test set
            definitions, parallel settings, decode directory, and model
            parameters.

    Returns:
        None: Produces ``hyp.scp`` and ``ref.scp`` files under
        ``config.decode_dir`` for each test set.

    Example:
        >>> inference(cfg)  # doctest: +SKIP
    """
    set_parallel(config.parallel)

    test_sets = [test_set.name for test_set in config.dataset.test]
    assert len(test_sets) > 0, "No test set found in dataset"
    assert len(test_sets) == len(set(test_sets)), "Duplicate test key found."

    for test_name in test_sets:
        print(f"===> Processing {test_name}")
        config.test_set = test_name
        provider = InferenceProvider(config)
        runner = InferenceRunner(
            provider=provider,
            async_mode=False,
        )
        dataset_length = len(provider.build_dataset(config))
        print(f"===> Processing {dataset_length} samples..")
        out = runner(list(range(dataset_length)))

        # create scp files
        (Path(config.decode_dir) / test_name).mkdir(parents=True, exist_ok=True)
        with open(Path(config.decode_dir) / test_name / "ref.scp", "w") as f:
            f.write("\n".join([f"{result['idx']} {result['ref']}" for result in out]))

        with open(Path(config.decode_dir) / test_name / "hyp.scp", "w") as f:
            f.write("\n".join([f"{result['idx']} {result['hyp']}" for result in out]))
