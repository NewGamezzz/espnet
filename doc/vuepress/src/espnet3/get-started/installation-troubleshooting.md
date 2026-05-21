# Installation troubleshooting

## Basic checks

```bash
python -c "import espnet3"
python -c "from espnet3.systems.asr.system import ASRSystem"
```

## Common failures

- CUDA or PyTorch mismatch: reinstall PyTorch first, then reinstall `espnet`.
- Missing optional package: install the matching extra such as `espnet[asr]`.
- Gradio missing for demo work: install `gradio`.
- Hugging Face upload missing tools: install `huggingface_hub` and log in with
  `huggingface-cli login`.
- WSL or Windows path confusion: keep recipe work inside one filesystem root.

## When to report an issue

Open an issue after you can show:

- your install command
- Python version
- PyTorch version
- the full traceback

