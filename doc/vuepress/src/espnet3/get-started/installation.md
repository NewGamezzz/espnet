# Installation

ESPnet3 uses the `espnet` Python package. Start with a plain install, then add
task extras if needed.

## pip

```bash
pip install espnet
pip install "espnet[asr]"
```

## uv

```bash
uv venv .venv
. .venv/bin/activate
uv pip install espnet
uv pip install "espnet[asr]"
```

## pixi

```bash
pixi init
pixi add python=3.10 pip
pixi run pip install espnet
pixi run pip install "espnet[asr]"
```

## Source install

This is the recommended path for recipe work and development.

```bash
git clone https://github.com/espnet/espnet.git
cd espnet/tools
. setup_uv.sh
cd ..
uv pip install -e ".[asr]"
```

## Notes

- CPU-only usage is supported for development and small tests.
- Some recipes need external installers under `tools/installers/`.
- Use [Installation troubleshooting](./installation-troubleshooting.md) when
  import or CUDA checks fail.

