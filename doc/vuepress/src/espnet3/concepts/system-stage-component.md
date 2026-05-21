# System, stage, component

This is the core ESPnet3 mental model.

## System

A `System` is the task entry point. It owns the stage methods and receives the
loaded configs. See `BaseSystem` and `ASRSystem` in `espnet3/systems/`.

## Stage

A stage is one named step in the pipeline, such as:

- `create_dataset`
- `train_tokenizer`
- `collect_stats`
- `train`
- `infer`
- `measure`
- `pack_model`
- `upload_model`
- `pack_demo`
- `upload_demo`

`egs3/TEMPLATE/asr/run.py` defines the default order.

## Component

Components are reusable building blocks used inside stages. Common examples are
dataset loading, dataloader construction, Lightning modules, metrics, and
parallel runners.

## Wiring

`run.py` does four things:

1. parses CLI flags
2. loads `training.yaml`, `inference.yaml`, `metrics.yaml`, `publication.yaml`,
   and `demo.yaml`
3. instantiates a `System`
4. runs the requested stages

