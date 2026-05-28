---
title: ESPnet3 Provider And Runner
author:
  name: "Masao Someki"
date: 2026-05-26
---

# ESPnet3 Provider And Runner

This page is for people who want to implement or modify a parallel workload.

Start with [ESPnet3 Parallel](./index.html) for the high-level flow.
This page focuses on:

- subclass contracts
- writer hooks
- shard-local files
- implementation snippets

## What this page covers

[`EnvironmentProvider`](../../../guide/espnet3/parallel/EnvironmentProvider.html)
and [`BaseRunner`](../../../guide/espnet3/parallel/BaseRunner.html) are the two
main extension points.

- [`EnvironmentProvider`](../../../guide/espnet3/parallel/EnvironmentProvider.html)
  decides how runtime objects are built
- [`BaseRunner`](../../../guide/espnet3/parallel/BaseRunner.html) decides how one
  shard is processed and how outputs are written

Use this page when you are:

- adding a new parallel inference-like task
- changing shard output formats
- debugging writer / merge behavior
- deciding which hook to override

## Read the real contracts first

Read the generated API docs before changing these classes:

- [`espnet3/parallel/env_provider.py`](../../../guide/espnet3/parallel/EnvironmentProvider.html)
- [`espnet3/parallel/base_runner.py`](../../../guide/espnet3/parallel/BaseRunner.html)

Useful concrete examples:

- [`espnet3/parallel/inference_provider.py`](../../../guide/espnet3/parallel/InferenceProvider.html)
- [`espnet3/systems/base/inference_runner.py`](../../../guide/espnet3/systems/InferenceRunner.html)

## EnvironmentProvider

[`EnvironmentProvider`](../../../guide/espnet3/parallel/EnvironmentProvider.html)
has exactly two required methods:

- [`build_env_local()`](../../../guide/espnet3/parallel/EnvironmentProvider.html)
- [`build_worker_setup_fn()`](../../../guide/espnet3/parallel/EnvironmentProvider.html)

<div class='custom-h3'><p>build_env_local<span class="small-bracket">()</span></p></div>


Use this for local execution.

It should build one env dict on the driver and return it directly.

```python
from espnet3.parallel.env_provider import EnvironmentProvider


class MyProvider(EnvironmentProvider):
    def build_env_local(self):
        return {
            "dataset": build_dataset(self.config),
            "model": build_model(self.config),
            "tokenizer": build_tokenizer(self.config),
        }
```

<div class='custom-h3'><p>build_worker_setup_fn<span class="small-bracket">()</span></p></div>


Use this for distributed execution.

It must return a zero-argument function.
That returned function runs once per worker.

```python
class MyProvider(EnvironmentProvider):
    def build_worker_setup_fn(self):
        config = self.config

        def setup():
            return {
                "dataset": build_dataset(config),
                "model": build_model(config),
                "tokenizer": build_tokenizer(config),
            }

        return setup
```

<div class='custom-h3'><p>Local vs worker timing</p></div>


Keep this rule in mind:

- local mode: build once on the driver
- Dask / SLURM mode: build once per worker

Do not build large objects inside `forward()`.

## InferenceProvider

If your task is inference-like, check
[`espnet3/parallel/inference_provider.py`](../../../guide/espnet3/parallel/InferenceProvider.html).

It prebuilds the local env once:

```python
class InferenceProvider(EnvironmentProvider, ABC):
    def __init__(self, config, params=None):
        super().__init__(config)
        self.params = params or {}
        self._local_env = self.build_worker_setup_fn()()

    def build_env_local(self):
        return dict(self._local_env)
```

That pattern is useful when:

- local mode should avoid repeated model loading
- worker setup logic and local setup logic should stay identical

## BaseRunner

[`BaseRunner`](../../../guide/espnet3/parallel/BaseRunner.html) handles:

- batching indices
- shard planning
- resume
- local vs Dask dispatch
- shard-local writer lifecycle
- final merge

The main method you must implement is
[`forward(...)`](../../../guide/espnet3/parallel/BaseRunner.html).

## Minimal runner

```python
from espnet3.parallel.base_runner import BaseRunner


class MyRunner(BaseRunner):
    @staticmethod
    def forward(idx, dataset, model, **env):
        sample = dataset[idx]
        return model(sample)
```

Important constraints:

- keep `forward()` as `@staticmethod`
- do not capture `self`
- env keys are injected by name

## Name-based env injection

If the provider returns:

```python
{
    "dataset": dataset,
    "model": model,
    "device": device,
}
```

then `forward()` can declare:

```python
@staticmethod
def forward(idx, dataset, model, device, **env):
    ...
```

The parameter names must match the env dict keys.

## Batch-aware forward()

If `batch_size` is set on the runner, `idx` may be a batch.

Write `forward()` so both forms are valid when needed.

```python
@staticmethod
def forward(idx, dataset, model, **env):
    if isinstance(idx, int):
        return model(dataset[idx])

    batch = [dataset[i] for i in idx]
    return model(batch)
```

## Which hook should you override?

Use this rule:

- only compute one result in memory: override `forward()`
- write shard-local files incrementally: override writer hooks
- combine shard files on the driver: override `merge()`

The main hooks are documented in the
[`BaseRunner` API reference](../../../guide/espnet3/parallel/BaseRunner.html):

- [`open_writers(shard_dir, **env)`](../../../guide/espnet3/parallel/BaseRunner.html)
- [`write_record(writers, result, state, **env)`](../../../guide/espnet3/parallel/BaseRunner.html)
- [`close_writers(writers)`](../../../guide/espnet3/parallel/BaseRunner.html)
- [`merge(shard_dirs)`](../../../guide/espnet3/parallel/BaseRunner.html)

Lower-level state hooks also exist:

- [`init_state(...)`](../../../guide/espnet3/parallel/BaseRunner.html)
- [`reduce_state(...)`](../../../guide/espnet3/parallel/BaseRunner.html)
- [`finalize_state(...)`](../../../guide/espnet3/parallel/BaseRunner.html)

Most subclasses should not override those lower-level methods first.

## Writer lifecycle

One shard roughly runs like this:

```python
state = cls.init_state(shard_id=shard_id, **env)

for item in items:
    result = cls.forward(item, **env)
    state = cls.reduce_state(state, result, shard_id=shard_id, **env)

cls.finalize_state(state, shard_id=shard_id, **env)
```

And by default:

- `init_state()` creates `split.N/`
- `open_writers()` returns a writer dict
- `write_record()` appends to `state["records"]`
- `close_writers()` closes handles
- a `done` file is written after successful completion

## Minimal file-writing runner

This is the smallest useful pattern when results should be streamed to disk.

```python
from pathlib import Path

from espnet3.parallel.base_runner import BaseRunner


class MyTextRunner(BaseRunner):
    @staticmethod
    def forward(idx, dataset, model, **env):
        sample = dataset[idx]
        hyp = model(sample)
        return {"utt_id": sample["utt_id"], "text": hyp}

    @staticmethod
    def open_writers(shard_dir: Path, **env):
        return {
            "text": (shard_dir / "text").open("w", encoding="utf-8"),
        }

    @staticmethod
    def write_record(writers, result, state, **env):
        writers["text"].write(f'{result["utt_id"]} {result["text"]}\n')

    @staticmethod
    def close_writers(writers):
        for handle in writers.values():
            handle.close()
        return None

    def merge(self, shard_dirs):
        out_dir = self.output_dir / self.shard_subdir if self.shard_subdir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "text").open("w", encoding="utf-8") as out_f:
            for shard_dir in sorted(shard_dirs):
                shard_path = shard_dir / "text"
                if not shard_path.exists():
                    continue
                out_f.write(shard_path.read_text(encoding="utf-8"))
        return {}
```

## State-accumulating runner

If outputs are small, you may not need writers.

The default `write_record()` already appends each result into `state["records"]`.

```python
class MyCollectRunner(BaseRunner):
    @staticmethod
    def forward(idx, dataset, model, **env):
        return {"idx": idx, "score": float(model(dataset[idx]))}

    def merge(self, shard_dirs):
        # read shard-local state or ignore merge if caller only needs side effects
        return None
```

If you keep everything in memory, check carefully whether that still scales for
your dataset size.

## Real example: InferenceRunner

[`espnet3/systems/base/inference_runner.py`](../../../guide/espnet3/systems/InferenceRunner.html)
is the best reference for writer-style parallel output.

Key ideas from that implementation:

- [`open_writers()`](../../../guide/espnet3/systems/InferenceRunner.html)
  prepares shard-local SCP metadata
- [`write_record()`](../../../guide/espnet3/systems/InferenceRunner.html)
  validates one result and writes `<field>.scp`
- [`close_writers()`](../../../guide/espnet3/systems/InferenceRunner.html)
  closes handles and writes `field_keys.txt`
- [`merge()`](../../../guide/espnet3/systems/InferenceRunner.html)
  concatenates shard-local SCP fragments into final outputs

The write path looks like this:

```python
@staticmethod
def open_writers(shard_dir, output_artifacts=None, **env):
    return {
        "shard_dir": shard_dir,
        "artifact_configs": output_artifacts or {},
        "scp_handles": {},
        "field_keys": set(),
    }
```

```python
@staticmethod
def write_record(writers, result, state, idx_key="utt_id", **env):
    for output in _iter_outputs(result):
        idx_value = output[idx_key]
        for field_key in field_keys:
            handle = writers["scp_handles"].get(field_key)
            if handle is None:
                handle = (writers["shard_dir"] / f"{field_key}.scp").open(
                    "w", encoding="utf-8"
                )
                writers["scp_handles"][field_key] = handle
            handle.write(f"{idx_value} {value}\n")
```

And merge is just ordered shard-file concatenation:

```python
for field_key in field_keys:
    concatenate_shard_files(
        ordered_shard_dirs,
        f"{field_key}.scp",
        base_dir / f"{field_key}.scp",
    )
```

See
[`concatenate_shard_files()`](../../../guide/espnet3/parallel/concatenate_shard_files.html)
for the exact file merge behavior.

That pattern is the right choice when:

- each result becomes one or more output files
- per-shard streaming is cheaper than large Python lists
- final outputs should look like normal ESPnet artifacts

## Output directory behavior

If `output_dir` is set, shard-local work is written under:

```text
output_dir/
  shard_subdir/
    manifest.json
    split.0/
    split.1/
    ...
```

The done marker is:

```text
split.N/done
```

Resume behavior depends on that file.

If `resume=True`, completed shards are skipped.

## Common implementation patterns

<div class='custom-h3'><p>Pattern 1: plain local computation</p></div>


Use:

- simple provider
- [`forward()`](../../../guide/espnet3/parallel/BaseRunner.html) only
- no writer hooks

Good for:

- debugging
- small outputs
- tests

<div class='custom-h3'><p>Pattern 2: inference output writing</p></div>


Use:

- provider that builds dataset/model
- [`forward()`](../../../guide/espnet3/parallel/BaseRunner.html) returning normalized dicts
- [`open_writers()`](../../../guide/espnet3/parallel/BaseRunner.html) /
  [`write_record()`](../../../guide/espnet3/parallel/BaseRunner.html) /
  [`close_writers()`](../../../guide/espnet3/parallel/BaseRunner.html)
- [`merge()`](../../../guide/espnet3/parallel/BaseRunner.html) that assembles final files

Good for:

- SCP outputs
- JSONL fragments
- per-utterance artifacts

<div class='custom-h3'><p>Pattern 3: worker-local initialization</p></div>


Use:

- [`build_worker_setup_fn()`](../../../guide/espnet3/parallel/EnvironmentProvider.html)
  to construct heavy objects on each worker
- env injection by name into `forward()`

Good for:

- GPU models
- datasets with file handles
- large tokenizer/model objects

## Common mistakes

<div class='custom-h3'><p>Capturing self in forward<span class="small-bracket">()</span></p></div>


Do not do this:

```python
class BadRunner(BaseRunner):
    def forward(self, idx):  # wrong
        ...
```

Use:

```python
class GoodRunner(BaseRunner):
    @staticmethod
    def forward(idx, dataset, model, **env):
        ...
```

<div class='custom-h3'><p>Rebuilding the model inside forward<span class="small-bracket">()</span></p></div>


Do not load a checkpoint per item.

Build it in the provider.

<div class='custom-h3'><p>Returning the env instead of a setup function</p></div>


Wrong:

```python
def build_worker_setup_fn(self):
    return {"dataset": ..., "model": ...}
```

Correct:

```python
def build_worker_setup_fn(self):
    def setup():
        return {"dataset": ..., "model": ...}
    return setup
```

<div class='custom-h3'><p>Using mismatched env names</p></div>


Wrong:

```python
return {"ds": dataset, "net": model}
```

with

```python
def forward(idx, dataset, model, **env):
    ...
```

Correct the names or read from `**env` explicitly.

<div class='custom-h3'><p>Forgetting shard merge semantics</p></div>


If your subclass writes shard-local files, but `merge()` does nothing, the
final outputs stay split across `split.N/`.

That may be fine for debugging.
It is usually wrong for stage-facing outputs.

## Practical debugging checklist

When a new runner does not behave correctly, check these first:

1. `forward()` is `@staticmethod`
2. provider env keys match `forward()` parameter names
3. `output_dir` is set when using writer hooks
4. shard directories contain expected files
5. `done` is written only after successful completion
6. `merge()` reads shards in stable order
7. `resume=True` is not hiding stale shard outputs during debugging

## See also

<DocCards :cols="3">
  <DocCard
    title="ESPnet3 Parallel"
    desc="Return to the high-level parallel execution overview."
    icon="tabler:route"
    href="./index.html"
  />
  <DocCard
    title="Inference Provider"
    desc="See the inference-stage provider pattern and YAML wiring."
    icon="tabler:cpu"
    href="./inference_provider.html"
  />
  <DocCard
    title="Parallel Config"
    desc="Review local, local GPU, and cluster backend settings."
    icon="tabler:settings"
    href="../config/parallel.html"
  />
  <DocCard
    title="EnvironmentProvider API"
    desc="Read the generated contract for local and worker env setup."
    icon="tabler:book"
    href="../../../guide/espnet3/parallel/EnvironmentProvider.html"
  />
  <DocCard
    title="BaseRunner API"
    desc="Read the generated contract for forward, writer hooks, and merge."
    icon="tabler:book"
    href="../../../guide/espnet3/parallel/BaseRunner.html"
  />
  <DocCard
    title="InferenceRunner API"
    desc="Inspect the writer-style runner used by base inference."
    icon="tabler:file-code"
    href="../../../guide/espnet3/systems/InferenceRunner.html"
  />
</DocCards>
