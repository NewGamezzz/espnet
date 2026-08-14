# ESPnet3 F5-TTS Base recipe: Emilia EN+ZH

Reproduces F5-TTS **Base** (336M params, arXiv:2410.06885) on the full
**Emilia EN+ZH** corpus (~37M utterances, ~92k hours), on PSC Bridges-2.
Evaluated on Seed-TTS test-en / test-zh / test-hard.

Design doc: `Design - F5-TTS Base on Emilia.md` (vault, project ESPnet3
F5-TTS). This README covers only what an operator needs to run the recipe;
the design doc has the full rationale.

## Corpus location and ownership

The corpus is already staged, read-only, at
`/ocean/projects/cis210027p/ttrachu/emilia_dataset/raw/`.
It is owned by another user (`dsharma4`) at the parent level; `ttrachu`'s
access is a readable copy under `raw/emilia/{EN,ZH}/<shard>/`, one flat
directory per shard holding `<utt_id>.mp3` + `<utt_id>.json` pairs.

**Nothing in this recipe may write into `emilia_dataset/`.**
All recipe outputs (manifests, shape files, checkpoints, logs) go to the
recipe's own `data/` and `exp/` directories, resolved from `recipe_dir` in
the training config, never into the staged corpus tree.
A second, separate copy of Emilia also exists under `dsharma4/f5tts_data`
on the same shared allocation; it is not ours to write into or reclaim
space from either.

## The staged `processed/` manifests: do not reuse them

`emilia_dataset/processed/{EN,ZH}/{train.csv,val.csv,duration.json}` looks
like a ready-made manifest, but it has three independent, verified defects
that make it unusable as-is. This recipe rebuilds manifests from the raw
JSON sidecars instead (`local/EmiliaBuilder`, stage `create_dataset`).
Do not shortcut back to `processed/`:

1. **Permissions.** Every audio path inside these CSVs points at
   `/ocean/projects/cis210027p/dsharma4/f5tts_data/raw/emilia/...`, which
   returns **Permission denied** for `ttrachu`. The readable tree is
   `ttrachu/emilia_dataset/raw/emilia/`, which these CSVs never reference.
2. **The speaker blocklist never fired.** The staging script matched
   `obj["id"]` (a 4-field, per-*window* utterance id, e.g.
   `EN_B00000_S00000_W000000`) against blocklists that hold 3-field
   *speaker* ids (`EN_B00000_S00000`). A 4-field string can never equal a
   3-field one, so the match condition was always false and all 77
   blocklisted speakers (71 EN, 6 ZH) passed through unfiltered.
3. **`duration.json` is not row-aligned with `train.csv`.** Durations were
   accumulated in batch-directory order, but the rows were shuffled
   afterward before the train/val split, so `duration.json[i]` does not
   describe `train.csv` row `i`. The element counts happen to match, which
   makes the file look aligned when it is not. Verified empirically by
   reading each row's true duration from its own JSON sidecar, e.g. row 0
   is truly `EN_B00012_S06221_W000069` at 6.791 s, while
   `duration.json[0]` reports 4.295 s.

## Stage list

| Stage | What it does | Present? |
|---|---|---|
| `create_dataset` | `EmiliaBuilder` walks the 2,060 raw shard dirs, reads each JSON sidecar, applies upstream F5's filters (blocklist fixed per above) and the 0.3-30 s duration bound, writes merged TSV manifests. | Yes |
| `remove_long_short` | LibriTTS's stage that filters by re-reading audio headers. | **Dropped.** Duration already comes from the JSON; re-reading 37M audio headers to get a number the corpus already tells you is strictly worse. |
| `create_token_list` | LibriTTS's stage that derives a token list from the corpus. | **Dropped.** This recipe uses F5's own shipped `Emilia_ZH_EN_pinyin/vocab.txt` (`downloads/vocab.txt`, 2545 tokens) instead of a corpus-derived vocab, so the trained checkpoint stays token-list-compatible with the official F5TTS_Base release (this is what makes an anchor run against the released weights possible later). |
| `create_shape` | Synthesizes `feats_shape` analytically from the builder's own `duration` field (`1 + duration * sample_rate // hop_length`), instead of decoding audio. | **New**, replaces `collect_stats`. |
| `collect_stats` | espnet2's stage that decodes every utterance and runs `feats_extract` to get shapes plus mean/variance stats. | **Dropped**, replaced by `create_shape`. `normalize: null` in the training config means no mean/variance stats are needed either, so nothing else is lost. Avoids ~37M mp3 decodes plus mel extraction on a stage that would otherwise run once over the whole corpus. |
| `train` | Standard espnet3 training loop, F5-TTS Base architecture and Table 9 batch size. | Yes |
| `infer` | Seed-TTS test-en / test-zh / test-hard, LibriSpeech-PC test-clean. | Yes |
| `measure` | Same VERSA metrics stack as the LibriTTS recipe. | Yes |

## Batch arithmetic

The paper's Base run is 8x A100 at 38,400 frames/GPU with no gradient
accumulation: **307,200 mel frames per optimizer update**.
The `numel`/`numel_array` batch sampler counts `T * D` (D = 100 mel
channels), so:

```
batch_bins = 30,720,000 / (accumulate_grad_batches * num_device)
```

The production config (`conf/training_f5_tts_base.yaml`) uses
`batch_bins: 480000`, `accumulate_grad_batches: 8`, `num_device: 8`:

```
480000 * 8 * 8 / 100 == 307200   # exact
```

**To re-solve for a different GPU count**, pick `accumulate_grad_batches`
and `num_device` for the new hardware, then solve for `batch_bins`.
Example, 4 GPUs instead of 8 (same accumulation):

```
batch_bins = 30,720,000 / (8 * 4) = 960000
```

Two constraints that don't change with GPU count:

- `min_batch_size` must stay `>= num_device` (the world_size floor
  `espnet3/components/data/dataloader.py:281` enforces; the smaller the
  world_size, the more headroom there is here).
- `max_samples: 64` (upstream's hard per-batch sample cap, `numel_array`'s
  addition over stock `numel`) is **independent of GPU count** and does
  not need re-solving; it bounds the short end of the length-sorted batch
  order regardless of how many GPUs are training.

## Precision: fp32, no exceptions

`conf/training_f5_tts_base.yaml` sets no `trainer.precision` key, which
means fp32.
Bridges-2 GPU nodes are V100-32, which has **no bf16 support**, and every
verified result from the LibriTTS work on this branch is fp32.
fp16 is a separate, opt-in throughput experiment with its own decision
point (raised via the GPU smoke's peak-memory measurement below, if fp32
turns out not to fit); it is not a dependency of getting this recipe
launch-ready.

## Running the GPU smoke

The smoke (`local/submit_smoke.sbatch` +
`conf/training_f5_tts_base_smoke.yaml`) produces the single number the
whole campaign budget depends on: updates per hour at production
`batch_bins`.

The smoke config is a **full standalone copy** of the production Base
config with exactly five things changed: `max_steps: 200`,
`val_check_interval: 100`, `limit_val_batches: 5`, `exp_tag:
smoke_base_emilia`, and `trainer.logger` swapped from `WandbLogger` to an
offline `CSVLogger`.
Everything else -- especially `batch_bins`, `accumulate_grad_batches` and
`num_device` -- is untouched, because the entire point is to measure
production throughput, not a smaller run.
`tests/test_training_config.py::test_smoke_config_only_differs_from_base_in_five_places`
enforces this; a config merge (`load_and_merge_config`) never inherits
from another recipe config, only from the near-empty `egs3.TEMPLATE.tts`
default, so a thin override file would silently fall back to template
defaults instead of Base's real settings.

`logger: false` can never work here: `default_callbacks.py` installs a
`LearningRateMonitor()` unconditionally regardless of what's in
`trainer.callbacks`, and Lightning refuses that callback without a real
logger. Use the offline `CSVLogger`, never `false`.

```bash
mkdir -p logs                       # Slurm will not create this for you
sbatch local/submit_smoke.sbatch
```

Poll on a log marker, not `pgrep` (an ssh-wrapped `pgrep -f` matches its
own `bash -c` wrapper and reports "alive" forever):

```bash
ssh psc "grep -q 'Training completed' logs/smoke_<jobid>.out && echo DONE || echo RUN"
```

Four measurements to pull from the completed job log (full detail and
exact grep targets are in `local/submit_smoke.sbatch`'s header comment):

1. **Updates/hour** at production `batch_bins`, extrapolated to the
   1,200,000-update paper target.
2. **Peak GPU memory** per rank, confirming Base fits on V100-32 in fp32.
3. **Peak host RSS and wall time of sampler construction.** This is what
   Task 12's `NumElementsArraySampler` (already shipped, `type:
   numel_array` in both configs) exists to keep low across a months-long
   chained run that restarts often.
4. **mp3 decode throughput** as a fraction of step time, read straight off
   `MetricsLogger`'s `iter_time` vs. `forward_time`/`backward_time`/
   `optim_step_time` split. If the loader is starving the GPU, the fix is
   to raise `dataloader.train.num_workers` (currently 4) -- **but** the
   smoke's own `--ntasks-per-node=8 --cpus-per-task=5` already uses all 40
   cores of the node (8 ranks x 5 CPUs), so raising `num_workers` requires
   also raising `--cpus-per-task`, which reduces how many ranks fit on the
   node. It is not a free lever on a single-node job.

**Corrected success criterion for `numel_array`/`max_samples: 64` (Task
12):** the plan's original wording -- "confirm peak RSS fell and
`len(batches)` is unchanged from the stock run" -- is unsatisfiable as
written.
With `max_samples: 64` active, `len(batches)` **must** increase, because
the cap splits the short end of the length-sorted order into more, smaller
batches.
The corrected form, which is what `local/submit_smoke.sbatch`'s header and
the tests actually check for:

> RSS fell; batch count increased only at the short end; no batch exceeds
> 64.

## Chained training and the quota guard

`local/submit_train.sbatch` checks the chain-depth cap, then runs
`local/quota_guard.sh`, then loads the config and runs the completion
check, then **queues its own successor via `--dependency=afterany`
BEFORE training starts**, and only then runs `srun`.
`fit.ckpt_path: ${exp_dir}/last.ckpt` (already in the production config)
is what makes each hop resume where the last left off.

```bash
mkdir -p logs
sbatch local/submit_train.sbatch
```

**Why the successor is queued before training, not after:** the first
version of this script queued the next hop as its last statement, after
`srun` returned.
That is broken for the primary scenario chaining exists for: when
Bridges-2 hits the `--time` limit, Slurm delivers `SIGTERM` to the job's
entire process tree, including this parent script, not just the `srun`
step, and bash's default disposition for `SIGTERM` with no trap is
immediate termination -- so a trailing resubmit line never runs and the
chain silently stops at every walltime boundary, noticeable only because
the queue goes empty.
Queuing the successor first means it is already in the queue regardless
of how this job later dies: clean short-of-`max_steps` exit, walltime
`SIGTERM`, node failure, preemption, or manual `scancel`.

**The quota guard and chain-depth cap run before the successor is
queued**, so a sustained problem (full disk, a chain that's run away)
actually halts further submission instead of queuing through it; a
completion check (below) also runs before queuing, so a finished run
does not queue one more job that immediately exits, forever.
As of 2026-08-13 the project allocation
(`/ocean/projects/cis210027p`) is at **976.56 TiB quota, 956.12 TiB used,
20.4 TiB free (97.9% full)**, shared with other group members, and inodes
are a non-issue (4.6B free).
This 97.9%-full condition is exactly what truncated `step96048.ckpt` and
`step96600.ckpt` in July 2026; a Base checkpoint is roughly 5 GB and this
run writes many.
`local/quota_guard.sh` queries `my_quotas`, selects the **project** block
specifically (it prints a home block in GiB before a project block in TiB;
comparing the wrong block or the wrong unit yields a nonsense number), and
**fails closed**: any parse failure, missing block, or unit mismatch
aborts the job rather than letting it proceed on a guess. Threshold
defaults to 2 TiB free, overridable via `MIN_FREE_TIB`. (It computes the
free-space comparison with `awk`, not the brief's original `bc`, to avoid
a dependency on `bc` being present on the compute node.)

`--dependency=afterany` fires on success, failure and cancellation alike
-- an unguarded chain would resubmit forever if a job died instantly (bad
config, a quota-guard abort, a first-batch OOM).
`submit_train.sbatch` guards against that two ways, both checked before
the successor is queued: a `CHAIN_DEPTH` counter (exported across hops via
`sbatch --export=ALL,CHAIN_DEPTH=...`) that refuses to queue a further hop
past `MAX_CHAIN_DEPTH` (default 500), and a completion check that exits 0
without queuing a successor if `last.ckpt`'s step count already reached
`trainer.max_steps`, or a `STOP` sentinel file exists in `exp_dir`.

**`scancel` behaviour, since `afterany` fires on cancellation too:** once
a job has passed the queuing step, its successor is already sitting in
the queue.
Cancelling the currently running job with `scancel <jobid>` does **not**
cancel that already-queued successor -- this is a direct consequence of
using `afterany`, which is also what lets the chain survive preemption
(which looks like a cancellation to Slurm).
To actually stop the chain:

- **Preferred:** touch `${exp_dir}/STOP`. The already-queued successor
  will start, hit the completion check before training or queuing
  anything further, and exit 0. A brief, harmless job start is the
  accepted cost of guaranteed resume-after-preemption elsewhere in the
  chain.
- **Alternative:** also `scancel` the queued successor's job id (find it
  with `squeue -u $USER`) if even that brief start is undesirable.

### Rank-0 OOM masquerading as an NCCL timeout

The plan describes a diagnostic that "wraps the training step in a
try/except that logs `torch.cuda.memory_summary()` and calls
`dist.abort()`".
**Deviation from the plan, recorded here:** `torch.distributed` has no
`abort()` in the installed torch (2.12.0; verified directly), and the
actual failure mode -- documented in `[[psc-conversational-f5-oom-hang]]`
-- is a launcher-level problem, not a training-step-level one.
When rank 0 hits a CUDA OOM, it dies with a normal exception (whose own
message already reports allocated/reserved/capacity, so re-deriving that
via `memory_summary()` buys little); the other ranks are left sitting in
`lightning_module.py`'s `_sync2skip` `torch.distributed.barrier()`, waiting
for the dead rank, until the 30-minute NCCL watchdog finally times out.
**The visible symptom is an ALLREDUCE timeout, not an OOM.**

So: if a training job on PSC dies at the first batch (or any batch) with
what looks like a collective/watchdog timeout, **look for a rank crash
first**, before assuming a network or NCCL problem.
`submit_train.sbatch` mitigates this at the launcher level rather than in
Python: `srun --kill-on-bad-exit=1` tears down the whole step as soon as
one rank exits nonzero instead of leaving siblings in the barrier, and
`TORCH_NCCL_ASYNC_ERROR_HANDLING=1` makes the watchdog itself abort
promptly on a collective failure rather than hang.
For live triage, the memory note also points at `py-spy` in the recipe
venv and `/jet/home/ttrachu/collect_stacks.sh <jobid>` (dumps stacks
across all ranks via node ssh).

## Checkpoint retention trap

`espnet3/components/callbacks/default_callbacks.py`'s
`get_default_callbacks` saves `last.ckpt` as a **symlink**
(`save_last="link"`) pointing at the most recent `step{step}.ckpt`, with
`save_weights_only=False` -- this is the only checkpoint carrying
optimizer, scheduler and EMA state.
The best-K checkpoints (`best_model_criterion` in the training config) are
saved with `save_weights_only=True`.

**If `last.ckpt`'s target file corrupts or truncates (exactly the July
2026 failure mode from the quota section above), training cannot resume
as-is: no best-K checkpoint can substitute, because none of them carry
optimizer/scheduler/EMA state.**
`submit_train.sbatch`'s resume path (`fit.ckpt_path: ${exp_dir}/last.ckpt`)
depends entirely on this one file staying intact; keep the quota guard
enabled and do not disable it to "just get one more job through".

## Accepted SIM deviation

VERSA (the metrics stack `measure` uses) cannot load the official
UniSpeech `wavlm_large_finetune.pth` checkpoint, so paper-comparable SIM
still requires the external official scorer rather than VERSA's own WavLM
SIM implementation.
This was accepted and documented on 2026-08-05 for the LibriTTS work and
carries over unchanged here; it is not re-litigated by this recipe.

## Recipe stages, end to end

```bash
# 1) Build manifests from the raw corpus (once; ~37M JSON sidecars, runs
#    under `parallel:`, one task per shard)
python run.py --stages create_dataset --training_config conf/training_f5_tts_base.yaml

# 2) Synthesize feats_shape analytically (no audio decode)
python run.py --stages create_shape --training_config conf/training_f5_tts_base.yaml

# 3) GPU smoke (see above) -- confirms throughput before committing to the
#    full run
sbatch local/submit_smoke.sbatch

# 4) Train (chained across walltime; see above)
sbatch local/submit_train.sbatch

# 5) Seed-TTS inference
python run.py --stages infer \
    --training_config conf/training_f5_tts_base.yaml \
    --inference_config conf/inference_f5_seedtts.yaml

# 6) Metrics
python run.py --stages measure \
    --training_config conf/training_f5_tts_base.yaml \
    --inference_config conf/inference_f5_seedtts.yaml \
    --metrics_config conf/metrics.yaml
```

`create_dataset` and `create_shape` require real access to the staged
corpus and are not run at authoring time; see the design doc's open probes
for what's still unverified end to end (`TTSTask.build_model` against a
token list with no `<unk>`/`<sos/eos>`, resolved in Task 7's probes; the
duration-distribution short tail; and the real updates/hour number, which
only the GPU smoke can produce).
