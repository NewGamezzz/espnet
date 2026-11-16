#!/bin/bash
# Delta: stage one Emilia language from Hugging Face into the speaker_tree
# layout (local/stage_emilia.py), for a language whose shared copy cannot be
# read. Resumable: finished archives are skipped, so resubmit until the log
# ends with STAGE_COMPLETE (exit 3 and STAGE_INCOMPLETE mean archives remain).
#
# Needs, in local/delta/site_env.sh or the environment:
#   EMILIA_STAGE_DEST   corpus root to write into (about 1.3 TB for ZH, plus
#                       2 GB per worker while an archive is being extracted)
# and a Hugging Face login with access to the gated dataset (hf auth login).
#
# Measured on Delta's /work/hdd for ZH (920 archives, 20.0M utterances): about
# 90 s to download an archive and about 750 s to extract it with 8 workers in
# one job; as 8 array tasks of 16 workers, 898 archives took 2 h 15 min to
# 3 h 04 min per task (peak memory 49 GB), the largest archives (83,000
# utterances) over an hour each. The last run, which only merges, took 5.5 min
# (peak memory 27 GB).
#
# To read the result together with a shared copy of the other languages, link
# them into the same root and point EMILIA_CORPUS_ROOT at it:
#   ln -s <shared root>/emilia/EN "$EMILIA_STAGE_DEST/emilia/EN"
#
# Usage, from the recipe dir (the account is given at submit time):
#   sbatch --account=<cpu account> local/delta/stage_emilia.sbatch [LANG]
#   EMILIA_STAGE_LIMIT=8 sbatch --account=<cpu account> \
#     --partition=cpu-interactive --time=01:00:00 local/delta/stage_emilia.sbatch
# Several nodes at once, then one last run to write the batch jsonl files:
#   sbatch --account=<cpu account> --array=0-7 local/delta/stage_emilia.sbatch
#   sbatch --account=<cpu account> --dependency=afterany:<array job id> \
#     local/delta/stage_emilia.sbatch
#SBATCH --job-name=emilia_stage
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/stage_emilia_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/local/delta/env.sh"
: "${EMILIA_STAGE_DEST:?set EMILIA_STAGE_DEST in local/delta/site_env.sh or the environment}"
STAGE_LANG=${1:-ZH}
# As an array job of N tasks (--array=0-<N-1>), each task stages its own slice.
SLICE=${EMILIA_STAGE_SLICE:-}
if [ -z "$SLICE" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  SLICE="$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"
fi
# Writing ~30,000 small files per archive is the slow part, not the download.
WORKERS=${EMILIA_STAGE_WORKERS:-16}
echo "stage: $STAGE_LANG -> $EMILIA_STAGE_DEST  workers: $WORKERS  limit: ${EMILIA_STAGE_LIMIT:-none}  slice: ${SLICE:-all}"
git -C "$ROOT" log -1 --format='tree: %h %s' || true
date
$PY local/delta/check_imports.py "$ROOT"
rc=0
$PY local/stage_emilia.py --lang "$STAGE_LANG" --dest "$EMILIA_STAGE_DEST" \
  --workers "$WORKERS" ${EMILIA_STAGE_LIMIT:+--limit "$EMILIA_STAGE_LIMIT"} \
  ${SLICE:+--slice "$SLICE"} || rc=$?
date
exit "$rc"
