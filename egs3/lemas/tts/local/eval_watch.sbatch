#!/bin/bash
# Hourly watcher: submits the subset evaluation arm for every backup
# checkpoint of either run that has not been scored yet (local/eval_watch.py),
# then queues itself again one hour later. Stop it with exp/eval_watch/STOP.
# Usage (from the recipe dir on Delta): sbatch local/eval_watch.sbatch
#SBATCH --job-name=lemas_eval_watch
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/eval_watch_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/local/delta_env.sh"
mkdir -p logs exp/eval_watch
LEFT=${WATCH_LEFT:-720}   # hourly checks for a month
if [ "$LEFT" -gt 1 ] && [ ! -f exp/eval_watch/STOP ]; then
  WATCH_LEFT=$((LEFT - 1)) sbatch --begin=now+1hour "$SLURM_SUBMIT_DIR/local/eval_watch.sbatch" >> logs/eval_watch_chain.log
else
  echo "eval watch stopped (WATCH_LEFT=$LEFT)" >> logs/eval_watch_chain.log
fi
$PY local/eval_watch.py
