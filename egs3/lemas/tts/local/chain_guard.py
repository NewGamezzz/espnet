"""Stop a training chain whose links no longer advance the checkpoint.

``submit_train.sbatch`` calls this at the start of every link, before it
queues the successor. The guard compares ``<exp_dir>/last.ckpt`` with what
the previous link saw: espnet3 keeps ``last.ckpt`` as a symlink to
``step<N>.ckpt``, so the link target names the step and no checkpoint is
loaded. After ``max_stalls`` consecutive links without an advance it writes
``<exp_dir>/STOP`` (which the sbatch script honours) and asks the caller to
exit, so that a failure that repeats on every link costs a few links and
not the whole chain budget.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

STOP_EXIT = 3
STATE_FILE = "chain_state.json"


def checkpoint_marker(exp_dir: Path) -> str:
    """Return a string that changes whenever ``last.ckpt`` is rewritten."""
    last = Path(exp_dir) / "last.ckpt"
    if last.is_symlink():
        return os.readlink(last)
    if last.is_file():
        st = last.stat()
        return f"{st.st_size}:{st.st_mtime_ns}"
    return "none"


def check(exp_dir, max_stalls: int = 2) -> int:
    """Record this link's view of the checkpoint and decide whether to run.

    Args:
        exp_dir: Experiment directory (created when missing).
        max_stalls: Consecutive links without a checkpoint advance after
            which the chain stops.

    Returns:
        ``0`` to run the link, ``STOP_EXIT`` when the chain must stop; in
        that case ``<exp_dir>/STOP`` holds the reason.

    Example:
        >>> [check("exp/tag", max_stalls=2) for _ in range(3)]  # no checkpoint
        [0, 0, 3]
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    state_path = exp_dir / STATE_FILE
    marker = checkpoint_marker(exp_dir)
    stalls = 0
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if state["marker"] == marker:
            stalls = int(state["stalls"]) + 1
    state_path.write_text(json.dumps({"marker": marker, "stalls": stalls}))
    if stalls < max_stalls:
        print(f"chain guard: last.ckpt = {marker}, links without advance = {stalls}")
        return 0
    reason = (
        f"no progress: last.ckpt = {marker} after {stalls} consecutive links;"
        " read the newest logs/train_<jobid>.out, then remove this file and"
        f" {STATE_FILE} to restart the chain\n"
    )
    (exp_dir / "STOP").write_text(reason)
    print(f"chain guard: STOP, {reason}", end="")
    return STOP_EXIT


def main(argv=None) -> int:
    """Command line entry: ``chain_guard.py <exp_dir> [--max_stalls N]``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("exp_dir")
    parser.add_argument("--max_stalls", type=int, default=2)
    args = parser.parse_args(argv)
    return check(args.exp_dir, args.max_stalls)


if __name__ == "__main__":
    sys.exit(main())
