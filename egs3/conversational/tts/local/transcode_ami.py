"""Transcode the AMI test-partition headsets into one 4-channel 24 kHz FLAC
per meeting (``<root>/ami_flac/<MID>.flac``).  CPU only, minutes per meeting;
run it on cpu-interactive, not the login node (24 meetings x 4 x ~35 min).

Usage:
    python local/transcode_ami.py --root /work/hdd/bbjs/ttrachu/dataset/ami [--meetings ES2004a ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from egs3.conversational.tts.dataset.preprocessing.ami import (  # noqa: E402
    TEST_MEETINGS,
    transcode_meeting,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument(
        "--flac-dir", type=Path, default=None, help="default <root>/ami_flac"
    )
    ap.add_argument("--meetings", nargs="*", default=list(TEST_MEETINGS))
    args = ap.parse_args(argv)
    flac_dir = args.flac_dir or args.root / "ami_flac"
    for mid in args.meetings:
        rec = transcode_meeting(args.root, mid, flac_dir)
        print(f"{mid}\t{rec.duration:.1f}s\t{rec.audio_relpath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
