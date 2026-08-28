"""DNSMOS session blocklists (design 2026-08-27, stage 2).

cornell2's ``blocklist_{candor,fisher}.json`` list CHANNEL keys under
``"drop"`` (``<uuid>-ch0/-ch1`` for CANDOR, ``fe_03_NNNNN-A/-B`` for Fisher).
A session is blocked when ANY of its channels is listed (whole-conversation
filtering, ratified 2026-08-27).  Applied at dataset load so the manifests
copied from stage 1 stay untouched.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Sequence

from .sessions import SessionRecord

logger = logging.getLogger(__name__)

_SUFFIX_RE = re.compile(r"^(.*)-(?:ch\d+|[AB])$")


def channel_key_to_session(key: str) -> str:
    """Map a blocklist channel key to its session id.

    Args:
        key: A channel key such as ``<uuid>-ch0`` or ``fe_03_00001-A``.

    Returns:
        The session id (the key without its channel suffix).

    Raises:
        ValueError: when ``key`` carries no recognized channel suffix.
    """
    m = _SUFFIX_RE.match(key)
    if m is None:
        raise ValueError(
            f"blocklist key {key!r} has no -chK / -A / -B channel suffix"
        )
    return m.group(1)


def load_blocked_sessions(paths: str | Path | Sequence[str | Path]) -> set[str]:
    """Union of session ids blocked by one or more blocklist JSON files.

    Args:
        paths: One path or a sequence of paths to ``blocklist_*.json``.

    Returns:
        The set of session ids with at least one blocked channel.

    Raises:
        ValueError: when a file has no ``"drop"`` object.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    blocked: set[str] = set()
    for p in paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        drop = data.get("drop")
        if not isinstance(drop, dict):
            raise ValueError(
                f"{p}: expected a 'drop' object mapping channel keys to reasons"
            )
        blocked |= {channel_key_to_session(k) for k in drop}
    return blocked


def apply_session_blocklist(
    sessions: Sequence[SessionRecord],
    blocked: set[str],
    *,
    source: str,
    strict: bool = True,
) -> list[SessionRecord]:
    """Drop blocked sessions and log the tally.

    Blocklists cover a whole corpus while a manifest holds one split of it,
    so unmatched blocked ids are expected; what is NOT expected is a blocklist
    that matches nothing at all (wrong corpus, or ids renamed), which
    ``strict`` turns into an error instead of a silent no-op.

    Args:
        sessions: Session records of one manifest.
        blocked: Output of ``load_blocked_sessions``.
        source: Label for the log line (the manifest path).
        strict: Raise when no blocked id matches any session.

    Returns:
        The kept sessions, order preserved.

    Raises:
        ValueError: in strict mode, when ``blocked`` is non-empty and matches
            no session in ``sessions``.
    """
    ids = {s.session_id for s in sessions}
    if strict and blocked and not (blocked & ids):
        sample = sorted(blocked)[:3]
        raise ValueError(
            f"{source}: none of the {len(blocked)} blocklisted session ids "
            f"(e.g. {sample}) match a session in this manifest; wrong "
            "blocklist for this corpus?"
        )
    kept = [s for s in sessions if s.session_id not in blocked]
    dropped = [s for s in sessions if s.session_id in blocked]
    logger.info(
        "session blocklist %s: kept %d / dropped %d sessions (%.1f h dropped)",
        source,
        len(kept),
        len(dropped),
        sum(s.duration for s in dropped) / 3600.0,
    )
    return kept
