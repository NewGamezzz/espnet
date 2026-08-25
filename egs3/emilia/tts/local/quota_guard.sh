#!/usr/bin/env bash
# Refuse to submit when /ocean project free space is below the threshold.
#
# The project allocation was at 97.9% (20.4 TiB free of 976.56 TiB) on
# 2026-08-13, and it is shared. A disk-full condition truncated
# step96048.ckpt and step96600.ckpt in July 2026; a Base checkpoint is
# about 5 GB and this run writes many.
#
# `my_quotas` prints TWO blocks, home first, then project, and their units
# differ: home is GiB, project is TiB.
#
#   The quota for home directory /jet/home/ttrachu
#   Storage quota: 25.00GiB
#    Storage used: 19.80GiB
#     Inode quota: 0
#     Inodes used: 480,858
#
#   The quota for project directory /ocean/projects/cis210027p
#   Storage quota: 976.56TiB
#    Storage used: 956.12TiB
#     Inode quota: 6,070,000,000
#     Inodes used: 1,438,880,301
#
# This script selects the PROJECT block specifically (a naive `tr -d 'TiB'`
# or a global grep on the wrong line silently mixes GiB and TiB numbers) and
# fails CLOSED: any parse failure, missing block, or unexpected unit aborts
# rather than allowing submission to proceed on a guess.
#
# DEVIATION from task-13-brief.md's example: arithmetic below uses `awk`
# throughout instead of `bc`. This drops a dependency on `bc` being
# installed on the compute node (not guaranteed the way awk is), and keeps
# the numeric comparison in the same tool as the parsing above it.
set -euo pipefail

MIN_FREE_TIB="${MIN_FREE_TIB:-2}"

if ! command -v my_quotas >/dev/null 2>&1; then
    echo "ABORT: my_quotas not found on PATH (PSC quota-check environment" \
        "not loaded)" >&2
    exit 1
fi

quota_output="$(my_quotas)"

# Select lines inside "The quota for project directory ..." only, stopping
# at the next "The quota for ..." block (i.e. do not fall through into a
# third block if my_quotas ever adds one). Prints exactly two lines when
# the project block is well-formed: the quota value, then the used value.
parsed="$(
    printf '%s\n' "$quota_output" | awk '
        /^The quota for project directory/ { in_block = 1; next }
        /^The quota for/                   { in_block = 0 }
        in_block && /Storage quota:/       { print $NF }
        in_block && /Storage used:/        { print $NF }
    '
)"

quota_raw="$(printf '%s\n' "$parsed" | sed -n '1p')"
used_raw="$(printf '%s\n' "$parsed" | sed -n '2p')"

if [[ -z "$quota_raw" || -z "$used_raw" ]]; then
    echo "ABORT: could not find a 'project directory' quota block in" \
        "my_quotas output; refusing to guess. Raw output was:" >&2
    printf '%s\n' "$quota_output" >&2
    exit 1
fi

# Fail closed on units: the project block MUST be TiB. If my_quotas' format
# ever changes (block order, a PiB allocation, a renamed label), comparing
# TiB against GiB (or anything else) as if they matched would silently
# either block a healthy submission or allow one onto a full filesystem.
if [[ "$quota_raw" != *TiB || "$used_raw" != *TiB ]]; then
    echo "ABORT: expected TiB units in the project quota block, got" \
        "quota='$quota_raw' used='$used_raw'" >&2
    exit 1
fi

quota_num="${quota_raw%TiB}"
used_num="${used_raw%TiB}"

num_re='^[0-9]+(\.[0-9]+)?$'
if ! [[ "$quota_num" =~ $num_re ]] || ! [[ "$used_num" =~ $num_re ]]; then
    echo "ABORT: could not parse quota/used as plain numbers" \
        "(quota='$quota_raw' used='$used_raw')" >&2
    exit 1
fi

free_tib="$(awk -v q="$quota_num" -v u="$used_num" 'BEGIN { printf "%.4f", q - u }')"

echo "/ocean project free: ${free_tib} TiB (used ${used_num} TiB of" \
    "${quota_num} TiB, threshold ${MIN_FREE_TIB} TiB)"

below="$(awk -v f="$free_tib" -v m="$MIN_FREE_TIB" \
    'BEGIN { print (f < m) ? "1" : "0" }')"
if [[ "$below" == "1" ]]; then
    echo "ABORT: insufficient free space on /ocean project allocation" \
        "(${free_tib} TiB < ${MIN_FREE_TIB} TiB)" >&2
    exit 1
fi
