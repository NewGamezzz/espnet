"""Pure-function test for the K-speaker cpSIM assignment (no models, no
zipvoice import needed).  Run from this directory:
``python -m pytest test_cpsim_perm.py -q``."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpsim_arm_k import best_permutation  # noqa: E402


def test_k2_matches_stock_rule():
    p = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    e = [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])]
    score, perm = best_permutation(p, e)
    assert perm == (1, 0) and abs(score - 1.0) < 1e-6


def test_k2_direct_wins_when_better():
    p = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    e = [torch.tensor([0.9, 0.1]), torch.tensor([0.2, 0.8])]
    score, perm = best_permutation(p, e)
    direct = (torch.nn.functional.cosine_similarity(p[0], e[0], dim=-1)
              + torch.nn.functional.cosine_similarity(p[1], e[1], dim=-1)).item() / 2
    assert perm == (0, 1) and abs(score - direct) < 1e-6


def test_k3_finds_the_assignment():
    p = [torch.tensor([1.0, 0, 0]), torch.tensor([0, 1.0, 0]), torch.tensor([0, 0, 1.0])]
    e = [p[2], p[0], p[1]]
    score, perm = best_permutation(p, e)
    assert perm == (1, 2, 0) and abs(score - 1.0) < 1e-6
