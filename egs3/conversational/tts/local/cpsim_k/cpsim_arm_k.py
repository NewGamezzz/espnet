"""K-speaker cpSIM (ZipVoice-Dialog protocol generalized).

pyannote forced to K speakers, per-label concatenation, WavLM-large ECAPA
embeddings, MAX over the K! prompt-to-speaker assignments of the mean cosine.
K = 2 reproduces ``cpsim_arm.py`` exactly (``max(direct, swapped) / 2``).

Canonical source lives in the recipe (``local/cpsim_k/``); the Delta copy at
``/work/hdd/bbjs/ttrachu/scripts/cpsim/`` runs inside the ZipVoice cpSIM env
(``run_cpsim_all.sbatch`` recipe: nvme pixi env, models under
``/work/nvme/bbjs/ttrachu/tts_eval_models``, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1).

Usage:
    python cpsim_arm_k.py --model-dir DIR --manifest exp/.../manifest.jsonl \\
        --base-dir exp/.../k3 --wav-path <mix dir> --out FILE.tsv [--extension wav]
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import time

import torch
import torch.nn.functional as F


def best_permutation(prompt_embs, eval_embs):
    """Max over assignments of the mean prompt-to-speaker cosine; returns
    ``(score, perm)`` where ``perm[i]`` is the diarized speaker matched to
    prompt ``i``.  Pure torch, no model."""
    k = len(prompt_embs)
    if len(eval_embs) != k:
        raise ValueError(f"{k} prompts but {len(eval_embs)} diarized speakers")
    best, best_perm = -2.0, None
    for perm in itertools.permutations(range(k)):
        s = sum(
            F.cosine_similarity(prompt_embs[i], eval_embs[perm[i]], dim=-1).item()
            for i in range(k)
        ) / k
        if s > best:
            best, best_perm = s, perm
    return best, best_perm


try:  # the scorer itself needs the ZipVoice eval env; the pure part above does not
    from zipvoice.eval.speaker_similarity.cpsim import CpSpeakerSimilarity, load_waveform
except ImportError:  # pragma: no cover - exercised only outside the eval env
    CpSpeakerSimilarity = None  # type: ignore[assignment]
    load_waveform = None  # type: ignore[assignment]


if CpSpeakerSimilarity is not None:

    class KCpSpeakerSimilarity(CpSpeakerSimilarity):
        """The stock scorer with ``num_speakers`` a parameter."""

        fallback = 0
        error = 0

        @torch.no_grad()
        def embeddings_with_diarization_k(self, audio_path, num_speakers):
            speech = load_waveform(
                audio_path, self.sample_rate, device=self.device, max_seconds=120
            )
            diar = self.diarization_pipeline(
                {"waveform": speech.unsqueeze(0), "sample_rate": self.sample_rate},
                num_speakers=num_speakers,
            )
            chunks = {}
            for turn, _, label in diar.itertracks(yield_label=True):
                a = int(turn.start * self.sample_rate)
                b = int(turn.end * self.sample_rate)
                chunks.setdefault(label, []).append(speech[a:b])
            labels = sorted(chunks)
            if len(labels) < num_speakers:
                logging.debug(
                    f"Insufficient speaker chunks in {audio_path} "
                    f"({len(labels)} < {num_speakers}); full audio for every speaker"
                )
                self.fallback += 1
                return [self.sv_model([speech]) for _ in range(num_speakers)]
            try:
                return [
                    self.sv_model([torch.cat(chunks[lab], dim=0)])
                    for lab in labels[:num_speakers]
                ]
            except Exception as exc:  # same fallback as the stock scorer
                logging.debug(f"embedding error {exc}; full audio for every speaker")
                self.error += 1
                return [self.sv_model([speech]) for _ in range(num_speakers)]

        @torch.no_grad()
        def prompt_embeddings(self, paths):
            return [
                self.sv_model([load_waveform(p, self.sample_rate, device=self.device)])
                for p in paths
            ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--manifest", required=True, help="export's manifest.jsonl")
    ap.add_argument("--base-dir", required=True, help="dir the prompt_wav paths are relative to")
    ap.add_argument("--wav-path", required=True, help="dir of <window_id>.<ext> mixdowns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extension", default="wav")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if CpSpeakerSimilarity is None:
        raise SystemExit("zipvoice is not importable: run inside the cpSIM eval env")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    d = os.path.join(a.model_dir, "speaker_similarity/")
    cp = KCpSpeakerSimilarity(
        d + "wavlm_large_finetune.pth", d + "wavlm_large/", d + "pyannote/"
    )
    rows = [json.loads(line) for line in open(a.manifest) if line.strip()]
    if a.limit:
        rows = rows[: a.limit]
    scores, missing, t0 = [], [], time.time()
    with open(a.out, "w") as fo:
        fo.write("Name\tK\tcpSIM\tperm\n")
        for r in rows:
            wid, k = r["window_id"], int(r["num_channels"])
            e = os.path.join(a.wav_path, f"{wid}.{a.extension}")
            if not os.path.exists(e):
                missing.append(wid)
                continue
            pe = cp.prompt_embeddings(
                [os.path.join(a.base_dir, c["prompt_wav"]) for c in r["channels"]]
            )
            ee = cp.embeddings_with_diarization_k(e, k)
            s, perm = best_permutation(pe, ee)
            scores.append(s)
            fo.write(f"{wid}\t{k}\t{s:.6f}\t{','.join(map(str, perm))}\n")
    summ = {
        "wav_path": a.wav_path,
        "manifest": a.manifest,
        "n": len(scores),
        "missing": missing,
        "cpSIM_mean": float(sum(scores) / len(scores)) if scores else None,
        "fallback_fewer_than_k": cp.fallback,
        "embed_error_fallback": cp.error,
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(summ, open(a.out + ".summary.json", "w"), indent=2)
    print(json.dumps(summ))


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
