"""local/pool_shard_metrics.py: WER pooled from per-window counts across
shards; scalar summaries pooled as window-weighted means."""
import json

from egs3.conversational.tts.local.pool_shard_metrics import main, pool_summaries, pool_wer


def _shard(root, name, windows, metrics):
    d = root / name / "valid" / "scoring" / "conversation_asr"
    d.mkdir(parents=True)
    (d / "windows.jsonl").write_text("".join(json.dumps(w) + "\n" for w in windows))
    (root / name / "metrics.json").write_text(json.dumps(metrics))


def test_pooling(tmp_path):
    def win(wid, counts_a, counts_b):
        return {"window_id": wid, "channels": [{"counts": counts_a}, {"counts": counts_b}]}
    c = lambda h, s, d, i: {"hits": h, "substitutions": s, "deletions": d, "insertions": i}
    _shard(tmp_path, "shard0", [win("w0", c(90, 5, 5, 0), c(50, 0, 0, 0))],
           {"M": {"valid": {"wer_channel": 0.0667, "utmos_ipu_mean": 3.0, "sim_o_mean": 0.8}}})
    _shard(tmp_path, "shard1", [win("w1", c(10, 0, 0, 10), c(10, 0, 0, 0)), win("w2", c(20, 0, 0, 0), c(20, 0, 0, 0))],
           {"M": {"valid": {"wer_channel": 0.1429, "utmos_ipu_mean": 4.0, "sim_o_mean": 0.9}}})
    shards = sorted(p for p in tmp_path.glob("shard*"))
    wer = pool_wer(shards, "valid")
    # pooled: errors 5+5+10 = 20 over N = (90+5+5)+50+10+10+20+20 = 210
    assert wer["n_windows"] == 3 and abs(wer["wer_channel"] - 20 / 210) < 1e-9
    s = pool_summaries(shards)
    # window-weighted: (3.0*1 + 4.0*2)/3
    assert abs(s["utmos_ipu_mean"] - 11 / 3) < 1e-9 and s["n_shards"] == 2
    main([str(tmp_path), "--out", str(tmp_path / "pooled.json")])
    out = json.loads((tmp_path / "pooled.json").read_text())
    assert abs(out["pooled"]["wer_channel"] - 20 / 210) < 1e-9 and out["n_windows"] == 3
