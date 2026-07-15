# Conversational LM-TTS (BagPiper + TAC)

POC recipe: TAC branch exchange injected into BagPiper (ESPnet speechlm, Qwen3-8B + Xcodec).
Design doc: vault note "Design - TAC in BagPiper LM-Based TTS POC" (2026-07-13).
Run tests: `PYTHONPATH=<worktree>:$(pwd) <espnet>/envs/bin/python -m pytest src/branch_exchange/tests tests -v`
