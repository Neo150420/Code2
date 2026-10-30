# SAC-RWB baseline implementation for Neo

**Latest saved state:** 8 training episodes completed; episode 9 failed a
transition-coverage guard. Training is stopped. See [current status](CURRENT_STATUS.md)
and `artifacts/snapshot-2026-10-30/` for the captured results, failure log and
latest valid full resume checkpoint. Model files use Git LFS.

This is a corrected implementation using the authors' supplied SUMO assets, before
any uncertainty/quantum-inspired extension. See [the correction ledger](CHANGELOG_BASELINE.md).
Published paper results have not been reproduced. Upstream is pinned in provenance.json.

The current cloud is CPU-only. Python 3.10/Torch 1.13.1 CPU is installed at
`/workspace/research/.venv`; SUMO 1.12.0 is separately installed in
`/workspace/research/.sumo-venv` because its official wheel metadata targets Python 3.8.
Both environments have locked dependencies; no original dependency manifest was
silently edited. Windows usage should use Ubuntu 22.04 under WSL and adapt paths.

```bash
cd /workspace/Code2
export PATH=/workspace/research/.venv/bin:/workspace/research/.sumo-venv/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
python -m unittest -v test_correctness
python baseline.py train --out runs/my-pilot --episodes 3 --horizon 1000 --warmup 500
python baseline.py evaluate --checkpoint runs/my-pilot/checkpoint-0003.pt --out runs/my-evaluation --episodes 3 --horizon 1000 --warmup 500
```

Output directories must be new, to avoid overwriting evidence. Check `results.json`,
episode `summary.json`, `losses.json` and `steps.jsonl`. Evaluation loads a saved
actor and disables risk prediction and the training-only stopping proxy. AWT/AQL
and the collision-involvement proxy are explicitly named; read their definitions
in the ledger before comparing to published tables.
Protocol v3's `--horizon 1000` means 1000 control-loop steps after reset (as in
the authors' training script); reset time is additional. The monitor window is
still explicitly [500,1000). Earlier diagnostic pilots used a different horizon
convention and are labelled accordingly.

`checkpoint-NNNN.pt` stores models and optimizer states and verifies actor reload.
It is an evaluation checkpoint. `resume.pt` additionally stores replay and risk
buffers, target networks, RNG states and completed-episode metadata. It resumes
at an episode boundary, using the same output directory and settings:

```bash
python baseline.py train --out runs/my-pilot --episodes 500 --horizon 1000 --warmup 500 --resume runs/my-pilot/resume.pt
```

An interrupted partial episode is archived before replaying it. Code hashes and
core settings must match. Resume checks include matching the next SAC update
after restoring parameters, optimizer and replay state. Evaluation checkpoints
cannot be passed off as full training-resume checkpoints.

The intended full budget is 500 episodes, conditional on passing real-data risk
and SAC update, checkpoint/evaluation, protocol and runtime checks. Do not interpret
the unit tests or the earlier fixed-action smoke as learned-policy performance.

The full single-seed job has been started using:

```bash
python run_experiment.py --name full-seed-0 --episodes 500 --seed 0 --eval-sumo-seed 10001
```

Read `runs/full-seed-0/status.json`, `training/progress.json` and completed
`training/results.json`; verify the recorded process is still alive before
trusting a saved running status. The manager prevents duplicate instances. After
training it evaluates the final actor on three **disjoint traffic seeds**
10001–10003, then writes `evaluation-summary.json`. This is one trained model,
not a three-training-seed study. See [execution evidence](SESSION_VALIDATION.md).
Resume requires the same code/core settings. Do not publish/restart the cloud
expecting the live process to survive; full episode-boundary state is on disk.

Early checkpoint selection uses a separate fixed validation set; see
[validation protocol](VALIDATION_PROTOCOL.md). Checkpoints 1 and 3 are compared
using traffic seeds 7001–7003, with the training proxy/predictor disabled. The
table and paired per-seed differences are under `runs/full-seed-0/validation/`.
Final test seeds 10001–10003 remain untouched.

```bash
python checkpoint_validation.py --checkpoints 1 3 --pause-training --workers 2
python checkpoint_validation.py --checkpoints 10 --watch --pause-on-ready --hold-after-validation
```

The checkpoint-10 watcher waits for the complete episode/resume state, pauses
training, evaluates the same three validation seeds, then holds for review.
Read `validation/status.json` and `validation/training-pause.json` before assuming
that an alive trainer is actively training. A stopped process (`T`) is paused.
Do not start a second trainer to bypass a review gate. Resume the existing process
only after reviewing results and verifying its PID still belongs to this run.
If the instance stopped, use the complete `resume.pt` and preserve review history.
These validation comparisons do not establish final-test or journal performance.
