# Cloud execution gate before local GPU training

This gate checks execution correctness on the original supplied reference assets.
It does not establish paper-result reproduction, convergence, or GPU execution.
Original v3 training and its episode-9 failure remain under `runs/full-seed-0` and
the committed historical snapshot. Fresh v4 training uses a different directory.

**Verified outcome, 30 October 2026:** passed. Fifteen tests and 22 data/source
checks passed; four fresh training episodes (including actual resume), the full
final-source episode-9 diagnostic replay and six fixed-seed evaluations completed.
See [current status](CURRENT_STATUS.md) for counts, early metrics and limitations.

Required evidence:

1. Reproduce the original failure with the original source hashes and parent
   episode-8 resume state. Record before/after road, SUMO presence and virtual ID.
2. Replay that episode after the terminal-transition correction. Confirm the
   original failing vehicle receives one terminal transition at time 288 s and
   the full 1000-step episode completes with actual SAC/risk updates.
3. Include collision-teleport participants using the event lane. Regression-test
   the observed event and exclusions for approach roads and other junction IDs.
4. Train a fresh v4 pilot for three 1000-control-step episodes. Resume its full
   checkpoint for episode four with unchanged code/settings; preserve the original
   config, resume config, source snapshots, weights, losses and step records.
5. Evaluate v4 checkpoints 1 and 3 using the same validation traffic seeds
   7001–7003, deterministic actors, no training proxy and no predictor inference.
   Check transition coverage, finite outputs, exposure, collision counts and queues.
6. Run the correctness/aggregation tests, including restored next-update agreement.
   Verify supplied simulator asset hashes and code hashes in train/evaluation.

Final traffic seeds 10001–10003 are reserved and excluded from this gate.
Passing means this bounded cloud pipeline ran successfully. It is not a guarantee
that every later episode is bug-free. Additional milestone reviews remain needed.
The current cloud has no GPU; CUDA/driver/VRAM checks must be run separately when
the code moves to Neo's local GPU. No physical-vehicle experiment is required.

Example fresh pilot and real episode-boundary resume:

```bash
export PATH=/workspace/research/.venv/bin:/workspace/research/.sumo-venv/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
python -m unittest -v test_correctness test_validation
python baseline.py train --out runs/readiness-v4/training --episodes 3
python baseline.py train --out runs/readiness-v4/training --episodes 4 \
  --resume runs/readiness-v4/training/resume.pt
python checkpoint_validation.py --run readiness-v4 --checkpoints 1 3 --workers 2
python check_readiness.py --run runs/readiness-v4 \
  --diagnostic-run runs/transition-replay-v4-final
```

Use a fresh run name when repeating the gate. Never bypass the old experiment's
source-hash guard or present a corrected-source diagnostic as a v3 continuation.
