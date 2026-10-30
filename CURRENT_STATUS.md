# Project status — 30 October 2026

The corrected v4 baseline passed the bounded CPU execution gate. The published
paper has **not** been reproduced; CUDA execution and 500-episode training have
not been completed. No full trainer is running.

## Confirmed repairs

The original episode-9 failure was reproduced with identical first 261 step-log
records. At simulation time 288 s a junction collision teleported vehicle
`877763350#5.48` onto an outgoing edge. The old code skipped its reward/transition.
It also missed that collider in selected-junction outcome labels because it
checked the vehicle's post-teleport road rather than the collision event lane.
Both defects are corrected. The transition-coverage assertion remains enabled.

Final v4 source replayed the complete failing episode: 1000 control steps,
70,619 transitions, zero missing transitions, five SAC updates and five risk
training calls. Both participants of the observed junction collision were
recorded. This is a labelled diagnostic using the old episode-8 state, not a
silent continuation of the old learning curve.

## Fresh v4 execution evidence

- Four completed 1000-step training episodes, with a real process restart and
  full-checkpoint resume between episodes 3 and 4.
- 70,764 transitions, 20 SAC updates, 19 risk-training calls and zero missing
  transitions. Actor and risk parameters changed; recorded losses were finite.
- 15 correctness/aggregation tests passed, including next-update agreement after
  checkpoint restoration. The additional readiness verifier passed 22 checks.
- Original simulator input hashes and train/evaluation source hashes match.
- Checkpoints 1 and 3 each completed evaluation on seeds 7001–7003. Predictor
  inference and the training conflict proxy were disabled. Final test seeds
  10001–10003 remain untouched. These are one training run's early checkpoints.

| Validation metric | Checkpoint 1 | Checkpoint 3 |
|---|---:|---:|
| Mean AWT (s) | 141.045 | 194.553 |
| Mean AQL (vehicles) | 0.866 | 1.105 |
| Pooled junction collision-pair onset events | 41 | 2 |
| Pooled junction entries | 1426 | 1082 |
| Events / entries (%) | 2.875 | 0.185 |

Waiting increased on every paired seed. Collision events decreased, with lower
junction entry flow. This is an early safety–efficiency tradeoff, not convergence
or a demonstrated overall improvement. Events/entries is an operational metric;
it is not established as identical to the paper's CR.

## Saved evidence and remaining work

The fresh full resume is `runs/readiness-v4/training/resume.pt` (four completed
v4 episodes). The old v3 run stopped after eight episodes and remains preserved
separately. Do not combine their episode counts or bypass source-hash guards.

The committed export is [cloud readiness snapshot](artifacts/cloud-readiness-2026-10-30/SNAPSHOT.json),
including checkpoints, raw steps, source snapshots, per-seed evaluations, test
logs, the readiness report and a SHA256 manifest. Model files use Git LFS;
run `git lfs pull` after cloning to download the actual checkpoint data.
The [older snapshot](artifacts/snapshot-2026-10-30/SNAPSHOT.json) retains the v3
failure and parent checkpoint. Historical absolute paths/PIDs are provenance,
not live services. See [gate procedure](CLOUD_READINESS.md) and the correction ledger.

Exact journal-result reproduction still requires resolving the paper's symmetric
single-intersection description versus the supplied multi-junction assets,
Table II/III evaluation protocols and the precise successful-passage denominator
and update schedule. Do not infer missing settings by tuning toward the table.
Before a long local GPU run, verify CUDA execution on Neo's hardware and settle
which documented benchmark configuration is being reproduced. The cloud gate
establishes bounded execution correctness, not that every later episode is bug-free.
