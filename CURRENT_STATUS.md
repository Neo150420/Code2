# Project status — 30 October 2026

Checkpoint 10 and its fixed-seed validation are complete. The original corrected
baseline is continuing unchanged toward checkpoint 25; an execution campaign
monitor is configured to advance the frozen 500-episode budget through later
checks. Full training and reserved final testing are **not complete**. The
published paper has not been reproduced, and CUDA has not been tested here.

## Completed baseline evidence

- Ten completed episodes, 10,000 control steps, 366,106 transitions, 50 SAC
  updates and zero missing transitions in the checkpoint-10 snapshot.
- Checkpoints 1, 3 and 10 evaluated on the same seeds 7001–7003, deterministic
  actors, no predictor inference and no training stopping proxy.
- 24 correctness, aggregation, experiment-identity and campaign-gate tests pass.
- Earlier collision-teleport transition/outcome-label fixes remain in place;
  baseline source, rewards and original simulator assets are unchanged.

| Baseline checkpoint | AWT (s) | AQL (veh) | Collision events | Junction entries |
|---|---:|---:|---:|---:|
| 1 | 141.045 | 0.866 | 41 | 1426 |
| 3 | 194.553 | 1.105 | 2 | 1082 |
| 10 | 283.948 | 1.181 | 0 | 504 |

Checkpoint 10 increased waiting and reduced entry flow on all three paired seeds.
Zero observed collisions is not established safety superiority.

## Policy diagnosis and decision

On the same 4096 training-replay states, the checkpoint-10 actor strongly braked
on all 1167 empty-junction states (mean -2.846 m/s²). Its critic preferred braking
on all those empty states. The replay contains 366,106 transitions, but only 50
SAC updates had occurred. Additional SAC optimization on that fixed replay
changes the conditional actions. This supports under-optimization as a
contributor; it does not establish a sole cause or a new runtime defect.

| Diagnostic actor | Total SAC updates | AWT (s) | AQL | Collision events | Entries | Events / entries (%) |
|---|---:|---:|---:|---:|---:|---:|
| Original checkpoint 10 | 50 | 283.948 | 1.181 | 0 | 504 | 0.000 |
| Same parent +250 offline updates | 300 | 160.601 | 0.740 | 1 | 810 | 0.123 |
| Same parent +1000 offline updates | 1050 | 36.628 | 0.615 | 18 | 3676 | 0.490 |

The diagnostic actors use unequal optimization budgets. They are not proposed
methods, equal-budget improvements or published-paper reproductions. Their
validation shows a flow/collision tradeoff; neither is established as safer.
They do not replace the primary baseline. See [full diagnosis](POLICY_DIAGNOSIS.md).

The primary baseline continues with its original reward and one SAC update per
200 steps. The campaign checks execution integrity at the declared milestones
and preserves negative metrics. It does not automatically tune the method or
claim a scientific win. Failure, nonfinite results, missing transitions, source/
weight/seed mismatch or zero validation exposure stops automatic continuation.

## Live state and archived evidence

Read `runs/readiness-v4/status.json`, `campaign-status.json`,
`training/progress.json` and completed `training/results.json`; verify processes
are alive. Saved launch receipts and PIDs are historical observations. Cloud
shutdown may stop processes; full episode checkpoints support recovery.
Final seeds 10001–10030 remain unused until the full training/selection process.

The [checkpoint-10 snapshot](artifacts/checkpoint10-review-2026-10-30/SNAPSHOT.json)
contains its full resume, ten checkpoints/episodes, source snapshots, common-state
panel, diagnostic actors, nine baseline and six new diagnostic evaluation
rollouts, paired summaries, tests and a SHA256 manifest. Model files use Git LFS;
run `git lfs pull` after cloning. This frozen parent remains reproducible even
when live training advances. Earlier v4 readiness and failed v3 evidence remain
in their separate historical snapshots; do not combine their learning curves.

The experiment is a [declared-assumption reconstruction](REPRODUCTION_PROTOCOL.md)
using the authors' actual assets. Its geometry, measurement window and event/
entry collision percentage are not established as identical to the journal's
Tables II/III. Beating SOTA or achieving a journal contribution is not claimed.
