# Checkpoint-10 policy diagnosis

Completed CPU simulation diagnostics, 30 October 2026. Original baseline source, reward and assets are unchanged.
The primary 500-episode reconstruction continues separately; these offline actors do not replace it.

## Verified common-state evidence

Checkpoint 10 has 366,106 replay transitions but only 50 SAC gradient updates.
All actors were probed on the same 4096 checkpoint-10 training-replay states, including 1167 with zero junction occupancy.
This panel is not an independent held-out safety test.

| Actor | Mean acceleration, empty junction (m/s²) | Mean acceleration, occupied junction (m/s²) |
|---|---:|---:|
| Baseline checkpoint 1 | -0.060 | -0.023 |
| Baseline checkpoint 3 | -0.370 | -0.298 |
| Baseline checkpoint 10 | -2.846 | -2.772 |
| Checkpoint 10 + 250 offline updates | 0.863 | -2.642 |
| Checkpoint 10 + 1000 offline updates | 2.882 | -1.362 |

The checkpoint-10 critic prefers the -3 m/s² grid action on all 1167 empty-junction panel states.
Its actor strongly brakes even when the junction is empty and the vehicle is already stopped.
The existing no-conflict reward favours acceleration; no new reward-sign bug has been established.
Extra SAC optimization on identical replay shifts empty-junction actions toward acceleration.
This supports early SAC under-optimization as a contributor, rather than proving a sole cause.

## Matched traffic evaluation

Same original assets, 1000 steps after reset, window [500,1000), deterministic actors,
seeds 7001–7003, no predictor inference and no training conflict proxy. Final seeds remain unused.

| Actor | Total SAC updates | AWT (s) | AQL (veh) | Collision events | Junction entries | Events / entries (%) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline checkpoint 10 | 50 | 283.948 | 1.181 | 0 | 504 | 0.000 |
| Checkpoint 10 + 250 offline updates | 300 | 160.601 | 0.740 | 1 | 810 | 0.123 |
| Checkpoint 10 + 1000 offline updates | 1050 | 36.628 | 0.615 | 18 | 3676 | 0.490 |

Extra optimization improves flow in this screen but changes collision burden. Neither diagnostic is established as safer.
The event/entry percentage is our operational exposure metric, not verified as the paper CR.
These unequal optimization budgets cannot substantiate a novel-method improvement or beating the published table.

## Decision

Continue the unchanged primary baseline to the declared full budget, with monitored execution gates.
Keep reward weights, traffic and observation inputs fixed. Use the offline actors only to diagnose optimization sensitivity.
The campaign automates source/checkpoint/seed, finite-value, exposure, process and storage checks;
it does not automatically declare a scientific win or tune hyperparameters.
Independent training seeds, a strong matched baseline and untouched final evaluation are still required for a journal comparison.

The archived executed diagnostic script predates a later guard that forces use of the frozen episode-10 parent;
its hash is preserved with the raw report. The current root script defaults to that archived parent.
Risk predictor fit on the balanced training-buffer sample is descriptive and must not be called calibration.
