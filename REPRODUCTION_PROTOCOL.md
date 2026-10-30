# SAC-RWB reconstruction with declared assumptions

Neo authorized fixing missing experimental settings as explicit assumptions on
30 October 2026. The primary experiment is a **corrected author-asset
reconstruction**, not a claim of exact numerical reproduction of the final paper.
The frozen machine-readable contract is [assumptions-v1.json](reproduction/assumptions-v1.json).
It records source, simulator input, dependency and execution-script hashes.

The contract was frozen after four pilot episodes and checkpoint 1/3 validation
had already been seen. Those observations are disclosed in CURRENT_STATUS.md;
this is not a claim of preregistration before any data were observed. The model,
reward, traffic and update schedule are unchanged from that verified v4 pilot.

| Setting | Fixed choice | Evidence / limitation |
|---|---|---|
| Network and demand | Unchanged authors' network/routes; junctions 229, 499, 332, 334 | Repository assets are primary; final paper's symmetric single-intersection description differs |
| Traffic population | 100% RV assignment; original departures/turns and SUMO dynamics | Author configuration; no new human-behaviour or occlusion problem |
| State/action | 98 features, one acceleration in [-3,3] m/s²; 30 m control area, 10 m/s limit | Paper architecture and corrected implementation |
| Training target | 500 episodes × 1000 control steps after reset | Paper training budget; reset advance is additional |
| SAC updates | One gradient update per 200 control-loop steps | Observed author-code behaviour; paper update multiplicity is ambiguous |
| Risk updates | Eligible calls every 200 steps; 3 epochs; original balanced buffers | Preserve supplied implementation; no selective budget reduction |
| Measurement | Monitor/simulator indices [500,1000) | Declared inherited window; not assumed identical to Table II |
| Collision percentage | Physical junction collision-pair onset events / junction entries | Operational proxy, not established as paper successful-passage CR; report raw numerator and exposure |
| Evaluation controller | Deterministic actor; predictor and stopping proxy disabled | Paper test-time architecture |
| Screening seeds | 7001, 7002, 7003 for every checkpoint | Early screening only, not independent training seeds |
| Final seeds | 10001–10030, after full training and checkpoint decisions | Declared stronger exposure budget; do not tune using these results |
| Execution | Current CPU cloud, two Torch threads, pinned dependencies | GPU execution is a separate portability check |

Published hyperparameters retained in the executable contract include the SAC
256–256 networks, actor/critic learning rates 3e-4/4e-4, batch 256, replay one
million, initial alpha 0.12, and risk Transformer history 10, dimension 128,
two layers, four heads, beta 0.2, batch 128 and learning rate 1e-5.
Implementation corrections and reward-sign interpretation remain in the ledger.

## Running and reviewing

**Current execution:** the checkpoint-10 review is complete; see POLICY_DIAGNOSIS.md.
The unchanged baseline continues toward 25. A separate campaign monitor now
automates execution checks at 25, 50, 100 and 200, then continues the fixed
500-episode target and its reserved final evaluation. It stops for input/weight/
seed mismatches, failed or missing processes, nonfinite results, missing
transitions, zero evaluation exposure or insufficient disk space. These are
execution checks, not automatic claims of scientific improvement. It never
tunes rewards, observations, assets or hyperparameters. Negative early metrics
remain in the record; the full baseline budget is needed before judging convergence.

```bash
python run_reproduction_campaign.py --run readiness-v4 \
  --contract reproduction/assumptions-v1.json
```

The campaign can adopt an existing valid manager. Start only one campaign and
one manager. Inspect `runs/readiness-v4/campaign-status.json` alongside manager
status and training progress. If either reports an error, inspect before any
restart. The command below documents the original initial review segment;
do not launch it alongside the active campaign.

The existing fresh v4 lineage is `runs/readiness-v4/training`; its first four
episodes are preserved in the committed cloud-readiness snapshot. It is now
continued under this contract. Historical v3 episodes are never added to it.

```bash
cd /workspace/Code2
export PATH=/workspace/research/.venv/bin:/workspace/research/.sumo-venv/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
python run_experiment.py --name readiness-v4 --episodes 500 --review-at 10 \
  --seed 0 --eval-sumo-seed 10001 --eval-episodes 30 \
  --contract reproduction/assumptions-v1.json
```

The controller stops training naturally at the end of episode 10, saves the full
resume checkpoint, evaluates checkpoint 10 on the same three validation seeds
and compares it with the verified checkpoints 1 and 3. It then records
`awaiting_checkpoint_review`. It does not consume final-test seeds at this gate.
Subsequent declared review milestones are 25, 50, 100 and 200; a continuation
changes `--review-at`, retaining the same contract and full 500-episode target.
The final continuation omits `--review-at` and evaluates 30 final traffic seeds
only after the complete training budget. Do not start a duplicate manager.

Inspect `runs/readiness-v4/status.json`, `training/progress.json`, completed
`training/results.json` and `validation/comparison.md`. A saved PID/status is not
proof of a live process. The controller records subprocess failure and prevents
duplicate instances. After cloud shutdown, explicitly restart the same command;
it resumes the last complete episode. Live processes are not guaranteed to
survive environment shutdown.

Review raw collision events, junction entries, waiting, queues and remaining
vehicles together. Increased waiting on all seeds with reduced flow is a reason
to investigate conservatism, not to declare success from a low collision ratio.
Do not change the denominator, remove an unfavourable seed, or tune toward the
published table. Update-budget sensitivity must be a separately labelled run;
the proposal cannot receive more training/tuning budget than its baseline.

## What a journal comparison may claim

Missing settings are assumptions; measured results come from simulation. Published
Table II/III numbers remain external reference values and are not treated as
directly matched results under this different supplied geometry/protocol.
A paper-description geometry reconstruction, if made later, must be a separate
track with its own demand assumptions and retrained matched baselines; it has
not been substituted for this benchmark.

Any proposed method uses the same assets, observations, traffic seeds, reward/
cost definitions, tuning and compute budgets and evaluation exposure as its
controls. A changed human-driver/behaviour environment is another labelled task,
on which all methods are retrained. Confirm promising comparisons across
independent training seeds and report uncertainty before superiority claims.
This first seed is an engineering/performance reproduction attempt, not that
confirmatory study. Beating the baseline or SOTA is an empirical question.
