# Early-checkpoint validation protocol

This is a screening/continuation protocol for the corrected reference-asset
baseline, not a claim of reproducing the journal or selecting a novel method.

- Validation traffic seeds are fixed at 7001, 7002, 7003. Training uses SUMO
  seeds 1..500. Final test traffic seeds 10001..10003 remain untouched by
  checkpoint selection. Changes chosen using validation are reported explicitly.
- First comparison: completed checkpoints 1 and 3 from the same training run.
  Next review milestone: checkpoint 10. Later milestones are 25, 50, 100, 200,
  500, subject to review. These are policy checkpoints, not independent models.
- Same original network/route files, four controlled junctions, 1000 control
  steps after reset, monitor window [500,1000), and three matched traffic seeds
  for every compared checkpoint. Use deterministic actor actions. No predictor
  inference, training-only stopping proxy, extra sensing, or added safety shield.
- Validation processes use one Torch thread each and never update the checkpoint.
  Record checkpoint SHA256, source hashes, per-seed output and aggregate output.
- Report direction/time-mean AWT and AQL, selected-junction collision-pair onset
  event counts, junction entry counts, their pooled event/exposure percentage,
  network arrivals after reset and remaining vehicles. Network totals are
  explicitly distinct from controlled-junction counts. A zero-entry collision
  rate is undefined, not evidence of safety.
- Review paired per-seed differences as well as aggregate results. A rate can
  fall merely because the policy blocks traffic: examine raw event counts,
  entries, waiting and unfinished vehicles together. Do not infer improvement
  from training reward, BCE loss or a single favourable seed.
- Three traffic seeds provide an early diagnostic screen. They do not establish
  safety superiority, statistical convergence or robustness across training
  seeds. Rare-event conclusions require much larger, prospectively specified
  exposure and independent training runs. Final-test results cannot be used
  repeatedly to tune checkpoints and still be called an untouched test.

## Continuation decisions

1. Hold for diagnosis if evaluations fail, checkpoint/source/seed settings differ,
   nonfinite outputs occur, training mechanisms leak into evaluation, or exposure
   disappears. Fix infrastructure before making a performance claim.
2. If later checkpoints consistently increase waiting/backlog while reducing
   throughput, or increase collision burden without an acceptable efficiency
   tradeoff, investigate policy/reward/metric behaviour. Do not conceal the
   regression by changing denominators, dropping a seed, or rewriting the task.
3. Mixed/noisy results at checkpoints 1 and 3 are inconclusive, especially with
   only 5 versus 15 SAC updates. They can justify continuing to checkpoint 10
   with the same settings; they do not justify a claim that training will work.
4. Checkpoint 10 is a review gate before proceeding further. Preserve complete
   episode state, review the paired table, and document continuation or changes.
   A material code/protocol change creates a separately labelled run and must
   not be silently resumed into the old learning curve.

The author's Table II scenario/evaluation/CR correspondence remains unresolved.
This table compares our own frozen checkpoints under one declared protocol;
it does not make our metric definition identical to the paper's.
