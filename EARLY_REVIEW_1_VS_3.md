# Early checkpoint review: episodes 1 versus 3

Both checkpoints were evaluated on the same three validation traffic seeds
7001, 7002, 7003, with deterministic actor control, 1000 control-loop steps,
the [500,1000) measurement window, the same simulator assets, and both risk
inference and the training-only stopping proxy disabled. Final test seeds
10001–10003 have not been used. All six evaluations completed successfully.

| Metric | Checkpoint 1 | Checkpoint 3 | Interpretation |
|---|---:|---:|---|
| Mean AWT (s) | 148.499 | 209.647 | 41.2% higher |
| Mean AQL (vehicles) | 0.925 | 1.168 | 26.3% higher |
| Collision-pair onset events, pooled | 41 | 2 | Fewer events |
| Junction entries, pooled | 1312 | 1180 | 10.1% fewer |
| Events per junction entry (%) | 3.125 | 0.169 | Lower operational collision rate |
| Mean network arrivals after reset | 1337.0 | 1031.7 | Fewer arrivals |
| Mean network vehicles remaining | 2170.3 | 2323.0 | 7.0% more remaining |

Waiting increased on all three paired seeds (+71.60, +19.88, +91.96 s).
Entry counts decreased on two seeds and increased on one (-143, +74, -63).
Collision-event differences were -30, 0, -9. This is an early safety–efficiency
tradeoff, not established overall improvement. The lower event/exposure rate
is a positive safety signal under this operational definition, but it does not
establish calibrated risk, real-world safety or statistical superiority.
Remaining network vehicles include vehicles normally in transit and are not
equivalent to failed trips. The operational collision definition is not verified
as identical to the paper CR. No proximity-to-paper result is claimed.

The models had only 5 versus 15 SAC gradient updates. These results are too early
to establish convergence or predict whether 500 episodes will succeed. Continue
the unchanged baseline only to the next diagnostic milestone, checkpoint 10,
then review the same fixed-seed table before continuing further. Do not silently
change reward, update schedule, simulator safety rules or reporting denominators
to obtain better-looking numbers.

Training was resumed after completing the six evaluations. A checkpoint-10
watcher is active with `--watch --pause-on-ready --hold-after-validation`: it
waits for the full completed episode/resume state, pauses the trainer, performs
the matched validation and leaves the trainer paused for review. Core training
source hashes remain identical to the running experiment.

Outputs: `runs/full-seed-0/validation/comparison.md`, `comparison.csv`,
`paired-deltas.json`, per-checkpoint summaries and raw per-seed evaluation data.
The environment draft's install/start instructions were updated for the 12
passing targeted tests and checkpoint-review workflow. Saving is not publication.
Review/save and publish in environment settings after stopping or completing live
jobs; publication does not preserve running processes.
