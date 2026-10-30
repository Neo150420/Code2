# Baseline execution evidence

This records the implementation milestone, not completion of 500-episode training
or reproduction of the journal tables. Consult live run files for current progress.

| Check | Observed outcome |
|---|---|
| Dependency compatibility | Both isolated environments pass checks |
| Original simulator assets | All three SHA256 hashes match author reference |
| Targeted correctness tests | 8 passed |
| Three-episode diagnostic pilot | 47,620 transitions; 14 SAC updates; 13 risk update calls / 705 minibatches |
| Real risk learning | 32 risk state tensors changed between pilot checkpoints 1 and 2 |
| Actor save/load | Identical deterministic actions after independent reload |
| Evaluation mode | Risk predictor and training conflict proxy disabled |
| Same-checkpoint/seed reproducibility | Exact 200-step event/control trace match |
| Full training-state restore | Unit next-update equality plus actual simulator episode continuation passed |
| Full-run first completed episode | 1000 control steps, 16,115 transitions, 5 SAC updates, 84 risk minibatches; actor and predictor parameters changed |
| Full 500-episode experiment | Started; see runs/full-seed-0/status.json and training/progress.json |

Diagnostic pilot `pilot-002` used the older horizon/throughput reporting and its
performance metrics must not be used for journal comparison. `budget-v3-check`
verified the final loop convention: 200 control steps plus reset time. The full
run uses protocol v3: 500 episodes with 1000 control-loop steps each.

The running manager targets one training seed (0), followed by evaluation with
traffic seeds 10001, 10002, 10003, outside training traffic seeds 1..500. Three
traffic seeds are not three independently trained models. The manager briefly
restarted from the first complete resume checkpoint to activate the disjoint
evaluation seed range. An interrupted partial episode was archived; it is not
counted as completed training. Interruption exposed and repaired a manager
signal-handler wait-lock issue; full training resumed from the retained state.

One earlier pipeline evaluation of the **one-episode diagnostic pilot** measured
AWT 129.33 s, AQL 0.8863, and selected-junction collision-pair-onset events per
entry 1.566%. These are short-pilot diagnostics with an explicit operational
collision definition, not results of 500-episode training, not directly matched
to the paper's CR, and not evidence of approaching the published performance.

CPU first-episode time was about 239 s. A simple extrapolation is about 33 hours
for 500 episodes before allowing for fuller risk buffers, evaluation, interruptions
and variability. The earlier three-episode pilot ran at 225–338 s per episode
while other validation processes were also active. This is a rough resource
estimate, not a completion-time guarantee.

Live processes require this cloud instance to remain running. Publication/restart
does not preserve the process; `resume.pt` preserves complete episode-boundary
training state. Setup and startup instructions were saved to the cloud draft;
they were not published. Review/save and publish the prepared environment after
training is stopped or complete, rather than expecting a running job to survive
publication. New-task restoration has not been independently verified.

The authors' final map/evaluation/physical-collision protocol still needs
reconciliation. Full-run output will be labelled corrected reference-asset
results unless that correspondence can be established. No quantum extension has
been added and no SOTA claim is made.
