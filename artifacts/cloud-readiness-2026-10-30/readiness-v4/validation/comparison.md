# Fixed-seed early validation

Traffic seeds: 7001, 7002, 7003. Final test seeds 10001–10003 are not used.
One trained model at different checkpoints; three traffic seeds are not three training seeds.
Collision rate below is selected-junction collision-pair onset events / junction entries,
not established as identical to the paper CR. These are validation results, not final test results.

| Checkpoint | AWT (s) | AQL (veh) | Collision events | Entries | Events / entry (%) | Mean network arrivals* | Mean vehicles remaining |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 141.045 | 0.866 | 41 | 1426 | 2.875 | 1153.7 | 2286.7 |
| 3 | 194.553 | 1.105 | 2 | 1082 | 0.185 | 1225.0 | 2158.3 |

*Network arrivals counted after Env.reset; not a completed-trip fraction.
Do not reward an apparent collision-rate decrease caused by reduced intersection entry.
Review waiting, entry flow, arrivals and remaining vehicles together.
