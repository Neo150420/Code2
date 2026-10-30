# Fixed-seed early validation

Traffic seeds: 7001, 7002, 7003. Final test seeds 10001–10003 are not used.
One trained model at different checkpoints; three traffic seeds are not three training seeds.
Collision rate below is selected-junction collision-pair onset events / junction entries,
not established as identical to the paper CR. These are validation results, not final test results.

| Checkpoint | AWT (s) | AQL (veh) | Collision events | Entries | Events / entry (%) | Mean network arrivals* | Mean vehicles remaining |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 148.499 | 0.925 | 41 | 1312 | 3.125 | 1337.0 | 2170.3 |
| 3 | 209.647 | 1.168 | 2 | 1180 | 0.169 | 1031.7 | 2323.0 |

*Network arrivals counted after Env.reset; not a completed-trip fraction.
Do not reward an apparent collision-rate decrease caused by reduced intersection entry.
Review waiting, entry flow, arrivals and remaining vehicles together.
