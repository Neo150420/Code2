# Project status — 30 October 2026

This repository contains a corrected SAC-RWB implementation, simulator inputs,
research reports and a frozen snapshot of available experiment artifacts.
The original paper has **not** been reproduced.

## Latest training state

- Eight full training episodes completed and have model checkpoints.
- Episode 9 failed the explicit transition-coverage check: one acted-on vehicle
  (`877763350#5.48`) had no matching reward/transition. The guard stopped training
  instead of silently dropping the transition.
- `artifacts/snapshot-2026-10-30/runs/full-seed-0/training/resume.pt` is the latest
  complete episode-boundary state, corresponding to eight completed episodes.
- Checkpoint 10 was not reached. Its waiting monitor was stopped because the
  training process had failed. Training is not running or successfully completed.
- The rare transition path requires diagnosis and a documented correction before
  further training. Source-hash resume checks intentionally reject silent code
  changes to an existing experiment.

## Available validation

Checkpoints 1 and 3 were evaluated on fixed traffic seeds 7001–7003. Checkpoint 3
had fewer collision-pair events but higher waiting/queue values and lower pooled
junction entries. This is an early safety–efficiency tradeoff, not overall success.
See `EARLY_REVIEW_1_VS_3.md` and the snapshot's validation table/raw per-seed data.
Final test seeds 10001–10003 were not evaluated.

## Artifact layout

- `artifacts/snapshot-2026-10-30/SNAPSHOT.json` records the captured run state.
- `MANIFEST.json` lists artifact sizes and SHA256 hashes before Git LFS filtering.
- `runs/` below the snapshot includes completed and explicitly labelled partial/
  diagnostic runs, saved source versions, logs, metrics, evaluations and weights.
- `additional-logs/` preserves setup, correctness, evaluation and failure logs.
- Torch checkpoints (`*.pt`) are stored using Git LFS. Use `git lfs pull` after
  cloning to obtain the actual model and resume files.
- Dependency environments/caches, transient locks, and uploaded source-paper PDFs
  are not committed. Paper identities/links and the audit are documented in
  `docs/RESEARCH_AUDIT.md` and `provenance.json`.

The snapshot preserves historical absolute cloud paths and process IDs as evidence;
they are not live endpoints or a claim that those processes still exist.
