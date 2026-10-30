"""Summarize the completed checkpoint-10 optimization diagnostics honestly."""
import json
from pathlib import Path

from checkpoint_validation import summarize
from reproduction_contract import file_sha256


def main():
    root=Path(__file__).resolve().parent
    snapshot=root/'artifacts/checkpoint10-review-2026-10-30'
    parent=json.loads((snapshot/'training/results.json').read_text())
    profile=json.loads((snapshot/'offline-diagnostic/report.json').read_text())
    if file_sha256(snapshot/'training/resume.pt')!=profile['parent_resume_sha256']:
        raise ValueError('Diagnostic parent checkpoint differs')
    if file_sha256(snapshot/'offline-diagnostic/diagnose_policy.py')!=profile['diagnostic_script_sha256']:
        raise ValueError('Executed diagnostic source snapshot differs')
    base_updates=sum(r['sac_updates'] for r in parent['episodes'])
    baseline=json.loads((snapshot/'validation/checkpoint-0010/summary.json').read_text())
    summaries=[dict(baseline,label='Baseline checkpoint 10',extra_offline_updates=0,total_sac_updates=base_updates)]
    for updates in (250,1000):
        result=json.loads((root/f'runs/offline-{updates:04d}-validation/results.json').read_text())
        if result['config']['code_sha256']!=parent['config']['code_sha256']:
            raise ValueError('Diagnostic evaluation source differs from baseline')
        if result['config']['checkpoint_sha256']!=file_sha256(snapshot/f'offline-diagnostic/offline-{updates:04d}.pt'):
            raise ValueError('Diagnostic evaluation weights differ')
        row=summarize(updates,result)
        summaries.append(dict(row,label=f'Checkpoint 10 + {updates} offline updates',
                              extra_offline_updates=updates,total_sac_updates=base_updates+updates))
    paired=[]
    for candidate in summaries[1:]:
        for reference,row in zip(baseline['episode_results'],candidate['episode_results']):
            paired.append(dict(extra_offline_updates=candidate['extra_offline_updates'],sumo_seed=row['sumo_seed'],
                awt_delta_seconds=row['awt_lane_time_mean_seconds']-reference['awt_lane_time_mean_seconds'],
                collision_events_delta=row['junction_collision_pair_events_in_window']-reference['junction_collision_pair_events_in_window'],
                entries_delta=row['junction_entries_in_window']-reference['junction_entries_in_window']))
    report=dict(diagnostic_only=True,original_paper_reproduced=False,final_test_evaluated=False,
                source='Frozen checkpoint-10 replay; common validation seeds 7001–7003',
                training_replay_size=profile['training_replay_size'],parent_sac_updates=base_updates,
                comparisons=summaries,paired_deltas=paired,
                interpretation='Additional SAC optimization changes conditional actions and traffic outcomes; under-optimization is supported, but no safety superiority or new method is established',
                limitations=['Only one parent training seed and three reused validation seeds',
                             'Offline actors have additional optimization budget and are not equal-budget novel-method comparisons',
                             'Risk profile uses balanced training-buffer outcomes; not calibration or causal action-effect evidence',
                             'Frozen-replay SAC can extrapolate to poorly covered actions; actors are diagnostic only'])
    (snapshot/'diagnostic-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    text=['# Checkpoint-10 policy diagnosis','',
          'Completed CPU simulation diagnostics, 30 October 2026. Original baseline source, reward and assets are unchanged.',
          'The primary 500-episode reconstruction continues separately; these offline actors do not replace it.','',
          '## Verified common-state evidence','',
          f"Checkpoint 10 has {profile['training_replay_size']:,} replay transitions but only {base_updates} SAC gradient updates.",
          'All actors were probed on the same 4096 checkpoint-10 training-replay states, including 1167 with zero junction occupancy.',
          'This panel is not an independent held-out safety test.','',
          '| Actor | Mean acceleration, empty junction (m/s²) | Mean acceleration, occupied junction (m/s²) |',
          '|---|---:|---:|']
    for n in (1,3,10):
        p=profile['checkpoints'][str(n)]['policy']
        text.append(f"| Baseline checkpoint {n} | {p['empty_junction']['mean']:.3f} | {p['occupied_junction']['mean']:.3f} |")
    for n in (250,1000):
        p=profile['offline_updates'][str(n)]['policy']
        text.append(f"| Checkpoint 10 + {n} offline updates | {p['empty_junction']['mean']:.3f} | {p['occupied_junction']['mean']:.3f} |")
    text+=['','The checkpoint-10 critic prefers the -3 m/s² grid action on all 1167 empty-junction panel states.',
           'Its actor strongly brakes even when the junction is empty and the vehicle is already stopped.',
           'The existing no-conflict reward favours acceleration; no new reward-sign bug has been established.',
           'Extra SAC optimization on identical replay shifts empty-junction actions toward acceleration.',
           'This supports early SAC under-optimization as a contributor, rather than proving a sole cause.','',
           '## Matched traffic evaluation','',
           'Same original assets, 1000 steps after reset, window [500,1000), deterministic actors,',
           'seeds 7001–7003, no predictor inference and no training conflict proxy. Final seeds remain unused.','',
           '| Actor | Total SAC updates | AWT (s) | AQL (veh) | Collision events | Junction entries | Events / entries (%) |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        text.append(f"| {r['label']} | {r['total_sac_updates']} | {r['awt_seconds']:.3f} | {r['aql_vehicles']:.3f} | {r['collision_pair_events']} | {r['junction_entries']} | {r['collision_events_per_entry_percent']:.3f} |")
    text+=['','Extra optimization improves flow in this screen but changes collision burden. Neither diagnostic is established as safer.',
           'The event/entry percentage is our operational exposure metric, not verified as the paper CR.',
           'These unequal optimization budgets cannot substantiate a novel-method improvement or beating the published table.','',
           '## Decision','',
           'Continue the unchanged primary baseline to the declared full budget, with monitored execution gates.',
           'Keep reward weights, traffic and observation inputs fixed. Use the offline actors only to diagnose optimization sensitivity.',
           'The campaign automates source/checkpoint/seed, finite-value, exposure, process and storage checks;',
           'it does not automatically declare a scientific win or tune hyperparameters.',
           'Independent training seeds, a strong matched baseline and untouched final evaluation are still required for a journal comparison.','',
           'The archived executed diagnostic script predates a later guard that forces use of the frozen episode-10 parent;',
           'its hash is preserved with the raw report. The current root script defaults to that archived parent.',
           'Risk predictor fit on the balanced training-buffer sample is descriptive and must not be called calibration.','']
    (root/'POLICY_DIAGNOSIS.md').write_text('\n'.join(text))
    print(json.dumps(dict(comparisons=[{k:r[k] for k in ('label','awt_seconds','collision_pair_events','junction_entries')} for r in summaries])))


if __name__=='__main__':main()
