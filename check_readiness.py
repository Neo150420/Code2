"""Verify a completed four-episode cloud pilot and matched checkpoint evaluations."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/readiness-v4')
    parser.add_argument('--diagnostic-run', default='runs/transition-replay-v4-final')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    run = Path(args.run).resolve()
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)

    train = json.loads((run/'training/results.json').read_text())
    config, rows = train['config'], train['episodes']
    check('fresh_v4_protocol', config['protocol'] == 'corrected_reference_assets_v4')
    check('four_full_episodes', [r['episode'] for r in rows] == list(range(4)) and
          all(r['control_steps'] == 1000 for r in rows))
    check('complete_transitions_and_real_updates', all(
        r['missing_transition_keys'] == 0 and r['sac_updates'] == 5 and
        r['nonterminal_transitions'] > 0 and r['terminal_transitions'] > 0 and
        r['risk_update_calls'] > 0 and r['actor_parameters_changed'] and
        r['risk_parameters_changed'] for r in rows))
    check('finite_episode_results', all(math.isfinite(r[k]) for r in rows for k in
          ('reward_sum', 'awt_lane_time_mean_seconds', 'aql_lane_time_mean_vehicles')))
    resume = json.loads((run/'training/resume-config.json').read_text())
    check('actual_same_source_resume', bool(resume.get('resumed_from')) and
          resume['code_sha256'] == config['code_sha256'])
    check('current_source_matches_training', all(hashlib.sha256((root/n).read_bytes()).hexdigest() == h
          for n, h in config['code_sha256'].items()))
    check('original_simulator_assets', all(hashlib.sha256((root/n).read_bytes()).hexdigest() == h
          for n, h in config['original_assets']['sha256'].items()))

    step_files = list((run/'training').glob('episode-*/steps.jsonl'))
    consistent_events = True
    exits = 0
    for path in step_files:
        steps = [json.loads(line) for line in path.read_text().splitlines()]
        check(f'{path.parent.name}_1000_step_records', len(steps) == 1000)
        for step in steps:
            expected = {vid for e in step['junction_collision_records'] for vid in (e['collider'], e['victim'])}
            consistent_events &= expected == set(step['junction_collision_vehicle_ids'])
            exits += len(step['unmapped_control_exit_ids'])
    check('collision_participants_match_recorded_events', consistent_events and len(step_files) == 4)
    check('exit_log_summary_agreement', exits == sum(r['unmapped_control_exit_transitions'] for r in rows))
    diagnostic = Path(args.diagnostic_run).resolve()
    replay = json.loads((diagnostic/'result.json').read_text())['summary']
    replay_config = json.loads((diagnostic/'diagnostic-config.json').read_text())
    check('final_source_failing_episode_replay', replay['control_steps'] == 1000 and
          replay['missing_transition_keys'] == 0 and replay['sac_updates'] == 5 and
          replay['risk_update_calls'] == 5 and replay_config['diagnostic_only'] and
          replay_config['source_sha256'] == config['code_sha256'])
    events = [json.loads(line) for line in (diagnostic/'transition-events.jsonl').read_text().splitlines()]
    check('original_288s_teleport_handled', any(e['simulation_time'] == 288. and
          '877763350#5.48' in e['handled_real_ids'] and not e['missing_keys'] and
          e['vehicles']['877763350#5.48']['done'] for e in events))
    replay_steps = [json.loads(line) for line in (diagnostic/'episode-0008/steps.jsonl').read_text().splitlines()]
    check('teleported_collision_participants_preserved', any(s['simulation_time'] == 288. and
          {'877763350#5.48', '36913228.3'} <= set(s['junction_collision_vehicle_ids']) for s in replay_steps))
    evaluations = []
    for checkpoint in (1, 3):
        dest = run/'validation'/f'checkpoint-{checkpoint:04d}'/'evaluation'
        result = json.loads((dest/'results.json').read_text())
        eval_rows = result['episodes']
        check(f'checkpoint_{checkpoint}_same_source', result['config']['code_sha256'] == config['code_sha256'])
        check(f'checkpoint_{checkpoint}_fixed_seeds', [r['sumo_seed'] for r in eval_rows] == [7001, 7002, 7003])
        check(f'checkpoint_{checkpoint}_evaluation_contract', all(
            r['control_steps'] == 1000 and r['missing_transition_keys'] == 0 and
            not r['training_conflict_proxy'] and not r['risk_at_inference'] and
            r['junction_entries_in_window'] > 0 for r in eval_rows))
        evaluations.extend(eval_rows)
    report = dict(passed=all(checks.values()), checks=checks,
                  training_episodes=len(rows), validation_episodes=len(evaluations),
                  unmapped_control_exit_transitions=exits,
                  original_paper_reproduced=False, gpu_execution_tested=False,
                  scope='Bounded CPU execution gate; not convergence, safety superiority, or exact paper reproduction.')
    (run/'readiness-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
