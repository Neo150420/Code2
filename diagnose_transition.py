"""Replay one saved episode for debugging; never modifies the source experiment."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import baseline


def vehicle_state(env, key):
    real = env.convert_virtual_id_to_real_id(key)
    veh = env.rl_vehicles.get(real)
    state = dict(real_id=real, mapping=env.veh_name_mapping_table.get(real),
                 present_in_sumo=real in env.sumo_interface.tc.vehicle.getIDList())
    if veh is not None:
        state.update({k: veh.get(k) for k in ('road_id', 'lane_index', 'laneposition', 'speed')})
        state['junction_direction'] = env.map.get_veh_moving_direction(veh) if veh.road_id else None
        state['needs_control'] = env.need_to_control(veh)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--allow-code-change', action='store_true',
                        help='Explicitly label a diagnostic replay under corrected source, not an original-run resume')
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    baseline.seed_all(0)
    agent = baseline.build_agent('cpu')
    predictor = baseline.TransformerRiskPredictor(98, 1)
    metadata = baseline.restore_training(Path(args.resume).resolve(), agent, predictor)
    config = metadata['config']
    hashes = {name: hashlib.sha256((baseline.ROOT/name).read_bytes()).hexdigest()
              for name in config['code_sha256']}
    changed = [name for name in hashes if hashes[name] != config['code_sha256'][name]]
    if changed and not args.allow_code_change:
        raise ValueError(f'Diagnostic source differs: {changed}; explicit diagnostic flag required')
    (out/'diagnostic-config.json').write_text(json.dumps(dict(
        parent_checkpoint=str(Path(args.resume).resolve()), source_sha256=hashes,
        changed_source=changed, diagnostic_only=True, original_paper_reproduced=False,
        parent_completed_episodes=metadata['completed_episodes']), indent=2))
    original_step = baseline.Env.step

    def traced_step(env, actions=None, risks=None):
        actions = actions or {}
        before = {k: vehicle_state(env, k) for k in actions}
        result = original_step(env, actions, risks)
        missing = set(actions) - set(result[1])
        handled = getattr(env, 'last_unmapped_control_exit_ids', set())
        if missing or handled:
            interesting = missing | {k for k in actions if env.convert_virtual_id_to_real_id(k) in handled}
            report = dict(simulation_time=env.sumo_interface.tc.simulation.getTime(),
                          missing_keys=sorted(missing), handled_real_ids=sorted(handled),
                          starting_teleports=list(env.sumo_interface.tc.simulation.getStartingTeleportIDList()),
                          ending_teleports=list(env.sumo_interface.tc.simulation.getEndingTeleportIDList()),
                          actual_collision_ids=sorted(env.last_actual_collision_ids),
                          collisions=[dict(collider=e.collider, victim=e.victim, lane=e.lane, type=e.type)
                                      for e in env.sumo_interface.tc.simulation.getCollisions()],
                          vehicles={k: dict(before=before[k], after=vehicle_state(env, k),
                                           reward=result[1].get(k), done=result[2].get(k)) for k in sorted(interesting)})
            with (out/'transition-events.jsonl').open('a') as stream:
                stream.write(json.dumps(report)+'\n')
            if missing:
                (out/'missing-transitions.json').write_text(json.dumps(report, indent=2))
        return result

    baseline.Env.step = traced_step
    options = SimpleNamespace(**config)
    options.resume = args.resume
    options.out = str(out)
    options.threads = 2
    summary, step = baseline.run_episode(agent, predictor, options, out,
                                        metadata['completed_episodes'], metadata['training_loop_steps'])
    (out/'result.json').write_text(json.dumps(dict(summary=summary, global_step=step), indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
