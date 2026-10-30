"""Corrected SAC-RWB reference-asset implementation, not verified paper reproduction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch

from Env import Env
from algos.transformer_risk_model import TransformerRiskPredictor
from sac.SAC import SAC_countinuous

ROOT = Path(__file__).resolve().parent


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_agent(device, capacity=1_000_000):
    return SAC_countinuous(state_dim=98, action_dim=1, net_width=256,
                          dvc=torch.device(device), a_lr=3e-4, c_lr=4e-4,
                          gamma=0.99, batch_size=256, alpha=0.12,
                          adaptive_alpha=True, replay_capacity=capacity)


def risk_batch(model, env, obs, actions, device):
    keys = list(obs)
    states, acts = [], []
    for key in keys:
        vid = env.convert_virtual_id_to_real_id(key)
        model.buffer.add(vid, obs[key], np.array([actions[key]], dtype=np.float32))
        s, a = model.buffer.get_sequence(vid)
        states.append(s[0])
        acts.append(a[0])
    model.eval()
    with torch.no_grad():
        scores = model(torch.as_tensor(np.stack(states), dtype=torch.float32, device=device),
                       torch.as_tensor(np.stack(acts), dtype=torch.float32, device=device))
    if not torch.isfinite(scores).all():
        raise RuntimeError('Nonfinite risk prediction')
    return dict(zip(keys, scores.cpu().tolist()))


def make_env(out, seed, horizon, train, predictor):
    tree = ET.parse(ROOT / 'real_data/osm.sumocfg')
    cfg = tree.getroot()
    for child in cfg.find('input'):
        child.set('value', str(ROOT / 'real_data' / child.get('value')))
    cfg.find('gui_only/gui-settings-file').set('value', str(ROOT / 'real_data/osm.view.xml'))
    # Avoid enormous per-timestep full-network dumps. Metrics are collected with TraCI.
    cfg.remove(cfg.find('output'))
    cfg.find('random/seed').set('value', str(seed))
    filename = out / 'simulation.sumocfg'
    tree.write(filename)
    config = dict(junction_list=['229', '499', '332', '334'], spawn_rl_prob={},
                  probablity_RL=1.0, cfg=str(filename), render=False,
                  map_xml=str(ROOT / 'real_data/CSeditClean_1.net_threelegs.xml'),
                  max_episode_steps=horizon, conflict_mechanism='flexible',
                  traffic_light_program=dict(disable_state='G', disable_light_start=0),
                  use_safety_predictor=train, risk_predictor=predictor,
                  training_conflict_proxy=train, sumo_seed=seed)
    return Env(config), config


def save_checkpoint(path, agent, predictor, metadata):
    data = dict(actor=agent.actor.state_dict(), critic=agent.q_critic.state_dict(),
                target_critic=agent.q_critic_target.state_dict(),
                risk=predictor.state_dict(), metadata=metadata,
                actor_optimizer=agent.actor_optimizer.state_dict(),
                critic_optimizer=agent.q_critic_optimizer.state_dict(),
                alpha_optimizer=agent.alpha_optim.state_dict(),
                log_alpha=agent.log_alpha.detach().cpu(),
                torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                python_rng=random.getstate(), resumable=False,
                resume_limitation='Replay and risk buffers not serialized; evaluation checkpoint only.')
    temp = path.with_suffix('.tmp')
    torch.save(data, temp)
    temp.replace(path)
    # Verify an independent model reload, not merely that a file exists.
    check = build_agent(agent.dvc, capacity=8)
    check.actor.load_state_dict(torch.load(path, map_location=agent.dvc)['actor'])
    test_s = np.linspace(0, 1, 98, dtype=np.float32)[None]
    np.testing.assert_array_equal(agent.select_action(test_s, True), check.select_action(test_s, True))


def run_episode(agent, predictor, args, out, episode, global_step):
    training = args.mode == 'train'
    sim_seed = args.sumo_seed + episode
    seed_all(args.seed + episode)
    episode_dir = out / f'episode-{episode:04d}'
    episode_dir.mkdir()
    os.chdir(episode_dir)
    if training:
        predictor.buffer.to_use_buffer.clear()  # Never link history across simulator resets.
    env, config = make_env(episode_dir, sim_seed, args.horizon, training, predictor)
    summary = dict(episode=episode, seed=args.seed + episode, sumo_seed=sim_seed,
                   stage='training pilot' if training else 'checkpoint evaluation',
                   training_conflict_proxy=training, risk_at_inference=training,
                   control_steps=0, transitions=0, nonterminal_transitions=0,
                   terminal_transitions=0, sac_updates=0, risk_update_calls=0,
                   risk_minibatches=0, missing_transition_keys=0, reward_sum=0.0)
    collision_ids, proxy_ids = set(), set()
    window_collision_ids = set()
    departed, arrived = set(), set()
    measurement_steps = 0
    started = time.monotonic()
    losses = []
    initial_actor = [p.detach().clone() for p in agent.actor.parameters()]
    metrics_file = (episode_dir / 'steps.jsonl').open('w')
    try:
        obs, _ = env.reset()
        while env.sumo_interface.tc.simulation.getTime() < args.horizon:
            tc = env.sumo_interface.tc
            if obs:
                keys = list(obs)
                state_array = np.stack([obs[k] for k in keys])
                if state_array.shape != (len(keys), 98) or not np.isfinite(state_array).all():
                    raise RuntimeError('Observation shape/finite check failed')
                a = agent.select_action(state_array, deterministic=not training)
                if a.shape != (len(keys), 1) or not np.isfinite(a).all():
                    raise RuntimeError('Scalar action contract failed')
                actions = dict(zip(keys, (a[:, 0] * env.max_acc).tolist()))
                risks = risk_batch(predictor, env, obs, actions, args.device) if training else None
            else:
                keys, actions, risks = [], {}, None
            next_obs, rewards, dones, truncated, info, transition_obs = env.step(actions, risks)
            summary['control_steps'] += 1
            global_step += 1
            if set(rewards) != set(actions):
                summary['missing_transition_keys'] += len(set(actions) ^ set(rewards))
                raise RuntimeError(f'Transition coverage mismatch: actions={set(actions)}, rewards={set(rewards)}')
            if training:
                for key in keys:
                    # Entering the junction terminates this access-control decision process.
                    terminal = bool(dones.get(key, False))
                    if key not in transition_obs:
                        raise RuntimeError(f'Missing next observation for {key}')
                    reward = float(rewards[key])
                    if not np.isfinite(reward): raise RuntimeError('Nonfinite reward')
                    agent.replay_buffer.add(obs[key], np.array([a[keys.index(key), 0]], dtype=np.float32),
                                            reward, transition_obs[key], terminal)
                    summary['transitions'] += 1
                    summary['terminal_transitions' if terminal else 'nonterminal_transitions'] += 1
                    summary['reward_sum'] += reward
                if global_step % args.update_every == 0 and agent.replay_buffer.size >= agent.batch_size:
                    batch_losses = agent.train()
                    if not all(v is None or np.isfinite(v) for v in batch_losses.values()):
                        raise RuntimeError('Nonfinite SAC loss')
                    losses.append(dict(step=global_step, **batch_losses))
                    summary['sac_updates'] += 1
                    half = 64
                    if len(predictor.buffer.active_buffer) >= half and len(predictor.buffer.risk_active_buffer) >= half:
                        minibatches = max(1, len(predictor.buffer) // 128) * 3
                        risk_loss = predictor.train_model(batch_size=128, epochs=3, lr=1e-5, device=args.device)
                        if not np.isfinite(risk_loss): raise RuntimeError('Nonfinite risk loss')
                        summary['risk_update_calls'] += 1
                        summary['risk_minibatches'] += minibatches
                        losses[-1]['risk_loss'] = risk_loss
            for key, done in dones.items():
                if key != '__all__' and done: next_obs.pop(key, None)
            obs = next_obs
            sim_t = tc.simulation.getTime()
            collision_ids.update(env.last_actual_collision_ids)
            proxy_ids.update(env.last_proxy_ids)
            departed.update(tc.simulation.getDepartedIDList())
            arrived.update(tc.simulation.getArrivedIDList())
            in_window = args.warmup <= sim_t < args.horizon
            if in_window:
                measurement_steps += 1
                window_collision_ids.update(env.last_actual_collision_ids)
            metrics_file.write(json.dumps(dict(simulation_time=sim_t,
                                               controlled=len(keys), actual_collision_vehicle_ids=sorted(env.last_actual_collision_ids),
                                               stopped_proxy_ids=sorted(env.last_proxy_ids))) + '\n')
        lo, hi = args.warmup, min(args.horizon, 4999)
        waits, queues = [], []
        crossings = 0
        for junction in env.junction_list:
            for direction in env.keywords_order:
                records = env.monitor.data_record[junction][direction]
                waits.extend(records['queue_wait'][lo:hi].tolist())
                queues.extend(records['queue_length'][lo:hi].tolist())
                crossings += int(records['throughput'][lo:hi].sum())
        summary.update(elapsed_seconds=time.monotonic()-started,
                       simulation_seconds=float(env.sumo_interface.tc.simulation.getTime()),
                       measurement_window_indices=[lo, hi], measurement_steps=measurement_steps,
                       awt_lane_time_mean_seconds=float(np.mean(waits)) if waits else None,
                       aql_lane_time_mean_vehicles=float(np.mean(queues)) if queues else None,
                       junction_entries_in_window=crossings,
                       unique_collision_vehicle_ids_in_window=len(window_collision_ids),
                       collision_vehicle_ids_per_junction_entry_percent=(100*len(window_collision_ids)/crossings if crossings else None),
                       unique_actual_collision_vehicles_total=len(collision_ids),
                       unique_training_proxy_vehicles=len(proxy_ids),
                       departures_after_reset=len(departed), arrivals_after_reset=len(arrived),
                       vehicles_remaining=len(env.sumo_interface.tc.vehicle.getIDList()),
                       risk_safe_buffer=len(predictor.buffer.active_buffer),
                       risk_collision_buffer=len(predictor.buffer.risk_active_buffer),
                       actor_parameters_changed=any(not torch.equal(a,b) for a,b in zip(initial_actor, agent.actor.parameters())),
                       last_losses=losses[-1] if losses else None)
        if training and (not summary['sac_updates'] or not summary['nonterminal_transitions']):
            raise RuntimeError('Pilot did not exercise nonterminal SAC training')
        (episode_dir/'losses.json').write_text(json.dumps(losses, indent=2))
        (episode_dir/'summary.json').write_text(json.dumps(summary, indent=2))
        return summary, global_step
    finally:
        metrics_file.close()
        env.close()
        os.chdir(ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'evaluate'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--horizon', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sumo-seed', type=int, default=1)
    parser.add_argument('--update-every', type=int, default=200)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.warmup < args.horizon < 5000:
        parser.error('Require 0 <= warmup < horizon < 5000')
    if args.episodes < 1 or args.update_every < 1: parser.error('Positive budgets required')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable; no silent CPU fallback')
    if args.mode == 'evaluate' and not args.checkpoint: parser.error('Evaluation requires a checkpoint')
    out = Path(args.out).resolve()
    if out.exists(): parser.error('Output directory already exists; choose a fresh run directory')
    out.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    agent = build_agent(args.device, capacity=1_000_000 if args.mode=='train' else 8)
    predictor = TransformerRiskPredictor(98, 1).to(args.device)
    config = dict(vars(args), original_paper_reproduced=False,
                  protocol='corrected_reference_assets_v1', original_assets=json.loads((ROOT/'provenance.json').read_text()),
                  metric_note='Lane-time queue/wait averages and unique collision-involved vehicle IDs per junction entry; not guaranteed identical to paper CR.',
                  simulator_horizon_includes_reset_warmup=True)
    if args.checkpoint:
        path = Path(args.checkpoint).resolve()
        data = torch.load(path, map_location=args.device)
        agent.actor.load_state_dict(data['actor'])
        config['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'config.json').write_text(json.dumps(config, indent=2))
    summaries, step = [], 0
    for episode in range(args.episodes):
        summary, step = run_episode(agent, predictor, args, out, episode, step)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        if args.mode == 'train':
            save_checkpoint(out/f'checkpoint-{episode+1:04d}.pt', agent, predictor,
                            dict(config=config, completed_episodes=episode+1, training_loop_steps=step))
        (out/'results.json').write_text(json.dumps(dict(config=config, episodes=summaries), indent=2))


if __name__ == '__main__':
    main()
