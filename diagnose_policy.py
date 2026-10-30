"""Common-state action/critic probes and an explicitly offline SAC budget test.

No simulator rollout, risk retraining, new method or performance reproduction is
claimed by this diagnostic. Original checkpoints and source remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from baseline import build_agent, restore_training, seed_all
from algos.transformer_risk_model import TransformerRiskPredictor


def action_stats(acc):
    return dict(n=len(acc),mean=float(acc.mean()),p05=float(np.quantile(acc,.05)),
                median=float(np.median(acc)),p95=float(np.quantile(acc,.95)),
                braking_fraction=float(np.mean(acc<-.3)),
                strong_braking_fraction=float(np.mean(acc<-1.5)),
                acceleration_fraction=float(np.mean(acc>.3))) if len(acc) else dict(n=0)


def probe(agent, states):
    actions=agent.select_action(states,True)[:,0]*3
    empty=np.max(states[:,18:],axis=1)==0
    stopped=states[:,1]<.05
    grid=np.array([-1.,-.5,0.,.5,1.],dtype=np.float32)
    values=[]
    with torch.no_grad():
        s=torch.as_tensor(states,dtype=torch.float32)
        for a in grid:
            q1,q2=agent.q_critic(s,torch.full((len(states),1),float(a)))
            values.append(torch.minimum(q1,q2).numpy()[:,0])
    values=np.stack(values,axis=1)
    best=grid[np.argmax(values,axis=1)]*3
    return dict(all_states=action_stats(actions),empty_junction=action_stats(actions[empty]),
                empty_junction_stopped=action_stats(actions[empty&stopped]),
                occupied_junction=action_stats(actions[~empty]),
                critic_grid_accelerations=(grid*3).tolist(),
                critic_mean_grid_q=values.mean(axis=0).tolist(),
                critic_grid_preferred_actions=action_stats(best),
                critic_empty_junction_preferred_actions=action_stats(best[empty]))


def risk_probe(model, buffer, rng):
    samples=[]
    for source in (list(buffer.active_buffer),list(buffer.risk_active_buffer)):
        samples.extend(source[i] for i in rng.choice(len(source),min(128,len(source)),replace=False))
    s=torch.as_tensor(np.stack([r['state_seq'] for r in samples]),dtype=torch.float32)
    a=torch.as_tensor(np.stack([r['action_seq'] for r in samples]),dtype=torch.float32)
    y=np.array([r['label'] for r in samples])
    model.eval()
    with torch.no_grad():
        native=model(s,a).numpy()
        candidates={}
        for last_action in (-3.,0.,3.):
            changed=a.clone();changed[:,-1,0]=last_action
            candidates[str(last_action)]=float(model(s,changed).mean())
    p=np.clip(native,1e-6,1-1e-6)
    return dict(n=len(y),labelled_safe_mean=float(native[y==0].mean()),
                labelled_risk_mean=float(native[y==1].mean()),
                brier=float(np.mean((native-y)**2)),
                nll=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
                changed_last_action_mean_prediction=candidates,
                limitation='Balanced training-buffer samples: descriptive fit, not held-out calibration or causal action effects')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',default='artifacts/checkpoint10-review-2026-10-30/training')
    parser.add_argument('--out',required=True)
    parser.add_argument('--updates',nargs='+',type=int,default=[50,250,1000])
    args=parser.parse_args()
    if not args.updates or min(args.updates)<1:parser.error('Positive update budgets required')
    train=Path(args.training).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);seed_all(90210)
    data=torch.load(train/'resume.pt',map_location='cpu')
    if data['metadata']['completed_episodes']!=10:
        raise ValueError('This diagnostic requires the frozen completed episode-10 parent, not a later live resume')
    rng=np.random.default_rng(90210)
    ids=rng.choice(data['replay']['size'],min(4096,data['replay']['size']),replace=False)
    states=data['replay']['tensors']['s'][ids].numpy()
    np.savez_compressed(out/'common-state-panel.npz',indices=ids,states=states)
    report=dict(diagnostic_only=True,original_paper_reproduced=False,
                data_source='Fixed training-replay state panel from completed checkpoint 10; not final test data',
                parent_resume_sha256=hashlib.sha256((train/'resume.pt').read_bytes()).hexdigest(),
                diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                training_replay_size=data['replay']['size'],checkpoints={},offline_updates={})
    for n in (1,3,10):
        ck=torch.load(train/f'checkpoint-{n:04d}.pt',map_location='cpu')
        agent=build_agent('cpu',capacity=8)
        agent.actor.load_state_dict(ck['actor']);agent.q_critic.load_state_dict(ck['critic'])
        risk=TransformerRiskPredictor(98,1);risk.load_state_dict(ck['risk'])
        report['checkpoints'][str(n)]=dict(policy=probe(agent,states),
                                         risk=risk_probe(risk,data['risk_buffer'],np.random.default_rng(90211)))
    del data
    agent=build_agent('cpu');risk=TransformerRiskPredictor(98,1)
    restore_training(train/'resume.pt',agent,risk)
    seed_all(90210)
    for step in range(1,max(args.updates)+1):
        loss=agent.train()
        if not all(v is None or np.isfinite(v) for v in loss.values()):raise RuntimeError('Nonfinite offline loss')
        if step in args.updates:
            report['offline_updates'][str(step)]=dict(policy=probe(agent,states),loss=loss)
            torch.save(dict(actor=agent.actor.state_dict(),critic=agent.q_critic.state_dict(),
                            metadata=dict(diagnostic_only=True,offline_updates=step,parent=str(train/'resume.pt'))),
                       out/f'offline-{step:04d}.pt')
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
