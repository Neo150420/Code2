"""Run one full corrected baseline seed, then evaluate its final actor.

Status is persisted; no claim is made that a process survives environment shutdown.
Re-run this command after shutdown to resume at the last complete training episode.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='full-seed-0')
    parser.add_argument('--episodes',type=int,default=500)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--eval-sumo-seed',type=int,default=10001)
    args=parser.parse_args()
    if Path(args.name).name != args.name: parser.error('Run name must be a simple directory name')
    if args.eval_sumo_seed <= args.episodes:
        parser.error('Evaluation traffic seeds must be outside training SUMO seeds 1..episodes')
    run=ROOT/'runs'/args.name
    run.mkdir(parents=True,exist_ok=True)
    run_lock=(run/'manager.lock').open('a')
    try:
        fcntl.flock(run_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('This experiment is already running; inspect status.json instead of starting a duplicate')
    train=run/'training'
    evaluation=run/'evaluation'
    status_file=run/'status.json'
    status=dict(experiment='corrected reference-asset SAC-RWB; exact paper reproduction unverified',
                episodes_target=args.episodes, training_directory=str(train),
                evaluation_directory=str(evaluation), manager_pid=os.getpid(),
                started_unix=time.time(), device='cpu',training_seed=args.seed,
                evaluation_traffic_seeds=list(range(args.eval_sumo_seed,args.eval_sumo_seed+3)))
    env=os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE']='1'
    env['PATH']=str(Path(sys.executable).parent)+':/workspace/research/.sumo-venv/bin:'+env['PATH']
    child=None
    stop_requested=False

    def save(stage,**fields):
        status.update(stage=stage,updated_unix=time.time(),**fields)
        tmp=status_file.with_suffix('.tmp')
        tmp.write_text(json.dumps(status,indent=2))
        tmp.replace(status_file)

    def stop(sig,frame):
        nonlocal stop_requested
        stop_requested=True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            # Never call wait reentrantly from a signal handler: the interrupted
            # Popen.wait may already hold its internal waitpid lock.
            return
        save('interrupted',reason='Process interrupted; restart explicitly from saved episode checkpoint')
        raise SystemExit(130)

    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)

    def execute(command,log,stage):
        nonlocal child
        save(stage,command=command)
        with log.open('a') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            save(stage,child_pid=child.pid)
            code=child.wait()
        if stop_requested:
            save('interrupted',reason='Process interrupted; restart explicitly from saved episode checkpoint')
            raise SystemExit(130)
        if code:
            save('failed',failed_stage=stage,exit_code=code,log=str(log))
            raise SystemExit(code)

    train_command=[sys.executable,str(ROOT/'baseline.py'),'train','--out',str(train),
                   '--episodes',str(args.episodes),'--horizon','1000','--warmup','500','--seed',str(args.seed)]
    if train.exists():
        resume=train/'resume.pt'
        if not resume.exists():
            save('blocked',reason='Existing training directory has no full resume checkpoint; preserve and inspect it.')
            raise SystemExit(2)
        train_command+=['--resume',str(resume)]
    execute(train_command,run/'training.log','training')
    checkpoint=train/f'checkpoint-{args.episodes:04d}.pt'
    if evaluation.exists():
        save('blocked',reason='Evaluation directory already exists; inspect existing evidence before rerunning.')
        raise SystemExit(2)
    execute([sys.executable,str(ROOT/'baseline.py'),'evaluate','--out',str(evaluation),
             '--checkpoint',str(checkpoint),'--episodes','3','--horizon','1000',
             '--warmup','500','--sumo-seed',str(args.eval_sumo_seed),'--seed','1000'],run/'evaluation.log','evaluation')
    results=json.loads((evaluation/'results.json').read_text())
    rows=results['episodes']
    report=dict(label='Single trained seed; three traffic evaluation seeds. Not three independent training seeds.',
                exact_paper_reproduction=False, checkpoint=str(checkpoint),
                metrics={name:(sum(r[name] for r in rows)/len(rows) if all(r[name] is not None for r in rows) else None) for name in
                         ('awt_lane_time_mean_seconds','aql_lane_time_mean_vehicles',
                          'collision_pair_events_per_junction_entry_percent')},
                metric_warning='Paper protocol/map/CR correspondence unresolved; do not claim SOTA or exact numerical reproduction.',
                episodes=rows)
    (run/'evaluation-summary.json').write_text(json.dumps(report,indent=2))
    save('completed',evaluation_summary=str(run/'evaluation-summary.json'))


if __name__=='__main__': main()
