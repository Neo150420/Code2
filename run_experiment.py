"""Run one full corrected baseline seed, then evaluate its final actor.

Status is persisted; no claim is made that a process survives environment shutdown.
Re-run this command after shutdown to resume at the last complete training episode.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from reproduction_contract import bind_contract, file_sha256, validate_contract
from checkpoint_validation import SEEDS

ROOT=Path(__file__).resolve().parent


def aggregate_final_evaluation(rows):
    if not rows or any(r['training_conflict_proxy'] or r['risk_at_inference'] or
                       r['missing_transition_keys'] for r in rows):
        raise ValueError('Invalid final evaluation or training-only mechanism present')
    means={k:sum(r[k] for r in rows)/len(rows) for k in
           ('awt_lane_time_mean_seconds','aql_lane_time_mean_vehicles')}
    if not all(math.isfinite(value) for value in means.values()):
        raise ValueError('Nonfinite final efficiency metrics')
    events=sum(r['junction_collision_pair_events_in_window'] for r in rows)
    entries=sum(r['junction_entries_in_window'] for r in rows)
    means['collision_pair_events_per_junction_entry_percent']=100*events/entries if entries else None
    return means,events,entries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='full-v4-seed-0')
    parser.add_argument('--episodes',type=int,default=500)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--eval-sumo-seed',type=int,default=10001)
    parser.add_argument('--eval-episodes',type=int,default=3)
    parser.add_argument('--review-at',type=int,help='End training naturally at this episode, validate, then await review')
    parser.add_argument('--contract',help='Frozen declared-assumption reproduction contract')
    args=parser.parse_args()
    if args.episodes < 1: parser.error('Positive training target required')
    if Path(args.name).name != args.name: parser.error('Run name must be a simple directory name')
    if args.eval_sumo_seed <= args.episodes:
        parser.error('Evaluation traffic seeds must be outside training SUMO seeds 1..episodes')
    if args.eval_episodes < 1: parser.error('Positive evaluation budget required')
    if args.review_at is not None and not 1 <= args.review_at < args.episodes:
        parser.error('Review milestone must be positive and below the full training target')
    run=ROOT/'runs'/args.name
    run.mkdir(parents=True,exist_ok=True)
    run_lock=(run/'manager.lock').open('a')
    try:
        fcntl.flock(run_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        run_lock.close()
        parser.error('This experiment is already running; inspect status.json instead of starting a duplicate')
    try:
        return run_locked(args,run,parser)
    finally:
        run_lock.close()


def run_locked(args,run,parser):
    contract=None
    contract_hash=None
    if (run/'reproduction-contract.json').exists() and not args.contract:
        parser.error('This run has a frozen contract; provide --contract on every continuation')
    if args.contract:
        try:
            contract=json.loads(Path(args.contract).read_text())
        except (ValueError,FileNotFoundError) as exc:
            parser.error(str(exc))
        if (args.seed!=contract['fixed_cli']['seed'] or args.episodes!=contract['training_target_episodes'] or
            args.eval_sumo_seed!=contract['final_test_seeds'][0] or
            args.eval_episodes!=len(contract['final_test_seeds'])):
            parser.error('Manager budget/seeds differ from the frozen reproduction contract')
        if args.review_at is not None and args.review_at not in contract['review_milestones']:
            parser.error('Review milestone not declared in frozen contract')
        expected_cli=dict(seed=args.seed,sumo_seed=1,horizon=1000,warmup=500,update_every=200,threads=2,device='cpu')
        if (contract['fixed_cli']!=expected_cli or contract['validation_traffic_seeds']!=SEEDS or
            contract['final_test_seeds']!=list(range(args.eval_sumo_seed,args.eval_sumo_seed+args.eval_episodes))):
            parser.error('Execution settings differ from frozen contract')
        try:
            contract,contract_hash=bind_contract(ROOT,run,args.contract)
        except (ValueError,KeyError,FileNotFoundError) as exc:
            parser.error(str(exc))
    train=run/'training'
    evaluation=run/'evaluation'
    status_file=run/'status.json'
    status=dict(experiment='corrected reference-asset SAC-RWB; exact paper reproduction unverified',
                episodes_target=args.episodes, current_training_budget=args.review_at or args.episodes,
                contract_sha256=contract_hash, classification=(contract['classification'] if contract else 'corrected_reference_assets'),
                training_directory=str(train),
                evaluation_directory=str(evaluation), manager_pid=os.getpid(),
                started_unix=time.time(), device='cpu',training_seed=args.seed,
                evaluation_traffic_seeds=list(range(args.eval_sumo_seed,args.eval_sumo_seed+args.eval_episodes)))
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
                   '--episodes',str(args.review_at or args.episodes),'--horizon','1000','--warmup','500',
                   '--seed',str(args.seed),'--sumo-seed','1','--update-every','200','--threads','2','--device','cpu']
    if train.exists():
        resume=train/'resume.pt'
        if not resume.exists():
            save('blocked',reason='Existing training directory has no full resume checkpoint; preserve and inspect it.')
            raise SystemExit(2)
        train_command+=['--resume',str(resume)]
    execute(train_command,run/'training.log','training')
    if args.review_at is not None:
        milestones=sorted({n for n in (1,3,args.review_at) if n<=args.review_at})
        # Existing pilot evaluations may be reused only for these exact weights
        # and source. A stale directory must never silently count as validation.
        for n in milestones:
            saved=run/'validation'/f'checkpoint-{n:04d}'/'summary.json'
            if saved.exists():
                old=json.loads(saved.read_text())
                checkpoint=train/f'checkpoint-{n:04d}.pt'
                eval_config=json.loads((saved.parent/'evaluation/config.json').read_text())
                train_config=json.loads((train/'config.json').read_text())
                if (old['checkpoint_sha256']!=file_sha256(checkpoint) or
                    eval_config['code_sha256']!=train_config['code_sha256']):
                    save('blocked',reason=f'Existing validation identity differs: checkpoint {n}')
                    raise SystemExit(2)
        execute([sys.executable,str(ROOT/'checkpoint_validation.py'),'--run',args.name,
                 '--checkpoints',*[str(n) for n in milestones],'--workers','1'],
                run/'review-validation.log','checkpoint_validation')
        save('awaiting_checkpoint_review',completed_training_episodes=args.review_at,
             comparison=str(run/'validation/comparison.md'),
             final_test_evaluated=False,
             reason='Episode-boundary stop; inspect paired validation before explicit continuation')
        return
    checkpoint=train/f'checkpoint-{args.episodes:04d}.pt'
    if contract:
        try:
            validate_contract(ROOT,contract,json.loads((train/'results.json').read_text())['config'])
        except (ValueError,KeyError,FileNotFoundError) as exc:
            save('blocked',reason=str(exc))
            raise SystemExit(2)
    if evaluation.exists():
        save('blocked',reason='Evaluation directory already exists; inspect existing evidence before rerunning.')
        raise SystemExit(2)
    execute([sys.executable,str(ROOT/'baseline.py'),'evaluate','--out',str(evaluation),
             '--checkpoint',str(checkpoint),'--episodes',str(args.eval_episodes),'--horizon','1000',
             '--warmup','500','--sumo-seed',str(args.eval_sumo_seed),'--seed','1000'],run/'evaluation.log','evaluation')
    results=json.loads((evaluation/'results.json').read_text())
    rows=results['episodes']
    if [r['sumo_seed'] for r in rows]!=status['evaluation_traffic_seeds']:
        save('blocked',reason='Final traffic seed identity differs')
        raise SystemExit(2)
    trained_config=json.loads((train/'results.json').read_text())['config']
    if results['config']['code_sha256']!=trained_config['code_sha256']:
        save('blocked',reason='Final training/evaluation source differs')
        raise SystemExit(2)
    metrics,events,entries=aggregate_final_evaluation(rows)
    report=dict(label=f'Single trained seed; {args.eval_episodes} traffic evaluation seeds. Not independent training seeds.',
                exact_paper_reproduction=False, checkpoint=str(checkpoint),
                metrics=metrics,pooled_collision_pair_events=events,pooled_junction_entries=entries,
                metric_warning='Paper protocol/map/CR correspondence unresolved; do not claim SOTA or exact numerical reproduction.',
                episodes=rows)
    (run/'evaluation-summary.json').write_text(json.dumps(report,indent=2))
    save('completed',evaluation_summary=str(run/'evaluation-summary.json'))


if __name__=='__main__': main()
