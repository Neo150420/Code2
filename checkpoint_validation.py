"""Matched early-checkpoint validation, separated from final test traffic seeds."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
SEEDS=[7001,7002,7003]


def atomic_json(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2))
    temp.replace(path)


def summarize(checkpoint,results):
    rows=results['episodes']
    if len(rows)!=3 or [r['sumo_seed'] for r in rows]!=SEEDS:
        raise ValueError('Expected exactly three matched validation traffic seeds')
    if any(r['training_conflict_proxy'] or r['risk_at_inference'] for r in rows):
        raise ValueError('Training-only mechanism present during validation')
    if any(r['missing_transition_keys'] for r in rows):
        raise ValueError('Missing evaluation transitions')
    entries=sum(r['junction_entries_in_window'] for r in rows)
    events=sum(r['junction_collision_pair_events_in_window'] for r in rows)
    return dict(checkpoint=checkpoint,traffic_seeds=SEEDS,
                awt_seconds=sum(r['awt_lane_time_mean_seconds'] for r in rows)/3,
                aql_vehicles=sum(r['aql_lane_time_mean_vehicles'] for r in rows)/3,
                collision_pair_events=events,junction_entries=entries,
                collision_events_per_entry_percent=100*events/entries if entries else None,
                mean_arrivals_after_reset=sum(r['arrivals_after_reset'] for r in rows)/3,
                mean_vehicles_remaining=sum(r['vehicles_remaining'] for r in rows)/3,
                episode_results=rows)


def write_comparison(out,summaries):
    rows=sorted(summaries,key=lambda r:r['checkpoint'])
    fields=['checkpoint','awt_seconds','aql_vehicles','collision_pair_events','junction_entries',
            'collision_events_per_entry_percent','mean_arrivals_after_reset','mean_vehicles_remaining']
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)
    text=['# Fixed-seed early validation','',
          'Traffic seeds: 7001, 7002, 7003. Final test seeds 10001–10003 are not used.',
          'One trained model at different checkpoints; three traffic seeds are not three training seeds.',
          'Collision rate below is selected-junction collision-pair onset events / junction entries,',
          'not established as identical to the paper CR. These are validation results, not final test results.','',
          '| Checkpoint | AWT (s) | AQL (veh) | Collision events | Entries | Events / entry (%) | Mean network arrivals* | Mean vehicles remaining |',
          '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        rate='undefined' if r['collision_events_per_entry_percent'] is None else f"{r['collision_events_per_entry_percent']:.3f}"
        text.append(f"| {r['checkpoint']} | {r['awt_seconds']:.3f} | {r['aql_vehicles']:.3f} | {r['collision_pair_events']} | {r['junction_entries']} | {rate} | {r['mean_arrivals_after_reset']:.1f} | {r['mean_vehicles_remaining']:.1f} |")
    text+=['','*Network arrivals counted after Env.reset; not a completed-trip fraction.',
           'Do not reward an apparent collision-rate decrease caused by reduced intersection entry.',
           'Review waiting, entry flow, arrivals and remaining vehicles together.']
    if len(rows)>=2:
        first,last=rows[0],rows[-1]
        pairs=[]
        for a,b in zip(first['episode_results'],last['episode_results']):
            pairs.append(dict(sumo_seed=a['sumo_seed'],
                awt_delta_seconds=b['awt_lane_time_mean_seconds']-a['awt_lane_time_mean_seconds'],
                collision_events_delta=b['junction_collision_pair_events_in_window']-a['junction_collision_pair_events_in_window'],
                entries_delta=b['junction_entries_in_window']-a['junction_entries_in_window'],
                vehicles_remaining_delta=b['vehicles_remaining']-a['vehicles_remaining']))
        atomic_json(out/'paired-deltas.json',dict(reference=first['checkpoint'],candidate=last['checkpoint'],pairs=pairs,
            warning='Three paired traffic seeds are a screening check, not a statistically established safety improvement.'))
    (out/'comparison.md').write_text('\n'.join(text)+'\n')
    atomic_json(out/'summary.json',dict(validation_seeds=SEEDS,final_test_seeds_untouched=[10001,10002,10003],
        checkpoints=rows,paper_reproduction=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='full-v4-seed-0')
    parser.add_argument('--checkpoints',nargs='+',type=int,required=True)
    parser.add_argument('--pause-training',action='store_true')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--pause-on-ready',action='store_true',help='Wait without pausing, then pause when the requested checkpoint is complete')
    parser.add_argument('--hold-after-validation',action='store_true',help='Leave training paused for review after evaluation')
    args=parser.parse_args()
    if Path(args.run).name!=args.run:parser.error('Run must be a simple directory name')
    if args.watch and args.pause_training:parser.error('Do not pause training while waiting for future checkpoints')
    if args.hold_after_validation and not (args.pause_training or args.pause_on_ready):
        parser.error('A review hold requires an explicit training-pause option')
    if any(n<1 for n in args.checkpoints):parser.error('Checkpoints must be positive episode numbers')
    if not 1<=args.workers<=2:parser.error('Use one or two validation processes on this cloud')
    run=ROOT/'runs'/args.run
    out=run/'validation'
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'monitor.lock').open('a')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:parser.error('Validation monitor already active')
    status_file=out/'status.json'
    summaries=[]
    for p in sorted(out.glob('checkpoint-*/summary.json')):
        summaries.append(json.loads(p.read_text()))
    completed={r['checkpoint'] for r in summaries}
    requested=sorted(set(args.checkpoints))
    paused_pid=None
    succeeded=False

    def pause_training():
        nonlocal paused_pid
        if paused_pid is not None:return
        s=json.loads((run/'status.json').read_text())
        if s['stage']!='training':raise RuntimeError('Manager is not in the training stage')
        pid=s['child_pid']
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if str(ROOT/'baseline.py').encode() not in cmd or b'train' not in cmd:
            raise RuntimeError('Recorded PID is not the expected training process')
        os.kill(pid,signal.SIGSTOP)
        paused_pid=pid
        atomic_json(out/'training-pause.json',dict(pid=pid,paused=True,reason='Matched checkpoint validation',
                                                  paused_unix=time.time()))

    def status(stage,**extra):
        atomic_json(status_file,dict(stage=stage,monitor_pid=os.getpid(),requested=requested,
                    completed=sorted(completed),validation_seeds=SEEDS,updated_unix=time.time(),**extra))

    def interrupted(sig,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)

    def evaluate(n):
        checkpoint=run/'training'/f'checkpoint-{n:04d}.pt'
        training_config=json.loads((run/'training/config.json').read_text())
        for filename,digest in training_config['code_sha256'].items():
            if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'Training/evaluation source mismatch: {filename}')
        dest=out/f'checkpoint-{n:04d}'
        if dest.exists():
            # Preserve a failed/partial validation, rather than merging old result files.
            archive=out/'interrupted'
            archive.mkdir(exist_ok=True)
            dest.rename(archive/f'{dest.name}-{time.time_ns()}')
        dest.mkdir()
        evaluation=dest/'evaluation'
        env=os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE']='1'
        env['PATH']=str(Path(sys.executable).parent)+':/workspace/research/.sumo-venv/bin:'+env['PATH']
        command=[sys.executable,str(ROOT/'baseline.py'),'evaluate','--out',str(evaluation),
                 '--checkpoint',str(checkpoint),'--episodes','3','--horizon','1000','--warmup','500',
                 '--sumo-seed',str(SEEDS[0]),'--seed','2000','--threads','1']
        with (dest/'evaluation.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Checkpoint {n} validation exited {result.returncode}; see {dest}')
        results=json.loads((evaluation/'results.json').read_text())
        summary=summarize(n,results)
        summary['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        atomic_json(dest/'summary.json',summary)
        return summary

    try:
        if args.pause_training:
            pause_training()
        while not set(requested)<=completed:
            # results.json is written after the full resume file commits. Waiting
            # for it avoids pausing halfway through saving episode-boundary state.
            try:
                trained_episodes=len(json.loads((run/'training/results.json').read_text())['episodes'])
            except (FileNotFoundError,json.JSONDecodeError):
                trained_episodes=0
            ready=[n for n in requested if n not in completed and n<=trained_episodes and (run/'training'/f'checkpoint-{n:04d}.pt').exists()]
            if not ready:
                if not args.watch:raise RuntimeError('Requested checkpoints are not yet available')
                status('waiting_for_checkpoints')
                time.sleep(10)
                continue
            if args.pause_on_ready:pause_training()
            status('evaluating',active=ready)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures={pool.submit(evaluate,n):n for n in ready}
                for future in as_completed(futures):
                    summary=future.result()
                    summaries.append(summary)
                    completed.add(summary['checkpoint'])
                    write_comparison(out,summaries)
                    status('evaluating',active=[n for n in ready if n not in completed])
        succeeded=True
        status('completed_awaiting_review' if args.hold_after_validation else 'completed',comparison=str(out/'comparison.md'))
    except BaseException as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if paused_pid is not None:
            if args.hold_after_validation:
                atomic_json(out/'training-pause.json',dict(pid=paused_pid,paused=True,
                    reason='Awaiting checkpoint review' if succeeded else 'Validation failed; inspect before resuming',
                    checkpoint_review=requested,updated_unix=time.time()))
            else:
                os.kill(paused_pid,signal.SIGCONT)
                atomic_json(out/'training-pause.json',dict(pid=paused_pid,paused=False,resumed_unix=time.time()))


if __name__=='__main__':main()
