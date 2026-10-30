"""Continue the frozen baseline budget through monitored execution gates.

This automates execution integrity checks, not scientific superiority decisions.
No rewards, assets, hyperparameters or final-test selection are changed.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from reproduction_contract import file_sha256, validate_contract

ROOT=Path(__file__).resolve().parent


def atomic_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def manager_alive(pid):
    try:
        state=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()[0]
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return state!='Z' and str(ROOT/'run_experiment.py').encode() in cmd
    except (FileNotFoundError,IndexError,ProcessLookupError):
        return False


def review_execution(root,run,contract,episode):
    results=json.loads((run/'training/results.json').read_text())
    validate_contract(root,contract,results['config'])
    rows=results['episodes']
    if len(rows)!=episode or [r['episode'] for r in rows]!=list(range(episode)):
        raise ValueError('Completed episode identity differs at review gate')
    if any(r['control_steps']!=1000 or r['missing_transition_keys'] for r in rows):
        raise ValueError('Incomplete training or missing transitions')
    for row in rows:
        if not math.isfinite(row['reward_sum']) or any(
            v is not None and not math.isfinite(v) for k,v in row['last_losses'].items()):
            raise ValueError('Nonfinite training result')
    dest=run/'validation'/f'checkpoint-{episode:04d}'
    summary=json.loads((dest/'summary.json').read_text())
    evaluated=json.loads((dest/'evaluation/results.json').read_text())
    if summary['checkpoint_sha256']!=file_sha256(run/'training'/f'checkpoint-{episode:04d}.pt'):
        raise ValueError('Validation checkpoint identity differs')
    if evaluated['config']['code_sha256']!=contract['critical_source_sha256']:
        raise ValueError('Validation source identity differs')
    if [r['sumo_seed'] for r in evaluated['episodes']]!=contract['validation_traffic_seeds']:
        raise ValueError('Validation seed identity differs')
    for row in evaluated['episodes']:
        if (row['missing_transition_keys'] or row['training_conflict_proxy'] or row['risk_at_inference'] or
            row['control_steps']!=1000 or row['junction_entries_in_window']<=0):
            raise ValueError('Evaluation integrity or positive exposure check failed')
        if not all(math.isfinite(row[k]) for k in ('awt_lane_time_mean_seconds','aql_lane_time_mean_vehicles')):
            raise ValueError('Nonfinite validation metric')
    following=[n for n in contract['review_milestones'] if n>episode]
    return dict(episode=episode,execution_checks_passed=True,next_review=min(following) if following else None,
                awt_seconds=summary['awt_seconds'],aql_vehicles=summary['aql_vehicles'],
                collision_events=summary['collision_pair_events'],junction_entries=summary['junction_entries'],
                decision='Continue fixed-budget baseline with unchanged settings; these checks do not establish safety or superiority',
                original_paper_reproduced=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='readiness-v4')
    parser.add_argument('--contract',default='reproduction/assumptions-v1.json')
    args=parser.parse_args()
    if Path(args.run).name!=args.run:parser.error('Simple run directory name required')
    run=ROOT/'runs'/args.run
    contract_path=Path(args.contract).resolve();raw=contract_path.read_bytes();contract=json.loads(raw)
    if (run/'reproduction-contract.json').read_bytes()!=raw:parser.error('Bound contract differs')
    with (run/'campaign.lock').open('a') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.error('Campaign already active')
        status_file=run/'campaign-status.json'
        child=None
        def status(stage,**extra):
            atomic_json(status_file,dict(stage=stage,campaign_pid=os.getpid(),updated_unix=time.time(),
                contract_sha256=hashlib.sha256(raw).hexdigest(),script_sha256=file_sha256(__file__),
                target_episodes=contract['training_target_episodes'],original_paper_reproduced=False,**extra))
        try:
            while True:
                if child is not None:child.poll()  # Reap managers started by this campaign.
                state=json.loads((run/'status.json').read_text())
                if child is not None and child.poll() is None and state.get('manager_pid')!=child.pid:
                    status('waiting_for_new_manager',manager_pid=child.pid)
                    time.sleep(3);continue
                if state.get('contract_sha256')!=hashlib.sha256(raw).hexdigest():
                    raise ValueError('Live manager contract differs')
                if state['stage'] in ('failed','blocked','interrupted'):
                    raise ValueError(f"Manager needs inspection: {state['stage']}")
                if state['stage'] in ('training','checkpoint_validation','evaluation'):
                    if not manager_alive(state['manager_pid']):
                        raise ValueError('Manager is not alive; inspect possible orphan child before restart')
                    status('monitoring_manager',manager_stage=state['stage'],manager_pid=state['manager_pid'])
                    time.sleep(10);continue
                if state['stage']=='completed':
                    final=json.loads((run/'evaluation/results.json').read_text())
                    trained=json.loads((run/'training/results.json').read_text())
                    if (len(trained['episodes'])!=contract['training_target_episodes'] or
                        [r['sumo_seed'] for r in final['episodes']]!=contract['final_test_seeds']):
                        raise ValueError('Full training/final evaluation budget differs')
                    status('completed_fixed_budget',final_test_evaluated=True,
                           note='One trained seed; assumed reconstruction, not exact paper replication or superiority')
                    return
                if state['stage']!='awaiting_checkpoint_review':
                    raise ValueError(f"Unexpected manager state: {state['stage']}")
                if manager_alive(state['manager_pid']):
                    time.sleep(2);continue  # Allow the old manager to release its lock.
                episode=state['completed_training_episodes']
                decision=review_execution(ROOT,run,contract,episode)
                target=decision['next_review'] or contract['training_target_episodes']
                required=((target-episode)*(run/'training'/f'checkpoint-{episode:04d}.pt').stat().st_size+
                          2*(run/'training/resume.pt').stat().st_size+1024**3)
                if shutil.disk_usage(run).free<required:raise ValueError('Insufficient space for next fixed-budget segment')
                dest=run/'reviews'/f'checkpoint-{episode:04d}';dest.mkdir(parents=True,exist_ok=True)
                atomic_json(dest/'automatic-execution-review.json',decision)
                command=[sys.executable,str(ROOT/'run_experiment.py'),'--name',args.run,
                         '--episodes',str(contract['training_target_episodes']),'--seed',str(contract['fixed_cli']['seed']),
                         '--eval-sumo-seed',str(contract['final_test_seeds'][0]),
                         '--eval-episodes',str(len(contract['final_test_seeds'])),'--contract',str(contract_path)]
                if decision['next_review'] is not None:command+=['--review-at',str(decision['next_review'])]
                with (run/f'campaign-manager-to-{target:04d}.log').open('a') as stream:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                           env=os.environ.copy(),start_new_session=True)
                status('starting_segment',manager_pid=child.pid,target_segment=target)
                time.sleep(3)
                if child.poll() is not None and child.returncode:
                    raise ValueError(f'New segment manager failed with code {child.returncode}')
        except BaseException as exc:
            status('stopped_for_inspection',reason=f'{type(exc).__name__}: {exc}',final_test_evaluated=False)
            raise


if __name__=='__main__':main()
