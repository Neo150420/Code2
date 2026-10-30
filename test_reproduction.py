"""Check experiment identity and review stops without spending RL training time."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_experiment
from reproduction_contract import bind_contract, file_sha256, validate_contract


class ReconstructionControl(unittest.TestCase):
    def test_final_metrics_pool_collision_exposure_and_reject_training_leakage(self):
        def row(events,entries):
            return dict(training_conflict_proxy=False,risk_at_inference=False,missing_transition_keys=0,
                        awt_lane_time_mean_seconds=10.,aql_lane_time_mean_vehicles=2.,
                        junction_collision_pair_events_in_window=events,junction_entries_in_window=entries)
        rows=[row(1,10),row(0,100)]
        metrics,events,entries=run_experiment.aggregate_final_evaluation(rows)
        self.assertAlmostEqual(metrics['collision_pair_events_per_junction_entry_percent'],100/110)
        self.assertEqual((events,entries),(1,110))
        self.assertIsNone(run_experiment.aggregate_final_evaluation([row(0,0)])[0]['collision_pair_events_per_junction_entry_percent'])
        rows[0]['risk_at_inference']=True
        with self.assertRaises(ValueError):run_experiment.aggregate_final_evaluation(rows)

    def test_contract_rejects_changed_source_and_changed_resume_settings(self):
        with tempfile.TemporaryDirectory() as dirname:
            root=Path(dirname)
            (root/'model.py').write_text('original source')
            contract=dict(classification='corrected_author_asset_reconstruction',protocol='v4',
                          critical_source_sha256={'model.py':file_sha256(root/'model.py')},
                          asset_sha256={},dependency_lock_sha256={},fixed_cli={'seed':0})
            previous=dict(protocol='v4',code_sha256=contract['critical_source_sha256'],seed=0)
            validate_contract(root,contract,previous)
            with self.assertRaisesRegex(ValueError,'setting differs: seed'):
                validate_contract(root,contract,dict(previous,seed=1))
            (root/'model.py').write_text('changed source')
            with self.assertRaisesRegex(ValueError,'input changed'):
                validate_contract(root,contract,previous)

    def test_bound_contract_cannot_be_replaced_even_if_sources_still_match(self):
        with tempfile.TemporaryDirectory() as dirname:
            root=Path(dirname);run=root/'run';run.mkdir()
            contract=dict(classification='corrected_author_asset_reconstruction',
                          critical_source_sha256={},asset_sha256={},dependency_lock_sha256={})
            path=root/'contract.json';path.write_text(json.dumps(contract))
            bind_contract(root,run,path)
            path.write_text(json.dumps(dict(contract,changed_assumption=True)))
            with self.assertRaisesRegex(ValueError,'different frozen contract'):
                bind_contract(root,run,path)

    def run_fake_review(self, root, validation_code):
        commands=[]
        class Child:
            pid=1234
            def __init__(self,command,**kwargs):
                self.command=command;commands.append(command)
                (root/'runs/trial/training').mkdir(exist_ok=True)
            def wait(self):
                return validation_code if 'checkpoint_validation.py' in self.command[1] else 0
        argv=['run_experiment.py','--name','trial','--episodes','500','--review-at','10']
        with patch.object(run_experiment,'ROOT',root),patch('sys.argv',argv),patch.object(run_experiment.subprocess,'Popen',Child):
            if validation_code:
                with self.assertRaises(SystemExit):run_experiment.main()
            else:
                run_experiment.main()
        return commands,json.loads((root/'runs/trial/status.json').read_text())

    def test_review_budget_stops_at_boundary_and_never_evaluates_final_seeds(self):
        with tempfile.TemporaryDirectory() as dirname:
            commands,status=self.run_fake_review(Path(dirname),0)
            self.assertEqual(status['stage'],'awaiting_checkpoint_review')
            self.assertFalse(status['final_test_evaluated'])
            self.assertEqual(commands[0][commands[0].index('--episodes')+1],'10')
            self.assertEqual(len(commands),2)
            self.assertNotIn('evaluate',commands[0])
            self.assertNotIn('--sumo-seed',commands[1])
            self.assertIn('10',commands[1])

    def test_failed_checkpoint_validation_never_reports_review_ready(self):
        with tempfile.TemporaryDirectory() as dirname:
            _,status=self.run_fake_review(Path(dirname),1)
            self.assertEqual(status['stage'],'failed')
            self.assertEqual(status['failed_stage'],'checkpoint_validation')


if __name__=='__main__':unittest.main()
