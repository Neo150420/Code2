"""Ensure automatic continuation rejects corrupt or unexposed evaluations."""
import json
from pathlib import Path
import tempfile
import unittest

from reproduction_contract import file_sha256
from run_reproduction_campaign import review_execution


class CampaignGate(unittest.TestCase):
    def fixture(self,root):
        run=root/'run';train=run/'training';train.mkdir(parents=True)
        contract=dict(protocol='v4',classification='corrected_author_asset_reconstruction',
                      critical_source_sha256={},fixed_cli={},validation_traffic_seeds=[7001,7002,7003],
                      review_milestones=[10,25,50,100,200])
        rows=[dict(episode=i,control_steps=1000,missing_transition_keys=0,reward_sum=1.,last_losses={'q_loss':1.}) for i in range(25)]
        result=dict(config=dict(protocol='v4',code_sha256={}),episodes=rows)
        (train/'results.json').write_text(json.dumps(result))
        checkpoint=train/'checkpoint-0025.pt';checkpoint.write_bytes(b'fixed checkpoint identity')
        dest=run/'validation/checkpoint-0025';(dest/'evaluation').mkdir(parents=True)
        summary=dict(checkpoint_sha256=file_sha256(checkpoint),awt_seconds=10.,aql_vehicles=1.,
                     collision_pair_events=0,junction_entries=300)
        (dest/'summary.json').write_text(json.dumps(summary))
        episodes=[dict(sumo_seed=s,missing_transition_keys=0,training_conflict_proxy=False,risk_at_inference=False,
                       control_steps=1000,junction_entries_in_window=100,
                       awt_lane_time_mean_seconds=10.,aql_lane_time_mean_vehicles=1.) for s in (7001,7002,7003)]
        evaluation=dict(config=dict(code_sha256={}),episodes=episodes)
        (dest/'evaluation/results.json').write_text(json.dumps(evaluation))
        return run,contract,result,evaluation

    def test_intact_gate_advances_fixed_budget_without_claiming_reproduction(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run,c,_,_=self.fixture(root)
            decision=review_execution(root,run,c,25)
            self.assertEqual(decision['next_review'],50)
            self.assertFalse(decision['original_paper_reproduced'])

    def test_zero_entry_seed_blocks_automatic_continuation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run,c,_,evaluation=self.fixture(root)
            evaluation['episodes'][0]['junction_entries_in_window']=0
            (run/'validation/checkpoint-0025/evaluation/results.json').write_text(json.dumps(evaluation))
            with self.assertRaisesRegex(ValueError,'positive exposure'):
                review_execution(root,run,c,25)

    def test_changed_checkpoint_is_not_validated_by_old_results(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run,c,_,_=self.fixture(root)
            (run/'training/checkpoint-0025.pt').write_bytes(b'different weights')
            with self.assertRaisesRegex(ValueError,'checkpoint identity'):
                review_execution(root,run,c,25)

    def test_nonfinite_training_result_stops_campaign(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run,c,result,_=self.fixture(root)
            result['episodes'][0]['last_losses']['q_loss']=float('nan')
            (run/'training/results.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'Nonfinite training'):
                review_execution(root,run,c,25)


if __name__=='__main__':unittest.main()
