"""Targeted checks for errors that invalidate the learning/evaluation pipeline."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from baseline import build_agent, save_checkpoint, restore_training
from algos.transformer_risk_model import TransformerRiskPredictor, RiskExperienceBuffer
from Env import Env


class BaselineCorrectness(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)

    def test_scalar_actions_and_replay_shape_guard(self):
        agent = build_agent('cpu', capacity=32)
        self.assertEqual(agent.select_action(np.zeros((7,98)), True).shape, (7,1))
        with self.assertRaises(ValueError):
            agent.replay_buffer.add(np.zeros(98), np.zeros(98), 0., np.zeros(98), False)

    def test_nonterminal_bellman_bootstrap_and_actual_parameter_updates(self):
        agent = build_agent('cpu', capacity=512)
        # Twin models receive identical samples/RNG. Changing only terminal masks must
        # change the actual critic loss: it catches the all-terminal training bug.
        state = np.ones(98, dtype=np.float32)*0.1
        for _ in range(300):
            agent.replay_buffer.add(state, np.zeros(1,dtype=np.float32), 0.5, state, False)
        import copy
        terminal_agent = copy.deepcopy(agent)
        terminal_agent.replay_buffer.dw.fill_(True)
        before = [p.clone() for p in agent.actor.parameters()]
        torch.manual_seed(17)
        live_loss = agent.train()
        torch.manual_seed(17)
        terminal_loss = terminal_agent.train()
        self.assertNotAlmostEqual(live_loss['q_loss'], terminal_loss['q_loss'], places=6)
        self.assertTrue(any(not torch.equal(p,b) for p,b in zip(agent.actor.parameters(),before)))
        self.assertTrue(all(v is None or np.isfinite(v) for v in live_loss.values()))

    def test_risk_labels_require_observed_outcomes_and_padding_matches(self):
        buffer = RiskExperienceBuffer()
        s, a = np.ones(98), np.ones(1)
        buffer.add('vehicle', s, a)
        live_s, live_a = buffer.get_sequence('vehicle')
        buffer.update_data('vehicle',1)
        self.assertEqual(len(buffer.active_buffer),0)  # No fabricated braking-safe example.
        self.assertEqual(len(buffer.risk_active_buffer),1)
        np.testing.assert_array_equal(live_s[0],buffer.risk_active_buffer[0]['state_seq'])
        np.testing.assert_array_equal(live_a[0],buffer.risk_active_buffer[0]['action_seq'])

    def test_efficiency_reward_handles_ties_and_risk_has_negative_sign(self):
        env = object.__new__(Env)
        env.max_acc = 3.
        env.reward_record = {}
        class Vehicle: id='test'
        no_risk = env.reward_compute(Vehicle(), np.ones(8), 3., 'j','d',0.,0)
        risk = env.reward_compute(Vehicle(), np.ones(8), 3., 'j','d',1.,0)
        self.assertAlmostEqual(no_risk,10.)  # Equal ranks have zero efficiency bonus.
        self.assertAlmostEqual(risk-no_risk,-3.)

    def test_evaluation_checkpoint_round_trip(self):
        agent = build_agent('cpu',capacity=8)
        predictor = TransformerRiskPredictor(98,1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'checkpoint.pt'
            save_checkpoint(path,agent,predictor,dict(test=True))
            data = torch.load(path)
            self.assertFalse(data['resumable'])
            self.assertIn('actor',data)

    def test_rotation_preserves_all_eight_directions(self):
        env = object.__new__(Env)
        env.keywords_order = ['topstraight','topleft','rightstraight','rightleft',
                              'bottomstraight','bottomleft','leftstraight','leftleft']
        class Map:
            def get_facing_intersection(self, road): return '229'
            def qurey_edge_direction(self, road, lane): return 'straight', 'right'
        class Vehicle:
            road_id='incoming'
            lane_index=0
        env.map=Map()
        rotated=env.rotated_keywords_order(Vehicle())
        self.assertEqual(rotated[0], 'rightstraight')
        self.assertEqual(set(rotated),set(env.keywords_order))
        self.assertEqual(len(set(rotated)),8)

    def test_full_resume_restores_buffers_optimizer_and_next_update(self):
        agent = build_agent('cpu',capacity=16)
        risk = TransformerRiskPredictor(98,1)
        state = np.ones(98,dtype=np.float32)*0.2
        for _ in range(12):
            agent.replay_buffer.add(state,np.zeros(1,dtype=np.float32),0.3,state,False)
        agent.train()
        risk.buffer.add('r',state,np.zeros(1))
        risk.buffer.update_data('r',1)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'resume.pt'
            save_checkpoint(path,agent,risk,dict(completed_episodes=1),include_replay=True)
            expected=agent.train()
            restored=build_agent('cpu',capacity=16)
            restored_risk=TransformerRiskPredictor(98,1)
            metadata=restore_training(path,restored,restored_risk)
            self.assertEqual(metadata['completed_episodes'],1)
            self.assertEqual(restored.replay_buffer.size,12)
            self.assertEqual(len(restored_risk.buffer.risk_active_buffer),1)
            observed=restored.train()
            for name in expected: self.assertAlmostEqual(expected[name],observed[name],places=6)

    def test_throughput_counts_entry_once_when_vehicle_moves_inside_junction(self):
        from copy import deepcopy
        from core.costomized_data_structures import Vehicle
        env=object.__new__(Env)
        env.print_debug=False
        env.junction_list=['229']
        env.keywords_order=['topstraight','topleft','rightstraight','rightleft',
                            'bottomstraight','bottomleft','leftstraight','leftleft']
        env.use_deadlock_resolver=False
        env.init_env()
        env.global_obs={'229':0.}
        env.prev_inner=deepcopy(env.inner_lane_obs)
        class Map:
            def qurey_inner_edge_direction(self,road,lane): return 'straight','right'
            def edge_length(self,road): return 100.
        env.map=Map()
        vehicle=Vehicle(id='v',road_id=':229_0',lane_index=0,laneposition=10.,speed=5.)
        env.vehicles['v']=vehicle
        env._update_obs()
        self.assertEqual(len(env.inner_lane_newly_enter['229']['rightstraight']),1)
        env.prev_inner=deepcopy(env.inner_lane_obs)
        vehicle.laneposition=20.
        vehicle.speed=6.
        env._update_obs()
        self.assertEqual(len(env.inner_lane_newly_enter['229']['rightstraight']),0)


if __name__=='__main__': unittest.main()
