"""Checks for misleading checkpoint aggregation and evaluation leakage."""
import unittest
from checkpoint_validation import summarize


def episode(seed,events,entries):
    return dict(sumo_seed=seed,training_conflict_proxy=False,risk_at_inference=False,
                missing_transition_keys=0,awt_lane_time_mean_seconds=10.,
                aql_lane_time_mean_vehicles=2.,junction_collision_pair_events_in_window=events,
                junction_entries_in_window=entries,arrivals_after_reset=80,vehicles_remaining=20)


class ValidationAggregation(unittest.TestCase):
    def test_collision_rate_uses_pooled_exposure_not_mean_of_percentages(self):
        result=summarize(1,dict(episodes=[episode(7001,1,10),episode(7002,0,100),episode(7003,0,100)]))
        self.assertAlmostEqual(result['collision_events_per_entry_percent'],100/210)
        self.assertEqual(result['collision_pair_events'],1)
        self.assertEqual(result['junction_entries'],210)

    def test_zero_entries_is_undefined_not_safe_zero(self):
        result=summarize(1,dict(episodes=[episode(s,0,0) for s in [7001,7002,7003]]))
        self.assertIsNone(result['collision_events_per_entry_percent'])

    def test_missing_or_mismatched_traffic_seed_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize(1,dict(episodes=[episode(7001,0,10),episode(7002,0,10)]))
        with self.assertRaises(ValueError):
            summarize(1,dict(episodes=[episode(s,0,10) for s in [7001,7002,10001]]))

    def test_training_proxy_or_risk_inference_leakage_is_rejected(self):
        rows=[episode(s,0,10) for s in [7001,7002,7003]]
        rows[1]['training_conflict_proxy']=True
        with self.assertRaises(ValueError):summarize(1,dict(episodes=rows))
        rows[1]['training_conflict_proxy']=False
        rows[2]['risk_at_inference']=True
        with self.assertRaises(ValueError):summarize(1,dict(episodes=rows))


if __name__=='__main__':unittest.main()
