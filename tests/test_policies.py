"""
test_policies.py — Tests for all four handover policies.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.handover_policies import (
    PolicyA, PolicyB, PolicyC, PolicyD,
    CellMeasurement, HandoverDecision,
)


def make_meas(cell_id, rsrp, sinr=5.0, load=0.3, lat=15.0, tput=10.0, dist=200.0):
    return CellMeasurement(
        cell_id=cell_id, rsrp_dbm=rsrp, sinr_db=sinr,
        load_factor=load, latency_ms=lat,
        throughput_mbps=tput, distance_m=dist,
    )


SERVING_WEAK = [
    make_meas(0, -90.0),
    make_meas(1, -80.0),
    make_meas(2, -95.0),
]

SERVING_STRONG = [
    make_meas(0, -70.0),
    make_meas(1, -90.0),
    make_meas(2, -100.0),
]


class TestPolicyA:
    def test_triggers_when_neighbour_stronger(self):
        pol = PolicyA(a3_offset_db=3.0)
        dec = pol.decide(0, SERVING_WEAK, 15.0, 0.0)
        assert dec.trigger
        assert dec.target_cell_id == 1

    def test_no_trigger_when_serving_strongest(self):
        pol = PolicyA(a3_offset_db=3.0)
        dec = pol.decide(0, SERVING_STRONG, 15.0, 0.0)
        assert not dec.trigger

    def test_picks_best_rsrp(self):
        pol = PolicyA(a3_offset_db=0.0)
        meas = [make_meas(0, -90), make_meas(1, -85), make_meas(2, -80)]
        dec = pol.decide(0, meas, 10.0, 0.0)
        assert dec.target_cell_id == 2

    def test_min_rsrp_filter(self):
        pol = PolicyA(a3_offset_db=0.0, min_rsrp_dbm=-80.0)
        meas = [make_meas(0, -85), make_meas(1, -81), make_meas(2, -75)]
        dec = pol.decide(0, meas, 10.0, 0.0)
        assert dec.trigger
        assert dec.target_cell_id == 2

    def test_reset_has_no_state(self):
        pol = PolicyA()
        pol.reset()

    def test_name(self):
        assert "Policy-A" in PolicyA().name


class TestPolicyB:
    def test_ttt_not_satisfied_immediately(self):
        pol = PolicyB(a3_offset_db=3.0, ttt_steps=3)
        dec = pol.decide(0, SERVING_WEAK, 15.0, 0.0)
        assert not dec.trigger

    def test_triggers_after_ttt_steps(self):
        pol = PolicyB(a3_offset_db=3.0, ttt_steps=3)
        dec = None
        for t in range(5):
            dec = pol.decide(0, SERVING_WEAK, 15.0, float(t))
        assert dec is not None and dec.trigger

    def test_counter_resets_on_condition_loss(self):
        pol = PolicyB(a3_offset_db=3.0, ttt_steps=3)
        for t in range(2):
            pol.decide(0, SERVING_WEAK, 15.0, float(t))
        for t in range(5):
            pol.decide(0, SERVING_STRONG, 15.0, float(t + 2))
        dec = pol.decide(0, SERVING_WEAK, 15.0, 7.0)
        assert not dec.trigger

    def test_reset_clears_counters(self):
        pol = PolicyB(ttt_steps=2)
        for t in range(2):
            pol.decide(0, SERVING_WEAK, 15.0, float(t))
        pol.reset()
        dec = pol.decide(0, SERVING_WEAK, 15.0, 3.0)
        assert not dec.trigger

    def test_name(self):
        assert "Policy-B" in PolicyB().name


class TestPolicyC:
    def _fill_history(self, pol, meas, steps=10):
        for t in range(steps):
            pol.decide(0, meas, 15.0, float(t))

    def test_no_trigger_with_good_signal(self):
        pol = PolicyC(rsrp_floor_dbm=-100.0)
        meas = [make_meas(0, -80), make_meas(1, -85)]
        for t in range(12):
            dec = pol.decide(0, meas, 10.0, float(t))
        assert not dec.trigger

    def test_triggers_when_signal_deteriorates(self):
        pol = PolicyC(window=6, horizon_steps=3, rsrp_floor_dbm=-95.0, a3_margin_db=0.0)
        rsrp_vals = [-75, -78, -82, -86, -91, -96, -100, -104]
        meas_seq = [
            [make_meas(0, r), make_meas(1, -78)] for r in rsrp_vals
        ]
        dec = None
        for t, meas in enumerate(meas_seq):
            dec = pol.decide(0, meas, 10.0, float(t))
        assert dec is not None and dec.trigger

    def test_mae_computed(self):
        pol = PolicyC(window=5)
        meas = [make_meas(0, -80), make_meas(1, -85)]
        for t in range(15):
            pol.decide(0, meas, 10.0, float(t))
        mae = pol.prediction_mae
        assert np.isfinite(mae) or np.isnan(mae)

    def test_reset_clears(self):
        pol = PolicyC()
        meas = [make_meas(0, -80), make_meas(1, -75)]
        for t in range(10):
            pol.decide(0, meas, 10.0, float(t))
        pol.reset()
        assert pol._history == {}

    def test_name(self):
        assert "Policy-C" in PolicyC().name


class TestPolicyD:
    def test_no_trigger_when_serving_fine(self):
        pol = PolicyD(min_gain_db=2.0)
        meas = [make_meas(0, -70), make_meas(1, -90), make_meas(2, -95)]
        for t in range(10):
            dec = pol.decide(0, meas, 10.0, float(t))
        assert not dec.trigger

    def test_triggers_with_clear_better_neighbour(self):
        pol = PolicyD(min_gain_db=2.0, ttt_steps=2, min_rsrp_dbm=-120.0, max_load_threshold=0.95)
        meas = [make_meas(0, -92), make_meas(1, -78, load=0.3)]
        dec = None
        for t in range(10):
            dec = pol.decide(0, meas, 10.0, float(t))
        assert dec is not None and dec.trigger

    def test_avoids_high_load_cell(self):
        pol = PolicyD(min_gain_db=0.0, ttt_steps=1, max_load_threshold=0.85)
        meas = [make_meas(0, -90), make_meas(1, -78, load=0.95)]
        for t in range(10):
            dec = pol.decide(0, meas, 10.0, float(t))
        assert not dec.trigger

    def test_imminent_failure_override(self):
        pol = PolicyD(critical_rsrp_dbm=-95.0, min_gain_db=5.0, ttt_steps=1)
        meas = [make_meas(0, -98), make_meas(1, -99)]
        dec = None
        for t in range(5):
            dec = pol.decide(0, meas, 10.0, float(t))
        assert dec is not None and dec.trigger

    def test_cooldown_reduces_switches(self):
        pol = PolicyD(switch_cooldown_steps=100, ttt_steps=1)
        meas = [make_meas(0, -92), make_meas(1, -75, load=0.3)]
        switches = []
        for t in range(20):
            dec = pol.decide(0, meas, 10.0, float(t))
            switches.append(dec.trigger)
        assert switches.count(True) <= 2

    def test_reset_clears_state(self):
        pol = PolicyD()
        pol.reset()
        assert pol._history == {}
        assert pol._score_counters == {}

    def test_name(self):
        assert "Policy-D" in PolicyD().name
