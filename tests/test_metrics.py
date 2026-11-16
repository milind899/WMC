"""
test_metrics.py — Tests for metrics collection, KPI computation, and edge cases.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.metrics import (
    MetricsCollector,
    StepRecord,
    HandoverEvent,
    AggregatedKPIs,
    safe_div,
    _count_ping_pongs,
    PING_PONG_WINDOW_S,
)


def make_step(t=0.0, serving=0, rsrp=-80.0, sinr=10.0, tput=15.0, lat=15.0, load=0.3,
              ho=False, ho_success=True, rlf=False):
    return StepRecord(
        t=t, x=500.0, y=500.0, serving_cell=serving,
        rsrp_dbm=rsrp, sinr_db=sinr, throughput_mbps=tput,
        latency_ms=lat, load_factor=load,
        handover_occurred=ho, handover_success=ho_success, rlf=rlf,
    )


def make_ho(t, from_c, to_c, success=True, rsrp=-80.0):
    return HandoverEvent(t=t, from_cell=from_c, to_cell=to_c, success=success, rsrp_at_ho=rsrp)


class TestSafeDiv:
    def test_normal(self):
        assert abs(safe_div(3, 4) - 0.75) < 1e-9

    def test_zero_denominator(self):
        assert safe_div(1, 0) == 0.0

    def test_custom_default(self):
        assert safe_div(1, 0, default=-1.0) == -1.0


class TestPingPong:
    def test_no_events(self):
        assert _count_ping_pongs([]) == 0

    def test_single_event(self):
        assert _count_ping_pongs([make_ho(0, 0, 1)]) == 0

    def test_ping_pong_detected(self):
        events = [make_ho(0.0, 0, 1), make_ho(5.0, 1, 0)]
        assert _count_ping_pongs(events) == 1

    def test_no_ping_pong_outside_window(self):
        events = [make_ho(0.0, 0, 1), make_ho(PING_PONG_WINDOW_S + 1, 1, 0)]
        assert _count_ping_pongs(events) == 0

    def test_multiple_ping_pongs(self):
        events = [
            make_ho(0, 0, 1), make_ho(3, 1, 0), make_ho(5, 0, 1), make_ho(8, 1, 0)
        ]
        assert _count_ping_pongs(events) == 3


class TestMetricsCollector:
    def test_empty_gives_zero_kpis(self):
        c = MetricsCollector()
        kpis = c.compute_kpis()
        assert kpis.total_steps == 0
        assert kpis.handover_count == 0
        assert kpis.rlf_rate == 0.0

    def test_basic_kpis(self):
        c = MetricsCollector()
        for t in range(100):
            c.record_step(make_step(t=float(t)))
        kpis = c.compute_kpis()
        assert kpis.total_steps == 100
        assert kpis.handover_count == 0
        assert kpis.rlf_rate == 0.0
        assert kpis.mean_throughput_mbps > 0

    def test_rlf_counting(self):
        c = MetricsCollector()
        for t in range(10):
            c.record_step(make_step(t=float(t), rlf=(t < 3)))
        kpis = c.compute_kpis()
        assert kpis.rlf_count == 3
        assert abs(kpis.rlf_rate - 0.3) < 1e-6

    def test_handover_failure_rate(self):
        c = MetricsCollector()
        for t in range(10):
            c.record_step(make_step(t=float(t)))
        c.record_handover(make_ho(2.0, 0, 1, success=True))
        c.record_handover(make_ho(5.0, 1, 2, success=False))
        kpis = c.compute_kpis()
        assert kpis.handover_count == 2
        assert abs(kpis.handover_failure_rate - 0.5) < 1e-6

    def test_ping_pong_rate(self):
        c = MetricsCollector()
        for t in range(20):
            c.record_step(make_step(t=float(t)))
        c.record_handover(make_ho(2.0, 0, 1))
        c.record_handover(make_ho(5.0, 1, 0))
        c.record_handover(make_ho(8.0, 0, 1))
        kpis = c.compute_kpis()
        assert kpis.ping_pong_count >= 1

    def test_latency_percentile(self):
        c = MetricsCollector()
        for t in range(100):
            lat = 10.0 if t < 95 else 100.0
            c.record_step(make_step(t=float(t), lat=lat))
        kpis = c.compute_kpis()
        assert kpis.p95_latency_ms >= kpis.mean_latency_ms

    def test_pdr_good_sinr(self):
        c = MetricsCollector()
        for t in range(100):
            c.record_step(make_step(t=float(t), sinr=20.0))
        kpis = c.compute_kpis()
        assert kpis.pdr > 0.9

    def test_pdr_bad_sinr(self):
        c = MetricsCollector()
        for t in range(100):
            c.record_step(make_step(t=float(t), sinr=-20.0))
        kpis = c.compute_kpis()
        assert kpis.pdr < 0.1

    def test_prediction_mae_passthrough(self):
        c = MetricsCollector()
        c.record_step(make_step())
        kpis = c.compute_kpis(prediction_mae=1.5)
        assert kpis.prediction_mae == 1.5
