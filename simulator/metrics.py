"""
metrics.py — Performance metric collection and computation for MobiSense.

All denominators are guarded against zero division.
Metric definitions are documented inline.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np


PING_PONG_WINDOW_S = 10.0
RLF_RSRP_THRESHOLD_DBM = -120.0
INTERRUPTION_PENALTY_MS = 50.0


@dataclass
class HandoverEvent:
    t: float
    from_cell: int
    to_cell: int
    success: bool
    rsrp_at_ho: float


@dataclass
class StepRecord:
    t: float
    x: float
    y: float
    serving_cell: int
    rsrp_dbm: float
    sinr_db: float
    throughput_mbps: float
    latency_ms: float
    load_factor: float
    handover_occurred: bool
    handover_success: bool
    rlf: bool


@dataclass
class AggregatedKPIs:
    handover_count: int = 0
    handover_failure_rate: float = 0.0
    ping_pong_rate: float = 0.0
    rlf_rate: float = 0.0
    mean_interruption_ms: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    mean_throughput_mbps: float = 0.0
    pdr: float = 0.0
    prediction_mae: float | None = None
    total_steps: int = 0
    rlf_count: int = 0
    ping_pong_count: int = 0
    handover_failure_count: int = 0


class MetricsCollector:
    """
    Accumulates per-step records and handover events, then computes KPIs.

    Ping-pong definition: a return to cell A within PING_PONG_WINDOW_S
    seconds after leaving it.

    RLF definition: RSRP falls below RLF_RSRP_THRESHOLD_DBM.

    Handover failure: handover triggered but RSRP of target cell is below
    min_rsrp_dbm immediately after switching (approximated here as the
    serving RSRP dropping below threshold on the next step).
    """

    def __init__(self):
        self._steps: list[StepRecord] = []
        self._ho_events: list[HandoverEvent] = []

    def record_step(self, record: StepRecord) -> None:
        self._steps.append(record)

    def record_handover(self, event: HandoverEvent) -> None:
        self._ho_events.append(event)

    @property
    def steps(self) -> list[StepRecord]:
        return self._steps

    @property
    def handover_events(self) -> list[HandoverEvent]:
        return self._ho_events

    def compute_kpis(self, prediction_mae: float | None = None) -> AggregatedKPIs:
        kpis = AggregatedKPIs()
        kpis.total_steps = len(self._steps)
        kpis.prediction_mae = prediction_mae

        if not self._steps:
            return kpis

        kpis.handover_count = len(self._ho_events)

        ho_failures = [e for e in self._ho_events if not e.success]
        kpis.handover_failure_count = len(ho_failures)
        kpis.handover_failure_rate = safe_div(len(ho_failures), len(self._ho_events))

        kpis.ping_pong_count = _count_ping_pongs(self._ho_events)
        kpis.ping_pong_rate = safe_div(kpis.ping_pong_count, len(self._ho_events))

        rlf_steps = [s for s in self._steps if s.rlf]
        kpis.rlf_count = len(rlf_steps)
        kpis.rlf_rate = safe_div(len(rlf_steps), len(self._steps))

        interruption_ms_list = [
            INTERRUPTION_PENALTY_MS for e in self._ho_events if not e.success
        ]
        kpis.mean_interruption_ms = float(np.mean(interruption_ms_list)) if interruption_ms_list else 0.0

        latencies = [s.latency_ms for s in self._steps]
        kpis.mean_latency_ms = float(np.mean(latencies))
        kpis.p95_latency_ms = float(np.percentile(latencies, 95))

        throughputs = [s.throughput_mbps for s in self._steps]
        kpis.mean_throughput_mbps = float(np.mean(throughputs))

        kpis.pdr = _estimate_pdr(self._steps)

        return kpis


def _count_ping_pongs(events: list[HandoverEvent]) -> int:
    count = 0
    for i in range(1, len(events)):
        prev = events[i - 1]
        curr = events[i]
        if curr.to_cell == prev.from_cell and (curr.t - prev.t) <= PING_PONG_WINDOW_S:
            count += 1
    return count


def _estimate_pdr(steps: list[StepRecord]) -> float:
    """
    Estimate packet delivery ratio from SINR.
    PDR ~ sigmoid of (SINR - threshold). Threshold = 0 dB for QPSK.
    This is an approximation, not a PHY-layer simulation.
    """
    if not steps:
        return 0.0
    delivered = sum(1 for s in steps if _sinr_to_pdr(s.sinr_db) > 0.5)
    return safe_div(delivered, len(steps))


def _sinr_to_pdr(sinr_db: float) -> float:
    return 1.0 / (1.0 + np.exp(-0.5 * (sinr_db - 2.0)))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator
