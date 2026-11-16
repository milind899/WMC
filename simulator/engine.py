"""
engine.py — Discrete-time simulation engine for MobiSense.

Ties together: mobility, radio model, handover policy, and metrics collector.

The engine runs for a fixed number of time steps (dt_s each). At each step:
  1. Advance user position via MobilityEngine.
  2. Compute RSRP, SINR, throughput, and latency for every base station.
  3. Add measurement noise and shadowing samples.
  4. Query the active policy for a handover decision.
  5. Apply handover if triggered; update serving cell.
  6. Record step metrics.

A handover is modeled as instantaneous for simplicity. A fixed interruption
penalty is added to latency in the step where the switch occurs.
This is an approximation; actual L2/L3 handover latency is not modeled.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from .radio_model import (
    compute_rsrp,
    compute_sinr,
    sinr_to_throughput,
    estimate_latency,
    add_measurement_noise,
    generate_shadowing,
    RLF_RSRP_DBM_THRESHOLD as RLF_RSRP_THRESHOLD_DBM,
)
from .mobility import MobilityEngine, MobilityConfig
from .handover_policies import (
    BasePolicy,
    CellMeasurement,
)
from .metrics import (
    MetricsCollector,
    StepRecord,
    HandoverEvent,
    AggregatedKPIs,
)
from .scenarios import Scenario, BaseStation


@dataclass
class SimulationResult:
    scenario_name: str
    policy_name: str
    seed: int
    steps: list[StepRecord]
    handover_events: list[HandoverEvent]
    kpis: AggregatedKPIs
    trajectory: list[tuple[float, float]]
    bs_positions: list[tuple[int, float, float]]


class SimulationEngine:
    """
    Runs a full scenario simulation with one policy and returns a SimulationResult.
    """

    def __init__(
        self,
        scenario: Scenario,
        policy: BasePolicy,
        seed: int | None = None,
        noise_override_db: float | None = None,
        speed_override_mps: float | None = None,
    ):
        self._scenario = scenario
        self._policy = policy
        self._seed = seed if seed is not None else scenario.seed
        self._noise_db = noise_override_db if noise_override_db is not None else scenario.measurement_noise_db
        self._speed = speed_override_mps if speed_override_mps is not None else scenario.speed_mps

    def run(self) -> SimulationResult:
        sc = self._scenario
        rng = np.random.default_rng(self._seed)
        self._policy.reset()

        mob_cfg = MobilityConfig(
            start_x=sc.start_x,
            start_y=sc.start_y,
            waypoints=list(sc.waypoints),
            speed_mps=self._speed,
            speed_std_mps=max(self._speed * 0.05, 0.5),
            heading_noise_std_rad=0.08,
            area_width_m=2000.0,
            area_height_m=2000.0,
            dt_s=sc.dt_s,
        )
        mob = MobilityEngine(mob_cfg, rng)
        collector = MetricsCollector()

        bs_list = sc.base_stations
        shadowing: dict[int, float] = {
            bs.cell_id: generate_shadowing(bs.shadowing_sigma_db, rng)
            for bs in bs_list
        }

        serving_cell_id = _initial_serving_cell(sc.start_x, sc.start_y, bs_list)
        prev_cell_id = serving_cell_id
        handover_cooldown = 0

        for step in range(sc.duration_steps):
            t = step * sc.dt_s
            user = mob.step()

            if step % 10 == 0:
                for bs in bs_list:
                    shadowing[bs.cell_id] = generate_shadowing(bs.shadowing_sigma_db, rng)

            load_map = _get_loads(bs_list, sc, step)

            measurements: list[CellMeasurement] = []
            for bs in bs_list:
                dist = np.hypot(user.x - bs.x, user.y - bs.y)
                rsrp_true = compute_rsrp(dist, bs.tx_power_dbm, bs.fc_mhz, shadowing[bs.cell_id])
                rsrp_meas = add_measurement_noise(rsrp_true, self._noise_db, rng)
                interferers = [
                    compute_rsrp(np.hypot(user.x - b.x, user.y - b.y), b.tx_power_dbm, b.fc_mhz)
                    for b in bs_list if b.cell_id != bs.cell_id
                ]
                sinr = compute_sinr(rsrp_meas, interferers)
                load = load_map[bs.cell_id]
                tput = sinr_to_throughput(sinr, load_factor=load)
                lat = estimate_latency(load)
                measurements.append(CellMeasurement(
                    cell_id=bs.cell_id,
                    rsrp_dbm=rsrp_meas,
                    sinr_db=sinr,
                    load_factor=load,
                    latency_ms=lat,
                    throughput_mbps=tput,
                    distance_m=dist,
                ))

            decision = self._policy.decide(
                serving_cell_id=serving_cell_id,
                measurements=measurements,
                user_speed_mps=user.speed,
                t=t,
            )

            ho_occurred = False
            ho_success = False
            interruption_add = 0.0

            if decision.trigger and decision.target_cell_id is not None:
                target_m = _find_meas(measurements, decision.target_cell_id)
                success = target_m is not None and target_m.rsrp_dbm >= RLF_RSRP_THRESHOLD_DBM

                prev_cell_id = serving_cell_id
                serving_cell_id = decision.target_cell_id if success else serving_cell_id
                ho_occurred = True
                ho_success = success

                collector.record_handover(HandoverEvent(
                    t=t,
                    from_cell=prev_cell_id,
                    to_cell=decision.target_cell_id,
                    success=success,
                    rsrp_at_ho=(target_m.rsrp_dbm if target_m else -999.0),
                ))
                if not success:
                    interruption_add = sc.handover_interruption_ms

            serving_m = _find_meas(measurements, serving_cell_id)
            if serving_m is None:
                serving_m = measurements[0]

            is_rlf = serving_m.rsrp_dbm < RLF_RSRP_THRESHOLD_DBM

            collector.record_step(StepRecord(
                t=t,
                x=user.x,
                y=user.y,
                serving_cell=serving_cell_id,
                rsrp_dbm=serving_m.rsrp_dbm,
                sinr_db=serving_m.sinr_db,
                throughput_mbps=serving_m.throughput_mbps,
                latency_ms=serving_m.latency_ms + interruption_add,
                load_factor=serving_m.load_factor,
                handover_occurred=ho_occurred,
                handover_success=ho_success,
                rlf=is_rlf,
            ))

        pred_mae = None
        if hasattr(self._policy, "prediction_mae"):
            pred_mae = self._policy.prediction_mae

        kpis = collector.compute_kpis(prediction_mae=pred_mae)

        return SimulationResult(
            scenario_name=sc.name,
            policy_name=self._policy.name,
            seed=self._seed,
            steps=collector.steps,
            handover_events=collector.handover_events,
            kpis=kpis,
            trajectory=mob.trajectory,
            bs_positions=[(bs.cell_id, bs.x, bs.y) for bs in bs_list],
        )


def _initial_serving_cell(x: float, y: float, bs_list: list[BaseStation]) -> int:
    best = min(bs_list, key=lambda bs: np.hypot(x - bs.x, y - bs.y))
    return best.cell_id


def _find_meas(measurements: list[CellMeasurement], cell_id: int) -> CellMeasurement | None:
    for m in measurements:
        if m.cell_id == cell_id:
            return m
    return None


def _get_loads(bs_list: list[BaseStation], scenario: Scenario, step: int) -> dict[int, float]:
    loads = {bs.cell_id: bs.base_load for bs in bs_list}
    for cell_id, periods in scenario.congestion_schedule.items():
        for start_step, end_step, load_value in periods:
            if start_step <= step < end_step:
                loads[cell_id] = load_value
    return loads
