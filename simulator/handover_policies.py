"""
handover_policies.py — Handover decision algorithms for MobiSense.

Policy A — Strongest-cell baseline (A3-event style)
Policy B — Hysteresis + Time-to-Trigger (3GPP A3 with TTT)
Policy C — Predictive signal-based (linear regression forecast)
Policy D — MobiSense: risk-aware, QoS-aware composite scoring

These are software decision algorithms implemented for research simulation.
They are NOT implementations of standardised 3GPP RRC procedures.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CellMeasurement:
    cell_id: int
    rsrp_dbm: float
    sinr_db: float
    load_factor: float
    latency_ms: float
    throughput_mbps: float
    distance_m: float


@dataclass
class HandoverDecision:
    trigger: bool
    target_cell_id: int | None
    reason: str
    score: float = 0.0


class BasePolicy(ABC):
    @abstractmethod
    def decide(
        self,
        serving_cell_id: int,
        measurements: list[CellMeasurement],
        user_speed_mps: float,
        t: float,
    ) -> HandoverDecision:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class PolicyA(BasePolicy):
    """
    Strongest-cell baseline.
    Triggers handover when a neighbour RSRP exceeds serving RSRP by at least
    `a3_offset_db` (A3 event margin). No timer is applied — instant trigger.
    """

    def __init__(self, a3_offset_db: float = 3.0, min_rsrp_dbm: float = -110.0):
        self._offset = a3_offset_db
        self._min_rsrp = min_rsrp_dbm

    @property
    def name(self) -> str:
        return "Policy-A (Strongest Cell)"

    def reset(self) -> None:
        pass

    def decide(
        self,
        serving_cell_id: int,
        measurements: list[CellMeasurement],
        user_speed_mps: float,
        t: float,
    ) -> HandoverDecision:
        serving = _find_cell(measurements, serving_cell_id)
        if serving is None:
            return HandoverDecision(False, None, "No serving cell measurement")

        candidates = [
            m for m in measurements
            if m.cell_id != serving_cell_id
            and m.rsrp_dbm > serving.rsrp_dbm + self._offset
            and m.rsrp_dbm >= self._min_rsrp
        ]
        if not candidates:
            return HandoverDecision(False, None, "No A3 event triggered")

        best = max(candidates, key=lambda m: m.rsrp_dbm)
        return HandoverDecision(
            True, best.cell_id,
            f"A3 triggered: neighbour RSRP {best.rsrp_dbm:.1f} > serving {serving.rsrp_dbm:.1f} + {self._offset}",
            score=best.rsrp_dbm,
        )


class PolicyB(BasePolicy):
    """
    Hysteresis + Time-to-Trigger baseline (3GPP A3 with TTT).
    A neighbour must sustain the A3 margin for `ttt_steps` consecutive steps.
    """

    def __init__(
        self,
        a3_offset_db: float = 3.0,
        ttt_steps: int = 3,
        hysteresis_db: float = 1.0,
        min_rsrp_dbm: float = -110.0,
    ):
        self._offset = a3_offset_db
        self._ttt_steps = ttt_steps
        self._hysteresis = hysteresis_db
        self._min_rsrp = min_rsrp_dbm
        self._trigger_counters: dict[int, int] = {}

    @property
    def name(self) -> str:
        return "Policy-B (Hysteresis + TTT)"

    def reset(self) -> None:
        self._trigger_counters = {}

    def decide(
        self,
        serving_cell_id: int,
        measurements: list[CellMeasurement],
        user_speed_mps: float,
        t: float,
    ) -> HandoverDecision:
        serving = _find_cell(measurements, serving_cell_id)
        if serving is None:
            return HandoverDecision(False, None, "No serving cell measurement")

        threshold = serving.rsrp_dbm + self._offset + self._hysteresis
        active_neighbour_ids = set()

        for m in measurements:
            if m.cell_id == serving_cell_id:
                continue
            if m.rsrp_dbm >= threshold and m.rsrp_dbm >= self._min_rsrp:
                self._trigger_counters[m.cell_id] = self._trigger_counters.get(m.cell_id, 0) + 1
                active_neighbour_ids.add(m.cell_id)
            else:
                self._trigger_counters[m.cell_id] = 0

        stale = [cid for cid in self._trigger_counters if cid not in active_neighbour_ids]
        for cid in stale:
            self._trigger_counters[cid] = 0

        ready = [
            m for m in measurements
            if m.cell_id != serving_cell_id
            and self._trigger_counters.get(m.cell_id, 0) >= self._ttt_steps
            and m.rsrp_dbm >= self._min_rsrp
        ]
        if not ready:
            return HandoverDecision(False, None, "TTT not satisfied for any neighbour")

        best = max(ready, key=lambda m: m.rsrp_dbm)
        self._trigger_counters[best.cell_id] = 0
        return HandoverDecision(
            True, best.cell_id,
            f"TTT met ({self._ttt_steps} steps): target {best.cell_id} RSRP {best.rsrp_dbm:.1f}",
            score=best.rsrp_dbm,
        )


class PolicyC(BasePolicy):
    """
    Predictive signal-based policy using linear regression over recent RSRP history.

    For each cell, fits a linear trend to the last `window` RSRP samples and
    projects the value `horizon_steps` ahead. Triggers handover when:
      - Predicted serving RSRP < `rsrp_floor_dbm` within the horizon, AND
      - A neighbour's predicted RSRP > predicted serving + margin.

    Prediction MAE is tracked for reporting.
    """

    def __init__(
        self,
        window: int = 8,
        horizon_steps: int = 5,
        rsrp_floor_dbm: float = -100.0,
        a3_margin_db: float = 2.0,
        min_rsrp_dbm: float = -110.0,
    ):
        self._window = window
        self._horizon = horizon_steps
        self._floor = rsrp_floor_dbm
        self._margin = a3_margin_db
        self._min_rsrp = min_rsrp_dbm
        self._history: dict[int, deque] = {}
        self._prediction_errors: list[float] = []

    @property
    def name(self) -> str:
        return "Policy-C (Predictive Linear)"

    def reset(self) -> None:
        self._history = {}
        self._prediction_errors = []

    @property
    def prediction_mae(self) -> float:
        if not self._prediction_errors:
            return float("nan")
        return float(np.mean(np.abs(self._prediction_errors)))

    def _update_history(self, measurements: list[CellMeasurement]) -> None:
        for m in measurements:
            if m.cell_id not in self._history:
                self._history[m.cell_id] = deque(maxlen=self._window)
            self._history[m.cell_id].append(m.rsrp_dbm)

    def _predict(self, cell_id: int) -> float | None:
        hist = self._history.get(cell_id)
        if hist is None or len(hist) < 3:
            return None
        y = np.array(hist, dtype=float)
        x = np.arange(len(y), dtype=float)
        coeffs = np.polyfit(x, y, 1)
        return float(np.polyval(coeffs, len(y) - 1 + self._horizon))

    def decide(
        self,
        serving_cell_id: int,
        measurements: list[CellMeasurement],
        user_speed_mps: float,
        t: float,
    ) -> HandoverDecision:
        self._update_history(measurements)
        serving = _find_cell(measurements, serving_cell_id)
        if serving is None:
            return HandoverDecision(False, None, "No serving cell measurement")

        pred_serving = self._predict(serving_cell_id)
        if pred_serving is None:
            return HandoverDecision(False, None, "Insufficient history for prediction")

        track_error = self._history.get(serving_cell_id)
        if track_error and len(track_error) >= 2:
            prev_hist = list(track_error)[:-1]
            if len(prev_hist) >= 3:
                y = np.array(prev_hist, dtype=float)
                x = np.arange(len(y), dtype=float)
                c = np.polyfit(x, y, 1)
                projected = float(np.polyval(c, len(y)))
                actual = list(track_error)[-1]
                self._prediction_errors.append(projected - actual)

        if pred_serving >= self._floor:
            return HandoverDecision(False, None, f"Predicted serving RSRP OK ({pred_serving:.1f} dBm)")

        candidates = []
        for m in measurements:
            if m.cell_id == serving_cell_id:
                continue
            pred_n = self._predict(m.cell_id)
            if pred_n is None:
                pred_n = m.rsrp_dbm
            if pred_n > pred_serving + self._margin and pred_n >= self._min_rsrp:
                candidates.append((m, pred_n))

        if not candidates:
            return HandoverDecision(False, None, "No suitable neighbour predicted")

        best_m, best_pred = max(candidates, key=lambda x: x[1])
        return HandoverDecision(
            True, best_m.cell_id,
            f"Prediction: serving→{pred_serving:.1f} dBm, target→{best_pred:.1f} dBm in {self._horizon}s",
            score=best_pred,
        )


class PolicyD(BasePolicy):
    """
    MobiSense: Risk-aware, QoS-aware composite handover policy.

    Scoring function (all terms normalized to [0,1] before weighting):
      score = w_rsrp * norm_rsrp_gain
            + w_load * (1 - target_load)
            + w_lat  * norm_latency_improvement
            - w_spd  * speed_penalty
            - w_switch * switch_penalty

    Eligibility rules (all must pass):
      1. Target RSRP >= min_rsrp_dbm
      2. Target RSRP >= serving RSRP + min_gain_db  (or imminent failure override)
      3. TTT lite: candidate must have scored ≥ score_threshold for ttt_steps consecutive steps
      4. Target load < max_load_threshold

    Imminent failure override: if predicted serving RSRP < critical_rsrp_dbm,
    any eligible neighbour (RSRP >= min_rsrp_dbm) is considered regardless of gain.

    NOTE: This algorithm is a research prototype, not a 3GPP standardised procedure.
    """

    def __init__(
        self,
        window: int = 8,
        horizon_steps: int = 5,
        min_rsrp_dbm: float = -110.0,
        critical_rsrp_dbm: float = -105.0,
        rsrp_floor_dbm: float = -100.0,
        min_gain_db: float = 2.0,
        max_load_threshold: float = 0.90,
        ttt_steps: int = 2,
        w_rsrp: float = 0.35,
        w_load: float = 0.30,
        w_lat: float = 0.15,
        w_spd: float = 0.10,
        w_switch: float = 0.10,
        switch_cooldown_steps: int = 5,
    ):
        self._window = window
        self._horizon = horizon_steps
        self._min_rsrp = min_rsrp_dbm
        self._critical_rsrp = critical_rsrp_dbm
        self._rsrp_floor = rsrp_floor_dbm
        self._min_gain = min_gain_db
        self._max_load = max_load_threshold
        self._ttt_steps = ttt_steps
        self._weights = dict(w_rsrp=w_rsrp, w_load=w_load, w_lat=w_lat, w_spd=w_spd, w_switch=w_switch)
        self._cooldown_steps = switch_cooldown_steps
        self._history: dict[int, deque] = {}
        self._score_counters: dict[int, int] = {}
        self._last_switch_step: float = -999
        self._prediction_errors: list[float] = []

    @property
    def name(self) -> str:
        return "Policy-D (MobiSense QoS-Aware)"

    def reset(self) -> None:
        self._history = {}
        self._score_counters = {}
        self._last_switch_step = -999
        self._prediction_errors = []

    @property
    def prediction_mae(self) -> float:
        if not self._prediction_errors:
            return float("nan")
        return float(np.mean(np.abs(self._prediction_errors)))

    def _update_and_predict(self, cell_id: int, rsrp: float) -> float:
        if cell_id not in self._history:
            self._history[cell_id] = deque(maxlen=self._window)
        self._history[cell_id].append(rsrp)
        hist = self._history[cell_id]
        if len(hist) < 3:
            return rsrp
        y = np.array(hist, dtype=float)
        x = np.arange(len(y), dtype=float)
        coeffs = np.polyfit(x, y, 1)
        return float(np.polyval(coeffs, len(y) - 1 + self._horizon))

    def _track_error(self, cell_id: int) -> None:
        hist = self._history.get(cell_id)
        if hist and len(hist) >= 4:
            prev = list(hist)[:-1]
            y = np.array(prev, dtype=float)
            x = np.arange(len(y), dtype=float)
            c = np.polyfit(x, y, 1)
            projected = float(np.polyval(c, len(y)))
            actual = list(hist)[-1]
            self._prediction_errors.append(projected - actual)

    def decide(
        self,
        serving_cell_id: int,
        measurements: list[CellMeasurement],
        user_speed_mps: float,
        t: float,
    ) -> HandoverDecision:
        for m in measurements:
            self._update_and_predict(m.cell_id, m.rsrp_dbm)
            self._track_error(m.cell_id)

        serving = _find_cell(measurements, serving_cell_id)
        if serving is None:
            return HandoverDecision(False, None, "No serving cell measurement")

        pred_serving = self._update_and_predict(serving_cell_id, serving.rsrp_dbm)
        imminent_failure = pred_serving < self._critical_rsrp

        cooldown_active = (t - self._last_switch_step) < self._cooldown_steps

        candidates = []
        for m in measurements:
            if m.cell_id == serving_cell_id:
                continue
            if m.rsrp_dbm < self._min_rsrp:
                continue
            if m.load_factor >= self._max_load:
                continue
            if not imminent_failure:
                if m.rsrp_dbm < serving.rsrp_dbm + self._min_gain:
                    continue
            pred_n = self._update_and_predict(m.cell_id, m.rsrp_dbm)
            candidates.append((m, pred_n))

        if not candidates:
            return HandoverDecision(False, None, "No eligible neighbours" + (" (imminent failure)" if imminent_failure else ""))

        rsrp_values = [m.rsrp_dbm for m, _ in candidates] + [serving.rsrp_dbm]
        rsrp_min, rsrp_max = min(rsrp_values), max(rsrp_values)
        rsrp_range = max(rsrp_max - rsrp_min, 1.0)

        lat_values = [m.latency_ms for m, _ in candidates] + [serving.latency_ms]
        lat_min, lat_max = min(lat_values), max(lat_values)
        lat_range = max(lat_max - lat_min, 1.0)

        speed_norm = min(user_speed_mps / 50.0, 1.0)
        w = self._weights

        scored = []
        for m, pred_n in candidates:
            rsrp_gain_norm = (m.rsrp_dbm - serving.rsrp_dbm) / rsrp_range
            lat_improvement_norm = (serving.latency_ms - m.latency_ms) / lat_range
            load_score = 1.0 - m.load_factor
            switch_pen = 0.5 if cooldown_active else 0.0
            speed_pen = speed_norm * 0.3

            score = (
                w["w_rsrp"] * rsrp_gain_norm
                + w["w_load"] * load_score
                + w["w_lat"] * lat_improvement_norm
                - w["w_spd"] * speed_pen
                - w["w_switch"] * switch_pen
            )

            cid = m.cell_id
            if score > 0.05:
                self._score_counters[cid] = self._score_counters.get(cid, 0) + 1
            else:
                self._score_counters[cid] = 0

            ttt_ok = self._score_counters.get(cid, 0) >= self._ttt_steps or imminent_failure
            scored.append((m, score, pred_n, ttt_ok))

        ready = [(m, s, p) for m, s, p, ok in scored if ok]
        if not ready:
            return HandoverDecision(False, None, "TTT lite not satisfied")

        best_m, best_score, best_pred = max(ready, key=lambda x: x[1])
        self._last_switch_step = t
        self._score_counters[best_m.cell_id] = 0

        reason = (
            f"{'IMMINENT FAILURE OVERRIDE — ' if imminent_failure else ''}"
            f"score={best_score:.3f}, "
            f"target RSRP={best_m.rsrp_dbm:.1f} dBm, "
            f"pred={best_pred:.1f} dBm, "
            f"load={best_m.load_factor:.2f}"
        )
        return HandoverDecision(True, best_m.cell_id, reason, score=best_score)


def _find_cell(measurements: list[CellMeasurement], cell_id: int) -> CellMeasurement | None:
    for m in measurements:
        if m.cell_id == cell_id:
            return m
    return None
