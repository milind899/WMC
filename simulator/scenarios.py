"""
scenarios.py — Predefined simulation scenarios for MobiSense benchmarks.

Each scenario specifies BS layout, user start position, waypoints,
speed, seed, duration, and congestion profile.

All scenarios use the same simulation area: 2000 m × 2000 m.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class BaseStation:
    cell_id: int
    x: float
    y: float
    tx_power_dbm: float = 46.0
    fc_mhz: float = 2100.0
    shadowing_sigma_db: float = 4.0
    base_load: float = 0.3


@dataclass
class Scenario:
    name: str
    description: str
    base_stations: list[BaseStation]
    start_x: float
    start_y: float
    waypoints: list[tuple[float, float]]
    speed_mps: float
    seed: int
    duration_steps: int
    dt_s: float = 1.0
    measurement_noise_db: float = 2.0
    handover_interruption_ms: float = 50.0
    congestion_schedule: dict[int, list[tuple[int, int, float]]] = field(default_factory=dict)


SCENARIOS: dict[str, Scenario] = {
    "normal_mobility": Scenario(
        name="Normal Mobility",
        description=(
            "Moderate-speed user traversing a standard 3-cell layout under stable "
            "radio conditions. Baseline scenario for policy comparison."
        ),
        base_stations=[
            BaseStation(cell_id=0, x=300, y=1000, shadowing_sigma_db=4.0, base_load=0.30),
            BaseStation(cell_id=1, x=1000, y=1700, shadowing_sigma_db=4.0, base_load=0.30),
            BaseStation(cell_id=2, x=1700, y=1000, shadowing_sigma_db=4.0, base_load=0.30),
        ],
        start_x=200, start_y=200,
        waypoints=[(500, 1000), (1000, 1500), (1500, 1000), (1800, 500)],
        speed_mps=15.0,
        seed=42,
        duration_steps=200,
    ),

    "high_speed": Scenario(
        name="High-Speed Transit",
        description=(
            "Vehicle travelling at highway speed through multiple overlapping cells. "
            "Tests how policies handle rapid cell transitions and potential ping-pong."
        ),
        base_stations=[
            BaseStation(cell_id=0, x=200, y=1000, shadowing_sigma_db=6.0, base_load=0.25),
            BaseStation(cell_id=1, x=600, y=1000, shadowing_sigma_db=6.0, base_load=0.25),
            BaseStation(cell_id=2, x=1000, y=1000, shadowing_sigma_db=6.0, base_load=0.25),
            BaseStation(cell_id=3, x=1400, y=1000, shadowing_sigma_db=6.0, base_load=0.25),
            BaseStation(cell_id=4, x=1800, y=1000, shadowing_sigma_db=6.0, base_load=0.25),
        ],
        start_x=50, start_y=1000,
        waypoints=[(1950, 1000)],
        speed_mps=35.0,
        seed=7,
        duration_steps=120,
    ),

    "overlapping_cells": Scenario(
        name="Overlapping Cells (Fluctuating Signal)",
        description=(
            "Dense cell layout with high shadowing variance. Signal quality fluctuates "
            "significantly. Tests stability and unnecessary handover avoidance."
        ),
        base_stations=[
            BaseStation(cell_id=0, x=500, y=500, shadowing_sigma_db=10.0, base_load=0.40),
            BaseStation(cell_id=1, x=1500, y=500, shadowing_sigma_db=10.0, base_load=0.40),
            BaseStation(cell_id=2, x=500, y=1500, shadowing_sigma_db=10.0, base_load=0.40),
            BaseStation(cell_id=3, x=1500, y=1500, shadowing_sigma_db=10.0, base_load=0.40),
            BaseStation(cell_id=4, x=1000, y=1000, shadowing_sigma_db=10.0, base_load=0.35),
        ],
        start_x=1000, start_y=100,
        waypoints=[(500, 500), (1500, 1500), (500, 1500), (1500, 500)],
        speed_mps=12.0,
        seed=99,
        duration_steps=300,
    ),

    "congested_target": Scenario(
        name="Strong but Congested Target",
        description=(
            "A strong neighbour cell is heavily loaded. Tests whether QoS-aware policies "
            "avoid switching to a cell that offers weak service despite good RSRP."
        ),
        base_stations=[
            BaseStation(cell_id=0, x=500, y=1000, shadowing_sigma_db=4.0, base_load=0.25),
            BaseStation(cell_id=1, x=1200, y=1000, shadowing_sigma_db=4.0, base_load=0.92),
            BaseStation(cell_id=2, x=1700, y=1000, shadowing_sigma_db=4.0, base_load=0.30),
        ],
        start_x=300, start_y=1000,
        waypoints=[(1000, 1000), (1900, 1000)],
        speed_mps=12.0,
        seed=55,
        duration_steps=200,
    ),

    "sudden_deterioration": Scenario(
        name="Sudden Signal Deterioration",
        description=(
            "Serving cell signal degrades abruptly mid-simulation (simulated via "
            "trajectory moving rapidly away). Tests how quickly policies react to "
            "impending link failure."
        ),
        base_stations=[
            BaseStation(cell_id=0, x=300, y=300, shadowing_sigma_db=5.0, base_load=0.30),
            BaseStation(cell_id=1, x=1700, y=1700, shadowing_sigma_db=5.0, base_load=0.30),
            BaseStation(cell_id=2, x=1000, y=1000, shadowing_sigma_db=5.0, base_load=0.35),
        ],
        start_x=350, start_y=350,
        waypoints=[(1900, 1900)],
        speed_mps=20.0,
        seed=13,
        duration_steps=180,
    ),

    "unseen_route": Scenario(
        name="Unseen Route (Generalization Test)",
        description=(
            "A diagonal route not matching any training/tuning trajectory. Tests "
            "policy robustness to novel mobility patterns."
        ),
        base_stations=[
            BaseStation(cell_id=0, x=400, y=1600, shadowing_sigma_db=5.0, base_load=0.35),
            BaseStation(cell_id=1, x=1000, y=1000, shadowing_sigma_db=5.0, base_load=0.35),
            BaseStation(cell_id=2, x=1600, y=400, shadowing_sigma_db=5.0, base_load=0.35),
        ],
        start_x=100, start_y=1900,
        waypoints=[(1000, 1000), (1900, 100)],
        speed_mps=18.0,
        seed=314,
        duration_steps=200,
    ),
}


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]
