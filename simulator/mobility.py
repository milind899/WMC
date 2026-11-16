"""
mobility.py — Waypoint-based user mobility model for MobiSense.

Users follow a sequence of waypoints at a configured speed. Direction noise
models realistic trajectory variation. Boundary reflection keeps users inside
the simulation area.

Speed units: m/s. Position units: metres. Time step: seconds.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np


@dataclass
class UserState:
    x: float
    y: float
    speed: float
    heading_rad: float
    waypoint_index: int = 0


@dataclass
class MobilityConfig:
    start_x: float = 0.0
    start_y: float = 0.0
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    speed_mps: float = 15.0
    speed_std_mps: float = 1.0
    heading_noise_std_rad: float = 0.05
    area_width_m: float = 2000.0
    area_height_m: float = 2000.0
    dt_s: float = 1.0


class MobilityEngine:
    """
    Discrete-time waypoint mobility model.

    The user moves toward successive waypoints. When the final waypoint is
    reached the user holds position. Speed and heading jitter model realistic
    movement uncertainty.
    """

    def __init__(self, config: MobilityConfig, rng: np.random.Generator):
        self._cfg = config
        self._rng = rng
        self._state = UserState(
            x=config.start_x,
            y=config.start_y,
            speed=config.speed_mps,
            heading_rad=0.0,
        )
        self._trajectory: list[tuple[float, float]] = [(config.start_x, config.start_y)]

    @property
    def state(self) -> UserState:
        return self._state

    @property
    def trajectory(self) -> list[tuple[float, float]]:
        return self._trajectory

    def step(self) -> UserState:
        cfg = self._cfg
        s = self._state

        waypoints = cfg.waypoints
        if waypoints and s.waypoint_index < len(waypoints):
            target_x, target_y = waypoints[s.waypoint_index]
            dx = target_x - s.x
            dy = target_y - s.y
            dist = np.hypot(dx, dy)
            if dist < cfg.speed_mps * cfg.dt_s:
                s.waypoint_index += 1
            if dist > 1e-3:
                s.heading_rad = np.arctan2(dy, dx)
        else:
            target_x, target_y = s.x, s.y

        heading = s.heading_rad + self._rng.normal(0.0, cfg.heading_noise_std_rad)
        speed = max(
            0.1,
            cfg.speed_mps + self._rng.normal(0.0, cfg.speed_std_mps),
        )

        new_x = s.x + speed * np.cos(heading) * cfg.dt_s
        new_y = s.y + speed * np.sin(heading) * cfg.dt_s

        new_x, new_y, heading = _reflect_boundary(
            new_x, new_y, heading, cfg.area_width_m, cfg.area_height_m
        )

        self._state = UserState(
            x=new_x,
            y=new_y,
            speed=speed,
            heading_rad=heading,
            waypoint_index=s.waypoint_index,
        )
        self._trajectory.append((new_x, new_y))
        return self._state


def _reflect_boundary(
    x: float, y: float, heading: float, width: float, height: float
) -> tuple[float, float, float]:
    if x < 0:
        x = -x
        heading = np.pi - heading
    elif x > width:
        x = 2 * width - x
        heading = np.pi - heading

    if y < 0:
        y = -y
        heading = -heading
    elif y > height:
        y = 2 * height - y
        heading = -heading

    x = float(np.clip(x, 0, width))
    y = float(np.clip(y, 0, height))
    return x, y, heading
