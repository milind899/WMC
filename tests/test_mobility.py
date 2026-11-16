"""
test_mobility.py — Tests for waypoint mobility, boundary conditions, reproducibility.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.mobility import MobilityEngine, MobilityConfig


def make_engine(seed=42, speed=15.0, waypoints=None):
    if waypoints is None:
        waypoints = [(1000, 1000), (1900, 1000)]
    cfg = MobilityConfig(
        start_x=100, start_y=500,
        waypoints=waypoints,
        speed_mps=speed,
        area_width_m=2000, area_height_m=2000,
    )
    rng = np.random.default_rng(seed)
    return MobilityEngine(cfg, rng)


class TestMobility:
    def test_moves_from_start(self):
        eng = make_engine()
        s = eng.step()
        assert (s.x, s.y) != (100, 500)

    def test_stays_in_bounds(self):
        eng = make_engine(speed=100.0)
        for _ in range(300):
            s = eng.step()
            assert 0 <= s.x <= 2000
            assert 0 <= s.y <= 2000

    def test_speed_approximate(self):
        eng = make_engine(speed=20.0)
        speeds = [eng.step().speed for _ in range(50)]
        mean_speed = np.mean(speeds)
        assert 15 < mean_speed < 25

    def test_trajectory_grows(self):
        eng = make_engine()
        for _ in range(10):
            eng.step()
        assert len(eng.trajectory) == 11

    def test_reproducibility(self):
        eng1 = make_engine(seed=123)
        eng2 = make_engine(seed=123)
        for _ in range(50):
            s1 = eng1.step()
            s2 = eng2.step()
            assert abs(s1.x - s2.x) < 1e-9
            assert abs(s1.y - s2.y) < 1e-9

    def test_different_seeds_differ(self):
        eng1 = make_engine(seed=1)
        eng2 = make_engine(seed=2)
        traj1 = [eng1.step() for _ in range(20)]
        traj2 = [eng2.step() for _ in range(20)]
        diffs = [abs(a.x - b.x) + abs(a.y - b.y) for a, b in zip(traj1, traj2)]
        assert max(diffs) > 0.1

    def test_waypoint_progress(self):
        eng = make_engine(speed=50.0, waypoints=[(200, 500)])
        for _ in range(20):
            eng.step()
        assert eng.state.waypoint_index >= 1

    def test_no_waypoints_still_moves(self):
        cfg = MobilityConfig(start_x=1000, start_y=1000, waypoints=[], speed_mps=10.0)
        rng = np.random.default_rng(0)
        eng = MobilityEngine(cfg, rng)
        for _ in range(10):
            s = eng.step()
            assert 0 <= s.x <= 2000
