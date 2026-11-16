"""
test_scenarios.py — Tests that every scenario runs to completion without error.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.scenarios import SCENARIOS, get_scenario
from simulator.handover_policies import PolicyA, PolicyD
from simulator.engine import SimulationEngine


@pytest.mark.parametrize("scenario_name", list(SCENARIOS.keys()))
def test_scenario_runs_to_completion_policy_a(scenario_name):
    sc = get_scenario(scenario_name)
    policy = PolicyA()
    engine = SimulationEngine(sc, policy)
    result = engine.run()
    assert result.kpis.total_steps == sc.duration_steps
    assert len(result.steps) == sc.duration_steps
    assert result.policy_name == policy.name


@pytest.mark.parametrize("scenario_name", list(SCENARIOS.keys()))
def test_scenario_runs_to_completion_policy_d(scenario_name):
    sc = get_scenario(scenario_name)
    policy = PolicyD()
    engine = SimulationEngine(sc, policy)
    result = engine.run()
    assert result.kpis.total_steps == sc.duration_steps


def test_get_scenario_invalid_raises():
    with pytest.raises(ValueError):
        get_scenario("does_not_exist")


def test_scenario_has_expected_fields():
    sc = get_scenario("normal_mobility")
    assert len(sc.base_stations) >= 2
    assert len(sc.waypoints) >= 1
    assert sc.duration_steps > 0
    assert sc.speed_mps > 0


def test_simulation_result_has_trajectory():
    sc = get_scenario("normal_mobility")
    result = SimulationEngine(sc, PolicyA()).run()
    assert len(result.trajectory) == sc.duration_steps + 1
    assert all(len(pt) == 2 for pt in result.trajectory)
