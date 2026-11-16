"""
test_reproducibility.py — Verify identical seeds produce identical simulation results.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.scenarios import get_scenario
from simulator.handover_policies import PolicyA, PolicyB, PolicyC, PolicyD
from simulator.engine import SimulationEngine


def run_sim(scenario_name, policy_cls, seed):
    sc = get_scenario(scenario_name)
    result = SimulationEngine(sc, policy_cls(), seed=seed).run()
    rsrp_series = [s.rsrp_dbm for s in result.steps]
    return rsrp_series, result.kpis.handover_count


@pytest.mark.parametrize("policy_cls", [PolicyA, PolicyB, PolicyC, PolicyD])
def test_same_seed_same_result(policy_cls):
    r1, ho1 = run_sim("normal_mobility", policy_cls, seed=42)
    r2, ho2 = run_sim("normal_mobility", policy_cls, seed=42)
    assert r1 == r2
    assert ho1 == ho2


@pytest.mark.parametrize("policy_cls", [PolicyA, PolicyD])
def test_different_seeds_differ(policy_cls):
    r1, _ = run_sim("normal_mobility", policy_cls, seed=1)
    r2, _ = run_sim("normal_mobility", policy_cls, seed=2)
    assert r1 != r2


def test_seed_override():
    sc = get_scenario("normal_mobility")
    r1, _ = run_sim("normal_mobility", PolicyA, seed=99)
    result2 = SimulationEngine(sc, PolicyA(), seed=99).run()
    r2 = [s.rsrp_dbm for s in result2.steps]
    assert r1 == r2
