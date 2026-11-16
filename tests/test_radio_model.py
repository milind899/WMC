"""
test_radio_model.py — Tests for path-loss, RSRP, SINR, and throughput computations.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.radio_model import (
    path_loss_uma,
    compute_rsrp,
    compute_sinr,
    sinr_to_throughput,
    estimate_latency,
    dbm_to_mw,
    mw_to_dbm,
    add_measurement_noise,
    THERMAL_NOISE_DBM,
)


class TestPathLoss:
    def test_increases_with_distance(self):
        pl_near = path_loss_uma(100)
        pl_far = path_loss_uma(1000)
        assert pl_far > pl_near

    def test_minimum_distance_clamped(self):
        pl_zero = path_loss_uma(0)
        pl_one = path_loss_uma(1)
        assert abs(pl_zero - pl_one) < 1e-6

    def test_positive_loss(self):
        for d in [10, 100, 500, 2000]:
            assert path_loss_uma(d) > 0

    def test_frequency_effect(self):
        pl_low = path_loss_uma(500, fc_mhz=700)
        pl_high = path_loss_uma(500, fc_mhz=2600)
        assert pl_high > pl_low


class TestRSRP:
    def test_decreases_with_distance(self):
        rsrp_near = compute_rsrp(100)
        rsrp_far = compute_rsrp(2000)
        assert rsrp_far < rsrp_near

    def test_bounded_by_tx_power(self):
        rsrp = compute_rsrp(1, tx_power_dbm=46.0)
        assert rsrp <= 46.0

    def test_shadowing_effect(self):
        base = compute_rsrp(500, shadowing_db=0.0)
        with_shadowing = compute_rsrp(500, shadowing_db=10.0)
        assert base > with_shadowing

    def test_typical_rsrp_range(self):
        rsrp = compute_rsrp(2000)
        assert -130 < rsrp < -60


class TestSINR:
    def test_no_interference_higher_than_with(self):
        sinr_no_int = compute_sinr(-80.0, [])
        sinr_with_int = compute_sinr(-80.0, [-85.0, -90.0])
        assert sinr_no_int > sinr_with_int

    def test_finite_sinr(self):
        sinr = compute_sinr(-90.0, [-95.0])
        assert np.isfinite(sinr)

    def test_thermal_noise_floor(self):
        sinr = compute_sinr(THERMAL_NOISE_DBM - 20, [])
        assert sinr < -15


class TestThroughput:
    def test_positive(self):
        for sinr in [-5, 0, 5, 15, 25]:
            tput = sinr_to_throughput(sinr)
            assert tput > 0

    def test_increases_with_sinr(self):
        t1 = sinr_to_throughput(0)
        t2 = sinr_to_throughput(20)
        assert t2 > t1

    def test_load_factor_reduces_throughput(self):
        t_low_load = sinr_to_throughput(10, load_factor=0.1)
        t_high_load = sinr_to_throughput(10, load_factor=0.9)
        assert t_low_load > t_high_load

    def test_saturation_cap(self):
        t_very_high_sinr = sinr_to_throughput(100)
        t_moderate = sinr_to_throughput(30)
        ratio = t_very_high_sinr / max(t_moderate, 1e-6)
        assert ratio < 2.0


class TestLatency:
    def test_increases_with_load(self):
        lat_low = estimate_latency(0.1)
        lat_high = estimate_latency(0.9)
        assert lat_high > lat_low

    def test_minimum_base(self):
        lat = estimate_latency(0.0)
        assert lat >= 10.0


class TestHelpers:
    def test_dbm_mw_roundtrip(self):
        for dbm in [-100, -80, -60, -40]:
            assert abs(mw_to_dbm(dbm_to_mw(dbm)) - dbm) < 1e-9

    def test_noise_adds_variation(self):
        rng = np.random.default_rng(0)
        samples = [add_measurement_noise(-80.0, 3.0, rng) for _ in range(100)]
        assert np.std(samples) > 0.5
