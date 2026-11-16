"""
radio_model.py — Wireless propagation and radio link model for MobiSense.

Propagation model: 3GPP TR 38.901 UMa (simplified free-space + shadowing).
  PL(d) = 32.44 + 20*log10(fc_MHz) + 20*log10(d_m/1000)   [dB]

RSRP: Ptx_dBm - PL(d)   [dBm]
SINR: RSRP_serving - 10*log10(I + N)  where N = kTB noise floor, I = sum of interferers
Throughput: Shannon bound with cell load factor (approximation only).
Latency: Base RTT + queuing contribution from utilization.

DISCLAIMER: This is a simplified simulation model, not a commercial 4G/5G radio measurement.
"""

import numpy as np

BOLTZMANN_K = 1.38e-23
TEMPERATURE_K = 290.0
BANDWIDTH_HZ = 10e6
NOISE_FIGURE_DB = 7.0
THERMAL_NOISE_DBM = 10 * np.log10(BOLTZMANN_K * TEMPERATURE_K * BANDWIDTH_HZ * 1e3) + NOISE_FIGURE_DB
RLF_RSRP_DBM_THRESHOLD = -120.0


def path_loss_uma(distance_m: float, fc_mhz: float = 2100.0) -> float:
    """
    Simplified UMa path-loss (3GPP TR 38.901 UMa LOS approximation).
    Valid for d_m >= 1 m.
    Returns path loss in dB.
    """
    d = max(distance_m, 1.0)
    return 32.44 + 20 * np.log10(fc_mhz) + 20 * np.log10(d / 1000.0)


def compute_rsrp(
    distance_m: float,
    tx_power_dbm: float = 46.0,
    fc_mhz: float = 2100.0,
    shadowing_db: float = 0.0,
) -> float:
    """
    RSRP [dBm] = Ptx - PathLoss - Shadowing
    """
    pl = path_loss_uma(distance_m, fc_mhz)
    return tx_power_dbm - pl - shadowing_db


def compute_sinr(
    serving_rsrp_dbm: float,
    interferer_rsrp_dbm_list: list[float],
) -> float:
    """
    SINR [dB] = serving_power / (interference_power + noise_power)
    All in linear domain, then converted back to dB.
    """
    serving_mw = dbm_to_mw(serving_rsrp_dbm)
    noise_mw = dbm_to_mw(THERMAL_NOISE_DBM)
    interference_mw = sum(dbm_to_mw(r) for r in interferer_rsrp_dbm_list) if interferer_rsrp_dbm_list else 0.0
    sinr_linear = serving_mw / (interference_mw + noise_mw)
    return 10 * np.log10(max(sinr_linear, 1e-12))


def sinr_to_throughput(sinr_db: float, bandwidth_hz: float = BANDWIDTH_HZ, load_factor: float = 1.0) -> float:
    """
    Shannon-bound throughput estimate [Mbps], de-rated by cell load factor.
    Efficiency cap applied at 0.75 to reflect practical MCS limitations.
    load_factor: fraction of RBs already consumed by other users (0.0–1.0).

    Returns throughput in Mbps. This is an approximation; not a MAC-layer simulation.
    """
    sinr_linear = 10 ** (sinr_db / 10.0)
    spectral_efficiency = min(np.log2(1 + sinr_linear), 7.0) * 0.75
    available_bw = bandwidth_hz * max(1.0 - load_factor, 0.05)
    return (spectral_efficiency * available_bw) / 1e6


def estimate_latency(load_factor: float, base_rtt_ms: float = 10.0) -> float:
    """
    Estimated one-way latency [ms].
    Models queuing delay using M/M/1 approximation: W = 1 / (mu - lambda).
    load_factor: server utilization rho (0 to 1).
    Returns total latency in ms.
    """
    rho = min(load_factor, 0.99)
    queuing_delay_ms = (rho / (1.0 - rho)) * 2.0
    return base_rtt_ms + queuing_delay_ms


def dbm_to_mw(dbm: float) -> float:
    return 10 ** (dbm / 10.0)


def mw_to_dbm(mw: float) -> float:
    return 10 * np.log10(max(mw, 1e-30))


def add_measurement_noise(value_db: float, sigma_db: float, rng: np.random.Generator) -> float:
    """Add Gaussian measurement noise to a dB-domain value."""
    if sigma_db <= 0:
        return value_db
    return value_db + rng.normal(0.0, sigma_db)


def generate_shadowing(sigma_db: float, rng: np.random.Generator) -> float:
    """
    Log-normal shadowing sample [dB]. Zero-mean Gaussian in dB domain.
    sigma_db typical: 4–10 dB for UMa.
    """
    if sigma_db <= 0:
        return 0.0
    return rng.normal(0.0, sigma_db)
