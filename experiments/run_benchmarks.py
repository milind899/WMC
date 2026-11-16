"""
run_benchmarks.py — Reproducible multi-scenario, multi-policy benchmark runner.

Usage:
    python experiments/run_benchmarks.py [--scenarios s1,s2] [--seeds 42,7,99] [--output-dir outputs]

Outputs:
  outputs/run_<scenario>_<policy>_seed<N>.json   per-run raw KPIs
  outputs/benchmark_summary.csv                  aggregated mean ± std across seeds
  outputs/benchmark_comparison.png               comparison bar chart (requires kaleido)
"""

import sys
import json
import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from simulator.scenarios import SCENARIOS, get_scenario
from simulator.handover_policies import PolicyA, PolicyB, PolicyC, PolicyD
from simulator.engine import SimulationEngine


POLICIES = {
    "policy_a": lambda: PolicyA(a3_offset_db=3.0),
    "policy_b": lambda: PolicyB(a3_offset_db=3.0, ttt_steps=3, hysteresis_db=1.0),
    "policy_c": lambda: PolicyC(window=8, horizon_steps=5),
    "policy_d": lambda: PolicyD(window=8, horizon_steps=5),
}

DEFAULT_SEEDS = [42, 7, 99, 13, 55]


def run_benchmark(
    scenario_names: list[str],
    policy_names: list[str],
    seeds: list[int],
    output_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = []

    total = len(scenario_names) * len(policy_names) * len(seeds)
    done = 0

    for sc_name in scenario_names:
        scenario = get_scenario(sc_name)
        for pol_name in policy_names:
            for seed in seeds:
                policy = POLICIES[pol_name]()
                engine = SimulationEngine(scenario, policy, seed=seed)
                result = engine.run()
                kpis = result.kpis

                record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "scenario": sc_name,
                    "policy": pol_name,
                    "policy_display": result.policy_name,
                    "seed": seed,
                    "total_steps": kpis.total_steps,
                    "handover_count": kpis.handover_count,
                    "handover_failure_rate": kpis.handover_failure_rate,
                    "ping_pong_rate": kpis.ping_pong_rate,
                    "rlf_rate": kpis.rlf_rate,
                    "mean_latency_ms": kpis.mean_latency_ms,
                    "p95_latency_ms": kpis.p95_latency_ms,
                    "mean_throughput_mbps": kpis.mean_throughput_mbps,
                    "pdr": kpis.pdr,
                    "mean_interruption_ms": kpis.mean_interruption_ms,
                    "prediction_mae": kpis.prediction_mae,
                }

                fname = output_dir / f"run_{sc_name}_{pol_name}_seed{seed}.json"
                with open(fname, "w") as f:
                    json.dump(record, f, indent=2)

                all_records.append(record)
                done += 1
                pct = done / total * 100
                print(f"[{done:3d}/{total}] ({pct:5.1f}%)  {sc_name:25s}  {pol_name:12s}  seed={seed}  "
                      f"HO={kpis.handover_count:3d}  "
                      f"RLF={kpis.rlf_rate:.3f}  "
                      f"tput={kpis.mean_throughput_mbps:.2f}Mbps")

    return all_records


def aggregate_results(records: list[dict], output_dir: Path) -> Path:
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["scenario"], r["policy"])].append(r)

    metric_keys = [
        "handover_count", "handover_failure_rate", "ping_pong_rate",
        "rlf_rate", "mean_latency_ms", "p95_latency_ms",
        "mean_throughput_mbps", "pdr", "mean_interruption_ms",
    ]

    rows = []
    for (sc, pol), recs in groups.items():
        row = {"scenario": sc, "policy": pol, "policy_display": recs[0]["policy_display"], "n_seeds": len(recs)}
        for k in metric_keys:
            vals = [r[k] for r in recs if r[k] is not None]
            row[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{k}_std"] = float(np.std(vals)) if len(vals) > 1 else float("nan")
        rows.append(row)

    summary_path = output_dir / "benchmark_summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary saved to: {summary_path}")
    return summary_path


def generate_plots(summary_path: Path, output_dir: Path) -> None:
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd

        df = pd.read_csv(summary_path)
        metrics_to_plot = [
            ("handover_count_mean", "Handover Count"),
            ("rlf_rate_mean", "RLF Rate"),
            ("ping_pong_rate_mean", "Ping-Pong Rate"),
            ("mean_throughput_mbps_mean", "Mean Throughput (Mbps)"),
            ("mean_latency_ms_mean", "Mean Latency (ms)"),
            ("pdr_mean", "Packet Delivery Ratio"),
        ]

        for scenario in df["scenario"].unique():
            sc_df = df[df["scenario"] == scenario]
            fig = go.Figure()
            for m_col, m_label in metrics_to_plot:
                if m_col not in sc_df.columns:
                    continue
                fig.add_trace(go.Bar(
                    x=sc_df["policy_display"],
                    y=sc_df[m_col],
                    name=m_label,
                    visible=True,
                ))

            fig.update_layout(
                title=f"Policy Comparison — {scenario}",
                barmode="group",
                template="plotly_dark",
                legend=dict(orientation="h"),
            )

            png_path = output_dir / f"benchmark_{scenario}.png"
            try:
                fig.write_image(str(png_path))
                print(f"Plot saved: {png_path}")
            except Exception as e:
                html_path = output_dir / f"benchmark_{scenario}.html"
                fig.write_html(str(html_path))
                print(f"Plot saved as HTML (kaleido not available): {html_path}")
    except Exception as e:
        print(f"Plot generation skipped: {e}")


def main():
    parser = argparse.ArgumentParser(description="MobiSense Benchmark Runner")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS.keys()),
                        help="Comma-separated scenario names")
    parser.add_argument("--policies", default=",".join(POLICIES.keys()),
                        help="Comma-separated policy names")
    parser.add_argument("--seeds", default="42,7,99",
                        help="Comma-separated random seeds")
    parser.add_argument("--output-dir", default="outputs",
                        help="Output directory")
    args = parser.parse_args()

    scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip() in SCENARIOS]
    policy_names = [p.strip() for p in args.policies.split(",") if p.strip() in POLICIES]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    output_dir = Path(args.output_dir)

    print(f"MobiSense Benchmark Runner")
    print(f"Scenarios : {scenario_names}")
    print(f"Policies  : {policy_names}")
    print(f"Seeds     : {seeds}")
    print(f"Output dir: {output_dir}")
    print("-" * 70)

    records = run_benchmark(scenario_names, policy_names, seeds, output_dir)
    summary_path = aggregate_results(records, output_dir)
    generate_plots(summary_path, output_dir)

    print("\n=== Benchmark complete ===")
    print(f"Total runs: {len(records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
