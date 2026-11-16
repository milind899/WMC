"""
export_results.py — Load and reformat saved benchmark outputs.
"""

import sys
import json
import csv
from pathlib import Path

import pandas as pd


def load_all_runs(output_dir: Path) -> pd.DataFrame:
    records = []
    for f in output_dir.glob("run_*.json"):
        with open(f) as fh:
            records.append(json.load(fh))
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def export_csv(df: pd.DataFrame, out_path: Path) -> None:
    df.to_csv(out_path, index=False)
    print(f"Exported to {out_path}")


if __name__ == "__main__":
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs")
    df = load_all_runs(output_dir)
    if df.empty:
        print("No run files found.")
    else:
        export_csv(df, output_dir / "all_runs_export.csv")
        print(df[["scenario", "policy", "seed", "handover_count", "mean_throughput_mbps", "rlf_rate"]].to_string())
