"""
Load saved population/evolution results; compute nondominated set; plot/save Pareto front per condition.
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from pathlib import Path


def dominates(row_a, row_b, obj1="obj1_performance", obj2="obj2_cost"):
    """Maximize obj1, minimize obj2."""
    better1 = row_a[obj1] >= row_b[obj1]
    better2 = row_a[obj2] <= row_b[obj2]
    strict1 = row_a[obj1] > row_b[obj1]
    strict2 = row_a[obj2] < row_b[obj2]
    return better1 and better2 and (strict1 or strict2)


def nondominated_set(df, obj1="obj1_performance", obj2="obj2_cost"):
    """Return DataFrame of nondominated rows."""
    rows = []
    for i, r in df.iterrows():
        if not any(dominates(df.loc[j], r, obj1, obj2) for j in df.index if j != i):
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="+", help="population.csv or run dirs")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--obj1", type=str, default="obj1_performance")
    parser.add_argument("--obj2", type=str, default="obj2_cost")
    args = parser.parse_args()

    all_dfs = []
    for path in args.input:
        p = Path(path)
        if p.is_dir():
            csv_path = p / "population.csv"
            if not csv_path.exists():
                csv_path = p / "pareto.csv"
            path = str(csv_path)
        if not os.path.isfile(path):
            print(f"Skip {path}")
            continue
        df = pd.read_csv(path)
        if "obj1_performance" not in df.columns and "obj1" in df.columns:
            df = df.rename(columns={"obj1": "obj1_performance", "obj2": "obj2_cost"})
        nd = nondominated_set(df, args.obj1, args.obj2)
        nd["source"] = path
        all_dfs.append(nd)
    if not all_dfs:
        print("No data")
        return
    out = pd.concat(all_dfs, ignore_index=True)
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"Wrote {args.output}")
    else:
        print(out.to_string())


if __name__ == "__main__":
    main()
