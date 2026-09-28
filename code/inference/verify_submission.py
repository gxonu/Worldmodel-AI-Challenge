#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    required = {"sample_id", "feature_component", "feature_json"}
    if set(df.columns) != required:
        raise SystemExit(f"bad columns: {list(df.columns)}")
    expected_components = {"Video Feature Component", "DINO Component", "Action Component"}
    if len(df) != 648 or df.sample_id.nunique() != 216:
        raise SystemExit(f"bad shape: rows={len(df)} unique_samples={df.sample_id.nunique()}")
    counts = df.feature_component.value_counts().to_dict()
    if set(counts) != expected_components or any(v != 216 for v in counts.values()):
        raise SystemExit(f"bad component counts: {counts}")
    expected_shapes = {
        "Video Feature Component": (512,),
        "DINO Component": (16, 384),
        "Action Component": (1, 1),
    }
    for row in df.itertuples(index=False):
        value = np.asarray(json.loads(row.feature_json), dtype=np.float64)
        if value.shape != expected_shapes[row.feature_component] or not np.isfinite(value).all():
            raise SystemExit(f"bad feature: {row.sample_id} {row.feature_component}")
    print(f"VALID: {args.csv} | 216 samples | 648 rows | no NaN/Inf")


if __name__ == "__main__":
    main()
