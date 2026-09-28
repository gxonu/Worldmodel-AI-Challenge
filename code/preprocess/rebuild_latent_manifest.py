"""Validate latent files and rebuild a manifest from the episode index.

The repository may ship a manifest generated from an older episode ordering.
Latent filenames intentionally use the *filtered train-row index*, so rebuilding
episode_index.parquet can keep the same number of windows while changing many
filenames.  Deriving paths and motion from the current action parquet avoids
opening 100k small latent files and guarantees the sampler matches this cache.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))

from so100_dataset import INDEX_PATH, NUM_FRAMES, TRAIN_ROOT


HERE = Path(__file__).resolve().parent
LATENT_DIR = Path(os.environ.get("COSMOS_LATENT_DIR", HERE / "latents"))
MANIFEST = Path(os.environ.get("COSMOS_LATENT_MANIFEST", HERE / "latent_manifest.parquet"))
STRIDE = int(os.environ.get("LATENT_STRIDE", "4"))
SPLIT = os.environ.get("LATENT_SPLIT", "train")
EXPECTED_LATENT_SHAPE = (16, 5, 40, 64)
EXPECTED_ACTION_SHAPE = (16, 6)


def validate_cache(path: Path, expected_action: np.ndarray, expected_motion: float) -> str | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(payload, dict):
            return "payload is not a dictionary"
        latent = payload.get("latent")
        action = payload.get("act")
        if not isinstance(latent, torch.Tensor) or tuple(latent.shape) != EXPECTED_LATENT_SHAPE:
            return f"invalid latent shape: {getattr(latent, 'shape', None)}"
        if latent.dtype != torch.bfloat16:
            return f"invalid latent dtype: {latent.dtype}"
        if not isinstance(action, torch.Tensor) or tuple(action.shape) != EXPECTED_ACTION_SHAPE:
            return f"invalid action shape: {getattr(action, 'shape', None)}"
        if action.dtype != torch.float16:
            return f"invalid action dtype: {action.dtype}"
        if not torch.equal(action, torch.from_numpy(expected_action).to(torch.float16)):
            return "cached action does not match the train parquet"
        motion = float(payload.get("motion"))
        if not np.isfinite(motion) or not np.isclose(motion, expected_motion, rtol=0, atol=1e-12):
            return "cached motion does not match the train parquet"
    except (EOFError, OSError, RuntimeError, TypeError, ValueError) as error:
        return f"unreadable cache: {error}"
    return None


def main() -> None:
    index = pd.read_parquet(INDEX_PATH)
    if SPLIT not in {"train", "val"}:
        raise SystemExit(f"LATENT_SPLIT must be train or val, got {SPLIT!r}")
    episodes = index[index.split == SPLIT].reset_index(drop=True)
    stats = json.loads((TRAIN_ROOT / "so100_action_statistics.json").read_text(encoding="utf-8"))
    action_mean = np.asarray(stats["mean"], dtype=np.float32)
    action_std = np.asarray(stats["std"], dtype=np.float32)

    rows: list[dict[str, object]] = []
    expected: set[str] = set()
    invalid_count = 0
    invalid_examples: list[dict[str, str]] = []
    for ep_i, row in enumerate(tqdm(episodes.itertuples(index=False), total=len(episodes), desc="manifest")):
        actions = np.stack(
            pd.read_parquet(TRAIN_ROOT / row.parquet, columns=["action"])["action"].to_numpy()
        ).astype(np.float32)
        actions = (actions - action_mean) / action_std
        for start in range(0, int(row.num_frames) - NUM_FRAMES + 1, STRIDE):
            name = f"{ep_i:06d}_{start:04d}.pt"
            act = actions[start : start + NUM_FRAMES - 1]
            motion = float(np.abs(np.diff(act, axis=0)).mean()) if len(act) > 1 else 0.0
            rows.append({"path": name, "motion": motion})
            expected.add(name)
            cache_path = LATENT_DIR / name
            if cache_path.is_file():
                problem = validate_cache(cache_path, act, motion)
                if problem:
                    invalid_count += 1
                    if len(invalid_examples) < 5:
                        invalid_examples.append({"path": name, "problem": problem})

    actual = {
        entry.name
        for entry in os.scandir(LATENT_DIR)
        if entry.is_file() and entry.name.endswith(".pt")
    }
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    report = {
        "manifest_windows": len(rows),
        "cached_windows": len(actual),
        "missing": len(missing),
        "extra": len(extra),
        "invalid": invalid_count,
        "missing_examples": missing[:5],
        "extra_examples": extra[:5],
        "invalid_examples": invalid_examples,
    }
    print(json.dumps(report, indent=2), flush=True)
    if missing or extra or invalid_count or not rows:
        raise SystemExit("latent cache does not exactly match the current episode index")

    frame = pd.DataFrame(rows)
    temp = MANIFEST.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False)
    temp.replace(MANIFEST)
    print(f"saved -> {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
