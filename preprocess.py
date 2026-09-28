#!/usr/bin/env python3
"""Build the deterministic train index and Cosmos latent caches on one GPU."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def run(script: Path, env: dict[str, str], *args: str) -> None:
    command = [sys.executable, str(script), *args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=script.parent, env=env, check=True)


def require(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=Path("data/train"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("pretrained/checkpoints"))
    parser.add_argument("--work-root", type=Path, default=Path("work"))
    parser.add_argument("--gpu", default="0", help="one visible GPU index")
    parser.add_argument("--latent-batch", type=int, default=8)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    train_root = absolute(args.train_root)
    checkpoint_root = absolute(args.checkpoint_root)
    work_root = absolute(args.work_root)
    index_path = work_root / "episode_index.parquet"
    train_latents = work_root / "latents_stride4"
    val_latents = work_root / "val_latents"
    manifest_s4 = work_root / "latent_manifest_s4.parquet"
    manifest_s8 = work_root / "latent_manifest_s8.parquet"
    manifest_val = work_root / "latent_manifest_val.parquet"
    split_reference = ROOT / "assets" / "episode_index.parquet"

    require(train_root / "so100_action_statistics.json", "train action statistics")
    require(checkpoint_root / "tokenizer.pth", "Cosmos tokenizer")
    require(split_reference, "submitted episode split reference")
    require(
        checkpoint_root / "robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt",
        "Cosmos action-conditioned base model",
    )
    require(
        checkpoint_root / "robot/action-cond/cr1_empty_string_text_embeddings.pt",
        "Cosmos empty-text embedding",
    )
    if args.check_only:
        print("preprocess inputs OK")
        return

    work_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ("REVERSE", "LATENT_TQDM_ALL", "LATENT_SPLIT", "LATENT_STRIDE", "LATENT_BATCH"):
        env.pop(key, None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "SO100_TRAIN_ROOT": str(train_root),
            "SO100_INDEX_PATH": str(index_path),
            "SO100_SPLIT_REFERENCE": str(split_reference),
            "COSMOS_CHECKPOINT_DIR": str(checkpoint_root),
            "LATENT_BATCH": str(args.latent_batch),
            "LATENT_STRIDE": "4",
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
        }
    )
    pre = CODE / "preprocess"
    run(pre / "build_index.py", env)

    env["LATENT_SPLIT"] = "train"
    env["COSMOS_LATENT_DIR"] = str(train_latents)
    run(pre / "precompute_latents.py", env)
    env["COSMOS_LATENT_MANIFEST"] = str(manifest_s4)
    run(pre / "rebuild_latent_manifest.py", env)

    # Stage 1 historically used stride 8. Its windows are an exact subset of
    # the stride-4 cache, so no second VAE pass is needed.
    import pandas as pd

    frame = pd.read_parquet(manifest_s4)
    starts = frame["path"].str.rsplit("_", n=1).str[-1].str.removesuffix(".pt").astype(int)
    stage1 = frame.loc[(starts % 8) == 0].reset_index(drop=True)
    if len(frame) != 203_165 or len(stage1) != 104_368:
        raise RuntimeError(
            f"unexpected manifest size: stride4={len(frame):,}, stride8={len(stage1):,}"
        )
    stage1.to_parquet(manifest_s8, index=False)
    print(f"saved {manifest_s8}: {len(stage1):,} stride-8 windows", flush=True)

    env["LATENT_SPLIT"] = "val"
    env["COSMOS_LATENT_DIR"] = str(val_latents)
    run(pre / "precompute_latents.py", env)
    env["COSMOS_LATENT_MANIFEST"] = str(manifest_val)
    run(pre / "rebuild_latent_manifest.py", env)

    print("preprocessing complete", flush=True)
    print(f"  index: {index_path}")
    print(f"  train latents: {train_latents}")
    print(f"  stage1 manifest: {manifest_s8}")
    print(f"  stage2 manifest: {manifest_s4}")
    print(f"  validation latents: {val_latents}")
    print(f"  validation manifest: {manifest_val}")


if __name__ == "__main__":
    main()
