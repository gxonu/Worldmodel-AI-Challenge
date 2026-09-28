# 17프레임 창을 독립적으로 VAE 인코딩해 저장 — 추론과 동일한 인코딩 방식 보장.
# 산출물: latents/<episode행번호>_<시작프레임>.pt = {latent (16,5,40,64) bf16, act (16,6) fp16, motion float}
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
import bootstrap  # noqa: F401

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from load_dit import CKPT_DIR
from so100_dataset import TRAIN_ROOT, INDEX_PATH, NUM_FRAMES, letterbox_batch, read_frames

OUT_DIR = Path(os.environ.get(
    "COSMOS_LATENT_DIR",
    str(Path(__file__).resolve().parent / "latents"),
))
STRIDE = int(os.environ.get("LATENT_STRIDE", "4"))
SPLIT = os.environ.get("LATENT_SPLIT", "train")
ENCODE_BATCH = int(os.environ.get("LATENT_BATCH", "8"))
EXPECTED_LATENT_SHAPE = (16, 5, 40, 64)
EXPECTED_ACTION_SHAPE = (16, 6)


def valid_cache(
    path: Path,
    expected_action: np.ndarray | None = None,
    expected_motion: float | None = None,
) -> bool:
    """Return whether a completed latent cache has the submitted tensor schema."""
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        valid = (
            isinstance(payload, dict)
            and isinstance(payload.get("latent"), torch.Tensor)
            and tuple(payload["latent"].shape) == EXPECTED_LATENT_SHAPE
            and payload["latent"].dtype == torch.bfloat16
            and isinstance(payload.get("act"), torch.Tensor)
            and tuple(payload["act"].shape) == EXPECTED_ACTION_SHAPE
            and payload["act"].dtype == torch.float16
            and np.isfinite(float(payload.get("motion")))
        )
        if not valid:
            return False
        if expected_action is not None and not torch.equal(
            payload["act"], torch.from_numpy(expected_action).to(torch.float16)
        ):
            return False
        if expected_motion is not None and not np.isclose(
            float(payload["motion"]), expected_motion, rtol=0, atol=1e-12
        ):
            return False
        return True
    except (EOFError, OSError, RuntimeError, TypeError, ValueError):
        return False


def atomic_save(payload: dict[str, object], output: Path, rank: int) -> None:
    """Publish a cache only after torch.save completed successfully."""
    temporary = output.with_name(f".{output.name}.rank{rank}.tmp")
    try:
        torch.save(payload, temporary)
        if not valid_cache(temporary):
            raise RuntimeError(f"latent cache validation failed: {temporary}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import WanVAE

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    vae = WanVAE(z_dim=16, vae_pth=str(CKPT_DIR / "tokenizer.pth"), dtype=torch.bfloat16, device="cuda")
    stats = json.loads((TRAIN_ROOT / "so100_action_statistics.json").read_text(encoding="utf-8"))
    a_mean = np.array(stats["mean"], dtype=np.float32)
    a_std = np.array(stats["std"], dtype=np.float32)

    if SPLIT not in {"train", "val"}:
        raise SystemExit(f"LATENT_SPLIT must be train or val, got {SPLIT!r}")
    episodes = pd.read_parquet(INDEX_PATH)
    episodes = episodes[episodes.split == SPLIT].reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    order = list(range(len(episodes)))
    if os.environ.get("REVERSE") == "1":
        order = order[::-1]  # legacy two-worker mode
    order = order[rank::world_size]
    print(
        f"latent worker split={SPLIT} rank={rank}/{world_size} "
        f"device={device} episodes={len(order)} out={OUT_DIR} batch={ENCODE_BATCH}",
        flush=True,
    )

    done, t0 = 0, time.time()
    rows = list(episodes.itertuples())
    show_all_bars = os.environ.get("LATENT_TQDM_ALL", "0") == "1"
    progress = tqdm(
        order,
        total=len(order),
        desc=f"{SPLIT} latent rank {rank}/{world_size}",
        position=rank if show_all_bars else 0,
        disable=not (show_all_bars or rank == 0),
        dynamic_ncols=True,
        unit="episode",
    )
    for n_proc, ep_i in enumerate(progress):
        row = rows[ep_i]
        acts_all = np.stack(
            pd.read_parquet(TRAIN_ROOT / row.parquet, columns=["action"])["action"].to_numpy()
        ).astype(np.float32)
        starts = list(range(0, row.num_frames - NUM_FRAMES + 1, STRIDE))
        todo = []
        for s in starts:
            expected_action = (acts_all[s : s + NUM_FRAMES - 1] - a_mean) / a_std
            expected_motion = (
                float(np.abs(np.diff(expected_action, axis=0)).mean())
                if len(expected_action) > 1 else 0.0
            )
            if not valid_cache(
                OUT_DIR / f"{ep_i:06d}_{s:04d}.pt",
                expected_action,
                expected_motion,
            ):
                todo.append(s)
        if not todo:
            continue
        try:
            # 에피소드 영상은 한 번만 디코드
            frames_all = read_frames(TRAIN_ROOT / row.video, 0, row.num_frames)
        except Exception as e:
            print(f"skip ep{ep_i}: {e}", flush=True)
            continue
        cursor = 0
        active_batch = max(1, ENCODE_BATCH)
        while cursor < len(todo):
            chunk_starts = todo[cursor : cursor + active_batch]
            # Keep the exact float32 CPU letterbox contract used by the dataset,
            # then combine clips for one VAE call.  Only the expensive encode is
            # batched; saved files remain one window each.
            clips = []
            for s in chunk_starts:
                frames = (
                    torch.from_numpy(frames_all[s : s + NUM_FRAMES])
                    .float()
                    .permute(0, 3, 1, 2)
                    / 255.0
                )
                clips.append(letterbox_batch(frames).permute(1, 0, 2, 3))
            clip_batch = torch.stack(clips) * 2.0 - 1.0  # (B,3,17,320,512)
            try:
                with torch.no_grad():
                    latent_batch = vae.encode(clip_batch.to(device, torch.bfloat16)).cpu()
            except torch.OutOfMemoryError:
                del clip_batch
                torch.cuda.empty_cache()
                if active_batch == 1:
                    raise
                active_batch = max(1, active_batch // 2)
                print(
                    f"rank {rank}: latent batch OOM, retry with LATENT_BATCH={active_batch}",
                    flush=True,
                )
                continue

            for latent, s in zip(latent_batch, chunk_starts):
                act = (acts_all[s : s + NUM_FRAMES - 1] - a_mean) / a_std  # (16,6)
                motion = float(np.abs(np.diff(act, axis=0)).mean()) if len(act) > 1 else 0.0
                output = OUT_DIR / f"{ep_i:06d}_{s:04d}.pt"
                atomic_save(
                    {
                        # A slice of a batched CPU tensor may retain the complete
                        # backing storage. Clone it so each cache file contains
                        # only one window.
                        "latent": latent.clone(),
                        "act": torch.from_numpy(act).to(torch.float16),
                        "motion": motion,
                    },
                    output,
                    rank,
                )
                done += 1
            cursor += len(chunk_starts)
        if show_all_bars or rank == 0:
            progress.set_postfix(windows=done, elapsed_min=f"{(time.time() - t0) / 60:.1f}")
        if (n_proc + 1) % 200 == 0:
            el = time.time() - t0
            print(f"proc {n_proc + 1}/{len(order)} windows {done} elapsed {el / 60:.0f}min", flush=True)
    print(f"done: {done} windows", flush=True)


if __name__ == "__main__":
    main()
