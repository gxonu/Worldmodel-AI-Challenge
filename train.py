#!/usr/bin/env python3
"""Run the submitted two-stage single-GPU training recipe."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN_CODE = ROOT / "code" / "train"


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def require(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def run(script: Path, env: dict[str, str], log_path: Path, *args: str) -> None:
    command = [sys.executable, str(script), *args]
    print("+", " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=script.parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def clean_recipe_env(env: dict[str, str]) -> None:
    # The training engine also contains historical experiment branches. Clear
    # every flag understood by that engine before applying the submitted recipe
    # so an invoking shell cannot silently select another architecture, loss,
    # warm start, sampler, or validation schedule.
    for key in (
        "DRAFT", "ATOK", "ACTX_ONLY", "ACT_DROP", "FULLFT", "RUN_CKPT",
        "DELTA", "AUX", "AUX_JUDGE", "AUX_WARMUP", "W_AUX", "P_AUX",
        "CLEAN", "CLEAN_FILE", "DRAFT_K", "DRAFT_P", "W_R", "W_ANCHOR",
        "JOINT_W", "MAX_STEPS", "RESUME_FROM", "WIDEN_FROM", "MERGE_FROM",
        "LR_NEW", "NOREPL", "UNIFIED_V2", "UNIFIED_WARMUP", "W_X0",
        "W_TEMPORAL", "W_PRESERVE", "W_COUNTERFACTUAL", "COUNTERFACTUAL_P",
        "COUNTERFACTUAL_MARGIN", "RELADA", "REF_ADAPTER", "PIXANCHOR",
        "W_PIX", "ABS_AUG", "ABS_AUG_SCALE", "VAL_N", "VAL_EVERY",
        "RUN_NAME", "STOP_AT", "SCRATCH", "LORA_R", "BATCH",
        "EFFECTIVE_BATCH", "LR", "MOTION_POW", "SAMPLER_SEED", "SAVE_EVERY",
    ):
        env.pop(key, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=Path("data/train"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("pretrained/checkpoints"))
    parser.add_argument("--work-root", type=Path, default=Path("work"))
    parser.add_argument("--gpu", default="0", help="one visible GPU index")
    parser.add_argument("--latent-batch", type=int, default=8)
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="require an already complete work cache instead of preparing it automatically",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run one real Stage-1 forward/backward/update step and stop",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "stage1", "idm", "stage2", "soup"),
        default="all",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    train_root = absolute(args.train_root)
    checkpoint_root = absolute(args.checkpoint_root)
    work_root = absolute(args.work_root)
    runs_root = work_root / ("smoke_runs" if args.smoke_test else "runs")
    logs_root = work_root / ("smoke_logs" if args.smoke_test else "logs")
    index_path = work_root / "episode_index.parquet"
    latents = work_root / "latents_stride4"
    val_latents = work_root / "val_latents"
    manifest_s8 = work_root / "latent_manifest_s8.parquet"
    manifest_s4 = work_root / "latent_manifest_s4.parquet"
    manifest_val = work_root / "latent_manifest_val.parquet"
    lidm_path = runs_root / "latent_idm2" / "lidm2_best.pt"

    require(train_root / "so100_action_statistics.json", "train action statistics")
    require(checkpoint_root / "tokenizer.pth", "Cosmos tokenizer")
    require(
        checkpoint_root / "robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt",
        "Cosmos action-conditioned base model",
    )
    require(
        checkpoint_root / "robot/action-cond/cr1_empty_string_text_embeddings.pt",
        "Cosmos empty-text embedding",
    )
    cache_inputs = (
        index_path, latents, val_latents, manifest_s8, manifest_s4, manifest_val,
    )
    if not all(path.exists() for path in cache_inputs):
        if args.check_only:
            missing = [str(path) for path in cache_inputs if not path.exists()]
            print("raw training inputs and base checkpoints OK")
            print(f"preprocessed cache not present yet ({len(missing)} paths); default --stage all will create it")
            return
        if args.skip_preprocess or args.smoke_test or args.stage != "all":
            missing = [str(path) for path in cache_inputs if not path.exists()]
            raise FileNotFoundError(
                "preprocessed training inputs are missing; run the default --stage all command "
                f"or preprocess.py first: {missing}"
            )
        preprocess_command = [
            sys.executable,
            str(ROOT / "preprocess.py"),
            "--train-root", str(train_root),
            "--checkpoint-root", str(checkpoint_root),
            "--work-root", str(work_root),
            "--gpu", str(args.gpu),
            "--latent-batch", str(args.latent_batch),
        ]
        print("preprocessed cache missing; running preprocessing first", flush=True)
        print("+", " ".join(preprocess_command), flush=True)
        subprocess.run(preprocess_command, cwd=ROOT, check=True)
    if args.check_only:
        print("training inputs and preprocessed cache OK")
        return

    runs_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    clean_recipe_env(env)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "SO100_TRAIN_ROOT": str(train_root),
            "SO100_INDEX_PATH": str(index_path),
            "COSMOS_CHECKPOINT_DIR": str(checkpoint_root),
            "COSMOS_LATENT_DIR": str(latents),
            "COSMOS_VAL_LATENT_DIR": str(val_latents),
            "RUNS_ROOT": str(runs_root),
            "LIDM_OUT": str(lidm_path),
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "LORA_R": "32",
            "BATCH": "8",
            "EFFECTIVE_BATCH": "8",
            "LR": "5e-5",
            "MOTION_POW": "1",
            "SAMPLER_SEED": "1234",
            "SAVE_EVERY": "625",
            "AUX_WARMUP": "0",
            "VAL_N": "96",
            "VAL_EVERY": "625",
        }
    )

    requested = "stage1" if args.smoke_test else args.stage
    if requested in {"all", "stage1"}:
        stage_env = env.copy()
        stage_env.update(
            {
                "COSMOS_LATENT_MANIFEST": str(manifest_s8),
                "RUN_NAME": "smoke_stage1" if args.smoke_test else "v1_lora",
                "STOP_AT": "1" if args.smoke_test else "4375",
                "SCRATCH": "1",
            }
        )
        if args.smoke_test:
            smoke_val = work_root / "smoke_empty_val"
            smoke_val.mkdir(parents=True, exist_ok=True)
            stage_env.update(
                {
                    "COSMOS_VAL_LATENT_DIR": str(smoke_val),
                    "BATCH": "1",
                    "EFFECTIVE_BATCH": "1",
                    "SAVE_EVERY": "1",
                    "VAL_N": "0",
                }
            )
        run(TRAIN_CODE / "train_lora.py", stage_env, logs_root / "stage1.log")

    if requested in {"all", "idm"}:
        idm_env = env.copy()
        idm_env["COSMOS_LATENT_MANIFEST"] = str(manifest_s8)
        run(TRAIN_CODE / "train_latent_idm2.py", idm_env, logs_root / "latent_idm2.log")

    if requested in {"all", "stage2"}:
        stage1 = runs_root / "v1_lora" / "latest.pt"
        require(stage1, "Stage-1 checkpoint; run --stage stage1 first")
        require(lidm_path, "latent IDM checkpoint; run --stage idm first")
        stage_env = env.copy()
        stage_env.update(
            {
                "COSMOS_LATENT_MANIFEST": str(manifest_s4),
                "RUN_NAME": "v3_aux",
                "STOP_AT": "25000",
                "RESUME_FROM": str(stage1),
                "AUX": "1",
                "AUX_JUDGE": "v2",
                "W_AUX": "0.3",
                "P_AUX": "0.3",
            }
        )
        run(TRAIN_CODE / "train_lora.py", stage_env, logs_root / "stage2.log")

    if requested in {"all", "soup"}:
        soup_dir = runs_root / "v3_aux"
        for step in (23125, 23750, 24375, 25000):
            require(soup_dir / f"step_{step:06d}.pt", f"Stage-2 checkpoint {step}")
        soup_env = env.copy()
        soup_env["SOUP_DIR"] = str(soup_dir)
        steps = ("23125", "23750", "24375", "25000")
        run(TRAIN_CODE / "make_soup.py", soup_env, logs_root / "soup.log", *steps)
        source = soup_dir / "soup_23125_23750_24375_25000.pt"
        output = work_root / "v014_stride4_200k_soup_retrained.pt"
        shutil.copy2(source, output)
        print(f"final retrained weight: {output}", flush=True)


if __name__ == "__main__":
    main()
