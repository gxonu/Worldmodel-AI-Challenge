#!/usr/bin/env python3
"""Single-GPU generation, deterministic postprocessing, and official CSV export."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "code" / "inference"
PACKAGED_WEIGHT = Path("weights/v014_stride4_200k_soup.pt")
RETRAINED_WEIGHT = Path("work/v014_stride4_200k_soup_retrained.pt")


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def absolute_executable(path: Path) -> Path:
    """Make an executable path absolute without dereferencing venv symlinks."""
    return path if path.is_absolute() else (ROOT / path).absolute()


def require(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def run(command: list[str], env: dict[str, str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def scoring_runtime_ok(python: Path, kit: Path, env: dict[str, str]) -> bool:
    """Return whether Python can load the unmodified official CSV kit."""
    if not python.is_file():
        return False
    probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import torch, pytorch_lightning, timm, pandas; "
                "from action_extractor import load_so100_action_extractor_checkpoint"
            ),
        ],
        cwd=kit,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return probe.returncode == 0


def select_scoring_python(
    requested: Path | None, kit: Path, env: dict[str, str]
) -> Path:
    """Select the explicit interpreter or the documented local scoring venv."""
    if requested is not None:
        candidate = absolute_executable(requested)
        require(candidate, "scoring Python")
        if not scoring_runtime_ok(candidate, kit, env):
            raise RuntimeError(
                f"scoring Python cannot import the official kit dependencies: {candidate}"
            )
        return candidate

    candidates = (ROOT / ".scoring-venv" / "bin" / "python", Path(sys.executable))
    for candidate in candidates:
        candidate = absolute_executable(candidate)
        if scoring_runtime_ok(candidate, kit, env):
            return candidate
    raise RuntimeError(
        "official CSV-kit dependencies were not found; create .scoring-venv as "
        "described in README.md or pass --scoring-python"
    )


def video_count(path: Path) -> int:
    return len(list(path.glob("sample_*.mp4"))) if path.exists() else 0


def expected_generation_manifest(weight: Path) -> dict[str, object]:
    """Configuration written by generate_eval.py for the fixed recipe."""
    return {
        "ckpt": str(weight),
        "steps": 30,
        "shift": 5.0,
        "seed": 7,
        "guidance": 0.0,
        "acfg": 0.0,
        "anull": 0,
        "acfg_lo": 0.0,
        "acfg_hi": 1.0,
        "acfg_rescale": 0.0,
        "auto": 0.6,
        "navg": 1,
        "lora_scale": 1.0,
        "atok": 0,
        "atok_gate_scale": 1.0,
        "relada": 0,
        "relada_hidden": 512,
        "reference_adapter": 0,
        "reference_blocks": "3,9,15,21",
        "reference_bottleneck": 128,
        "pix_fmt": "libx264-default(yuv420p)",
        "encode": "crf12/preset-slow",
    }


def clean_inference_env(env: dict[str, str]) -> None:
    """Prevent inherited experiment flags from changing the submitted recipe."""
    for key in (
        "ATOK", "ATOK_GATE_SCALE", "PERBLOCK", "DELTA", "ABLATE_REFERENCE",
        "ABLATE_RELATIVE", "INDEPENDENT_WORKERS", "SUBSET", "ANULL",
        "ACFG_LO", "ACFG_HI", "ACFG_RESCALE", "PIX_FMT", "REF_BLOCKS",
        "REF_BOTTLENECK", "RELADA_HIDDEN",
    ):
        env.pop(key, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--submission-kit", type=Path, default=Path("submission_kit"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("pretrained/checkpoints"))
    parser.add_argument("--cosmos-repo", type=Path, default=Path("pretrained/cosmos-predict2.5"))
    parser.add_argument(
        "--weight-mode",
        choices=("packaged", "retrained", "custom"),
        default="packaged",
        help=(
            "packaged: submitted final weight; retrained: output of train.py; "
            "custom: path supplied with --weight"
        ),
    )
    parser.add_argument(
        "--weight",
        type=Path,
        default=None,
        help="checkpoint path; required only for --weight-mode custom",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--generator-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--postprocess-python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--scoring-python",
        type=Path,
        default=None,
        help="official-kit Python; defaults to .scoring-venv/bin/python, then this Python",
    )
    parser.add_argument("--gpu", default="0", help="one visible GPU index")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-csv", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="generate and postprocess one real evaluation sample, without CSV export",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    data_root = absolute(args.data_root)
    eval_root = data_root / "eval"
    train_root = data_root / "train"
    kit = absolute(args.submission_kit)
    checkpoint_root = absolute(args.checkpoint_root)
    cosmos_repo = absolute(args.cosmos_repo)
    if args.weight_mode == "packaged":
        if args.weight is not None:
            parser.error("--weight cannot be combined with --weight-mode packaged")
        weight = absolute(PACKAGED_WEIGHT)
    elif args.weight_mode == "retrained":
        if args.weight is not None:
            parser.error("--weight cannot be combined with --weight-mode retrained")
        weight = absolute(RETRAINED_WEIGHT)
    else:
        if args.weight is None:
            parser.error("--weight is required with --weight-mode custom")
        weight = absolute(args.weight)
    default_output = Path("outputs/smoke" if args.smoke_test else "outputs/final")
    output_root = absolute(args.output_root or default_output)
    raw_videos = output_root / "raw_videos"
    final_videos = output_root / "videos"
    output_csv = output_root / "submission_features.csv"
    generator_python = absolute_executable(args.generator_python)
    postprocess_python = absolute_executable(args.postprocess_python)
    scoring_python: Path | None = None

    require(eval_root / "images", "evaluation images")
    require(eval_root / "actions", "evaluation actions")
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
    require(weight, "fine-tuned weight")
    require(generator_python, "generator Python")
    require(postprocess_python, "postprocess Python")
    run(
        [
            str(generator_python),
            str(ENGINE / "check_weight.py"),
            str(weight),
        ],
        os.environ.copy(),
        ENGINE,
    )
    make_csv = not (args.skip_csv or args.smoke_test)
    if make_csv:
        require(kit / "make_submission_csv.py", "official submission kit")
        require(kit / "checkpoints/action_extractor.ckpt", "official action extractor")
        scoring_python = select_scoring_python(args.scoring_python, kit, os.environ.copy())
        print(f"scoring Python: {scoring_python}", flush=True)
    if args.check_only:
        print(f"inference inputs and {args.weight_mode} weight OK")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    clean_inference_env(env)
    python_path = [str(cosmos_repo)] if cosmos_repo.exists() else []
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONPATH": os.pathsep.join(python_path),
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": "1",
            "COSMOS_CHECKPOINT_DIR": str(checkpoint_root),
            "SO100_EVAL_ROOT": str(eval_root),
            "RUN_CKPT": str(weight),
            "OUTDIR": str(raw_videos),
            "RELADA": "0",
            "REF_ADAPTER": "0",
            "STEPS": "30",
            "SHIFT": "5",
            "SEED": "7",
            "GUIDANCE": "0",
            "ACFG": "0",
            "ANULL": "0",
            "ACFG_LO": "0",
            "ACFG_HI": "1",
            "ACFG_RESCALE": "0",
            "AUTO_G": "0.6",
            "NAVG": "1",
            "LORA_SCALE": "1",
            "ATOK": "0",
            "ATOK_GATE_SCALE": "1",
            "PERBLOCK": "0",
            "DELTA": "0",
            "RELADA_HIDDEN": "512",
            "REF_BLOCKS": "3,9,15,21",
            "REF_BOTTLENECK": "128",
            "PIX_FMT": "",
            "RESUME": "0",
            "OVERWRITE": "1" if args.overwrite else "0",
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
        }
    )
    if args.smoke_test:
        env["SUBSET"] = "216"

    started = time.monotonic()
    expected_count = 1 if args.smoke_test else 216
    current_raw = video_count(raw_videos)
    if current_raw == expected_count and not args.overwrite:
        manifest_path = raw_videos / "manifest.json"
        require(manifest_path, "generation manifest")
        actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest = expected_generation_manifest(weight)
        if actual_manifest != expected_manifest:
            raise RuntimeError(
                "complete raw output uses a different generation configuration; "
                "use a fresh --output-root"
            )
        print(f"reuse complete raw output: {current_raw}/{expected_count}", flush=True)
    else:
        run([str(generator_python), str(ENGINE / "generate_eval.py")], env, ENGINE)
    if video_count(raw_videos) != expected_count:
        raise RuntimeError(
            f"generation incomplete: {video_count(raw_videos)}/{expected_count}"
        )

    current_final = video_count(final_videos)
    if current_final == expected_count and not args.overwrite:
        require(final_videos / "routing.csv", "postprocess routing record")
        print(f"reuse complete postprocess output: {current_final}/{expected_count}", flush=True)
    elif current_final and not args.overwrite:
        raise RuntimeError(
            f"postprocess output is not empty ({current_final}/{expected_count}); "
            "use a fresh --output-root or rerun with --overwrite"
        )
    else:
        command = [
            str(postprocess_python),
            str(ENGINE / "postprocess.py"),
            "--prediction-root", str(raw_videos),
            "--image-root", str(eval_root / "images"),
            "--action-root", str(eval_root / "actions"),
            "--output-root", str(final_videos),
            "--expected-count", str(expected_count),
        ]
        if args.overwrite:
            command.append("--overwrite")
        run(command, env, ENGINE)
    if video_count(final_videos) != expected_count:
        raise RuntimeError(
            f"postprocessing incomplete: {video_count(final_videos)}/{expected_count}"
        )

    if make_csv:
        assert scoring_python is not None
        # Do not leak Cosmos PYTHONPATH/thread settings into the unmodified kit.
        # They can change its dependency resolution and floating-point reductions.
        score_env = os.environ.copy()
        score_env.pop("PYTHONPATH", None)
        score_env.pop("OMP_NUM_THREADS", None)
        score_env.pop("MKL_NUM_THREADS", None)
        score_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        score_env["MPLCONFIGDIR"] = str(output_root / ".mplconfig")
        command = [
            str(scoring_python),
            str(kit / "make_submission_csv.py"),
            "--prediction-root", str(final_videos),
            "--challenge-root", str(eval_root),
            "--output-csv", str(output_csv),
            "--action-stats-path", str(train_root / "so100_action_statistics.json"),
            "--action-extractor-ckpt", str(kit / "checkpoints/action_extractor.ckpt"),
            "--feature-batch-size", str(args.feature_batch_size),
        ]
        run(command, score_env, kit)
        run([str(scoring_python), str(ENGINE / "verify_submission.py"), str(output_csv)], score_env)
        print(f"CSV: {output_csv}", flush=True)

    print(f"total elapsed: {(time.monotonic() - started) / 60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
