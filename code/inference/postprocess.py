"""Apply the submitted action-routed background-anchor postprocessing.

The route is determined only from each sample's supplied action condition.  It
does not inspect sample IDs, submission-kit features, future frames, or another
sample.  The fixed submitted threshold and its provenance warning are recorded
in README.md.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import av
import cv2
import numpy as np


SHOULDER_LIFT_INDEX = 1
DEFAULT_DOMAIN_THRESHOLD_DEG = 45.0


def read_video(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        frames = [
            frame.to_ndarray(format="rgb24")
            for frame in container.decode(container.streams.video[0])
        ]
    if len(frames) != 16:
        raise ValueError(f"expected 16 frames, got {len(frames)}: {path}")
    return np.stack(frames)


def letterbox(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if image.shape[:2] == (target_h, target_w):
        return image
    height, width = image.shape[:2]
    scale = min(target_h / height, target_w / width)
    resized = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - resized.shape[0]) // 2
    left = (target_w - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def background_anchor(
    prediction: np.ndarray,
    image: np.ndarray,
    threshold: float = 0.060,
    dilation: int = 15,
    sigma: float = 3.0,
) -> np.ndarray:
    """Restore the supplied first image outside the predicted motion support."""
    image = letterbox(image, prediction.shape[1], prediction.shape[2])
    base = image.astype(np.float32) / 255.0
    video = prediction.astype(np.float32) / 255.0
    difference = np.max(np.abs(video - base[None]), axis=-1)
    open_kernel = np.ones((3, 3), np.uint8)
    dilate_kernel = np.ones((dilation, dilation), np.uint8)
    masks = []
    for frame_difference in difference:
        mask = (frame_difference >= threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.dilate(mask, dilate_kernel)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
        masks.append(np.clip(mask, 0.0, 1.0))
    masks = np.stack(masks)[..., None]
    output = base[None] * (1.0 - masks) + video * masks
    output[0] = base
    return np.rint(output * 255.0).clip(0, 255).astype(np.uint8)


def write_video(path: Path, frames: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=6)
    stream.width = int(frames.shape[2])
    stream.height = int(frames.shape[1])
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "12", "preset": "medium"}
    try:
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def copy_unmodified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def use_anchor(action: np.ndarray, domain_threshold: float) -> tuple[bool, float]:
    if action.shape != (16, 6):
        raise ValueError(f"expected action shape (16, 6), got {action.shape}")
    shoulder_lift_mean = float(action[:, SHOULDER_LIFT_INDEX].mean())
    return shoulder_lift_mean > domain_threshold, shoulder_lift_mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--action-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--domain-threshold",
        type=float,
        default=DEFAULT_DOMAIN_THRESHOLD_DEG,
        help="anchor when mean supplied shoulder_lift is above this value",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-count", type=int, default=216)
    args = parser.parse_args()

    videos = sorted(args.prediction_root.glob("sample_*.mp4"))
    if len(videos) != args.expected_count:
        raise RuntimeError(
            f"expected {args.expected_count} input videos, found {len(videos)}"
        )

    anchored = 0
    untouched = 0
    routed = []
    for index, video_path in enumerate(videos, 1):
        sample_id = video_path.stem
        output_path = args.output_root / video_path.name
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists (use --overwrite): {output_path}")

        action = np.load(args.action_root / f"{sample_id}.npy").astype(np.float32)
        selected, shoulder_lift_mean = use_anchor(action, args.domain_threshold)
        routed.append((sample_id, shoulder_lift_mean, "anchor" if selected else "raw"))
        if selected:
            image_bgr = cv2.imread(str(args.image_root / f"{sample_id}.png"))
            if image_bgr is None:
                raise FileNotFoundError(args.image_root / f"{sample_id}.png")
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            output = background_anchor(read_video(video_path), image)
            write_video(output_path, output)
            anchored += 1
        else:
            copy_unmodified(video_path, output_path)
            untouched += 1
        print(
            f"[{index:03d}/{args.expected_count}] {sample_id} "
            f"shoulder_lift_mean={shoulder_lift_mean:.6f} "
            f"route={routed[-1][2]}",
            flush=True,
        )

    route_csv = args.output_root / "routing.csv"
    route_csv.write_text(
        "sample_id,shoulder_lift_mean_deg,route\n"
        + "".join(f"{sid},{value:.8f},{route}\n" for sid, value, route in routed),
        encoding="utf-8",
    )
    print(
        f"done: anchored={anchored}, raw={untouched}, threshold={args.domain_threshold:g}, "
        f"routing={route_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
