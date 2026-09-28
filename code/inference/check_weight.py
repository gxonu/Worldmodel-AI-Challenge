#!/usr/bin/env python3
"""Validate that a checkpoint has the state layout required by final inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict", checkpoint.get("state", checkpoint))
    else:
        state = None
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint must be a state dictionary or contain state_dict")
    tensor_state = {key: value for key, value in state.items() if torch.is_tensor(value)}
    if len(tensor_state) < 500:
        raise RuntimeError(f"checkpoint contains too few tensors: {len(tensor_state)}")

    lora_a = [value for key, value in tensor_state.items() if ".lora_A." in key]
    lora_b = [value for key, value in tensor_state.items() if ".lora_B." in key]
    if not lora_a or len(lora_a) != len(lora_b):
        raise RuntimeError(
            f"invalid LoRA tensors: A={len(lora_a)}, B={len(lora_b)}"
        )
    ranks = {int(value.shape[0]) for value in lora_a if value.ndim == 2}
    if ranks != {32}:
        raise RuntimeError(f"expected LoRA rank 32, found {sorted(ranks)}")
    if not any("action_embedder" in key for key in tensor_state):
        raise RuntimeError("checkpoint has no action-conditioning tensors")

    step = checkpoint.get("step", "unknown") if isinstance(checkpoint, dict) else "unknown"
    print(
        f"weight OK: tensors={len(tensor_state)}, LoRA pairs={len(lora_a)}, "
        f"rank=32, step={step}",
        flush=True,
    )


if __name__ == "__main__":
    main()
