#!/usr/bin/env python3
"""Run SAM3 for one image/prompt and save mask artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sam3_task_e_masks import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SAM3_ROOT,
    build_predictor,
    render_view_overlay,
    resolve_device,
    run_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Input RGB image.")
    parser.add_argument("--prompt", required=True, help="Text prompt for SAM3.")
    parser.add_argument(
        "--label",
        default=None,
        help="Stable output label. Defaults to a normalized version of --prompt.",
    )
    parser.add_argument(
        "--view-label",
        default="image",
        help="Human-readable view name drawn on the overlay.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--sam3-root", type=Path, default=DEFAULT_SAM3_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--use-fa3", action="store_true")
    parser.add_argument("--use-rope-real", action="store_true")
    return parser.parse_args()


def normalized_label(prompt: str) -> str:
    label = "".join(ch.lower() if ch.isalnum() else "_" for ch in prompt.strip())
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_") or "mask"


def choose_best_mask(masks: np.ndarray, scores: list[float | None]) -> int | None:
    if len(masks) == 0:
        return None
    scored = []
    for idx, mask in enumerate(masks):
        score = scores[idx] if idx < len(scores) and scores[idx] is not None else 0.0
        area = int(mask.sum())
        scored.append((float(score), area, idx))
    scored.sort(reverse=True)
    return scored[0][2]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    label = args.label or normalized_label(args.prompt)

    image = Image.open(args.image).convert("RGB")
    device = resolve_device(args.device)
    print(
        f"[INFO] torch device={device} image={args.image} prompt={args.prompt!r}",
        flush=True,
    )
    predictor = build_predictor(args, device=device)
    result = run_prompt(predictor, image, args.prompt)

    masks = result["masks"]
    boxes = result["boxes"]
    scores = result["scores"]
    best_idx = choose_best_mask(masks, scores)
    if best_idx is None:
        best_mask = np.zeros((image.height, image.width), dtype=np.uint8)
    else:
        best_mask = masks[best_idx].astype(np.uint8) * 255

    mask_path = args.output / f"{label}_mask.png"
    overlay_path = args.output / f"{label}_overlay.png"
    json_path = args.output / f"{label}_detections.json"

    Image.fromarray(best_mask, mode="L").save(mask_path)
    overlay = render_view_overlay(
        image=image,
        masks=masks,
        boxes=boxes,
        scores=scores,
        prompt=args.prompt,
        view_label=args.view_label,
    )
    overlay.save(overlay_path)

    areas = [int(mask.sum()) for mask in masks]
    payload = {
        "image": str(args.image.resolve()),
        "prompt": args.prompt,
        "label": label,
        "view_label": args.view_label,
        "device": device,
        "mask": mask_path.name,
        "overlay": overlay_path.name,
        "mask_count": int(len(masks)),
        "best_index": best_idx,
        "areas_px": areas,
        "boxes_xyxy": boxes.astype(float).tolist(),
        "scores": scores,
        "sam3_root": str(args.sam3_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[INFO] Saved {mask_path}")
    print(f"[INFO] Saved {overlay_path}")
    print(f"[INFO] Saved {json_path}")


if __name__ == "__main__":
    main()
