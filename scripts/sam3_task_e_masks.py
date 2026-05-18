#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


DEFAULT_SAM3_ROOT = Path("/home/steven/Projects/Projects/sam3")
DEFAULT_CHECKPOINT = DEFAULT_SAM3_ROOT / "checkpoints/sam3.1/sam3.1_multiplex.pt"

VIEW_SPECS = [
    ("video_rgb", "eye-to-hand / video_cam", "video_rgb.png"),
    ("ee_rgb", "eye-in-hand / ee_camera", "ee_rgb.png"),
]

PROMPT_SPECS = [
    ("banana", "banana"),
    ("mustard_bottle", "mustard bottle"),
    ("box_object", "yellow and white box"),
    ("purple_basket", "purple basket"),
]

MASK_COLORS = [
    (59, 130, 246),
    (245, 158, 11),
    (16, 185, 129),
    (239, 68, 68),
    (168, 85, 247),
    (236, 72, 153),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 on Task E captured video_rgb and ee_rgb frames."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Capture directory containing video_rgb.png and ee_rgb.png.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to --input.",
    )
    parser.add_argument(
        "--sam3-root",
        type=Path,
        default=DEFAULT_SAM3_ROOT,
        help="Local SAM3 repository path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="SAM3.1 multiplex checkpoint path.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable SAM3 model compilation.",
    )
    parser.add_argument(
        "--use-fa3",
        action="store_true",
        help="Enable FlashAttention 3 explicitly.",
    )
    parser.add_argument(
        "--use-rope-real",
        action="store_true",
        help="Enable real-valued RoPE in the SAM3.1 path.",
    )
    return parser.parse_args()


def resolve_input_dir(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    if path.name == "latest":
        latest_txt = path.with_name("latest.txt")
        if latest_txt.exists():
            return Path(latest_txt.read_text(encoding="utf-8").strip()).resolve()
    raise FileNotFoundError(
        f"Input directory does not exist: {path}\n"
        "Run the capture step first and confirm it creates video_rgb.png and ee_rgb.png:\n"
        "  ATEC_SAM3_CAPTURE=1 python scripts/play_atec_task.py "
        "--task ATEC-TaskE-Piper --enable_cameras --headless"
    )


def resolve_device(device: str) -> str:
    cuda_available = torch.cuda.is_available()
    if device == "cuda" and not cuda_available:
        raise RuntimeError(
            "SAM3 was requested with --device cuda, but torch.cuda.is_available() "
            "is False in this process."
        )
    if device != "auto":
        return device
    return "cuda" if cuda_available else "cpu"


def build_predictor(args: argparse.Namespace, device: str):
    sam3_root = args.sam3_root.resolve()
    if not sam3_root.exists():
        raise FileNotFoundError(f"SAM3 root does not exist: {sam3_root}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"SAM3 checkpoint does not exist: {args.checkpoint}")
    if str(sam3_root) not in sys.path:
        sys.path.insert(0, str(sam3_root))

    from sam3.model_builder import build_sam3_predictor

    return build_sam3_predictor(
        checkpoint_path=str(args.checkpoint),
        version="sam3.1",
        compile=args.compile,
        async_loading_frames=False,
        device=device,
        use_fa3=args.use_fa3,
        use_rope_real=args.use_rope_real,
        enable_bf16_autocast=False,
    )


def normalize_array(payload) -> np.ndarray:
    if payload is None:
        return np.asarray([])
    if isinstance(payload, torch.Tensor):
        return payload.detach().cpu().numpy()
    return np.asarray(payload)


def normalize_masks(mask_payload, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    masks = normalize_array(mask_payload)
    if masks.size == 0:
        return np.zeros((0, height, width), dtype=bool)

    if masks.ndim == 2:
        masks = masks[None, :, :]
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, :, :]
    elif masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[:, :, :, 0]
    if masks.ndim != 3:
        raise ValueError(f"Unsupported mask shape: {masks.shape}")

    masks = masks.astype(bool)
    if masks.shape[-2:] == (height, width):
        return masks

    resized = []
    for mask in masks:
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)
        resized.append(np.asarray(mask_image) > 0)
    return np.stack(resized, axis=0) if resized else np.zeros((0, height, width), dtype=bool)


def boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    boxes = np.zeros((len(masks), 4), dtype=np.float32)
    for idx, mask in enumerate(masks):
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            continue
        boxes[idx] = [xs.min(), ys.min(), xs.max(), ys.max()]
    return boxes


def normalize_boxes(box_payload, box_format: str, masks: np.ndarray) -> np.ndarray:
    boxes = normalize_array(box_payload)
    if boxes.size == 0:
        return boxes_from_masks(masks)
    boxes = boxes.reshape(-1, 4).astype(np.float32)
    if box_format == "xywh":
        x0 = boxes[:, 0]
        y0 = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        boxes = np.stack([x0, y0, x0 + w, y0 + h], axis=-1)
    elif box_format != "xyxy":
        raise ValueError(f"Unsupported box format: {box_format}")
    if len(boxes) != len(masks):
        return boxes_from_masks(masks)
    return boxes


def normalize_scores(score_payload, mask_count: int) -> list[float | None]:
    scores = normalize_array(score_payload)
    if scores.size == 0:
        return [None for _ in range(mask_count)]
    scores = scores.reshape(-1).astype(np.float32)
    result = [float(score) for score in scores[:mask_count]]
    result.extend([None for _ in range(mask_count - len(result))])
    return result


def run_prompt(predictor, image: Image.Image, prompt: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="sam3_task_e_") as tmp_dir:
        frame_path = Path(tmp_dir) / "00000.jpg"
        image.convert("RGB").save(frame_path, format="JPEG", quality=95)
        session_id = predictor.handle_request(
            {"type": "start_session", "resource_path": str(Path(tmp_dir))}
        )["session_id"]
        try:
            response = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt,
                }
            )
        finally:
            predictor.handle_request(
                {
                    "type": "close_session",
                    "session_id": session_id,
                    "run_gc_collect": False,
                }
            )

    outputs = response["outputs"]
    masks = normalize_masks(outputs.get("out_binary_masks", []), image.size)
    boxes = normalize_boxes(outputs.get("out_boxes_xywh"), "xywh", masks)
    scores = normalize_scores(outputs.get("out_probs"), len(masks))
    return {"masks": masks, "boxes": boxes, "scores": scores}


def draw_banner(
    image: Image.Image,
    prompt: str,
    view_label: str,
    mask_count: int,
) -> Image.Image:
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    lines = [f"Prompt: {prompt}", f"View: {view_label}", f"Masks: {mask_count}"]
    if mask_count == 0:
        lines.append("No mask")
    text = "\n".join(lines)
    left, top = 12, 12
    bbox = draw.multiline_textbbox((0, 0), text, spacing=4)
    right = left + (bbox[2] - bbox[0]) + 18
    bottom = top + (bbox[3] - bbox[1]) + 16
    draw.rectangle((left, top, right, bottom), fill=(0, 0, 0, 170))
    draw.multiline_text((left + 9, top + 8), text, fill=(255, 255, 255, 255), spacing=4)
    return overlay.convert("RGB")


def render_view_overlay(
    image: Image.Image,
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: list[float | None],
    prompt: str,
    view_label: str,
) -> Image.Image:
    base = image.convert("RGBA")
    width, height = image.size

    for idx, mask in enumerate(masks):
        color = MASK_COLORS[idx % len(MASK_COLORS)]
        layer = np.zeros((height, width, 4), dtype=np.uint8)
        layer[mask] = (*color, 100)
        base = Image.alpha_composite(base, Image.fromarray(layer, mode="RGBA"))

        draw = ImageDraw.Draw(base)
        x0, y0, x1, y1 = [int(round(v)) for v in boxes[idx]]
        x0 = max(0, min(width - 1, x0))
        x1 = max(0, min(width - 1, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(0, min(height - 1, y1))
        draw.rectangle((x0, y0, x1, y1), outline=color + (255,), width=3)

        score = scores[idx] if idx < len(scores) else None
        if score is not None:
            label = f"{score:.2f}"
            bbox = draw.textbbox((0, 0), label)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_top = max(0, y0 - text_h - 8)
            draw.rectangle(
                (x0, text_top, x0 + text_w + 10, text_top + text_h + 6),
                fill=(0, 0, 0, 170),
            )
            draw.text((x0 + 5, text_top + 3), label, fill=(255, 255, 255, 255))

    return draw_banner(base.convert("RGB"), prompt, view_label, len(masks))


def combine_views(left: Image.Image, right: Image.Image) -> Image.Image:
    gap = 16
    width = left.width + right.width + gap
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (width, height), (17, 24, 39))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas


def save_overlays(input_dir: Path, output_dir: Path, predictor) -> dict:
    images = {}
    for view_name, view_label, file_name in VIEW_SPECS:
        image_path = input_dir / file_name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing captured image: {image_path}")
        images[view_name] = {
            "label": view_label,
            "image": Image.open(image_path).convert("RGB"),
            "file": file_name,
        }

    detections = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "prompts": {},
    }
    for label, prompt in PROMPT_SPECS:
        prompt_record = {"prompt": prompt, "overlay": f"{label}_overlay.png", "views": {}}
        rendered_views = []
        for view_name, view_data in images.items():
            result = run_prompt(predictor, view_data["image"], prompt)
            rendered = render_view_overlay(
                image=view_data["image"],
                masks=result["masks"],
                boxes=result["boxes"],
                scores=result["scores"],
                prompt=prompt,
                view_label=view_data["label"],
            )
            rendered_views.append(rendered)
            prompt_record["views"][view_name] = {
                "image": view_data["file"],
                "mask_count": int(len(result["masks"])),
                "boxes_xyxy": result["boxes"].astype(float).tolist(),
                "scores": result["scores"],
            }

        combined = combine_views(rendered_views[0], rendered_views[1])
        combined.save(output_dir / prompt_record["overlay"])
        detections["prompts"][label] = prompt_record
        print(f"[INFO] Saved {output_dir / prompt_record['overlay']}")
    return detections


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(args.input)
    output_dir = (args.output or input_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(
        f"[INFO] torch.cuda.is_available()={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count()} using device={device}"
    )

    predictor = build_predictor(args, device=device)
    detections = save_overlays(input_dir, output_dir, predictor)
    detections["checkpoint"] = str(args.checkpoint.resolve())
    detections["sam3_root"] = str(args.sam3_root.resolve())
    detections["device"] = device

    detections_path = output_dir / "detections.json"
    detections_path.write_text(json.dumps(detections, indent=2), encoding="utf-8")
    print(f"[INFO] Saved {detections_path}")


if __name__ == "__main__":
    main()
