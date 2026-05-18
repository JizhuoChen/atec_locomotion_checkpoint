# Task E SAM3 Usage

This stage turns camera PNGs into masks, overlays, and detection metadata.

## Environment

Run SAM3 in the isolated env:

```bash
conda run -n sam3_full python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

Expected on this machine: `True`.

## Single-Image Mask Command

Example for the external camera banana mask:

```bash
conda run -n sam3_full python scripts/sam3_single_image_mask.py \
  --image outputs/task_e_camera_capture/seed7_table/video_rgb.png \
  --prompt "curved yellow banana fruit" \
  --label video_curved_yellow_banana \
  --view-label "eye-to-hand / video_cam" \
  --output outputs/task_e_camera_capture/seed7_table/sam3 \
  --device cuda
```

Outputs:

```text
sam3/
  video_curved_yellow_banana_mask.png
  video_curved_yellow_banana_overlay.png
  video_curved_yellow_banana_detections.json
```

The JSON records the prompt, view, boxes, scores, mask count, SAM3 root, checkpoint, and device.

## Prompt Notes

For the banana scene, plain `banana` sometimes selected the mustard bottle because it is also yellow and upright. The more specific prompt below selected the curved banana in the saved test:

```text
curved yellow banana fruit
```

For the yellow box, use:

```text
yellow and white box
```

## Pseudo-Grasp From Mask

After SAM3, create the pseudo-grasp JSON and visualization:

```bash
python scripts/pseudo_grasp_from_task_e_output.py \
  --input outputs/task_e_banana_pipeline/latest \
  --pose-source sam3-mask \
  --sam3-mask outputs/task_e_banana_pipeline/latest/pseudo_grasp/sam3_prompt_test/video_curved_yellow_banana_mask.png \
  --grasp-xy-offset 0.03 -0.006 \
  --quat-source default-topdown
```

Key outputs:

```text
pseudo_grasp/
  pseudo_grasp.json
  pseudo_grasp_video_overlay.png
  pseudo_grasp_cloud_overlay.png
```

`pseudo_grasp.json` is then converted to the unified motion request used by the MoveIt/IK stage.
