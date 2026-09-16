# Panoramic YOLO training (400 frames, 4096×1152)

## Why not square 640

4096×1152 into 640×640 squashes horizontal 6.4× vs vertical 1.8×. A 30px
far-touchline player becomes ~5px tall. Train at stride-clean **32:9** instead:
default **2048×576** (2× uniform downsample, 30px → ~15px).

## Label caveat

Only ~2% of boxes are under 40px height — pre-annotation missed most
far-touchline players. Val recall on `small` / `far+small` is a lower bound;
Pass 1 **fragment median span** is the real-world check.

## Steps (RunPod)

```bash
cd /workspace/Second-software/sports-main/examples/soccer

# 1. Split + GT stats (stratified val so rare small/far tags stay balanced)
python scripts/prepare_panoramic_dataset.py \
  --source /workspace/ds_full \
  --dest /workspace/ds_yolo

# 2. Train + ONNX export (reduce --batch if OOM)
python scripts/train_panoramic_yolo.py \
  --data /workspace/ds_yolo/data.yaml \
  --base-model yolov8m.pt \
  --imgsz-preset 2048x576 \
  --batch 4 \
  --epochs 150

# 3. Compare vs stock RF-DETR on the same held-out val images
python scripts/eval_far_third_recall.py \
  --data /workspace/ds_yolo/data.yaml \
  --weights runs/panoramic/yolo32x9/weights/best.onnx \
  --imgsz 576 2048

# Optional conf sweep (same threshold on both):
python scripts/eval_far_third_recall.py \
  --data /workspace/ds_yolo/data.yaml \
  --weights runs/panoramic/yolo32x9/weights/best.onnx \
  --imgsz 576 2048 \
  --yolo-conf 0.15 --rfdetr-conf 0.15

# 4. Pipeline smoke test (match training imgsz — do NOT use square 1920)
export PYTHONPATH=../../
python main.py \
  --source_video_path "data/base_datasets/Sample Videos/Stationary_Camera_14_08.mp4" \
  --target_video_path /workspace/out_finetune.mp4 \
  --detector yolo \
  --model runs/panoramic/yolo32x9/weights/best.onnx \
  --imgsz 576 2048 \
  --mode PLAYER_TRACKING \
  --start_frame 26289 --max_frames 33114 \
  --no_render --run_label yolo_finetune_10min
```

## Metrics that matter

| Metric | Where | Target |
|--------|-------|--------|
| **small recall** | `eval_far_third_recall.py` | Beat RF-DETR on val |
| **far_third recall** | same | Beat RF-DETR on val |
| **fragment median span** | Pass 1 log, 10-min window | Above ~3.6s |
| **identity count** | identity_map summary | Trend down (long term) |

Ignore headline mAP if `small` / `far_third` recall does not move.

## Augmentation (fixed camera)

Training script sets: `degrees=0`, `shear=0`, `flipud=0`, `fliplr=0.5`,
`scale=0.10`, `translate=0.03`, `multi_scale=False`.

## Class order

Default `data.yaml` names: `0=goalkeeper, 1=player, 2=referee`. If Roboflow
labels use a different order, pass `--class-names` to `prepare_panoramic_dataset.py`.

## Mac / offline

Copy `best.onnx` to the Mac. Ultralytics loads it directly. Pipeline:

```bash
--detector yolo --model path/to/best.onnx --imgsz 576 2048
```
