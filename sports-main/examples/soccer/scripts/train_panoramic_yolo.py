#!/usr/bin/env python3
"""Train YOLOv8 on 4096x1152 panoramic football footage without square stretch.

4096:1152 is 32:9. Training uses rect=True and a stride-clean (32k, 9k) imgsz so
a 30px far-touchline player is not squashed to ~5px (what 4096->640 square does).

Default imgsz [576, 2048] = 2x uniform downsample from full frame (30px -> ~15px).

After training, exports ONNX for offline Mac / RunPod inference:
  ultralytics YOLO('best.onnx') or main.py --model path/to/best.onnx

Usage:
  python scripts/prepare_panoramic_dataset.py --source /workspace/ds_full
  python scripts/train_panoramic_yolo.py --data /workspace/ds_yolo/data.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Stride-clean 32:9 pairs (multiples of 32 x multiples of 9 -> both divisible by 32)
IMGSZ_PRESETS = {
    '2048x576': (576, 2048),   # 2x down from 4096x1152 — default
    '1024x288': (288, 1024),   # 4x down — lighter GPU
    '1280x360': (360, 1280),   # ~3.2x down (360 = 9*40, 1280 = 32*40)
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=Path('/workspace/ds_yolo/data.yaml'))
    ap.add_argument('--base-model', default='yolov8m.pt',
                    help='COCO checkpoint (yolov8s.pt for faster iteration)')
    ap.add_argument('--imgsz-preset', choices=list(IMGSZ_PRESETS), default='2048x576',
                    help='Aspect-correct input size (height, width)')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch', type=int, default=4,
                    help='Reduce to 2 if OOM at 2048x576')
    ap.add_argument('--device', default='0')
    ap.add_argument('--project', default='runs/panoramic')
    ap.add_argument('--name', default='yolo32x9')
    ap.add_argument('--patience', type=int, default=40)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-export', action='store_true',
                    help='Skip ONNX export after training')
    ap.add_argument('--export-opset', type=int, default=12)
    args = ap.parse_args()

    if not args.data.is_file():
        raise SystemExit(f'data.yaml not found: {args.data}\n'
                         'Run prepare_panoramic_dataset.py first.')

    imgsz = list(IMGSZ_PRESETS[args.imgsz_preset])
    print(f'Training imgsz (h,w) = {imgsz}  preset={args.imgsz_preset}  rect=True')
    print('Do NOT use square 640 — that crushes far-touchline players to ~5px tall.')

    from ultralytics import YOLO

    model = YOLO(args.base_model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=imgsz,
        rect=True,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        # Preserve aspect; avoid multi-scale that reintroduces square batches
        multi_scale=False,
        # Fixed panoramic camera: upright players, symmetric pitch.
        # Keep scale jitter small — down-scaling already-marginal far-field
        # players makes them harder, not easier.
        mosaic=1.0,
        copy_paste=0.1,
        scale=0.10,
        translate=0.03,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        close_mosaic=20,
        save=True,
        plots=True,
        val=True,
    )

    best_pt = Path(results.save_dir) / 'weights' / 'best.pt'
    print(f'\nBest weights: {best_pt}')

    if args.no_export:
        return

    print('Exporting ONNX...')
    best = YOLO(str(best_pt))
    onnx_path = best.export(
        format='onnx',
        imgsz=imgsz,
        simplify=True,
        opset=args.export_opset,
        dynamic=False,
    )
    print(f'ONNX export: {onnx_path}')
    print('\nPipeline smoke test (from sports-main/examples/soccer):')
    print(f'  PYTHONPATH=../../ python main.py --detector yolo '
          f'--model {onnx_path} --mode PLAYER_TRACKING ...')
    print('\nEval vs stock RF-DETR on held-out val:')
    print(f'  python scripts/eval_far_third_recall.py '
          f'--data {args.data} --weights {onnx_path} --imgsz {imgsz[0]} {imgsz[1]}')


if __name__ == '__main__':
    main()
