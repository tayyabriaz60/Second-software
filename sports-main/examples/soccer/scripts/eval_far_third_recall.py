#!/usr/bin/env python3
"""Compare detectors on held-out val images: small-box and far-third recall.

Headline mAP hides the failure mode: far-touchline players (top third of the
4096x1152 panoramic) at 24-38px height. Labels only contain ~2% of boxes under
40px because pre-annotation missed them too — treat val GT counts as a lower
bound and confirm gains on Pass 1 fragment span.

Runs the same val images through:
  - stock RF-DETR (pipeline default, rfdetr_onnx.py)
  - fine-tuned YOLO (.pt or .onnx)

Reports recall @ IoU 0.5 for:
  - all GT boxes
  - far third   : GT centre y < H/3
  - small       : GT height < 40 px at full resolution
  - far + small : both (target population, may be very few in val)

Usage (from sports-main/examples/soccer):
  python scripts/eval_far_third_recall.py \\
    --data /workspace/ds_yolo/data.yaml \\
    --weights runs/panoramic/yolo32x9/weights/best.onnx \\
    --imgsz 576 2048
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# rfdetr_onnx lives in the soccer example root
_SOCCER_ROOT = Path(__file__).resolve().parents[1]
if str(_SOCCER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOCCER_ROOT))


@dataclass
class BucketStats:
    gt: int = 0
    matched: int = 0

    @property
    def recall(self) -> float:
        return self.matched / self.gt if self.gt else float('nan')


@dataclass
class EvalStats:
    all: BucketStats = field(default_factory=BucketStats)
    far_third: BucketStats = field(default_factory=BucketStats)
    small: BucketStats = field(default_factory=BucketStats)
    far_small: BucketStats = field(default_factory=BucketStats)


def _load_gt(label_path: Path, w: int, h: int) -> np.ndarray:
    if not label_path.is_file():
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    for line in label_path.read_text(encoding='utf-8').strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        rows.append([
            (cx - bw / 2) * w, (cy - bh / 2) * h,
            (cx + bw / 2) * w, (cy + bh / 2) * h,
        ])
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 4))


def _box_h(xyxy: np.ndarray) -> float:
    return float(xyxy[3] - xyxy[1])


def _box_cy(xyxy: np.ndarray) -> float:
    return float((xyxy[1] + xyxy[3]) / 2)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-6)


def _update(stats: EvalStats, gt: np.ndarray, pred: np.ndarray,
            frame_h: int, small_h: float, far_frac: float,
            iou_thr: float = 0.5) -> None:
    far_y = frame_h * far_frac
    for i in range(len(gt)):
        box = gt[i]
        h = _box_h(box)
        cy = _box_cy(box)
        is_far = cy < far_y
        is_small = h < small_h
        stats.all.gt += 1
        if is_far:
            stats.far_third.gt += 1
        if is_small:
            stats.small.gt += 1
        if is_far and is_small:
            stats.far_small.gt += 1

    if len(gt) == 0:
        return

    ious = _iou_matrix(gt, pred) if len(pred) else np.zeros((len(gt), 0))
    used: set[int] = set()
    for gi in range(len(gt)):
        if len(pred) == 0:
            break
        best_pi = int(np.argmax(ious[gi]))
        if ious[gi, best_pi] < iou_thr or best_pi in used:
            continue
        used.add(best_pi)
        box = gt[gi]
        h = _box_h(box)
        cy = _box_cy(box)
        is_far = cy < far_y
        is_small = h < small_h
        stats.all.matched += 1
        if is_far:
            stats.far_third.matched += 1
        if is_small:
            stats.small.matched += 1
        if is_far and is_small:
            stats.far_small.matched += 1


def _person_yolo_ids(model: YOLO) -> set[int]:
    names = {str(v).strip().lower(): int(k) for k, v in model.names.items()}
    return {names[n] for n in ('goalkeeper', 'player', 'referee') if n in names}


# Pipeline class IDs after rfdetr_onnx._CLASS_MAP
_RFDETR_PERSON = {1, 2, 3}  # goalkeeper, player, referee


def _run_yolo(model_path: Path, val_img: Path, val_lbl: Path,
              conf: float, imgsz: list[int], rect: bool) -> EvalStats:
    model = YOLO(str(model_path))
    person_ids = _person_yolo_ids(model)
    stats = EvalStats()
    for img_path in sorted(val_img.glob('*')):
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gt = _load_gt(val_lbl / f'{img_path.stem}.txt', w, h)
        res = model.predict(frame, conf=conf, imgsz=imgsz, rect=rect, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            pred = np.zeros((0, 4))
        else:
            cls = res.boxes.cls.cpu().numpy().astype(int)
            mask = np.isin(cls, list(person_ids))
            pred = res.boxes.xyxy.cpu().numpy()[mask]
        _update(stats, gt, pred, h, small_h=40.0, far_frac=1.0 / 3.0)
    return stats


def _run_rfdetr(val_img: Path, val_lbl: Path, conf: float) -> EvalStats:
    import rfdetr_onnx

    stats = EvalStats()
    for img_path in sorted(val_img.glob('*')):
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gt = _load_gt(val_lbl / f'{img_path.stem}.txt', w, h)
        dets = rfdetr_onnx.detect(frame, conf=conf)
        if len(dets) == 0 or dets.class_id is None:
            pred = np.zeros((0, 4))
        else:
            mask = np.isin(dets.class_id, list(_RFDETR_PERSON))
            pred = dets.xyxy[mask].astype(np.float32)
        _update(stats, gt, pred, h, small_h=40.0, far_frac=1.0 / 3.0)
    return stats


def _print_gt(stats: EvalStats) -> None:
    print('Val GT counts (held-out labels — far-touchline players under-counted):')
    print(f'  all       : {stats.all.gt}')
    print(f'  far third : {stats.far_third.gt}')
    print(f'  small     : {stats.small.gt}')
    print(f'  far+small : {stats.far_small.gt}\n')


def _print_row(name: str, stats: EvalStats) -> None:
    def fmt(b: BucketStats) -> str:
        if b.gt == 0:
            return 'n/a (0 gt)'
        return f'{b.recall:.1%} ({b.matched}/{b.gt})'

    print(f'{name:14}  all={fmt(stats.all):>20}  '
          f'far_third={fmt(stats.far_third):>20}  '
          f'small={fmt(stats.small):>20}  '
          f'far+small={fmt(stats.far_small):>20}')


def _gt_only(val_img: Path, val_lbl: Path) -> EvalStats:
    stats = EvalStats()
    for img_path in sorted(val_img.glob('*')):
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gt = _load_gt(val_lbl / f'{img_path.stem}.txt', w, h)
        _update(stats, gt, np.zeros((0, 4)), h, small_h=40.0, far_frac=1.0 / 3.0)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, required=True,
                    help='data.yaml from prepare_panoramic_dataset.py')
    ap.add_argument('--weights', type=Path, required=True,
                    help='Fine-tuned YOLO .pt or .onnx')
    ap.add_argument('--yolo-conf', type=float, default=0.25,
                    help='Confidence for fine-tuned YOLO')
    ap.add_argument('--rfdetr-conf', type=float, default=0.20,
                    help='Confidence for stock RF-DETR (pipeline weak tier)')
    ap.add_argument('--imgsz', type=int, nargs=2, default=[576, 2048],
                    metavar=('H', 'W'), help='Must match training / ONNX export')
    ap.add_argument('--no-rect', action='store_true',
                    help='Disable rect inference for YOLO (not recommended)')
    ap.add_argument('--skip-rfdetr', action='store_true',
                    help='Only evaluate fine-tuned YOLO')
    args = ap.parse_args()

    root = args.data.parent
    val_img = root / 'val' / 'images'
    val_lbl = root / 'val' / 'labels'
    if not val_img.is_dir():
        raise SystemExit(f'Missing {val_img}')

    imgsz = list(args.imgsz)
    rect = not args.no_rect
    print(f'Val: {val_img}')
    print(f'YOLO conf={args.yolo_conf}  imgsz={imgsz}  rect={rect}')
    print(f'RF-DETR conf={args.rfdetr_conf}  input=576x576 stretch (stock)\n')
    print('far_third = GT centre y < H/3   small = GT height < 40 px\n')

    gt_stats = _gt_only(val_img, val_lbl)
    _print_gt(gt_stats)

    if not args.skip_rfdetr:
        print('--- STOCK RF-DETR ---')
        _print_row('RF-DETR', _run_rfdetr(val_img, val_lbl, args.rfdetr_conf))
        print()

    print('--- FINE-TUNED YOLO ---')
    _print_row('YOLO', _run_yolo(args.weights, val_img, val_lbl,
                                 args.yolo_conf, imgsz, rect))

    print('\nHow to read this:')
    print('  • Primary: small recall and far_third recall vs RF-DETR on the same val images.')
    print('  • far+small may show n/a if val has <5 GT — labels miss most far players.')
    print('  • Sweep conf if needed: --yolo-conf 0.15  --rfdetr-conf 0.15')
    print('  • Confirm on video: Pass 1 fragment median span on the 10-min window.')


if __name__ == '__main__':
    main()
