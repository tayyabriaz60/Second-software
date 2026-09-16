#!/usr/bin/env python3
"""Split flat YOLO dataset into train/val and write data.yaml.

Source layout (flat):
  ds_full/images/*.jpg
  ds_full/labels/*.txt   (YOLO normalised: class cx cy w h)

Output layout:
  ds_yolo/
    data.yaml
    train/images, train/labels
    val/images, val/labels

Images stay at native 4096x1152. Labels are copied as-is (normalised coords
survive any training resize).

The val split is stratified by image tags (has small box, has far-third box)
so the ~2% of boxes under 40px are not accidentally concentrated in one split.
"""
from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LabelStats:
    boxes: int = 0
    small: int = 0          # height < small_h px
    far_third: int = 0      # centre y < H/3
    far_small: int = 0      # both
    heights: list[float] = field(default_factory=list)

    def merge(self, other: LabelStats) -> None:
        self.boxes += other.boxes
        self.small += other.small
        self.far_third += other.far_third
        self.far_small += other.far_small
        self.heights.extend(other.heights)

    def report(self, title: str, frame_h: int, small_h: float) -> None:
        if not self.heights:
            print(f'{title}: no boxes')
            return
        hs = sorted(self.heights)
        n = len(hs)
        p10 = hs[max(0, int(n * 0.10) - 1)]
        med = hs[n // 2]
        print(f'{title}:')
        print(f'  boxes      : {self.boxes}')
        print(f'  height px  : min={hs[0]:.0f}  p10={p10:.0f}  median={med:.0f}  max={hs[-1]:.0f}')
        print(f'  small (<{small_h:.0f}px) : {self.small} ({100 * self.small / self.boxes:.1f}%)')
        print(f'  far third  : {self.far_third} ({100 * self.far_third / self.boxes:.1f}%)')
        print(f'  far+small  : {self.far_small} ({100 * self.far_small / self.boxes:.1f}%)')


@dataclass
class ImageRecord:
    img: Path
    lbl: Path
    stats: LabelStats
    has_small: bool
    has_far: bool


def _stats_from_label(label_path: Path, frame_h: int, small_h: float,
                      far_frac: float) -> LabelStats:
    st = LabelStats()
    if not label_path.is_file():
        return st
    far_y = frame_h * far_frac
    for line in label_path.read_text(encoding='utf-8').strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        h_px = bh * frame_h
        cy_px = cy * frame_h
        st.boxes += 1
        st.heights.append(h_px)
        is_small = h_px < small_h
        is_far = cy_px < far_y
        if is_small:
            st.small += 1
        if is_far:
            st.far_third += 1
        if is_small and is_far:
            st.far_small += 1
    return st


def _collect(source: Path, frame_h: int, small_h: float,
             far_frac: float) -> list[ImageRecord]:
    img_dir = source / 'images'
    lbl_dir = source / 'labels'
    if not img_dir.is_dir():
        raise SystemExit(f'Missing {img_dir}')
    if not lbl_dir.is_dir():
        raise SystemExit(f'Missing {lbl_dir}')

    exts = {'.jpg', '.jpeg', '.png', '.webp'}
    records: list[ImageRecord] = []
    missing_labels: list[str] = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in exts:
            continue
        lbl = lbl_dir / f'{img.stem}.txt'
        if not lbl.is_file():
            missing_labels.append(img.name)
            continue
        st = _stats_from_label(lbl, frame_h, small_h, far_frac)
        records.append(ImageRecord(
            img=img, lbl=lbl, stats=st,
            has_small=st.small > 0,
            has_far=st.far_third > 0,
        ))

    if missing_labels:
        print(f'WARNING: {len(missing_labels)} image(s) without labels '
              f'(first: {missing_labels[0]})')
    if not records:
        raise SystemExit(f'No image/label pairs under {source}')
    return records


def _stratified_split(records: list[ImageRecord], val_frac: float,
                      seed: int) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Split within (has_small, has_far) strata so rare buckets stay balanced."""
    rng = random.Random(seed)
    buckets: dict[tuple[bool, bool], list[ImageRecord]] = {}
    for rec in records:
        key = (rec.has_small, rec.has_far)
        buckets.setdefault(key, []).append(rec)

    train: list[ImageRecord] = []
    val: list[ImageRecord] = []
    for key in sorted(buckets):
        group = buckets[key][:]
        rng.shuffle(group)
        n_val = max(0, int(round(len(group) * val_frac)))
        if n_val == 0 and len(group) > 4:
            n_val = 1
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _copy_split(records: list[ImageRecord], dest: Path, split: str) -> LabelStats:
    img_out = dest / split / 'images'
    lbl_out = dest / split / 'labels'
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    total = LabelStats()
    for rec in records:
        shutil.copy2(rec.img, img_out / rec.img.name)
        shutil.copy2(rec.lbl, lbl_out / rec.lbl.name)
        total.merge(rec.stats)
    return total


def write_data_yaml(dest: Path, names: list[str]) -> Path:
    yaml_path = dest / 'data.yaml'
    lines = [
        f'path: {dest.as_posix()}',
        'train: train/images',
        'val: val/images',
        f'nc: {len(names)}',
        'names:',
    ]
    for i, n in enumerate(names):
        lines.append(f'  {i}: {n}')
    yaml_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, default=Path('/workspace/ds_full'),
                    help='Flat dataset root (images/ + labels/)')
    ap.add_argument('--dest', type=Path, default=Path('/workspace/ds_yolo'),
                    help='Output YOLO split directory')
    ap.add_argument('--val-frac', type=float, default=0.20,
                    help='Validation fraction (default 0.20 -> ~320/80 for 400 frames)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--frame-h', type=int, default=1152,
                    help='Native frame height for GT stats (4096x1152)')
    ap.add_argument('--small-h', type=float, default=40.0,
                    help='Small-box threshold in px for stats / eval alignment')
    ap.add_argument('--class-names', nargs='+',
                    default=['goalkeeper', 'player', 'referee'],
                    help='Class order must match label indices in .txt files')
    args = ap.parse_args()

    records = _collect(args.source, args.frame_h, args.small_h, 1.0 / 3.0)
    train_recs, val_recs = _stratified_split(records, args.val_frac, args.seed)

    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.mkdir(parents=True, exist_ok=True)

    all_stats = LabelStats()
    for rec in records:
        all_stats.merge(rec.stats)

    train_stats = _copy_split(train_recs, args.dest, 'train')
    val_stats = _copy_split(val_recs, args.dest, 'val')
    yaml_path = write_data_yaml(args.dest, args.class_names)

    print(f'Source images: {len(records)}')
    print(f'Train / val   : {len(train_recs)} / {len(val_recs)} (stratified by small+far tags)\n')
    all_stats.report('All labels', args.frame_h, args.small_h)
    print()
    train_stats.report('Train labels', args.frame_h, args.small_h)
    print()
    val_stats.report('Val labels (held-out for eval)', args.frame_h, args.small_h)
    print(f'\ndata.yaml     : {yaml_path}')
    print('Class map     :', dict(enumerate(args.class_names)))
    if val_stats.far_small < 5:
        print('\nNOTE: val has very few far+small GT boxes — eval recall on that '
              'bucket will be noisy. Labels under-represent far-touchline players '
              '(pre-annotation missed them too). Pass 1 fragment span is the '
              'real-world check after training.')


if __name__ == '__main__':
    main()
