#!/usr/bin/env python3
"""Export labelled jersey digit crops from a jersey_ocr probe report + track dump.

Step 1 (data only): no training. Uses OCR labels already in the report; does not
re-run OCR.

Usage (from sports-main/examples/soccer):
  python tools/build_digit_dataset.py \\
    --report data/jersey_ocr_v5/jersey_ocr_report.json \\
    --dump data/id_lists/track_dump_clip10min_deliver_v2.json \\
    --video /workspace/clip10min.mp4 \\
    --out-dir data/digits

Layout:
  digits/<N>/frag<id>_f<frame>.jpg     - consistent read, label N (N != 1)
  digits/_review_1/                    - consistent read was 1 (likely 10; relabel by hand)
  digits/_unlabelled/                  - no consistent read

Crop: upper-back number band (square), cy-0.28h .. cy-0.05h, width = band height.
Centre and per-frame h match jersey_ocr_probe.py (post ccb739e).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name in ('cv2', 'numpy'):
        raise SystemExit(
            'Missing OpenCV/NumPy in this Python. Use the same env as '
            'jersey_ocr_probe (main tracking venv), e.g.:\n'
            '  source /workspace/venv/bin/activate   # if you use one\n'
            '  pip install "numpy==1.26.4" opencv-python-headless==4.10.0.84\n'
            'Then re-run build_digit_dataset.py.'
        ) from exc
    raise

_SOC = Path(__file__).resolve().parents[1]
if str(_SOC) not in sys.path:
    sys.path.insert(0, str(_SOC))

from tools.jersey_ocr_probe import (
    fit_y_height_model,
    iter_dump_tracks,
    per_track_heights,
    read_video_frames_sequential,
    sample_centre_and_height,
    upscale,
)

# Upper back / number patch (units of box height h).
DIGIT_Y_TOP = 0.28    # y1 = cy - DIGIT_Y_TOP * h
DIGIT_Y_ABOVE_CENTRE = 0.05   # y2 = cy - DIGIT_Y_ABOVE_CENTRE * h


def digit_back_crop(
    frame,
    cx: float,
    cy: float,
    h: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Tight square-ish crop on the jersey number (upper back)."""
    if h <= 0 or frame.size == 0:
        return frame[0:0, 0:0].copy(), (0, 0, 0, 0)
    fh, fw = frame.shape[:2]
    band_h = float(h) * (DIGIT_Y_TOP - DIGIT_Y_ABOVE_CENTRE)
    half = max(4.0, band_h / 2.0)
    y1 = int(round(cy - DIGIT_Y_TOP * h))
    y2 = int(round(cy - DIGIT_Y_ABOVE_CENTRE * h))
    x1 = int(round(cx - half))
    x2 = int(round(cx + half))
    x1 = max(0, min(x1, fw - 1))
    x2 = max(x1 + 1, min(x2, fw))
    y1 = max(0, min(y1, fh - 1))
    y2 = max(y1 + 1, min(y2, fh))
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def dest_subdir(frag: dict) -> str:
    if not frag.get('consistent'):
        return '_unlabelled'
    maj = frag.get('majority_number')
    if maj is None:
        return '_unlabelled'
    if int(maj) == 1:
        return '_review_1'
    return str(int(maj))


def frames_needed(report_frags: list[dict]) -> set[int]:
    need: set[int] = set()
    for frag in report_frags:
        for ps in frag.get('per_sample') or []:
            need.add(int(ps['frame']))
    return need


def build_dataset(
    report: dict,
    dump: dict,
    frame_cache: dict[int, np.ndarray],
    track_by_id: dict[int, dict],
    frame_h: float,
    y_model,
    out_dir: Path,
    upscale_factor: float,
) -> Counter:
    counts: Counter = Counter()
    frags = report.get('fragments') or []
    for frag in frags:
        fid = int(frag['fragment_id'])
        tr = track_by_id.get(fid)
        if tr is None:
            print(f'  WARNING: fragment {fid} not in dump — skip')
            continue
        sub = dest_subdir(frag)
        dest = out_dir / sub
        dest.mkdir(parents=True, exist_ok=True)
        frames = tr['frames']
        hs = per_track_heights(tr, frame_h, y_model, dump)
        if len(hs) != len(frames):
            hs = (hs + [0.0] * len(frames))[:len(frames)]
        for ps in frag.get('per_sample') or []:
            fnum = int(ps['frame'])
            frame = frame_cache.get(fnum)
            if frame is None:
                print(f'  WARNING: missing frame {fnum} frag {fid}')
                continue
            try:
                fi = frames.index(fnum)
            except ValueError:
                print(f'  WARNING: frame {fnum} not on track {fid}')
                continue
            cx, cy, h = sample_centre_and_height(tr, fi, hs)
            crop, _ = digit_back_crop(frame, cx, cy, h)
            if upscale_factor and upscale_factor != 1.0:
                crop = upscale(crop, upscale_factor)
            if crop.size == 0:
                continue
            name = f'frag{fid}_f{fnum}.jpg'
            cv2.imwrite(str(dest / name), crop)
            counts[sub] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--report', type=Path, required=True,
                    help='jersey_ocr_report.json from probe (e.g. v5)')
    ap.add_argument('--dump', type=Path, required=True,
                    help='track_dump used for the same probe run')
    ap.add_argument('--video', type=Path, required=True,
                    help='Source video (clip-relative frames + start_frame)')
    ap.add_argument('--out-dir', type=Path, default=Path('data/digits'))
    ap.add_argument('--upscale', type=float, default=2.5,
                    help='Same as probe upscale (2.5); set 1 for native px')
    ap.add_argument('--clean', action='store_true',
                    help='Remove --out-dir before writing')
    args = ap.parse_args()

    report = json.loads(args.report.read_text(encoding='utf-8'))
    dump = json.loads(args.dump.read_text(encoding='utf-8'))
    start_frame = int(dump.get('start_frame') or 0)
    frame_h = float(dump.get('height') or 0)

    tracks = iter_dump_tracks(dump)
    track_by_id = {int(tr['id']): tr for tr in tracks}
    y_model = fit_y_height_model(tracks, frame_h, dump)

    frags = report.get('fragments') or []
    need = frames_needed(frags)
    print(f'Report: {args.report.name} ({len(frags)} fragments, '
          f'{len(need)} unique frames to decode)')
    print(f'Dump start_frame={start_frame}, video={args.video}')

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print('Sequential video decode...')
    frame_cache = read_video_frames_sequential(
        args.video, start_frame, need)
    got = len(need & set(frame_cache.keys()))
    print(f'  Decoded {got}/{len(need)} target frame(s)')

    counts = build_dataset(
        report, dump, frame_cache, track_by_id, frame_h, y_model,
        args.out_dir, args.upscale)

    n_frags = len(frags)
    n_cons = sum(1 for f in frags if f.get('consistent'))
    n_review = sum(1 for f in frags if dest_subdir(f) == '_review_1')
    n_unlab = sum(1 for f in frags if dest_subdir(f) == '_unlabelled')

    print('\n=== Digit dataset export ===')
    print(f'  Output: {args.out_dir.resolve()}')
    print(f'  Fragments in report     : {n_frags}')
    print(f'  Consistent labels       : {n_cons}')
    print(f'  Sent to _review_1 (maj=1): {n_review} fragments')
    print(f'  Unlabelled fragments    : {n_unlab}')
    print(f'  Crop: y=[cy-{DIGIT_Y_TOP}h, cy-{DIGIT_Y_ABOVE_CENTRE}h], '
          f'square width=band height, upscale={args.upscale}')
    print('\n  Crops per folder:')
    for key in sorted(counts.keys(), key=lambda k: (k.startswith('_'), k)):
        print(f'    {key}: {counts[key]}')
    print(f'    TOTAL: {sum(counts.values())}')


if __name__ == '__main__':
    main()
