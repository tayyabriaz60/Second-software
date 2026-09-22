#!/usr/bin/env python3
"""Feasibility probe: can jersey digits be read on large enough fragments?

Uses a track_dump JSON only — does not modify main.py or assign_identities.py.

Usage (RunPod, from sports-main/examples/soccer):
  pip install easyocr opencv-python-headless
  python tools/jersey_ocr_probe.py \\
    --dump data/id_lists/track_dump_clip10min_deliver_v2.json \\
    --video /workspace/clip10min.mp4 \\
    --out-dir data/jersey_ocr_probe

Video is read sequentially (grab/read forward). Do not use per-frame seek on
variable-frame-rate sources.

Pass: >=30%% of probed fragments get a consistent number (3+ agreeing reads,
majority >=60%% of successful reads) and the contact sheet looks like real
jersey digits. Below 10%% -> stop.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REFEREE_CLASS_ID = 2


def _spread_indices(idxs: list[int], k: int) -> list[int]:
    if len(idxs) <= k:
        return idxs[:]
    if k <= 1:
        return [idxs[len(idxs) // 2]]
    out = []
    for j in range(k):
        pos = int(round(j * (len(idxs) - 1) / (k - 1)))
        out.append(idxs[pos])
    return out


def box_from_centre_h(cx: float, cy: float, h: float,
                        width_frac: float) -> tuple[int, int, int, int]:
    w = max(8.0, h * width_frac)
    x1 = int(round(cx - w / 2))
    x2 = int(round(cx + w / 2))
    y1 = int(round(cy - h / 2))
    y2 = int(round(cy + h / 2))
    return x1, y1, x2, y2


def torso_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """15–60% box height, middle 60% width."""
    fh, fw = frame.shape[:2]
    x1 = max(0, min(x1, fw - 1))
    x2 = max(x1 + 1, min(x2, fw))
    y1 = max(0, min(y1, fh - 1))
    y2 = max(y1 + 1, min(y2, fh))
    bw, bh = x2 - x1, y2 - y1
    tx1 = x1 + int(0.20 * bw)
    tx2 = x1 + int(0.80 * bw)
    ty1 = y1 + int(0.15 * bh)
    ty2 = y1 + int(0.60 * bh)
    if tx2 <= tx1 or ty2 <= ty1:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return frame[ty1:ty2, tx1:tx2].copy()


def upscale(crop: np.ndarray, scale: float) -> np.ndarray:
    if crop.size == 0:
        return crop
    nh = max(8, int(round(crop.shape[0] * scale)))
    nw = max(8, int(round(crop.shape[1] * scale)))
    return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)


def parse_jersey_number(text: str) -> int | None:
    digits = re.sub(r'\D', '', text or '')
    if not digits:
        return None
    n = int(digits[:2]) if len(digits) >= 2 else int(digits[0])
    if 1 <= n <= 99:
        return n
    return None


class OcrEngine:
    def __init__(self, gpu: bool):
        self._reader = None
        self._gpu = gpu

    def _lazy_init(self):
        if self._reader is not None:
            return
        import easyocr
        self._reader = easyocr.Reader(['en'], gpu=self._gpu, verbose=False)

    def read_digits(self, bgr: np.ndarray) -> list[tuple[int, float]]:
        self._lazy_init()
        if bgr.size == 0:
            return []
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out: list[tuple[int, float]] = []
        for _bbox, text, conf in self._reader.readtext(
                rgb, allowlist='0123456789', paragraph=False):
            num = parse_jersey_number(text)
            if num is not None:
                out.append((num, float(conf)))
        return out


def majority_stats(reads: list[tuple[int, float]]) -> tuple[int | None, int, float]:
    """Majority number, agreeing count, mean confidence of that number."""
    if not reads:
        return None, 0, 0.0
    c = Counter(n for n, _ in reads)
    top, count = c.most_common(1)[0]
    confs = [conf for n, conf in reads if n == top]
    return top, count, float(sum(confs) / len(confs))


def is_consistent(reads: list[tuple[int, float]]) -> tuple[bool, int | None, int, int]:
    if not reads:
        return False, None, 0, 0
    maj, count, _ = majority_stats(reads)
    n = len(reads)
    ok = count >= 3 and (count / n) >= 0.60
    return ok, maj, count, n


def eligible_fragments(
    dump: dict,
    min_h: float,
    min_large_frames: int,
    max_samples: int,
    skip_referees: bool,
) -> list[dict]:
    eligible = []
    for tr in dump.get('tracks', []):
        if skip_referees and int(tr.get('class', 1)) == REFEREE_CLASS_ID:
            continue
        frames = tr['frames']
        hs = tr['h']
        large_idx = [i for i, h in enumerate(hs) if float(h) >= min_h]
        if len(large_idx) < min_large_frames:
            continue
        sample_idx = _spread_indices(large_idx, max_samples)
        eligible.append({
            'id': int(tr['id']),
            'team': tr.get('team'),
            'class': tr.get('class'),
            'n_large': len(large_idx),
            'sample_idx': sample_idx,
            'frames': frames,
        })
    return eligible


def frames_to_decode(eligible: list[dict]) -> set[int]:
    need: set[int] = set()
    for meta in eligible:
        for si in meta['sample_idx']:
            need.add(int(meta['frames'][si]))
    return need


def read_video_frames_sequential(
    video_path: Path,
    start_frame: int,
    needed: set[int],
) -> dict[int, np.ndarray]:
    """Forward-only decode — no cap.set(CAP_PROP_POS_FRAMES)."""
    if not needed:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f'Cannot open video: {video_path}')

    max_f = max(needed)
    out: dict[int, np.ndarray] = {}
    frame_idx = 0
    try:
        for _ in range(start_frame):
            if not cap.grab():
                return out
            frame_idx += 1

        while frame_idx <= max_f:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in needed:
                out[frame_idx] = frame
                if len(out) == len(needed):
                    break
            frame_idx += 1
    finally:
        cap.release()

    missing = needed - set(out.keys())
    if missing:
        print(f'  WARNING: {len(missing)} frames not read (eof or short video)')
    return out


def overlay_read_on_crop(crop: np.ndarray, label: str) -> np.ndarray:
    if crop.size == 0:
        return crop
    img = crop.copy()
    h = img.shape[0]
    cv2.rectangle(img, (0, 0), (img.shape[1], min(h, 22)), (0, 0, 0), -1)
    cv2.putText(img, label[:24], (4, min(h, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def build_contact_sheet(
    entries: list[dict],
    out_path: Path,
    thumb_h: int = 120,
    cols: int = 8,
) -> None:
    if not entries:
        return
    rows = int(math.ceil(len(entries) / cols))
    pad = 8
    label_h = 36
    cell_w = int(thumb_h * 0.85) + pad * 2
    cell_h = thumb_h + label_h + pad * 2
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 240, dtype=np.uint8)

    for i, ent in enumerate(entries):
        r, c = divmod(i, cols)
        y0 = r * cell_h + pad
        x0 = c * cell_w + pad
        crop = ent['thumb']
        if crop.size == 0:
            continue
        scale = thumb_h / max(crop.shape[0], 1)
        tw = max(1, int(crop.shape[1] * scale))
        thumb = cv2.resize(crop, (tw, thumb_h), interpolation=cv2.INTER_AREA)
        sheet[y0:y0 + thumb_h, x0:x0 + tw] = thumb
        for li, line in enumerate(ent['label'].split('\n')[:2]):
            cv2.putText(
                sheet, line[:32], (x0, y0 + thumb_h + 14 + li * 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dump', type=Path, required=True,
                    help='track_dump_*deliver_v2*.json')
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, default=Path('data/jersey_ocr_probe'))
    ap.add_argument('--min-h', type=float, default=200.0)
    ap.add_argument('--min-large-frames', type=int, default=20)
    ap.add_argument('--max-samples', type=int, default=30)
    ap.add_argument('--upscale', type=float, default=2.5,
                    help='Torso crop upscale (2–3 typical)')
    ap.add_argument('--width-frac', type=float, default=0.45,
                    help='Estimated box width as fraction of height')
    ap.add_argument('--cpu', action='store_true',
                    help='EasyOCR on CPU (default: GPU)')
    ap.add_argument('--contact-n', type=int, default=40)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-fragments', type=int, default=0,
                    help='Cap probed fragments (0 = all eligible)')
    args = ap.parse_args()

    dump = json.loads(args.dump.read_text(encoding='utf-8'))
    start_frame = int(dump.get('start_frame') or 0)
    eligible = eligible_fragments(
        dump, args.min_h, args.min_large_frames, args.max_samples,
        skip_referees=True)

    if args.max_fragments and len(eligible) > args.max_fragments:
        rng = random.Random(args.seed)
        eligible = rng.sample(eligible, args.max_fragments)

    needed_frames = frames_to_decode(eligible)
    print(f'Dump: {args.dump.name}')
    print(f'Eligible fragments (>={args.min_large_frames} frames h>={args.min_h}): '
          f'{len(eligible)}')
    print(f'Unique frames to decode: {len(needed_frames)}')
    print(f'Sequential read from dump start_frame={start_frame} (no seek)...')

    frame_cache = read_video_frames_sequential(
        args.video, start_frame, needed_frames)

    track_by_id = {int(tr['id']): tr for tr in dump['tracks']}
    ocr = OcrEngine(gpu=not args.cpu)
    results = []

    for meta in eligible:
        tid = meta['id']
        tr = track_by_id[tid]
        frames = meta['frames']
        xy = tr['xy']
        hs = tr['h']
        all_reads: list[tuple[int, float]] = []
        per_sample = []
        best_crop = None
        best_conf = -1.0

        for si in meta['sample_idx']:
            fnum = int(frames[si])
            frame = frame_cache.get(fnum)
            if frame is None:
                continue
            cx, cy = float(xy[si][0]), float(xy[si][1])
            h = float(hs[si])
            x1, y1, x2, y2 = box_from_centre_h(cx, cy, h, args.width_frac)
            crop = torso_crop(frame, x1, y1, x2, y2)
            up = upscale(crop, args.upscale)
            reads = ocr.read_digits(up)
            for num, conf in reads:
                all_reads.append((num, conf))
            per_sample.append({
                'frame': fnum,
                'h': h,
                'reads': [{'number': n, 'confidence': round(c, 3)} for n, c in reads],
            })
            if reads and max(c for _, c in reads) > best_conf:
                best_conf = max(c for _, c in reads)
                best_crop = up.copy()

        consistent, maj, maj_count, n_reads = is_consistent(all_reads)
        maj_num, _, maj_mean_conf = majority_stats(all_reads)

        results.append({
            'fragment_id': tid,
            'team': meta.get('team'),
            'class': meta.get('class'),
            'n_large_frames': meta['n_large'],
            'n_samples': len(per_sample),
            'n_successful_reads': n_reads,
            'all_reads': [{'number': n, 'confidence': round(c, 3)}
                          for n, c in all_reads],
            'majority_number': maj_num,
            'majority_count': maj_count,
            'majority_mean_confidence': round(maj_mean_conf, 3) if maj_num else None,
            'consistent': consistent,
            'per_sample': per_sample,
            '_thumb': best_crop,
            '_overlay': (
                f"#{maj_num} ({maj_count}/{n_reads})" if maj_num else 'no read'),
        })

    n_probed = len(results)
    n_any = sum(1 for r in results if r['n_successful_reads'] > 0)
    n_cons = sum(1 for r in results if r['consistent'])
    pct_any = 100.0 * n_any / n_probed if n_probed else 0.0
    pct_cons = 100.0 * n_cons / n_probed if n_probed else 0.0

    by_team: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        if r['majority_number'] is None:
            continue
        key = 'team_' + str(r['team']) if r['team'] is not None else 'team_unknown'
        by_team[key][r['majority_number']] += 1

    print('\n=== Jersey OCR probe report ===')
    print(f'  Fragments probed              : {n_probed}')
    print(f'  Any digit read                : {n_any} ({pct_any:.1f}%)')
    print(f'  Consistent (3+ & >=60% agree) : {n_cons} ({pct_cons:.1f}%)')
    print('\n  Majority number distribution by team (fragment counts):')
    for team_key in sorted(by_team.keys()):
        dist = by_team[team_key]
        print(f'    {team_key}: {len(dist)} distinct numbers, top={dist.most_common(12)}')

    if pct_cons >= 30.0:
        verdict = 'PASS (>=30% consistent) — worth a pipeline experiment'
    elif pct_cons < 10.0:
        verdict = 'STOP (<10% consistent) — jersey OCR unlikely to anchor identity here'
    else:
        verdict = f'MARGINAL ({pct_cons:.1f}% consistent) — eyeball contact sheet before continuing'

    print(f'\n  Verdict: {verdict}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / 'jersey_ocr_report.json'
    serializable = [{k: v for k, v in r.items() if not k.startswith('_')}
                    for r in results]
    payload = {
        'dump': str(args.dump),
        'video': str(args.video),
        'params': {k: v for k, v in vars(args).items()},
        'summary': {
            'fragments_probed': n_probed,
            'any_read': n_any,
            'consistent': n_cons,
            'pct_any_read': round(pct_any, 2),
            'pct_consistent': round(pct_cons, 2),
            'verdict': verdict,
        },
        'team_number_distribution': {k: dict(v) for k, v in by_team.items()},
        'fragments': serializable,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'\n  Report: {report_path}')

    rng = random.Random(args.seed)
    contact_pool = [r for r in results if r['_thumb'] is not None]
    contact_pick = rng.sample(
        contact_pool, min(args.contact_n, len(contact_pool)))
    contact_entries = []
    for r in contact_pick:
        thumb = overlay_read_on_crop(r['_thumb'], r['_overlay'])
        label = (f"frag {r['fragment_id']} team {r['team']}\n"
                 f"{r['_overlay']} {'OK' if r['consistent'] else 'weak'}")
        contact_entries.append({'thumb': thumb, 'label': label})

    sheet_path = args.out_dir / 'jersey_ocr_contact_sheet.jpg'
    build_contact_sheet(contact_entries, sheet_path)
    print(f'  Contact sheet: {sheet_path}')


if __name__ == '__main__':
    main()
