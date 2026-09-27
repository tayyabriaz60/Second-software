#!/usr/bin/env python3
"""Full-clip jersey digit probe using the trained classifier (not OCR).

Memory-safe for CPU pods:
  - Streams the track dump (ijson) — never json.loads the whole file
  - Strips appearance vectors
  - Streams video one frame at a time

  pip install ijson

Usage:
  python tools/jersey_digit_probe.py \\
    --dump data/id_lists/track_dump_clip10min_deliver_v2.json \\
    --video /workspace/clip10min.mp4 \\
    --checkpoint runs/digit_cls_v2/best.pt \\
    --replay-report data/jersey_ocr_v5/jersey_ocr_report.json \\
    --out-dir data/jersey_digit_v1
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

_SOC = Path(__file__).resolve().parents[1]
if str(_SOC) not in sys.path:
    sys.path.insert(0, str(_SOC))

try:
    import torch
    from torchvision import transforms
except ModuleNotFoundError as exc:
    raise SystemExit('pip install torch torchvision') from exc

try:
    import ijson
except ModuleNotFoundError as exc:
    raise SystemExit(
        'pip install ijson\n'
        '(Needed to stream the track dump without OOM on CPU pods.)'
    ) from exc

from tools.build_digit_dataset import digit_back_crop
from tools.infer_digit_classifier import load_model
from tools.jersey_ocr_probe import (
    is_consistent,
    json_safe,
    majority_stats,
    rebuild_contact_sheet,
    sample_centre_and_height,
    upscale,
    _spread_indices,
    resolve_referee_class_id,
)


def _tfm(img_size: int):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict_crop(
        model, transform, crop_bgr: np.ndarray, device: torch.device,
        idx_to_class: dict[int, str],
) -> tuple[int | None, float, str]:
    if crop_bgr.size == 0:
        return None, 0.0, ''
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = transform(rgb).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0]
    conf, pred = probs.max(dim=0)
    label = idx_to_class[int(pred.item())]
    try:
        num = int(label)
    except ValueError:
        return None, float(conf.item()), label
    return num, float(conf.item()), label


def sample_frames_from_report(frag_report: dict) -> list[int]:
    return [int(ps['frame']) for ps in (frag_report.get('per_sample') or [])]


def read_dump_header(path: Path) -> dict:
    """Read fps/width/height/start_frame from the start of the JSON (no full parse)."""
    with open(path, 'rb') as f:
        head = f.read(4096).decode('utf-8', errors='ignore')
    out: dict = {}
    for key in ('fps', 'width', 'height', 'start_frame', 'team_sep'):
        m = re.search(rf'"{key}"\s*:\s*([0-9.eE+-]+)', head)
        if m:
            raw = m.group(1)
            out[key] = float(raw) if '.' in raw or 'e' in raw.lower() else int(raw)
    return out


def stream_load_tracks(
        path: Path,
        want_ids: set[int] | None,
) -> list[dict]:
    """Stream tracks.item; keep only want_ids (if set); drop appearance."""
    tracks: list[dict] = []
    n_seen = 0
    with open(path, 'rb') as f:
        for tr in ijson.items(f, 'tracks.item'):
            n_seen += 1
            if n_seen % 200 == 0:
                print(f'  ...scanned {n_seen} tracklets, kept {len(tracks)}',
                      flush=True)
            tid = int(tr.get('id', -1))
            if want_ids is not None and tid not in want_ids:
                continue
            tr.pop('appearance', None)
            # Drop nothing else — need frames, xy, h, class, team
            tracks.append(tr)
    print(f'  Stream done: scanned {n_seen}, kept {len(tracks)}', flush=True)
    return tracks


def heights_from_track(tr: dict) -> list[float]:
    raw = tr.get('h') or tr.get('box_height_px') or tr.get('heights') or []
    if not isinstance(raw, list):
        return [0.0] * len(tr.get('frames') or [])
    out = []
    for v in raw:
        try:
            out.append(float(v) if v not in (None, '') else 0.0)
        except (TypeError, ValueError):
            out.append(0.0)
    n = len(tr.get('frames') or [])
    if len(out) < n:
        out.extend([0.0] * (n - len(out)))
    return out[:n]


def build_eligible_from_tracks(
        tracks: list[dict],
        min_h: float,
        min_large_frames: int,
        max_samples: int,
        skip_referees: bool,
        referee_class: int,
        report_by_id: dict[int, dict],
) -> list[dict]:
    """Eligibility using dump h only (no full-dump y-fallback model)."""
    eligible = []
    for tr in tracks:
        if skip_referees and int(tr.get('class', 1)) == referee_class:
            continue
        tid = int(tr['id'])
        frames = tr.get('frames') or []
        hs = heights_from_track(tr)
        if tid in report_by_id:
            # Trust OCR-v5 eligibility; use report sample frames later
            sample_frames = sample_frames_from_report(report_by_id[tid])
            # Map to indices if possible; heights aligned to dump frames
            eligible.append({
                'id': tid,
                'team': tr.get('team'),
                'class': tr.get('class'),
                'n_large': sum(1 for h in hs if h >= min_h),
                'sample_idx': [],  # unused when report frames set
                'sample_frames': sample_frames,
                'frames': frames,
                'heights': hs,
            })
            continue
        large_idx = [i for i, h in enumerate(hs) if float(h) >= min_h]
        if len(large_idx) < min_large_frames:
            continue
        sample_idx = _spread_indices(large_idx, max_samples)
        eligible.append({
            'id': tid,
            'team': tr.get('team'),
            'class': tr.get('class'),
            'n_large': len(large_idx),
            'sample_idx': sample_idx,
            'sample_frames': [int(frames[i]) for i in sample_idx],
            'frames': frames,
            'heights': hs,
        })
    return eligible


def stream_classify(
        video: Path,
        start_frame: int,
        need_clip: set[int],
        jobs_by_frame: dict[int, list[dict]],
        model,
        transform,
        device: torch.device,
        idx_to_class: dict[int, str],
        upscale_factor: float,
        frag_state: dict[int, dict],
) -> int:
    if not need_clip:
        return 0
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f'Cannot open video: {video}')

    needed_abs = {f + int(start_frame) for f in need_clip}
    max_abs = max(needed_abs)
    remaining = set(needed_abs)
    n_hit = 0
    frame_idx = 0
    try:
        for _ in range(int(start_frame)):
            if not cap.grab():
                return n_hit
            frame_idx += 1

        span = max(1, max_abs - frame_idx)
        log_every = max(500, span // 20)
        while frame_idx <= max_abs and remaining:
            if frame_idx in remaining:
                ok, frame = cap.read()
                if not ok:
                    break
                clip_idx = frame_idx - int(start_frame)
                for job in jobs_by_frame.get(clip_idx, []):
                    tid = job['tid']
                    cx, cy, h = job['cx'], job['cy'], job['h']
                    crop, _ = digit_back_crop(frame, cx, cy, h)
                    up = upscale(crop, upscale_factor)
                    num, conf, raw = predict_crop(
                        model, transform, up, device, idx_to_class)
                    st = frag_state[tid]
                    if num is not None:
                        st['all_reads'].append((num, conf))
                    st['per_sample'].append({
                        'frame': clip_idx,
                        'h': h,
                        'reads': (
                            [{'number': num, 'confidence': round(conf, 3),
                              'raw_digits': raw}]
                            if num is not None else []
                        ),
                    })
                    if num is not None and conf > st['best_conf']:
                        st['best_conf'] = conf
                        st['_thumb'] = up.copy()
                remaining.discard(frame_idx)
                n_hit += 1
                del frame
            else:
                if not cap.grab():
                    break
            if log_every and frame_idx % log_every == 0:
                print(f'  decode: video frame {frame_idx}/{max_abs}, '
                      f'hits {n_hit}/{len(need_clip)}', flush=True)
            frame_idx += 1
    finally:
        cap.release()
    return n_hit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dump', type=Path, required=True)
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, default=Path('data/jersey_digit_v1'))
    ap.add_argument('--replay-report', type=Path, default=None,
                    help='OCR jersey_ocr_report.json: same frags + sample frames')
    ap.add_argument('--min-h', type=float, default=200.0)
    ap.add_argument('--min-large-frames', type=int, default=20)
    ap.add_argument('--max-samples', type=int, default=30)
    ap.add_argument('--upscale', type=float, default=2.5)
    ap.add_argument('--contact-n', type=int, default=40)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-fragments', type=int, default=0)
    ap.add_argument('--only-fragments', type=str, default=None)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--include-referees', action='store_true')
    ap.add_argument('--referee-class', type=int, default=None)
    args = ap.parse_args()

    use_cuda = bool(args.gpu and torch.cuda.is_available())
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Device: {device}', flush=True)

    model, idx_to_class, img_size = load_model(args.checkpoint, device)
    transform = _tfm(img_size)
    print(f'Classes: {[idx_to_class[i] for i in range(len(idx_to_class))]}',
          flush=True)

    report_by_id: dict[int, dict] = {}
    want_ids: set[int] | None = None
    if args.replay_report and args.replay_report.is_file():
        prior = json.loads(args.replay_report.read_text(encoding='utf-8'))
        report_by_id = {
            int(f['fragment_id']): f for f in prior.get('fragments') or []
        }
        want_ids = set(report_by_id.keys())
        print(f'Replay report: {len(want_ids)} fragment ids', flush=True)

    if args.only_fragments:
        only = {
            int(x.strip()) for x in args.only_fragments.split(',') if x.strip()
        }
        want_ids = only if want_ids is None else (want_ids & only)
        print(f'Only fragments: {sorted(want_ids) if want_ids else only}',
              flush=True)

    header = read_dump_header(args.dump)
    start_frame = int(header.get('start_frame') or 0)
    print(f'Dump header: {header}', flush=True)
    print(f'Streaming dump tracks from {args.dump.name} '
          f'(want_ids={len(want_ids) if want_ids else "ALL"})...', flush=True)

    tracks = stream_load_tracks(args.dump, want_ids)
    gc.collect()

    # Minimal dump dict for referee class resolution
    dump_meta = dict(header)
    dump_meta['tracks'] = tracks
    ref_cls = resolve_referee_class_id(dump_meta, args.referee_class)

    eligible = build_eligible_from_tracks(
        tracks,
        args.min_h,
        args.min_large_frames,
        args.max_samples,
        skip_referees=not args.include_referees,
        referee_class=ref_cls,
        report_by_id=report_by_id,
    )

    if report_by_id:
        order = {fid: i for i, fid in enumerate(report_by_id.keys())}
        eligible.sort(key=lambda e: order.get(int(e['id']), 10**9))

    if args.max_fragments and len(eligible) > args.max_fragments:
        rng = random.Random(args.seed)
        eligible = rng.sample(eligible, args.max_fragments)

    track_by_id = {int(t['id']): t for t in tracks}
    need: set[int] = set()
    frag_frame_list: dict[int, list[int]] = {}
    for e in eligible:
        tid = int(e['id'])
        frames = e.get('sample_frames') or []
        frag_frame_list[tid] = frames
        need.update(frames)

    print(f'Eligible / probing: {len(eligible)}', flush=True)
    print(f'Unique frames to decode: {len(need)}', flush=True)
    if need:
        print(f'  Clip frame range: {min(need)}..{max(need)} '
              f'(dump start_frame={start_frame})', flush=True)

    jobs_by_frame: dict[int, list[dict]] = defaultdict(list)
    frag_state: dict[int, dict] = {}
    meta_by_id = {int(e['id']): e for e in eligible}

    for tid, sample_frames in frag_frame_list.items():
        tr = track_by_id.get(tid)
        meta = meta_by_id.get(tid)
        if tr is None or meta is None:
            continue
        frames = tr['frames']
        hs = meta['heights']
        if len(hs) != len(frames):
            hs = (hs + [0.0] * len(frames))[:len(frames)]
        frag_state[tid] = {
            'meta': meta,
            'all_reads': [],
            'per_sample': [],
            'best_conf': -1.0,
            '_thumb': None,
        }
        for fnum in sample_frames:
            try:
                si = frames.index(fnum)
            except ValueError:
                continue
            cx, cy, h = sample_centre_and_height(tr, si, hs)
            jobs_by_frame[fnum].append({
                'tid': tid, 'cx': cx, 'cy': cy, 'h': h,
            })

    del tracks
    del track_by_id
    dump_meta['tracks'] = []
    gc.collect()

    print('Streaming video decode + classify...', flush=True)
    n_hit = stream_classify(
        args.video, start_frame, need, jobs_by_frame,
        model, transform, device, idx_to_class, args.upscale, frag_state)
    print(f'  Hit {n_hit}/{len(need)} target frame(s)', flush=True)

    results = []
    for tid in [int(e['id']) for e in eligible]:
        st = frag_state.get(tid)
        if st is None:
            continue
        meta = st['meta']
        all_reads = st['all_reads']
        consistent, maj, maj_count, n_reads = is_consistent(all_reads)
        maj_num, _, maj_mean_conf = majority_stats(all_reads)
        results.append({
            'fragment_id': tid,
            'team': meta.get('team'),
            'class': meta.get('class'),
            'n_samples': len(st['per_sample']),
            'n_successful_reads': n_reads,
            'all_reads': [
                {'number': n, 'confidence': round(c, 3)} for n, c in all_reads
            ],
            'majority_number': maj_num,
            'majority_count': maj_count,
            'majority_mean_confidence': (
                round(maj_mean_conf, 3) if maj_num else None),
            'consistent': consistent,
            'per_sample': st['per_sample'],
            '_thumb': st['_thumb'],
            '_overlay': (
                f'#{maj_num} ({maj_count}/{n_reads})' if maj_num else 'no read'),
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
        key = (
            'team_' + str(r['team']) if r['team'] is not None else 'team_unknown'
        )
        by_team[key][r['majority_number']] += 1

    print('\n=== Jersey digit classifier probe ===')
    print(f'  Fragments probed              : {n_probed}')
    print(f'  Any digit read                : {n_any} ({pct_any:.1f}%)')
    print(f'  Consistent (3+ & >=60% agree) : {n_cons} ({pct_cons:.1f}%)')
    print('\n  Majority number distribution by team:')
    for team_key in sorted(by_team.keys()):
        dist = by_team[team_key]
        print(f'    {team_key}: {len(dist)} distinct, top={dist.most_common(12)}')

    if pct_cons >= 30.0:
        verdict = 'PASS (>=30% consistent) — worth merge experiment'
    elif pct_cons < 10.0:
        verdict = 'STOP (<10% consistent) — need more labels / model'
    else:
        verdict = (f'MARGINAL ({pct_cons:.1f}% consistent) — '
                   f'eyeball contact sheet; compare to OCR ~18.5%')

    print(f'\n  Verdict: {verdict}')
    print('  Compare: OCR v5 was ~18.5% consistent on same clip/eligibility.')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / 'jersey_digit_report.json'
    serializable = [
        {k: v for k, v in r.items() if not k.startswith('_')} for r in results
    ]
    payload = {
        'dump': str(args.dump),
        'video': str(args.video),
        'checkpoint': str(args.checkpoint),
        'device': str(device),
        'replay_report': str(args.replay_report) if args.replay_report else None,
        'params': json_safe(vars(args)),
        'dump_header': header,
        'summary': {
            'fragments_probed': n_probed,
            'any_read': n_any,
            'consistent': n_cons,
            'pct_any_read': round(pct_any, 2),
            'pct_consistent': round(pct_cons, 2),
            'verdict': verdict,
            'ocr_v5_pct_consistent_ref': 18.5,
        },
        'team_number_distribution': {k: dict(v) for k, v in by_team.items()},
        'fragments': serializable,
    }
    report_path.write_text(
        json.dumps(json_safe(payload), indent=2), encoding='utf-8')
    print(f'\n  Report: {report_path}')

    sheet_path = args.out_dir / 'jersey_digit_contact_sheet.jpg'
    replay_ids = list(report_by_id.keys()) if report_by_id else None
    rebuild_contact_sheet(
        sheet_path, results, replay_ids, args.contact_n, args.seed)


if __name__ == '__main__':
    main()
