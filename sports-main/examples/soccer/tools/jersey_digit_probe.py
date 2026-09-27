#!/usr/bin/env python3
"""Full-clip jersey digit probe using the trained classifier (not OCR).

Memory-safe for small CPU pods: strips dump appearance vectors, streams the
video one frame at a time (does not cache thousands of 4K frames).

Usage:
  cd sports-main/examples/soccer
  export PYTHONPATH=/workspace/Second-software/sports-main
  source /workspace/venv_jersey_paddle/bin/activate

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

from tools.build_digit_dataset import digit_back_crop
from tools.infer_digit_classifier import load_model
from tools.jersey_ocr_probe import (
    eligible_fragments,
    is_consistent,
    iter_dump_tracks,
    json_safe,
    majority_stats,
    rebuild_contact_sheet,
    sample_centre_and_height,
    upscale,
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


def slim_dump_tracks(dump: dict) -> None:
    """Drop appearance embeddings (large) to fit small CPU RAM."""
    for tr in iter_dump_tracks(dump):
        if 'appearance' in tr:
            tr['appearance'] = None


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
    """One forward pass over the video; process crops; free each frame."""
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
                        # Keep a small thumb only (contact sheet); drop later if none
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
    ap.add_argument('--gpu', action='store_true',
                    help='Use CUDA if available (default: CPU)')
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

    print(f'Loading dump (may take a minute)... {args.dump}', flush=True)
    dump = json.loads(args.dump.read_text(encoding='utf-8'))
    slim_dump_tracks(dump)
    gc.collect()
    print('  Dump loaded; appearance vectors stripped for RAM.', flush=True)

    start_frame = int(dump.get('start_frame') or 0)

    eligible, height_diag = eligible_fragments(
        dump, args.min_h, args.min_large_frames, args.max_samples,
        skip_referees=not args.include_referees,
        referee_class=args.referee_class)
    track_by_id = {int(t['id']): t for t in iter_dump_tracks(dump)}

    report_by_id: dict[int, dict] = {}
    if args.replay_report and args.replay_report.is_file():
        prior = json.loads(args.replay_report.read_text(encoding='utf-8'))
        report_by_id = {
            int(f['fragment_id']): f for f in prior.get('fragments') or []
        }
        want = set(report_by_id.keys())
        eligible = [e for e in eligible if int(e['id']) in want]
        order = {fid: i for i, fid in enumerate(report_by_id.keys())}
        eligible.sort(key=lambda e: order.get(int(e['id']), 10**9))
        print(f'Replay OCR report: {len(eligible)} fragments with prior samples',
              flush=True)

    if args.only_fragments:
        want_only = {
            int(x.strip()) for x in args.only_fragments.split(',') if x.strip()
        }
        eligible = [e for e in eligible if int(e['id']) in want_only]

    if args.max_fragments and len(eligible) > args.max_fragments:
        rng = random.Random(args.seed)
        eligible = rng.sample(eligible, args.max_fragments)

    need: set[int] = set()
    frag_frame_list: dict[int, list[int]] = {}
    for e in eligible:
        tid = int(e['id'])
        if tid in report_by_id:
            frames = sample_frames_from_report(report_by_id[tid])
        else:
            frames = [int(e['frames'][si]) for si in e['sample_idx']]
        frag_frame_list[tid] = frames
        need.update(frames)

    print(f'Dump: {args.dump.name}', flush=True)
    print(f'  Height diagnostics: {height_diag}', flush=True)
    print(f'Eligible / probing: {len(eligible)}', flush=True)
    print(f'Unique frames to decode: {len(need)}', flush=True)
    if need:
        print(f'  Clip frame range: {min(need)}..{max(need)} '
              f'(dump start_frame={start_frame})', flush=True)

    # Build per-frame job list (no full-frame cache)
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

    # Free track maps we no longer need for decode (keep frag_state / meta)
    del track_by_id
    dump['tracks'] = []
    gc.collect()

    print('Streaming video decode + classify (low RAM)...', flush=True)
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
        'height_diagnostics': height_diag,
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
