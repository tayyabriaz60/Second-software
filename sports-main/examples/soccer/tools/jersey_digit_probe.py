#!/usr/bin/env python3
"""Full-clip jersey digit probe using the trained classifier (not OCR).

CPU-pod safe path:
  1) Stream dump with ijson (only needed tracks)
  2) Extract needed frames to JPEG on disk (ffmpeg, no torch in RAM)
  3) Load classifier and run on JPEGs

  pip install ijson
  apt-get install -y ffmpeg

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
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

_SOC = Path(__file__).resolve().parents[1]
if str(_SOC) not in sys.path:
    sys.path.insert(0, str(_SOC))

try:
    import ijson
except ModuleNotFoundError as exc:
    raise SystemExit(
        'pip install ijson\n'
        '(Needed to stream the track dump without OOM on CPU pods.)'
    ) from exc

from tools.build_digit_dataset import digit_back_crop
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


def sample_frames_from_report(frag_report: dict) -> list[int]:
    return [int(ps['frame']) for ps in (frag_report.get('per_sample') or [])]


def read_dump_header(path: Path) -> dict:
    with open(path, 'rb') as f:
        head = f.read(4096).decode('utf-8', errors='ignore')
    out: dict = {}
    for key in ('fps', 'width', 'height', 'start_frame', 'team_sep'):
        m = re.search(rf'"{key}"\s*:\s*([0-9.eE+-]+)', head)
        if m:
            raw = m.group(1)
            out[key] = float(raw) if '.' in raw or 'e' in raw.lower() else int(raw)
    return out


def stream_load_tracks(path: Path, want_ids: set[int] | None) -> list[dict]:
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
    eligible = []
    for tr in tracks:
        if skip_referees and int(tr.get('class', 1)) == referee_class:
            continue
        tid = int(tr['id'])
        frames = tr.get('frames') or []
        hs = heights_from_track(tr)
        if tid in report_by_id:
            sample_frames = sample_frames_from_report(report_by_id[tid])
            eligible.append({
                'id': tid,
                'team': tr.get('team'),
                'class': tr.get('class'),
                'n_large': sum(1 for h in hs if h >= min_h),
                'sample_idx': [],
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


def extract_frames_jpeg(
        video: Path,
        start_frame: int,
        need_clip: set[int],
        out_dir: Path,
        frame_w: int,
        frame_h: int,
) -> dict[int, Path]:
    """Sequential ffmpeg raw decode; write only needed frames as JPEG.

    Runs before torch is loaded so RAM stays low. One frame buffer at a time.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not need_clip:
        return {}
    if frame_w <= 0 or frame_h <= 0:
        raise SystemExit('Dump header missing width/height')

    need_abs = {int(f) + int(start_frame) for f in need_clip}
    max_abs = max(need_abs)
    remaining = set(need_abs)
    mapping: dict[int, Path] = {}
    # Resume: already on disk
    for abs_f in list(need_abs):
        clip_f = abs_f - int(start_frame)
        dest = out_dir / f'clip{clip_f}_abs{abs_f}.jpg'
        if dest.is_file() and dest.stat().st_size > 0:
            mapping[clip_f] = dest
            remaining.discard(abs_f)

    if not remaining:
        print(f'Reusing {len(mapping)} existing JPEGs in {out_dir}', flush=True)
        return mapping

    nbytes = int(frame_w) * int(frame_h) * 3
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-threads', '1',
        '-i', str(video),
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-vsync', '0',
        '-',
    ]
    print(f'Sequential extract {len(remaining)} frames '
          f'(through abs {max_abs}), ~{nbytes / 1e6:.1f} MB/frame...',
          flush=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=nbytes)
    assert proc.stdout is not None
    frame_idx = 0
    n_wrote = 0
    try:
        while frame_idx <= max_abs and remaining:
            raw = proc.stdout.read(nbytes)
            if raw is None or len(raw) < nbytes:
                err = ''
                if proc.stderr:
                    err = proc.stderr.read().decode('utf-8', errors='ignore')[:300]
                print(f'  ffmpeg EOF at {frame_idx}: {err}', flush=True)
                break
            if frame_idx in remaining:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (int(frame_h), int(frame_w), 3))
                clip_f = frame_idx - int(start_frame)
                dest = out_dir / f'clip{clip_f}_abs{frame_idx}.jpg'
                cv2.imwrite(str(dest), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                mapping[clip_f] = dest
                remaining.discard(frame_idx)
                n_wrote += 1
                if n_wrote == 1:
                    print(f'  first JPEG ok at abs {frame_idx}', flush=True)
                del frame
                if n_wrote % 25 == 0:
                    print(f'  wrote {n_wrote}, left {len(remaining)}', flush=True)
                    gc.collect()
            del raw
            if frame_idx % 2000 == 0 and frame_idx > 0:
                print(f'  pass frame {frame_idx}/{max_abs}', flush=True)
            frame_idx += 1
    finally:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    print(f'  JPEGs ready: {len(mapping)}/{len(need_clip)}', flush=True)
    return mapping


def classify_from_jpegs(
        frame_paths: dict[int, Path],
        jobs_by_frame: dict[int, list[dict]],
        frag_state: dict[int, dict],
        model,
        transform,
        device,
        idx_to_class: dict[int, str],
        upscale_factor: float,
) -> int:
    import torch

    n_hit = 0
    for clip_f in sorted(frame_paths.keys()):
        path = frame_paths[clip_f]
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        n_hit += 1
        for job in jobs_by_frame.get(clip_f, []):
            tid = job['tid']
            crop, _ = digit_back_crop(frame, job['cx'], job['cy'], job['h'])
            up = upscale(crop, upscale_factor)
            if up.size == 0:
                continue
            rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
            x = transform(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0]
                conf, pred = probs.max(dim=0)
            label = idx_to_class[int(pred.item())]
            try:
                num = int(label)
            except ValueError:
                num = None
            conf_f = float(conf.item())
            st = frag_state[tid]
            if num is not None:
                st['all_reads'].append((num, conf_f))
            st['per_sample'].append({
                'frame': clip_f,
                'h': job['h'],
                'reads': (
                    [{'number': num, 'confidence': round(conf_f, 3),
                      'raw_digits': label}]
                    if num is not None else []
                ),
            })
            if num is not None and conf_f > st['best_conf']:
                st['best_conf'] = conf_f
                st['_thumb'] = up.copy()
        del frame
        if n_hit % 20 == 0:
            gc.collect()
    return n_hit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dump', type=Path, required=True)
    ap.add_argument('--video', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, default=Path('data/jersey_digit_v1'))
    ap.add_argument('--replay-report', type=Path, default=None)
    ap.add_argument('--min-h', type=float, default=200.0)
    ap.add_argument('--min-large-frames', type=int, default=20)
    ap.add_argument('--max-samples', type=int, default=30)
    ap.add_argument('--upscale', type=float, default=2.5)
    ap.add_argument('--contact-n', type=int, default=40)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-fragments', type=int, default=0)
    ap.add_argument('--only-fragments', type=str, default=None)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--frames-dir', type=Path, default=None,
                    help='JPEG cache dir (default: out-dir/frames)')
    ap.add_argument('--skip-extract', action='store_true',
                    help='Reuse existing JPEGs in --frames-dir')
    ap.add_argument('--keep-frames', action='store_true',
                    help='Do not delete extracted JPEGs at end')
    ap.add_argument('--include-referees', action='store_true')
    ap.add_argument('--referee-class', type=int, default=None)
    args = ap.parse_args()

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
        print(f'Only fragments: {sorted(want_ids)}', flush=True)

    header = read_dump_header(args.dump)
    start_frame = int(header.get('start_frame') or 0)
    print(f'Dump header: {header}', flush=True)
    print(f'Streaming dump tracks (want_ids='
          f'{len(want_ids) if want_ids else "ALL"})...', flush=True)

    tracks = stream_load_tracks(args.dump, want_ids)
    gc.collect()

    dump_meta = dict(header)
    dump_meta['tracks'] = tracks
    ref_cls = resolve_referee_class_id(dump_meta, args.referee_class)

    eligible = build_eligible_from_tracks(
        tracks, args.min_h, args.min_large_frames, args.max_samples,
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
    print(f'Unique frames to extract: {len(need)}', flush=True)
    if need:
        print(f'  Clip frame range: {min(need)}..{max(need)}', flush=True)

    jobs_by_frame: dict[int, list[dict]] = defaultdict(list)
    frag_state: dict[int, dict] = {}
    for tid, sample_frames in frag_frame_list.items():
        tr = track_by_id.get(tid)
        meta = next((e for e in eligible if int(e['id']) == tid), None)
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

    frames_dir = args.frames_dir or (args.out_dir / 'frames')
    if args.skip_extract:
        frame_paths = {}
        for clip_f in need:
            abs_f = clip_f + start_frame
            dest = frames_dir / f'clip{clip_f}_abs{abs_f}.jpg'
            if dest.is_file():
                frame_paths[clip_f] = dest
        print(f'Reusing {len(frame_paths)} JPEGs from {frames_dir}', flush=True)
    else:
        frame_paths = extract_frames_jpeg(
            args.video, start_frame, need, frames_dir,
            frame_w=int(header.get('width') or 0),
            frame_h=int(header.get('height') or 0),
        )

    print('Loading torch + classifier (after extract)...', flush=True)
    try:
        import torch
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise SystemExit('pip install torch torchvision') from exc

    from tools.infer_digit_classifier import load_model
    from tools.train_digit_classifier import SmallDigitCNN  # noqa: F401

    use_cuda = bool(args.gpu and torch.cuda.is_available())
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Device: {device}', flush=True)
    model, idx_to_class, img_size = load_model(args.checkpoint, device)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print(f'Classes: {[idx_to_class[i] for i in range(len(idx_to_class))]}',
          flush=True)

    n_hit = classify_from_jpegs(
        frame_paths, jobs_by_frame, frag_state,
        model, transform, device, idx_to_class, args.upscale)
    print(f'  Classified from {n_hit} JPEG frame(s)', flush=True)

    if not args.keep_frames and not args.skip_extract:
        # keep frames by default for resume; only delete if user wants later
        pass

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
