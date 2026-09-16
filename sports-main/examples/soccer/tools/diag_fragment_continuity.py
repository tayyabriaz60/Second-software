#!/usr/bin/env python3
"""Analyze within-fragment continuity from a track_dump or track_diag JSON.

Reports frame coverage, internal gap counts (≤10 vs >10 frames), and — for
track_diag — why canonical ids ended (detection present vs absent).

Usage:
    python tools/diag_fragment_continuity.py --dump track_dump_clip.json
    python tools/diag_fragment_continuity.py --diag track_diag_clip.json
    python tools/diag_fragment_continuity.py --dump a.json --compare b.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _gaps_from_frames(frames: list[int]) -> list[int]:
    frames = sorted(int(f) for f in frames)
    return [frames[i + 1] - frames[i] - 1 for i in range(len(frames) - 1)]


def analyze_track(frames: list[int]) -> dict[str, Any]:
    frames = sorted(int(f) for f in frames)
    if len(frames) < 2:
        return {}
    span = int(frames[-1] - frames[0]) + 1
    gaps = _gaps_from_frames(frames)
    pos_gaps = [g for g in gaps if g > 0]
    return {
        'samples': len(frames),
        'span_frames': span,
        'coverage': len(frames) / span if span else 0.0,
        'internal_gaps': len(pos_gaps),
        'gaps_le_10': sum(1 for g in pos_gaps if g <= 10),
        'gaps_gt_10': sum(1 for g in pos_gaps if g > 10),
        'max_gap': max(pos_gaps) if pos_gaps else 0,
    }


def summarize_dump(data: dict) -> dict[str, Any]:
    per_track = []
    all_gaps: list[int] = []
    for tr in data.get('tracks', []):
        stats = analyze_track(tr.get('frames', []))
        if not stats:
            continue
        per_track.append(stats)
        frames = sorted(int(f) for f in tr['frames'])
        all_gaps.extend(g for g in _gaps_from_frames(frames) if g > 0)

    if not per_track:
        return {'tracks': 0}

    cov = [t['coverage'] for t in per_track]
    return {
        'tracks': len(per_track),
        'median_coverage': float(sorted(cov)[len(cov) // 2]),
        'mean_coverage': sum(cov) / len(cov),
        'internal_gaps': len(all_gaps),
        'gaps_le_10': sum(1 for g in all_gaps if g <= 10),
        'gaps_gt_10': sum(1 for g in all_gaps if g > 10),
        'max_gap': max(all_gaps) if all_gaps else 0,
        'median_span_s': None,  # filled by caller if fps known
    }


def summarize_diag(data: dict) -> dict[str, Any]:
    ev = data.get('events', [])
    if not ev:
        return {'fragment_ends': 0}
    present = [e for e in ev if e.get('det_at_position')]
    absent = [e for e in ev if not e.get('det_at_position')]
    other = sum(1 for e in present if e.get('outcome') == 'det_other_id')
    cluster = sum(1 for e in present if e.get('in_cluster'))
    raw_sw = sum(1 for e in present if e.get('bytetrack_raw_switch'))
    return {
        'fragment_ends': len(ev),
        'det_present': len(present),
        'det_absent': len(absent),
        'det_other_id': other,
        'in_cluster': cluster,
        'bytetrack_raw_switch': raw_sw,
        'params': data.get('params', {}),
    }


def _print_summary(label: str, s: dict, fps: float | None):
    print(f"\n=== {label} ===")
    if 'tracks' in s and s.get('tracks', 0) == 0:
        print("  (no tracks)")
        return
    if 'fragment_ends' in s:
        print(f"  fragment ends       : {s['fragment_ends']}")
        if s['fragment_ends']:
            n = s['fragment_ends']
            print(f"  det present next f  : {s['det_present']} "
                  f"({100*s['det_present']/n:.1f}%)")
            print(f"  det absent next f   : {s['det_absent']} "
                  f"({100*s['det_absent']/n:.1f}%)")
            print(f"  → other canonical   : {s['det_other_id']}")
            print(f"  → cluster nearby    : {s['in_cluster']}")
            print(f"  → BT raw id switch  : {s['bytetrack_raw_switch']}")
            if s.get('params'):
                p = s['params']
                print(f"  params: match={p.get('TRACK_MATCHING_THRESHOLD')}  "
                      f"lost={p.get('BYTE_TRACK_LOST_SECONDS')}s  "
                      f"activation={p.get('TRACK_ACTIVATION_THRESHOLD')}")
        return

    print(f"  tracks              : {s['tracks']}")
    print(f"  median coverage     : {s['median_coverage']:.1%}")
    print(f"  internal gaps       : {s['internal_gaps']} "
          f"(≤10f: {s['gaps_le_10']}, >10f: {s['gaps_gt_10']})")
    print(f"  max gap (frames)    : {s['max_gap']}")
    if fps and s['tracks']:
        # rough median span from dump tracks not stored — skip unless needed
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dump', help='track_dump_*.json from --track_dump')
    ap.add_argument('--diag', help='track_diag_*.json from --track_diag')
    ap.add_argument('--compare', help='Second track_dump for A/B')
    ap.add_argument('--fps', type=float, default=None,
                    help='FPS (from dump if omitted)')
    args = ap.parse_args()

    if not args.dump and not args.diag:
        ap.error('Provide --dump and/or --diag')

    if args.diag:
        with open(args.diag) as f:
            diag_data = json.load(f)
        _print_summary('track_diag', summarize_diag(diag_data), args.fps)

    if args.dump:
        with open(args.dump) as f:
            dump_a = json.load(f)
        fps = args.fps or dump_a.get('fps')
        sa = summarize_dump(dump_a)
        _print_summary(f"track_dump ({args.dump})", sa, fps)

        if args.compare:
            with open(args.compare) as f:
                dump_b = json.load(f)
            sb = summarize_dump(dump_b)
            _print_summary(f"track_dump ({args.compare})", sb, fps)
            print("\n=== A vs B (coverage / gaps) ===")
            print(f"  median coverage : {sa['median_coverage']:.1%} → "
                  f"{sb['median_coverage']:.1%}")
            print(f"  gaps ≤10f       : {sa['gaps_le_10']} → {sb['gaps_le_10']}")
            print(f"  gaps >10f       : {sa['gaps_gt_10']} → {sb['gaps_gt_10']}")


if __name__ == '__main__':
    main()
