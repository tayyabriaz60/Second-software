#!/usr/bin/env python3
"""Recompute track_diag det_at_position with the motion-scaled gate.

Old runs used match_radius_px = max(18, 0.55 * box_h), which labelled normal
inter-frame motion as no_det. New gate (main.py):

  radius_bh = TRACK_DIAG_BASE_BODY_H + MAX_BODY_HEIGHTS_PER_SEC * gap_sec
  radius_px = radius_bh * box_h

Does not re-run tracking — only re-reads saved events.

Usage:
  python tools/reclassify_track_diag_gate.py --diag data/id_lists/track_diag_....json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import constants from main without loading models
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as m  # noqa: E402


def motion_radius_px(box_h: float, frame_gap: int, fps: float) -> float:
    bh = max(float(box_h), float(m.MIN_BOX_HEIGHT_PX))
    gap = max(1, int(frame_gap))
    gap_sec = gap / max(fps, 1e-6)
    radius_bh = m.TRACK_DIAG_BASE_BODY_H + m.MAX_BODY_HEIGHTS_PER_SEC * gap_sec
    return max(float(m.MIN_BOX_HEIGHT_PX), radius_bh * bh)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--diag', required=True, help='track_diag_*.json')
    args = ap.parse_args()

    data = json.loads(Path(args.diag).read_text())
    fps = float(data.get('fps') or 30.0)
    events = data.get('events') or []

    old_present = sum(1 for e in events if e.get('det_at_position'))
    new_present = 0
    dist_bh: list[float] = []

    for e in events:
        dist = e.get('nearest_dist_px')
        box_h = e.get('box_h') or m.MIN_BOX_HEIGHT_PX
        gap = e.get('frame_gap')
        if gap is None:
            gap = max(1, int(e.get('next_frame', 0) - e.get('last_frame', 0)))
        rad = motion_radius_px(box_h, gap, fps)
        if dist is not None and dist <= rad:
            new_present += 1
        if dist is not None and box_h:
            dist_bh.append(float(dist) / max(float(box_h), float(m.MIN_BOX_HEIGHT_PX)))

    n = len(events)
    print(f"File: {args.diag}")
    print(f"Events: {n}  fps={fps:.2f}")
    print(f"Gate: {m.TRACK_DIAG_BASE_BODY_H}bh + "
          f"{m.MAX_BODY_HEIGHTS_PER_SEC}bh/s × gap_sec")
    print(f"det_at_position (stored):     {old_present} ({100*old_present/n:.1f}%)")
    print(f"det_at_position (recomputed): {new_present} ({100*new_present/n:.1f}%)")
    if dist_bh:
        dist_bh.sort()
        mid = dist_bh[len(dist_bh) // 2]
        within3 = sum(1 for d in dist_bh if d <= 3.0)
        within5 = sum(1 for d in dist_bh if d <= 5.0)
        print(f"nearest_dist (body-heights): median={mid:.2f}  "
              f"≤3bh: {100*within3/len(dist_bh):.1f}%  "
              f"≤5bh: {100*within5/len(dist_bh):.1f}%")


if __name__ == '__main__':
    main()
