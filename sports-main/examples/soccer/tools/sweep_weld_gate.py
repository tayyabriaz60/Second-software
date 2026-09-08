"""Offline sweep of SPLIT_PATH_NET_MIN_DURATION_S against an existing dump.

split_path_net_welds() (main.py) changes what gets cut during PASS 1, so
sweeping its min-duration gate properly needs the fragment set from BEFORE
any path/net cut was applied — not the already-cut dump used for the
ceiling sweep (whose fragments are the CONSEQUENCE of one specific gate
choice, and don't carry the parent/child lineage needed to undo a cut).

That pre-cut dump is cheap to produce once: run main.py with
--no_path_net_split --track_dump (one GPU pass, same as any other dump).
Every downstream duration value is then free — this script reimplements
split_path_net_welds' exact cut algorithm (ceiling + min-duration gate) in
plain numpy against the saved dump, no detection, no GPU, no supervision/
cv2 import. It then feeds the resulting fragments through
assign_identities.assign() (CPU, ceiling=25 — the value the ceiling sweep
already settled) to report identities_over_60s alongside the pass-1 numbers,
the same "clean-and-over-60s" figure used throughout this investigation.

Usage
    python tools/sweep_weld_gate.py --dump track_dump_..._precut.json \
        --durations 10 20 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import assign_identities as ai

WELD_PATH_NET_CEILING = 25.0          # settled by the earlier ceiling sweep
MIN_LATERAL_EXCURSION_PX = 25.0       # same floor main.py's cut search uses


class _Frag:
    """Minimal stand-in for main.py's id_history entries: (frame, x, y, h)."""

    __slots__ = ("id", "frames", "xy", "h", "meta")

    def __init__(self, fid, frames, xy, h, meta):
        self.id = fid
        self.frames = frames
        self.xy = xy
        self.h = h
        self.meta = meta


def _path_net_ratio(xy: np.ndarray) -> float:
    """Verbatim port of main.py's PlayerReIDTracker.path_net_ratio()."""
    path = float(np.sum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))))
    net = float(np.hypot(xy[-1, 0] - xy[0, 0], xy[-1, 1] - xy[0, 1]))
    return path if net < 1.0 else path / net


def _next_id(counter: List[int]) -> int:
    counter[0] += 1
    return counter[0]


def split_path_net_welds(frags: List[_Frag], fps: float, ceiling: float,
                         min_duration_s: float, id_counter: List[int]
                         ) -> Tuple[List[_Frag], int]:
    """Faithful port of main.py's PlayerReIDTracker.split_path_net_welds().

    Operates on a plain fragment list instead of self.id_history, and
    creates new _Frag objects instead of self._apply_track_cuts — same
    cut decision logic, same iteration cap, same lateral-excursion floor.
    """
    out: List[_Frag] = []
    splits = 0
    for frag in frags:
        cur = frag
        for _ in range(10):
            if len(cur.xy) < 2:
                break
            ratio = _path_net_ratio(cur.xy)
            if ratio <= ceiling:
                break
            if len(cur.frames) < 4:
                break
            if (cur.frames[-1] - cur.frames[0]) / fps < min_duration_s:
                break
            pos = cur.xy.astype(float)
            chord = pos[-1] - pos[0]
            chord_len = float(np.linalg.norm(chord)) + 1e-6
            unit = chord / chord_len
            best_k, best_d = 1, -1.0
            for k in range(1, len(pos) - 1):
                rel = pos[k] - pos[0]
                proj = min(max(float(rel @ unit), 0.0), chord_len)
                perp = float(np.linalg.norm(rel - proj * unit))
                if perp > best_d:
                    best_d, best_k = perp, k
            if best_d < MIN_LATERAL_EXCURSION_PX:
                break
            new_id = _next_id(id_counter)
            tail = _Frag(new_id, cur.frames[best_k:], cur.xy[best_k:],
                        cur.h[best_k:], cur.meta)
            cur = _Frag(cur.id, cur.frames[:best_k], cur.xy[:best_k],
                       cur.h[:best_k], cur.meta)
            out.append(tail)
            splits += 1
        out.append(cur)
    return out, splits


def load_dump(path: str):
    data = json.load(open(path))
    fps = float(data.get('fps', 30.0))
    frags, max_id = [], 0
    for rec in data['tracks']:
        frames = np.asarray(rec['frames'], dtype=np.int64)
        if len(frames) < 2:
            continue
        xy = np.asarray(rec['xy'], dtype=np.float32)
        h = np.asarray(rec.get('h') or [0.0] * len(frames), dtype=np.float32)
        frags.append(_Frag(int(rec['id']), frames, xy, h, rec))
        max_id = max(max_id, int(rec['id']))
    return frags, fps, data, max_id


def to_ai_fragments(frags: List[_Frag]) -> List[ai.Fragment]:
    out = []
    for f in frags:
        if len(f.frames) < 2:
            continue
        meta = f.meta
        out.append(ai.Fragment(
            id=int(f.id), frames=f.frames, xy=f.xy, h=f.h,
            cls=int(meta.get('class', 2)),
            team=meta.get('team'), team_conf=meta.get('team_conf'),
            appearance=None,   # appearance is per-original-fragment only;
                               # dropped on a synthetic split piece, same as
                               # main.py's _apply_track_cuts does not carry
                               # a fresh embedding for the new piece either.
            stable=bool(meta.get('stable', False))))
    out.sort(key=lambda fr: fr.start)
    return out


def report(label: str, frags: List[_Frag], fps: float, n_splits: int,
          team_sep: float):
    spans = np.asarray([(f.frames[-1] - f.frames[0]) / fps for f in frags
                        if len(f.frames) >= 2])
    ai_frags = to_ai_fragments(frags)
    chains, refused = ai.assign(ai_frags, fps, team_sep,
                                roster_min=22, roster_max=26,
                                path_net_ceiling=WELD_PATH_NET_CEILING,
                                verbose=False)
    total_frames = int(max(f.end for f in ai_frags)) + 1
    imap = ai.to_identity_map(chains, fps, total_frames, {}, refused=refused,
                              path_net_ceiling=WELD_PATH_NET_CEILING)
    s = imap['summary']
    print(f"{label:>10}: fragments={len(frags):5d}  "
          f"median_span_s={float(np.median(spans)):6.2f}  "
          f"pass1_welds={n_splits:4d}  "
          f"merger_welds={s['physics_refusals']:4d}  "
          f"clean_over_60s={s['identities_over_60s']:4d}")


def main():
    global WELD_PATH_NET_CEILING
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dump', required=True,
                    help='track_dump_*.json produced with --no_path_net_split '
                         '(the PRE-cut fragment set — see module docstring)')
    ap.add_argument('--durations', type=float, nargs='+', default=[10, 20, 30],
                    help='min-duration gate values to sweep, in seconds')
    ap.add_argument('--ceiling', type=float, default=WELD_PATH_NET_CEILING,
                    help='path/net ceiling for BOTH stages (default 25.0, '
                         'the value the earlier sweep settled on)')
    args = ap.parse_args()
    WELD_PATH_NET_CEILING = args.ceiling

    frags0, fps, data, max_id = load_dump(args.dump)
    team_sep = float(data.get('team_sep', 0.0))
    print(f"Loaded {len(frags0)} pre-cut fragments @ {fps:.2f} fps, "
          f"ceiling={WELD_PATH_NET_CEILING}, team_sep={team_sep:.2f}")
    print()

    for min_dur in args.durations:
        id_counter = [max_id]
        cut_frags, n_splits = split_path_net_welds(
            frags0, fps, WELD_PATH_NET_CEILING, min_dur, id_counter)
        report(f"{min_dur:g}s", cut_frags, fps, n_splits, team_sep)


if __name__ == '__main__':
    main()
