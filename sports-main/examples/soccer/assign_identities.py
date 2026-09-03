"""Offline fragment -> player identity assignment.

The online tracker (pass 1) leaves each real player as several fragments: on
the 10-min 14_08 clip 268 fragments for ~24 people, median fragment 25 s.
Earlier stages tried to FILTER this (keep 24-40 "stable" ids, call the rest
noise). That throws away data the client needs — a player's 10 minutes are
spread over ~11 fragments, and statistics need all of them on one id.

This module does the opposite: it ASSIGNS every fragment to an identity.

What makes this tractable on a stationary camera with a closed roster:

  * cannot-link — two fragments alive at the same time are never the same
    person. Chaining tail -> head with a positive time gap enforces this for
    free, and it prunes most of the pairwise space.
  * scale-free motion — pixel distance means nothing across the pitch depth,
    so gaps are judged in BODY HEIGHTS per second (box height as the ruler).
    A sprint is ~5.7 bh/s; the reach budget is that plus jitter slack.
  * team colour — hard veto when both fragments are confidently different
    kits under strong separation, soft penalty otherwise.
  * appearance — SigLIP cosine, used as a soft term only (it does not
    separate same-kit players reliably).
  * cardinality — the number of identities should approach the roster. The
    acceptance threshold relaxes progressively while the count is above the
    roster band, but never past physical feasibility, so the algorithm stops
    at "cannot be joined on this evidence" rather than force-welding.

Links are made by repeated global assignment (Hungarian over chain tails x
chain heads). Each accepted link records its cost and margin over the
runner-up, so the output can flag low-confidence joins for human review.

Output: identity_map.json

    {
      "fragment_to_identity": {"1043": 5, ...},
      "identities": {"5": {"fragments": [5, 1043, ...], "team": 0,
                            "duration_s": 512.3, "coverage": 0.85,
                            "links": [{"from": 5, "to": 1043,
                                       "gap_s": 3.2, "cost": 0.31,
                                       "margin": 0.6, "confident": true}]}},
      "summary": {...}, "params": {...}
    }

The same file is the correction interface: edit fragment_to_identity (or use
--assign / --split / --merge on the CLI) and re-render — pass 2 replays pass
1 boxes, so a fix propagates through every frame of the fragment.

Usage
    python assign_identities.py --dump data/id_lists/track_dump_X.json
    python assign_identities.py --dump ... --roster_min 22 --roster_max 26
    python assign_identities.py --dump ... --apply corrections.json
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:                                   # pragma: no cover
    linear_sum_assignment = None

# ---------------------------------------------------------------- params
SPRINT_BH_PER_SEC = 6.0        # ~10 m/s over a 1.75 m body
JITTER_SLACK_BH = 1.5          # detection/box jitter at the two ends
PREDICT_MAX_SECONDS = 0.8      # extrapolate exit velocity at most this far
MAX_GAP_SECONDS = 90.0         # beyond this, position is uninformative
GAP_WEIGHT = 0.30              # cost for using the gap budget
APPEARANCE_WEIGHT = 0.40       # cost per unit of (1 - cosine)
TEAM_SOFT_PENALTY = 0.50
TEAM_CONF_MIN = 0.70           # vote fraction to treat a team label as firm
TEAM_HARD_SEP_MIN = 2.0        # kit separation for the hard veto
SCALE_JUMP_PENALTY = 0.40      # box height ratio far from 1 across the gap
LONG_GAP_FLOOR_S = 8.0         # gaps above this ignore exit velocity entirely
CONFIDENT_MARGIN = 0.35        # abs margin over runner-up for "confident"

# Progressive acceptance thresholds. Pass 1 links only the obvious; later
# passes relax while the identity count is above the roster band.
THRESHOLDS = [0.35, 0.55, 0.75, 0.95, 1.15]


@dataclass
class Fragment:
    id: int
    frames: np.ndarray
    xy: np.ndarray
    h: np.ndarray
    cls: int
    team: Optional[int]
    team_conf: Optional[float]
    appearance: Optional[np.ndarray]
    stable: bool = False

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:
        return int(self.frames[-1])

    def _end_stats(self, tail: bool, fps: float):
        n = max(2, min(len(self.frames), int(0.5 * fps)))
        idx = slice(-n, None) if tail else slice(0, n)
        f = self.frames[idx].astype(float)
        p = self.xy[idx]
        hh = self.h[idx]
        hh = hh[hh > 0]
        h_ref = float(np.median(hh)) if len(hh) else float('nan')
        span = f[-1] - f[0]
        vel = (p[-1] - p[0]) / span if span > 0 else np.zeros(2)
        pos = p[-1] if tail else p[0]
        return pos, vel, h_ref

    def tail(self, fps):
        return self._end_stats(True, fps)

    def head(self, fps):
        return self._end_stats(False, fps)


def load_fragments(path: str) -> Tuple[List[Fragment], float, dict]:
    data = json.load(open(path))
    fps = float(data.get('fps', 30.0))
    frags = []
    for rec in data['tracks']:
        frames = np.asarray(rec['frames'], dtype=np.int64)
        if len(frames) < 2:
            continue
        xy = np.asarray(rec['xy'], dtype=np.float32)
        h = rec.get('h')
        h = (np.asarray(h, dtype=np.float32) if h is not None
             else np.zeros(len(frames), dtype=np.float32))
        app = rec.get('appearance')
        frags.append(Fragment(
            id=int(rec['id']), frames=frames, xy=xy, h=h,
            cls=int(rec.get('class', 2)),
            team=rec.get('team'), team_conf=rec.get('team_conf'),
            appearance=None if app is None else np.asarray(app, np.float32),
            stable=bool(rec.get('stable', False))))
    frags.sort(key=lambda f: f.start)
    return frags, fps, data


def fit_height_model(frags: List[Fragment]):
    """Box height as a function of image y, for dumps missing 'h'.

    On a stationary camera apparent height grows roughly linearly with y
    (distance from the horizon). Fit h = a*y + b on samples that have h; if
    none do, fall back to a flat 40 px ruler, which keeps costs monotone even
    if not scale-correct.
    """
    ys, hs = [], []
    for f in frags:
        m = f.h > 0
        if m.any():
            ys.append(f.xy[m, 1])
            hs.append(f.h[m])
    if not ys:
        return lambda y: np.full_like(np.asarray(y, float), 40.0)
    y = np.concatenate(ys).astype(float)
    h = np.concatenate(hs).astype(float)
    a, b = np.polyfit(y, h, 1)
    return lambda yy: np.maximum(8.0, a * np.asarray(yy, float) + b)


def _cos(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not na or not nb:
        return None
    return float(a @ b / (na * nb))


def link_cost(a: Fragment, b: Fragment, fps: float, hmodel, team_sep: float,
              max_gap_s: float = MAX_GAP_SECONDS) -> Optional[float]:
    """Cost that fragment b is the continuation of fragment a; None = impossible."""
    gap = b.start - a.end
    if gap <= 0:
        return None
    gap_s = gap / fps
    if gap_s > max_gap_s:
        return None

    pa, va, ha = a.tail(fps)
    pb, vb, hb = b.head(fps)
    if not np.isfinite(ha):
        ha = float(hmodel(pa[1]))
    if not np.isfinite(hb):
        hb = float(hmodel(pb[1]))
    h_ref = max(8.0, 0.5 * (ha + hb))

    # Predict a short way along the exit velocity, then hold position: after a
    # second or two a player's direction says nothing about where they went.
    if gap_s <= LONG_GAP_FLOOR_S:
        t_pred = min(gap, PREDICT_MAX_SECONDS * fps)
        pred = pa + va * t_pred
    else:
        pred = pa
    dist_bh = float(np.linalg.norm(pb - pred)) / h_ref
    reach = SPRINT_BH_PER_SEC * gap_s + JITTER_SLACK_BH
    if dist_bh > reach:
        return None

    cost = dist_bh / reach
    cost += GAP_WEIGHT * (gap_s / max_gap_s)

    # A player does not change apparent size much over a short gap; a big
    # jump means the two boxes sit at very different depths.
    ratio = ha / hb if hb else 1.0
    if ratio < 1 / 1.6 or ratio > 1.6:
        cost += SCALE_JUMP_PENALTY * (1.0 if gap_s < 5 else 0.5)

    # Team: hard veto only when both are firm under strong kit separation.
    if a.team is not None and b.team is not None and a.team != b.team:
        firm = ((a.team_conf or 0) >= TEAM_CONF_MIN and
                (b.team_conf or 0) >= TEAM_CONF_MIN)
        if firm and team_sep >= TEAM_HARD_SEP_MIN:
            return None
        cost += TEAM_SOFT_PENALTY

    sim = _cos(a.appearance, b.appearance)
    if sim is not None:
        cost += APPEARANCE_WEIGHT * (1.0 - sim)
    return cost


@dataclass
class Chain:
    frags: List[Fragment] = field(default_factory=list)
    links: List[dict] = field(default_factory=list)

    @property
    def head(self) -> Fragment:
        return self.frags[0]

    @property
    def tail(self) -> Fragment:
        return self.frags[-1]


def _assign_round(chains: List[Chain], fps, hmodel, team_sep, threshold,
                  max_gap_s) -> int:
    """One Hungarian round over chain tails x chain heads. Returns #links."""
    n = len(chains)
    BIG = 1e6
    C = np.full((n, n), BIG)
    for i, ca in enumerate(chains):
        for j, cb in enumerate(chains):
            if i == j:
                continue
            c = link_cost(ca.tail, cb.head, fps, hmodel, team_sep, max_gap_s)
            if c is not None:
                C[i, j] = c
    if not np.isfinite(C[C < BIG]).any():
        return 0
    if linear_sum_assignment is None:
        raise RuntimeError("scipy is required: pip install scipy")
    rows, cols = linear_sum_assignment(C)
    accepted = []
    for i, j in zip(rows, cols):
        c = C[i, j]
        if c >= threshold:
            continue
        # Margin over the runner-up in both the row and the column.
        row = np.delete(C[i], j)
        col = np.delete(C[:, j], i)
        alt = min(row.min() if len(row) else BIG,
                  col.min() if len(col) else BIG)
        margin = float(min(alt, BIG) - c) if alt < BIG else 10.0
        accepted.append((c, margin, i, j))

    # Merge greedily by cost; a chain can gain one predecessor and one
    # successor per round, so re-resolve indices as chains fuse.
    accepted.sort()
    parent = list(range(n))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used_tail, used_head = set(), set()
    merges = []
    for c, margin, i, j in accepted:
        if i in used_tail or j in used_head:
            continue
        ri, rj = root(i), root(j)
        if ri == rj:
            continue
        merges.append((i, j, c, margin))
        used_tail.add(i)
        used_head.add(j)
        parent[rj] = ri

    if not merges:
        return 0
    # Build merged chains in time order.
    groups: Dict[int, List[int]] = {}
    for k in range(n):
        groups.setdefault(root(k), []).append(k)
    new_chains = []
    link_by_pair = {(i, j): (c, m) for i, j, c, m in merges}
    for members in groups.values():
        members.sort(key=lambda k: chains[k].head.start)
        ch = Chain()
        for k in members:
            ch.frags.extend(chains[k].frags)
            ch.links.extend(chains[k].links)
        for a, b in zip(members[:-1], members[1:]):
            c, m = link_by_pair.get((a, b), (None, None))
            ch.links.append({
                'from': chains[a].tail.id, 'to': chains[b].head.id,
                'gap_s': round((chains[b].head.start - chains[a].tail.end)
                               / fps, 2),
                'cost': None if c is None else round(float(c), 3),
                'margin': None if m is None else round(float(m), 3),
                'confident': bool(m is not None and m >= CONFIDENT_MARGIN),
                'threshold': threshold,
            })
        new_chains.append(ch)
    chains[:] = new_chains
    return len(merges)


def assign(frags: List[Fragment], fps: float, team_sep: float,
           roster_min: int = 22, roster_max: int = 26,
           max_gap_s: float = MAX_GAP_SECONDS,
           thresholds=THRESHOLDS, verbose=True) -> List[Chain]:
    hmodel = fit_height_model(frags)
    chains = [Chain([f]) for f in frags]
    for t in thresholds:
        rounds = 0
        while True:
            n_links = _assign_round(chains, fps, hmodel, team_sep, t, max_gap_s)
            rounds += 1
            if n_links == 0 or rounds > 50:
                break
        if verbose:
            print(f"  threshold {t:.2f}: {len(chains)} identities")
        if len(chains) <= roster_max:
            break
    chains.sort(key=lambda c: -sum(len(f.frames) for f in c.frags))
    return chains


def summarise(chains: List[Chain], fps: float, total_frames: int) -> dict:
    durs = []
    covered = 0
    n_frag = 0
    low_conf = 0
    for ch in chains:
        fr = np.concatenate([f.frames for f in ch.frags])
        durs.append(len(np.unique(fr)) / fps)
        covered += len(np.unique(fr))
        n_frag += len(ch.frags)
        low_conf += sum(1 for l in ch.links if not l['confident'])
    durs = np.asarray(durs)
    return {
        'fragments': n_frag,
        'identities': len(chains),
        'median_identity_s': round(float(np.median(durs)), 1) if len(durs) else 0,
        'identities_over_60s': int((durs >= 60).sum()),
        'identities_over_300s': int((durs >= 300).sum()),
        'low_confidence_links': low_conf,
        'total_links': sum(len(c.links) for c in chains),
        'clip_seconds': round(total_frames / fps, 1),
    }


def to_identity_map(chains: List[Chain], fps: float, total_frames: int,
                    params: dict) -> dict:
    f2i, ids = {}, {}
    for pid, ch in enumerate(chains, start=1):
        fr = np.concatenate([f.frames for f in ch.frags])
        teams = [f.team for f in ch.frags if f.team is not None]
        team = (max(set(teams), key=teams.count) if teams else None)
        ids[str(pid)] = {
            'fragments': [f.id for f in ch.frags],
            'team': team,
            'first_frame': int(fr.min()), 'last_frame': int(fr.max()),
            'duration_s': round(len(np.unique(fr)) / fps, 1),
            'coverage': round(len(np.unique(fr)) / max(total_frames, 1), 3),
            'links': ch.links,
            'player_name': None,
        }
        for f in ch.frags:
            f2i[str(f.id)] = pid
    return {
        'fragment_to_identity': f2i,
        'identities': ids,
        'summary': summarise(chains, fps, total_frames),
        'params': params,
    }


def apply_corrections(imap: dict, corrections: dict) -> dict:
    """Human fixes, propagated at fragment granularity.

    corrections = {"assign": {"1043": 5}, "merge": [[5, 17]],
                   "split": {"5": [1043]}}   # split = move these fragments out
    """
    f2i = {k: int(v) for k, v in imap['fragment_to_identity'].items()}
    for frag, pid in corrections.get('assign', {}).items():
        f2i[str(frag)] = int(pid)
    for keep, drop in corrections.get('merge', []):
        for k, v in f2i.items():
            if v == int(drop):
                f2i[k] = int(keep)
    next_id = max(f2i.values(), default=0) + 1
    for _, frags in corrections.get('split', {}).items():
        for frag in frags:
            f2i[str(frag)] = next_id
        next_id += 1
    imap['fragment_to_identity'] = f2i
    # Rebuild identities from the mapping (links no longer meaningful).
    ids: Dict[str, dict] = {}
    for frag, pid in f2i.items():
        ids.setdefault(str(pid), {'fragments': [], 'links': [],
                                  'player_name': None})
        ids[str(pid)]['fragments'].append(int(frag))
    for pid, rec in ids.items():
        old = imap['identities'].get(pid, {})
        rec['team'] = old.get('team')
        rec['player_name'] = old.get('player_name')
    imap['identities'] = ids
    imap['corrections_applied'] = corrections
    return imap


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dump', required=True, help='track_dump_*.json from main.py --track_dump')
    ap.add_argument('--out', default=None, help='identity_map.json path')
    ap.add_argument('--roster_min', type=int, default=22)
    ap.add_argument('--roster_max', type=int, default=26)
    ap.add_argument('--max_gap_s', type=float, default=MAX_GAP_SECONDS)
    ap.add_argument('--apply', default=None,
                    help='corrections JSON to apply on top of the automatic map')
    args = ap.parse_args()

    frags, fps, data = load_fragments(args.dump)
    total_frames = int(max(f.end for f in frags)) + 1
    team_sep = float(data.get('team_sep', 0.0))
    print(f"Loaded {len(frags)} fragments @ {fps:.2f} fps, "
          f"{total_frames / fps:.0f}s of footage, team_sep={team_sep:.2f}, "
          f"heights={'yes' if any((f.h > 0).any() for f in frags) else 'NO (y-model fallback)'}, "
          f"appearance={'yes' if any(f.appearance is not None for f in frags) else 'no'}")
    params = dict(roster_min=args.roster_min, roster_max=args.roster_max,
                  max_gap_s=args.max_gap_s, thresholds=THRESHOLDS,
                  sprint_bh_per_sec=SPRINT_BH_PER_SEC)
    chains = assign(frags, fps, team_sep, args.roster_min, args.roster_max,
                    args.max_gap_s)
    imap = to_identity_map(chains, fps, total_frames, params)
    if args.apply:
        imap = apply_corrections(imap, json.load(open(args.apply)))
    s = imap['summary']
    print("\nAssignment summary")
    for k, v in s.items():
        print(f"  {k:24s}: {v}")
    print("\nTop identities (duration s / fragments / low-conf links):")
    for pid, rec in list(imap['identities'].items())[:30]:
        lc = sum(1 for l in rec.get('links', []) if not l.get('confident'))
        print(f"  #{pid:>3} {rec.get('duration_s', 0):7.1f}s  "
              f"{len(rec['fragments']):3d} frags  {lc} low-conf  team={rec.get('team')}")
    out = args.out or os.path.join(
        os.path.dirname(args.dump),
        os.path.basename(args.dump).replace('track_dump_', 'identity_map_'))
    with open(out, 'w') as f:
        json.dump(imap, f, indent=1)
    print(f"\nIdentity map saved to: {out}")


if __name__ == '__main__':
    main()
