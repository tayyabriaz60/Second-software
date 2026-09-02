"""Offline tracklet stitching — turn fragments into player identities.

The problem this solves: ByteTrack is an ONLINE tracker. It decides, frame by
frame, whether a detection continues a track, with no knowledge of what happens
next. But we process recorded video, so the whole match is available at once.

Three things an online tracker cannot use, and this can:

  1. THE FUTURE. A fragment that ends at (x,y) moving right, and another that
     begins 40 frames later further right, are almost certainly one player. The
     online tracker had to decide before seeing the second fragment.
  2. THE ROSTER. There are ~22 players and 2 goalkeepers. Fragments compete for
     a fixed number of slots rather than multiplying without limit.
  3. GAPS ARE SHORT. A player lost for two seconds moves predictably. Once two
     fragments are linked, the positions between them can be interpolated —
     and for distance covered and heatmaps, interpolated positions are fine.

WHY THIS WAS REWRITTEN
----------------------
The first version was GREEDY: walk tracklets in time order, take the cheapest
available link, and refuse when the second-best candidate was nearly as good.
It refused 64% of candidates. The obvious diagnosis was that fragments were too
short to extrapolate from, so we improved the tracker and fed it a third fewer,
longer fragments. The refusal rate moved from 64% to 61%.

That result is what motivated this file. Ambiguity here is STRUCTURAL, not a
consequence of fragmentation: with 22 densely packed players, almost every gap
has several candidates inside the plausible-travel radius, however long the
surrounding fragments are. No amount of upstream work removes that.

What breaks the tie is not a better local score but a GLOBAL one. Greedy asks
"what is the best continuation of this fragment?" in isolation. The right
question is "what assignment of all fragments to all continuations is cheapest
overall?" — because claiming A->B also means B is unavailable to C, and that
knock-on constraint is exactly the information greedy discards. A pairing that
looks marginally better locally is often rejected globally, since it forces an
expensive assignment elsewhere.

That is a linear assignment problem, solved exactly (Jonker-Volgenant, via
scipy) rather than approximately. Cycles are impossible by construction because
a link requires the successor to start strictly after the predecessor ends.

What this deliberately does NOT do: guess silently. Every link carries a MARGIN
— how much worse the next-best alternative was. Links with a thin margin are
reported for review rather than being dropped, so the manual validation pass
spends its attention where the evidence is weakest instead of on everything.

Input is the per-run track record written by main.py (see --track_dump).
"""
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# A link is considered only if the time gap is short enough that a player
# could not have gone far, and the predicted position lands near the candidate.
MAX_GAP_SECONDS = 6.0
# Tolerance on the predicted position, as a multiple of how far the player
# could travel in the gap at a sprint (~8 m/s, ~26 px/m on ultrawide footage).
PREDICT_TOLERANCE = 1.6
SPRINT_PX_PER_SEC = 210.0

# The price of ending a chain instead of linking it to something. This is the
# single most important knob: every candidate link whose cost exceeds it is
# declined, so it sets the trade between leaving a player fragmented and
# stitching two different players together.
#
# Set from measurement, not feel. Ground truth was manufactured by cutting real
# tracklets in half with a known gap and leaving every other tracklet in as a
# distractor (see the eval harness). At a 1s gap, 84 known pairs:
#
#   no_link  links  recall  precision
#     0.10     19     17%      100%
#     0.20     60     48%       91%
#     0.30    105     56%       81%
#     0.55    173     58%       68%
#     0.90    194     62%       66%
#
# READ THIS BEFORE TRUSTING THOSE NUMBERS. Manufactured cuts are EASIER than
# real breaks. A cut lands at an arbitrary point where motion continues
# smoothly; a real fragment breaks precisely because something hard happened —
# an occlusion, two players crossing — so the motion model is least reliable
# exactly where it is needed. The cost distributions show the gap plainly:
#
#   manufactured true pairs   median cost 0.20
#   real candidate links      median cost 0.70
#
# So a threshold tuned on cuts transfers badly. At 0.20 the real 36s clip got
# 14 links from 178 tracklets — a no-op, worse than the old greedy stitcher.
# 0.30 is the compromise actually shipped: 44 links, 134 identities, a third of
# them flagged thin for review. Comparable volume to greedy, but with per-link
# confidence and without greedy's collapse when pushed.
#
# Deliberately biased towards precision. The two failure modes are NOT
# symmetric: an unlinked fragment is visibly incomplete and the manual pass can
# join it, whereas a wrong link welds two players into one identity, looks
# perfectly normal, and silently corrupts every statistic derived from it.
NO_LINK_COST = 0.30

# A link whose next-best alternative was within this fraction of its own cost
# is thin evidence. Global assignment still proposes it, but by default the
# pipeline DECLINES thin links (see keep_thin / STITCH_KEEP_THIN) because
# measured correctness collapses:
#
#   confident       93% correct
#   thin (flagged)  29% correct
#
# The margin over the runner-up predicts correctness far better than the
# absolute cost does. Thin links remain in the returned `links` list for
# audit even when not applied.
THIN_MARGIN_RATIO = 0.85


class Tracklet:
    """One continuous run of a single canonical id."""

    def __init__(self, tid: int, frames: List[int], xy: np.ndarray, cls: int,
                 team: Optional[int] = None,
                 team_conf: Optional[float] = None,
                 team_sep: float = 0.0,
                 appearance=None):
        self.id = tid
        self.frames = frames
        self.xy = xy
        self.cls = cls
        # Optional team label. When both ends of a candidate link know their
        # team with high confidence and disagree under a strong kit separation,
        # the link is HARD-VETOED. Weaker evidence only adds a soft penalty —
        # see team_colour module docstring / TEAM_HARD_SEP_MIN.
        self.team = team
        self.team_conf = team_conf
        self.team_sep = team_sep
        self.appearance = appearance
        self.start, self.end = frames[0], frames[-1]

    @property
    def duration(self) -> int:
        return self.end - self.start + 1

    def exit_velocity(self, n: int = 8) -> np.ndarray:
        """Direction and speed as the tracklet ends, in px/frame."""
        if len(self.xy) < 2:
            return np.zeros(2)
        k = min(n, len(self.xy) - 1)
        span = self.frames[-1] - self.frames[-1 - k]
        if span <= 0:
            return np.zeros(2)
        return (self.xy[-1] - self.xy[-1 - k]) / span

    def predict(self, frame: int) -> np.ndarray:
        """Where this player would be at `frame`, extrapolating from the end."""
        return self.xy[-1] + self.exit_velocity() * (frame - self.end)


def load_tracklets(path: str) -> Tuple[List[Tracklet], float]:
    """Read main.py's track dump into tracklets, splitting on long gaps.

    A canonical id can already contain gaps if re-identification linked it back
    together, so split those into separate tracklets — this stage decides links
    on its own evidence rather than inheriting the online tracker's guesses.
    """
    data = json.load(open(path))
    fps = data.get('fps', 30.0)
    out = []
    for rec in data['tracks']:
        frames = rec['frames']
        xy = np.array(rec['xy'], dtype=np.float32)
        if len(frames) < 2:
            continue
        breaks = [0]
        for i in range(1, len(frames)):
            if frames[i] - frames[i - 1] > fps:      # >1s internal gap
                breaks.append(i)
        breaks.append(len(frames))
        for a, b in zip(breaks[:-1], breaks[1:]):
            if b - a >= 2:
                out.append(Tracklet(rec['id'], frames[a:b], xy[a:b],
                                    rec.get('class', 2)))
    return out, fps


def _cosine(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not na or not nb:
        return None
    return float(a @ b / (na * nb))


def link_cost(a: Tracklet, b: Tracklet, fps: float,
              max_gap_seconds: float = MAX_GAP_SECONDS,
              max_gap_frames: Optional[int] = None,
              appearance_min_cosine: Optional[float] = None) -> Optional[float]:
    """Cost of claiming tracklet b continues tracklet a, or None if impossible."""
    gap = b.start - a.end
    if gap <= 0:                                  # overlapping in time
        return None                               # cannot be the same player
    if max_gap_frames is not None and gap > max_gap_frames:
        return None
    gap_s = gap / fps
    if gap_s > max_gap_seconds:
        return None
    reach = SPRINT_PX_PER_SEC * gap_s * PREDICT_TOLERANCE
    dist = float(np.linalg.norm(b.xy[0] - a.predict(b.start)))
    if dist > reach:
        return None
    # Normalise so cost is comparable across different gap lengths: a link that
    # uses most of the plausible travel budget is worse than one that barely
    # moves, regardless of how long the gap was.
    cost = dist / max(reach, 1e-6) + 0.25 * (gap_s / max_gap_seconds)
    sim = _cosine(a.appearance, b.appearance)
    if appearance_min_cosine is not None and sim is not None:
        if sim < appearance_min_cosine:
            return None
        # Reward high cosine (lower cost).
        cost *= (1.0 - 0.35 * sim)
    # Team disagreement: HARD veto when both ends are confident and kit
    # separation is strong; otherwise a soft penalty (see team_colour).
    try:
        from team_colour import team_penalty, team_hard_conflict
        if team_hard_conflict(a.team, b.team, a.team_conf, b.team_conf,
                              max(a.team_sep, b.team_sep)):
            return None
        cost += team_penalty(a.team, b.team)
    except ImportError:
        pass
    return cost


def build_cost_matrix(tracklets: Sequence[Tracklet], fps: float,
                      max_gap_seconds: float = MAX_GAP_SECONDS,
                      max_gap_frames: Optional[int] = None,
                      appearance_min_cosine: Optional[float] = None,
                      sim_percentile: Optional[float] = None,
                      cost_percentile: Optional[float] = None) -> np.ndarray:
    """Dense cost of every ordered pair; np.inf where a link is impossible.

    When sim_percentile / cost_percentile are set (mot_sota_v6), only links in
    the top appearance band and best motion-cost band survive — everyone else
    is zeroed to inf before assignment.
    """
    n = len(tracklets)
    cost = np.full((n, n), np.inf, dtype=np.float64)
    sims = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = link_cost(tracklets[i], tracklets[j], fps, max_gap_seconds,
                          max_gap_frames=max_gap_frames,
                          appearance_min_cosine=appearance_min_cosine)
            if c is not None:
                cost[i, j] = c
                sims[i, j] = _cosine(tracklets[i].appearance,
                                     tracklets[j].appearance)

    finite = np.isfinite(cost)
    if not np.any(finite):
        return cost

    if sim_percentile is not None:
        sim_vals = sims[finite & np.isfinite(sims)]
        if len(sim_vals) >= 4:
            sim_floor = float(np.percentile(sim_vals, sim_percentile))
            # Only cull scored pairs below the band; unscored (no emb) keep
            # motion evidence for the cost-percentile stage.
            kill = finite & np.isfinite(sims) & (sims < sim_floor)
            cost[kill] = np.inf
            finite = np.isfinite(cost)

    if cost_percentile is not None and np.any(finite):
        cost_vals = cost[finite]
        if len(cost_vals) >= 4:
            cost_ceil = float(np.percentile(cost_vals, cost_percentile))
            kill = finite & (cost > cost_ceil)
            cost[kill] = np.inf

    return cost


def stitch_global(tracklets: Sequence[Tracklet], fps: float,
                  no_link_cost: float = NO_LINK_COST,
                  max_gap_seconds: float = MAX_GAP_SECONDS,
                  cost: Optional[np.ndarray] = None,
                  keep_thin: bool = True,
                  max_gap_frames: Optional[int] = None,
                  appearance_min_cosine: Optional[float] = None,
                  sim_percentile: Optional[float] = None,
                  cost_percentile: Optional[float] = None):
    """Assign successors globally: the cheapest consistent set of links.

    Formulated as a rectangular assignment problem. Rows are tracklets choosing
    a successor; the first n columns are "the successor is tracklet j", and n
    further columns are per-row slack meaning "this chain ends here", priced at
    `no_link_cost`. The solver therefore declines any link that costs more than
    ending the chain, and — the point of doing it globally — will decline a
    locally attractive link when taking it would force a worse assignment
    somewhere else.

    keep_thin: when False, thin-margin links are reported but NOT applied
    (mot_sota_v3: thin links ~29% correct vs ~93% confident). Prefer leaving
    fragments over silent wrong welds.

    Returns (identities, links), where identities are chains of tracklet
    indices and links carry the margin over the next-best alternative.
    """
    n = len(tracklets)
    if n == 0:
        return [], []
    if cost is None:
        cost = build_cost_matrix(
            tracklets, fps, max_gap_seconds,
            max_gap_frames=max_gap_frames,
            appearance_min_cosine=appearance_min_cosine,
            sim_percentile=sim_percentile,
            cost_percentile=cost_percentile)

    # linear_sum_assignment cannot take inf, so price impossible links above
    # any slack column — they will never be chosen.
    big = no_link_cost * 1000.0
    block = np.where(np.isinf(cost), big, cost)
    slack = np.full((n, n), big, dtype=np.float64)
    np.fill_diagonal(slack, no_link_cost)
    rows, cols = linear_sum_assignment(np.hstack([block, slack]))

    successor: Dict[int, int] = {}
    links = []
    for i, j in zip(rows, cols):
        if j >= n or not np.isfinite(cost[i, j]):
            continue                              # chain ends here
        # Margin: how much worse the runner-up was. Compare against the best
        # alternative FOR THIS ROW and the best alternative claim ON THIS
        # COLUMN — a link is only well-evidenced if neither side had a close
        # second choice.
        row_alt = np.min(np.delete(cost[i], j))
        col_alt = np.min(np.delete(cost[:, j], i))
        alt = min(row_alt, col_alt)
        thin = np.isfinite(alt) and cost[i, j] >= alt * THIN_MARGIN_RATIO
        links.append({'from': tracklets[i].id, 'to': tracklets[j].id,
                      'cost': float(cost[i, j]),
                      'runner_up': float(alt) if np.isfinite(alt) else None,
                      'thin': bool(thin)})
        if thin and not keep_thin:
            continue                              # decline; leave fragmented
        successor[i] = j

    claimed = set(successor.values())
    identities = []
    for s in range(n):
        if s in claimed:
            continue
        chain, cur = [], s
        while cur is not None:
            chain.append(cur)
            cur = successor.get(cur)
        identities.append(chain)
    return identities, links


def stitch_to_roster(tracklets: Sequence[Tracklet], fps: float, roster: int,
                     max_gap_seconds: float = MAX_GAP_SECONDS,
                     tol: float = 0.01):
    """Pick the no-link price that yields about `roster` identities.

    We know how many people are on the pitch, which is a stronger constraint
    than any cost threshold chosen by feel. Raising the price of ending a chain
    monotonically reduces the number of chains, so bisect on it.

    Treat the result as a target, not a guarantee: if the evidence does not
    support that many links they will not be made at any price, and forcing the
    count would just invent merges. Check the returned count.

    USE WITH CARE, and check how many links come back thin. Driving 178 real
    tracklets down to 22 identities needed 128 links of which 104 were thin —
    and thin links measured 29% correct. Landing on the right NUMBER of players
    says nothing about them being the right players. Prefer the measured
    NO_LINK_COST default unless you have a reason not to.
    """
    cost = build_cost_matrix(tracklets, fps, max_gap_seconds)
    lo, hi = 0.0, 2.0
    best = None
    for _ in range(40):
        mid = (lo + hi) / 2
        ident, links = stitch_global(tracklets, fps, mid, max_gap_seconds, cost)
        best = (ident, links, mid)
        if len(ident) > roster:
            lo = mid                              # too fragmented, link more
        else:
            hi = mid
        if hi - lo < tol:
            break
    return best


def stitch_greedy(tracklets: Sequence[Tracklet], fps: float,
                  ambiguity_ratio: float = 0.75):
    """The original greedy stitcher. Superseded by stitch_global.

    Kept because it is the baseline the global version is measured against —
    on 36s of the 14_09 stationary it refused 61-64% of candidates regardless
    of how good its input was, which is what proved local scoring insufficient.
    """
    order = sorted(range(len(tracklets)), key=lambda i: tracklets[i].start)
    successor: Dict[int, int] = {}
    claimed = set()
    ambiguous = []

    for i in order:
        cands = []
        for j in order:
            if j == i or j in claimed:
                continue
            c = link_cost(tracklets[i], tracklets[j], fps)
            if c is not None:
                cands.append((c, j))
        if not cands:
            continue
        cands.sort()
        best_c, best_j = cands[0]
        if len(cands) > 1 and cands[1][0] * ambiguity_ratio <= best_c:
            ambiguous.append((tracklets[i].id, tracklets[best_j].id,
                              tracklets[cands[1][1]].id))
            continue
        successor[i] = best_j
        claimed.add(best_j)

    starts = [i for i in order if i not in claimed]
    identities = []
    for s in starts:
        chain, cur = [], s
        while cur is not None:
            chain.append(cur)
            cur = successor.get(cur)
        identities.append(chain)
    return identities, ambiguous


def interpolate(chain: List[Tracklet]) -> Dict:
    """Fill gaps between linked tracklets with straight-line motion."""
    frames, xy, filled = [], [], 0
    for k, t in enumerate(chain):
        if k > 0:
            prev = chain[k - 1]
            gap = t.start - prev.end
            if gap > 1:
                for g in range(1, gap):
                    a = g / gap
                    frames.append(prev.end + g)
                    xy.append(prev.xy[-1] * (1 - a) + t.xy[0] * a)
                    filled += 1
        frames += list(t.frames)
        xy += list(t.xy)
    return {'frames': frames, 'xy': np.array(xy), 'interpolated': filled}
