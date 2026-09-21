"""A 2D player/ball map drawn into the corner of the render.

WHY THIS EXISTS
---------------
Reading team shape off the annotated video is hard: players are 60-120px tall in
a 4096px-wide frame, they bunch up, and a ring that flickers off for a second is
invisible among 22 others. The map puts every player in one small panel where a
gap, a swap or a missing player is immediately obvious.

It is a DIAGNOSTIC first and a feature second. If identity is broken, this is
where you will see it.

WHY IT IS NOT A TRUE TOP-DOWN PITCH
-----------------------------------
A real overhead view needs a homography from image space to pitch space, and
this footage cannot supply one: `football-pitch-detection.pt` finds ZERO
landmarks on a dry worn pitch at any confidence down to 0.05, which is why
`render_radar()` / `ViewTransformer` cannot be used here.

So this maps feet positions directly into the bounding box of the pitch polygon.
The result is perspective-distorted — the far half is compressed — but it
preserves left/right, near/far and relative spacing, which is what makes shape
and continuity readable. A four-point manual calibration would straighten it
(the camera is fixed, so it is a one-time job) and would additionally let the
stitcher extrapolate in pitch space, where constant-speed running is linear,
instead of image space, where perspective makes it curve.

GAPS ARE DRAWN DIFFERENTLY ON PURPOSE
-------------------------------------
Positions carried through a tracking gap are HOLLOW; measured positions are
filled. A straight line through a two-second gap is a guess, and a map that
renders guesses identically to measurements invites trusting fabricated data.
Same principle as the stitcher flagging thin links.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Carry a lost player this long before dropping them from the map. Beyond a
# couple of seconds a straight-line guess is worthless — players change
# direction — and a stale dot is worse than an absent one.
MAX_GHOST_SECONDS = 2.5
# Half-width of the moving average applied to each trajectory, in frames.
# Detection boxes jitter by a few px per frame; without this the map looks like
# the players are vibrating, which hides the real motion.
SMOOTH_HALF_WINDOW = 4
TRAIL_SECONDS = 1.5

PANEL_FRACTION = 0.30           # panel width as a fraction of the frame width
# Near-opaque on purpose: at 0.72 the pitch and a sponsor banner showed through
# and the dots became unreadable against bright grass, which defeats the point
# of having a panel you can read at a glance.
PANEL_ALPHA = 0.90

TEAM_COLOURS = [(235, 180, 60), (128, 0, 0)]        # BGR: sky blue, navy (#000080)
REFEREE_COLOUR = (60, 230, 235)
KEEPER_COLOUR = (120, 235, 120)
BALL_COLOUR = (0, 255, 255)


def _smooth(xy: np.ndarray, half: int = SMOOTH_HALF_WINDOW) -> np.ndarray:
    """Moving average along a trajectory, edges handled by clamping."""
    if len(xy) < 3 or half < 1:
        return xy
    pad = np.concatenate([np.repeat(xy[:1], half, axis=0), xy,
                          np.repeat(xy[-1:], half, axis=0)])
    kern = np.ones(2 * half + 1) / (2 * half + 1)
    return np.stack([np.convolve(pad[:, 0], kern, 'valid'),
                     np.convolve(pad[:, 1], kern, 'valid')], axis=1)


def build_timeline(id_history: Dict[int, List[Tuple]], fps: float,
                   id_team: Optional[Dict[int, int]] = None,
                   id_class: Optional[Dict[int, int]] = None,
                   keep_ids: Optional[set] = None) -> Dict[int, List[dict]]:
    """Per-frame player positions, with short gaps filled and paths smoothed.

    This is the payoff of doing the work in a second pass: pass 1 has already
    seen the whole clip, so a player who vanishes at frame 100 and returns at
    frame 160 can be drawn through the gap using BOTH ends, rather than
    extrapolated blindly forward the way an online tracker would have to.

    Returns {frame: [{id, xy, team, cls, ghost}, ...]}.
    """
    id_team = id_team or {}
    id_class = id_class or {}
    max_gap = int(MAX_GHOST_SECONDS * fps)
    timeline: Dict[int, List[dict]] = {}

    for cid, hist in id_history.items():
        if keep_ids is not None and cid not in keep_ids:
            continue
        if len(hist) < 2:
            continue
        hist = sorted(hist)
        frames = np.array([h[0] for h in hist], dtype=int)
        xy = _smooth(np.array([[h[1], h[2]] for h in hist], dtype=np.float32))

        for i in range(len(frames)):
            timeline.setdefault(int(frames[i]), []).append({
                'id': cid, 'xy': xy[i], 'team': id_team.get(cid),
                'cls': id_class.get(cid), 'ghost': False})
            if i + 1 >= len(frames):
                continue
            gap = int(frames[i + 1] - frames[i])
            if gap <= 1 or gap > max_gap:
                continue
            # Interpolate BETWEEN two observations — both endpoints are known,
            # so this is bounded rather than an open-ended extrapolation.
            for g in range(1, gap):
                a = g / gap
                timeline.setdefault(int(frames[i]) + g, []).append({
                    'id': cid, 'xy': xy[i] * (1 - a) + xy[i + 1] * a,
                    'team': id_team.get(cid), 'cls': id_class.get(cid),
                    'ghost': True})
    return timeline


def build_ball_timeline(ball_hist: Sequence[Tuple[int, float, float]],
                        fps: float) -> Dict[int, np.ndarray]:
    """Smoothed ball positions, gaps filled the same way.

    The ball is detected in only a minority of frames — sanity checks remove the
    goalpost but also a legitimately stationary ball — so without filling, the
    marker strobes. Gaps are bridged only between two real observations.
    """
    if len(ball_hist) < 2:
        return {}
    hist = sorted(ball_hist)
    frames = np.array([h[0] for h in hist], dtype=int)
    xy = _smooth(np.array([[h[1], h[2]] for h in hist], dtype=np.float32), 2)
    out: Dict[int, np.ndarray] = {}
    max_gap = int(1.5 * fps)
    for i in range(len(frames)):
        out[int(frames[i])] = xy[i]
        if i + 1 < len(frames):
            gap = int(frames[i + 1] - frames[i])
            if 1 < gap <= max_gap:
                for g in range(1, gap):
                    a = g / gap
                    out[int(frames[i]) + g] = xy[i] * (1 - a) + xy[i + 1] * a
    return out


def bounds_from_timeline(timeline: Dict[int, List[dict]],
                         fallback: Tuple[int, int, int, int],
                         pct: float = 1.0,
                         margin: float = 0.04) -> Tuple[int, int, int, int]:
    """Frame the map on where players actually went.

    The obvious choice — the pitch polygon's bounding box — is wrong. That
    polygon runs to the bottom of the frame because the near touchline sits off
    frame, but players never reach there, so it left the lower half of the panel
    empty and compressed everyone into the top 40%.

    Percentiles rather than min/max, so one stray detection on a spectator does
    not zoom the whole map out.
    """
    pts = np.array([r['xy'] for recs in timeline.values() for r in recs])
    if len(pts) < 20:
        return fallback
    x0, x1 = np.percentile(pts[:, 0], [pct, 100 - pct])
    y0, y1 = np.percentile(pts[:, 1], [pct, 100 - pct])
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return (int(x0 - mx), int(y0 - my), int(x1 + mx), int(y1 + my))


class Minimap:
    """Renders the panel and composites it onto a frame."""

    def __init__(self, bounds: Tuple[int, int, int, int],
                 frame_w: int, frame_h: int, fps: float,
                 corner: str = 'bottom_left'):
        self.x0, self.y0, self.x1, self.y1 = bounds
        self.fps = fps
        self.corner = corner
        span_x = max(1, self.x1 - self.x0)
        span_y = max(1, self.y1 - self.y0)
        self.pw = int(frame_w * PANEL_FRACTION)
        # Keep the mapped region's aspect so spacing is not distorted further
        # than perspective already does.
        self.ph = max(90, int(self.pw * span_y / span_x))
        self.frame_w, self.frame_h = frame_w, frame_h
        self.trail_frames = int(TRAIL_SECONDS * fps)

    def _to_panel(self, xy) -> Tuple[int, int]:
        u = (xy[0] - self.x0) / max(1, self.x1 - self.x0)
        v = (xy[1] - self.y0) / max(1, self.y1 - self.y0)
        return (int(np.clip(u, 0, 1) * (self.pw - 1)),
                int(np.clip(v, 0, 1) * (self.ph - 1)))

    def _colour(self, rec) -> Tuple[int, int, int]:
        if rec.get('cls') == 3:
            return REFEREE_COLOUR
        if rec.get('cls') == 1:
            return KEEPER_COLOUR
        team = rec.get('team')
        return TEAM_COLOURS[team] if team in (0, 1) else (200, 200, 200)

    def draw(self, frame: np.ndarray, records: List[dict],
             ball_xy=None, trails: Optional[Dict[int, List]] = None,
             label: str = '') -> np.ndarray:
        panel = np.full((self.ph, self.pw, 3), 28, np.uint8)
        cv2.rectangle(panel, (0, 0), (self.pw - 1, self.ph - 1), (90, 90, 90), 2)
        cv2.line(panel, (self.pw // 2, 0), (self.pw // 2, self.ph - 1),
                 (70, 70, 70), 1)

        if trails:
            for cid, pts in trails.items():
                if len(pts) < 2:
                    continue
                p = [self._to_panel(q) for q in pts[-self.trail_frames:]]
                for j in range(1, len(p)):
                    cv2.line(panel, p[j - 1], p[j], (95, 95, 95), 1)

        for rec in records:
            cx, cy = self._to_panel(rec['xy'])
            col = self._colour(rec)
            if rec['ghost']:
                # Hollow: this position was inferred across a gap, not measured.
                cv2.circle(panel, (cx, cy), 5, col, 1)
            else:
                cv2.circle(panel, (cx, cy), 5, col, -1)
                cv2.circle(panel, (cx, cy), 5, (20, 20, 20), 1)

        if ball_xy is not None:
            bx, by = self._to_panel(ball_xy)
            cv2.circle(panel, (bx, by), 4, BALL_COLOUR, -1)
            cv2.circle(panel, (bx, by), 8, BALL_COLOUR, 1)

        n_real = sum(1 for r in records if not r['ghost'])
        n_ghost = len(records) - n_real
        cv2.putText(panel, f'{n_real} tracked  {n_ghost} inferred{label}',
                    (8, self.ph - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (215, 215, 215), 1)

        m = 24
        y = self.frame_h - self.ph - m if 'bottom' in self.corner else m
        x = m if 'left' in self.corner else self.frame_w - self.pw - m
        roi = frame[y:y + self.ph, x:x + self.pw]
        if roi.shape[:2] == panel.shape[:2]:
            frame[y:y + self.ph, x:x + self.pw] = cv2.addWeighted(
                panel, PANEL_ALPHA, roi, 1 - PANEL_ALPHA, 0)
        return frame
