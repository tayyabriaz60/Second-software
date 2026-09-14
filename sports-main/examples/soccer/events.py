"""Set-piece heuristics from ball track — fully offline, no goal-line calibration.

Ported from repo-root analyse.py zone logic. Uses ball stillness + pitch zones;
does not emit goals because the goal line is not calibrated on this camera.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

# Pitch zones as % of frame (stationary wide angle). Slightly wider than analyse.py
# on this panoramic — corners at y≈12% were missing the old 10% cut.
CORNER_X_PCT = 10.0
CORNER_Y_PCT = 15.0
GOAL_AREA_X_PCT = 12.0
GOAL_AREA_Y_PCT = (30.0, 70.0)
PENALTY_X_PCT = 18.0
PENALTY_Y_PCT = (20.0, 80.0)
# Touchline band for generic free_kick only — corners/goal_kicks checked first.
TOUCHLINE_Y_PCT = 15.0

# Ball stillness — proxy for a dead ball at a set piece
STILL_DIST_PX = 50.0
STILL_MAX_GAP_S = 2.5
STILL_MIN_DURATION_S = 2.5
DEDUP_WINDOW_S = 12.0
DEDUP_SPATIAL_PX = 120.0
MIN_BALL_CONF = 0.30

# Furniture signature — applied to free_kick only (corners/goal_kicks ARE at edge)
BALL_CELL_PX = 64
FURNITURE_MIN_FRAC = 0.01
FURNITURE_MAX_SPREAD_PX = 5.0
FURNITURE_EDGE_MARGIN_PX = 80.0

GOALS_NOTE = (
    "goals omitted — goal line not calibrated for this camera; "
    "ball zone heuristics cannot reliably detect scoring"
)


def px_to_pct(x_px: float, y_px: float, width: int, height: int) -> Tuple[float, float]:
    return (x_px / max(width, 1)) * 100.0, (y_px / max(height, 1)) * 100.0


def _at_pitch_edge(x_px: float, y_px: float, pitch_polygon,
                   edge_margin: float = FURNITURE_EDGE_MARGIN_PX) -> bool:
    if pitch_polygon is None:
        return False
    dist = cv2.pointPolygonTest(
        pitch_polygon, (float(x_px), float(y_px)), True)
    return dist < 0 or dist < edge_margin


def furniture_cells(
    ball_history: Sequence[Tuple],
    pitch_polygon=None,
    cell_px: int = BALL_CELL_PX,
    min_frac: float = FURNITURE_MIN_FRAC,
    max_spread_px: float = FURNITURE_MAX_SPREAD_PX,
    edge_margin: float = FURNITURE_EDGE_MARGIN_PX,
) -> Set[Tuple[int, int]]:
    """Fine cells that look like fixed pitch-edge objects, not set-piece stillness."""
    if not ball_history:
        return set()

    cell_pts: Dict[Tuple[int, int], list] = defaultdict(list)
    for rec in ball_history:
        x_px, y_px = float(rec[2]), float(rec[3])
        cell_pts[(int(x_px) // cell_px, int(y_px) // cell_px)].append((x_px, y_px))

    n_total = len(ball_history)
    min_hits = max(15, int(min_frac * n_total))
    banned: Set[Tuple[int, int]] = set()

    for cell, pts in cell_pts.items():
        if len(pts) < min_hits:
            continue
        xs, ys = zip(*pts)
        if (float(np.std(xs)) > max_spread_px
                or float(np.std(ys)) > max_spread_px):
            continue
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        if not _at_pitch_edge(cx, cy, pitch_polygon, edge_margin):
            continue
        banned.add(cell)
    return banned


def classify_zone(x_pct: float, y_pct: float) -> Optional[str]:
    """Map ball location to set-piece type; None if not a plausible event zone."""
    in_corner = (
        (y_pct < CORNER_Y_PCT or y_pct > (100.0 - CORNER_Y_PCT))
        and (x_pct < CORNER_X_PCT or x_pct > (100.0 - CORNER_X_PCT))
    )
    in_left_goal = (
        x_pct < GOAL_AREA_X_PCT
        and GOAL_AREA_Y_PCT[0] < y_pct < GOAL_AREA_Y_PCT[1]
    )
    in_right_goal = (
        x_pct > (100.0 - GOAL_AREA_X_PCT)
        and GOAL_AREA_Y_PCT[0] < y_pct < GOAL_AREA_Y_PCT[1]
    )
    in_penalty = (
        (x_pct < PENALTY_X_PCT and PENALTY_Y_PCT[0] < y_pct < PENALTY_Y_PCT[1])
        or (x_pct > (100.0 - PENALTY_X_PCT)
            and PENALTY_Y_PCT[0] < y_pct < PENALTY_Y_PCT[1])
    )

    if in_corner:
        return "corner"
    if in_left_goal or in_right_goal:
        return "goal_kick"
    if in_penalty:
        return "free_kick"
    # Mid-pitch free kick — exclude touchline bands where furniture sits
    if y_pct < TOUCHLINE_Y_PCT or y_pct > (100.0 - TOUCHLINE_Y_PCT):
        return None
    if x_pct < CORNER_X_PCT or x_pct > (100.0 - CORNER_X_PCT):
        return None
    return "free_kick"


def _zone_confidence_scale(event_type: str) -> float:
    return {"corner": 0.85, "goal_kick": 0.80, "free_kick": 0.55}.get(event_type, 0.5)


def _still_runs(
    ball_history: Sequence[Tuple],
    still_dist_px: float = STILL_DIST_PX,
    max_gap_s: float = STILL_MAX_GAP_S,
    min_duration_s: float = STILL_MIN_DURATION_S,
    min_conf: float = MIN_BALL_CONF,
) -> List[dict]:
    """Group consecutive ball detections that stay roughly in one place."""
    if not ball_history:
        return []

    pts = sorted(
        (
            float(rec[1]),
            float(rec[2]),
            float(rec[3]),
            float(rec[4]) if len(rec) > 4 else 0.0,
        )
        for rec in ball_history
        if float(rec[4] if len(rec) > 4 else 0.0) >= min_conf
    )
    if not pts:
        return []

    def _append_run(chunk: list) -> Optional[dict]:
        if not chunk:
            return None
        ts, te = chunk[0][0], chunk[-1][0]
        if te - ts < min_duration_s:
            return None
        mx = sum(p[1] for p in chunk) / len(chunk)
        my = sum(p[2] for p in chunk) / len(chunk)
        return {
            "start_s": ts,
            "end_s": te,
            "second": (ts + te) / 2.0,
            "x_px": mx,
            "y_px": my,
            "mean_conf": sum(p[3] for p in chunk) / len(chunk),
            "duration_s": te - ts,
            "samples": len(chunk),
        }

    runs: List[dict] = []
    run_start = 0
    for i in range(1, len(pts)):
        t0, x0, y0, _ = pts[i - 1]
        t1, x1, y1, _ = pts[i]
        dist = math.hypot(x1 - x0, y1 - y0)
        gap = t1 - t0
        if dist <= still_dist_px and gap <= max_gap_s:
            continue
        rec = _append_run(pts[run_start:i])
        if rec is not None:
            runs.append(rec)
        run_start = i

    rec = _append_run(pts[run_start:])
    if rec is not None:
        runs.append(rec)
    return runs


def _event_exists(events: List[dict], etype: str, second: float,
                  x_px: float, y_px: float,
                  window_s: float = DEDUP_WINDOW_S,
                  spatial_px: float = DEDUP_SPATIAL_PX) -> bool:
    for e in events:
        if e["event_type"] != etype:
            continue
        if abs(second - e["second"]) >= window_s:
            continue
        if math.hypot(x_px - e["x_px"], y_px - e["y_px"]) < spatial_px:
            return True
    return False


def detect_ball_events(
    ball_history: Sequence[Tuple],
    width: int,
    height: int,
    pitch_polygon=None,
) -> Tuple[List[dict], Dict]:
    """Return (events, meta). Goals are never emitted — see meta['goals_note']."""
    furniture = furniture_cells(ball_history, pitch_polygon)
    runs = _still_runs(ball_history)

    events: List[dict] = []
    zone_raw: Dict[str, int] = defaultdict(int)
    zone_candidates: Dict[str, int] = defaultdict(int)
    rejected_zone_none = 0
    rejected_furniture_free_kick = 0
    rejected_dedupe = 0
    emitted: Dict[str, int] = defaultdict(int)

    for run in runs:
        x_pct, y_pct = px_to_pct(run["x_px"], run["y_px"], width, height)
        etype = classify_zone(x_pct, y_pct)
        zone_raw[etype or "zone_none"] += 1

        if etype is None:
            rejected_zone_none += 1
            continue

        zone_candidates[etype] += 1

        # Furniture filter: free_kick only — corners/goal_kicks live at the edge.
        if etype == "free_kick":
            cell = (int(run["x_px"]) // BALL_CELL_PX,
                    int(run["y_px"]) // BALL_CELL_PX)
            if cell in furniture:
                rejected_furniture_free_kick += 1
                continue

        second = round(run["second"], 3)
        if _event_exists(events, etype, second, run["x_px"], run["y_px"]):
            rejected_dedupe += 1
            continue

        dur_boost = min(0.12, max(0.0, run["duration_s"] - STILL_MIN_DURATION_S) * 0.04)
        conf = min(
            0.95,
            run["mean_conf"] * _zone_confidence_scale(etype) + dur_boost,
        )
        events.append({
            "second": second,
            "event_type": etype,
            "x_px": round(run["x_px"], 1),
            "y_px": round(run["y_px"], 1),
            "confidence": round(conf, 4),
        })
        emitted[etype] += 1

    events.sort(key=lambda e: e["second"])
    by_type: Dict[str, int] = dict(emitted)

    meta = {
        "goals_note": GOALS_NOTE,
        "goals_emitted": 0,
        "method": "ball_stillness_zone_heuristic",
        "counts_by_type": by_type,
        "total": len(events),
        "funnel": {
            "still_runs": len(runs),
            "zone_raw": dict(zone_raw),
            "zone_candidates": dict(zone_candidates),
            "rejected_zone_none": rejected_zone_none,
            "rejected_furniture_free_kick": rejected_furniture_free_kick,
            "rejected_dedupe": rejected_dedupe,
            "furniture_cells": len(furniture),
            "emitted": dict(emitted),
        },
    }
    return events, meta


def write_events_csv(path: str, events: Sequence[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["second", "event_type", "x_px", "y_px", "confidence"],
        )
        w.writeheader()
        w.writerows(events)
