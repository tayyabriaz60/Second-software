"""Set-piece heuristics from ball track — fully offline, no goal-line calibration.

Ported from repo-root analyse.py zone logic. Uses ball stillness + pitch zones;
does not emit goals because the goal line is not calibrated on this camera.
"""
from __future__ import annotations

import csv
import math
from typing import Dict, List, Optional, Sequence, Tuple

# Pitch zones as % of frame (stationary wide angle), from analyse.py
CORNER_X_PCT = 8.0
CORNER_Y_PCT = 10.0
GOAL_AREA_X_PCT = 8.0
GOAL_AREA_Y_PCT = (35.0, 65.0)
PENALTY_X_PCT = 18.0
PENALTY_Y_PCT = (20.0, 80.0)

# Ball stillness — proxy for a dead ball at a set piece
STILL_DIST_PX = 50.0
STILL_MAX_GAP_S = 2.5
STILL_MIN_DURATION_S = 2.5
DEDUP_WINDOW_S = 12.0
MIN_BALL_CONF = 0.30

GOALS_NOTE = (
    "goals omitted — goal line not calibrated for this camera; "
    "ball zone heuristics cannot reliably detect scoring"
)


def px_to_pct(x_px: float, y_px: float, width: int, height: int) -> Tuple[float, float]:
    return (x_px / max(width, 1)) * 100.0, (y_px / max(height, 1)) * 100.0


def classify_zone(x_pct: float, y_pct: float) -> str:
    """Map ball location to the set-piece type suggested by analyse.py."""
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

    runs: List[dict] = []
    run_start = 0
    for i in range(1, len(pts)):
        t0, x0, y0, _ = pts[i - 1]
        t1, x1, y1, _ = pts[i]
        dist = math.hypot(x1 - x0, y1 - y0)
        gap = t1 - t0
        if dist <= still_dist_px and gap <= max_gap_s:
            continue
        ts, xs, ys, _ = pts[run_start]
        te, xe, ye, _ = pts[i - 1]
        if te - ts >= min_duration_s:
            chunk = pts[run_start:i]
            runs.append({
                "start_s": ts,
                "end_s": te,
                "second": (ts + te) / 2.0,
                "x_px": sum(p[1] for p in chunk) / len(chunk),
                "y_px": sum(p[2] for p in chunk) / len(chunk),
                "mean_conf": sum(p[3] for p in chunk) / len(chunk),
                "duration_s": te - ts,
                "samples": len(chunk),
            })
        run_start = i

    ts, xs, ys, _ = pts[run_start]
    te, xe, ye, _ = pts[-1]
    if te - ts >= min_duration_s:
        chunk = pts[run_start:]
        runs.append({
            "start_s": ts,
            "end_s": te,
            "second": (ts + te) / 2.0,
            "x_px": sum(p[1] for p in chunk) / len(chunk),
            "y_px": sum(p[2] for p in chunk) / len(chunk),
            "mean_conf": sum(p[3] for p in chunk) / len(chunk),
            "duration_s": te - ts,
            "samples": len(chunk),
        })
    return runs


def _event_exists(events: List[dict], etype: str, second: float,
                  window_s: float = DEDUP_WINDOW_S) -> bool:
    return any(
        e["event_type"] == etype and abs(second - e["second"]) < window_s
        for e in events
    )


def detect_ball_events(
    ball_history: Sequence[Tuple],
    width: int,
    height: int,
) -> Tuple[List[dict], Dict]:
    """Return (events, meta). Goals are never emitted — see meta['goals_note']."""
    events: List[dict] = []
    for run in _still_runs(ball_history):
        x_pct, y_pct = px_to_pct(run["x_px"], run["y_px"], width, height)
        etype = classify_zone(x_pct, y_pct)
        second = round(run["second"], 3)
        if _event_exists(events, etype, second):
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

    events.sort(key=lambda e: e["second"])
    by_type: Dict[str, int] = {}
    for e in events:
        by_type[e["event_type"]] = by_type.get(e["event_type"], 0) + 1

    meta = {
        "goals_note": GOALS_NOTE,
        "goals_emitted": 0,
        "method": "ball_stillness_zone_heuristic",
        "counts_by_type": by_type,
        "total": len(events),
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
