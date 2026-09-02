# Soccer AI Tracking — Client Update Brief
**Project:** Stationary panoramic camera player tracking (RF-DETR + ByteTrack)  
**Date:** 2 September 2026  
**Clip:** `Stationary_Camera_14_08` (kickoff window from ~7:57)

---

## 1. Client problem (your words)

> Detection works, but identity does not.  
> Frame 1’s Player 5 becomes Player 87 by Frame 500.  
> Without stable identity, statistics are useless — “who ran 8 km?” needs a reliable *who*.

**Deliverable:** ~10 minutes of footage with **one stable identity per player**, stats correctly merged across the full passage.

**Priority order:** (1) Player detection → (2) **Identification** → (3) Ball tracking

---

## 2. What improved (shipped)

| Area | Before | Now |
|------|--------|-----|
| **Detection** | Working (RF-DETR) | Still strong; GPU CUDA path fixed for RunPod RTX 4090 |
| **ID explosion (`100xxx` fake IDs)** | **428** collision-split IDs in a bad run (v4) | Cut to **34** (v5) by separating short ByteTrack linking from long-horizon ReID |
| **Blank / empty tracking video** | Pass 2 could render **zero** boxes (`Valid IDs: []`) when pitch keypoints failed | Pitch fallback + roster safety so detections and IDs **draw on video** |
| **Pitch filter crash** | Dry pitch → 0 keypoints → bad motion polygon wiped all players | Validated polygon / %-bounds fallback; players no longer mass-rejected |
| **Render speed (Pass 2)** | Re-ran SigLIP embeddings every frame (slow + noisy logs) | Pass 2 skips appearance; uses Pass‑1 identities to draw |

### Measured MOT ID pressure (same camera family)

| Run | Total IDs | Passed filter | Collision IDs (≥100000) |
|-----|-----------|---------------|-------------------------|
| mot_sota_v3 | 413 | 317 | 227 |
| mot_sota_v4 | 697 | 564 | 428 |
| mot_sota_v5 | **281** | **218** | **34** |

Collision swarm is largely under control. **True long-horizon identity (~24 players for 10 minutes) is still the open workstream** — that is the remaining gap to your deliverable.

---

## 3. What you can review now (demo package)

Please share / review these together:

1. **Demo video** — latest rendered tracking output (ellipses + IDs + minimap when enabled)  
   - Suggested name: `output_mot_v6_*.mp4` from RunPod  
2. **ID list JSON** — `player_id_list_*_mot_sota_v6_*.json`  
   - Columns include frames seen, path length, `path_net_ratio`, team (when colour separation works)  
3. **This brief** — context for what changed and what is still open

### How to watch the video (2 minutes)

- Do player boxes appear on the pitch (not blank frames)?  
- Do ~20–24 players look present in busy frames?  
- Pick **2–3 players** and follow for 30–60 seconds: does the number stay the same, or jump (5→87)?  
- Note any “teleport” (same ID leaping across the pitch) — that is an identity weld, not a detection miss.

---

## 4. Honest status vs deliverable

| Requirement | Status |
|-------------|--------|
| Player detection | **Met** |
| GPU inference on production box | **Met** (CUDA EP path fixed) |
| Visible tracking output for review | **Met** (blank-video failure fixed) |
| Stable identity Frame 1 → Frame 500+ for each player | **In progress** — fragmentation still above ~24–40 clean identities on long clips |
| “Who ran 8 km?” production stats | **Not yet** — blocked on identity |
| Ball tracking | **Not started** (correctly deferred per your priority) |

---

## 5. Next sprint (identification-only)

1. **10-minute production run** on the agreed kickoff window (full stats + video).  
2. Quantify remaining switches / welds from that run (`path_net_ratio`, lifetime per ID).  
3. Tighten online association + confident stitch until passed IDs sit in **~24–40** *and* manual review confirms continuity.  
4. Only then: ball trajectory and distance/possession stats on stable IDs.

---

## 6. One-line takeaway for stakeholders

**Detection and live tracking visualization are ready to review; identity stability is materially better (collision explosion fixed) but not yet at the bar for production “who” statistics over a full 10-minute passage.**

---

*Prepared from pipeline iterations `mot_sota_v3` → `v6` on the stationary ultrawide camera path.*
