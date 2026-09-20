# Soccer PoC — handover (Sean / RunPod / Mac)

Last updated: March 2026. Repo: `Second-software`, code under `sports-main/examples/soccer/`.

---

## 1. Quick start (production inference)

```bash
cd sports-main/examples/soccer
export PYTHONPATH=/workspace/Second-software/sports-main   # or ../../ on Mac

python main.py \
  --source_video_path /path/to/video.mp4 \
  --target_video_path /path/to/out.mp4 \
  --device cuda \
  --mode PLAYER_TRACKING \
  --detector yolo \
  --model runs/detect/runs/panoramic/yolo32x9/weights/best.onnx \
  --imgsz 576 2048 \
  --conf 0.15
```

Pass-1-only / calibration (no video):

```bash
python main.py ... --no_render --track_diag --run_label myrun
```

ByteTrack sweep (IoU threshold — **lower = more permissive**):

```bash
python main.py ... --no_render --track_diag --run_label bt070 --bt_match 0.70
python main.py ... --no_render --track_diag --run_label bt050 --bt_match 0.50
# optional: --bt_lost 1.25
```

Startup prints **must** show:

`ByteTrack (this run): minimum_matching_threshold=...`  
If that line does not match your `--bt_match`, the sweep did not apply.

---

## 2. Detector training (Sean can re-run)

Full runbook: `sports-main/examples/soccer/scripts/README_TRAINING.md`.

| Step | Script |
|------|--------|
| 80/20 split + `data.yaml` | `scripts/prepare_panoramic_dataset.py` |
| Train YOLO 32:9 + ONNX | `scripts/train_panoramic_yolo.py` |
| Val vs RF-DETR buckets | `scripts/eval_far_third_recall.py` |

**Why 576×2048:** 4096×1152 squashed into square 640 destroys far-touchline boxes (~30px → ~5px). Train and infer at **32:9**, `rect=True`.

**Label caveat:** Only ~2% of training boxes are &lt;40px tall — labels under-count far-touchline players. Val **small/far+small** recall is a lower bound; **Pass 1 fragment median span** on real video is the real check.

**Held-out val (same images, matched conf):**

| Bucket | Stock RF-DETR | Fine-tuned YOLO |
|--------|---------------|-----------------|
| All | 52.7% | **86.3%** |
| Far third | 56.8% | **87.2%** |
| Under 40px | 0/32 | 6/32 |

Labels derived from stock detector + pitch filter — some lift is teacher-matching; +30 pts far-third and 0→6 small boxes is still real signal.

**What detector fix changed on video (`clip10min`, fine-tuned YOLO):**

- Frame coverage inside fragments: **~84% → ~100%** (pre-bridge)
- Internal gaps: thousands of ≤10f dropouts → **hundreds**, mostly **&gt;10f** (occlusions)
- **Median fragment span still ~4.2s** — bottleneck moved to **tracker**, not detector

---

## 3. ByteTrack / identity (current engineering focus)

**Constants (defaults in `main.py`):**

- `TRACK_MATCHING_THRESHOLD = 0.90` → passed to Supervision `ByteTrack` as **`minimum_matching_threshold` (minimum IoU)**. At 0.90, a player moving ~0.5 body-heights between frames (~IoU 0.5–0.6) fails to match — obvious same-player rejections.
- `BYTE_TRACK_LOST_SECONDS = 0.75` → `lost_track_buffer` frames

**Instrumentation:** `--track_diag` logs every canonical-id drop; JSON under `data/id_lists/track_diag_<clip>_<run>.json`.

**Diag motion gate (fixed):** Old gate `0.55 × box_h` mis-labelled movement as “no detection”. New gate: `1.0 bh + 9.0 bh/s × gap_sec`, distance to **predicted** position. Reclassify old JSON without GPU:

```bash
python tools/reclassify_track_diag_gate.py --diag data/id_lists/track_diag_....json
```

**Baseline diag (match=0.90, `clip10min`):** ~1715 fragment ends; after reclassify ~**72%** had a detection within the motion gate; **~742** ends assigned detection to **another** canonical id (ByteTrack raw-id switch); **0** ReID reclaims at those moments.

**Sweep results so far:**

| Run | `--bt_match` | Fragments | Median span | Notes |
|-----|--------------|-----------|-------------|--------|
| baseline | 0.90 (default) | 567 | 4.21s | Reference |
| bt085 | 0.85 | 594 | 3.81s | **Worse** — higher min IoU is stricter; 0.85 was wrong direction |

**Planned sweep:** **0.90, 0.70, 0.50** (not higher than 0.90). Compare `fragments`, `median span`, track_diag raw-id switches.

**Do not yet:** team_colour rebuild, loose identity merge — wait until online tracks stabilize.

---

## 4. CSV / JSON outputs

Written to `sports-main/examples/soccer/data/id_lists/` (suffix includes video stem + `--run_label`).

| File | Contents |
|------|----------|
| `player_id_list_*.json` | Per **fragment** stats + `run_metrics` |
| `identity_map_*.json` | Fragment → roster identity; human correction interface |
| `players_*.csv` | Per-identity aggregates (after assign) |
| `positions_*.csv` | Per-frame rows: identity, second, x_px, y_px, box_height_px, … |
| `ball_*.csv` | Ball trajectory when `SHOW_BALL` on |
| `events_*.csv` | Heuristic events (e.g. free_kick); goals omitted without goal-line calibration |
| `track_dump_*.json` | Optional `--track_dump`; per-fragment frame lists for offline tools |
| `track_diag_*.json` | Optional `--track_diag`; fragment-end drop reasons |

**Correction workflow:** Edit `identity_map_*.json` (or use assign pipeline), re-run with `--identity_map path/to/fixed.json`.

---

## 5. Ball tracking

Roughly **~14%** of processed frames get a ball row in pass 1 on `clip10min` (detector finds ball class intermittently; confidence often 0.30–0.55). Rolling static suppressor + pitch bounds reduce furniture; **precision is not production-grade** (~order 50% usable without manual review — treat as overlay/trajectory hint, not stats).

Gates: `BALL_MIN_CONF`, `BALL_ACCEPT_MIN_CONF` in `main.py`.

---

## 6. Physics guard & merge

- Online/post: cuts when implied speed **&gt; 9 body-heights/s** at fragment joins (`assign_identities` physics guard).
- Path/net ceiling default **25** (`--path_net_ceiling` overrides **both** pass-1 splits and merge guard — echoed at startup).
- **Non-determinism:** Same clip can yield **387 / 396 / 400** merged identities across runs (GPU, assignment order, thin links). Always quote **run_label** and log file when comparing before/after.

---

## 7. Team colour (honest status)

Shipped path uses **CIELAB torso** voting in `team_colour.py` + soft penalties in merge — **not** the experimental high-separation variant sometimes discussed in chat (~10% error claims). On this footage, **navy vs sky-blue differ mainly in L**; shadows break lightness-only separation. **Different away kit colour** would help more than tuning L thresholds.

Do not treat team assignment as solved.

---

## 8. Approaches tried (high level)

1. ByteTrack + ReID (SigLIP largely off on this footage)  
2. Offline stitch / weld guard / path-net splits  
3. `assign_identities` merge with physics + appearance bands  
4. Panoramic YOLO fine-tune (**win on detection**)  
5. `--track_diag` + ByteTrack IoU sweep (**in progress**)  
6. Operational: camera height / team sheet / kit colour (Sean-facing, zero code cost)

---

## 9. Not done / out of scope for current PoC

- **Metre-based** stats (homography / calibration not trusted on this ground)  
- **~25 stable identities × 10 min** deliverable — still **~150+** merged IDs on 10-min window after detector fix  
- Reliable **goal** detection (note in logs: goal line not calibrated)  
- Jersey OCR / numbers  

---

## 10. Sean-facing recommendations (copy-ready)

**Camera:** Far-touchline players **24–38 px** vs **~67 px** median. Raising/lowering camera to **~60–70 px** at the far line helps detection, appearance, and track length more than parameter tuning alone.

**Ops:** Team sheet → real names on identities. **Away kit ≠ second blue** → team assignment becomes tractable.

---

## 11. Git / RunPod notes

- Pull before runs: `git pull origin main`  
- `--no_render` does not require `--target_video_path` (recent main)  
- Motion pitch polygon at startup can take **30–70 min** first time; subsequent runs on same machine still pay unless polygon cached or skipped  
- Weights path on pod: `runs/detect/runs/panoramic/yolo32x9/weights/best.onnx`

---

## 12. Key commits (tracker)

- Panoramic training + inference fixes  
- `--track_diag`, pre-bridge gap metrics  
- `--bt_match`, `--bt_lost` CLI + startup echo  
- Track diag motion gate + `tools/reclassify_track_diag_gate.py`
