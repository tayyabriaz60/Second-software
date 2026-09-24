# Bosco FC — handover

What was built, how to run it, what the numbers mean, and what is not done.

Everything runs offline. No API keys, no per-frame cost.

---

## Quick start

```bash
cd sports-main/examples/soccer
export PYTHONPATH=/path/to/sports-main

python main.py \
  --source_video_path your_clip.mp4 \
  --target_video_path annotated.mp4 \
  --mode PLAYER_TRACKING --detector rfdetr \
  --track_dump \
  --run_label myrun
```

Add `--device cuda` on an NVIDIA machine. On a Mac leave it off; the detector
falls back to CPU on its own.

Add `--no_render` to skip the video and get the numbers in about a fifth of the
time. Useful when iterating.

### What lands in `data/id_lists/`

| File | Contents |
|---|---|
| `players_<clip>_<label>.csv` | One row per identity: team, first and last second, duration, frames seen, distance, mean speed, path/net ratio |
| `positions_<clip>_<label>.csv` | One row per identity per second: x, y, box height, whether the position was inferred across a gap |
| `ball_<clip>_<label>.csv` | Ball position per second with confidence |
| `events_<clip>_<label>.csv` | Set pieces: second, type, position |
| `identity_map_<clip>_<label>.json` | Which fragments make up each identity, plus every merge the physics guard refused and why |
| `track_dump_<clip>_<label>.json` | Raw per-frame tracking, for offline analysis |
| `pitch_polygon_<clip>_<label>.npy` | The playing area, derived from motion |

Times are in seconds, not frame indices. This footage is variable frame rate —
measured inter-frame gaps run 8.33 to 30.0 ms across nine discrete values — so
frame numbers do not map cleanly onto time.

Distances are in pixels and the column names say so. Metre calibration is not
done, and a fabricated metre figure would be worse than an honest pixel one.

---

## Two bugs fixed in the original pipeline

**Duplicate rows.** `merge_stitched` concatenated two identity histories and
sorted them without collapsing frames present in both. Measured at 41,983
duplicate rows out of 546,778 on a ten-minute clip, 7.7%. Every per-detection
statistic was inflated by that margin, distance covered included.

**Time from frame index.** `run_full_analysis` computed `secs = frame_n / fps`.
On variable-frame-rate footage with gaps ranging 8.33 to 30.0 ms, that carries
up to 4x error into anything integrated over time. Now reads presentation
timestamps from the container clock.

Both feed distance covered, which is the headline number.

---

## What else changed

**GPU detection.** `rfdetr_onnx.py` hardcoded `CPUExecutionProvider`. Correct
for a Mac — CoreML measured slower than CPU here — but it meant the detector
never touched the GPU on any CUDA machine. Now picks CUDA when available and
falls back, so Mac behaviour is unchanged.

**Team label reaches the outputs.** The colour classifier votes per tracklet,
but `player_id_list` wrote `"team": None` unconditionally and the track dump
carried no team field at all. A standalone stitch therefore had less evidence
than the in-process one.

**Render uses the tracklet vote.** Pass 2 was colouring players by whichever
team label their first fragment happened to get, rather than the per-identity
majority vote the classifier already computes. That vote is what lifts per-crop
accuracy to per-player accuracy, and first-wins discarded it.

**Draw filter separated from the stats filter.** `valid_ids()` drops anything
under 1.5 seconds and anything that barely moved. That is right for statistics
and wrong for drawing: applied to the render it hid short tracks, which is
exactly what fragmentation looks like, so the video under-reported the tracker.
Minimap positions went from 50,704 to 65,312 with it off.

**Team hard veto removed from merging.** `stitch_tracks.link_cost` was vetoing
links on team mismatch, contradicting the comments saying team should only ever
be a soft penalty. A veto was measured making things worse — 279 identities
became 328 — because at roughly 90% team accuracy a hard rule blocks one correct
link in ten, and a blocked correct link is invisible. Soft penalty only now.

**Ball tracking enabled.** `SHOW_BALL` defaulted to false, so the ball detector
never ran in the tracking path. "Ball in 0 frames" was a flag, not the footage.

**Constants unified.** `TEAM_HARD_SEP_MIN` was 1.35 in one module and 2.0 in
another; `TEAM_MISMATCH_PENALTY` was 0.25 against 0.50. Both now come from
`team_colour.py`.

---

## Identity: what the numbers say

Measured on a ten-minute passage cut reproducibly with
`ffmpeg -ss 477 -t 600`, 33,114 frames.

| | Stock ByteTrack | This pipeline |
|---|---|---|
| Identities | 1,214 | 171 |
| Median identity duration | 1.8 s | 62.3 s |
| Holding over 60 s | — | 90 |
| Clean and holding over 60 s | — | 56 |

"Clean" means the identity passes a path-to-displacement check, so it is one
player rather than several welded together.

Roughly 25 identities holding the full ten minutes is the target. This is not
that.

### Why ten minutes is not reachable on this footage

Each player exists as about 44 fragments across ten minutes. Carrying one
through means chaining 44 links correctly, and link accuracy compounds:

| Per-link accuracy | Chance one player survives 44 links |
|---|---|
| 93% (measured on confident links) | 4% |
| 99% | 64% |
| 98.4% | 50% |

Reaching the contract as written needs **98.4% per link** on 67px crops of two
near-identical blue kits, where jersey numbers are legible in 10.6% of
detections. That is not achievable, and it is a property of the footage rather
than the code.

99% of merge links are flagged low-confidence. That is not a bug — it is the
algorithm honestly declining to force joins it cannot evidence.

### Where fragments come from

| Cause | Share |
|---|---|
| Broke mid-pitch | 99% |
| Exited side edge | 1% |
| Exited bottom edge | 0% |

Tracks do not break because players leave frame. They break in open play, in
view.

---

## The physics guard

`assign_identities.py` merges fragments into identities. It originally had no
physics check at all — just a gap and a distance, with a twelve-second window —
so pass one would split a weld apart and the merger would weld it straight back
together.

It also used a flat 1400 px/s speed threshold. On footage running 24px at the
far touchline against 520px near camera, that is a teleport for one player and a
jog for another.

Now every candidate merge is checked at **9.0 body-heights per second** plus a
**path-to-displacement ceiling of 25**, matching `split_implausible_tracks`.
Speed divided by box height is perspective-free, so it holds at both ends of the
pitch.

Effect: 8,016 merges refused, largest identity down from 101 fragments to 8,
player coverage up from 5-13 tracked at a time to 23.

The ceiling was swept offline against a saved dump:

| Ceiling | Identities | Clean and over 60 s | Largest identity |
|---|---|---|---|
| 25 | 319 | 123 | 44 fragments |
| 50 | 157 | 32 | 83 |
| 75 | 126 | 18 | 90 |
| 100 | 114 | 19 | 104 |

Loosening it makes the roster count look better and the identities worse. 25 is
correct and should not be raised.

---

## ByteTrack matching threshold

`--bt_match` overrides `TRACK_MATCHING_THRESHOLD`. Swept first with the
fine-tuned detector on a two-minute clip:

| Threshold | Fragments | Median fragment span | Max path/net |
|---|---|---|---|
| 0.70 | 296 | 1.14 s | 209 |
| 0.90 (default) | 214 | 2.83 s | 96 |
| 0.95 | 199 | 4.15 s | 25.5 |
| 0.99 | 201 | 2.97 s | 23.7 |

Note the direction: in supervision's ByteTrack, *higher* is more permissive
here, which is the opposite of what a comment in the code used to claim.

**But 0.95 did not carry over.** Re-run with the shipped RF-DETR detector on the
full ten-minute window, 0.95 came out worse than the default: clean identities
holding over 60 seconds dropped from 56 to 48, median identity duration from
62.3s to 57.5s, and one identity welded badly enough to reach a path/net ratio
of 2,911. So **the default of 0.90 is what ships.** The lesson is that a sweep on
a short clip with a different detector is a hint, not a result — confirm on the
deliverable window before changing a default.

---

## A detector trained on this footage

The client's own notes called for this: "generic broadcast-trained model vs Veo
screen recording, train custom YOLO". Every model in the original pipeline was
trained on broadcast footage where players are 100-200px.

400 frames were sampled evenly across the full eighteen minutes, pre-annotated
with the stock detector, filtered to the pitch polygon, and split 80/20
stratified so the few small boxes were not all in one half. Trained at 2048x576
with `rect=True` — a uniform 2x downsample from 4096x1152, so a 30px player
arrives at about 15px rather than the 5px you get squashing into a square.

Scripts are in `scripts/`:

```bash
python scripts/prepare_panoramic_dataset.py --source /path/to/frames --dest /path/to/split
python scripts/train_panoramic_yolo.py --data /path/to/split/data.yaml --imgsz-preset 2048x576
python scripts/eval_far_third_recall.py --data ... --weights ... --imgsz 576 2048
```

### Detection results, held-out validation

| | Stock RF-DETR | Fine-tuned |
|---|---|---|
| All boxes | 52.7% | 86.3% |
| Far third of frame | 56.8% | 87.2% |
| Boxes under 40px | 0.0% | 18.8% |

Far-third recall went from 57% to 87%, and stock RF-DETR finds none of the small
boxes at all.

Downstream it fixed detection continuity: frame coverage inside a fragment went
from 84% to 100%, and internal gaps from 12,023 to 426, with all remaining gaps
longer than ten frames.

### But identity got worse

| | RF-DETR | Fine-tuned |
|---|---|---|
| Identities | 171 | 155 |
| Median identity duration | 62.3 s | 28.1 s |
| Clean and over 60 s | 56 | 36 |
| Fragments per identity | 12 | 3 |

The pieces are cleaner — three fragments per identity rather than twelve — and
they are not joining. The fine-tuned model finds far-touchline players the old
one never saw, which creates many new short tracks, and the merger has no
evidence to chain them.

**So RF-DETR is what ships.** The detector work is real and measurable, and it
does not move the deliverable on this pipeline. The weights and scripts are kept
because the labels are the limiting factor, not the method: only 2% of the
training boxes were under 40px, because the pre-annotation that produced them
missed the same players.

---

## Ball tracking

Enabled by default, `--no_ball` to turn it off.

On the ten-minute passage: 5,084 detections across 33,087 frames, 15.4%.
Confidence runs 0.30 to 0.78, median 0.384.

**Precision is roughly 50%.** 100 detections were sampled at random, cropped
from the frames and checked by eye. About half are actually the ball, so true
coverage is nearer 7-8% of frames.

The failures are not random. They cluster on three things: goal netting and
posts, which are white and mesh-like at this resolution; white parts of player
kit, mostly socks; and sideline flags. Confidence does not separate them — the
false ones sit in the same range as the true ones — so raising the threshold
will not help. Excluding the goal mouths geometrically would.

`--ball_static_suppress` is an opt-in filter for fixed white furniture. It
requires a cell to appear in over 1% of ball rows, be immobile across the whole
clip, and sit outside or at the edge of the pitch. An earlier, more aggressive
version banned 237 cells and cut ball detections from 11,204 to about 500 — it
was suppressing the ball for behaving like a ball.

---

## Set-piece events

`events.py`, ported from logic that was sitting disconnected in the legacy
`analyse.py` and calling a hosted API, which broke the offline requirement.

Corners, goal kicks and free kicks come from ball position and stillness. Goals
are deliberately not emitted, and the output says why: detecting one needs the
goal line calibrated, and it is not.

**This barely works.** Across the full ten minutes the ball comes to rest only
four times, so there are four candidate set pieces in total. The zone logic is
not the problem, the stillness detection is: with the ball genuinely visible in
7-8% of frames it rarely holds still long enough to register.

Fixing ball precision would fix this. Until then treat the events CSV as
indicative rather than a count.

---

## Correction tooling

The stitcher knows which of its links are weak. A link whose next-best
alternative was nearly as cheap is flagged thin, and thin links measure 29%
correct against 93% for confident ones. So a reviewer never needs to look at
every identity — the doubtful evidence is already isolated.

```bash
python tools/auto_review.py --dump track_dump_....json --interactive
python tools/link_crops.py --dump track_dump_....json --video clip.mp4 \
       --decisions auto_decisions.json --outdir review/
```

On a ten-minute clip that took 48 flagged links down to 9 questions.

Decisions propagate. Accepting A→B and B→C merges all three without ever asking
about A→C.

`link_crops.py` renders the last frame of the outgoing fragment beside the first
frame of the incoming one, with the tracked player ringed. Nobody can adjudicate
"3.2s gap, 91px prediction error" from numbers, and these links break precisely
when players bunch up, so a crop of a crowd without a marker does not say which
of five shirts is meant.

`assign_identities.py --apply corrections.json` applies decisions to an existing
identity map without re-running detection.

**An honest limit.** Some links cannot be resolved by anyone. Five light-blue
players clustered at 67px with no legible numbers is not a judgement a human can
make from two frames either.

---

## Approaches tried and ruled out

Each was implemented and measured on this footage, not estimated.

**GTA-Link** (ACCV 2024, state of the art for sports tracklet association). Its
splitter works. Its connector merges one pair per iteration and rebuilds the
full distance matrix each pass — at their default threshold it ran 30+ minutes
without converging, and at a third of that it still did not. With two blue kits
at 67px, most pairs sit under any useful threshold, so it would eventually
collapse the match into a single identity.

**Stitching cost sweep.** Bottoms out around 270 identities; doubling the price
of ending a chain from 2.0 to 4.0 buys six. Every gain comes from thin links.

**Team as a hard veto.** 279 identities became 328. Covered above.

**Detection confidence at 0.15.** The extra detections are partial boxes and
crowd rather than missed players. Identities rose 117 to 126 and ID handovers
rose 50%.

**Tiled inference.** The model declares 576x576 with stretch-to preprocessing,
so a 4096x1152 frame squeezes 7.1x horizontally against 2x vertically. Tiling
fixes the distortion and raises raw detections 45 to 74 — but on-pitch
detections stay at 18.5, and tracking got worse: 101 tracks became 144, median
duration 7.5s down to 5.3s. Median box height went *up*, which says it was
producing larger duplicates of players already found rather than finding small
ones.

**SigLIP for team classification.** 221 frames yielded 15 crops and UMAP could
not fit. The client's own notes have it splitting 60 v 2. Both kits differ in
lightness rather than hue, and a whole-box embedding is dominated by pitch,
shadow and pose. Median CIELAB torso colour with per-tracklet voting is what
works.

---

## Known issues

**76% of raw detections are off-pitch** — crowd, sideline, the neighbouring
field. Pitch filtering removes them, but the detector spends three quarters of
its work on people who are not playing. With the fine-tuned model that drops to
about 22%.

**The pipeline is not fully deterministic.** Identical input has produced 387,
396 and 400 identities across runs, because the motion-based pitch polygon
differs slightly each time. Pin it with `--pitch_polygon` before quoting any
before/after comparison.

**The pitch keypoint model finds zero landmarks** on this dry pitch, confirmed
live at every confidence down to 0.05. The motion-based polygon fallback is what
actually works. `CLAUDE.md` still describes the homography as working.

**`CLAUDE.md` contradicts the shipped code** in several places — it describes
YOLO plus SigLIP team classification where the code defaults to RF-DETR plus
CIELAB.

**The legacy `analyse.py` uses the hosted Roboflow API**, which breaks the
fully-offline constraint. That path should be closed before this goes anywhere
near production.

**The shade-aware `team_colour.py` is missing.** An 18KB version with shade
normalisation and weighted voting was measured at 10.4% error on light kit in
shade. It only ever existed in a pre-bid handover folder and is in no
repository. What ships is the 6.5KB naive version, which errs 51.5% on that same
case. A rebuild from the documented behaviour came out worse than the naive one
— separation 2.31 down to 2.06 — because ground luminance was sampled from the
bottom of the crop, which is boots rather than grass. Doing it properly needs
`main.py` to store a ground patch alongside each crop.

---

## Not done

**Metre calibration.** Everything is in image space. The panoramic is a stitched
render with curved lines and a depth-error gradient of roughly fifty to one
between near and far touchline, so a single global homography will not hold.
This is the fiddliest remaining piece and it needs a per-region approach.

**Jersey numbers — feasibility not decided yet.** See [Jersey OCR probe](#jersey-ocr-probe)
below. A number gives a fragment an absolute label, so identity stops depending
on chaining and the 4% survival figure above stops applying. Sean (client) has
been told twice that jersey reads are the next step; the v4 probe result drives
the next client update either way.

---

## Jersey OCR probe

Offline tool only: `sports-main/examples/soccer/tools/jersey_ocr_probe.py`. It
does **not** modify `main.py` or `assign_identities.py` unless a follow-up
explicitly adds jersey anchors to the merger.

**Client context.** Sean wants one stable ID per player across ten minutes. The
deliverable pipeline is at roughly **170–190 identities** with **median identity
duration ~1 minute** on the clip10min window (not the full 62 s figure from an
older analysis pass — quote the same window and dump when comparing). Jersey
numbers are the obvious way to fuse fragments without compounding link error.

### What was wrong in every run before v4

Two independent bugs made all prior probe numbers **void** (including reported
~28% and ~26% “consistent fragment” rates):

1. **Video frame keys.** Dump `frames[]` are clip-relative pass-1 indices;
   `start_frame` in the dump must be added when decoding the file. Fixed in
   `dc7d318`.

2. **Box height source.** main.py `--track_dump` stores native-pixel `h` per
   frame. The probe must use that list directly — not a y→height estimate and
   not infer-space scaling when `inference_imgsz` is in the JSON (that doubled
   `h` and blew up torso crops to legs/netting). Diagnostics now report
   `tracks_with_dump_h` and `tracks_using_y_height_fallback`.

3. **Crop geometry (centre band).** Dump `xy` is the **detector box centre**,
   not the feet. The probe built a full box then took 15–60% from the **top** of
   that box — equivalent to a feet-anchored band around `cy − 0.85h … cy − 0.50h`
   when `xy` is read as centre. **Every OCR read was grass** (pitch texture,
   shadow edges → EasyOCR `1`, `I`, `ll`, etc.). Fixed in `669278e` with
   centre-based torso crop:
   - vertical: `cy − 0.30h` to `cy + 0.05h`
   - horizontal: `cx ± 0.18h`

Team colour in `main.py` is **not** on the same bug path: it crops full
`detections.xyxy` and `team_colour.py` takes 15–55% of **that** box.

### Run in flight (v4)

Pod / workspace example:

```bash
cd sports-main/examples/soccer
export PYTHONPATH=/path/to/sports-main
git pull   # need >= 669278e

python tools/jersey_ocr_probe.py \
  --dump data/id_lists/track_dump_clip10min_deliver_v2.json \
  --video /workspace/clip10min.mp4 \
  --out-dir data/jersey_ocr_v4
```

**Eligibility (unchanged):** fragments with ≥20 frames at box height ≥200 px;
referee class skipped by default (RF-DETR referee = 3). On deliver_v2 that is
**168 eligible** fragments (174 height-eligible tracklets minus referees).

**Verify crops before trusting OCR:**

```bash
python tools/jersey_ocr_probe.py ... \
  --only-fragments 178 \
  --debug-fragment 178 \
  --export-crops-only \
  --out-dir data/jersey_ocr_crops178
```

JPGs land in `out-dir/crops_frag178/` (`*_up.jpg`, optional `*_pre.jpg`). If
the shirt is not in the crop, stop — nothing downstream matters.

Debug with EasyOCR + optional PaddleOCR compare: `--debug-fragment 178
--compare-engines --debug-only` (primary verdict is **raw upscale**, not CLAHE).

### RunPod: fix cv2/numpy after a bad Paddle pip install

Installing Paddle in the **same** env as the tracking stack often breaks OpenCV:
pip may remove `opencv-contrib-python` and leave **NumPy 2.x** while **cv2** was
built for NumPy 1.x (`numpy.core.multiarray failed to import`). The `blinker`
uninstall error is a red herring — ignore it; repair numpy/opencv first.

**Restore the main env (EasyOCR probe + main.py):**

```bash
pip install --force-reinstall "numpy==1.26.4" \
  "opencv-contrib-python==4.10.0.84" "opencv-python-headless==4.10.0.84"
python -c "import cv2, numpy; print('ok', numpy.__version__, cv2.__version__)"
```

**Disk quota exceeded?** Paddle + EasyOCR need several GB. Before any venv:

```bash
pip cache purge
du -sh /workspace/* /root/.cache/pip 2>/dev/null | sort -h
rm -rf /workspace/venv_jersey_paddle   # drop failed partial venvs
```

If Paddle cannot install, you can still run EasyOCR debug on frag 178 from the
**main** env (after numpy/opencv repair) with `--no-paddle-debug --debug-only`.
That answers “does EasyOCR read 10 on the fixed crop?” without Paddle.

**Paddle compare only — use a separate venv** (do not pip install Paddle into
the tracking env again; needs free disk):

```bash
python3.11 -m venv /workspace/venv_jersey_paddle
source /workspace/venv_jersey_paddle/bin/activate
pip install "numpy==1.26.4" opencv-python-headless==4.10.0.84 easyocr
pip install paddlepaddle-gpu==2.6.2 paddleocr==2.7.3
cd /workspace/Second-software/sports-main/examples/soccer
export PYTHONPATH=/workspace/Second-software/sports-main
export PADDLEOCR_LEGACY=1
python tools/jersey_ocr_probe.py \
  --dump data/id_lists/track_dump_clip10min_deliver_v2.json \
  --video /workspace/clip10min.mp4 \
  --out-dir data/jersey_ocr_engine178 \
  --only-fragments 178 --debug-fragment 178 \
  --compare-engines --debug-only
```

### When v4 finishes — three checks

1. **Fragment 178.** Shirt clearly shows **10**; every pre-v4 run voted **1**
   (grass). If v4 still reads **1** with crops that show the shirt, the crop is
   fixed but **off-the-shelf OCR may not resolve two digits** at ~35 px digit
   height (72×91 torso upscaled 2.5×) — a different conclusion from “wrong crop”.

2. **Number distribution** in `jersey_ocr_report.json` majority counts. A squad
   should look like **~11 distinct numbers per team**, each appearing a few
   times. If one digit dominates again, something is still wrong (crop, frames,
   or engine).

3. **Contact sheet** `jersey_ocr_contact.jpg`. Do overlaid `#N` labels match the
   shirts by eye? Metrics can lie; this cannot.

Probe “consistent” definition (unchanged): ≥3 agreeing reads on sampled frames,
majority ≥60% of successful reads on that fragment.

### Decision rule

| Outcome | Meaning |
|---|---|
| **≥30%** of probed fragments consistent **and** contact sheet agrees | Worth building on: use reads as **identity anchors** (next step). |
| **<10%** consistent | Off-the-shelf OCR (EasyOCR / Paddle at this resolution) is not enough; route is a **digit model trained on crops like these**. |
| Between 10% and 30% | Judgment call; contact sheet and frag 178 decide. |

**Discard** all feasibility percentages from runs before v4.

### If it passes — next engineering step

Use consistent jersey reads in **`assign_identities.py`**: fragments with the
**same team** and the **same confident number** should prefer the same identity,
subject to the existing **physics guard** (body-heights/s, path/net ceiling 25).
Must be measured on the **same ten-minute window** as everything else:

- identity count
- median identity duration
- clean identities holding ≥60 s

Do not change merger defaults without that before/after on `clip10min` +
`track_dump_clip10min_deliver_v2.json`.

### Tool reference (commits)

| Commit | Change |
|---|---|
| `dc7d318` | Clip vs absolute video frame keys; EasyOCR 2-tuple rows |
| `70727e2` | Glyph mapping experiment (later **removed** — fabricated digits) |
| `669278e` | Centre-based torso crop; crop export; Paddle debug path |

Outputs under `--out-dir`: `jersey_ocr_report.json`, `jersey_ocr_contact.jpg`,
optional `debug_fragment_*.json`.

---

## On phase 2 throughput

A ten-minute pass takes roughly 20 minutes without rendering, and around two
hours with. A 90-minute match is five times that, and a season of 30-40 matches
is 75-100 hours of compute, before any reprocessing.

The intention is to run this locally on a Mac, which is slower than the machine
these numbers come from. That arithmetic is worth settling before phase 2 is
priced.

---

## One thing worth asking

Far-touchline players come out at 24 to 38 pixels against a 67 pixel median, and
that single number explains most of what is hard here. It is why the detector
loses them, why appearance matching cannot separate two players in the same kit,
and why a track only holds a few seconds at that end of the pitch.

At 60 to 70 pixels all three of those get easier at once. If the camera can sit
lower or closer to the pitch, it would do more than any amount of tuning, and it
costs nothing.
