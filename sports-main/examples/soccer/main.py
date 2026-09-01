import argparse
import itertools
import json
import os
from collections import Counter, defaultdict, deque
from datetime import datetime
from enum import Enum
from typing import Iterator, List

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallAnnotator, BallTracker
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH  = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH   = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

# Defaults matching the stock Roboflow model. These are re-read from whatever
# model is actually loaded — see sync_class_ids() — because a custom model can
# easily arrive with a different order. Training on a dataset with no `ball`
# annotations, for instance, can drop that class and shift every index down by
# one, which would silently swap players and referees rather than error.
BALL_CLASS_ID       = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID     = 2
REFEREE_CLASS_ID    = 3


def sync_class_ids(model) -> None:
    """Point the class-ID globals at this model's own class names."""
    global BALL_CLASS_ID, GOALKEEPER_CLASS_ID, PLAYER_CLASS_ID, REFEREE_CLASS_ID
    names = getattr(model, 'names', None)
    if not names:
        print("  Class IDs: model exposes no class names, keeping defaults")
        return

    by_name = {str(v).strip().lower(): int(k) for k, v in names.items()}
    wanted = {
        'ball':       ('BALL_CLASS_ID', BALL_CLASS_ID),
        'goalkeeper': ('GOALKEEPER_CLASS_ID', GOALKEEPER_CLASS_ID),
        'player':     ('PLAYER_CLASS_ID', PLAYER_CLASS_ID),
        'referee':    ('REFEREE_CLASS_ID', REFEREE_CLASS_ID),
    }
    resolved, missing, changed = {}, [], []
    for name, (const, current) in wanted.items():
        if name in by_name:
            resolved[const] = by_name[name]
            if by_name[name] != current:
                changed.append(f'{name}: {current} -> {by_name[name]}')
        else:
            missing.append(name)

    if 'PLAYER_CLASS_ID' not in resolved:
        raise ValueError(
            f"Model has no 'player' class — found {sorted(by_name)}. "
            "Tracking would silently return nothing, so refusing to continue.")

    BALL_CLASS_ID       = resolved.get('BALL_CLASS_ID', BALL_CLASS_ID)
    GOALKEEPER_CLASS_ID = resolved.get('GOALKEEPER_CLASS_ID', GOALKEEPER_CLASS_ID)
    PLAYER_CLASS_ID     = resolved.get('PLAYER_CLASS_ID', PLAYER_CLASS_ID)
    REFEREE_CLASS_ID    = resolved.get('REFEREE_CLASS_ID', REFEREE_CLASS_ID)

    print(f"  Class IDs from model: {dict(sorted(by_name.items(), key=lambda x: x[1]))}")
    if changed:
        print(f"  Remapped: {', '.join(changed)}")
    if missing:
        # Not fatal: a model trained without ball annotations is expected, and
        # ball detection uses its own separate model anyway.
        print(f"  Not in this model (keeping defaults): {', '.join(missing)}")

STRIDE = 30
CONFIG = SoccerPitchConfiguration()

# ================================================================
# FILTERING CONSTANTS — tweak these to tune detection quality
# ================================================================
# An id must live this long to be drawn or counted. Lowered from 3.0s after a
# render showed 1-2 clearly visible players carrying no identity ring.
# Measured over 33s of 14_09: 0.5 detections/frame were being dropped, from 11
# rejected ids whose lifetimes had a median of 1.7s — and ALL ELEVEN were
# rejected purely for being short. None failed the movement test.
#
# The 3.0s floor was set when detection was noisy and short tracks were usually
# spurious. That is no longer the regime: detection now finds 23.1 of ~24 people
# per frame, and stitching merges fragments before this filter runs, so a
# surviving short track is a real player briefly held rather than a phantom.
# The movement test still guards against static furniture, which is the thing
# this floor was really protecting against.
MIN_SECONDS_TO_KEEP = 1.5
# Movement floor as a RATE, not a total. A fixed pixel count silently gets
# stricter the shorter the clip: 500px is trivial over 143s but a lot over 30s,
# so on a 30s render it filtered out ten players who were tracked the whole
# time — goalkeepers and slow defenders, which is exactly who barely moves.
#
# Measured px/sec on the ultrawide (path / seconds present):
#   definitely static : id 16 = 5.1 (net displacement 1px), id 11 = 1.5-4.0
#   real players      : 36-149, clip median 48-89 depending on passage of play
#
# Set low on purpose. Keeping a bystander costs one row to dismiss during
# manual validation; dropping a real player makes them invisible, and that
# failure is much harder to notice.
MIN_SPEED_PX_PER_SEC = 8.0

# ...but speed alone still discards goalkeepers, who patrol slowly and never
# sprint. A keeper and a fence post can look identical by speed; they differ in
# whether they ever RELOCATE. Measured on a 30s ultrawide clip, both present
# for ~100% of it:
#     goalkeeper  4.0 px/s, net displacement 187px
#     static obj  5.1 px/s, net displacement   1px
# So an ID is kept if it is quick enough OR it ended up somewhere else.
MIN_NET_DISPLACEMENT_PX = 100

# Model input size. Players here are only ~45px tall in a 3024px-wide frame,
# so at 1280 the model was seeing them at ~19px and missing roughly 4 of the
# 22 per frame. Costs about 2x runtime.
INFERENCE_IMGSZ = 1920

# Detection confidence floor. Ultralytics defaults to 0.25, which is right for
# a model on footage it knows. A model applied to an unfamiliar ground/kit/light
# combination stays correct but gets UNSURE — on a new match the ultrawide model
# peaked at 0.47 confidence versus 0.69 on its own footage, so everything fell
# below 0.25 and the pipeline saw nothing at all. Lower this when moving to
# unfamiliar footage; raise it back once a model has been trained on it.
INFERENCE_CONF = 0.25

# --- ByteTrack's two tiers -------------------------------------------------
# ByteTrack is designed to be given BOTH confident and unsure detections. A
# detection above the activation threshold may START a new track; anything
# below it may only CONTINUE an existing one. That is the whole point of the
# algorithm: a faint detection where a player was a moment ago is probably that
# player, while the same faint detection in open space is probably noise.
#
# For most of this project we defeated that design without noticing — we
# filtered detections to a single confidence floor BEFORE the tracker, so it
# never saw a weak tier and the activation threshold did nothing. A sweep
# confirmed it: 0.25 and 0.15 gave byte-identical results in every pairing.
#
# Measured on 500 frames of the 14_09 stationary (~9s), same cached detections
# replayed through each configuration, so the tracker is the only variable:
#
#   floor  activation   ids  alive>=90%  >=75%  >=50%
#    0.40      0.25      50       5       13     20    <- old single-tier
#    0.20      0.40      43       7       14     25    <- now
#    0.15      0.45      40       7       14     19
#    0.10      0.50      32       7       14     22
#
# Fewer ids AND better continuity together — those normally trade against each
# other, so fragments are being rejoined rather than tracks suppressed.
# How many crops to keep per id for team colour, and how often to sample one.
# 10 crops spread over a run is plenty for a majority vote, and consecutive
# frames of one player are near-identical so sampling costs nothing in accuracy.
# --- Ball sanity checks (see get_ball) -------------------------------------
# A goalpost joint looks like a ball at this scale, and raw ball detections had
# only 21% of hits inside the pitch with one static x band supplying 58% of them.
BALL_MIN_CONF = 0.30
BALL_CELL_PX = 64
BALL_HISTORY_FRAMES = 90
# A cell producing a ball in more than this fraction of recent frames is
# furniture. A real ball crosses a 64px cell in a fraction of a second, so it
# cannot occupy one for 25% of a 90-frame window; a goalpost does exactly that.
BALL_STATIC_FRACTION = 0.25

# 2D player/ball map in the corner of the render. On by default: reading team
# shape off a 4096px frame is hard, and a dropped ring is invisible among 22
# players but obvious on the map. See minimap.py.
# Cut tracks that move impossibly fast. Everything else in the pipeline merges;
# this is the only stage that can undo an over-merge. See
# split_implausible_tracks().
SPLIT_IMPLAUSIBLE = True

SHOW_MINIMAP = True
MINIMAP_CORNER = 'bottom_left'

# Maximum plausible speed, in BODY HEIGHTS per second — scale-free, so it means
# the same thing near the camera and at the far touchline without needing a
# homography. A 10 m/s sprint at ~1.75m tall is ~5.7 body-heights/sec; this sits
# above that deliberately, because the job is catching teleports (one id handed
# between two players) rather than adjudicating fast running. A wrong cut costs
# a fragment, which stitching may re-join; a missed teleport corrupts a player's
# entire track.
MAX_BODY_HEIGHTS_PER_SEC = 9.0
# Baseline over which speed is measured. Long enough that detection jitter
# averages out, short enough that a real teleport still registers.
SPEED_WINDOW_SECONDS = 0.20

TEAM_CROPS_PER_ID = 10
TEAM_CROP_STRIDE = 8

TRACK_ACTIVATION_THRESHOLD = 0.40
# Matching gate for associating a detection with an existing track; higher is
# more permissive. Swept 0.80/0.90/0.95/0.99 on the same clip: ids 55->52 and
# alive>=75% 10->12. RF-DETR's boxes jitter more than the YOLO boxes ByteTrack's
# 0.80 default was tuned against, so a moving player fell outside the gate.
TRACK_MATCHING_THRESHOLD = 0.99

# Fold offline-stitched fragments into single identities after pass 1. On by
# default: measured on 36s of 14_09 it joined ~44 fragments, and the stitcher
# declines rather than guesses (see stitch_tracks.NO_LINK_COST). Turn off with
# --no_stitch to see the raw online tracker.
STITCH = True

# Which detector to use. 'yolo' is our local ultralytics weights; 'rfdetr' is
# the Roboflow football-players v20 transformer, which handles this footage far
# better (25.9 detections/frame on 12_08's moving view versus 0.0 for our
# best YOLO). See rfdetr_onnx.py. Its natural threshold is ~0.45, not 0.25.
DETECTOR = 'yolo'

# Track referees alongside players. Off by default because PLAYER_TRACKING is
# about players, and a referee tracked as a player pollutes per-player stats.
# Worth turning on to see officials, or to check the detector is separating them.
INCLUDE_REFEREES = False

# Draw the ball. RF-DETR detects it as a class (~58% of frames on the 14_09
# stationary, ~21px, confidence 0.36-0.50), which is enough for a trajectory
# once gaps are interpolated — it does not need the follow-cam.
SHOW_BALL = False

# Skip this many frames before processing. Recordings often start well before
# kickoff — the 14_09 stationary has 8 minutes of warm-up, where extra balls and
# people wandering the pitch make any tracking result meaningless.
START_FRAME = 0

# Record every canonical id's position each frame, for offline tracklet
# stitching (see stitch_tracks.py). Off by default because a full match is
# roughly 25 ids x 300k frames of coordinates.
TRACK_DUMP = False

# Re-identification. These are expressed in seconds because frame counts mean
# different things on 30fps and 57fps footage — the old fixed 150-frame window
# was 5s as intended at 30fps but only 2.6s here.
REID_WINDOW_SECONDS = 5.0   # how long a lost track stays re-identifiable

# Spatial gating for re-id, as a fraction of the frame diagonal (~394px here).
#
# Two apparently-better schemes were measured on game.mp4 and both made things
# considerably worse, so don't re-derive them from first principles:
#   - Radius growing with the gap (4.5 px/frame, the measured sprint speed):
#     IDs 30 -> 51. Most re-ids follow gaps of only a few frames, where that
#     gives a 20-40px radius — tighter than a small player's own box jitter.
#   - Refusing matches when two lost tracks are similarly close: IDs 30 -> 215.
#     With ~25 players in a narrow band there is nearly always a second
#     candidate nearby, so it refused almost every match and ByteTrack opened
#     a new ID instead.
# The flat radius does over-merge occasionally (one ID landing on two players),
# which is worth revisiting — but with appearance features, not geometry.
REID_DISTANCE_FRACTION = 0.12

# How long a track must have been unseen before another detection may adopt its
# canonical id. Previously any track not seen THIS frame was fair game, so a
# momentary detection miss let a second player steal a live id — the "player 2
# hands his number over" failure. A player genuinely lost is absent for more
# than a frame or two.
REID_MIN_LOST_FRAMES = 3

# Appearance tie-breaker for re-id. Among lost tracks already inside the
# spatial radius, prefer the one that also LOOKS most like this detection,
# rather than blindly taking the nearest — which is how one ID ends up on two
# different players standing close together.
#
# Measured separation (probability a same-player crop pair scores above a
# different-player pair), and note how much it depends on crop size:
#
#                         game.mp4 (45px)   ultrawide (77px)
#   torso colour hist          0.705             0.737
#   SigLIP embeddings          0.730             0.883
#
# Colour barely improved; the LEARNED features are what needed resolution. At
# 45px the two were equivalent so the free one won, but on ultrawide footage
# SigLIP is far stronger and worth the compute.
#
# Still ranking, not vetoing. Even at 0.883 the distributions overlap, and a
# hard threshold that rejects matches is how an earlier attempt turned 30 IDs
# into 215. Teammates in identical kit remain indistinguishable by
# construction; this only helps when the confusion is between opposing teams.
# MEASURED OFF. Even at 0.883 separation on ultrawide footage, weighting
# appearance made over-merges WORSE on a 30s segment (8 -> 12 at weight 0.3/0.5)
# and pure distance won on every metric:
#     weight 0.0 -> 33 ids, 23 valid, 8 over-merged, 27 alive>=75%
#     weight 0.5 -> 33 ids, 21 valid, 12 over-merged, 26 alive>=75%
#
# The likely reason is that this hook is the wrong place: _find_lost_match only
# runs when ByteTrack hands us a NEW track id. When ByteTrack's own internal
# association merges two players, that decision never reaches this code, so no
# amount of appearance evidence here can overrule it. Fixing merges means
# changing the association inside the tracker, or splitting merged tracks
# afterwards — not re-weighting this step.
#
# Left wired up (and measured) so it can be revisited cheaply.
APPEARANCE_WEIGHT = 0.0     # 0 = pure distance, 1 = pure appearance

# Embedding every detection every frame would mean ~222k transformer passes on
# a 2.4-minute clip. We only need one when deciding a NEW track's identity, or
# to refresh a known track's signature occasionally.
APPEARANCE_REFRESH_FRAMES = 60
APPEARANCE_MIN_CROP_PX    = 12   # smaller than this carries no usable signal

# Movement is measured as path length, sampled every few frames. Comparing
# first position to last position (the old approach) scores a player who runs
# all game and finishes near where they started the same as a fence post.
MOVE_SAMPLE_EVERY = 15     # frames between position samples
MOVE_MIN_STEP_PX  = 5      # ignore smaller steps — that's box jitter, not travel

# Pitch boundary as % of frame — used for the sides and the near edge, and as
# the fallback for the far edge when the touchline curve can't be fitted.
PITCH_LEFT_PCT   = 5
PITCH_RIGHT_PCT  = 95
PITCH_TOP_PCT    = 10
PITCH_BOTTOM_PCT = 90

# Far-touchline filter. On panoramic footage the pitch bows, so a straight
# percentage cut-off leaves the crowd standing behind the far touchline inside
# the frame — measured on game.mp4 it discarded nothing at all. Fitting the
# touchline as a curve removes ~10 non-players per frame instead.
TOUCHLINE_BUFFER_PX  = 20   # feet must be this far below the line to count
TOUCHLINE_FIT_FRAMES = 12   # frames sampled to fit the curve
TOUCHLINE_MIN_CONF   = 0.5  # keypoint confidence floor
TOUCHLINE_MAX_RESID  = 25   # px — reject the fit if the landmarks disagree more
# Fitting once only works if the camera holds still. Measured landmark spread
# across sampled frames: 18px on game.mp4's fixed camera, 210px on a broadcast
# clip that pans. A moving camera can still produce a low-residual fit, so
# checking the residual alone is not enough to catch it.
TOUCHLINE_MAX_MOTION_PX = 60

# Nested-box suppression. NMS drops a box when it overlaps another by IoU, but
# a small box sitting inside a large one has LOW IoU precisely because their
# areas differ — a quarter-sized box fully inside another scores 0.25, well
# under the 0.7 threshold, so it can never be suppressed. The detector emits a
# full-body box plus separate head/torso/leg boxes for the same player, and
# every survivor becomes its own track. Measured ~2.6 per frame on game.mp4.
#
# Training does not fix this: the custom model, trained on hand-cleaned labels
# with one box per player, produced 2.6/frame against the stock model's 2.4.
# 0.90 rather than 0.75: the duplicates are near-total overlaps, so a strict
# threshold still catches them. At 0.75 it also deleted players standing partly
# behind another — their box is >75% inside the front player's — which produced
# 5 over-merged IDs (one ID covering two players). Measured per frame:
#   0.75 -> 2.38 removed, 23.0 players    0.90 -> 1.38 removed, 24.0 players
CONTAINMENT_THRESHOLD = 0.90   # fraction of the smaller box inside the larger
# ================================================================

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']

VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(c) for c in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5, text_thickness=1, text_scale=0.5, text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'), thickness=2, edges=CONFIG.edges)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'), base=20, height=15)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS), thickness=2)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS), thickness=2)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5, text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5, text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    PITCH_DETECTION     = 'PITCH_DETECTION'
    PLAYER_DETECTION    = 'PLAYER_DETECTION'
    BALL_DETECTION      = 'BALL_DETECTION'
    PLAYER_TRACKING     = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR               = 'RADAR'
    FULL_ANALYSIS       = 'FULL_ANALYSIS'


# ================================================================
# HELPERS
# ================================================================

def load_player_model(device: str):
    """Load the player/detection model and align class IDs with its own names.

    Every mode goes through here so a swapped-in custom model can't quietly
    disagree with the hardcoded indices.
    """
    model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    sync_class_ids(model)
    return model


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def video_frames(source_video_path: str, stride: int = 1, max_frames: int = None,
                 start_frame: int = 0):
    """Frame generator, optionally starting late and stopping after max_frames.

    Seeks once to start_frame then reads sequentially. Seeking into a large
    high-bitrate recording is slow (the 14_09 stationary is 7.3GB for 18 min),
    so we pay that cost once rather than per frame.
    """
    if start_frame:
        # Skip by GRABBING frames rather than seeking. Seeking into a large
        # high-bitrate recording leaves the decoder in a state that fails a few
        # thousand frames later — on the 14_09 stationary a seek to 8 min died
        # at 10 min, while a straight sequential read decodes all 18 min. grab()
        # skips without the cost of fully decoding each frame.
        cap = cv2.VideoCapture(source_video_path)
        for _ in range(int(start_frame)):
            if not cap.grab():
                break

        def _gen():
            n = 0
            try:
                while max_frames is None or n < max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if n % stride == 0:
                        yield frame
                    n += 1
            finally:
                cap.release()
        return _gen()

    gen = sv.get_video_frames_generator(
        source_path=source_video_path, stride=stride)
    return itertools.islice(gen, max_frames) if max_frames else gen


# Set from --run_label, or a timestamp. Keeps each run's outputs distinct.
RUN_LABEL = None


def output_path_for(source_video_path: str, suffix: str) -> str:
    """Name outputs after the source clip AND the run.

    Naming by source alone is not enough: re-running the same video — to try a
    different threshold, say — still destroyed the previous result, including
    any player names filled in by hand. Every run now gets its own file.
    """
    stem = os.path.splitext(os.path.basename(source_video_path))[0]
    label = RUN_LABEL or datetime.now().strftime('%Y%m%d-%H%M')
    out_dir = os.path.join(PARENT_DIR, 'data', 'id_lists')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f'{suffix}_{stem}_{label}.json')


def in_pitch_boundary(cx_pct: float, cy_pct: float) -> bool:
    return (PITCH_LEFT_PCT <= cx_pct <= PITCH_RIGHT_PCT and
            PITCH_TOP_PCT  <= cy_pct <= PITCH_BOTTOM_PCT)


# Fitted by fit_far_touchline() at startup; None means fall back to the % box.
FAR_TOUCHLINE = None
# Built by build_pitch_polygon(), or loaded from --pitch_polygon.
PITCH_POLYGON = None
PITCH_POLYGON_PATH = None


def build_pitch_polygon_from_motion(source_video_path, frame_w, frame_h,
                                    n_samples=40, cell=64, static_frac=0.60):
    """Outline the pitch WITHOUT pitch markings, using motion.

    The landmark route below needs `football-pitch-detection.pt` to find
    markings, and on a dry worn pitch it finds ZERO at any confidence down to
    0.05 — so grounds like 14_09 get no polygon at all and fall back to the
    percentage bounds, which admit the crowd and the neighbouring field.

    This uses a signal that needs no markings: CROWD IS STATIC, PLAYERS MOVE.
    Histogram detection feet into cells across the clip. A cell occupied in
    most sampled frames is furniture — a spectator, a parked car, a clubhouse
    door. A cell occupied occasionally is pitch that players pass through. The
    far touchline sits just below the lowest static cell in each column.

    Columns with no static detection are filled from their neighbours: crowd
    lines a touchline continuously, so an empty column means detection missed
    them there, not that the crowd stopped. Without that fill the boundary
    drops to the frame top in gaps and lets the car park back in.

    Measured on 14_09 against ~24 people actually on the pitch:
        percentage bounds       31 detections/frame
        this method             24.2

    The near touchline is at or below the frame bottom on an elevated camera,
    so the polygon closes along the bottom edge.
    """
    import collections
    info = sv.VideoInfo.from_video_path(source_video_path)
    fps = info.fps or 30.0
    step = max(1, int(3 * fps))

    # Detect directly rather than through the pipeline's own helper: this runs
    # BEFORE any polygon exists, so it must not apply the filter it is trying
    # to build. Nested-box suppression is still wanted — duplicate boxes on one
    # spectator would look like a very static cell.
    if DETECTOR == 'rfdetr':
        import rfdetr_onnx
        detect = lambda f: rfdetr_onnx.detect(f, conf=INFERENCE_CONF)
    else:
        _m = load_player_model('mps')
        detect = lambda f: sv.Detections.from_ultralytics(
            _m(f, imgsz=INFERENCE_IMGSZ, conf=INFERENCE_CONF,
               agnostic_nms=True, verbose=False)[0])

    occ = collections.Counter()
    taken = n = 0
    for frame in video_frames(source_video_path, start_frame=START_FRAME):
        if n % step == 0:
            det = suppress_contained_boxes(detect(frame))
            if len(det) and det.class_id is not None:
                det = det[np.isin(det.class_id, [GOALKEEPER_CLASS_ID,
                                                 PLAYER_CLASS_ID,
                                                 REFEREE_CLASS_ID])]
                for (x1, _, x2, y2) in det.xyxy:
                    occ[(int((x1 + x2) / 2) // cell, int(y2) // cell)] += 1
            taken += 1
            if taken >= n_samples:
                break
        n += 1
    if taken < 5:
        print("  Pitch polygon: too few frames sampled for the motion method")
        return None

    n_cols = frame_w // cell + 1
    raw = {}
    for xb in range(n_cols):
        static = [yb for (x, yb), c in occ.items()
                  if x == xb and c >= static_frac * taken]
        raw[xb] = (max(static) + 1) * cell if static else None
    if all(v is None for v in raw.values()):
        print("  Pitch polygon: no static band found — no crowd to exclude")
        return None

    filled = []
    for xb in range(n_cols):
        if raw[xb] is not None:
            filled.append(raw[xb])
            continue
        near = [raw[x] for x in range(max(0, xb - 6), min(n_cols, xb + 7))
                if raw[x] is not None]
        filled.append(max(near) if near else 0)
    # Local max so the boundary never dips below a neighbouring column's crowd.
    smooth = [int(max(filled[max(0, i - 2):i + 3])) for i in range(n_cols)]

    cap = int(frame_h * 0.36)          # never cut more than the top third away
    pts = [[xb * cell, min(smooth[xb], cap)] for xb in range(0, n_cols, 3)]
    poly = np.array([[0, pts[0][1]]] + pts +
                    [[frame_w, smooth[-1]], [frame_w, frame_h], [0, frame_h]],
                    dtype=np.int32)
    print(f"  Pitch polygon (motion): {len(poly)} points, "
          f"far edge y={min(p[1] for p in pts)}..{max(p[1] for p in pts)}, "
          f"{cv2.contourArea(poly) / (frame_w * frame_h):.0%} of frame")
    return poly


def build_pitch_polygon(source_video_path, device, frame_w, frame_h):
    """Outline the playing area in image space.

    The far touchline alone is not enough. On the ultrawide footage the pitch
    occupies only x=1036..2888 of a 4096-wide frame at its far edge, so the
    percentage side-bounds (5%-95%) let in the neighbouring field entirely.

    Sides come from the goal lines: landmarks that share a pitch x-coordinate
    (0 or length) but differ in y give the angle the goal line leans at, which
    is what bounds the pitch left and right. Extended to the frame edges, and
    closed along the bottom since the near touchline sits below frame.
    """
    verts = np.array(CONFIG.vertices, dtype=np.float32)
    model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    info = sv.VideoInfo.from_video_path(source_video_path)
    stride = max(1, (info.total_frames or 1200) // (TOUCHLINE_FIT_FRAMES + 2))

    seen = {}
    for n, frame in enumerate(video_frames(source_video_path, stride=stride)):
        if n >= TOUCHLINE_FIT_FRAMES:
            break
        kp = sv.KeyPoints.from_ultralytics(model(frame, verbose=False)[0])
        if kp.xy is None or len(kp.xy) == 0:
            continue
        xy = kp.xy[0]
        conf = kp.confidence[0] if kp.confidence is not None else np.ones(len(xy))
        for i in range(len(xy)):
            if conf[i] > TOUCHLINE_MIN_CONF and xy[i][0] > 1 and xy[i][1] > 1:
                seen.setdefault(i, []).append((float(xy[i][0]), float(xy[i][1])))
    mean = {i: np.mean(v, axis=0) for i, v in seen.items() if len(v) >= 3}

    # Slope of a goal line in image space, from landmarks sharing a pitch x.
    slopes = []
    for target_x in (0.0, float(CONFIG.length)):
        pts = [(mean[i], verts[i][1]) for i in mean if verts[i][0] == target_x]
        if len(pts) < 2:
            continue
        pts.sort(key=lambda p: p[1])
        (p0, _), (p1, _) = pts[0], pts[-1]
        if abs(p1[1] - p0[1]) > 5:
            slopes.append(abs((p1[0] - p0[0]) / (p1[1] - p0[1])))
    if not slopes or FAR_TOUCHLINE is None:
        print("  Pitch polygon: not enough goal-line landmarks — "
              "falling back to the % side bounds")
        return None
    slope = float(np.mean(slopes))

    corners = [mean[i][0] for i in mean if verts[i][1] == 0]
    if len(corners) < 2:
        print("  Pitch polygon: no far-touchline corners — using % side bounds")
        return None
    x_left, x_right = min(corners), max(corners)

    buf = TOUCHLINE_BUFFER_PX
    top = [[int(x), int(FAR_TOUCHLINE(x) + buf)]
           for x in np.linspace(x_left, x_right, 16)]
    y_left  = FAR_TOUCHLINE(x_left) + buf + x_left / slope
    y_right = FAR_TOUCHLINE(x_right) + buf + (frame_w - x_right) / slope
    poly = np.array([[0, int(y_left)]] + top +
                    [[frame_w, int(y_right)], [frame_w, frame_h], [0, frame_h]],
                    dtype=np.int32)
    print(f"  Pitch polygon: far edge x={x_left:.0f}..{x_right:.0f}, "
          f"goal-line slope {slope:.2f}px/px, buffer {buf}px")
    return poly


def fit_far_touchline(source_video_path: str, device: str):
    """Fit the far touchline as a curve in image space: x -> y.

    Why not map to real pitch coordinates and test against the rectangle? On
    this footage a homography carries ~150cm of error even on the landmarks it
    was fitted from, because the panoramic stitch bows the pitch and a
    homography can only map straight lines to straight lines. A quadratic
    through the same landmarks fits to about a pixel — worse geometry, but all
    we need in order to tell players from the crowd standing behind them.

    Returns np.poly1d, or None if the pitch model can't place enough landmarks.
    """
    verts   = np.array(CONFIG.vertices, dtype=np.float32)
    far_idx = set(np.where(verts[:, 1] == 0)[0].tolist())   # landmarks on y=0

    info  = sv.VideoInfo.from_video_path(source_video_path)
    total = info.total_frames or 0
    stride = max(1, total // (TOUCHLINE_FIT_FRAMES + 2)) if total else 100

    model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    xs, ys, seen = [], [], set()
    per_landmark = {}
    frames = sv.get_video_frames_generator(
        source_path=source_video_path, stride=stride)
    for n, frame in enumerate(frames):
        if n >= TOUCHLINE_FIT_FRAMES:
            break
        kp = sv.KeyPoints.from_ultralytics(model(frame, verbose=False)[0])
        # A zoomed-in or panning shot can show no recognisable pitch landmarks
        # at all, in which case the keypoint model returns nothing.
        if kp.xy is None or len(kp.xy) == 0:
            continue
        xy = kp.xy[0]
        conf = (kp.confidence[0] if kp.confidence is not None
                else np.ones(len(xy)))
        for i in range(len(xy)):
            if (i in far_idx and conf[i] > TOUCHLINE_MIN_CONF
                    and xy[i][0] > 1 and xy[i][1] > 1):
                xs.append(float(xy[i][0]))
                ys.append(float(xy[i][1]))
                seen.add(i)
                per_landmark.setdefault(i, []).append(
                    (float(xy[i][0]), float(xy[i][1])))

    # Does the same landmark stay put between frames? If not the camera is
    # panning, one fitted curve cannot describe the whole clip, and the
    # residual won't necessarily reveal that on its own.
    spread = [float(np.linalg.norm(np.array(pts).std(axis=0)))
              for pts in per_landmark.values() if len(pts) >= 2]
    if spread and np.median(spread) > TOUCHLINE_MAX_MOTION_PX:
        print(f"  Touchline fit: landmarks move {np.median(spread):.0f}px "
              f"between frames — camera isn't stationary, falling back to "
              f"the {PITCH_TOP_PCT}% boundary")
        return None

    # A quadratic needs landmarks at three different x positions; with two we
    # can still fit a line, and with fewer there is nothing to fit.
    degree = 2 if len(seen) >= 3 else 1
    if len(seen) < 2 or len(xs) < degree + 1:
        print(f"  Touchline fit: only {len(seen)} landmark(s) found "
              f"— falling back to the {PITCH_TOP_PCT}% boundary")
        return None

    curve = np.poly1d(np.polyfit(np.array(xs), np.array(ys), degree))
    resid = np.abs(np.array(ys) - curve(np.array(xs)))
    if np.median(resid) > TOUCHLINE_MAX_RESID:
        print(f"  Touchline fit: landmarks disagree by "
              f"{np.median(resid):.0f}px — falling back to the % boundary")
        return None

    print(f"  Touchline fitted from {len(xs)} samples across {len(seen)} "
          f"landmarks (degree {degree}, residual {np.median(resid):.1f}px "
          f"median / {resid.max():.1f}px max)")
    return curve


def suppress_contained_boxes(
    detections: sv.Detections,
    threshold: float = CONTAINMENT_THRESHOLD
) -> sv.Detections:
    """Drop boxes that sit mostly inside a larger box.

    See CONTAINMENT_THRESHOLD for why NMS cannot do this itself.

    Only the *smaller* box of a nested pair is dropped, regardless of
    confidence: a confident torso-only box is still wrong. Two players standing
    side by side overlap at the edges rather than nesting, so they survive.
    """
    n = len(detections)
    if n < 2:
        return detections

    xyxy = detections.xyxy
    areas = ((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])).astype(float)
    drop = np.zeros(n, dtype=bool)

    for i in range(n):
        if drop[i]:
            continue
        for j in range(n):
            if i == j or drop[j] or areas[j] >= areas[i] or areas[j] <= 0:
                continue
            ix1 = max(xyxy[i][0], xyxy[j][0])
            iy1 = max(xyxy[i][1], xyxy[j][1])
            ix2 = min(xyxy[i][2], xyxy[j][2])
            iy2 = min(xyxy[i][3], xyxy[j][3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / areas[j] > threshold:
                drop[j] = True

    if not drop.any():
        return detections
    return detections[~drop]


def clean_detections(
    detections: sv.Detections,
    frame_w: int,
    frame_h: int
) -> sv.Detections:
    """Remove nested duplicate boxes, then anything off the pitch."""
    if len(detections) == 0:
        return detections

    detections = suppress_contained_boxes(detections)
    if len(detections) == 0:
        return detections

    xyxy   = detections.xyxy
    cx_pct = ((xyxy[:, 0] + xyxy[:, 2]) / 2) / frame_w * 100
    cy_pct = ((xyxy[:, 1] + xyxy[:, 3]) / 2) / frame_h * 100

    # Sides and near edge always apply.
    mask = ((cx_pct >= PITCH_LEFT_PCT) & (cx_pct <= PITCH_RIGHT_PCT) &
            (cy_pct <= PITCH_BOTTOM_PCT))

    # Feet, not box centre: "is this person standing on the grass" is a
    # question about the ground plane.
    feet = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    if PITCH_POLYGON is not None:
        # The polygon already bounds left/right by the goal lines, which is
        # stricter and more accurate than the percentage side-bounds.
        mask = np.array([
            cv2.pointPolygonTest(PITCH_POLYGON, (float(x), float(y)), False) >= 0
            for x, y in feet])
    elif FAR_TOUCHLINE is not None:
        mask &= feet[:, 1] > FAR_TOUCHLINE(feet[:, 0]) + TOUCHLINE_BUFFER_PX
    else:
        mask &= cy_pct >= PITCH_TOP_PCT

    if not mask.any():
        return sv.Detections.empty()
    return detections[mask]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.ndarray,
    goalkeepers: sv.Detections
) -> np.ndarray:
    goalkeepers_xy  = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy      = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    result = []
    for gxy in goalkeepers_xy:
        d0 = np.linalg.norm(gxy - team_0_centroid)
        d1 = np.linalg.norm(gxy - team_1_centroid)
        result.append(0 if d0 < d1 else 1)
    return np.array(result)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy             = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(xy)
    radar = draw_pitch(config=CONFIG)
    for i, color in enumerate(COLORS):
        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=transformed_xy[color_lookup == i],
            face_color=sv.Color.from_hex(color),
            radius=20, pitch=radar)
    return radar


# ================================================================
# PLAYER RE-ID TRACKER
# ================================================================

class PlayerReIDTracker:

    def __init__(self, frame_width: int, frame_height: int, fps: float = 30.0,
                 device: str = 'cpu'):
        self.fps           = fps or 30.0
        self.window_frames = max(1, int(REID_WINDOW_SECONDS * self.fps))
        self.min_frames    = max(1, int(MIN_SECONDS_TO_KEEP * self.fps))
        self.tracker = sv.ByteTrack(
            minimum_consecutive_frames=1,
            lost_track_buffer=self.window_frames,
            track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
            minimum_matching_threshold=TRACK_MATCHING_THRESHOLD,
        )
        self.frame_diag = np.sqrt(frame_width**2 + frame_height**2)
        self.frame_w    = frame_width
        self.frame_h    = frame_height
        self.id_map:    dict = {}
        self.last_seen: dict = {}
        self.id_frame_count: defaultdict = defaultdict(int)
        self.id_first_pos:   dict = {}
        self.id_last_pos:    dict = {}
        self.id_path_px:     defaultdict = defaultdict(float)
        self.id_sample_pos:  dict = {}   # cid -> (x, y, frame) last sampled
        self.id_appearance:  dict = {}   # cid -> SigLIP embedding
        self.id_class_votes: defaultdict = defaultdict(lambda: defaultdict(int))
        self.id_history:     defaultdict = defaultdict(list)   # cid -> [(frame,x,y)]
        self.id_crops:       defaultdict = defaultdict(list)   # cid -> [crop]
        self.id_team:        dict = {}
        self.id_splits:      dict = {}   # cid -> [(from_frame, new_cid), ...]
        self._claimed_this_frame: set = set()
        self._id_counter = 100000      # fresh ids start well clear of ByteTrack's
        self.id_embed_frame: dict = {}   # cid -> frame its embedding was taken
        self.device = device
        self._embedder = None
        self.frame_n = 0

    def _next_free_id(self) -> int:
        """A canonical id not yet used by any track."""
        self._id_counter = max(self._id_counter + 1,
                               max(self.id_frame_count.keys(), default=0) + 1)
        return self._id_counter

    def _centre(self, xyxy):
        return np.array([(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2])

    def _embed(self, crops):
        """SigLIP embeddings for a batch of crops.

        Loaded on first use so runs with APPEARANCE_WEIGHT=0 never pay for it.
        """
        if not crops:
            return []
        if self._embedder is None:
            self._embedder = TeamClassifier(device=self.device, batch_size=32)
        try:
            return list(self._embedder.extract_features(crops))
        except Exception as e:
            # Never let appearance take down a 30-minute tracking run; fall
            # back to pure-distance matching for this frame.
            print(f"  [appearance] embedding failed, using distance only: "
                  f"{type(e).__name__}")
            return [None] * len(crops)

    @staticmethod
    def _similarity(a, b):
        if a is None or b is None:
            return None
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na and nb else None

    def _find_lost_match(self, cx, cy, emb=None):
        """Find which lost track this detection most likely continues.

        Candidates must fall inside the spatial radius — see
        REID_DISTANCE_FRACTION for the tighter variants that measured worse.
        Among those, appearance breaks the tie; see APPEARANCE_WEIGHT for why
        it ranks rather than vetoes.
        """
        radius = REID_DISTANCE_FRACTION * self.frame_diag
        best_id, best_score = None, -1.0
        for cid, (lx, ly, last_frame) in self.last_seen.items():
            frames_ago = self.frame_n - last_frame
            # A candidate must be genuinely LOST, not merely unseen this frame.
            # Claiming a still-active id is how one id ends up on two players at
            # once — measured on 14_09, 37 of 46 ids held two players at some
            # point before this guard existed.
            if frames_ago < REID_MIN_LOST_FRAMES or frames_ago > self.window_frames:
                continue
            if cid in self._claimed_this_frame:
                continue
            dist = np.sqrt((cx-lx)**2 + (cy-ly)**2)
            if dist >= radius:
                continue

            score = 1.0 - dist / radius            # 1 = touching, 0 = at the edge
            sim = self._similarity(emb, self.id_appearance.get(cid))
            if sim is not None:
                score = ((1 - APPEARANCE_WEIGHT) * score
                         + APPEARANCE_WEIGHT * max(0.0, sim))
            if score > best_score:
                best_score, best_id = score, cid
        return best_id

    def _accumulate_path(self, cid, cx, cy):
        """Add this ID's travel since its last sampled position.

        Sampling every MOVE_SAMPLE_EVERY frames rather than every frame, and
        ignoring sub-MOVE_MIN_STEP_PX steps, keeps box jitter from accumulating
        into a large fake distance for something that never actually moves.
        """
        prev = self.id_sample_pos.get(cid)
        if prev is None:
            self.id_sample_pos[cid] = (cx, cy, self.frame_n)
            return
        px, py, pframe = prev
        gap = self.frame_n - pframe
        if gap < MOVE_SAMPLE_EVERY:
            return
        step = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        # A large jump after a long absence is a re-id landing on someone else,
        # not a player covering ground, so don't credit it as travel.
        if step >= MOVE_MIN_STEP_PX and gap <= MOVE_SAMPLE_EVERY * 3:
            self.id_path_px[cid] += step
        self.id_sample_pos[cid] = (cx, cy, self.frame_n)

    def update(self, detections: sv.Detections, frame=None) -> sv.Detections:
        detections = self.tracker.update_with_detections(detections)
        if detections.tracker_id is None or len(detections) == 0:
            self.frame_n += 1
            return sv.Detections.empty()

        # Decide which detections actually need an embedding this frame, and
        # do them in ONE batch — a per-crop transformer call every frame would
        # dominate the runtime. Needed for new tracks (to decide who they are)
        # and periodically for known tracks (to keep the signature current).
        embeddings = {}
        if frame is not None and APPEARANCE_WEIGHT > 0:
            wanted, crops = [], []
            for i, raw_id in enumerate(detections.tracker_id):
                cid = self.id_map.get(int(raw_id))
                stale = (cid is None or
                         self.frame_n - self.id_embed_frame.get(cid, -10**9)
                         >= APPEARANCE_REFRESH_FRAMES)
                if not stale:
                    continue
                x1, y1, x2, y2 = [int(v) for v in detections.xyxy[i]]
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if (crop.size == 0 or crop.shape[0] < APPEARANCE_MIN_CROP_PX
                        or crop.shape[1] < 4):
                    continue
                wanted.append(i)
                crops.append(crop)
            for i, emb in zip(wanted, self._embed(crops)):
                embeddings[i] = emb

        self._claimed_this_frame = {
            self.id_map[int(r)] for r in detections.tracker_id
            if int(r) in self.id_map
        }
        canonical_ids = []
        used_this_frame: set = set()
        for i, raw_id in enumerate(detections.tracker_id):
            raw_id = int(raw_id)
            cx, cy = self._centre(detections.xyxy[i])
            emb = embeddings.get(i)

            if raw_id in self.id_map:
                cid = self.id_map[raw_id]
            else:
                matched = self._find_lost_match(cx, cy, emb)
                cid     = matched if matched is not None else raw_id
                self.id_map[raw_id] = cid

            # HARD INVARIANT: one canonical id per detection per frame.
            #
            # Re-id hands a lost track's id to a new track, but ByteTrack keeps
            # its own lost-track buffer and can REVIVE the original later. Both
            # raw tracks then point at one canonical id and collide on every
            # frame after. Measured before this guard: 82% of frames contained
            # at least one doubled id, and every case came from a mapping made
            # earlier — so preventing it at adoption time is not enough.
            #
            # Whoever arrives second gets a fresh id. A split is visible and
            # fixable later; a merge silently blends two players' positions.
            if cid in used_this_frame:
                cid = self._next_free_id()
                self.id_map[raw_id] = cid
            used_this_frame.add(cid)
            canonical_ids.append(cid)
            self.last_seen[cid] = (cx, cy, self.frame_n)
            if emb is not None:
                # Blend rather than replace: a single blurred or occluded crop
                # shouldn't overwrite an otherwise stable signature.
                prev = self.id_appearance.get(cid)
                self.id_appearance[cid] = (
                    emb if prev is None else 0.7 * prev + 0.3 * emb)
                self.id_embed_frame[cid] = self.frame_n
            self.id_frame_count[cid] += 1
            if detections.class_id is not None:
                self.id_class_votes[cid][int(detections.class_id[i])] += 1
            if cid not in self.id_first_pos:
                self.id_first_pos[cid] = (cx, cy)
            self.id_last_pos[cid] = (cx, cy)
            # Keep a few crops per id for team colour. Sampled, not every
            # frame: consecutive crops of one player are near-identical, so
            # they add memory without adding evidence.
            if (STITCH and frame is not None and
                    len(self.id_crops[cid]) < TEAM_CROPS_PER_ID and
                    self.frame_n % TEAM_CROP_STRIDE == 0):
                x1, y1, x2, y2 = [int(v) for v in detections.xyxy[i]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 - x1 >= 8 and y2 - y1 >= 16:
                    self.id_crops[cid].append(frame[y1:y2, x1:x2].copy())
            if TRACK_DUMP or STITCH:
                # Stitching needs this too, not just --track_dump. Roughly
                # 80 bytes per detection per frame: ~260MB over a 45-minute
                # match at 55fps, which is fine, but it is why this stays gated
                # rather than always on.
                # Box height comes along because it is our only scale reference.
                # Pixel speed is meaningless on its own — 40px/frame near the
                # camera is a jog, the same 40px at the far touchline is
                # teleportation — but height is proportional to closeness, so
                # distance/height is a perspective-corrected speed that needs no
                # homography. See split_implausible_tracks().
                box_h = float(detections.xyxy[i][3] - detections.xyxy[i][1])
                self.id_history[cid].append((self.frame_n, float(cx), float(cy),
                                             box_h))
            self._accumulate_path(cid, cx, cy)

        detections = sv.Detections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
            tracker_id=np.array(canonical_ids, dtype=int)
        )
        self.frame_n += 1
        return detections

    def dominant_class(self, cid):
        """The class this ID was labelled most often, or None."""
        votes = self.id_class_votes.get(cid)
        return max(votes, key=votes.get) if votes else None

    def net_displacement(self, cid) -> float:
        """Straight-line distance from first sighting to last.

        Reported for context only — it is not a movement filter. A player who
        ends the clip near where they started scores close to zero on it.
        """
        if cid not in self.id_first_pos or cid not in self.id_last_pos:
            return 0.0
        fx, fy = self.id_first_pos[cid]
        lx, ly = self.id_last_pos[cid]
        return float(np.sqrt((lx - fx) ** 2 + (ly - fy) ** 2))

    def split_implausible_tracks(self, fps: float):
        """Cut a track where it moves faster than a human can run.

        Everything else in this pipeline MERGES — re-id, two-tier tracking,
        stitching. Nothing ever cuts. So when ByteTrack hands one id to two
        different players, that splice is permanent and later stages stitch
        further onto it. On the 2D map it reads as a dot teleporting across the
        pitch. This is the only stage that can undo it.

        The physical claim is simple: a player cannot cross the pitch in half a
        second. Turning that into a test needs care, because PIXEL speed is not
        physical speed — 40px/frame near the camera is a jog and the same 40px
        at the far touchline is impossible. Box height is proportional to how
        close a player is, so `distance / box_height` is a scale-free speed:
        body-heights per second, which means the same thing anywhere in frame.
        That avoids needing the homography we cannot fit here.

        A sprint is ~10 m/s and a player is ~1.75m, so ~5.7 body-heights/sec.
        The threshold sits well above that: the goal is to catch teleports, not
        to adjudicate fast running, and a wrong cut costs a fragment that
        stitching may re-join anyway.

        Measured over a short window rather than frame to frame — at 55fps one
        frame is 18ms, over which detection jitter rivals real motion.
        """
        splits = 0
        self.id_splits = {}
        for cid in list(self.id_history.keys()):
            hist = sorted(self.id_history[cid])
            if len(hist) < 4:
                continue
            frames = np.array([h[0] for h in hist], dtype=float)
            pos = np.array([[h[1], h[2]] for h in hist], dtype=float)
            heights = np.array([h[3] if len(h) > 3 else 0.0 for h in hist])
            # Smooth before differencing. Raw box centres jitter by tens of
            # pixels frame to frame, and at 55fps a single frame is 18ms, so
            # dividing raw jitter by that dt manufactures impossible speeds. A
            # first attempt without this cut 1075 tracks and turned 58 ids into
            # 595 — measuring noise, not teleports.
            if len(pos) >= 5:
                k = np.ones(5) / 5
                pad = np.vstack([np.repeat(pos[:1], 2, axis=0), pos,
                                 np.repeat(pos[-1:], 2, axis=0)])
                pos = np.stack([np.convolve(pad[:, 0], k, 'valid'),
                                np.convolve(pad[:, 1], k, 'valid')], axis=1)

            win = max(1, int(SPEED_WINDOW_SECONDS * fps))
            cuts = []
            for i in range(1, len(hist)):
                gap = frames[i] - frames[i - 1]
                if gap > 1:
                    # Across a real tracking gap, compare the two ends directly:
                    # dt is large enough that jitter is irrelevant, and this is
                    # exactly where an id gets handed to a different player.
                    j = i - 1
                else:
                    j = max(0, i - win)
                    if frames[i] - frames[j] < win * 0.5:
                        continue                # not enough baseline yet
                dt = (frames[i] - frames[j]) / fps
                if dt <= 0:
                    continue
                h = max(float(np.mean(heights[j:i + 1])), 1e-6)
                if h < 8:                      # too small to trust as a ruler
                    continue
                dist = float(np.hypot(pos[i][0] - pos[j][0],
                                      pos[i][1] - pos[j][1]))
                if dist / h / dt > MAX_BODY_HEIGHTS_PER_SEC:
                    cuts.append(i)
            # Collapse cuts that fire on consecutive samples — one teleport
            # trips the test for a whole window, and each extra cut is a
            # spurious identity.
            cuts = [c for k, c in enumerate(cuts)
                    if k == 0 or c - cuts[k - 1] > win]
            if not cuts:
                continue
            # Everything from each cut onward becomes a new identity.
            boundaries = []
            for c in cuts:
                new_cid = self._next_free_id()
                boundaries.append((hist[c][0], new_cid))
                splits += 1
            self.id_splits[cid] = boundaries

            # Rewrite this id's own state so stats and the map see the split.
            segs, start = [], 0
            for c in cuts + [len(hist)]:
                segs.append(hist[start:c])
                start = c
            ids = [cid] + [b[1] for b in boundaries]
            for seg_id, seg in zip(ids, segs):
                if not seg:
                    continue
                self.id_history[seg_id] = seg
                self.id_frame_count[seg_id] = len(seg)
                self.id_first_pos[seg_id] = (seg[0][1], seg[0][2])
                self.id_last_pos[seg_id] = (seg[-1][1], seg[-1][2])
                if seg_id != cid:
                    # Class votes cannot be split by time, so copy the parent's
                    # majority rather than losing the class entirely.
                    self.id_class_votes[seg_id][self.dominant_class(cid) or
                                                PLAYER_CLASS_ID] = len(seg)
                    self.id_crops[seg_id] = self.id_crops.get(cid, [])[:]
                path = 0.0
                for a, b in zip(seg[:-1], seg[1:]):
                    path += float(np.hypot(b[1] - a[1], b[2] - a[2]))
                self.id_path_px[seg_id] = path
        return splits

    def merge_stitched(self, fps: float):
        """Fold offline-stitched tracklets together into single identities.

        Runs after pass 1, when the whole clip is known. The online tracker had
        to decide frame by frame; this can look at the future, which is the
        only reason it can join fragments the tracker could not.

        Merging into the tracker's OWN state rather than remapping at render
        time keeps everything downstream consistent — the movement filter, the
        stats, the validation JSON and the drawn ids all see one identity. It
        also fixes a subtle bug in doing it later: a stitched player's speed is
        computed from their whole path, so two fragments that each looked like
        loitering no longer get filtered out separately.

        Pass 2 seeds its id_map from ours, so remapping id_map here is what
        makes the rendered ids match with no further plumbing.
        """
        import stitch_tracks as stitch

        # Team colour, if the kits allow it. Assigned per canonical id by
        # majority vote over that id's crops, then attached to every tracklet
        # cut from it — a tracklet is a segment of one id, so it inherits the
        # id's team. link_cost() turns a disagreement into a penalty, never a
        # veto, because the labelling is ~93% and a wrong veto silently leaves
        # a player fragmented.
        self.id_team = {}
        try:
            from team_colour import TeamColourClassifier
            crops_all = [c for v in self.id_crops.values() for c in v]
            if len(crops_all) >= 20:
                clf = TeamColourClassifier().fit(crops_all)
                if clf.centres is not None and clf.separation >= 1.0:
                    for cid, crops in self.id_crops.items():
                        t = clf.predict_tracklet(crops)
                        if t is not None:
                            self.id_team[cid] = t
                    counts = Counter(self.id_team.values())
                    print(f"Team colour: separation {clf.separation:.2f}, "
                          f"{dict(counts)} across {len(self.id_team)} ids")
                else:
                    sep = clf.separation if clf.centres is not None else 0.0
                    print(f"Team colour: separation {sep:.2f} — kits too "
                          f"similar to separate, skipping the team term")
        except ImportError:
            pass

        tracklets = []
        for cid, hist in self.id_history.items():
            if len(hist) < 2:
                continue
            frames = [int(h[0]) for h in hist]
            xy = np.array([[h[1], h[2]] for h in hist], dtype=np.float32)
            # Split internal gaps: a canonical id may already span a re-id
            # jump, and this stage should judge on its own evidence.
            breaks = [0] + [i for i in range(1, len(frames))
                            if frames[i] - frames[i - 1] > fps] + [len(frames)]
            for a, b in zip(breaks[:-1], breaks[1:]):
                if b - a >= 2:
                    tracklets.append(stitch.Tracklet(
                        cid, frames[a:b], xy[a:b],
                        self.dominant_class(cid) or PLAYER_CLASS_ID,
                        team=self.id_team.get(cid)))
        if len(tracklets) < 2:
            return 0, 0

        identities, links = stitch.stitch_global(tracklets, fps)

        # Each chain collapses onto its lowest id, which keeps numbers stable
        # and small rather than renaming everyone.
        remap = {}
        for chain in identities:
            ids = {tracklets[i].id for i in chain}
            if len(ids) < 2:
                continue
            keep = min(ids)
            for cid in ids:
                if cid != keep:
                    remap[cid] = keep
        if not remap:
            return len(links), sum(1 for l in links if l['thin'])

        for src, dst in remap.items():
            self.id_frame_count[dst] += self.id_frame_count.pop(src, 0)
            self.id_path_px[dst] += self.id_path_px.pop(src, 0.0)
            for k, v in self.id_class_votes.pop(src, {}).items():
                self.id_class_votes[dst][k] += v
            self.id_history[dst] = sorted(self.id_history[dst] +
                                          self.id_history.pop(src, []))
            self.id_first_pos.pop(src, None)
            self.id_last_pos.pop(src, None)
        # Endpoints have to come from the merged history, not from whichever
        # fragment happened to be written last, or net displacement is wrong.
        for dst in set(remap.values()):
            hist = self.id_history.get(dst)
            if hist:
                self.id_first_pos[dst] = (hist[0][1], hist[0][2])
                self.id_last_pos[dst] = (hist[-1][1], hist[-1][2])
        for raw, cid in list(self.id_map.items()):
            if cid in remap:
                self.id_map[raw] = remap[cid]
        return len(links), sum(1 for l in links if l['thin'])

    def valid_ids(self) -> set:
        valid = set()
        for cid, count in self.id_frame_count.items():
            if count < self.min_frames:
                continue
            # Goalkeepers are exempt from the movement test entirely. Standing
            # still IS the job, and a keeper who never leaves their box fails
            # both the speed floor and the displacement rescue — which is how
            # one ended up detected all clip but never drawn. The detector
            # identifies the class confidently (0.71 on ultrawide), so trust it
            # rather than inferring "not a player" from stillness.
            if self.dominant_class(cid) != GOALKEEPER_CLASS_ID:
                seconds = count / self.fps
                speed = self.id_path_px[cid] / seconds if seconds > 0 else 0.0
                if (speed < MIN_SPEED_PX_PER_SEC and
                        self.net_displacement(cid) < MIN_NET_DISPLACEMENT_PX):
                    continue
            valid.add(cid)
        return valid


# ================================================================
# MODE FUNCTIONS
# ================================================================

def run_pitch_detection(source_video_path: str, device: str,
                        max_frames: int = None) -> Iterator[np.ndarray]:
    model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    for frame in video_frames(source_video_path, max_frames=max_frames, start_frame=START_FRAME):
        result    = model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        annotated = VERTEX_LABEL_ANNOTATOR.annotate(
            frame.copy(), keypoints, CONFIG.labels)
        yield annotated


def run_player_detection(source_video_path: str, device: str,
                         max_frames: int = None) -> Iterator[np.ndarray]:
    model      = load_player_model(device)
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    for frame in video_frames(source_video_path, max_frames=max_frames, start_frame=START_FRAME):
        result     = model(frame, imgsz=INFERENCE_IMGSZ, conf=INFERENCE_CONF, agnostic_nms=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = clean_detections(
            detections, video_info.width, video_info.height)
        annotated  = BOX_ANNOTATOR.annotate(frame.copy(), detections)
        annotated  = BOX_LABEL_ANNOTATOR.annotate(annotated, detections)
        yield annotated


def run_ball_detection(source_video_path: str, device: str,
                       max_frames: int = None) -> Iterator[np.ndarray]:
    model          = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    ball_tracker   = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(img: np.ndarray) -> sv.Detections:
        return sv.Detections.from_ultralytics(
            model(img, imgsz=640, verbose=False)[0])

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )
    for frame in video_frames(source_video_path, max_frames=max_frames, start_frame=START_FRAME):
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        yield ball_annotator.annotate(frame.copy(), detections)


def run_player_tracking(
    source_video_path: str,
    device: str,
    focus_id: int = None,
    max_frames: int = None,
    no_render: bool = False
) -> Iterator[np.ndarray]:
    # RF-DETR path never calls the Ultralytics YOLO weights. Loading them here
    # only to discard them forces football-player-detection.pt onto every
    # --detector rfdetr run. Class IDs stay at the module defaults, which already
    # match rfdetr_onnx._CLASS_MAP (ball=0, gk=1, player=2, ref=3).
    model = None if DETECTOR == 'rfdetr' else load_player_model(device)
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    fps        = video_info.fps or 30

    def detect_raw(frame):
        """One inference per frame, shared by the player and ball paths.

        Pass 2 used to call the detector TWICE per frame — once for players,
        once for the ball — which is pure waste since a single forward pass
        already returns both classes. Folding them together roughly halves
        render time and lets pass 1 collect ball positions for free.
        """
        if DETECTOR == 'rfdetr':
            import rfdetr_onnx
            return rfdetr_onnx.detect(frame, conf=min(INFERENCE_CONF,
                                                      BALL_MIN_CONF))
        result = model(frame, imgsz=INFERENCE_IMGSZ,
                       conf=min(INFERENCE_CONF, BALL_MIN_CONF),
                       agnostic_nms=True, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    def get_player_detections(frame, raw=None):
        detections = detect_raw(frame) if raw is None else raw
        if len(detections) and detections.confidence is not None:
            detections = detections[detections.confidence >= INFERENCE_CONF]
        detections = clean_detections(
            detections, video_info.width, video_info.height)
        if detections.class_id is not None and len(detections) > 0:
            wanted = [PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID]
            if INCLUDE_REFEREES:
                wanted.append(REFEREE_CLASS_ID)
            detections = detections[np.isin(detections.class_id, wanted)]
        return detections

    _ball_recent = deque(maxlen=BALL_HISTORY_FRAMES)

    def get_ball(frame, raw=None):
        """The ball, with the three checks the raw detections lack.

        Measured over 33s of 14_09: 1473 ball detections above 0.30, in 1107 of
        1800 frames, with 303 frames offering more than one. Only 309 of the
        1473 were inside the pitch polygon, and a single 256px-wide x band held
        853 of them — a goalpost, detected as a ball over and over.

        Size cannot separate them: false balls measured 19px median against the
        real ball's 19px. Position and persistence can.

          1. inside the pitch polygon — removed ~79% on this footage
          2. not in a STATIC hotspot — the same trick that fixed the pitch
             polygon. A real ball moves; a cell that keeps producing a ball
             across many recent frames is furniture. Rolling, so it needs no
             extra pass over the video.
          3. one ball per frame — there is only one, so keep the most
             confident survivor rather than drawing every candidate.
        """
        if not SHOW_BALL:
            return None
        d = detect_raw(frame) if raw is None else raw
        if len(d) and d.confidence is not None:
            d = d[d.confidence >= BALL_MIN_CONF]
        if len(d) == 0 or d.class_id is None:
            return None
        b = d[d.class_id == BALL_CLASS_ID]
        if len(b) == 0:
            return None

        cx = (b.xyxy[:, 0] + b.xyxy[:, 2]) / 2
        cy = (b.xyxy[:, 1] + b.xyxy[:, 3]) / 2
        cells = [(int(x) // BALL_CELL_PX, int(y) // BALL_CELL_PX)
                 for x, y in zip(cx, cy)]

        keep = np.ones(len(b), dtype=bool)
        if PITCH_POLYGON is not None:
            keep &= np.array([
                cv2.pointPolygonTest(PITCH_POLYGON, (float(x), float(y)),
                                     False) >= 0
                for x, y in zip(cx, b.xyxy[:, 3])])
        if len(_ball_recent) >= BALL_HISTORY_FRAMES // 2:
            seen = Counter(c for frame_cells in _ball_recent
                           for c in set(frame_cells))
            limit = BALL_STATIC_FRACTION * len(_ball_recent)
            keep &= np.array([seen[c] < limit for c in cells])

        # History records every candidate, filtered or not — a hotspot has to
        # stay visible for the suppressor to keep suppressing it.
        _ball_recent.append(cells)

        b = b[keep]
        if len(b) == 0:
            return None
        return b[[int(np.argmax(b.confidence))]]

    ball_history = []

    # ---- PASS 1: build lifetime stats (no frames stored in memory) ----
    print("Pass 1: building tracker lifetime stats...")
    tracker1 = PlayerReIDTracker(video_info.width, video_info.height, fps, device)
    for frame in tqdm(
        video_frames(source_video_path, max_frames=max_frames, start_frame=START_FRAME),
        desc='Pass 1'
    ):
        raw = detect_raw(frame)
        tracker1.update(get_player_detections(frame, raw), frame)
        # Ball in pass 1 costs nothing now that detection is shared, and gives
        # the map a full trajectory to smooth rather than a strobing marker.
        if SHOW_BALL:
            b = get_ball(frame, raw)
            if b is not None and len(b):
                ball_history.append((tracker1.frame_n,
                                     float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                                     float(b.xyxy[0][3])))

    raw_id_count = len(tracker1.id_frame_count)
    if SPLIT_IMPLAUSIBLE:
        # Cut BEFORE stitching: undo bad splices first, then let stitching
        # re-link the pieces on their own evidence (including team colour).
        n_splits = tracker1.split_implausible_tracks(fps)
        if n_splits:
            print(f"\nPhysics check: {n_splits} track(s) cut where motion "
                  f"exceeded {MAX_BODY_HEIGHTS_PER_SEC} body-heights/sec "
                  f"— an id had been handed between two players")
    if STITCH:
        n_links, n_thin = tracker1.merge_stitched(fps)
        print(f"\nStitching: {n_links} links ({n_thin} thin), "
              f"{raw_id_count} -> {len(tracker1.id_frame_count)} identities")
        if n_thin:
            print(f"  {n_thin} links had a close runner-up. Thin links measured "
                  f"29% correct against 93% for confident ones — check these "
                  f"ids first when validating.")

    good_ids = tracker1.valid_ids()
    all_ids  = set(tracker1.id_frame_count.keys())

    print(f"\nPass 1 complete:")
    print(f"  Total canonical IDs : {len(all_ids)}")
    print(f"  Passed filters      : {len(good_ids)}")
    print(f"  Removed as noise    : {len(all_ids) - len(good_ids)}")
    print(f"  Valid IDs           : {sorted(good_ids)}")

    if focus_id is not None and focus_id not in good_ids:
        print(f"  Warning: #{focus_id} didn't pass filters — showing anyway")
        good_ids.add(focus_id)

    # Save ID list JSON for manual validation
    id_stats = []
    for cid in sorted(all_ids):
        id_stats.append({
            "canonical_id":        int(cid),
            "frames_seen":         int(tracker1.id_frame_count[cid]),
            "path_length_px":      round(float(tracker1.id_path_px[cid]), 1),
            "speed_px_per_sec":    round(float(tracker1.id_path_px[cid]) /
                                         max(tracker1.id_frame_count[cid] / fps, 1e-6), 1),
            "net_displacement_px": round(tracker1.net_displacement(cid), 1),
            "class":               {BALL_CLASS_ID: 'ball',
                                    GOALKEEPER_CLASS_ID: 'goalkeeper',
                                    PLAYER_CLASS_ID: 'player',
                                    REFEREE_CLASS_ID: 'referee'}.get(
                                        tracker1.dominant_class(cid), 'unknown'),
            "passed_filter":       bool(cid in good_ids),
            "player_name":         None,
            "team":                None,
        })
    id_path = output_path_for(source_video_path, 'player_id_list')
    with open(id_path, 'w') as f:
        json.dump({"ids": id_stats}, f, indent=2)
    if TRACK_DUMP:
        dump = {'fps': float(fps), 'width': int(video_info.width),
                'height': int(video_info.height), 'tracks': []}
        for cid, hist in tracker1.id_history.items():
            if len(hist) < 2:
                continue
            dump['tracks'].append({
                'id': int(cid),
                'class': int(tracker1.dominant_class(cid) or PLAYER_CLASS_ID),
                'frames': [int(h[0]) for h in hist],
                'xy': [[round(h[1], 1), round(h[2], 1)] for h in hist],
            })
        dpath = output_path_for(source_video_path, 'track_dump')
        with open(dpath, 'w') as f:
            json.dump(dump, f)
        print(f"Track dump saved to: {dpath}  ({len(dump['tracks'])} tracklets)")

    print(f"\nID list saved to: {id_path}")
    print("Open it, fill in player_name and team for each ID.\n")

    if no_render:
        # Every number above comes from pass 1; pass 2 exists only to produce
        # the video. Skipping it roughly halves a calibration run.
        print("Skipping pass 2 (--no_render).")
        return

    # ---- PASS 2: render output video (video read again from disk) ----
    print("Pass 2: rendering output video...")
    tracker2 = PlayerReIDTracker(video_info.width, video_info.height, fps, device)
    # Seed the id_map so canonical IDs match pass 1
    tracker2.id_map = tracker1.id_map.copy()

    FOCUS_COLOUR = (0, 255, 128)
    trail        = []
    frame_n      = 0

    # The 2D map is built from PASS 1's completed history, not from what pass 2
    # happens to detect. That is the whole point of a second pass: a player lost
    # at frame 100 and found again at 160 can be drawn through the gap using
    # both ends, where the live tracker could only have guessed forward.
    minimap_obj = None
    timeline = ball_timeline = {}
    if SHOW_MINIMAP:
        import minimap as mm
        timeline = mm.build_timeline(
            tracker1.id_history, fps,
            id_team=tracker1.id_team,
            id_class={c: tracker1.dominant_class(c)
                      for c in tracker1.id_frame_count},
            keep_ids=good_ids)
        ball_timeline = mm.build_ball_timeline(ball_history, fps)
        # Bounds from where players ACTUALLY went, not from the pitch polygon.
        # The polygon runs to the frame bottom because the near touchline is off
        # frame, but players never reach there, so polygon bounds left the lower
        # half of the panel empty and squashed everyone into the top 40%.
        bounds = mm.bounds_from_timeline(
            timeline, fallback=(0, 0, video_info.width, video_info.height))
        minimap_obj = mm.Minimap(bounds, video_info.width, video_info.height,
                                 fps, corner=MINIMAP_CORNER)
        n_ghost = sum(1 for recs in timeline.values() for r in recs if r['ghost'])
        n_real = sum(len(r) for r in timeline.values()) - n_ghost
        print(f"Minimap: {n_real} measured positions, {n_ghost} inferred "
              f"across gaps ({n_ghost / max(n_real + n_ghost, 1):.0%}), "
              f"ball in {len(ball_timeline)} frames")
        map_trails = defaultdict(list)

    # Splits cannot travel through id_map: merging is time-independent (all raw
    # ids of merged tracks collapse to one), but a split means the SAME raw id
    # is a different player before and after a frame, which id_map has no way to
    # express. Pass 2 re-runs the same deterministic tracking, so its ids match
    # pass 1's pre-split ids and can be remapped per frame here instead.
    split_map = dict(getattr(tracker1, 'id_splits', {}) or {})

    def apply_splits(cid, frame_no):
        bounds = split_map.get(cid)
        if not bounds:
            return cid
        out = cid
        for from_frame, new_cid in bounds:
            if frame_no >= from_frame:
                out = new_cid
        return out

    for frame in video_frames(source_video_path, max_frames=max_frames, start_frame=START_FRAME):
        raw        = detect_raw(frame)
        detections = get_player_detections(frame, raw)
        ball       = get_ball(frame, raw)
        detections = tracker2.update(detections, frame)
        if split_map and detections.tracker_id is not None and len(detections):
            detections.tracker_id = np.array(
                [apply_splits(int(t), tracker2.frame_n)
                 for t in detections.tracker_id])
        annotated  = frame.copy()
        if ball is not None:
            for bb, bc in zip(ball.xyxy, ball.confidence):
                cx, cy = int((bb[0]+bb[2])/2), int((bb[1]+bb[3])/2)
                r = max(10, int((bb[2]-bb[0])))
                cv2.circle(annotated, (cx, cy), r, (0, 255, 255), 3)
                cv2.putText(annotated, f'BALL {bc:.2f}', (cx-30, cy-r-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if detections.tracker_id is not None and len(detections) > 0:
            # Keep only IDs that passed filters
            valid_mask = np.isin(detections.tracker_id, list(good_ids))
            detections = detections[valid_mask]

        if detections.tracker_id is not None and len(detections) > 0:
            if focus_id is not None:
                focus_mask = detections.tracker_id == focus_id
                others     = detections[~focus_mask]
                focused    = detections[focus_mask]

                # Dim all other players
                for i in range(len(others)):
                    box = others.xyxy[i]
                    cx  = int((box[0]+box[2])/2)
                    bot = int(box[3])
                    cv2.ellipse(annotated, (cx, bot),
                                (int((box[2]-box[0])/2), 8),
                                0, -45, 235, (80, 80, 80), 2)

                # Highlight focused player with trail
                if len(focused) > 0:
                    box = focused.xyxy[0]
                    cx  = int((box[0]+box[2])/2)
                    bot = int(box[3])
                    cv2.ellipse(annotated, (cx, bot),
                                (int((box[2]-box[0])/2), 10),
                                0, -45, 235, FOCUS_COLOUR, 3)
                    cv2.putText(annotated, f"#{focus_id}",
                                (cx-20, bot+22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                FOCUS_COLOUR, 2)
                    trail.append((cx, bot))
                    if len(trail) > 60:
                        trail.pop(0)
                    for j in range(1, len(trail)):
                        alpha = j / len(trail)
                        col   = tuple(int(c * alpha) for c in FOCUS_COLOUR)
                        cv2.line(annotated, trail[j-1], trail[j], col, 2)
            else:
                labels    = [f"#{int(tid)}" for tid in detections.tracker_id]
                annotated = ELLIPSE_ANNOTATOR.annotate(annotated, detections)
                annotated = ELLIPSE_LABEL_ANNOTATOR.annotate(
                    annotated, detections, labels=labels)

        # HUD counter
        cv2.rectangle(annotated, (0, 0), (360, 36), (0, 0, 0), -1)
        n_visible = len(detections) if detections.tracker_id is not None else 0
        cv2.putText(annotated,
                    f"Visible: {n_visible}  |  Unique IDs: {len(good_ids)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1)

        if minimap_obj is not None:
            recs = timeline.get(frame_n, [])
            for r in recs:
                map_trails[r['id']].append(r['xy'])
                if len(map_trails[r['id']]) > minimap_obj.trail_frames:
                    map_trails[r['id']].pop(0)
            annotated = minimap_obj.draw(
                annotated, recs, ball_xy=ball_timeline.get(frame_n),
                trails=map_trails)

        frame_n += 1
        yield annotated

    print(f"\nDone. Processed {frame_n} frames.")


def run_team_classification(source_video_path: str, device: str,
                            max_frames: int = None) -> Iterator[np.ndarray]:
    model      = load_player_model(device)
    video_info = sv.VideoInfo.from_video_path(source_video_path)

    print("Pass 1: collecting player crops...")
    all_crops = []
    for frame in tqdm(
        video_frames(source_video_path, stride=STRIDE, max_frames=max_frames, start_frame=START_FRAME),
        desc='collecting crops'
    ):
        result     = model(frame, imgsz=INFERENCE_IMGSZ, conf=INFERENCE_CONF, agnostic_nms=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = clean_detections(
            detections, video_info.width, video_info.height)
        all_crops += get_crops(
            frame, detections[detections.class_id == PLAYER_CLASS_ID])

    print(f"Collected {len(all_crops)} crops. Fitting team classifier...")
    team_classifier = TeamClassifier(device=device, batch_size=32)
    team_classifier.fit(all_crops)
    print("Team classifier ready. Starting video processing...")

    tracker = sv.ByteTrack(minimum_consecutive_frames=1, lost_track_buffer=90)
    for frame in video_frames(source_video_path, max_frames=max_frames, start_frame=START_FRAME):
        result     = model(frame, imgsz=INFERENCE_IMGSZ, conf=INFERENCE_CONF, agnostic_nms=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = clean_detections(
            detections, video_info.width, video_info.height)
        detections = tracker.update_with_detections(detections)

        if detections.tracker_id is None or len(detections) == 0:
            yield frame.copy()
            continue

        players    = detections[detections.class_id == PLAYER_CLASS_ID]
        crops      = get_crops(frame, players)
        if len(crops) == 0:
            yield frame.copy()
            continue

        players_team_id     = team_classifier.predict(crops)
        goalkeepers         = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)
        referees            = detections[detections.class_id == REFEREE_CLASS_ID]

        detections   = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels    = [str(tid) for tid in detections.tracker_id]
        annotated = frame.copy()
        annotated = ELLIPSE_ANNOTATOR.annotate(
            annotated, detections, custom_color_lookup=color_lookup)
        annotated = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated, detections, labels, custom_color_lookup=color_lookup)
        yield annotated


def run_radar(source_video_path: str, device: str,
              max_frames: int = None) -> Iterator[np.ndarray]:
    player_model = load_player_model(device)
    pitch_model  = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    video_info   = sv.VideoInfo.from_video_path(source_video_path)

    print("Pass 1: collecting crops...")
    all_crops = []
    for frame in tqdm(
        video_frames(source_video_path, stride=STRIDE, max_frames=max_frames, start_frame=START_FRAME),
        desc='collecting crops'
    ):
        result     = player_model(frame, imgsz=INFERENCE_IMGSZ, conf=INFERENCE_CONF, agnostic_nms=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = clean_detections(
            detections, video_info.width, video_info.height)
        all_crops += get_crops(
            frame, detections[detections.class_id == PLAYER_CLASS_ID])

    print(f"Collected {len(all_crops)} crops. Fitting team classifier...")
    team_classifier = TeamClassifier(device=device, batch_size=32)
    team_classifier.fit(all_crops)
    print("Ready. Processing radar...")

    tracker = sv.ByteTrack(minimum_consecutive_frames=1, lost_track_buffer=90)
    for frame in video_frames(source_video_path, max_frames=max_frames, start_frame=START_FRAME):
        keypoints  = sv.KeyPoints.from_ultralytics(
            pitch_model(frame, verbose=False)[0])
        result     = player_model(frame, imgsz=INFERENCE_IMGSZ, conf=INFERENCE_CONF, agnostic_nms=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = clean_detections(
            detections, video_info.width, video_info.height)
        detections = tracker.update_with_detections(detections)

        if detections.tracker_id is None or len(detections) == 0:
            yield frame.copy()
            continue

        players    = detections[detections.class_id == PLAYER_CLASS_ID]
        crops      = get_crops(frame, players)
        if len(crops) == 0:
            yield frame.copy()
            continue

        players_team_id     = team_classifier.predict(crops)
        goalkeepers         = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)
        referees            = detections[detections.class_id == REFEREE_CLASS_ID]

        detections   = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels    = [str(tid) for tid in detections.tracker_id]
        annotated = frame.copy()
        annotated = ELLIPSE_ANNOTATOR.annotate(
            annotated, detections, custom_color_lookup=color_lookup)
        annotated = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated, detections, labels, custom_color_lookup=color_lookup)
        try:
            h, w, _ = frame.shape
            radar   = render_radar(detections, keypoints, color_lookup)
            radar   = sv.resize_image(radar, (w//2, h//2))
            rh, rw, _ = radar.shape
            rect    = sv.Rect(x=w//2-rw//2, y=h-rh, width=rw, height=rh)
            annotated = sv.draw_image(annotated, radar, opacity=0.5, rect=rect)
        except Exception:
            pass
        yield annotated


def run_full_analysis(source_video_path: str, device: str,
                      max_frames: int = None) -> Iterator[np.ndarray]:
    player_model = load_player_model(device)
    pitch_model  = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_model   = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    video_info   = sv.VideoInfo.from_video_path(source_video_path)
    fps          = video_info.fps or 30

    print("Pass 1: collecting crops for team classifier...")
    all_crops = []
    for frame in tqdm(
        video_frames(source_video_path, stride=STRIDE, max_frames=max_frames, start_frame=START_FRAME),
        desc='collecting crops'
    ):
        result     = player_model(frame, imgsz=INFERENCE_IMGSZ, conf=INFERENCE_CONF, agnostic_nms=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = clean_detections(
            detections, video_info.width, video_info.height)
        all_crops += get_crops(
            frame, detections[detections.class_id == PLAYER_CLASS_ID])

    print(f"Collected {len(all_crops)} crops. Fitting team classifier...")
    team_classifier = TeamClassifier(device=device, batch_size=32)
    team_classifier.fit(all_crops)
    print("Team classifier ready. Running full analysis...")

    tracker      = sv.ByteTrack(minimum_consecutive_frames=1, lost_track_buffer=90)
    ball_tracker = BallTracker(buffer_size=20)
    ball_ann     = BallAnnotator(radius=6, buffer_size=10)

    def ball_cb(img):
        return sv.Detections.from_ultralytics(
            ball_model(img, imgsz=640, verbose=False)[0])

    slicer = sv.InferenceSlicer(
        callback=ball_cb,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    player_positions = []
    ball_positions   = []
    possession_log   = []
    frame_n          = 0

    for frame in video_frames(source_video_path, max_frames=max_frames, start_frame=START_FRAME):
        secs       = round(frame_n / fps, 2)
        keypoints  = sv.KeyPoints.from_ultralytics(
            pitch_model(frame, verbose=False)[0])
        result     = player_model(frame, imgsz=INFERENCE_IMGSZ, conf=INFERENCE_CONF, agnostic_nms=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = clean_detections(
            detections, video_info.width, video_info.height)
        detections = tracker.update_with_detections(detections)

        ball_det = slicer(frame).with_nms(threshold=0.1)
        ball_det = ball_tracker.update(ball_det)

        if len(ball_det) > 0:
            bx = float(ball_det.xyxy[0][0] + ball_det.xyxy[0][2]) / 2
            by = float(ball_det.xyxy[0][1] + ball_det.xyxy[0][3]) / 2
            ball_positions.append({
                "second": secs,
                "x_pct":  round(bx / frame.shape[1] * 100, 1),
                "y_pct":  round(by / frame.shape[0] * 100, 1),
            })

        annotated = frame.copy()

        if detections.tracker_id is not None and len(detections) > 0:
            players     = detections[detections.class_id == PLAYER_CLASS_ID]
            crops       = get_crops(frame, players)
            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            referees    = detections[detections.class_id == REFEREE_CLASS_ID]

            if len(crops) > 0:
                players_team_id     = team_classifier.predict(crops)
                goalkeepers_team_id = resolve_goalkeepers_team_id(
                    players, players_team_id, goalkeepers)
                detections   = sv.Detections.merge([players, goalkeepers, referees])
                color_lookup = np.array(
                    players_team_id.tolist() +
                    goalkeepers_team_id.tolist() +
                    [REFEREE_CLASS_ID] * len(referees)
                )
                labels = [str(tid) for tid in detections.tracker_id]

                for i in range(len(players)):
                    box  = players.xyxy[i]
                    cx   = float((box[0]+box[2])/2)
                    cy   = float((box[1]+box[3])/2)
                    tid  = int(players.tracker_id[i]) \
                           if players.tracker_id is not None else -1
                    team = int(players_team_id[i]) \
                           if i < len(players_team_id) else -1
                    player_positions.append({
                        "second": secs, "tracker_id": tid, "team_id": team,
                        "x_pct":  round(cx/frame.shape[1]*100, 1),
                        "y_pct":  round(cy/frame.shape[0]*100, 1),
                    })

                t0r = sum(1 for i, t in enumerate(players_team_id)
                          if t == 0 and players.xyxy[i][0] > frame.shape[1]/2)
                t1r = sum(1 for i, t in enumerate(players_team_id)
                          if t == 1 and players.xyxy[i][0] > frame.shape[1]/2)
                possession_log.append(0 if t0r >= t1r else 1)

                annotated = ELLIPSE_ANNOTATOR.annotate(
                    annotated, detections, custom_color_lookup=color_lookup)
                annotated = ELLIPSE_LABEL_ANNOTATOR.annotate(
                    annotated, detections, labels, custom_color_lookup=color_lookup)

                try:
                    h, w, _ = frame.shape
                    radar   = render_radar(detections, keypoints, color_lookup)
                    radar   = sv.resize_image(radar, (w//2, h//2))
                    rh, rw, _ = radar.shape
                    rect    = sv.Rect(x=w//2-rw//2, y=h-rh, width=rw, height=rh)
                    annotated = sv.draw_image(
                        annotated, radar, opacity=0.5, rect=rect)
                except Exception:
                    pass

        annotated = ball_ann.annotate(annotated, ball_det)
        frame_n  += 1
        yield annotated

    total   = max(len(possession_log), 1)
    t0_poss = round(possession_log.count(0)/total*100, 1)
    t1_poss = round(possession_log.count(1)/total*100, 1)
    stats   = {
        "meta": {
            "video": source_video_path,
            "processed_at": datetime.utcnow().isoformat(),
            "fps": fps, "total_frames": frame_n,
            "duration_mins": round(frame_n/fps/60, 1),
        },
        "possession":       {"team_0_pct": t0_poss, "team_1_pct": t1_poss},
        "player_positions": player_positions,
        "ball_positions":   ball_positions,
    }
    sp = output_path_for(source_video_path, 'match_data')
    with open(sp, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved: {sp}")
    print(f"Possession — Team 0: {t0_poss}%  Team 1: {t1_poss}%")


# ================================================================
# MAIN
# ================================================================

def main(
    source_video_path: str,
    target_video_path: str,
    device: str,
    mode: Mode,
    focus_id: int = None,
    max_frames: int = None,
    no_render: bool = False
) -> None:
    # Fit once up front — the camera is stationary, so the touchline doesn't
    # move, and every mode's call to filter_detections_by_pitch picks this up.
    global FAR_TOUCHLINE
    print("Fitting far touchline from pitch keypoints...")
    FAR_TOUCHLINE = fit_far_touchline(source_video_path, device)
    global PITCH_POLYGON
    _vi = sv.VideoInfo.from_video_path(source_video_path)
    if PITCH_POLYGON_PATH:
        PITCH_POLYGON = np.load(PITCH_POLYGON_PATH).astype(np.int32)
        print(f"  Pitch polygon: loaded {len(PITCH_POLYGON)} points from "
              f"{os.path.basename(PITCH_POLYGON_PATH)}")
    else:
        PITCH_POLYGON = build_pitch_polygon(source_video_path, device,
                                            _vi.width, _vi.height)
        if PITCH_POLYGON is None:
            # The landmark route needs pitch markings, and on a dry worn pitch
            # the pitch model finds none at any confidence. Rather than drop to
            # the percentage bounds — which admit the crowd and the next field
            # — derive the boundary from what stays still.
            print("  Falling back to the motion-based polygon "
                  "(no pitch markings found)...")
            PITCH_POLYGON = build_pitch_polygon_from_motion(
                source_video_path, _vi.width, _vi.height)
        if PITCH_POLYGON is not None:
            _out = output_path_for(source_video_path, 'pitch_polygon').replace(
                '.json', '.npy')
            np.save(_out, PITCH_POLYGON)
            print(f"  Pitch polygon saved to {os.path.basename(_out)} — "
                  f"VIEW IT before trusting a run. A polygon that silently "
                  f"clips the near half of the pitch still produces plausible "
                  f"detection counts.")
    print(f"Start frame: {START_FRAME}")
    print(f"Detector: {DETECTOR}  imgsz: {INFERENCE_IMGSZ}  conf: {INFERENCE_CONF}  touchline buffer: {TOUCHLINE_BUFFER_PX}px")

    if max_frames:
        print(f"Limiting to the first {max_frames} frames.")

    if mode == Mode.PITCH_DETECTION:
        gen = run_pitch_detection(source_video_path, device, max_frames)
    elif mode == Mode.PLAYER_DETECTION:
        gen = run_player_detection(source_video_path, device, max_frames)
    elif mode == Mode.BALL_DETECTION:
        gen = run_ball_detection(source_video_path, device, max_frames)
    elif mode == Mode.PLAYER_TRACKING:
        gen = run_player_tracking(source_video_path, device, focus_id,
                                  max_frames, no_render)
    elif mode == Mode.TEAM_CLASSIFICATION:
        gen = run_team_classification(source_video_path, device, max_frames)
    elif mode == Mode.RADAR:
        gen = run_radar(source_video_path, device, max_frames)
    elif mode == Mode.FULL_ANALYSIS:
        gen = run_full_analysis(source_video_path, device, max_frames)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    if no_render:
        # Drain the generator without opening a video sink, so no empty or
        # half-written output file is left behind.
        for _ in tqdm(gen, desc='Processing (no render)'):
            pass
        return

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in tqdm(gen, desc='Processing'):
            sink.write_frame(frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soccer AI Analysis')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device',   type=str,  default='cpu')
    parser.add_argument('--mode',     type=Mode, default=Mode.PLAYER_DETECTION)
    parser.add_argument('--focus_id', type=int,  default=None,
        help='Highlight a specific canonical player ID in PLAYER_TRACKING mode.')
    parser.add_argument('--run_label', type=str, default=None,
        help='Tag for this run, used in output filenames. Defaults to a '
             'timestamp so re-running the same video never overwrites the '
             'previous result.')
    parser.add_argument('--touchline_buffer', type=int, default=None,
        help='Pixels below the fitted touchline that feet must sit. Needs '
             'calibrating per camera setup: the pitch-keypoint model can place '
             'the line well above the real touchline (on the ultrawide footage '
             'it lands in the carpark), and this compensates. 20 suits '
             'game.mp4, 60 suits Improved_Ultrawide.')
    parser.add_argument('--no_minimap', action='store_true',
                        help='Hide the 2D player/ball map in the corner.')
    parser.add_argument('--minimap_corner', type=str, default=None,
                        choices=['bottom_left','bottom_right','top_left','top_right'],
                        help='Where to place the 2D map (default bottom_left).')
    parser.add_argument('--no_stitch', action='store_true',
                        help='Skip offline tracklet stitching after pass 1, '
                             'showing the online tracker output unmodified.')
    parser.add_argument('--track_dump', action='store_true',
        help='Save every id\'s per-frame position, for offline tracklet '
             'stitching with stitch_tracks.py.')
    parser.add_argument('--start_frame', type=int, default=None,
        help='Frame to start processing at. Use to skip warm-up — kickoff is '
             'often several minutes into a recording.')
    parser.add_argument('--start_seconds', type=float, default=None,
        help='Same as --start_frame but in seconds of the source video.')
    parser.add_argument('--pitch_polygon', type=str, default=None,
        help='Path to a .npy Nx2 polygon of the playing area in image coords. '
             'Use when the pitch-keypoint model finds no landmarks (it finds '
             'none at all on dry, worn pitches).')
    parser.add_argument('--include_referees', action='store_true',
        help='Track referees as well as players.')
    parser.add_argument('--show_ball', action='store_true',
        help='Draw ball detections on the rendered video.')
    parser.add_argument('--detector', type=str, default=None, choices=['yolo','rfdetr'],
        help="Detector backend. 'rfdetr' uses the local Roboflow v20 transformer "
             "(see rfdetr_onnx.py) which generalises far better across grounds; "
             "'yolo' uses our own trained weights.")
    parser.add_argument('--model', type=str, default=None,
        help='Player-detection weights to use. Lets a run pick the right model '
             'per view — a model trained on ultrawide footage can be blind to a '
             'follow-cam of the same match, and vice versa.')
    parser.add_argument('--conf', type=float, default=None,
        help='Detection confidence floor (default 0.25). Lower it for footage '
             'the model was not trained on — it detects correctly but with low '
             'confidence, and the default silently discards everything.')
    parser.add_argument('--imgsz', type=int, default=None,
        help='Model input size. Higher recovers small players but costs runtime; '
             'a wider source downscales harder, so 4096-wide footage needs more '
             'than 3024-wide to give the model the same pixels.')
    parser.add_argument('--max_frames', type=int, default=None,
        help='Process only the first N frames — useful for equal-length comparisons.')
    parser.add_argument('--no_render', action='store_true',
        help='Skip pass 2. All tracking stats come from pass 1, so this roughly '
             'halves a calibration run and writes no video.')
    args = parser.parse_args()
    if args.no_minimap:
        SHOW_MINIMAP = False
    if args.minimap_corner:
        MINIMAP_CORNER = args.minimap_corner
    if args.no_stitch:
        STITCH = False
    if args.track_dump:
        TRACK_DUMP = True
    if args.include_referees:
        INCLUDE_REFEREES = True
    if args.show_ball:
        SHOW_BALL = True
    if args.detector:
        DETECTOR = args.detector
        if args.conf is None and args.detector == 'rfdetr':
            # This is the floor for the WEAK tier, not the working threshold —
            # see TRACK_ACTIVATION_THRESHOLD, which is what actually decides
            # when a new track may open. Detections between 0.20 and 0.40 are
            # passed to the tracker so they can continue a track a confident
            # detection already started.
            #
            # A single-tier sweep (everything below the floor discarded) put the
            # best value at 0.40 — 22.3 dets/frame against ~22 players, where
            # 0.30 gave 28.5 and WORSE continuity because the extras opened
            # tracks of their own. Note that an earlier sweep picked 0.30; it
            # ran while over-merging was silently absorbing fragments, so extra
            # detections looked like better continuity. Redone with the
            # per-frame id invariant in place.
            #
            # Splitting the tiers is what made the lower floor pay: the same
            # 0.20-0.40 detections that hurt as track seeds help as track
            # continuations.
            INFERENCE_CONF = 0.20
    if args.start_frame:
        START_FRAME = args.start_frame
    if args.start_seconds:
        START_FRAME = int(args.start_seconds *
                          (sv.VideoInfo.from_video_path(args.source_video_path).fps or 30))
    if args.pitch_polygon:
        PITCH_POLYGON_PATH = args.pitch_polygon
    if args.model:
        PLAYER_DETECTION_MODEL_PATH = args.model
    if args.imgsz:
        INFERENCE_IMGSZ = args.imgsz
    if args.conf is not None:
        INFERENCE_CONF = args.conf
    if args.touchline_buffer is not None:
        TOUCHLINE_BUFFER_PX = args.touchline_buffer
    if args.run_label:
        RUN_LABEL = args.run_label
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        focus_id=args.focus_id,
        max_frames=args.max_frames,
        no_render=args.no_render,
    )