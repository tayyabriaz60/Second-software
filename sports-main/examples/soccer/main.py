import argparse
import itertools
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict, deque
from datetime import datetime
from enum import Enum
from typing import Iterator, List, Optional

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO


class RobustVideoSink:
    """Write a playable MP4; prefer ffmpeg over OpenCV's fragile mp4v path.

    OpenCV VideoWriter can spam 'Failed to write frame' and exit 'Done' while
    leaving a file with no moov atom (unopenable). ffmpeg libx264 + faststart
    finalizes the container properly; we fail loud if the pipe dies.
    """

    def __init__(self, path: str, video_info: sv.VideoInfo):
        self.path = path
        self.fps = float(getattr(video_info, 'fps', 0) or 30.0)
        w = int(video_info.width)
        h = int(video_info.height)
        # libx264 / yuv420p need even dimensions
        self.width = w - (w % 2)
        self.height = h - (h % 2)
        self._proc: Optional[subprocess.Popen] = None
        self._writer = None
        self._n = 0

    def __enter__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or '.',
                    exist_ok=True)
        if shutil.which('ffmpeg'):
            cmd = [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{self.width}x{self.height}',
                '-r', str(self.fps),
                '-i', '-',
                '-an',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-preset', 'veryfast', '-crf', '23',
                '-movflags', '+faststart',
                self.path,
            ]
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"Video writer: ffmpeg libx264 -> {self.path}")
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._writer = cv2.VideoWriter(
                self.path, fourcc, self.fps, (self.width, self.height))
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter for {self.path}")
            print(f"Video writer: OpenCV mp4v -> {self.path} "
                  f"(install ffmpeg for safer output)")
        return self

    def write_frame(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        frame = np.ascontiguousarray(frame)
        if self._proc is not None:
            try:
                self._proc.stdin.write(frame.tobytes())
            except BrokenPipeError as exc:
                err = ''
                if self._proc.stderr:
                    err = self._proc.stderr.read().decode('utf-8', 'ignore')[-500:]
                raise RuntimeError(
                    f"ffmpeg pipe broke after {self._n} frames. "
                    f"Output is likely corrupt. stderr: {err}"
                ) from exc
        else:
            self._writer.write(frame)
        self._n += 1

    def __exit__(self, exc_type, exc, tb):
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except Exception:
                pass
            rc = self._proc.wait()
            err = ''
            if self._proc.stderr:
                err = self._proc.stderr.read().decode('utf-8', 'ignore')[-800:]
            if exc_type is None and rc != 0:
                raise RuntimeError(
                    f"ffmpeg exited {rc} after {self._n} frames — "
                    f"MP4 likely invalid. {err}"
                )
            if exc_type is None:
                print(f"Wrote {self._n} frames via ffmpeg (ok)")
        elif self._writer is not None:
            self._writer.release()
            if exc_type is None:
                print(f"Wrote {self._n} frames via OpenCV")
        return False

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
# Raised 1.5→3.0s for production panoramic matches: most ID-explosion fragments
# die well under 3s, while real players (and post-stitch identities) survive.
# Transient shadow / touchline artifacts must not keep a "valid" identity.
# Raised for production, but short --max_frames calibration runs must not
# empty the roster: PlayerReIDTracker caps this against clip length.
MIN_SECONDS_TO_KEEP = 3.0
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
MIN_SPEED_PX_PER_SEC = 10.0

# ...but speed alone still discards goalkeepers, who patrol slowly and never
# sprint. A keeper and a fence post can look identical by speed; they differ in
# whether they ever RELOCATE. Measured on a 30s ultrawide clip, both present
# for ~100% of it:
#     goalkeeper  4.0 px/s, net displacement 187px
#     static obj  5.1 px/s, net displacement   1px
# So an ID is kept if it is quick enough OR it ended up somewhere else.
# Stricter (100→150) to purge sideline coaches / parked spectators on long runs.
MIN_NET_DISPLACEMENT_PX = 150
# Reject tiny boxes before they ever enter ByteTrack — RF-DETR shadow noise and
# partial crowd blobs often sit under ~18px and mint throwaway tracklets.
MIN_BOX_HEIGHT_PX = 18
MIN_BOX_WIDTH_PX = 8

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
#    0.20      0.40      43       7       14     25    <- previous
#    0.15      0.45      40       7       14     19
#    0.10      0.50      32       7       14     22
#
# Production tweak for ID explosion on long panoramic clips: keep a LOW detect
# floor (weak tier continues tracks through shadow / soft RF-DETR scores) but
# raise activation slightly so those weak boxes cannot mint new identities.
# TRACK_DETECT_FLOOR is what get_player_detections keeps; activation is what
# ByteTrack uses to open tracks. (Classic ByteTrack names map as:
#   track_low_thresh ≈ TRACK_DETECT_FLOOR
#   track_high_thresh / new_track_thresh ≈ TRACK_ACTIVATION_THRESHOLD
#   match_thresh ≈ TRACK_MATCHING_THRESHOLD)
TRACK_DETECT_FLOOR = 0.15
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
# Path-efficiency / weld diagnostics (see WELD_GUARD below). Teleport speed
# cuts (SPLIT_IMPLAUSIBLE) remain the primary physics scissors; weld guard
# adds path/net inflection splits before metric export.
SPLIT_INEFFICIENT = False
EFFICIENCY_WINDOW_SECONDS = 3.0
MAX_PATH_NET_RATIO = 8.0
MIN_EFFICIENCY_PATH_PX = 180.0
# mot_sota_v6: cut welded identities at handoff inflection before export.
WELD_GUARD = True
WELD_PATH_NET_CEILING = 25.0      # whole-track path/net above this → scan for cut
WELD_TELEPORT_BODY_H_PER_SEC = 7.0  # stricter than general physics cut
# Rolling path/net cuts inside weld_guard. OFF: on the v8 10-min run this made
# 536 cuts and turned 219 identities into 755. A footballer stops and turns
# inside every 3 s window (path 100px, net 10px → ratio 10 > 8), so the test
# fires on ordinary play, not on welds. Teleport cuts above stay on.
WELD_GUARD_EFFICIENCY = False

TEAM_CROPS_PER_ID = 10
TEAM_CROP_STRIDE = 8

# Slightly above the measured 0.40 sweet spot: harder to OPEN a track on a soft
# shadow-edge detection, while TRACK_DETECT_FLOOR still lets those boxes
# CONTINUE an existing one. Reduces ID minting without starving association.
TRACK_ACTIVATION_THRESHOLD = 0.45
# Matching gate for associating a detection with an existing track; higher is
# more permissive. With a SHORT ByteTrack lost buffer (see BYTE_TRACK_LOST_SECONDS),
# association is IoU-local; 0.90 is the continuity/switch trade-off for RF-DETR jitter.
TRACK_MATCHING_THRESHOLD = 0.90
# Require two hits before a brand-new ByteTrack id is emitted. Single-frame
# RF-DETR flicker (tree shadow, duplicate query) was minting throwaway ids that
# then survived long enough to look like "real" fragments in the dump.
TRACK_MIN_CONSECUTIVE_FRAMES = 2
# Bridge detector/occlusion holes of 1..N frames in id_history (linear +
# constant-velocity coast). 10 frames covers ~0.18–0.33s depending on fps —
# the regime where papers densify tracklets rather than letting association
# open a duplicate identity. Does NOT invent online IDs; densifies existing
# ones so stitch / speed / minimap see continuity.
BRIDGE_MAX_FRAMES = 10
# Sprint reference for adaptive Re-ID gating (px/sec on ultrawide). Fast
# movers get a modestly wider gate; idle players stay tight (avoids the
# measured failure mode of a flat growing radius).
REID_SPRINT_PX_PER_SEC = 210.0
REID_RADIUS_SPEED_GAIN = 0.25
REID_VELOCITY_DECAY = 0.88
# Online CIELAB L-gap above which a lost-track candidate is rejected as the
# wrong team (navy vs sky-blue kits). Below this, colour is ignored online.
TEAM_LAB_HARD_DELTA = 28.0
# When offline team separation is strong enough, opposing labels HARD-VETO a
# stitch link (not just a soft penalty). Below this, keep soft penalty only —
# 93% colour accuracy is too weak for a blanket veto.
TEAM_HARD_SEP_MIN = 1.35
TEAM_HARD_VOTE_FRAC = 0.70

# Fold offline-stitched fragments into single identities after pass 1.
STITCH = True
# Thin stitch links measured ~29% correct — never apply them for v6.
STITCH_KEEP_THIN = False
# Confident-only stitch (mot_sota_v6): short gap + appearance p95 band.
STITCH_MAX_GAP_FRAMES = 30
STITCH_APPEARANCE_MIN_COSINE = 0.50
STITCH_SIM_PERCENTILE = 95.0   # only top-5% appearance matches among candidates
STITCH_COST_PERCENTILE = 5.0   # only best-5% motion costs among candidates

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

# ---------------------------------------------------------------------------
# IDENTITY ARCHITECTURE (mot_sota_v5 → v6)
# ---------------------------------------------------------------------------
# v5: ByteTrack short IoU + ReID long memory + purge on adopt → killed 100xxx
#     swarm (428 → 34) but left ~218 passed IDs (client needs ~24–40).
# v6: Mahalanobis + hard appearance cosine on ReID; confident-only stitch
#     (gap≤30f, no thin, p95 sim); weld-guard splits; roster filter 24–40.
# ---------------------------------------------------------------------------
BYTE_TRACK_LOST_SECONDS = 0.75   # IoU coast only; NOT long-term identity
REID_WINDOW_SECONDS = 12.0       # long-horizon reclaim after occlusions

# Spatial gating for re-id, as a fraction of the frame diagonal (~394px here).
REID_DISTANCE_FRACTION = 0.12

# How long a track must have been unseen before another detection may adopt its
# canonical id.
REID_MIN_LOST_FRAMES = 2

# Isotropic Mahalanobis gate on (x,y) residual vs CV prediction (v6).
# sigma grows with gap so a 2-frame flicker stays open but a duel teleport fails.
REID_POS_SIGMA_PX = 28.0
REID_VEL_SIGMA_PX = 6.0
REID_MAHA_MAX = 2.5

# Appearance: hard veto (not soft rank-only). Close-contact duels require
# cosine ≥ REID_APPEARANCE_MIN_COSINE when both embeddings exist; otherwise
# the reclaim is declined rather than guessed.
APPEARANCE_HARD_GATE = True
REID_APPEARANCE_MIN_COSINE = 0.42
REID_DUEL_RADIUS_FRAC = 0.55   # of adaptive radius: 2+ cands inside ⇒ duel
REID_DUEL_MIN_COSINE = 0.55    # stricter appearance in a duel

# Soft rank weight among candidates that already passed hard gates.
APPEARANCE_WEIGHT = 0.40

# Embedding every detection every frame would mean ~222k transformer passes on
# a 2.4-minute clip. We only need one when deciding a NEW track's identity, or
# to refresh a known track's signature occasionally.
APPEARANCE_REFRESH_FRAMES = 45
APPEARANCE_MIN_CROP_PX    = 12   # smaller than this carries no usable signal

# Client deliverable band: active players + main subs.
TARGET_PASSED_MIN = 24
TARGET_PASSED_MAX = 40
# Tracks above this path/net after weld-guard are treated as still-corrupt and
# excluded from the passed roster (stats would be meaningless).
MAX_PASSED_PATH_NET = 30.0
# Collision-split emergency IDs must not enter the client roster.
EXCLUDE_COLLISION_IDS_FROM_PASSED = True

# Movement is measured as path length, sampled every few frames. Comparing
# first position to last position (the old approach) scores a player who runs
# all game and finishes near where they started the same as a fence post.
MOVE_SAMPLE_EVERY = 15     # frames between position samples
MOVE_MIN_STEP_PX  = 5      # ignore smaller steps — that's box jitter, not travel

# Pitch boundary as % of frame — used for the sides and the near edge, and as
# the fallback for the far edge when the touchline curve / motion polygon
# cannot be trusted (dry pitches with 0 keypoints).
PITCH_LEFT_PCT   = 5
PITCH_RIGHT_PCT  = 95
PITCH_TOP_PCT    = 8
PITCH_BOTTOM_PCT = 95

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

# Motion-polygon safety: if a built polygon keeps fewer than this fraction of
# raw player feet (or covers too little of the frame), discard it and use the
# %-bounds. A bad motion polygon was wiping every detection on dry pitches
# where keypoints fail and players themselves look "static" across samples.
PITCH_POLYGON_MIN_AREA_FRAC = 0.38
PITCH_POLYGON_MIN_KEEP_FRAC = 0.50
PITCH_POLYGON_MAX_FAR_Y_FRAC = 0.42   # far edge must stay in the top ~42%
# After this many frames where the polygon would wipe all dets but %-bounds
# would keep some, disable the polygon for the rest of the run.
PITCH_POLYGON_FAIL_DISABLE = 20

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


# Set from --run_label, or defaults to mot_sota_v6 for the identification push.
RUN_LABEL = 'mot_sota_v6'


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
# Frames where polygon wiped all dets while %-bounds would have kept some.
_pitch_polygon_fail_frames = 0


def _percent_bounds_mask(detections: sv.Detections, frame_w: int,
                         frame_h: int) -> np.ndarray:
    """Loose side/top/bottom percentage gate — last-resort keep filter."""
    xyxy = detections.xyxy
    cx_pct = ((xyxy[:, 0] + xyxy[:, 2]) / 2) / frame_w * 100
    cy_pct = ((xyxy[:, 1] + xyxy[:, 3]) / 2) / frame_h * 100
    return ((cx_pct >= PITCH_LEFT_PCT) & (cx_pct <= PITCH_RIGHT_PCT) &
            (cy_pct >= PITCH_TOP_PCT) & (cy_pct <= PITCH_BOTTOM_PCT))


def _detect_raw_for_polygon(frame):
    """Detector call used while building/validating the pitch polygon."""
    if DETECTOR == 'rfdetr':
        import rfdetr_onnx
        return rfdetr_onnx.detect(frame, conf=INFERENCE_CONF)
    # Prefer CUDA/CPU over hard-coded 'mps' (breaks on RunPod Linux).
    device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '' else 'cpu'
    try:
        _m = load_player_model(device)
    except Exception:
        _m = load_player_model('cpu')
    return sv.Detections.from_ultralytics(
        _m(frame, imgsz=INFERENCE_IMGSZ, conf=INFERENCE_CONF,
           agnostic_nms=True, verbose=False)[0])


def build_pitch_polygon_from_motion(source_video_path, frame_w, frame_h,
                                    n_samples=40, cell=64, static_frac=0.75):
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

    static_frac defaults to 0.75 (was 0.60): at 0.60, standing/jogging players
    on a short sample window were labelled "crowd", the far edge dropped into
    midfield, and clean_detections wiped every valid player → Valid IDs [].
    """
    import collections
    info = sv.VideoInfo.from_video_path(source_video_path)
    fps = info.fps or 30.0
    step = max(1, int(3 * fps))

    occ = collections.Counter()
    taken = n = 0
    for frame in video_frames(source_video_path, start_frame=START_FRAME):
        if n % step == 0:
            det = suppress_contained_boxes(_detect_raw_for_polygon(frame))
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
        # Only cells in the TOP half of the frame can be "crowd" for an
        # elevated camera — treating midfield players as static is the
        # failure mode that emptied Valid IDs.
        y_limit = (frame_h // 2) // cell
        static = [yb for (x, yb), c in occ.items()
                  if x == xb and yb <= y_limit and c >= static_frac * taken]
        raw[xb] = (max(static) + 1) * cell if static else None
    if all(v is None for v in raw.values()):
        print("  Pitch polygon: no static band found — using %-bounds fallback")
        return None

    filled = []
    for xb in range(n_cols):
        if raw[xb] is not None:
            filled.append(raw[xb])
            continue
        near = [raw[x] for x in range(max(0, xb - 6), min(n_cols, xb + 7))
                if raw[x] is not None]
        filled.append(max(near) if near else 0)
    smooth = [int(max(filled[max(0, i - 2):i + 3])) for i in range(n_cols)]

    # Never cut more than the top ~42% away — deeper cuts remove real players.
    cap = int(frame_h * PITCH_POLYGON_MAX_FAR_Y_FRAC)
    smooth = [min(y, cap) for y in smooth]
    pts = [[xb * cell, smooth[xb]] for xb in range(0, n_cols, 3)]
    poly = np.array([[0, pts[0][1]]] + pts +
                    [[frame_w, smooth[-1]], [frame_w, frame_h], [0, frame_h]],
                    dtype=np.int32)
    print(f"  Pitch polygon (motion): {len(poly)} points, "
          f"far edge y={min(p[1] for p in pts)}..{max(p[1] for p in pts)}, "
          f"{cv2.contourArea(poly) / (frame_w * frame_h):.0%} of frame")
    return poly


def validate_pitch_polygon(poly, source_video_path, frame_w, frame_h,
                           n_check: int = 10) -> Optional[np.ndarray]:
    """Reject a motion/keypoint polygon that would wipe player detections."""
    if poly is None or len(poly) < 3:
        return None
    area_frac = float(cv2.contourArea(poly)) / max(frame_w * frame_h, 1)
    if area_frac < PITCH_POLYGON_MIN_AREA_FRAC:
        print(f"  Pitch polygon REJECTED: area {area_frac:.0%} of frame "
              f"< {PITCH_POLYGON_MIN_AREA_FRAC:.0%} — using %-bounds")
        return None
    far_ys = [int(p[1]) for p in poly[:-2]]  # exclude bottom closers
    if far_ys and max(far_ys) > int(frame_h * PITCH_POLYGON_MAX_FAR_Y_FRAC) + 5:
        print(f"  Pitch polygon REJECTED: far edge y={max(far_ys)} too deep "
              f"(>{PITCH_POLYGON_MAX_FAR_Y_FRAC:.0%} of frame) — using %-bounds")
        return None

    info = sv.VideoInfo.from_video_path(source_video_path)
    fps = info.fps or 30.0
    step = max(1, int(2 * fps))
    raw_n = keep_n = checked = 0
    for frame in video_frames(source_video_path, start_frame=START_FRAME):
        if checked % step != 0 and checked > 0:
            checked += 1
            continue
        det = suppress_contained_boxes(_detect_raw_for_polygon(frame))
        checked += 1
        if len(det) == 0:
            if checked >= n_check * step:
                break
            continue
        if det.class_id is not None:
            det = det[np.isin(det.class_id,
                              [PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID,
                               REFEREE_CLASS_ID])]
        if len(det) == 0:
            if checked >= n_check * step:
                break
            continue
        feet = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        raw_n += len(det)
        keep_n += sum(
            1 for x, y in feet
            if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0)
        if raw_n >= 40 or checked >= n_check * step:
            break
    if raw_n >= 12:
        frac = keep_n / raw_n
        print(f"  Pitch polygon validation: kept {keep_n}/{raw_n} "
              f"player feet ({frac:.0%})")
        if frac < PITCH_POLYGON_MIN_KEEP_FRAC:
            print(f"  Pitch polygon REJECTED: keep rate {frac:.0%} "
                  f"< {PITCH_POLYGON_MIN_KEEP_FRAC:.0%} — using %-bounds")
            return None
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
    """Remove nested duplicate boxes, then anything off the pitch.

    Safety: a bad motion polygon must NEVER wipe a frame that the loose
    %-bounds would have kept — that was emptying Valid IDs on dry pitches.
    """
    global PITCH_POLYGON, _pitch_polygon_fail_frames
    if len(detections) == 0:
        return detections

    detections = suppress_contained_boxes(detections)
    if len(detections) == 0:
        return detections

    pct_mask = _percent_bounds_mask(detections, frame_w, frame_h)
    feet = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    if PITCH_POLYGON is not None:
        poly_mask = np.array([
            cv2.pointPolygonTest(PITCH_POLYGON, (float(x), float(y)), False) >= 0
            for x, y in feet])
        if poly_mask.any():
            mask = poly_mask
        elif pct_mask.any():
            # Polygon would drop everyone; %-bounds still see players.
            _pitch_polygon_fail_frames += 1
            if _pitch_polygon_fail_frames == 1 or _pitch_polygon_fail_frames % 50 == 0:
                print(f"  Pitch filter: polygon wiped frame "
                      f"({_pitch_polygon_fail_frames}x) — using %-bounds")
            if _pitch_polygon_fail_frames >= PITCH_POLYGON_FAIL_DISABLE:
                print(f"  Pitch polygon DISABLED after "
                      f"{_pitch_polygon_fail_frames} wipe-frames — "
                      f"%-bounds for the rest of the run")
                PITCH_POLYGON = None
            mask = pct_mask
        else:
            mask = poly_mask
    elif FAR_TOUCHLINE is not None:
        mask = pct_mask & (
            feet[:, 1] > FAR_TOUCHLINE(feet[:, 0]) + TOUCHLINE_BUFFER_PX)
        # If touchline curve alone wipes the frame, keep %-bounds.
        if not mask.any() and pct_mask.any():
            mask = pct_mask
    else:
        mask = pct_mask

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
                 device: str = 'cpu', use_appearance: bool = True):
        self.fps           = fps or 30.0
        self.window_frames = max(1, int(REID_WINDOW_SECONDS * self.fps))
        self.bt_lost_frames = max(1, int(BYTE_TRACK_LOST_SECONDS * self.fps))
        self.min_frames    = max(1, int(MIN_SECONDS_TO_KEEP * self.fps))
        self.bridge_frames = max(2, int(BRIDGE_MAX_FRAMES))
        self.use_appearance = use_appearance
        self.team_sep: float = 0.0
        self.team_vote_frac: dict = {}   # cid -> vote majority fraction
        # frame_rate must match the video: without it ByteTrack assumes 30fps
        # and its internal time-based buffers drift on 50/55fps screen captures,
        # killing tracks early and minting replacement ids.
        #
        # lost_track_buffer is SHORT (BYTE_TRACK_LOST_SECONDS). Long-horizon
        # identity lives in ReID (window_frames), not in ByteTrack — sharing
        # one long buffer was the mot_sota_v4 double-claim failure mode.
        bt_kwargs = dict(
            minimum_consecutive_frames=TRACK_MIN_CONSECUTIVE_FRAMES,
            lost_track_buffer=self.bt_lost_frames,
            track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
            minimum_matching_threshold=TRACK_MATCHING_THRESHOLD,
        )
        try:
            self.tracker = sv.ByteTrack(
                **bt_kwargs, frame_rate=max(1, int(round(self.fps))))
        except TypeError:
            # Older supervision builds omit frame_rate.
            self.tracker = sv.ByteTrack(**bt_kwargs)
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
        self.id_lab:         dict = {}   # cid -> EMA CIELAB torso colour
        self.id_class_votes: defaultdict = defaultdict(lambda: defaultdict(int))
        self.id_history:     defaultdict = defaultdict(list)   # cid -> [(frame,x,y)]
        self.id_crops:       defaultdict = defaultdict(list)   # cid -> [crop]
        self.id_team:        dict = {}
        self.id_splits:      dict = {}   # cid -> [(from_frame, new_cid), ...]
        self.id_remap:       dict = {}   # stitched src cid -> dst cid
        self._claimed_this_frame: set = set()
        self._id_counter = 100000      # emergency only — see _assign_canonical
        self.id_embed_frame: dict = {}   # cid -> frame its embedding was taken
        self.device = device
        self._embedder = None
        self.frame_n = 0
        self.collision_mints = 0        # should stay ~0 after v5 architecture
        self.reid_adopts = 0
        self.stale_map_purges = 0

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
        if not self.use_appearance:
            return [None] * len(crops)
        if self._embedder is None:
            try:
                self._embedder = TeamClassifier(device=self.device, batch_size=32)
            except Exception as e:
                # transformers with an unsupported torch (e.g. "PyTorch >= 2.5
                # is required") makes SiglipVisionModel unusable. Do not kill
                # the run: fall back to distance-only ReID for the whole run.
                print(f"  [appearance] SigLIP unavailable ({type(e).__name__}: "
                      f"{str(e)[:120]}) — appearance ReID disabled, "
                      f"using distance only")
                self.use_appearance = False
                return [None] * len(crops)
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

    def _exit_velocity(self, cid, n: int = 8) -> np.ndarray:
        """px/frame velocity from the tail of id_history, or zeros."""
        hist = self.id_history.get(cid)
        if not hist or len(hist) < 2:
            return np.zeros(2, dtype=np.float64)
        tail = hist[-min(n + 1, len(hist)):]
        span = tail[-1][0] - tail[0][0]
        if span <= 0:
            return np.zeros(2, dtype=np.float64)
        return np.array([
            (tail[-1][1] - tail[0][1]) / span,
            (tail[-1][2] - tail[0][2]) / span,
        ], dtype=np.float64)

    def _adaptive_radius(self, cid, frames_ago: int) -> float:
        """Spatial gate scaled by own speed — not a flat growing radius.

        Soccer MOT literature widens association covariance for high-velocity
        targets (sprints / cuts leave larger residual after a miss). A flat
        radius grown with gap length previously exploded IDs on this footage;
        scaling by the track's own speed keeps idle players tightly gated while
        giving runners room to reclaim after a mid-pitch dropout.
        """
        base = REID_DISTANCE_FRACTION * self.frame_diag
        speed = float(np.linalg.norm(self._exit_velocity(cid))) * self.fps
        speed_scale = 1.0 + REID_RADIUS_SPEED_GAIN * min(
            1.0, speed / max(REID_SPRINT_PX_PER_SEC, 1e-6))
        # Tiny extra slack for longer gaps (occlusion uncertainty), capped.
        gap_scale = 1.0 + 0.15 * min(1.0, frames_ago / max(self.window_frames, 1))
        return base * speed_scale * gap_scale

    def _lab_team_conflict(self, cid, det_lab) -> bool:
        """True if detection CIELAB conflicts hard with the lost track's kit."""
        if det_lab is None:
            return False
        prev = self.id_lab.get(cid)
        if prev is None:
            return False
        # Navy vs sky-blue kits separate primarily on L; require a large gap
        # before vetoing so shade-induced L drops don't block same-player reclaim.
        return abs(float(det_lab[0]) - float(prev[0])) >= TEAM_LAB_HARD_DELTA

    def _purge_stale_maps(self, cid: int, keep_raw: int) -> int:
        """Remove every raw→cid mapping except keep_raw.

        After ReID hands canonical `cid` to a new ByteTrack track, any older
        raw id still mapped to `cid` must die — otherwise ByteTrack can revive
        that raw track and both detections claim the same identity (the
        double-claim that forced 100xxx minting in v3/v4).
        """
        purged = 0
        for r, c in list(self.id_map.items()):
            if c == cid and r != keep_raw:
                del self.id_map[r]
                purged += 1
        self.stale_map_purges += purged
        return purged

    def _pred_xy(self, cid, frames_ago: int = None):
        """Damped constant-velocity prediction of where cid should be now."""
        lx, ly, last_frame = self.last_seen[cid]
        if frames_ago is None:
            frames_ago = max(0, self.frame_n - last_frame)
        vel = self._exit_velocity(cid)
        damp = REID_VELOCITY_DECAY ** max(0, frames_ago - 1)
        return (lx + vel[0] * frames_ago * damp,
                ly + vel[1] * frames_ago * damp)

    def _motion_residual(self, cid, cx, cy) -> float:
        """Distance from (cx,cy) to predicted position of cid (px)."""
        if cid not in self.last_seen:
            return 1e9
        px, py = self._pred_xy(cid)
        return float(np.hypot(cx - px, cy - py))

    def _mahalanobis(self, cid, cx, cy, frames_ago: int) -> float:
        """Isotropic Mahalanobis distance of (cx,cy) to CV-predicted state."""
        pred_x, pred_y = self._pred_xy(cid, frames_ago)
        vel = self._exit_velocity(cid)
        speed = float(np.linalg.norm(vel))
        sigma = (REID_POS_SIGMA_PX
                 + frames_ago * max(REID_VEL_SIGMA_PX, 0.5 * speed))
        sigma = max(sigma, 1e-3)
        return float(np.hypot(cx - pred_x, cy - pred_y) / sigma)

    def _lost_match_score(self, cid, cx, cy, emb=None, det_lab=None,
                          duel: bool = False):
        """Score in [0,1] for reclaiming lost cid, or None if gated out.

        v6 gates (all must pass):
          1. lost-window + team Lab veto
          2. Mahalanobis residual ≤ REID_MAHA_MAX
          3. inside adaptive spatial radius
          4. hard appearance cosine when embeddings exist (stricter in duels)
        """
        if cid not in self.last_seen:
            return None
        lx, ly, last_frame = self.last_seen[cid]
        frames_ago = self.frame_n - last_frame
        if frames_ago < REID_MIN_LOST_FRAMES or frames_ago > self.window_frames:
            return None
        if cid in self._claimed_this_frame:
            return None
        if self._lab_team_conflict(cid, det_lab):
            return None
        maha = self._mahalanobis(cid, cx, cy, frames_ago)
        if maha > REID_MAHA_MAX:
            return None
        pred_x, pred_y = self._pred_xy(cid, frames_ago)
        dist = float(np.hypot(cx - pred_x, cy - pred_y))
        radius = self._adaptive_radius(cid, frames_ago)
        if dist >= radius:
            return None

        sim = self._similarity(emb, self.id_appearance.get(cid))
        if APPEARANCE_HARD_GATE:
            min_cos = REID_DUEL_MIN_COSINE if duel else REID_APPEARANCE_MIN_COSINE
            track_emb = self.id_appearance.get(cid)
            if emb is not None and track_emb is not None:
                if sim is None or sim < min_cos:
                    return None  # appearance veto — refuse mid-clip switch
            elif duel:
                # Close-contact duel without appearance evidence: do not guess.
                return None
            elif emb is None or track_emb is None:
                # Spatial-only reclaim only when Mahalanobis is very tight.
                if maha > 1.2:
                    return None

        spatial = 1.0 - dist / radius
        maha_term = max(0.0, 1.0 - maha / REID_MAHA_MAX)
        score = 0.5 * spatial + 0.5 * maha_term
        if sim is not None and APPEARANCE_WEIGHT > 0:
            score = ((1 - APPEARANCE_WEIGHT) * score
                     + APPEARANCE_WEIGHT * max(0.0, sim))
        return score

    def _find_lost_match(self, cx, cy, emb=None, det_lab=None, exclude=None):
        """Greedy best lost-track reclaim (used for single-det fallback)."""
        exclude = exclude or set()
        # Detect duel: multiple lost tracks inside a fraction of the radius.
        cands = []
        for cid in self.last_seen:
            if cid in exclude:
                continue
            if cid not in self.last_seen:
                continue
            frames_ago = self.frame_n - self.last_seen[cid][2]
            if frames_ago < REID_MIN_LOST_FRAMES or frames_ago > self.window_frames:
                continue
            radius = self._adaptive_radius(cid, frames_ago)
            pred_x, pred_y = self._pred_xy(cid, frames_ago)
            dist = float(np.hypot(cx - pred_x, cy - pred_y))
            if dist < radius * REID_DUEL_RADIUS_FRAC:
                cands.append(cid)
        duel = len(cands) >= 2
        best_id, best_score = None, -1.0
        for cid in self.last_seen:
            if cid in exclude:
                continue
            score = self._lost_match_score(
                cid, cx, cy, emb, det_lab, duel=duel)
            if score is not None and score > best_score:
                best_score, best_id = score, cid
        return best_id

    def _assign_lost_globally(self, need_match):
        """Bipartite assign new detections → lost canonical ids.

        need_match: list of dicts with keys i, cx, cy, emb, det_lab, raw_id.
        Returns dict i -> cid for successful reclaim. Unassigned stay unset
        so the caller falls back to raw ByteTrack id.

        Global assignment prevents the greedy failure where det A steals lost
        id 5 and det B (the true continuation) is forced to mint a fragment.
        """
        if not need_match:
            return {}
        lost = []
        for cid, (_, _, last_frame) in self.last_seen.items():
            frames_ago = self.frame_n - last_frame
            if (REID_MIN_LOST_FRAMES <= frames_ago <= self.window_frames
                    and cid not in self._claimed_this_frame):
                lost.append(cid)
        if not lost:
            return {}

        n_d, n_l = len(need_match), len(lost)
        BIG = 10.0
        cost = np.full((n_d, n_l), BIG, dtype=np.float64)
        for di, det in enumerate(need_match):
            # Per-detection duel: ≥2 lost cids inside duel radius.
            near = 0
            cx, cy = det['cx'], det['cy']
            for cid in lost:
                frames_ago = self.frame_n - self.last_seen[cid][2]
                radius = self._adaptive_radius(cid, frames_ago)
                pred_x, pred_y = self._pred_xy(cid, frames_ago)
                if np.hypot(cx - pred_x, cy - pred_y) < radius * REID_DUEL_RADIUS_FRAC:
                    near += 1
            duel = near >= 2
            for lj, cid in enumerate(lost):
                s = self._lost_match_score(
                    cid, cx, cy, det['emb'], det['det_lab'], duel=duel)
                if s is not None:
                    cost[di, lj] = 1.0 - s

        # Stricter no-match: require score ≥ 0.50 after hard gates.
        NO_MATCH = 0.50
        slack = np.full((n_d, n_d), BIG, dtype=np.float64)
        np.fill_diagonal(slack, NO_MATCH)
        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(np.hstack([cost, slack]))
        except Exception:
            out = {}
            used = set()
            for det in need_match:
                cid = self._find_lost_match(
                    det['cx'], det['cy'], det['emb'], det['det_lab'],
                    exclude=used)
                if cid is not None:
                    out[det['i']] = cid
                    used.add(cid)
            return out

        out = {}
        for r, c in zip(rows, cols):
            if c >= n_l:
                continue
            if cost[r, c] >= NO_MATCH:
                continue
            out[need_match[r]['i']] = lost[c]
        return out

    def interpolate_short_gaps(self) -> int:
        """Fill holes of 2..BRIDGE_MAX_FRAMES in each id_history.

        Uses linear interpolation in (x, y, box_h). Online ByteTrack already
        keeps lost tracks alive via lost_track_buffer; this densifies the
        recorded trajectory so offline stitch / physics / minimap do not treat
        a 3–10 frame detector miss as a tracklet break.
        """
        filled = 0
        max_gap = self.bridge_frames
        for cid, hist in list(self.id_history.items()):
            if len(hist) < 2:
                continue
            hist = sorted(hist)
            out = [hist[0]]
            for prev, cur in zip(hist, hist[1:]):
                gap = int(cur[0] - prev[0])
                if 2 <= gap <= max_gap:
                    for k in range(1, gap):
                        t = k / gap
                        bh_p = float(prev[3]) if len(prev) > 3 else 0.0
                        bh_c = float(cur[3]) if len(cur) > 3 else bh_p
                        out.append((
                            int(prev[0] + k),
                            float(prev[1] + t * (cur[1] - prev[1])),
                            float(prev[2] + t * (cur[2] - prev[2])),
                            float(bh_p + t * (bh_c - bh_p)),
                        ))
                        filled += 1
                        self.id_frame_count[cid] += 1
                out.append(cur)
            self.id_history[cid] = out
        return filled

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

        embeddings = {}
        need_appearance = (
            self.use_appearance
            and (APPEARANCE_HARD_GATE or APPEARANCE_WEIGHT > 0)
        )
        if frame is not None and need_appearance:
            wanted, crops = [], []
            for i, raw_id in enumerate(detections.tracker_id):
                cid = self.id_map.get(int(raw_id))
                # Always embed brand-new raw tracks (ReID decision); refresh known.
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

        # ---- Pass A: gather per-detection evidence ----
        n = len(detections)
        raw_ids = [int(r) for r in detections.tracker_id]
        centres = [self._centre(detections.xyxy[i]) for i in range(n)]
        det_labs = [None] * n
        boxes = [None] * n
        if frame is not None:
            for i in range(n):
                x1, y1, x2, y2 = [int(v) for v in detections.xyxy[i]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                boxes[i] = (x1, y1, x2, y2)
                if x2 - x1 >= 8 and y2 - y1 >= 16:
                    try:
                        from team_colour import torso_colour
                        det_labs[i] = torso_colour(frame[y1:y2, x1:x2])
                    except Exception:
                        det_labs[i] = None

        # Known raw tracks claim their canonicals first (short-horizon BT).
        self._claimed_this_frame = {
            self.id_map[r] for r in raw_ids if r in self.id_map
        }

        # New raw tracks: global lost-track assignment (long-horizon ReID).
        need_match = []
        for i, raw_id in enumerate(raw_ids):
            if raw_id in self.id_map:
                continue
            cx, cy = centres[i]
            need_match.append({
                'i': i, 'raw_id': raw_id, 'cx': cx, 'cy': cy,
                'emb': embeddings.get(i), 'det_lab': det_labs[i],
            })
        reid_hits = self._assign_lost_globally(need_match)

        # ---- Pass B: propose a canonical id per detection ----
        proposed = [None] * n
        for i, raw_id in enumerate(raw_ids):
            if raw_id in self.id_map:
                proposed[i] = self.id_map[raw_id]
            elif i in reid_hits:
                proposed[i] = reid_hits[i]
                self.reid_adopts += 1
            else:
                proposed[i] = raw_id  # brand-new identity = ByteTrack raw id

        # ---- Pass C: resolve same-frame collisions WITHOUT minting 100xxx ----
        #
        # When two dets propose the same cid, keep the one closer to the
        # motion-predicted position. Loser: try another lost id, else fall
        # back to its ByteTrack raw id. Emergency 100xxx mint only if even
        # the raw id is already taken this frame (should be near-zero).
        from collections import defaultdict as _dd
        by_cid = _dd(list)
        for i, cid in enumerate(proposed):
            by_cid[cid].append(i)

        used = set()
        canonical_ids = [None] * n
        for cid, idxs in by_cid.items():
            if len(idxs) == 1:
                i = idxs[0]
                canonical_ids[i] = cid
                used.add(cid)
                continue
            # Rank claimants by motion residual (lower = better continuation).
            ranked = sorted(
                idxs, key=lambda i: self._motion_residual(cid, *centres[i]))
            winner = ranked[0]
            canonical_ids[winner] = cid
            used.add(cid)
            for i in ranked[1:]:
                cx, cy = centres[i]
                alt = self._find_lost_match(
                    cx, cy, embeddings.get(i), det_labs[i], exclude=used)
                if alt is not None and alt not in used:
                    canonical_ids[i] = alt
                    used.add(alt)
                    self.reid_adopts += 1
                else:
                    # Revert to ByteTrack identity — NEVER the old 100xxx path.
                    rid = raw_ids[i]
                    if rid not in used:
                        canonical_ids[i] = rid
                        used.add(rid)
                    else:
                        canonical_ids[i] = self._next_free_id()
                        used.add(canonical_ids[i])
                        self.collision_mints += 1

        # ---- Pass D: commit id_map + purge stale raw→cid links on ReID ----
        for i, raw_id in enumerate(raw_ids):
            cid = canonical_ids[i]
            prev = self.id_map.get(raw_id)
            self.id_map[raw_id] = cid
            # If this raw newly adopted a long-lived canonical (ReID), kill
            # every other raw still pointing at it.
            if prev != cid and cid != raw_id:
                self._purge_stale_maps(cid, keep_raw=raw_id)
            elif prev is None and cid != raw_id:
                self._purge_stale_maps(cid, keep_raw=raw_id)

        self._claimed_this_frame = set(canonical_ids)

        # ---- Pass E: update track state ----
        for i, cid in enumerate(canonical_ids):
            cx, cy = centres[i]
            emb = embeddings.get(i)
            det_lab = det_labs[i]
            self.last_seen[cid] = (cx, cy, self.frame_n)
            if emb is not None:
                prev = self.id_appearance.get(cid)
                self.id_appearance[cid] = (
                    emb if prev is None else 0.7 * prev + 0.3 * emb)
                self.id_embed_frame[cid] = self.frame_n
            if det_lab is not None:
                prev_lab = self.id_lab.get(cid)
                self.id_lab[cid] = (
                    det_lab if prev_lab is None
                    else (0.7 * prev_lab + 0.3 * det_lab).astype(np.float32))
            self.id_frame_count[cid] += 1
            if detections.class_id is not None:
                self.id_class_votes[cid][int(detections.class_id[i])] += 1
            if cid not in self.id_first_pos:
                self.id_first_pos[cid] = (cx, cy)
            self.id_last_pos[cid] = (cx, cy)
            box = boxes[i]
            if (STITCH and frame is not None and box is not None and
                    len(self.id_crops[cid]) < TEAM_CROPS_PER_ID and
                    self.frame_n % TEAM_CROP_STRIDE == 0):
                x1, y1, x2, y2 = box
                if x2 - x1 >= 8 and y2 - y1 >= 16:
                    self.id_crops[cid].append(frame[y1:y2, x1:x2].copy())
            if TRACK_DUMP or STITCH:
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

    def path_net_ratio(self, cid) -> float:
        """Cumulative path / straight-line net. High values flag identity welds."""
        net = self.net_displacement(cid)
        path = float(self.id_path_px.get(cid, 0.0))
        if net < 1.0:
            return path  # unbounded wandering with no net progress
        return path / net

    def _apply_track_cuts(self, cid, hist, cuts) -> int:
        """Rewrite state so each cut boundary becomes a new canonical id."""
        if not cuts:
            return 0
        boundaries = []
        for c in cuts:
            new_cid = self._next_free_id()
            boundaries.append((hist[c][0], new_cid))
        existing = list(self.id_splits.get(cid, []))
        existing.extend(boundaries)
        self.id_splits[cid] = existing

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
                self.id_class_votes[seg_id][self.dominant_class(cid) or
                                            PLAYER_CLASS_ID] = len(seg)
                self.id_crops[seg_id] = self.id_crops.get(cid, [])[:]
                if cid in self.id_team:
                    self.id_team[seg_id] = self.id_team[cid]
            path = 0.0
            for a, b in zip(seg[:-1], seg[1:]):
                path += float(np.hypot(b[1] - a[1], b[2] - a[2]))
            self.id_path_px[seg_id] = path
        return len(cuts)

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
            splits += self._apply_track_cuts(cid, hist, cuts)
        return splits

    def split_inefficient_tracks(self, fps: float):
        """Cut slow identity welds that never teleport in one hop.

        Speed gating misses tracks like mot_sota_v3 id 367 (path/net ≈ 777):
        the id walks at human pace but zigzags between two players. Over a
        rolling window, cumulative path ≫ chord length means the track is
        visiting two bodies. Cut at the sample farthest from the chord — the
        natural hand-off point between the two players.
        """
        splits = 0
        win = max(2, int(EFFICIENCY_WINDOW_SECONDS * fps))
        for cid in list(self.id_history.keys()):
            hist = sorted(self.id_history[cid])
            if len(hist) < win + 2:
                continue
            frames = np.array([h[0] for h in hist], dtype=float)
            pos = np.array([[h[1], h[2]] for h in hist], dtype=float)
            # Cumulative path along the polyline.
            steps = np.zeros(len(pos), dtype=float)
            for i in range(1, len(pos)):
                steps[i] = steps[i - 1] + float(np.hypot(
                    pos[i][0] - pos[i - 1][0], pos[i][1] - pos[i - 1][1]))

            cuts = []
            i = win
            while i < len(hist):
                # Find earliest j whose frame span ≈ window.
                target = frames[i] - win
                j = int(np.searchsorted(frames, target, side='left'))
                j = min(max(j, 0), i - 2)
                if frames[i] - frames[j] < win * 0.5:
                    i += 1
                    continue
                path = float(steps[i] - steps[j])
                net = float(np.hypot(pos[i][0] - pos[j][0],
                                     pos[i][1] - pos[j][1]))
                if (path >= MIN_EFFICIENCY_PATH_PX and
                        path / max(net, 1.0) > MAX_PATH_NET_RATIO):
                    # Farthest point from the chord j→i is the weld hand-off.
                    chord = pos[i] - pos[j]
                    chord_len = float(np.linalg.norm(chord)) + 1e-6
                    unit = chord / chord_len
                    best_k, best_d = j + 1, -1.0
                    for k in range(j + 1, i):
                        rel = pos[k] - pos[j]
                        proj = float(rel @ unit)
                        proj = min(max(proj, 0.0), chord_len)
                        perp = float(np.linalg.norm(rel - proj * unit))
                        if perp > best_d:
                            best_d, best_k = perp, k
                    # Require a real lateral excursion, not jitter on a line.
                    if best_d >= 25.0 and best_k not in cuts:
                        cuts.append(best_k)
                        # Skip ahead past this window so one weld ≠ N cuts.
                        i = best_k + win
                        continue
                i += max(1, win // 3)

            # De-dupe / order, then apply.
            cuts = sorted(set(cuts))
            cuts = [c for k, c in enumerate(cuts)
                    if k == 0 or c - cuts[k - 1] > win // 2]
            n = self._apply_track_cuts(cid, hist, cuts)
            splits += n
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
        self.team_vote_frac = {}
        self.team_sep = 0.0
        try:
            from team_colour import TeamColourClassifier
            crops_all = [c for v in self.id_crops.values() for c in v]
            if len(crops_all) >= 20:
                clf = TeamColourClassifier().fit(crops_all)
                self.team_sep = float(clf.separation)
                if clf.centres is not None and clf.separation >= 1.0:
                    for cid, crops in self.id_crops.items():
                        t, frac = clf.predict_tracklet_confident(crops)
                        if t is not None:
                            self.id_team[cid] = t
                            self.team_vote_frac[cid] = frac
                    counts = Counter(self.id_team.values())
                    hard = (clf.separation >= TEAM_HARD_SEP_MIN)
                    print(f"Team colour: separation {clf.separation:.2f}, "
                          f"{dict(counts)} across {len(self.id_team)} ids"
                          f"{' [HARD veto armed]' if hard else ' [soft penalty]'}")
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
            xy = np.array([[h[1], h[2]] for h in hist], dtype=float)
            # Split internal gaps: a canonical id may already span a re-id
            # jump, and this stage should judge on its own evidence.
            breaks = [0] + [i for i in range(1, len(frames))
                            if frames[i] - frames[i - 1] > fps] + [len(frames)]
            for a, b in zip(breaks[:-1], breaks[1:]):
                if b - a >= 2:
                    tracklets.append(stitch.Tracklet(
                        cid, frames[a:b], xy[a:b],
                        self.dominant_class(cid) or PLAYER_CLASS_ID,
                        team=self.id_team.get(cid),
                        team_conf=self.team_vote_frac.get(cid),
                        team_sep=self.team_sep,
                        appearance=self.id_appearance.get(cid)))
        if len(tracklets) < 2:
            return 0, 0

        identities, links = stitch.stitch_global(
            tracklets, fps,
            keep_thin=STITCH_KEEP_THIN,
            max_gap_frames=STITCH_MAX_GAP_FRAMES,
            appearance_min_cosine=STITCH_APPEARANCE_MIN_COSINE,
            sim_percentile=STITCH_SIM_PERCENTILE,
            cost_percentile=STITCH_COST_PERCENTILE,
        )

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
        n_thin = sum(1 for l in links if l['thin'])
        n_applied = sum(1 for l in links
                        if STITCH_KEEP_THIN or not l['thin'])
        if not remap:
            return n_applied, n_thin

        for src, dst in remap.items():
            self.id_frame_count[dst] += self.id_frame_count.pop(src, 0)
            self.id_path_px[dst] += self.id_path_px.pop(src, 0.0)
            for k, v in self.id_class_votes.pop(src, {}).items():
                self.id_class_votes[dst][k] += v
            self.id_history[dst] = sorted(self.id_history[dst] +
                                          self.id_history.pop(src, []))
            self.id_first_pos.pop(src, None)
            self.id_last_pos.pop(src, None)
            if src in self.id_team and dst not in self.id_team:
                self.id_team[dst] = self.id_team.pop(src)
            else:
                self.id_team.pop(src, None)
        # Endpoints have to come from the merged history, not from whichever
        # fragment happened to be written last, or net displacement is wrong.
        for dst in set(remap.values()):
            hist = self.id_history.get(dst)
            if hist:
                self.id_first_pos[dst] = (hist[0][1], hist[0][2])
                self.id_last_pos[dst] = (hist[-1][1], hist[-1][2])
                # Recompute path after merge so efficiency stats stay honest.
                path = 0.0
                for a, b in zip(hist[:-1], hist[1:]):
                    path += float(np.hypot(b[1] - a[1], b[2] - a[2]))
                self.id_path_px[dst] = path
        for raw, cid in list(self.id_map.items()):
            if cid in remap:
                self.id_map[raw] = remap[cid]
        # Pass 2 replays pass 1's per-frame ids and needs the same folding.
        self.id_remap.update(remap)
        return n_applied, n_thin

    def weld_guard(self, fps: float) -> int:
        """Split identity welds before metric export (mot_sota_v6).

        Does NOT reset id_splits (physics pass may have already cut teleports).
        Adds: (1) stricter residual teleports, (2) rolling path/net inflection
        cuts on tracks whose whole-track path/net still exceeds the ceiling.
        """
        splits = 0
        win = max(1, int(SPEED_WINDOW_SECONDS * fps))
        for cid in list(self.id_history.keys()):
            hist = sorted(self.id_history[cid])
            if len(hist) < 4:
                continue
            frames = np.array([h[0] for h in hist], dtype=float)
            pos = np.array([[h[1], h[2]] for h in hist], dtype=float)
            heights = np.array([h[3] if len(h) > 3 else 0.0 for h in hist])
            if len(pos) >= 5:
                k = np.ones(5) / 5
                pad = np.vstack([np.repeat(pos[:1], 2, axis=0), pos,
                                 np.repeat(pos[-1:], 2, axis=0)])
                pos_s = np.stack([np.convolve(pad[:, 0], k, 'valid'),
                                  np.convolve(pad[:, 1], k, 'valid')], axis=1)
            else:
                pos_s = pos
            cuts = []
            for i in range(1, len(hist)):
                gap = frames[i] - frames[i - 1]
                j = i - 1 if gap > 1 else max(0, i - win)
                if gap <= 1 and frames[i] - frames[j] < win * 0.5:
                    continue
                dt = (frames[i] - frames[j]) / fps
                if dt <= 0:
                    continue
                h = max(float(np.mean(heights[j:i + 1])), 1e-6)
                if h < 8:
                    continue
                dist = float(np.hypot(pos_s[i][0] - pos_s[j][0],
                                      pos_s[i][1] - pos_s[j][1]))
                if dist / h / dt > WELD_TELEPORT_BODY_H_PER_SEC:
                    cuts.append(i)
            cuts = [c for k, c in enumerate(cuts)
                    if k == 0 or c - cuts[k - 1] > win]
            splits += self._apply_track_cuts(cid, hist, cuts)

        if not WELD_GUARD_EFFICIENCY:
            return splits
        # Rolling path/net, restricted to tracks that look welded overall —
        # running it on every track cut ordinary direction changes (v8).
        welded = {cid for cid in self.id_history
                  if self.path_net_ratio(cid) > WELD_PATH_NET_CEILING}
        if not welded:
            return splits
        keep = {cid: self.id_history[cid] for cid in list(self.id_history)
                if cid not in welded}
        for cid in keep:
            self.id_history.pop(cid)
        saved = globals()['MAX_PATH_NET_RATIO']
        globals()['MAX_PATH_NET_RATIO'] = min(saved, 8.0)
        try:
            n_eff = self.split_inefficient_tracks(fps)
        finally:
            globals()['MAX_PATH_NET_RATIO'] = saved
            self.id_history.update(keep)
        splits += n_eff
        return splits

    def valid_ids(self) -> set:
        """Client roster filter: clean long tracks in the 24–40 band.

        Never returns empty when tracks exist — empty good_ids makes pass 2
        draw zero ellipses (blank tracking video), which is worse than showing
        a few short/noisy fragments on a calibration clip.
        """
        def gather(max_path_net, min_frames, require_motion=True,
                   allow_collision=False):
            out = []
            for cid, count in self.id_frame_count.items():
                if count < min_frames:
                    continue
                if (EXCLUDE_COLLISION_IDS_FROM_PASSED and not allow_collision
                        and cid >= 100000):
                    continue
                if (require_motion
                        and self.dominant_class(cid) != GOALKEEPER_CLASS_ID):
                    seconds = count / self.fps
                    speed = (self.id_path_px[cid] / seconds
                             if seconds > 0 else 0.0)
                    if (speed < MIN_SPEED_PX_PER_SEC and
                            self.net_displacement(cid) < MIN_NET_DISPLACEMENT_PX):
                        continue
                pn = self.path_net_ratio(cid)
                if pn > max_path_net:
                    continue
                out.append((cid, count, pn))
            out.sort(key=lambda t: (-t[1], t[2]))
            return out

        min_f = self.min_frames
        candidates = gather(MAX_PASSED_PATH_NET, min_f)
        if len(candidates) < TARGET_PASSED_MIN:
            candidates = gather(MAX_PASSED_PATH_NET * 2.0, min_f)
        if len(candidates) < TARGET_PASSED_MIN:
            candidates = gather(1e9, max(2, min_f // 3), require_motion=False)
        # Absolute last resort: longest-lived tracks (≥2 frames). A 150-frame
        # clip at 55fps has no 1s continuous fragments after MOT splits — the
        # old ≥1s floor still returned [] and pass 2 rendered nothing.
        if not candidates and self.id_frame_count:
            ranked = sorted(self.id_frame_count.items(), key=lambda x: -x[1])
            candidates = [
                (cid, count, self.path_net_ratio(cid))
                for cid, count in ranked if count >= 2
            ]
            print(f"  valid_ids: STRICT FILTERS KEPT 0 — using top "
                  f"{min(len(candidates), TARGET_PASSED_MAX)} tracks by "
                  f"lifetime so pass 2 can draw boxes")
        if len(candidates) > TARGET_PASSED_MAX:
            candidates = candidates[:TARGET_PASSED_MAX]
        return {c[0] for c in candidates}


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
            # Request at the WEAK-tier floor so soft shadow-edge boxes still
            # reach ByteTrack; activation (not this conf) decides new tracks.
            return rfdetr_onnx.detect(frame, conf=min(TRACK_DETECT_FLOOR,
                                                      BALL_MIN_CONF))
        result = model(frame, imgsz=INFERENCE_IMGSZ,
                       conf=min(TRACK_DETECT_FLOOR, BALL_MIN_CONF),
                       agnostic_nms=True, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    def get_player_detections(frame, raw=None):
        detections = detect_raw(frame) if raw is None else raw
        if len(detections) and detections.confidence is not None:
            # Keep the weak tier (TRACK_DETECT_FLOOR .. ACTIVATION) so ByteTrack
            # can CONTINUE tracks through shadow; do NOT filter at activation.
            detections = detections[detections.confidence >= TRACK_DETECT_FLOOR]
        if len(detections):
            h = detections.xyxy[:, 3] - detections.xyxy[:, 1]
            w = detections.xyxy[:, 2] - detections.xyxy[:, 0]
            detections = detections[
                (h >= MIN_BOX_HEIGHT_PX) & (w >= MIN_BOX_WIDTH_PX)]
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
    if max_frames:
        # Short calibration clips must not demand long continuous tracking.
        # 10% of the clip (min 2 frames) — 150 frames → min_frames=15.
        cap = max(2, int(0.10 * max_frames))
        if tracker1.min_frames > cap:
            print(f"  min_frames capped {tracker1.min_frames} → {cap} "
                  f"(10% of --max_frames={max_frames})")
            tracker1.min_frames = cap
    # Per-frame (online canonical id, box, class) so pass 2 can REPLAY pass 1
    # instead of re-running detection + tracking. Re-running was never
    # faithful: pass 2 skipped appearance ReID and ByteTrack re-seeded, so its
    # ids diverged from pass 1's stable set and most players rendered grey.
    frame_boxes: dict = {}
    for frame in tqdm(
        video_frames(source_video_path, max_frames=max_frames, start_frame=START_FRAME),
        desc='Pass 1'
    ):
        raw = detect_raw(frame)
        idx = tracker1.frame_n
        det = tracker1.update(get_player_detections(frame, raw), frame)
        if not no_render and det.tracker_id is not None and len(det):
            frame_boxes[idx] = (
                det.tracker_id.astype(np.int64).copy(),
                det.xyxy.astype(np.float32).copy(),
                None if det.class_id is None else det.class_id.copy())
        # Ball in pass 1 costs nothing now that detection is shared, and gives
        # the map a full trajectory to smooth rather than a strobing marker.
        if SHOW_BALL:
            b = get_ball(frame, raw)
            if b is not None and len(b):
                ball_history.append((tracker1.frame_n,
                                     float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                                     float(b.xyxy[0][3])))

    raw_id_count = len(tracker1.id_frame_count)
    n_bridged = tracker1.interpolate_short_gaps()
    if n_bridged:
        print(f"\nGap bridge: filled {n_bridged} missing sample(s) "
              f"(≤{BRIDGE_MAX_FRAMES} frames detector/occlusion dropouts)")
    if SPLIT_IMPLAUSIBLE:
        # Cut BEFORE stitching: undo bad splices first, then let stitching
        # re-link the pieces on their own evidence (including team colour).
        n_splits = tracker1.split_implausible_tracks(fps)
        if n_splits:
            print(f"\nPhysics check: {n_splits} track(s) cut where motion "
                  f"exceeded {MAX_BODY_HEIGHTS_PER_SEC} body-heights/sec "
                  f"— an id had been handed between two players")
        if SPLIT_INEFFICIENT:
            n_eff = tracker1.split_inefficient_tracks(fps)
            if n_eff:
                print(f"Efficiency check: {n_eff} track(s) cut where path/net "
                      f"exceeded {MAX_PATH_NET_RATIO} over "
                      f"{EFFICIENCY_WINDOW_SECONDS:.1f}s — slow identity weld")
    print(f"\nOnline association: ReID adopts={tracker1.reid_adopts}, "
          f"stale-map purges={tracker1.stale_map_purges}, "
          f"emergency 100xxx mints={tracker1.collision_mints} "
          f"(target ≈ 0)")
    if STITCH:
        n_links, n_thin = tracker1.merge_stitched(fps)
        dropped = n_thin if not STITCH_KEEP_THIN else 0
        print(f"\nStitching: {n_links} links applied "
              f"({n_thin} thin{' — DROPPED' if dropped else ' kept'}), "
              f"gap≤{STITCH_MAX_GAP_FRAMES}f, sim≥p{STITCH_SIM_PERCENTILE:.0f}, "
              f"{raw_id_count} -> {len(tracker1.id_frame_count)} identities")
        if n_thin and STITCH_KEEP_THIN:
            print(f"  {n_thin} links had a close runner-up. Thin links measured "
                  f"29% correct against 93% for confident ones — check these "
                  f"ids first when validating.")
        elif dropped:
            print(f"  Declined {dropped} thin link(s) (STITCH_KEEP_THIN=False). "
                  f"Use --keep_thin_stitch to apply them for audit.")
    if WELD_GUARD:
        n_weld = tracker1.weld_guard(fps)
        print(f"\nWeld guard: {n_weld} cut(s) "
              f"(teleport>{WELD_TELEPORT_BODY_H_PER_SEC} bh/s or "
              f"path_net ceiling {WELD_PATH_NET_CEILING})")

    good_ids = tracker1.valid_ids()
    all_ids  = set(tracker1.id_frame_count.keys())
    # Belt-and-suspenders: never hand pass 2 an empty keep-set when the tracker
    # saw people — that produces a blank output video (no ellipses / minimap).
    if not good_ids and all_ids:
        ranked = sorted(tracker1.id_frame_count.items(), key=lambda x: -x[1])
        good_ids = {cid for cid, n in ranked[:TARGET_PASSED_MAX] if n >= 2}
        print(f"  WARNING: valid_ids empty — forcing render of "
              f"{len(good_ids)} longest tracks")

    print(f"\nPass 1 complete (mot_sota_v6):")
    print(f"  Total canonical IDs : {len(all_ids)}")
    print(f"  Passed filters      : {len(good_ids)} "
          f"(target {TARGET_PASSED_MIN}-{TARGET_PASSED_MAX})")
    print(f"  Removed as noise    : {len(all_ids) - len(good_ids)}")
    collision_ids = sorted(i for i in all_ids if i >= 100000)
    if collision_ids:
        print(f"  Cut-generated IDs   : {len(collision_ids)} (≥100000, from "
              f"physics/weld cuts; emergency mints={tracker1.collision_mints})")
    print(f"  Valid IDs           : {sorted(good_ids)}")

    if focus_id is not None and focus_id not in good_ids:
        print(f"  Warning: #{focus_id} didn't pass filters — showing anyway")
        good_ids.add(focus_id)

    # Save ID list JSON for manual validation
    id_stats = []
    for cid in sorted(all_ids):
        net = float(tracker1.net_displacement(cid))
        path = float(tracker1.id_path_px[cid])
        id_stats.append({
            "canonical_id":        int(cid),
            "frames_seen":         int(tracker1.id_frame_count[cid]),
            "path_length_px":      round(path, 1),
            "speed_px_per_sec":    round(path /
                                         max(tracker1.id_frame_count[cid] / fps, 1e-6), 1),
            "net_displacement_px": round(net, 1),
            "path_net_ratio":      round(tracker1.path_net_ratio(cid), 2),
            "class":               {BALL_CLASS_ID: 'ball',
                                    GOALKEEPER_CLASS_ID: 'goalkeeper',
                                    PLAYER_CLASS_ID: 'player',
                                    REFEREE_CLASS_ID: 'referee'}.get(
                                        tracker1.dominant_class(cid), 'unknown'),
            "passed_filter":       bool(cid in good_ids),
            "player_name":         None,
            "team":                tracker1.id_team.get(cid),
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
    # No detector, no tracker here: every box and id was recorded in pass 1
    # and is replayed through the same split/stitch folding that produced the
    # stats, so what is drawn is exactly what was measured.
    print(f"Pass 2: rendering output video (replaying {len(frame_boxes)} "
          f"frames of pass 1 identities, no re-detection)...")

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
        # Chain: efficiency may cut a fragment that speed already split off.
        out = cid
        seen = set()
        while out not in seen:
            seen.add(out)
            bounds = split_map.get(out)
            if not bounds:
                break
            nxt = out
            for from_frame, new_cid in bounds:
                if frame_no >= from_frame:
                    nxt = new_cid
            if nxt == out:
                break
            out = nxt
        return out

    remap = dict(tracker1.id_remap)

    def apply_remap(cid):
        seen = set()
        while cid in remap and cid not in seen:
            seen.add(cid)
            cid = remap[cid]
        return cid

    def final_id(cid, frame_no):
        # Same order pass 1 applied: physics/efficiency cuts (pre-merge ids),
        # stitched merge, then weld-guard cuts (post-merge ids). apply_splits
        # chains, so running it on both sides is safe.
        cid = apply_splits(cid, frame_no)
        cid = apply_remap(cid)
        return apply_splits(cid, frame_no)

    # ball_history frames were taken AFTER tracker1.update bumped frame_n.
    ball_by_frame = {int(f) - 1: (x, y) for f, x, y in ball_history}

    for frame in video_frames(source_video_path, max_frames=max_frames, start_frame=START_FRAME):
        rec = frame_boxes.get(frame_n)
        if rec is None:
            detections = sv.Detections.empty()
        else:
            ids, xyxy, cls = rec
            detections = sv.Detections(
                xyxy=xyxy,
                # annotators colour by class and reject class_id=None
                class_id=(cls if cls is not None
                          else np.full(len(ids), PLAYER_CLASS_ID, dtype=int)),
                tracker_id=np.array([final_id(int(c), frame_n) for c in ids],
                                    dtype=int))
        annotated  = frame.copy()
        ball = ball_by_frame.get(frame_n)
        if ball is not None:
            cx, cy = int(ball[0]), int(ball[1])
            cv2.circle(annotated, (cx, cy - 8), 12, (0, 255, 255), 3)
            cv2.putText(annotated, 'BALL', (cx-20, cy-28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Never hide a detected player. Earlier this dropped every detection
        # whose id was not in good_ids, so anyone on a short fragment id (or
        # an id that pass 2's tracker assigned differently from pass 1) simply
        # vanished from the video and looked like a detection failure.
        # Stable ids are drawn in colour with their number; the rest in grey.
        unstable = None
        n_detected = len(detections) if detections.tracker_id is not None else 0
        if detections.tracker_id is not None and len(detections) > 0:
            valid_mask = np.isin(detections.tracker_id, list(good_ids))
            unstable   = detections[~valid_mask]
            detections = detections[valid_mask]

        if unstable is not None and len(unstable) > 0 and focus_id is None:
            for i in range(len(unstable)):
                box = unstable.xyxy[i]
                cx  = int((box[0]+box[2])/2)
                bot = int(box[3])
                cv2.ellipse(annotated, (cx, bot),
                            (max(4, int((box[2]-box[0])/2)), 8),
                            0, -45, 235, (170, 170, 170), 2)

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
        cv2.rectangle(annotated, (0, 0), (520, 36), (0, 0, 0), -1)
        n_visible = len(detections) if detections.tracker_id is not None else 0
        cv2.putText(annotated,
                    f"Players: {n_detected}  |  Stable ID: {n_visible}  |  "
                    f"IDs: {len(good_ids)}",
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
        PITCH_POLYGON = validate_pitch_polygon(
            PITCH_POLYGON, source_video_path, _vi.width, _vi.height)
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
        # Reject polygons that would wipe players (dry pitch / short sample).
        PITCH_POLYGON = validate_pitch_polygon(
            PITCH_POLYGON, source_video_path, _vi.width, _vi.height)
        if PITCH_POLYGON is None:
            print("  Pitch filter: using percentage bounds "
                  f"({PITCH_LEFT_PCT}-{PITCH_RIGHT_PCT}% x, "
                  f"{PITCH_TOP_PCT}-{PITCH_BOTTOM_PCT}% y)")
        if PITCH_POLYGON is not None:
            _out = output_path_for(source_video_path, 'pitch_polygon').replace(
                '.json', '.npy')
            np.save(_out, PITCH_POLYGON)
            print(f"  Pitch polygon saved to {os.path.basename(_out)} — "
                  f"VIEW IT before trusting a run. A polygon that silently "
                  f"clips the near half of the pitch still produces plausible "
                  f"detection counts.")
    print(f"Start frame: {START_FRAME}")
    print(f"Detector: {DETECTOR}  imgsz: {INFERENCE_IMGSZ}  "
          f"detect_floor: {TRACK_DETECT_FLOOR}  "
          f"activation: {TRACK_ACTIVATION_THRESHOLD}  "
          f"match: {TRACK_MATCHING_THRESHOLD}  "
          f"bt_lost: {BYTE_TRACK_LOST_SECONDS}s  "
          f"reid: {REID_WINDOW_SECONDS}s  "
          f"touchline buffer: {TOUCHLINE_BUFFER_PX}px")

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
    with RobustVideoSink(target_video_path, video_info) as sink:
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
        help='Tag for this run, used in output filenames. Defaults to '
             'mot_sota_v6. Pass a custom label to keep parallel experiments.')
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
    parser.add_argument('--keep_thin_stitch', action='store_true',
                        help='Apply thin-margin stitch links (default: drop them; '
                             'thin links measured ~29%% correct).')
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
    if args.keep_thin_stitch:
        STITCH_KEEP_THIN = True
    if args.track_dump:
        TRACK_DUMP = True
    if args.include_referees:
        INCLUDE_REFEREES = True
    if args.show_ball:
        SHOW_BALL = True
    if args.detector:
        DETECTOR = args.detector
        if args.conf is None and args.detector == 'rfdetr':
            # INFERENCE_CONF stays as the documented "working" conf for logs /
            # YOLO modes. PLAYER_TRACKING itself feeds ByteTrack at
            # TRACK_DETECT_FLOOR (weak tier) and opens tracks only above
            # TRACK_ACTIVATION_THRESHOLD — see those constants.
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
            # soft detections that hurt as track seeds help as track
            # continuations.
            INFERENCE_CONF = 0.20
            TRACK_DETECT_FLOOR = 0.15
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