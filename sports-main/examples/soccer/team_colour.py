"""Team classification from kit colour, for kits a general embedding cannot split.

WHY NOT SigLIP
--------------
`sports/common/team.py` embeds the whole player box with SigLIP, reduces with
UMAP and clusters with KMeans. On 14_09 it split 60 v 2 — stable but splitting
on something that is not the kit. Measured on that footage:

    SigLIP + UMAP + KMeans        60 v 2 split
    blue-pixel brightness         30 v 14, visually correct

The reason is in the footage: BOTH TEAMS WEAR BLUE, dark navy against light sky
blue. They differ in LIGHTNESS, not hue. A whole-box embedding is dominated by
pitch, shadow and pose, and hue clustering keys on the one axis these two kits
happen to share. See data/reference/kits_14_09.png.

WHAT THIS DOES INSTEAD
----------------------
Look only at the torso, and cluster in a space where "navy vs sky blue" and
"red vs blue" are both large distances:

  1. crop the torso region, discarding head, legs and most background
  2. drop grey/washed-out pixels, which are shadow, skin and pitch line
  3. take the median colour in CIELAB, which is perceptually uniform — a fixed
     distance means a similar perceived difference anywhere in the space, so
     lightness differences count properly instead of being swamped by hue
  4. cluster those per-crop colours into two teams
  5. VOTE PER TRACKLET, not per crop — every crop of one tracklet is the same
     player by construction, so a majority vote turns a noisy per-crop label
     into a stable per-player one

Step 5 is what makes the result usable: per-crop labels are ~85% consistent,
per-tracklet votes measured 93%.

Referees and goalkeepers are NOT a third cluster here — they are handled by the
detector's own class ids, which are reliable (0.71 confidence on referees). Only
outfield players are clustered.

HONEST LIMITS
-------------
93% is not good enough to VETO a stitch link — 7% wrong labels would forbid
correct joins. Use `team_penalty()` to add a cost instead, so a mismatch has to
be outvoted by strong motion evidence rather than being ruled out.

Two teams in near-identical kit cannot be separated by any colour method, and
this will report low confidence rather than guess. Check `separation` before
relying on the result.
"""
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Torso window as a fraction of the box. Starts below the head and stops above
# the shorts: shirts are the only large uniform colour a player carries, and
# shorts are often a different colour from the shirt on the same kit.
TORSO_TOP, TORSO_BOTTOM = 0.15, 0.55
TORSO_LEFT, TORSO_RIGHT = 0.25, 0.75

# Pixels below this saturation are shadow, skin, white trim or pitch line —
# they carry no team information and would pull every median toward grey.
MIN_SATURATION = 45
# Below this many usable pixels the crop is too small or too occluded to judge.
MIN_PIXELS = 15

# A mismatch costs this much extra in stitch link cost. Chosen to be
# comparable to the no-link price (0.30) so a team disagreement roughly halves
# a link's chances without forbidding it outright — see the module docstring on
# why a blanket hard veto is wrong at 93% accuracy.
TEAM_MISMATCH_PENALTY = 0.25
# Hard veto only when cluster separation and per-tracklet vote margin are both
# strong. Tuned for navy-vs-sky kits on 14_09 panoramic footage.
TEAM_HARD_SEP_MIN = 1.35
TEAM_HARD_VOTE_FRAC = 0.70


def torso_colour(crop: np.ndarray) -> Optional[np.ndarray]:
    """Median CIELAB colour of the shirt region, or None if unreadable."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    roi = crop[int(h * TORSO_TOP):int(h * TORSO_BOTTOM),
               int(w * TORSO_LEFT):int(w * TORSO_RIGHT)]
    if roi.size == 0:
        return None
    sat = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 1]
    mask = sat > MIN_SATURATION
    if mask.sum() < MIN_PIXELS:
        return None
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    # Median, not mean: one bright sponsor patch or a sliver of pitch should
    # not move the estimate, and medians are robust to exactly that.
    return np.median(lab[mask].reshape(-1, 3), axis=0).astype(np.float32)


class TeamColourClassifier:
    """Two-team clustering on torso colour, with per-tracklet voting."""

    def __init__(self):
        self.centres: Optional[np.ndarray] = None
        self.separation: float = 0.0

    def fit(self, crops: Sequence[np.ndarray]) -> 'TeamColourClassifier':
        cols = [c for c in (torso_colour(x) for x in crops) if c is not None]
        if len(cols) < 4:
            return self
        X = np.stack(cols)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
        self.centres = km.cluster_centers_

        # How far apart the kits are, relative to how spread each cluster is.
        # Below ~1.0 the two teams are not meaningfully distinguishable by
        # colour and the caller should not trust the labels.
        gap = float(np.linalg.norm(self.centres[0] - self.centres[1]))
        spread = float(np.mean([np.linalg.norm(X[km.labels_ == k] -
                                               self.centres[k], axis=1).mean()
                                for k in (0, 1)])) or 1e-6
        self.separation = gap / spread
        return self

    def predict_crop(self, crop: np.ndarray) -> Optional[int]:
        col = torso_colour(crop)
        if col is None or self.centres is None:
            return None
        return int(np.argmin(np.linalg.norm(self.centres - col, axis=1)))

    def predict_tracklet(self, crops: Sequence[np.ndarray]) -> Optional[int]:
        """Majority vote over a tracklet's crops."""
        t, _ = self.predict_tracklet_confident(crops)
        return t

    def predict_tracklet_confident(
            self, crops: Sequence[np.ndarray]
    ) -> Tuple[Optional[int], float]:
        """Majority vote plus vote fraction (confidence in [0, 1])."""
        votes = [v for v in (self.predict_crop(c) for c in crops) if v is not None]
        if not votes:
            return None, 0.0
        counts = np.bincount(votes)
        team = int(counts.argmax())
        return team, float(counts[team] / len(votes))


def team_penalty(team_a: Optional[int], team_b: Optional[int]) -> float:
    """Extra stitch cost for linking two tracklets on different teams.

    A penalty rather than a veto for the common case. At 93% accuracy a blanket
    hard rule would forbid roughly one correct link in fourteen.
    """
    if team_a is None or team_b is None or team_a == team_b:
        return 0.0
    return TEAM_MISMATCH_PENALTY


def team_hard_conflict(
        team_a: Optional[int], team_b: Optional[int],
        conf_a: Optional[float], conf_b: Optional[float],
        separation: float,
        sep_min: float = TEAM_HARD_SEP_MIN,
        vote_min: float = TEAM_HARD_VOTE_FRAC,
) -> bool:
    """True when opposing team labels should HARD-VETO a stitch/re-id link."""
    if team_a is None or team_b is None or team_a == team_b:
        return False
    if separation < sep_min:
        return False
    ca = 0.0 if conf_a is None else float(conf_a)
    cb = 0.0 if conf_b is None else float(conf_b)
    return ca >= vote_min and cb >= vote_min