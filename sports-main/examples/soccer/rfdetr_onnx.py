"""RF-DETR (Roboflow football-players-detection v20) as a local ONNX detector.

Why this exists: this model detects our footage far better than any YOLO weights
we have. Measured on 12_08, a match no model had seen:

    our round2 (YOLO)   0.0 detections/frame on the moving view
    our stock  (YOLO)   2.2
    RF-DETR v20        25.9, median confidence 0.76, and it finds the ball

The reason is architectural, not a training recipe: RF-DETR is a transformer
detector, which copes with the huge scale range in this footage (players from
16px to 126px in one frame) far better than YOLO's anchor-based heads.

Runs entirely locally from cached weights — no API, no credits.

Notes on the format, since it differs from YOLO in every respect:
  input   [1,3,576,576], STRETCHED (not letterboxed), ImageNet-normalised
  dets    [1,300,4]   300 object queries, boxes as normalised cx,cy,w,h
  labels  [1,300,366] per-class logits; only indices 1..4 are meaningful
                      (0 is background), mapping to ball/goalkeeper/player/referee

There is no NMS inside the model — DETR-family models emit one query per object
by design — but duplicate queries still occur, so callers should keep using
suppress_contained_boxes() downstream.
"""
import os
from typing import Optional

import cv2
import numpy as np
import supervision as sv

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', '..', 'external', 'roboflow_v20', 'weights.onnx')
WEIGHTS = os.path.normpath(WEIGHTS)

INPUT_SIZE = 576
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ONNX class index -> our pipeline's class id. Index 0 is background.
_CLASS_MAP = {1: 0, 2: 1, 3: 2, 4: 3}   # ball, goalkeeper, player, referee

_session = None


def session(weights: Optional[str] = None):
    """Load the ONNX session once.

    Prefer CUDA on RunPod / GPU hosts; fall back to CPU if CUDA EP is missing
    (local Mac/CPU boxes). Provider order is explicit so ORT does not silently
    stay on CPU when a GPU is available.
    """
    global _session
    if _session is None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = os.cpu_count() or 4
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        _session = ort.InferenceSession(weights or WEIGHTS, so,
                                        providers=providers)
        active = _session.get_providers()
        print(f"  RF-DETR ONNX providers requested={providers} active={active}")
        if 'CUDAExecutionProvider' not in active:
            print("  WARNING: CUDAExecutionProvider not active — running on CPU. "
                  "Install onnxruntime-gpu on RunPod for GPU inference.")
    return _session


def detect(frame: np.ndarray, conf: float = 0.25) -> sv.Detections:
    """Detect on a BGR frame, returning boxes in that frame's coordinates."""
    h, w = frame.shape[:2]
    sess = session()

    # Stretch, not letterbox — matches the model's declared preprocessing.
    x = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))[:, :, ::-1].astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    inp = np.expand_dims(np.transpose(x, (2, 0, 1)), 0).astype(np.float32)

    boxes, logits = sess.run(None, {sess.get_inputs()[0].name: inp})
    boxes, logits = boxes[0], logits[0]                      # (300,4), (300,366)

    scores = 1.0 / (1.0 + np.exp(-logits))                   # sigmoid
    usable = scores[:, list(_CLASS_MAP)]                     # only real classes
    best = usable.argmax(axis=1)
    best_score = usable[np.arange(len(usable)), best]
    keep = best_score >= conf
    if not keep.any():
        return sv.Detections.empty()

    b = boxes[keep]
    cls = np.array([_CLASS_MAP[list(_CLASS_MAP)[i]] for i in best[keep]], dtype=int)
    # normalised cxcywh -> absolute xyxy in the ORIGINAL frame, undoing the stretch
    cx, cy, bw, bh = b[:, 0] * w, b[:, 1] * h, b[:, 2] * w, b[:, 3] * h
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

    return sv.Detections(
        xyxy=xyxy.astype(np.float32),
        confidence=best_score[keep].astype(np.float32),
        class_id=cls,
    )
