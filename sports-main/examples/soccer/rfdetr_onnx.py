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
import ctypes
import glob
import os
import site
from pathlib import Path
from typing import List, Optional

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
_nvidia_libs_ready = False


def _nvidia_wheel_lib_dirs() -> List[str]:
    """Locate pip-installed NVIDIA CUDA wheel lib directories.

    onnxruntime-gpu's CUDA EP needs libcudnn_adv.so.9 / libcublas etc. Those
    ship inside `nvidia-cudnn-cu12` / `nvidia-cublas-cu12` site-packages, but
    the dynamic linker does not search there unless LD_LIBRARY_PATH (or an
    explicit preload) points at them — otherwise ORT silently falls back to CPU.
    """
    dirs: List[str] = []
    roots: List[Path] = []

    try:
        import nvidia  # type: ignore
        for p in getattr(nvidia, '__path__', []):
            roots.append(Path(p))
    except Exception:
        pass

    for sp in list(site.getsitepackages()) + [site.getusersitepackages()]:
        if sp:
            roots.append(Path(sp) / 'nvidia')

    # Deduplicate while preserving order.
    seen = set()
    uniq_roots = []
    for r in roots:
        key = str(r.resolve()) if r.exists() else str(r)
        if key not in seen:
            seen.add(key)
            uniq_roots.append(r)

    subpkgs = (
        'cudnn', 'cublas', 'cuda_runtime', 'cufft', 'curand',
        'cusolver', 'cusparse', 'nccl', 'nvjitlink', 'cuda_nvrtc',
    )
    for root in uniq_roots:
        for name in subpkgs:
            lib = root / name / 'lib'
            if lib.is_dir():
                dirs.append(str(lib.resolve()))

    # Preserve order, drop dupes.
    out, seen2 = [], set()
    for d in dirs:
        if d not in seen2:
            seen2.add(d)
            out.append(d)
    return out


def ensure_nvidia_cuda_libs() -> List[str]:
    """Inject NVIDIA wheel lib paths into LD_LIBRARY_PATH and preload .so files.

    Must run BEFORE `import onnxruntime` / InferenceSession so CUDAExecutionProvider
    can resolve libcudnn_adv.so.9 on RunPod (RTX 4090) and similar hosts.
    """
    global _nvidia_libs_ready
    if _nvidia_libs_ready:
        return [p for p in os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep) if p]

    lib_dirs = _nvidia_wheel_lib_dirs()
    if lib_dirs:
        existing = [p for p in os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep) if p]
        # Prepend wheel libs so they win over incomplete system CUDA installs.
        merged = []
        for p in lib_dirs + existing:
            if p not in merged:
                merged.append(p)
        os.environ['LD_LIBRARY_PATH'] = os.pathsep.join(merged)
        print(f"  RF-DETR: prepended {len(lib_dirs)} NVIDIA lib dir(s) to LD_LIBRARY_PATH")
        for d in lib_dirs:
            print(f"    {d}")

        # Preload so dlopen succeeds even when the process inherited an empty
        # LD_LIBRARY_PATH at start (common under systemd / RunPod wrappers).
        preload_patterns = (
            'libcudnn_adv.so*',
            'libcudnn_ops.so*',
            'libcudnn_cnn.so*',
            'libcudnn.so*',
            'libcublasLt.so*',
            'libcublas.so*',
            'libcudart.so*',
            'libnvJitLink.so*',
        )
        loaded = 0
        rtld = getattr(ctypes, 'RTLD_GLOBAL', 0)
        for d in lib_dirs:
            # Windows: allow DLL resolution from the wheel lib dir.
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(d)
                except (OSError, FileNotFoundError):
                    pass
            for pat in preload_patterns:
                for so in sorted(glob.glob(os.path.join(d, pat))):
                    # Prefer versioned sonames (e.g. .so.9) over bare .so symlinks.
                    try:
                        ctypes.CDLL(so, mode=rtld)
                        loaded += 1
                    except OSError:
                        continue
        if loaded:
            print(f"  RF-DETR: preloaded {loaded} NVIDIA shared librar(ies) via ctypes")
    else:
        print("  RF-DETR: no pip nvidia/*/lib dirs found — CUDA EP may fall back to CPU "
              "(install nvidia-cudnn-cu12 nvidia-cublas-cu12)")

    _nvidia_libs_ready = True
    return lib_dirs


# Run at import time so any later `import onnxruntime` sees the paths.
ensure_nvidia_cuda_libs()


def session(weights: Optional[str] = None):
    """Load the ONNX session once.

    Prefer CUDA on RunPod / GPU hosts; fall back to CPU if CUDA EP is missing
    (local Mac/CPU boxes). Provider order is explicit so ORT does not silently
    stay on CPU when a GPU is available.
    """
    global _session
    if _session is None:
        # Re-assert paths immediately before ORT import/load (idempotent).
        ensure_nvidia_cuda_libs()
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = os.cpu_count() or 4
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            _session = ort.InferenceSession(weights or WEIGHTS, so,
                                            providers=providers)
        except Exception as e:
            print(f"  RF-DETR: CUDA EP init failed ({type(e).__name__}: {e}); "
                  f"retrying CPU-only")
            _session = ort.InferenceSession(weights or WEIGHTS, so,
                                            providers=['CPUExecutionProvider'])
        active = _session.get_providers()
        print(f"  RF-DETR ONNX providers requested={providers} active={active}")
        if 'CUDAExecutionProvider' not in active:
            print("  WARNING: CUDAExecutionProvider not active — running on CPU. "
                  "Check LD_LIBRARY_PATH / nvidia-cudnn-cu12 on RunPod.")
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
