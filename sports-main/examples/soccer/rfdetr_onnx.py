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
import re
import shutil
import site
import subprocess
from pathlib import Path
from typing import List, Optional, Set, Tuple

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

# Pip package stems for CUDA 12 / 13 wheel stacks (ORT CUDA EP linkage).
_NVIDIA_CU_PACKAGES = (
    'cublas', 'cudnn', 'cuda_runtime', 'cufft', 'curand',
    'cusolver', 'cusparse', 'nccl', 'nvjitlink', 'cuda_nvrtc',
)


def _dedupe(paths: List[str]) -> List[str]:
    out, seen = [], set()
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _system_cuda_lib_dirs() -> List[str]:
    """Host CUDA toolkit lib dirs (in case wheels are incomplete)."""
    dirs: List[str] = []
    patterns = (
        '/usr/local/cuda/lib64',
        '/usr/local/cuda/targets/x86_64-linux/lib',
        '/usr/local/cuda-*/lib64',
        '/usr/local/cuda-*/targets/x86_64-linux/lib',
        '/usr/lib/x86_64-linux-gnu',
    )
    for pat in patterns:
        for match in glob.glob(pat):
            if os.path.isdir(match):
                dirs.append(os.path.realpath(match))
    return dirs


def _nvidia_wheel_lib_dirs() -> List[str]:
    """Locate pip-installed NVIDIA CUDA wheel lib directories (cu12 and cu13)."""
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

    seen = set()
    uniq_roots = []
    for r in roots:
        key = str(r.resolve()) if r.exists() else str(r)
        if key not in seen:
            seen.add(key)
            uniq_roots.append(r)

    for root in uniq_roots:
        if not root.exists():
            continue
        for name in _NVIDIA_CU_PACKAGES:
            lib = root / name / 'lib'
            if lib.is_dir():
                dirs.append(str(lib.resolve()))
        # Also pick up any nested */lib that contains libcublas*.so*
        for lib in root.glob('*/lib'):
            if lib.is_dir():
                dirs.append(str(lib.resolve()))

    # Deep search: find libcublasLt.so.* parents under site-packages (unusual layouts).
    for sp in list(site.getsitepackages()) + [site.getusersitepackages()]:
        if not sp or not os.path.isdir(sp):
            continue
        for so in glob.glob(os.path.join(sp, '**', 'libcublasLt.so*'), recursive=True):
            dirs.append(str(Path(so).resolve().parent))
        for so in glob.glob(os.path.join(sp, '**', 'libcudnn_adv.so*'), recursive=True):
            dirs.append(str(Path(so).resolve().parent))

    return _dedupe(dirs + _system_cuda_lib_dirs())


def _find_soname(lib_dirs: List[str], soname: str) -> Optional[str]:
    for d in lib_dirs:
        exact = os.path.join(d, soname)
        if os.path.isfile(exact):
            return exact
        matches = sorted(glob.glob(os.path.join(d, soname + '*')))
        if matches:
            return matches[0]
    return None


def _list_cublaslt_versions(lib_dirs: List[str]) -> List[str]:
    found: Set[str] = set()
    for d in lib_dirs:
        for so in glob.glob(os.path.join(d, 'libcublasLt.so*')):
            base = os.path.basename(so)
            found.add(base)
    return sorted(found)


def _ort_cuda_provider_path() -> Optional[Path]:
    try:
        import onnxruntime as ort
    except Exception:
        return None
    capi = Path(ort.__file__).resolve().parent / 'capi'
    for name in ('libonnxruntime_providers_cuda.so',
                 'onnxruntime_providers_cuda.dll'):
        p = capi / name
        if p.is_file():
            return p
    matches = list(capi.glob('*providers_cuda*'))
    return matches[0] if matches else None


def _ort_required_cuda_sonames(cuda_provider: Path) -> List[str]:
    """Parse ldd / dumpbin-style deps for the CUDA EP shared library."""
    needed: List[str] = []
    ldd = shutil.which('ldd')
    if ldd and cuda_provider.suffix == '.so':
        try:
            out = subprocess.check_output(
                [ldd, str(cuda_provider)], text=True, stderr=subprocess.STDOUT)
        except (subprocess.CalledProcessError, OSError):
            out = ''
        for line in out.splitlines():
            # "libcublasLt.so.13 => not found" or "=> /path/libcublasLt.so.13"
            m = re.search(r'(lib(?:cublasLt|cublas|cudnn_adv|cudnn|cudart|nvJitLink)\.so(?:\.\d+)*)',
                          line)
            if m:
                needed.append(m.group(1))
    return _dedupe(needed)


def _print_cuda_mismatch_help(lib_dirs: List[str], missing: List[str]) -> None:
    have = _list_cublaslt_versions(lib_dirs)
    need13 = any('.so.13' in m or m.endswith('.13') for m in missing)
    have12 = any('so.12' in h for h in have)
    have13 = any('so.13' in h for h in have)
    print("  RF-DETR: CUDA EP shared-library mismatch — GPU provider will not load.")
    if missing:
        print(f"    missing: {', '.join(missing)}")
    print(f"    found libcublasLt: {', '.join(have) if have else '(none)'}")
    if need13 and have12 and not have13:
        print(
            "    Your onnxruntime-gpu build needs CUDA 13 (libcublasLt.so.13),\n"
            "    but only CUDA 12 wheels are installed (libcublasLt.so.12).\n"
            "    Fix — pick ONE:\n"
            "      A) Install CUDA 13 wheels to match this ORT:\n"
            "           pip install -U nvidia-cublas-cu13 nvidia-cudnn-cu13 \\\n"
            "             nvidia-cuda-runtime-cu13 nvidia-cuda-nvrtc-cu13 \\\n"
            "             nvidia-cufft-cu13 nvidia-curand-cu13 \\\n"
            "             nvidia-cusolver-cu13 nvidia-cusparse-cu13 \\\n"
            "             nvidia-nvjitlink-cu13\n"
            "      B) Or pin ORT to a CUDA 12 build to match existing cu12 wheels:\n"
            "           pip install -U 'onnxruntime-gpu==1.20.1'\n"
            "         then re-run (1.20.1 links libcublasLt.so.12)."
        )
    elif need13 and not have13:
        print(
            "    Install CUDA 13 NVIDIA pip wheels (see A above) or a matching "
            "system CUDA 13 toolkit on LD_LIBRARY_PATH."
        )


def ensure_nvidia_cuda_libs() -> List[str]:
    """Inject NVIDIA wheel lib paths into LD_LIBRARY_PATH and preload .so files.

    Must run BEFORE InferenceSession so CUDAExecutionProvider can resolve
    libcudnn_adv / libcublasLt on RunPod (RTX 4090) and similar hosts.

    Note: LD_LIBRARY_PATH injection cannot fix a CUDA **major** mismatch
    (ORT built for .so.13 while only .so.12 wheels are installed) — see
    `_print_cuda_mismatch_help`.
    """
    global _nvidia_libs_ready
    if _nvidia_libs_ready:
        return [p for p in os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep) if p]

    lib_dirs = _nvidia_wheel_lib_dirs()
    if lib_dirs:
        existing = [p for p in os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep) if p]
        merged = _dedupe(lib_dirs + existing)
        os.environ['LD_LIBRARY_PATH'] = os.pathsep.join(merged)
        print(f"  RF-DETR: prepended {len(lib_dirs)} NVIDIA/CUDA lib dir(s) to LD_LIBRARY_PATH")
        for d in lib_dirs[:12]:
            print(f"    {d}")
        if len(lib_dirs) > 12:
            print(f"    ... +{len(lib_dirs) - 12} more")

        # Prefer CUDA 13 sonames when present, else 12 — preload both families.
        preload_names = (
            'libcudnn_adv.so.9', 'libcudnn_ops.so.9', 'libcudnn_cnn.so.9',
            'libcudnn_adv.so', 'libcudnn_ops.so', 'libcudnn_cnn.so', 'libcudnn.so',
            'libcublasLt.so.13', 'libcublas.so.13',
            'libcublasLt.so.12', 'libcublas.so.12',
            'libcublasLt.so', 'libcublas.so',
            'libcudart.so.13', 'libcudart.so.12', 'libcudart.so',
            'libnvJitLink.so.13', 'libnvJitLink.so.12', 'libnvJitLink.so',
        )
        loaded = 0
        rtld = getattr(ctypes, 'RTLD_GLOBAL', 0)
        for d in lib_dirs:
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(d)
                except (OSError, FileNotFoundError):
                    pass
        for name in preload_names:
            path = _find_soname(lib_dirs, name)
            if not path:
                continue
            try:
                ctypes.CDLL(path, mode=rtld)
                loaded += 1
            except OSError:
                continue
        # Also preload any versioned matches not covered above.
        for d in lib_dirs:
            for pat in ('libcudnn_adv.so*', 'libcublasLt.so*', 'libcudart.so*'):
                for so in sorted(glob.glob(os.path.join(d, pat))):
                    try:
                        ctypes.CDLL(so, mode=rtld)
                        loaded += 1
                    except OSError:
                        continue
        if loaded:
            print(f"  RF-DETR: preloaded {loaded} NVIDIA shared librar(ies) via ctypes")
        print(f"  RF-DETR: libcublasLt available: "
              f"{', '.join(_list_cublaslt_versions(lib_dirs)) or '(none)'}")
    else:
        print("  RF-DETR: no NVIDIA/CUDA lib dirs found — CUDA EP may fall back to CPU")

    _nvidia_libs_ready = True
    return lib_dirs


# Run at import time so any later `import onnxruntime` sees the paths.
ensure_nvidia_cuda_libs()


def _diagnose_ort_cuda(lib_dirs: List[str]) -> Tuple[List[str], List[str]]:
    """Return (required_sonames, missing_sonames) for the installed ORT CUDA EP."""
    provider = _ort_cuda_provider_path()
    if provider is None:
        return [], ['libonnxruntime_providers_cuda.so (package missing)']
    required = _ort_required_cuda_sonames(provider)
    # Always check the critical sonames ORT 1.22+/CUDA13 builds need.
    for critical in ('libcublasLt.so.13', 'libcublasLt.so.12',
                     'libcudnn_adv.so.9', 'libcudart.so.13', 'libcudart.so.12'):
        if critical not in required:
            # Infer from provider linkage failure patterns / filesystem.
            pass
    missing = []
    # Prefer ldd "not found" list; if ldd gave nothing, probe common sonames.
    if required:
        for so in required:
            if _find_soname(lib_dirs, so) is None:
                missing.append(so)
    else:
        # No ldd — probe both majors; report which ORT is likely to need.
        have13 = _find_soname(lib_dirs, 'libcublasLt.so.13')
        have12 = _find_soname(lib_dirs, 'libcublasLt.so.12')
        if not have13 and not have12:
            missing.append('libcublasLt.so.12|13')
    return required, missing


def session(weights: Optional[str] = None):
    """Load the ONNX session once.

    Prefer CUDA on RunPod / GPU hosts; fall back to CPU if CUDA EP is missing
    (local Mac/CPU boxes). Provider order is explicit so ORT does not silently
    stay on CPU when a GPU is available.
    """
    global _session
    if _session is None:
        lib_dirs = ensure_nvidia_cuda_libs()
        import onnxruntime as ort

        required, missing_probe = _diagnose_ort_cuda(lib_dirs)
        has_13 = _find_soname(lib_dirs, 'libcublasLt.so.13') is not None
        has_12 = _find_soname(lib_dirs, 'libcublasLt.so.12') is not None
        ldd_misses_13 = any('cublasLt.so.13' in m for m in missing_probe)
        ldd_needs_13 = any('cublasLt.so.13' in r for r in required)
        if (ldd_misses_13 or ldd_needs_13) and not has_13:
            _print_cuda_mismatch_help(
                lib_dirs,
                missing_probe or required or ['libcublasLt.so.13'],
            )

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
            if 'cublasLt.so.13' in str(e) or 'CUDA 13' in str(e):
                _print_cuda_mismatch_help(lib_dirs, [str(e)])
            _session = ort.InferenceSession(weights or WEIGHTS, so,
                                            providers=['CPUExecutionProvider'])
        active = _session.get_providers()
        print(f"  RF-DETR ONNX providers requested={providers} active={active}")
        print(f"  RF-DETR onnxruntime {ort.__version__}; "
              f"available={ort.get_available_providers()}")
        if 'CUDAExecutionProvider' not in active:
            print("  WARNING: CUDAExecutionProvider not active — running on CPU.")
            if has_12 and not has_13:
                _print_cuda_mismatch_help(lib_dirs, ['libcublasLt.so.13'])
            elif not has_12 and not has_13:
                print("  No libcublasLt.so.12/13 found on LD_LIBRARY_PATH.")
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
