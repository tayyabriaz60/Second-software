#!/usr/bin/env python3
"""Feasibility probe: can jersey digits be read on large enough fragments?

Uses a track_dump JSON only — does not modify main.py or assign_identities.py.

Usage (RunPod, from sports-main/examples/soccer):
  pip install easyocr opencv-python-headless
  # Paddle: use a SEPARATE venv (see HANDOVER.md) — do not pip paddle into the
  # tracking env; it breaks cv2/numpy. Pins: paddlepaddle-gpu==2.6.2 paddleocr==2.7.3
  # PADDLEOCR_LEGACY=1 if 3.x paddlex fails on set_optimization_level.
  python tools/jersey_ocr_probe.py \\
    --dump data/id_lists/track_dump_clip10min_deliver_v2.json \\
    --video /workspace/clip10min.mp4 \\
    --out-dir data/jersey_ocr_probe

Video is read sequentially (grab/read forward). Do not use per-frame seek on
variable-frame-rate sources.

Pass: >=30%% of probed fragments get a consistent number (3+ agreeing reads,
majority >=60%% of successful reads) and the contact sheet looks like real
jersey digits. Below 10%% -> stop.

Quick runs (avoid decoding the full clip for all eligible fragments):
  python tools/jersey_ocr_probe.py --dump ... --video ... --feasibility
  python tools/jersey_ocr_probe.py --dump ... --eligible-only   # no video/OCR
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# Fallback when the dump has no class map (RF-DETR / main.py defaults:
# ball=0, goalkeeper=1, player=2, referee=3).
FALLBACK_REFEREE_CLASS_ID = 3

_HEIGHT_KEYS = (
    'box_height_px', 'box_heights', 'box_h', 'h', 'heights',
    'height_px', 'height',
)


def _spread_indices(idxs: list[int], k: int) -> list[int]:
    if len(idxs) <= k:
        return idxs[:]
    if k <= 1:
        return [idxs[len(idxs) // 2]]
    out = []
    for j in range(k):
        pos = int(round(j * (len(idxs) - 1) / (k - 1)))
        out.append(idxs[pos])
    return out


# Dump ``xy`` is the detector box centre (pass-1 ``_centre(xyxy)``), not the feet.
# Torso window in units of box height h (see centre_torso_crop).
TORSO_Y_ABOVE = 0.30   # y1 = cy - TORSO_Y_ABOVE * h
TORSO_Y_BELOW = 0.05   # y2 = cy + TORSO_Y_BELOW * h
TORSO_X_HALF = 0.18    # x1/x2 = cx ± TORSO_X_HALF * h


def centre_torso_crop(
    frame: np.ndarray,
    cx: float,
    cy: float,
    h: float,
    width_margin: float = 0.0,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop jersey torso from box centre (cx, cy) and height h."""
    if h <= 0 or frame.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8), (0, 0, 0, 0)
    fh, fw = frame.shape[:2]
    hx = max(8.0, float(h) * TORSO_X_HALF)
    hy_top = float(h) * TORSO_Y_ABOVE
    hy_bot = float(h) * TORSO_Y_BELOW
    x1 = int(round(cx - hx))
    x2 = int(round(cx + hx))
    y1 = int(round(cy - hy_top))
    y2 = int(round(cy + hy_bot))
    if width_margin > 0:
        bw = x2 - x1
        x1 += int(width_margin * bw)
        x2 -= int(width_margin * bw)
    x1 = max(0, min(x1, fw - 1))
    x2 = max(x1 + 1, min(x2, fw))
    y1 = max(0, min(y1, fh - 1))
    y2 = max(y1 + 1, min(y2, fh))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=np.uint8), (x1, y1, x2, y2)
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def upscale(crop: np.ndarray, scale: float) -> np.ndarray:
    if crop.size == 0:
        return crop
    nh = max(8, int(round(crop.shape[0] * scale)))
    nw = max(8, int(round(crop.shape[1] * scale)))
    return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)


def preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """CLAHE contrast on L + mild sharpen for small jersey digits."""
    if bgr.size == 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    l_ch = clahe.apply(l_ch)
    out = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    out = cv2.filter2D(out, -1, kernel)
    return out


def normalize_readtext_row(row) -> tuple[list, str, float]:
    """EasyOCR detail=1: (bbox, text, conf); paragraph=True omits confidence."""
    n = len(row)
    if n == 2:
        bbox, text = row
        return bbox, text, 1.0
    if n == 3:
        bbox, text, conf = row
        return bbox, text, float(conf)
    raise ValueError(f'Unexpected EasyOCR row length {n}: {row!r}')


def normalize_readtext_rows(
        rows: list) -> list[tuple[list, str, float]]:
    return [normalize_readtext_row(r) for r in rows]


def parse_jersey_number(text: str) -> int | None:
    digits = re.sub(r'\D', '', text or '')
    if not digits:
        return None
    n = int(digits[:2]) if len(digits) >= 2 else int(digits[0])
    if 1 <= n <= 99:
        return n
    return None


def prefer_two_digit_candidates(
        cands: list[tuple[int, float, str]]) -> list[tuple[int, float, str]]:
    """When a crop yields both '1' and '10', keep the two-digit read."""
    if len(cands) <= 1:
        return cands
    two = [c for c in cands if len(c[2]) >= 2]
    if not two:
        return cands
    two_nums = {c[0] for c in two}
    kept_one: list[tuple[int, float, str]] = []
    for n, conf, raw in cands:
        if len(raw) >= 2:
            continue
        if any(t // 10 == n and t != n for t in two_nums):
            continue
        kept_one.append((n, conf, raw))
    two.sort(key=lambda x: -x[1])
    kept_one.sort(key=lambda x: -x[1])
    return two + kept_one


def merge_split_digit_boxes(
        raw: list[tuple[list, str, float]]) -> list[tuple[str, float]]:
    """Join left-to-right digit boxes on the same text line (e.g. '1' + '0' -> '10')."""
    cells = []
    for bbox, text, conf in raw:
        digits = re.sub(r'\D', '', text or '')
        if not digits:
            continue
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        cells.append({
            'xmin': min(xs),
            'xmax': max(xs),
            'cy': sum(ys) / len(ys),
            'h': max(max(ys) - min(ys), 1.0),
            'digits': digits,
            'conf': float(conf),
        })
    if not cells:
        return []
    cells.sort(key=lambda c: (round(c['cy'] / c['h']), c['xmin']))
    groups: list[list[dict]] = []
    for c in cells:
        placed = False
        for g in groups:
            gcy = sum(x['cy'] for x in g) / len(g)
            gh = max(x['h'] for x in g)
            if abs(c['cy'] - gcy) > 0.55 * max(c['h'], gh):
                continue
            gap = c['xmin'] - max(x['xmax'] for x in g)
            if gap <= max(0.45 * max(c['h'], gh), 10.0):
                g.append(c)
                placed = True
                break
        if not placed:
            groups.append([c])
    merged: list[tuple[str, float]] = []
    for g in groups:
        g.sort(key=lambda c: c['xmin'])
        text = ''.join(c['digits'] for c in g)
        conf = sum(c['conf'] for c in g) / len(g)
        merged.append((text, conf))
    return merged


def serialize_easyocr_boxes(
        raw: list[tuple[list, str, float]]) -> list[dict]:
    out = []
    for bbox, text, conf in raw:
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        out.append({
            'text': text,
            'confidence': round(float(conf), 4),
            'bbox': [[round(x, 1), round(y, 1)] for x, y in bbox],
            'xmin': round(min(xs), 1),
            'xmax': round(max(xs), 1),
            'cy': round(sum(ys) / len(ys), 1),
        })
    return out


def dedupe_raw_boxes(
        raw: list[tuple[list, str, float]]) -> list[tuple[list, str, float]]:
    seen: set[tuple] = set()
    out = []
    for bbox, text, conf in raw:
        xs = tuple(round(float(p[0])) for p in bbox)
        ys = tuple(round(float(p[1])) for p in bbox)
        key = (text, xs, ys)
        if key in seen:
            continue
        seen.add(key)
        out.append((bbox, text, conf))
    return out


def require_easyocr() -> None:
    try:
        import easyocr  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            'easyocr is not installed in this Python environment.\n'
            '  Main RunPod env: pip install easyocr  (after numpy/opencv repair)\n'
            '  Or in venv: pip install easyocr  (needs ~2GB+ disk for torch)\n'
            'See HANDOVER.md — do not install full Paddle stack if disk is full.'
        ) from exc


class OcrEngine:
    def __init__(
            self,
            gpu: bool,
            paragraph: bool = True,
    ):
        self._reader = None
        self._gpu = gpu
        self.paragraph = paragraph

    def _lazy_init(self):
        if self._reader is not None:
            return
        require_easyocr()
        import easyocr
        self._reader = easyocr.Reader(['en'], gpu=self._gpu, verbose=False)

    def readtext_raw(
            self,
            bgr: np.ndarray,
            *,
            paragraph: bool,
            allowlist: str | None,
    ) -> list[tuple[list, str, float]]:
        self._lazy_init()
        if bgr.size == 0:
            return []
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        kw: dict = dict(paragraph=paragraph, detail=1)
        if allowlist is not None:
            kw['allowlist'] = allowlist
        return normalize_readtext_rows(self._reader.readtext(rgb, **kw))

    def _raw_allowlist(self, bgr: np.ndarray) -> list[tuple[list, str, float]]:
        combined: list[tuple[list, str, float]] = []
        combined.extend(
            self.readtext_raw(bgr, paragraph=False, allowlist='0123456789'))
        if self.paragraph:
            combined.extend(
                self.readtext_raw(bgr, paragraph=True, allowlist='0123456789'))
        return dedupe_raw_boxes(combined)

    def collect_raw_for_merge(self, bgr: np.ndarray) -> list[tuple[list, str, float]]:
        """Allowlist reads (paragraph False + optional True); dedupe before merge."""
        return self._raw_allowlist(bgr)

    def read_digits(self, bgr: np.ndarray) -> list[tuple[int, float, str]]:
        """Returns (jersey number, confidence, raw merged digit string)."""
        if bgr.size == 0:
            return []
        raw = self._raw_allowlist(bgr)
        cands: list[tuple[int, float, str]] = []
        for merged_text, conf in merge_split_digit_boxes(raw):
            num = parse_jersey_number(merged_text)
            if num is None:
                continue
            cands.append((num, float(conf), merged_text))
        return prefer_two_digit_candidates(cands)

    def debug_all_modes(self, bgr: np.ndarray) -> dict[str, list[dict]]:
        modes = {
            'paragraph_false_allowlist': (False, '0123456789'),
            'paragraph_true_allowlist': (True, '0123456789'),
            'paragraph_false_no_allowlist': (False, None),
            'paragraph_true_no_allowlist': (True, None),
        }
        report = {}
        for name, (para, allow) in modes.items():
            raw = self.readtext_raw(bgr, paragraph=para, allowlist=allow)
            report[name] = serialize_easyocr_boxes(raw)
        return report


def serialize_det_boxes(
        raw: list[tuple[list, str, float]]) -> list[dict]:
    """Serialize EasyOCR / PaddleOCR detection boxes for JSON debug."""
    out = []
    for bbox, text, conf in raw:
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        out.append({
            'text': text,
            'confidence': round(float(conf), 4),
            'bbox': [[round(x, 1), round(y, 1)] for x, y in bbox],
            'xmin': round(min(xs), 1),
            'xmax': round(max(xs), 1),
            'cy': round(sum(ys) / len(ys), 1),
        })
    return out


def _parse_paddle_v3_result(result) -> list[tuple[list, str, float]]:
    """PaddleOCR 3.x predict() -> [(box, text, conf), ...]."""
    out: list[tuple[list, str, float]] = []

    def _consume_page(page) -> None:
        if page is None:
            return
        data = page
        if hasattr(page, 'json'):
            j = page.json
            data = j() if callable(j) else j
        elif hasattr(page, 'res'):
            data = page.res
        if isinstance(data, dict) and 'res' in data:
            data = data['res']
        if not isinstance(data, dict):
            return
        texts = data.get('rec_texts') or data.get('texts') or []
        scores = data.get('rec_scores') or data.get('scores') or []
        boxes = data.get('rec_boxes') or data.get('dt_polys') or data.get('boxes')
        if texts and boxes is not None:
            for i, text in enumerate(texts):
                if not text or not str(text).strip():
                    continue
                conf = float(scores[i]) if i < len(scores) else 1.0
                box = boxes[i]
                if hasattr(box, 'tolist'):
                    box = box.tolist()
                out.append((box, str(text), conf))
            return
        # Single-line recognition-only payload
        if data.get('rec_text'):
            out.append(([[0, 0], [1, 0], [1, 1], [0, 1]],
                        str(data['rec_text']),
                        float(data.get('rec_score') or 1.0)))

    if isinstance(result, list):
        for page in result:
            _consume_page(page)
    else:
        _consume_page(result)
    return out


def _paddle_install_hint() -> str:
    return (
        'PaddleOCR init failed (often paddlepaddle vs paddlex mismatch). On the pod try:\n'
        '  pip uninstall -y paddleocr paddlex paddlepaddle paddlepaddle-gpu\n'
        '  pip install paddlepaddle-gpu==2.6.2 paddleocr==2.7.3\n'
        'Or upgrade paddle to match paddlex: pip install -U paddlepaddle-gpu\n'
        'Then re-run with env PADDLEOCR_LEGACY=1 to skip PaddleOCR 3.x pipeline.'
    )


class PaddleOcrEngine:
    def __init__(self, gpu: bool):
        self._ocr = None
        self._gpu = gpu
        self._api: str = 'v3'
        self._mode_label: str = ''

    def _lazy_init(self):
        if self._ocr is not None:
            return
        import os
        from paddleocr import PaddleOCR

        device = 'gpu' if self._gpu else 'cpu'
        errors: list[str] = []
        force_legacy = os.environ.get('PADDLEOCR_LEGACY', '').lower() in (
            '1', 'true', 'yes')

        def _try_v3():
            return PaddleOCR(
                lang='en',
                device=device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

        def _try_legacy():
            return PaddleOCR(
                use_angle_cls=False,
                lang='en',
                use_gpu=self._gpu,
            )

        def _try_rec_only():
            from paddleocr import TextRecognition
            try:
                return TextRecognition(device=device)
            except TypeError:
                return TextRecognition()

        if not force_legacy:
            try:
                self._ocr = _try_v3()
                self._api = 'v3'
                self._mode_label = 'PaddleOCR 3.x pipeline'
                print(f'PaddleOCR: {self._mode_label}')
                return
            except Exception as exc:
                errors.append(f'3.x pipeline: {exc!r}')

        try:
            self._ocr = _try_legacy()
            self._api = 'legacy'
            self._mode_label = 'PaddleOCR 2.x (det+rec)'
            print(f'PaddleOCR: {self._mode_label}')
            return
        except Exception as exc:
            errors.append(f'2.x det+rec: {exc!r}')

        try:
            self._ocr = _try_rec_only()
            self._api = 'rec_only'
            self._mode_label = 'PaddleOCR TextRecognition (rec only, no det)'
            print(f'PaddleOCR: {self._mode_label}')
            return
        except Exception as exc:
            errors.append(f'TextRecognition: {exc!r}')

        raise RuntimeError(
            _paddle_install_hint() + '\n' + '\n'.join(errors))

    def readtext_raw(self, bgr: np.ndarray) -> list[tuple[list, str, float]]:
        if bgr.size == 0:
            return []
        self._lazy_init()
        fh, fw = bgr.shape[:2]
        full_box = [[0, 0], [fw, 0], [fw, fh], [0, fh]]

        if self._api == 'rec_only':
            predict = self._ocr.predict
            try:
                result = predict(bgr)
            except TypeError:
                result = predict(input=bgr)
            parsed = _parse_paddle_v3_result(result)
            if not parsed and isinstance(result, list):
                for item in result:
                    if hasattr(item, 'rec_text'):
                        parsed.append((
                            full_box, str(item.rec_text),
                            float(getattr(item, 'rec_score', 1.0))))
            if parsed:
                return parsed
            return []

        if self._api == 'v3':
            predict = self._ocr.predict
            try:
                result = predict(bgr)
            except TypeError:
                result = predict(input=bgr)
            parsed = _parse_paddle_v3_result(result)
            if parsed:
                return parsed
            # Some builds still expose ocr() with 2.x-shaped output
            if hasattr(self._ocr, 'ocr'):
                pages = self._ocr.ocr(bgr, cls=False)
                if pages:
                    for page in pages:
                        if not page:
                            continue
                        for box, (text, conf) in page:
                            parsed.append((box, text, float(conf)))
            return parsed
        pages = self._ocr.ocr(bgr, cls=False)
        if not pages:
            return []
        out: list[tuple[list, str, float]] = []
        for page in pages:
            if not page:
                continue
            for box, (text, conf) in page:
                out.append((box, text, float(conf)))
        return out

    def read_digits(self, bgr: np.ndarray) -> list[tuple[int, float, str]]:
        raw = self.readtext_raw(bgr)
        cands: list[tuple[int, float, str]] = []
        for merged_text, conf in merge_split_digit_boxes(raw):
            num = parse_jersey_number(merged_text)
            if num is None:
                continue
            cands.append((num, float(conf), merged_text))
        for _bbox, text, conf in raw:
            num = parse_jersey_number(text)
            if num is not None:
                cands.append((num, float(conf), re.sub(r'\D', '', text or '')))
        seen: set[tuple[int, str]] = set()
        deduped: list[tuple[int, float, str]] = []
        for num, c, raw_s in sorted(cands, key=lambda x: -x[1]):
            key = (num, raw_s)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((num, c, raw_s))
        return prefer_two_digit_candidates(deduped)


def export_fragment_crops(
        fragment_id: int,
        meta: dict,
        tr: dict,
        frame_cache: dict[int, np.ndarray],
        args,
        out_dir: Path,
) -> Path:
    """Write upscaled (and optional preprocessed) torso crops — no OCR."""
    frames = meta['frames']
    hs = meta['heights']
    crops_dir = out_dir / f'crops_frag{fragment_id}'
    crops_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for si in meta['sample_idx']:
        fnum = int(frames[si])
        frame = frame_cache.get(fnum)
        if frame is None:
            continue
        cx, cy, h = sample_centre_and_height(tr, si, hs)
        crop, (x1, y1, x2, y2) = centre_torso_crop(
            frame, cx, cy, h, width_margin=args.torso_width_margin)
        up = upscale(crop, args.upscale)
        tag = f'frag{fragment_id}_f{fnum}'
        cv2.imwrite(str(crops_dir / f'{tag}_up.jpg'), up)
        if args.ocr_preprocess:
            cv2.imwrite(str(crops_dir / f'{tag}_pre.jpg'), preprocess_for_ocr(up))
        n += 1
    print(f'Wrote {n} crop(s) to {crops_dir}/')
    return crops_dir


def run_fragment_ocr_debug(
        fragment_id: int,
        meta: dict,
        tr: dict,
        frame_cache: dict[int, np.ndarray],
        ocr: OcrEngine,
        args,
        out_dir: Path,
        paddle: PaddleOcrEngine | None = None,
) -> None:
    """Print and write pre-merge EasyOCR boxes for one fragment."""
    frames = meta['frames']
    hs = meta['heights']
    samples_out = []
    print(f'\n=== OCR debug fragment {fragment_id} ===')
    print(f'  crop: centre (cx,cy)+h — y=[cy-{TORSO_Y_ABOVE}h, cy+{TORSO_Y_BELOW}h], '
          f'x=±{TORSO_X_HALF}h (dump xy is box centre, not feet)')
    print(f'  torso_width_margin={args.torso_width_margin} upscale={args.upscale}')
    print(f'  ocr_preprocess={args.ocr_preprocess} (debug w/ Paddle always logs raw+pre)')
    crops_dir = out_dir / f'crops_frag{fragment_id}'
    crops_dir.mkdir(parents=True, exist_ok=True)
    print(f'  Saving upscaled crops -> {crops_dir}/')

    for si in meta['sample_idx']:
        fnum = int(frames[si])
        frame = frame_cache.get(fnum)
        if frame is None:
            continue
        cx, cy, h = sample_centre_and_height(tr, si, hs)
        crop, (x1, y1, x2, y2) = centre_torso_crop(
            frame, cx, cy, h, width_margin=args.torso_width_margin)
        up = upscale(crop, args.upscale)
        debug_dual = paddle is not None
        up_pp = preprocess_for_ocr(up) if (args.ocr_preprocess or debug_dual) else up
        crop_info = {
            'frame': fnum,
            'centre_xy': [round(cx, 1), round(cy, 1)],
            'box_h': round(h, 1),
            'torso_xyxy': [x1, y1, x2, y2],
            'torso_px': [int(crop.shape[1]), int(crop.shape[0])],
            'upscaled_px': [int(up.shape[1]), int(up.shape[0])],
            'width_margin': args.torso_width_margin,
        }
        tag = f'frag{fragment_id}_f{fnum}'
        cv2.imwrite(str(crops_dir / f'{tag}_up.jpg'), up)
        if args.ocr_preprocess or debug_dual:
            cv2.imwrite(str(crops_dir / f'{tag}_pre.jpg'), up_pp)

        slim = getattr(args, 'compare_engines', False)

        def _easy_report(label: str, img: np.ndarray) -> dict:
            raw_allow = ocr._raw_allowlist(img)
            merged = merge_split_digit_boxes(raw_allow)
            reads = ocr.read_digits(img)
            print(f'\n  frame {fnum} [{label}] {crop_info["upscaled_px"]}')
            if not slim:
                modes = ocr.debug_all_modes(img)
                for mode_name, boxes in modes.items():
                    print(f'    [easyocr {mode_name}] {len(boxes)} box(es)')
                    for b in boxes:
                        print(f"      text={b['text']!r} conf={b['confidence']} "
                              f"x=[{b['xmin']},{b['xmax']}]")
            else:
                modes = {}
                for _bbox, text, conf in raw_allow:
                    print(f'      easyocr box text={text!r} conf={round(float(conf), 4)}')
            print(f'    [easyocr allowlist merge] {merged!r} read_digits={reads!r}')
            return {
                'easyocr_modes': modes,
                'allowlist_pre_merge': serialize_det_boxes(raw_allow),
                'merged_allowlist': [{'digits': t, 'confidence': c} for t, c in merged],
                'read_digits': [
                    {'number': n, 'confidence': c, 'raw_digits': raw}
                    for n, c, raw in reads
                ],
            }

        easy_raw = _easy_report('raw upscale', up)
        easy_pp = _easy_report('preprocessed', up_pp) if debug_dual else None

        paddle_raw = paddle_pp = None
        if paddle is not None:
            for label, img in (
                ('raw', up),
                ('preprocessed', up_pp),
            ):
                if label == 'preprocessed' and not debug_dual:
                    continue
                praw = paddle.readtext_raw(img)
                preads = paddle.read_digits(img)
                print(f'    [paddle {label}] {len(praw)} box(es) reads={preads!r}')
                for _bbox, text, conf in praw:
                    print(f'      text={text!r} conf={round(float(conf), 4)}')
                block = {
                    'boxes': serialize_det_boxes(praw),
                    'read_digits': [
                        {'number': n, 'confidence': c, 'raw_digits': raw}
                        for n, c, raw in preads
                    ],
                }
                if label == 'raw':
                    paddle_raw = block
                else:
                    paddle_pp = block

        samples_out.append({
            **crop_info,
            'crop_files': {
                'upscaled': str(crops_dir / f'{tag}_up.jpg'),
                'preprocessed': (
                    str(crops_dir / f'{tag}_pre.jpg')
                    if (args.ocr_preprocess or debug_dual) else None
                ),
            },
            'easyocr_raw': easy_raw,
            'easyocr_preprocessed': easy_pp,
            'paddle_raw': paddle_raw,
            'paddle_preprocessed': paddle_pp,
        })

    out_path = out_dir / f'debug_fragment_{fragment_id}.json'
    out_path.write_text(
        json.dumps(json_safe({
            'fragment_id': fragment_id,
            'crop_settings': {
                'xy_is_box_centre': True,
                'torso_y_above_h': TORSO_Y_ABOVE,
                'torso_y_below_h': TORSO_Y_BELOW,
                'torso_x_half_h': TORSO_X_HALF,
                'torso_width_margin': args.torso_width_margin,
                'upscale': args.upscale,
            },
            'samples': samples_out,
        }), indent=2),
        encoding='utf-8')
    print(f'\n  Wrote {out_path}')
    if paddle is not None:
        cmp = print_engine_compare_table(
            fragment_id, samples_out,
            include_preprocessed=debug_dual)
        cmp_path = out_dir / f'engine_compare_frag{fragment_id}.json'
        cmp_path.write_text(json.dumps(json_safe(cmp), indent=2), encoding='utf-8')
        print(f'  Wrote {cmp_path}')


def majority_stats(reads: list[tuple[int, float]]) -> tuple[int | None, int, float]:
    """Majority number, agreeing count, mean confidence of that number."""
    if not reads:
        return None, 0, 0.0
    c = Counter(n for n, _ in reads)
    top, count = c.most_common(1)[0]
    confs = [conf for n, conf in reads if n == top]
    return top, count, float(sum(confs) / len(confs))


def is_consistent(reads: list[tuple[int, float]]) -> tuple[bool, int | None, int, int]:
    if not reads:
        return False, None, 0, 0
    maj, count, _ = majority_stats(reads)
    n = len(reads)
    ok = count >= 3 and (count / n) >= 0.60
    return ok, maj, count, n


def _scale_heights_to_px(vals: list[float], frame_h: float) -> list[float]:
    if not vals or frame_h <= 0:
        return vals
    mx = max(vals)
    if mx <= 0:
        return vals
    # Normalised fraction of frame height (0..1).
    if mx <= 1.0:
        return [v * frame_h for v in vals]
    return vals


def _infer_imgsz_height(dump: dict) -> float | None:
    raw = dump.get('inference_imgsz') or dump.get('infer_imgsz')
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and raw:
        return float(raw[0])
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _scale_heights_infer_to_video(
        vals: list[float], frame_h: float, dump: dict) -> list[float]:
    """Heights from 576-tall inference space -> native video height."""
    vals = _scale_heights_to_px(vals, frame_h)
    if not vals or frame_h <= 0:
        return vals
    ih = _infer_imgsz_height(dump)
    if ih is None or ih <= 0 or ih >= frame_h:
        return vals
    mx = max(vals)
    # Native near-touchline boxes often exceed ~250px; infer-space tops out ~140.
    if mx > 0 and mx < frame_h * 0.22:
        scale = frame_h / ih
        return [v * scale for v in vals]
    return vals


def _looks_like_cxcywh(p: list) -> bool:
    if len(p) < 4:
        return False
    cx, cy, w, h = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
    if w <= 0 or h <= 0:
        return False
    if w >= cx and h >= cy:
        return False
    if w > 800 or h > 800:
        return False
    return True


def _heights_from_xyxy(tr: dict, n: int) -> list[float] | None:
    xyxy = tr.get('xyxy') or tr.get('boxes') or tr.get('bbox')
    if isinstance(xyxy, list) and len(xyxy) > 0 and n > 0:
        out = []
        for row in xyxy[:n]:
            if not row or len(row) < 4:
                out.append(0.0)
            elif _looks_like_cxcywh(row):
                out.append(float(row[3]))
            else:
                out.append(abs(float(row[3]) - float(row[1])))
        out.extend([0.0] * max(0, n - len(out)))
        if max(out, default=0) > 0:
            return out[:n]
    xy = tr.get('xy')
    if isinstance(xy, list) and len(xy) > 0 and n > 0 and len(xy[0]) >= 4:
        out = []
        for p in xy[:n]:
            if _looks_like_cxcywh(p):
                out.append(float(p[3]))
            else:
                out.append(abs(float(p[3]) - float(p[1])))
        out.extend([0.0] * max(0, n - len(out)))
        if max(out, default=0) > 0:
            return out[:n]
    return None


def _heights_from_record_list(tr: dict, n: int) -> list[float] | None:
    for key in ('history', 'records', 'samples'):
        recs = tr.get(key)
        if not isinstance(recs, list) or len(recs) < n:
            continue
        out = []
        for rec in recs[:n]:
            if not isinstance(rec, dict):
                out.append(0.0)
                continue
            h = None
            for hk in _HEIGHT_KEYS:
                if rec.get(hk) not in (None, ''):
                    h = float(rec[hk])
                    break
            if h is None and rec.get('xyxy') and len(rec['xyxy']) >= 4:
                h = abs(float(rec['xyxy'][3]) - float(rec['xyxy'][1]))
            out.append(float(h or 0.0))
        if max(out, default=0) > 0:
            return out
    return None


def _heights_from_list_field(tr: dict, n: int) -> list[float] | None:
    if n <= 0:
        return None
    for key in _HEIGHT_KEYS:
        raw = tr.get(key)
        if not isinstance(raw, list) or not raw:
            continue
        out = []
        for v in raw[:n]:
            try:
                out.append(float(v) if v not in (None, '') else 0.0)
            except (TypeError, ValueError):
                out.append(0.0)
        out.extend([0.0] * max(0, n - len(out)))
        if max(out, default=0) > 0:
            return out[:n]
    wh = tr.get('wh')
    if isinstance(wh, list) and len(wh) >= n:
        out = []
        for row in wh[:n]:
            if row and len(row) >= 2:
                out.append(float(row[1]))
            else:
                out.append(0.0)
        if max(out, default=0) > 0:
            return out
    return None


def _measured_heights(
        tr: dict, n: int, frame_h: float, dump: dict) -> list[float] | None:
    candidates: list[list[float]] = []
    for fn in (_heights_from_list_field, _heights_from_xyxy,
               _heights_from_record_list):
        hs = fn(tr, n)
        if hs is None:
            continue
        hs = _scale_heights_infer_to_video(hs, frame_h, dump)
        if max(hs, default=0) > 0:
            candidates.append(hs)
    if not candidates:
        return None
    return max(candidates, key=lambda h: max(h))


def referee_class_from_dump(dump: dict) -> int | None:
    if dump.get('referee_class_id') is not None:
        return int(dump['referee_class_id'])
    for key in ('class_ids', 'class_map', 'detector_classes', 'class_names'):
        raw = dump.get(key)
        if not isinstance(raw, dict) or not raw:
            continue
        # Name -> id, e.g. {"referee": 3, "player": 2}
        by_name = {str(k).strip().lower(): v for k, v in raw.items()}
        if 'referee' in by_name:
            return int(by_name['referee'])
        # Id -> name, e.g. {"0": "ball", "3": "referee"}
        for k, v in raw.items():
            if str(v).strip().lower() == 'referee':
                return int(k)
    return None


def resolve_referee_class_id(dump: dict, cli_override: int | None) -> int:
    """CLI override wins; else dump class map; else RF-DETR default (3)."""
    if cli_override is not None:
        return int(cli_override)
    from_dump = referee_class_from_dump(dump)
    if from_dump is not None:
        return from_dump
    return FALLBACK_REFEREE_CLASS_ID


def sample_centre_and_height(
        tr: dict, idx: int, hs: list[float]) -> tuple[float, float, float]:
    xy = tr['xy'][idx]
    h = float(hs[idx]) if idx < len(hs) else 0.0
    if len(xy) >= 4:
        p = [float(v) for v in xy[:4]]
        if _looks_like_cxcywh(p):
            cx, cy, bh = p[0], p[1], p[3]
            return cx, cy, bh if bh > 0 else h
        x1, y1, x2, y2 = p
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bh = abs(y2 - y1)
        return cx, cy, bh if bh > 0 else h
    return float(xy[0]), float(xy[1]), h


def fit_y_height_model(tracks: list[dict], frame_h: float, dump: dict):
    """Linear y -> box height from tracks that already have pixel heights."""
    ys, hs = [], []
    for tr in tracks:
        n = len(tr.get('frames') or [])
        if n < 2:
            continue
        hlist = _measured_heights(tr, n, frame_h, dump)
        if hlist is None:
            continue
        xy = tr.get('xy') or []
        for i in range(min(n, len(xy), len(hlist))):
            hf = float(hlist[i])
            if hf <= 0:
                continue
            pt = xy[i]
            if pt and len(pt) >= 2:
                ys.append(float(pt[1]))
                hs.append(hf)
    if len(ys) < 20:
        return None
    a, b = np.polyfit(np.asarray(ys, float), np.asarray(hs, float), 1)
    return lambda y: max(8.0, float(a * y + b))


def per_track_heights(
        tr: dict, frame_h: float, y_model, dump: dict) -> list[float]:
    n = len(tr.get('frames') or [])
    if n == 0:
        return []
    hs = _measured_heights(tr, n, frame_h, dump)
    if hs is not None:
        return hs
    xy = tr.get('xy') or []
    if y_model is not None and len(xy) >= n:
        return [float(y_model(float(xy[i][1]))) for i in range(n)]
    return [0.0] * n


def iter_dump_tracks(dump: dict) -> list[dict]:
    for key in ('tracks', 'tracklets', 'fragments'):
        arr = dump.get(key)
        if isinstance(arr, list) and arr:
            return arr
    return []


def dump_height_diagnostics(dump: dict, tracks: list[dict], frame_h: float) -> dict:
    y_model = fit_y_height_model(tracks, frame_h, dump)
    max_h = 0.0
    n_with = 0
    for tr in tracks:
        hs = per_track_heights(tr, frame_h, y_model, dump)
        if hs and max(hs) > 0:
            n_with += 1
            max_h = max(max_h, max(hs))
    sample = tracks[0] if tracks else {}
    return {
        'n_tracks': len(tracks),
        'tracks_with_height': n_with,
        'global_max_height_px': round(max_h, 1),
        'sample_track_keys': sorted(sample.keys()) if sample else [],
        'used_y_fallback_model': y_model is not None,
        'dump_width': dump.get('width'),
        'dump_height': dump.get('height'),
    }


def eligible_fragments(
    dump: dict,
    min_h: float,
    min_large_frames: int,
    max_samples: int,
    skip_referees: bool,
    referee_class: int | None = None,
) -> tuple[list[dict], dict]:
    frame_h = float(dump.get('height') or 0)
    tracks = iter_dump_tracks(dump)
    ref_cls = resolve_referee_class_id(dump, referee_class)
    y_model = fit_y_height_model(tracks, frame_h, dump)
    eligible = []
    n_referee_tracks_skipped = 0
    n_referee_skipped_height_eligible = 0
    for tr in tracks:
        frames = tr['frames']
        hs = per_track_heights(tr, frame_h, y_model, dump)
        if len(hs) != len(frames):
            hs = (hs + [0.0] * len(frames))[:len(frames)]
        large_idx = [i for i, h in enumerate(hs) if float(h) >= min_h]
        height_ok = len(large_idx) >= min_large_frames
        if skip_referees and int(tr.get('class', 1)) == ref_cls:
            n_referee_tracks_skipped += 1
            if height_ok:
                n_referee_skipped_height_eligible += 1
            continue
        if not height_ok:
            continue
        sample_idx = _spread_indices(large_idx, max_samples)
        eligible.append({
            'id': int(tr['id']),
            'team': tr.get('team'),
            'class': tr.get('class'),
            'n_large': len(large_idx),
            'sample_idx': sample_idx,
            'frames': frames,
            'heights': hs,
        })
    diag = dump_height_diagnostics(dump, tracks, frame_h)
    diag['referee_class_id'] = ref_cls
    diag['referee_tracks_skipped'] = n_referee_tracks_skipped
    diag['referee_skipped_height_eligible'] = n_referee_skipped_height_eligible
    return eligible, diag


def json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def frames_to_decode(eligible: list[dict]) -> set[int]:
    need: set[int] = set()
    for meta in eligible:
        for si in meta['sample_idx']:
            need.add(int(meta['frames'][si]))
    return need


def read_video_frames_sequential(
    video_path: Path,
    start_frame: int,
    needed: set[int],
) -> dict[int, np.ndarray]:
    """Forward-only decode — no cap.set(CAP_PROP_POS_FRAMES).

    ``needed`` uses clip-relative indices from the track dump (pass-1
    ``frame_n``). Absolute video indices are ``clip + start_frame``.
    """
    if not needed:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f'Cannot open video: {video_path}')

    needed_clip = {int(f) for f in needed}
    needed_abs = {f + int(start_frame) for f in needed_clip}
    max_abs = max(needed_abs)
    out: dict[int, np.ndarray] = {}
    frame_idx = 0
    try:
        for _ in range(start_frame):
            if not cap.grab():
                return out
            frame_idx += 1

        span = max(1, max_abs - frame_idx)
        log_every = max(500, span // 20)
        while frame_idx <= max_abs:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in needed_abs:
                clip_idx = frame_idx - int(start_frame)
                out[clip_idx] = frame
                if len(out) == len(needed_clip):
                    break
            if log_every and frame_idx % log_every == 0:
                print(f'  decode: video frame {frame_idx}/{max_abs} '
                      f'(clip ~{frame_idx - int(start_frame)}), '
                      f'have {len(out)}/{len(needed_clip)} targets', flush=True)
            frame_idx += 1
    finally:
        cap.release()

    missing = needed_clip - set(out.keys())
    if missing:
        print(f'  WARNING: {len(missing)} clip frame(s) not read '
              f'(eof or short video); sample missing: {sorted(missing)[:8]}')
    return out


def overlay_read_on_crop(crop: np.ndarray, label: str) -> np.ndarray:
    if crop.size == 0:
        return crop
    img = crop.copy()
    h = img.shape[0]
    cv2.rectangle(img, (0, 0), (img.shape[1], min(h, 22)), (0, 0, 0), -1)
    cv2.putText(img, label[:24], (4, min(h, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _thumb_fit_cell(
        crop: np.ndarray, thumb_h: int, thumb_w: int) -> np.ndarray:
    """Resize crop to fit inside (thumb_w x thumb_h), letterboxed."""
    canvas = np.full((thumb_h, thumb_w, 3), 240, dtype=np.uint8)
    if crop.size == 0:
        return canvas
    ch, cw = crop.shape[:2]
    scale = min(thumb_h / max(ch, 1), thumb_w / max(cw, 1))
    tw = max(1, int(round(cw * scale)))
    th = max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
    y0 = (thumb_h - th) // 2
    x0 = (thumb_w - tw) // 2
    canvas[y0:y0 + th, x0:x0 + tw] = resized
    return canvas


def build_contact_sheet(
    entries: list[dict],
    out_path: Path,
    thumb_h: int = 120,
    cols: int = 8,
) -> None:
    if not entries:
        return
    rows = int(math.ceil(len(entries) / cols))
    pad = 8
    label_h = 36
    thumb_w = int(thumb_h * 0.85)
    cell_w = thumb_w + pad * 2
    cell_h = thumb_h + label_h + pad * 2
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 240, dtype=np.uint8)

    for i, ent in enumerate(entries):
        r, c = divmod(i, cols)
        y0 = r * cell_h + pad
        x0 = c * cell_w + pad
        crop = ent['thumb']
        thumb = _thumb_fit_cell(crop, thumb_h, thumb_w)
        sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb
        for li, line in enumerate(ent['label'].split('\n')[:2]):
            cv2.putText(
                sheet, line[:32], (x0, y0 + thumb_h + 14 + li * 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def _best_read_num(reads: list[tuple[int, float, str]]) -> int | None:
    if not reads:
        return None
    return int(reads[0][0])


def _reads_summary(reads: list[tuple[int, float, str]]) -> str:
    if not reads:
        return '—'
    return ','.join(f'{n}({c:.2f})' for n, c, _ in reads[:3])


def _engine_compare_one(
        fragment_id: int,
        samples: list[dict],
        *,
        use_preprocessed: bool,
) -> dict:
    key_easy = 'easyocr_preprocessed' if use_preprocessed else 'easyocr_raw'
    key_pad = 'paddle_preprocessed' if use_preprocessed else 'paddle_raw'
    label = 'preprocessed (CLAHE+sharpen)' if use_preprocessed else 'raw upscale'
    print(f'\n=== Engine compare fragment {fragment_id} — {label} ===')
    print(f'{"frame":>8}  {"easy":>6}  {"paddle":>6}  easy reads          paddle reads')
    rows = []
    n_paddle_10 = n_easy_10 = 0
    n_paddle_any = 0
    for s in samples:
        fnum = s['frame']
        eb = s.get(key_easy) or {}
        pb = s.get(key_pad)
        easy_r = eb.get('read_digits') or []
        pad_r = (pb or {}).get('read_digits') or [] if pb else []
        en = easy_r[0]['number'] if easy_r else None
        pn = pad_r[0]['number'] if pad_r else None
        if pn is not None:
            n_paddle_any += 1
        if en == 10:
            n_easy_10 += 1
        if pn == 10:
            n_paddle_10 += 1
        es = en if en is not None else '—'
        ps = pn if pn is not None else '—'
        er = [(r['number'], r['confidence'], r.get('raw_digits', '')) for r in easy_r]
        pr = [(r['number'], r['confidence'], r.get('raw_digits', '')) for r in pad_r]
        print(f'{fnum:8d}  {es!s:>6}  {ps!s:>6}  {_reads_summary(er)!s:18}  '
              f'{_reads_summary(pr) if pad_r else "—"}')
        rows.append({'frame': fnum, 'easy': en, 'paddle': pn})
    print(f'  Frames with paddle read: {n_paddle_any}/{len(samples)}')
    print(f'  EasyOCR read 10 on {n_easy_10} frame(s); Paddle read 10 on {n_paddle_10} frame(s)')
    return {
        'crop_variant': label,
        'fragment_id': fragment_id,
        'rows': rows,
        'easy_frames_with_10': n_easy_10,
        'paddle_frames_with_10': n_paddle_10,
        'paddle_frames_with_any_read': n_paddle_any,
    }


def _verdict_from_compare(cmp: dict) -> str:
    n_easy_10 = cmp['easy_frames_with_10']
    n_paddle_10 = cmp['paddle_frames_with_10']
    if n_paddle_10 > n_easy_10:
        return ('Paddle reads 10 more often — try --ocr-engine paddle and re-run probe.')
    if n_paddle_10 == 0 and n_easy_10 == 0:
        return ('Neither engine read 10 — off-the-shelf OCR likely insufficient; '
                'train a digit model on these crops.')
    return 'Paddle did not beat EasyOCR on digit 10 — same resolution limit.'


def print_engine_compare_table(
        fragment_id: int,
        samples: list[dict],
        *,
        include_preprocessed: bool,
) -> dict:
    """Side-by-side EasyOCR vs Paddle; always raw upscale, optional preprocess pass."""
    raw_cmp = _engine_compare_one(fragment_id, samples, use_preprocessed=False)
    pre_cmp = None
    if include_preprocessed:
        pre_cmp = _engine_compare_one(fragment_id, samples, use_preprocessed=True)
    print(f'\n  Primary verdict (raw upscale): {_verdict_from_compare(raw_cmp)}')
    if pre_cmp is not None:
        print(f'  Preprocessed pass verdict: {_verdict_from_compare(pre_cmp)}')
    return {
        'fragment_id': fragment_id,
        'raw_upscale': raw_cmp,
        'preprocessed': pre_cmp,
        'primary_verdict': _verdict_from_compare(raw_cmp),
    }


def make_probe_ocr(args) -> OcrEngine | PaddleOcrEngine:
    gpu = not args.cpu
    if args.ocr_engine == 'paddle':
        return PaddleOcrEngine(gpu=gpu)
    return OcrEngine(
        gpu=gpu,
        paragraph=not args.no_ocr_paragraph,
    )


def thumbs_from_report(
        report_frags: list[dict],
        track_by_id: dict,
        frame_cache: dict[int, np.ndarray],
        frame_h: float,
        dump: dict,
        args,
) -> list[dict]:
    """Rebuild _thumb / _overlay rows from a saved report (no OCR)."""
    tracks = list(track_by_id.values())
    y_model = fit_y_height_model(tracks, frame_h, dump)
    out = []
    for f in report_frags:
        tid = int(f['fragment_id'])
        tr = track_by_id.get(tid)
        if tr is None:
            continue
        hs = per_track_heights(tr, frame_h, y_model, dump)
        frames = tr['frames']
        if len(hs) != len(frames):
            hs = (hs + [0.0] * len(frames))[:len(frames)]
        best_crop = None
        best_conf = -1.0
        for ps in f.get('per_sample', []):
            fnum = int(ps['frame'])
            frame = frame_cache.get(fnum)
            if frame is None:
                continue
            try:
                fi = frames.index(fnum)
            except ValueError:
                continue
            cx, cy, h = sample_centre_and_height(tr, fi, hs)
            crop, _ = centre_torso_crop(
                frame, cx, cy, h, width_margin=args.torso_width_margin)
            up = upscale(crop, args.upscale)
            img = (
                preprocess_for_ocr(up) if args.ocr_preprocess else up)
            reads = ps.get('reads') or []
            conf = max((float(r.get('confidence', 0)) for r in reads), default=0.0)
            if conf > best_conf or best_crop is None:
                best_conf = conf
                best_crop = img.copy()
        maj = f.get('majority_number')
        n_reads = int(f.get('n_successful_reads') or 0)
        maj_count = int(f.get('majority_count') or 0)
        overlay = (
            f'#{maj} ({maj_count}/{n_reads})' if maj is not None else 'no read')
        out.append({
            'fragment_id': tid,
            'team': f.get('team'),
            'consistent': bool(f.get('consistent')),
            '_thumb': best_crop,
            '_overlay': overlay,
        })
    return out


def pick_report_fragments_for_contact(
        report_frags: list[dict],
        replay_ids: list[int] | None,
        contact_n: int,
        seed: int,
) -> list[dict]:
    pool = [f for f in report_frags if f.get('per_sample')]
    if not pool:
        return []
    if replay_ids:
        by_id = {int(f['fragment_id']): f for f in pool}
        return [by_id[fid] for fid in replay_ids if fid in by_id]
    rng = random.Random(seed)
    return rng.sample(pool, min(contact_n, len(pool)))


def best_contact_frame_from_report(frag: dict) -> int | None:
    """One frame per fragment for contact rebuild (from saved reads, no OCR)."""
    per = frag.get('per_sample') or []
    if not per:
        return None
    best_f, best_c = None, -1.0
    for ps in per:
        reads = ps.get('reads') or []
        conf = max((float(r.get('confidence', 0)) for r in reads), default=0.0)
        if conf > best_c or best_f is None:
            best_c = conf
            best_f = int(ps['frame'])
    return best_f if best_f is not None else int(per[0]['frame'])


def rebuild_contact_sheet(
        out_path: Path,
        results_with_thumbs: list[dict],
        replay_ids: list[int] | None,
        contact_n: int,
        seed: int,
) -> None:
    """Build contact sheet from in-memory results (same pick logic as main)."""
    contact_pool = [r for r in results_with_thumbs if r.get('_thumb') is not None]
    if not contact_pool:
        print('  rebuild-contact-sheet: no thumbnails')
        return
    if replay_ids:
        by_id = {r['fragment_id']: r for r in contact_pool}
        contact_pick = [by_id[fid] for fid in replay_ids if fid in by_id]
    else:
        rng = random.Random(seed)
        contact_pick = rng.sample(
            contact_pool, min(contact_n, len(contact_pool)))
    contact_entries = []
    for r in contact_pick:
        thumb = overlay_read_on_crop(r['_thumb'], r['_overlay'])
        label = (f"frag {r['fragment_id']} team {r['team']}\n"
                 f"{r['_overlay']} {'OK' if r['consistent'] else 'weak'}")
        contact_entries.append({'thumb': thumb, 'label': label})
    build_contact_sheet(contact_entries, out_path)
    print(f'  Contact sheet: {out_path}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dump', type=Path, required=True,
                    help='track_dump_*deliver_v2*.json')
    ap.add_argument('--video', type=Path, default=None,
                    help='Source video (not needed with --eligible-only)')
    ap.add_argument('--eligible-only', action='store_true',
                    help='Eligibility + height stats from dump only (seconds)')
    ap.add_argument('--feasibility', action='store_true',
                    help='Cap at 40 fragments / contact sheet 40 (default quick run)')
    ap.add_argument('--out-dir', type=Path, default=Path('data/jersey_ocr_probe'))
    ap.add_argument('--min-h', type=float, default=200.0)
    ap.add_argument('--min-large-frames', type=int, default=20)
    ap.add_argument('--max-samples', type=int, default=30)
    ap.add_argument('--upscale', type=float, default=2.5,
                    help='Torso crop upscale (2–3 typical)')
    ap.add_argument('--width-frac', type=float, default=0.45,
                    help=argparse.SUPPRESS)
    ap.add_argument('--torso-width-margin', type=float, default=0.0,
                    help='Optional extra horizontal inset on centre torso crop')
    ap.add_argument('--replay-report', type=Path, default=None,
                    help='Re-probe the same fragment ids as a prior '
                         'jersey_ocr_report.json; contact sheet keeps that order')
    ap.add_argument('--only-fragments', type=str, default=None,
                    help='Comma-separated fragment ids to probe')
    ap.add_argument('--debug-fragment', type=int, default=None,
                    help='Print/write pre-merge EasyOCR boxes for this fragment id')
    ap.add_argument('--debug-only', action='store_true',
                    help='With --debug-fragment, run debug then exit (no full probe)')
    ap.add_argument('--export-crops-only', action='store_true',
                    help='With --debug-fragment: write upscaled crop JPGs only (no OCR)')
    ap.add_argument('--no-ocr-paragraph', action='store_true',
                    help='EasyOCR paragraph=False only (skip paragraph=True pass)')
    ap.add_argument('--ocr-preprocess', action='store_true',
                    help='CLAHE + sharpen before OCR on probe runs (default: raw upscale only)')
    ap.add_argument('--no-paddle-debug', action='store_true',
                    help='With --debug-fragment, skip PaddleOCR comparison')
    ap.add_argument('--compare-engines', action='store_true',
                    help='With --debug-fragment: require Paddle + print side-by-side table')
    ap.add_argument('--rebuild-contact-sheet', action='store_true',
                    help='Rebuild jersey_ocr_contact_sheet.jpg from existing '
                         'jersey_ocr_report.json + video (no full re-OCR)')
    ap.add_argument('--ocr-engine', choices=('easyocr', 'paddle'), default='easyocr',
                    help='OCR backend for probe runs (use paddle if engine compare wins)')
    ap.add_argument('--cpu', action='store_true',
                    help='EasyOCR on CPU (default: GPU)')
    ap.add_argument('--contact-n', type=int, default=40)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-fragments', type=int, default=0,
                    help='Cap probed fragments (0 = all eligible)')
    ap.add_argument('--referee-class', type=int, default=None,
                    help='Referee class id to skip (default: dump class map, '
                         'else 3 for RF-DETR). YOLO panoramic finetune uses 2.')
    ap.add_argument('--include-referees', action='store_true',
                    help='Do not skip referee-class fragments')
    args = ap.parse_args()

    if args.feasibility:
        if not args.max_fragments:
            args.max_fragments = 40
        if args.contact_n == 40:
            args.contact_n = 40

    dump = json.loads(args.dump.read_text(encoding='utf-8'))
    start_frame = int(dump.get('start_frame') or 0)
    eligible, height_diag = eligible_fragments(
        dump, args.min_h, args.min_large_frames, args.max_samples,
        skip_referees=not args.include_referees,
        referee_class=args.referee_class)

    n_eligible_total = len(eligible)
    replay_ids: list[int] | None = None
    if args.only_fragments:
        want_only = {int(x.strip()) for x in args.only_fragments.split(',') if x.strip()}
        eligible = [e for e in eligible if int(e['id']) in want_only]

    if args.replay_report:
        prior = json.loads(args.replay_report.read_text(encoding='utf-8'))
        replay_ids = [int(f['fragment_id']) for f in prior.get('fragments', [])]
        want = set(replay_ids)
        eligible = [e for e in eligible if int(e['id']) in want]
        order = {fid: i for i, fid in enumerate(replay_ids)}
        eligible.sort(key=lambda e: order.get(int(e['id']), 10**9))
        missing = want - {int(e['id']) for e in eligible}
        if missing:
            print(f'  replay-report: {len(missing)} id(s) not eligible now: '
                  f'{sorted(missing)[:20]}...')

    if args.max_fragments and len(eligible) > args.max_fragments and not replay_ids:
        rng = random.Random(args.seed)
        eligible = rng.sample(eligible, args.max_fragments)

    needed_frames = frames_to_decode(eligible)
    print(f'Dump: {args.dump.name}')
    ref_skip = height_diag.get('referee_tracks_skipped', 0)
    ref_skip_ok = height_diag.get('referee_skipped_height_eligible', 0)
    ref_cls = height_diag.get('referee_class_id')
    if n_eligible_total != len(eligible):
        print(f'Eligible fragments (>={args.min_large_frames} frames h>={args.min_h}): '
              f'{n_eligible_total} total, probing {len(eligible)}')
    else:
        print(f'Eligible fragments (>={args.min_large_frames} frames h>={args.min_h}): '
              f'{n_eligible_total}')
    if not args.include_referees:
        print(f'  Referee filter: class_id={ref_cls} — skipped {ref_skip} track(s)'
              f' ({ref_skip_ok} would have met height threshold)')
    print(f'  Height diagnostics: {height_diag}')
    if not eligible and ref_skip_ok and not args.include_referees:
        print('  WARNING: referee filter removed all height-eligible tracks — '
              f'check --referee-class (YOLO panoramic finetune: 2, RF-DETR: 3).')
    elif not eligible:
        print('  WARNING: 0 eligible — check dump has h/box_height_px/xyxy or '
              'centre-y with measured heights on other tracks.')
    print(f'Unique frames to decode: {len(needed_frames)}')
    if needed_frames:
        cmin, cmax = min(needed_frames), max(needed_frames)
        print(f'  Clip frame range: {cmin}..{cmax} '
              f'(video absolute {cmin + start_frame}..{cmax + start_frame}, '
              f'dump start_frame={start_frame})')
    print(f'Crop: centre torso y±[{TORSO_Y_ABOVE},{TORSO_Y_BELOW}]h x±{TORSO_X_HALF}h, '
          f'margin={args.torso_width_margin}, upscale={args.upscale}')
    print('  Tip: --export-crops-only --debug-fragment N to verify shirts before OCR.')

    if args.eligible_only:
        out = args.out_dir / 'jersey_ocr_eligible_summary.json'
        args.out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            'dump': str(args.dump),
            'eligible_total': n_eligible_total,
            'min_h': args.min_h,
            'min_large_frames': args.min_large_frames,
            'height_diagnostics': height_diag,
        }
        out.write_text(json.dumps(json_safe(summary), indent=2), encoding='utf-8')
        print(f'Wrote {out} (--eligible-only, no video/OCR)')
        return

    if args.video is None:
        raise SystemExit('--video is required unless --eligible-only')

    track_by_id = {int(tr['id']): tr for tr in iter_dump_tracks(dump)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_h = float(dump.get('height') or 0)

    if args.rebuild_contact_sheet:
        report_path = args.out_dir / 'jersey_ocr_report.json'
        if not report_path.is_file():
            raise SystemExit(f'--rebuild-contact-sheet needs {report_path}')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        all_frags = report.get('fragments') or []
        picked = pick_report_fragments_for_contact(
            all_frags, replay_ids, args.contact_n, args.seed)
        need: set[int] = set()
        for f in picked:
            bf = best_contact_frame_from_report(f)
            if bf is not None:
                need.add(bf)
        print(f'Rebuild contact sheet: {len(picked)} tile(s) from report '
              f'({len(all_frags)} fragments total), decode {len(need)} frame(s)')
        if need:
            cmin, cmax = min(need), max(need)
            print(f'  Clip frame range for tiles: {cmin}..{cmax}')
        frame_cache = read_video_frames_sequential(
            args.video, start_frame, need)
        thumbs = thumbs_from_report(
            picked, track_by_id, frame_cache, frame_h, dump, args)
        sheet_path = args.out_dir / 'jersey_ocr_contact_sheet.jpg'
        rebuild_contact_sheet(
            sheet_path, thumbs, replay_ids=None, contact_n=args.contact_n,
            seed=args.seed)
        return

    print(f'Sequential read from dump start_frame={start_frame} (no seek)...')

    frame_cache = read_video_frames_sequential(
        args.video, start_frame, needed_frames)
    if needed_frames:
        got = len(needed_frames & set(frame_cache.keys()))
        print(f'Decoded {got}/{len(needed_frames)} target frame(s) for probe')
        if got < len(needed_frames):
            miss = sorted(needed_frames - set(frame_cache.keys()))[:12]
            print(f'  WARNING: missing clip frame(s): {miss}...')

    if args.debug_fragment is not None:
        fid = int(args.debug_fragment)
        meta = next((e for e in eligible if int(e['id']) == fid), None)
        if meta is None:
            raise SystemExit(f'Fragment {fid} not in eligible set '
                             f'(use --only-fragments {fid} if filtered out)')
        if args.export_crops_only:
            export_fragment_crops(
                fid, meta, track_by_id[fid], frame_cache, args, args.out_dir)
            return
        require_easyocr()
        ocr = OcrEngine(
            gpu=not args.cpu,
            paragraph=not args.no_ocr_paragraph,
        )
        paddle_dbg: PaddleOcrEngine | None = None
        want_paddle = args.compare_engines or not args.no_paddle_debug
        if want_paddle:
            try:
                paddle_dbg = PaddleOcrEngine(gpu=not args.cpu)
                paddle_dbg._lazy_init()
            except Exception as exc:
                if args.compare_engines:
                    raise SystemExit(
                        f'--compare-engines requires PaddleOCR: {exc}\n'
                        + _paddle_install_hint()) from exc
                print(f'PaddleOCR not available ({exc}) — '
                      f'install paddlepaddle-gpu paddleocr to compare')
        run_fragment_ocr_debug(
            fid, meta, track_by_id[fid], frame_cache, ocr, args, args.out_dir,
            paddle=paddle_dbg)
        if args.debug_only:
            return

    ocr = make_probe_ocr(args)
    if args.ocr_engine == 'paddle':
        print('OCR engine: PaddleOCR')
    else:
        print('OCR engine: EasyOCR')
    results = []

    for fi, meta in enumerate(eligible):
        if (fi + 1) % 5 == 0 or fi == 0:
            print(f'  OCR fragment {fi + 1}/{len(eligible)}...', flush=True)
        tid = meta['id']
        tr = track_by_id[tid]
        frames = meta['frames']
        xy = tr['xy']
        hs = meta['heights']
        all_reads: list[tuple[int, float]] = []
        per_sample = []
        best_crop = None
        best_conf = -1.0

        for si in meta['sample_idx']:
            fnum = int(frames[si])
            frame = frame_cache.get(fnum)
            if frame is None:
                continue
            cx, cy, h = sample_centre_and_height(tr, si, hs)
            crop, _box = centre_torso_crop(
                frame, cx, cy, h, width_margin=args.torso_width_margin)
            up = upscale(crop, args.upscale)
            ocr_in = (
                preprocess_for_ocr(up) if args.ocr_preprocess else up)
            reads = ocr.read_digits(ocr_in)
            for num, conf, _raw in reads:
                all_reads.append((num, conf))
            per_sample.append({
                'frame': fnum,
                'h': h,
                'reads': [
                    {'number': n, 'confidence': round(c, 3), 'raw_digits': raw}
                    for n, c, raw in reads
                ],
            })
            if reads and max(c for _, c, _ in reads) > best_conf:
                best_conf = max(c for _, c, _ in reads)
                best_crop = up.copy()

        consistent, maj, maj_count, n_reads = is_consistent(all_reads)
        maj_num, _, maj_mean_conf = majority_stats(all_reads)

        results.append({
            'fragment_id': tid,
            'team': meta.get('team'),
            'class': meta.get('class'),
            'n_large_frames': meta['n_large'],
            'n_samples': len(per_sample),
            'n_successful_reads': n_reads,
            'all_reads': [{'number': n, 'confidence': round(c, 3)}
                          for n, c in all_reads],
            'majority_number': maj_num,
            'majority_count': maj_count,
            'majority_mean_confidence': round(maj_mean_conf, 3) if maj_num else None,
            'consistent': consistent,
            'per_sample': per_sample,
            '_thumb': best_crop,
            '_overlay': (
                f"#{maj_num} ({maj_count}/{n_reads})" if maj_num else 'no read'),
        })

    n_probed = len(results)
    n_any = sum(1 for r in results if r['n_successful_reads'] > 0)
    n_cons = sum(1 for r in results if r['consistent'])
    pct_any = 100.0 * n_any / n_probed if n_probed else 0.0
    pct_cons = 100.0 * n_cons / n_probed if n_probed else 0.0

    by_team: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        if r['majority_number'] is None:
            continue
        key = 'team_' + str(r['team']) if r['team'] is not None else 'team_unknown'
        by_team[key][r['majority_number']] += 1

    print('\n=== Jersey OCR probe report ===')
    print(f'  Fragments probed              : {n_probed}')
    print(f'  Any digit read                : {n_any} ({pct_any:.1f}%)')
    print(f'  Consistent (3+ & >=60% agree) : {n_cons} ({pct_cons:.1f}%)')
    print('\n  Majority number distribution by team (fragment counts):')
    for team_key in sorted(by_team.keys()):
        dist = by_team[team_key]
        print(f'    {team_key}: {len(dist)} distinct numbers, top={dist.most_common(12)}')

    if pct_cons >= 30.0:
        verdict = 'PASS (>=30% consistent) — worth a pipeline experiment'
    elif pct_cons < 10.0:
        verdict = 'STOP (<10% consistent) — jersey OCR unlikely to anchor identity here'
    else:
        verdict = f'MARGINAL ({pct_cons:.1f}% consistent) — eyeball contact sheet before continuing'

    print(f'\n  Verdict: {verdict}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / 'jersey_ocr_report.json'
    serializable = [{k: v for k, v in r.items() if not k.startswith('_')}
                    for r in results]
    payload = {
        'dump': str(args.dump),
        'video': str(args.video),
        'params': json_safe(vars(args)),
        'crop_settings': {
            'xy_is_box_centre': True,
            'torso_y_above_h': TORSO_Y_ABOVE,
            'torso_y_below_h': TORSO_Y_BELOW,
            'torso_x_half_h': TORSO_X_HALF,
            'torso_width_margin': args.torso_width_margin,
            'upscale': args.upscale,
        },
        'height_diagnostics': height_diag,
        'summary': {
            'fragments_probed': n_probed,
            'any_read': n_any,
            'consistent': n_cons,
            'pct_any_read': round(pct_any, 2),
            'pct_consistent': round(pct_cons, 2),
            'verdict': verdict,
        },
        'team_number_distribution': {k: dict(v) for k, v in by_team.items()},
        'fragments': serializable,
    }
    report_path.write_text(
        json.dumps(json_safe(payload), indent=2), encoding='utf-8')
    print(f'\n  Report: {report_path}')

    sheet_path = args.out_dir / 'jersey_ocr_contact_sheet.jpg'
    rebuild_contact_sheet(
        sheet_path, results, replay_ids, args.contact_n, args.seed)


if __name__ == '__main__':
    main()
