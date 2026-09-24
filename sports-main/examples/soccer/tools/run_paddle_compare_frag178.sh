#!/usr/bin/env bash
# EasyOCR vs Paddle on fragment 178 (shirt shows 10, Easy reads 1).
# Uses a SEPARATE venv so Paddle does not break numpy/opencv in the tracking env.
set -euo pipefail

ROOT="${ROOT:-/workspace/Second-software}"
SOC="${ROOT}/sports-main/examples/soccer"
VENV="${VENV:-/workspace/venv_jersey_paddle}"
DUMP="${DUMP:-${SOC}/data/id_lists/track_dump_clip10min_deliver_v2.json}"
VIDEO="${VIDEO:-/workspace/clip10min.mp4}"
OUT="${OUT:-${SOC}/data/jersey_ocr_engine178}"

export PYTHONPATH="${ROOT}/sports-main"

if [[ ! -d "${VENV}" ]]; then
  python3.11 -m venv "${VENV}"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  pip install -U pip
  pip install "numpy==1.26.4" opencv-python-headless==4.10.0.84 easyocr
  pip install paddlepaddle-gpu==2.6.2 paddleocr==2.7.3
  export PADDLEOCR_LEGACY=1
else
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

python -c "import cv2, numpy; print('env ok', numpy.__version__, cv2.__version__)"

cd "${SOC}"
export PADDLEOCR_LEGACY="${PADDLEOCR_LEGACY:-1}"

python tools/jersey_ocr_probe.py \
  --dump "${DUMP}" \
  --video "${VIDEO}" \
  --out-dir "${OUT}" \
  --only-fragments 178 \
  --debug-fragment 178 \
  --compare-engines \
  --debug-only

echo ""
echo "See: ${OUT}/engine_compare_frag178.json"
echo "If primary_verdict says switch to paddle:"
echo "  python tools/jersey_ocr_probe.py --dump ... --video ... --ocr-engine paddle --out-dir data/jersey_ocr_v5_paddle"
