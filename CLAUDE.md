# Bosco FC / Sean PoC — AI project context

Use this file as **Claude Project custom instructions** (paste or attach) and keep it in sync with `HANDOVER.md`. Think like a **senior computer vision engineer**: honest metrics, minimal scope, offline pipeline, no fake metre figures.

---

## Client and goal

**Client:** Sean (Bosco FC proof of concept).

**Core ask:** One **stable player identity** per real person across a **~10 minute** panoramic clip (`clip10min`, cut from `Stationary_Camera_14_08.mp4`, 4096×1152, both teams wear **blue** kits).

**Success bar (not yet met):** Identity count toward **~23 roster**, not **~168** merged IDs with ~60 s median duration. **Jersey numbers** are the agreed unlock for fragmentation; kit colour + motion alone plateaus on this footage.

**Constraints:** Fully offline (no Roboflow API in production path). Distances in **pixels** until metre calibration exists. VFR-safe **seconds** in CSVs, not naive frame/fps.

**Delivered to client (Sep 2025):** Google Drive zip `bosco_clip10min_deliver_v2_data` — track dump, positions, players, identity map, player_id_list, ball, events, README counts. Sean praised fragment provenance + physics refusals. **Do not re-run full tracking** unless explicitly comparing a new label.

---

## Footage and canonical run

| Item | Value |
|------|--------|
| Full video | `Stationary_Camera_14_08.mp4` (~6.8G), Drive folder `15-Du0aEfNRqOZjfJXA21NOMrlPeREFnA` |
| 10 min clip | `ffmpeg -ss 477 -t 600` → `/workspace/clip10min.mp4` (~3.8G) |
| Canonical tracking label | **`deliver_v2`** |
| Merged identities | **168**, median duration **~59.7 s** |
| Raw tracklets (dump) | **2264**; player_id_list **2353** rows |
| Physics refusals | **2251** (identity_map) |
| Ball | ~74.5% of integer seconds have ≥1 detection in export |

**Repo:** https://github.com/tayyabriaz60/Second-software  
**Workdir:** `sports-main/examples/soccer`  
**PYTHONPATH:** `sports-main` (parent of `examples/soccer`)

---

## What we tried (experiments chronology)

1. **Detector:** RF-DETR ONNX (Roboflow v20 weights in `external/roboflow_v20/weights.onnx`) ships vs generic YOLO; large recall gain on far players.
2. **Pipeline bugs fixed:** duplicate CSV rows; time from container clock not frame index; GPU ORT; team vote in exports; physics guard on merge (path/net ratio).
3. **Team colour:** CIELAB torso vote in `team_colour.py`. **Both teams blue** → team 0/1 is a **weak hint** on this clip, not ground truth.
4. **Jersey OCR v4+:** Fixed clip-relative frames, dump `h`, centre-torso crop (not grass band). Prior ~28% consistent rates **void** before those fixes.
5. **OCR v5 full decode:** **4472** unique frames → OOM/SIGKILL around **4045/4476** on 4K sequential cache.
6. **OCR batches:** `--max-fragments 90 --max-samples 15` → **1296** frames, completes. **New pod (Sep 29):** part1 seed 42 → **13.3% consistent (MARGINAL)**, 90 fragments.
7. **Digit path:** `build_digit_dataset.py` → ~1350 crops; `_review_1` → **10**; `train_digit_classifier.py` → **180** labelled, **135 class 10** → model predicts **10** only (6/16/28 **0%** all-data). **Not production ready.**
8. **`jersey_digit_probe`:** ffmpeg sequential extract **hangs** mid-clip; restart resumes JPEGs but slow. Needs **sparse extract / less RAM** code fix.
9. **Client message (Sep 2025):** Data package sent; jersey WIP; no promise of ~23 until full-clip probe passes internal gate (~**30%+ consistent** + contact sheet OK).

**Backups (Drive):** client zip; `jersey_backup_newpod.tar.gz` (OCR report, contact sheet, `digit_cls_v3/best.pt`) — **internal**, not client final.

---

## Code map (where things live)

| Area | Path |
|------|------|
| Main tracking | `examples/soccer/main.py` |
| Identity merge | `examples/soccer/assign_identities.py`, `stitch_tracks.py` |
| Team colour | `examples/soccer/team_colour.py` |
| RF-DETR | `examples/soccer/rfdetr_onnx.py`, `external/roboflow_v20/weights.onnx` |
| Jersey OCR probe | `tools/jersey_ocr_probe.py` |
| Digit dataset / train / probe | `tools/build_digit_dataset.py`, `train_digit_classifier.py`, `jersey_digit_probe.py` |
| Outputs | `data/id_lists/*_{clip}_{run_label}.*` |
| Human doc | `HANDOVER.md` (source of truth for commands) |

**Jersey tools do not change merger until** jersey anchors are explicitly added to `assign_identities.py` and validated before/after on `clip10min`.

---

## Operating rules (RunPod / engineering)

1. **Never** full OCR on all eligible fragments without batching on 4K clip (RAM).
2. Prefer: `--max-fragments 45–90`, `--max-samples 10–15`, multiple seeds, merge JSON reports.
3. One heavy job at a time (track **or** OCR **or** digit export).
4. Pod setup: numpy 1.26 + opencv; onnxruntime-gpu; `gdown` Drive folder; clone repo under `/workspace/Second-software`.
5. After bad Paddle pip in same env: force-reinstall numpy/opencv per HANDOVER.
6. Commits only when user asks. No secrets in git.

---

## Remaining work (priority)

1. **Code:** digit probe sparse ffmpeg; balanced digit training; OCR report merge script; jersey anchors in `assign_identities.py`.
2. **RunPod:** OCR part2 (`--seed 99`) + merge; rebuild digit dataset; train; probe vs OCR %.
3. **Gate:** full-clip or merged-batch probe **≥30% consistent**, contact sheet passes eyeball.
4. **Demo:** same clip identity count **before/after** jersey merge → Sean second drop.
5. **Not milestone 2 yet:** metre calibration; shade-aware team_colour rebuild; production Roboflow API path.

---

## Timeline (honest, for team lead)

| Milestone | ETA (focused dev + GPU) |
|-----------|-------------------------|
| Jersey probe package + before/after count on same clip | **~1–2 weeks** if OCR/digit improve |
| ~23 roster-level identities | **Only after gate passes**; often **2–4 weeks** total; do not date-commit at 13% OCR |
| Speedups | Parallel OCR batches, 64GB+ RAM pod, code fixes above; **cannot** skip quality gate |

---

## How to answer the client

- Be split: **tracking delivered** vs **jersey experimental**.
- Cite **168 / 2264 / median ~60 s / physics refusals** from `deliver_v2`.
- Jersey: numbers reduce **identity fragmentation**; **team** on dual-blue needs roster or separate fix.
- Unreadable numbers: fallback motion + physics; no read → mostly stay separate.
- No em-dashes in client email; no overpromising ~23.

---

## When stuck

Read `HANDOVER.md` Jersey OCR section. Verify crops with `--export-crops-only --debug-fragment N` before trusting OCR %. Compare Paddle vs EasyOCR on fragment **178** if two-digit / trailing-zero issues.
