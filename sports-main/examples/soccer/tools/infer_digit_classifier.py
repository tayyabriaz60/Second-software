#!/usr/bin/env python3
"""Run trained digit classifier on crop JPGs; aggregate by fragment (majority vote).

Examples (from examples/soccer):
  python tools/infer_digit_classifier.py \\
    --checkpoint runs/digit_cls_v2/best.pt \\
    --glob 'data/digits/10/*.jpg' --expect 10

  python tools/infer_digit_classifier.py \\
    --checkpoint runs/digit_cls_v2/best.pt \\
    --glob 'data/digits/10/frag178_*.jpg' --expect 10
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

_SOC = Path(__file__).resolve().parents[1]
if str(_SOC) not in sys.path:
    sys.path.insert(0, str(_SOC))

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
except ModuleNotFoundError as exc:
    raise SystemExit('pip install torch torchvision') from exc

from tools.train_digit_classifier import SmallDigitCNN, parse_frag_id


class PathCropDataset(Dataset):
    def __init__(self, paths: list[Path], img_size: int):
        self.paths = paths
        self.img_size = img_size
        self.t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        bgr = cv2.imread(str(p))
        if bgr is None:
            bgr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.t(rgb), str(p)


def load_model(checkpoint: Path, device: torch.device):
    try:
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location='cpu')
    names = ckpt['class_names']
    idx_to_class = {i: c for i, c in enumerate(names)}
    img_size = int(ckpt.get('img_size', 64))
    model = SmallDigitCNN(len(names))
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model, idx_to_class, img_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--glob', type=str, required=True,
                    help="Glob under cwd, e.g. 'data/digits/10/*.jpg'")
    ap.add_argument('--expect', type=str, default=None,
                    help='If set, flag fragments whose vote != this label')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model, idx_to_class, img_size = load_model(args.checkpoint, device)

    paths = sorted(Path('.').glob(args.glob))
    if not paths:
        raise SystemExit(f'No files for glob: {args.glob}')

    ds = PathCropDataset(paths, img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    by_frag: dict[int, list[tuple[str, int, float]]] = defaultdict(list)
    with torch.no_grad():
        for batch, path_strs in loader:
            batch = batch.to(device)
            probs = torch.softmax(model(batch), dim=1)
            conf, pred = probs.max(dim=1)
            for ps, c, p in zip(
                    path_strs, conf.cpu().tolist(), pred.cpu().tolist()):
                fid = parse_frag_id(Path(ps))
                if fid is None:
                    fid = -1
                by_frag[fid].append((ps, p, c))

    print(f'Checkpoint: {args.checkpoint}')
    print(f'Images: {len(paths)}  fragments: {len(by_frag)}')
    n_ok = n_bad = 0
    for fid in sorted(by_frag.keys(), key=lambda x: (x < 0, x)):
        votes = Counter(idx_to_class[p] for _, p, _ in by_frag[fid])
        maj, count = votes.most_common(1)[0]
        n = len(by_frag[fid])
        pct = 100.0 * count / n
        line = f'  frag {fid}: vote #{maj} ({count}/{n}, {pct:.0f}%)'
        if args.expect and maj != args.expect:
            line += f'  ** expected {args.expect}'
            n_bad += 1
        else:
            if args.expect:
                n_ok += 1
        print(line)
    if args.expect:
        print(f'\nFragments matching expect={args.expect}: {n_ok} ok, {n_bad} wrong')


if __name__ == '__main__':
    main()
