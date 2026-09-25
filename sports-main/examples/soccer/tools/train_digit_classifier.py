#!/usr/bin/env python3
"""Train a small jersey-digit classifier on data/digits/ (step 2).

Fragment-wise train/val split so the same tracklet does not leak across splits.
Skips folders whose names start with '_' (e.g. _unlabelled).

Usage (RunPod, from examples/soccer):
  pip install torch torchvision   # in venv with cv2; GPU optional but faster
  python tools/train_digit_classifier.py \\
    --data data/digits \\
    --out-dir runs/digit_cls_v1

  python tools/train_digit_classifier.py --data data/digits --eval-only \\
    --checkpoint runs/digit_cls_v1/best.pt
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
except ModuleNotFoundError as exc:
    raise SystemExit(
        'PyTorch required: pip install torch torchvision\n'
        '(Use GPU wheel on RunPod if available.)'
    ) from exc

FRAG_RE = re.compile(r'^frag(\d+)_f(\d+)\.jpg$', re.I)


def parse_frag_id(path: Path) -> int | None:
    m = FRAG_RE.match(path.name)
    return int(m.group(1)) if m else None


def discover_samples(data_root: Path) -> tuple[list[tuple[Path, str]], list[str]]:
    """Return ([(jpg_path, class_name), ...], sorted class names)."""
    if not data_root.is_dir():
        raise SystemExit(f'Not a directory: {data_root}')
    class_names = sorted(
        p.name for p in data_root.iterdir()
        if p.is_dir() and not p.name.startswith('_') and p.name.isdigit()
    )
    if not class_names:
        raise SystemExit(f'No digit class folders in {data_root} '
                         '(expected 5, 6, 10, ...)')
    samples: list[tuple[Path, str]] = []
    for cls in class_names:
        for jpg in sorted((data_root / cls).glob('*.jpg')):
            samples.append((jpg, cls))
    if not samples:
        raise SystemExit(f'No JPGs under {data_root}')
    return samples, class_names


def split_by_fragment(
        samples: list[tuple[Path, str]],
        val_frac: float,
        seed: int,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], dict]:
    """Hold out whole fragments; stratify by class so every digit appears in val."""
    frag_to_items: dict[int, list[tuple[Path, str]]] = defaultdict(list)
    no_frag: list[tuple[Path, str]] = []
    for path, cls in samples:
        fid = parse_frag_id(path)
        if fid is None:
            no_frag.append((path, cls))
        else:
            frag_to_items[fid].append((path, cls))

    cls_to_frags: dict[str, set[int]] = defaultdict(set)
    for fid, items in frag_to_items.items():
        classes = {cls for _, cls in items}
        if len(classes) != 1:
            # One fragment folder should be one label; use majority label.
            cls = Counter(c for _, c in items).most_common(1)[0][0]
        else:
            cls = next(iter(classes))
        cls_to_frags[cls].add(fid)

    rng = random.Random(seed)
    val_frags: set[int] = set()
    split_info: dict[str, dict] = {}
    for cls in sorted(cls_to_frags.keys()):
        frags = sorted(cls_to_frags[cls])
        rng.shuffle(frags)
        if len(frags) == 1:
            split_info[cls] = {
                'train_frags': frags, 'val_frags': [],
                'note': 'single fragment — all train',
            }
            continue
        n_val = max(1, int(round(len(frags) * val_frac)))
        n_val = min(n_val, len(frags) - 1)
        val_cls = set(frags[:n_val])
        val_frags |= val_cls
        split_info[cls] = {
            'train_frags': frags[n_val:],
            'val_frags': frags[:n_val],
        }

    train_frags = set(frag_to_items.keys()) - val_frags
    train, val = [], []
    for fid in sorted(train_frags):
        train.extend(frag_to_items[fid])
    for fid in sorted(val_frags):
        val.extend(frag_to_items[fid])
    train.extend(no_frag)
    split_info['_summary'] = {
        'n_train': len(train),
        'n_val': len(val),
        'val_frags': sorted(val_frags),
    }
    return train, val, split_info


def macro_val_score(val_m: dict) -> float:
    """Mean per-class acc on val (classes with n>0 only)."""
    accs = [
        float(v['acc']) for v in val_m.get('per_class', {}).values()
        if v.get('n') and v.get('acc') is not None
    ]
    return sum(accs) / len(accs) if accs else 0.0


class DigitCropDataset(Dataset):
    def __init__(
            self,
            items: list[tuple[Path, str]],
            class_to_idx: dict[str, int],
            img_size: int,
            augment: bool,
    ):
        self.items = items
        self.class_to_idx = class_to_idx
        self.img_size = img_size
        t_list = [transforms.ToPILImage()]
        if augment:
            t_list.extend([
                transforms.RandomApply([
                    transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
                ], p=0.7),
                transforms.RandomAffine(
                    degrees=8, translate=(0.06, 0.06), scale=(0.92, 1.08)),
            ])
        t_list.extend([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose(t_list)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, cls = self.items[i]
        bgr = cv2.imread(str(path))
        if bgr is None:
            bgr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(rgb)
        y = self.class_to_idx[cls]
        return x, y, str(path)


class SmallDigitCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


@torch.no_grad()
def evaluate(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        n_classes: int,
        idx_to_class: dict[int, str],
) -> dict:
    model.eval()
    correct = 0
    total = 0
    per_cls_correct = Counter()
    per_cls_total = Counter()
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for x, y, _paths in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.size(0)
        for t, p in zip(y.cpu().tolist(), pred.cpu().tolist()):
            per_cls_total[t] += 1
            conf[t, p] += 1
            if t == p:
                per_cls_correct[t] += 1
    acc = correct / max(total, 1)
    per_class = {}
    for i in range(n_classes):
        name = idx_to_class[i]
        n = per_cls_total[i]
        per_class[name] = {
            'n': n,
            'acc': round(per_cls_correct[i] / n, 4) if n else None,
        }
    return {
        'acc': round(acc, 4),
        'n': total,
        'per_class': per_class,
        'confusion': conf.tolist(),
    }


def train_main(args) -> None:
    samples, class_names = discover_samples(args.data)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    train_items, val_items, split_info = split_by_fragment(
        samples, args.val_frac, args.seed)

    print(f'Classes: {class_names}')
    print(f'Samples: {len(samples)} total, train {len(train_items)}, val {len(val_items)}')
    print(f'  (stratified fragment split, val_frac={args.val_frac})')
    for cls in class_names:
        si = split_info.get(cls, {})
        if si:
            print(f'    {cls}: val frags={si.get("val_frags")} '
                  f'({si.get("note") or "ok"})')

    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f'Device: {device}')

    train_ds = DigitCropDataset(
        train_items, class_to_idx, args.img_size, augment=True)
    val_ds = DigitCropDataset(
        val_items, class_to_idx, args.img_size, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SmallDigitCNN(len(class_names)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'split.json').write_text(
        json.dumps(split_info, indent=2), encoding='utf-8')
    best_macro = -1.0
    best_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_batch = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            n_batch += 1
        val_m = evaluate(model, val_loader, device, len(class_names), idx_to_class)
        train_loss = loss_sum / max(n_batch, 1)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val': val_m})
        print(f'Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  '
              f'val_acc={val_m["acc"]:.3f}')
        macro = macro_val_score(val_m)
        if macro > best_macro:
            best_macro = macro
            best_acc = val_m['acc']
            ckpt = {
                'model': model.state_dict(),
                'class_names': class_names,
                'class_to_idx': class_to_idx,
                'img_size': args.img_size,
                'val_acc': best_acc,
                'val_macro_acc': best_macro,
                'val_metrics': val_m,
            }
            torch.save(ckpt, args.out_dir / 'best.pt')
            (args.out_dir / 'val_metrics.json').write_text(
                json.dumps(val_m, indent=2), encoding='utf-8')

    (args.out_dir / 'history.json').write_text(
        json.dumps(history, indent=2), encoding='utf-8')
    print(f'\nBest val macro acc: {best_macro:.3f}  (overall {best_acc:.3f})')
    print(f'  -> {args.out_dir / "best.pt"}')
    print('Per-class val (best epoch):')
    best_m = json.loads((args.out_dir / 'val_metrics.json').read_text())
    for name in class_names:
        pc = best_m['per_class'].get(name, {})
        print(f'  {name}: n={pc.get("n")} acc={pc.get("acc")}')

    all_ds = DigitCropDataset(
        samples, class_to_idx, args.img_size, augment=False)
    all_loader = DataLoader(all_ds, batch_size=args.batch_size, shuffle=False)
    try:
        best_ckpt = torch.load(
            args.out_dir / 'best.pt', map_location=device, weights_only=False)
    except TypeError:
        best_ckpt = torch.load(args.out_dir / 'best.pt', map_location=device)
    model.load_state_dict(best_ckpt['model'])
    all_m = evaluate(model, all_loader, device, len(class_names), idx_to_class)
    (args.out_dir / 'all_data_metrics.json').write_text(
        json.dumps(all_m, indent=2), encoding='utf-8')
    print('\nAll-data eval (includes train frags — optimistic):')
    for name in class_names:
        pc = all_m['per_class'].get(name, {})
        print(f'  {name}: n={pc.get("n")} acc={pc.get("acc")}')


def eval_main(args) -> None:
    try:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
    class_names = ckpt['class_names']
    class_to_idx = ckpt['class_to_idx']
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    img_size = int(ckpt.get('img_size', args.img_size))
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = SmallDigitCNN(len(class_names))
    model.load_state_dict(ckpt['model'])
    model.to(device)

    samples, _ = discover_samples(args.data)
    train_items, val_items, _ = split_by_fragment(
        samples, args.val_frac, args.seed)
    if args.eval_all:
        items = samples
        label = 'all'
    else:
        items = val_items
        label = 'val'
    ds = DigitCropDataset(items, class_to_idx, img_size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    m = evaluate(model, loader, device, len(class_names), idx_to_class)
    print(f'Eval split={label} n={len(items)}')
    print(json.dumps(m, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=Path('data/digits'))
    ap.add_argument('--out-dir', type=Path, default=Path('runs/digit_cls_v1'))
    ap.add_argument('--checkpoint', type=Path, default=None,
                    help='With --eval-only, path to best.pt')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--eval-all', action='store_true',
                    help='With --eval-only: score all JPGs (not just val frags)')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--img-size', type=int, default=64)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    if args.eval_only:
        if not args.checkpoint or not args.checkpoint.is_file():
            raise SystemExit('--eval-only requires --checkpoint best.pt')
        eval_main(args)
    else:
        train_main(args)


if __name__ == '__main__':
    main()
