"""Fine-tune a pretrained ResNet18 on OrganAMNIST (MedMNIST v2) with mixed precision.

    python train.py                 # default: pretrained + AMP + augmentation
    python train.py --no-amp        # full FP32 for comparison
    python train.py --no-pretrained # train from scratch for comparison

Writes <out>/model.pt, artifacts/history.json, artifacts/metrics.json.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ART, CLASSES, build_model, device, load_split, to_tensor


def augment(x):
    """Random small rotation / zoom / shift, done on the GPU per sample.

    No horizontal flips on purpose: flipping would turn a left kidney into a
    right kidney and corrupt the label.
    """
    n = x.size(0)
    ang = (torch.rand(n, device=x.device) - 0.5) * math.radians(20)  # ±10°
    scale = 1 + (torch.rand(n, device=x.device) - 0.5) * 0.2          # 0.9–1.1
    tx, ty = ((torch.rand(2, n, device=x.device) - 0.5) * 0.16)       # ±8%
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)


def batches(x, y, bs, shuffle):
    idx = torch.randperm(len(x)) if shuffle else torch.arange(len(x))
    for i in range(0, len(x), bs):
        j = idx[i:i + bs]
        yield x[j], y[j]


@torch.no_grad()
def evaluate(model, x, y, dev, amp):
    model.eval()
    probs, loss_sum = [], 0.0
    for xb, yb in batches(x, y, 512, False):
        xb, yb = to_tensor(xb.to(dev)), yb.to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            out = model(xb)
        loss_sum += F.cross_entropy(out.float(), yb, reduction="sum").item()
        probs.append(out.float().softmax(1).cpu())
    probs = torch.cat(probs)
    return loss_sum / len(x), (probs.argmax(1) == y).float().mean().item(), probs


def amp_benchmark(dev, xtr, ytr, steps=30, bs=128):
    """Time the same training steps with and without mixed precision."""
    res = {}
    for amp in (False, True):
        model = build_model().to(dev)
        opt = torch.optim.AdamW(model.parameters(), 1e-3)
        xb, yb = to_tensor(xtr[:bs].to(dev)), ytr[:bs].to(dev)
        for k in range(steps + 5):  # first 5 = warm-up, not timed
            if k == 5:
                sync(dev); t = time.perf_counter()
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sync(dev)
        res["amp" if amp else "fp32"] = round((time.perf_counter() - t) / steps * 1000, 1)
    return res  # ms per step


def sync(dev):
    if dev.type == "cuda": torch.cuda.synchronize()
    if dev.type == "mps": torch.mps.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--out", default=str(ART), help="folder for model + metrics")
    args = p.parse_args()

    torch.manual_seed(0)
    dev, amp = device(), not args.no_amp
    print(f"device={dev}  amp={amp}  pretrained={not args.no_pretrained}")

    (xtr, ytr), (xva, yva), (xte, yte) = (load_split(s) for s in ("train", "val", "test"))
    # Keep images as uint8 in memory; convert each batch on the GPU.
    xtr, xva, xte, ytr, yva, yte = map(torch.from_numpy, (xtr, xva, xte, ytr, yva, yte))
    print(f"train={len(xtr)} val={len(xva)} test={len(xte)}")

    model = build_model(not args.no_pretrained).to(dev)
    # AdamW = Adam + decoupled weight decay (L2-style regularisation).
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(xtr) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    # GradScaler protects FP16 gradients from underflow (needed on CUDA).
    scaler = torch.amp.GradScaler(dev.type, enabled=amp and dev.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, seen, correct, loss_sum = time.perf_counter(), 0, 0, 0.0
        for xb, yb in batches(xtr, ytr, args.bs, True):
            xb, yb = to_tensor(xb.to(dev)), yb.to(dev)
            if not args.no_aug:
                xb = augment(xb)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                out = model(xb)
                loss = crit(out, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            seen += len(yb); loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
        sync(dev)
        va_loss, va_acc, _ = evaluate(model, xva, yva, dev, amp)
        row = dict(epoch=ep, train_loss=loss_sum / seen, train_acc=correct / seen,
                   val_loss=va_loss, val_acc=va_acc, secs=time.perf_counter() - t0)
        history.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if va_acc > best:
            best = va_acc
            torch.save(model.state_dict(), out_dir / "model.pt")

    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=dev))
    te_loss, te_acc, probs = evaluate(model, xte, yte, dev, amp)
    pred = probs.argmax(1).numpy()
    y = yte.numpy()
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    print(f"TEST acc={te_acc:.4f} loss={te_loss:.4f}")

    print("benchmarking fp32 vs amp ...")
    bench = amp_benchmark(dev, xtr, ytr)
    print(bench)

    json.dump(history, open(out_dir / "history.json", "w"), indent=2)
    json.dump(dict(
        test_acc=te_acc, test_loss=te_loss, macro_f1=float(f1.mean()),
        per_class={c: dict(precision=float(prec[i]), recall=float(rec[i]), f1=float(f1[i]),
                           support=int(cm[i].sum())) for i, c in enumerate(CLASSES)},
        confusion_matrix=cm.tolist(), amp_benchmark_ms_per_step=bench,
        config=dict(vars(args), device=str(dev), model="ResNet18 (ImageNet pretrained)"),
    ), open(out_dir / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
