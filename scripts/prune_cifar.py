import argparse
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[1]
DEV = "cuda"
DATASET = os.environ.get("DATASET", "cifar100")
_DS = {
    "cifar100": dict(repo="uoft-cs/cifar100", cfg="cifar100", label="fine_label", ncls=100,
                     mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
    "cifar10": dict(repo="uoft-cs/cifar10", cfg="plain_text", label="label", ncls=10,
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
}[DATASET]
NCLS = _DS["ncls"]

class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + self.sc(x))

class ResNet18(nn.Module):
    def __init__(self, ncls=NCLS):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.layers = nn.Sequential(
            Block(64, 64, 1), Block(64, 64, 1),
            Block(64, 128, 2), Block(128, 128, 1),
            Block(128, 256, 2), Block(256, 256, 1),
            Block(256, 512, 2), Block(512, 512, 1),
        )
        self.fc = nn.Linear(512, ncls)

    def forward(self, x):
        x = self.layers(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

class CIFARmem(Dataset):
    def __init__(self, imgs, labels, transform):
        self.imgs, self.labels, self.t = imgs, labels, transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.t(Image.fromarray(self.imgs[i])), int(self.labels[i])

def _load_split(split):

    cache = ROOT / "data" / f"{DATASET}_hf" / f"{split}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["imgs"], z["labels"]
    fn = f"{_DS['cfg']}/{split}-00000-of-00001.parquet"
    p = hf_hub_download(_DS["repo"], filename=fn, repo_type="dataset", local_dir=ROOT / "data" / f"{DATASET}_hf")
    t = pq.read_table(p)
    labels = np.array(t.column(_DS["label"]).to_pylist(), dtype=np.int64)
    col = t.column("img").to_pylist()
    imgs = np.zeros((len(labels), 32, 32, 3), dtype=np.uint8)
    for i, d in enumerate(col):
        imgs[i] = np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))
    np.savez(cache, imgs=imgs, labels=labels)
    return imgs, labels

def loaders():
    mean, std = _DS["mean"], _DS["std"]
    tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    te = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    xtr, ytr = _load_split("train")
    xte, yte = _load_split("test")
    train = CIFARmem(xtr, ytr, tr)
    train_noaug = CIFARmem(xtr, ytr, te)
    test = CIFARmem(xte, yte, te)
    return train, train_noaug, test

def train_model(train_ds, test_ds, epochs, seed, track_ds=None, n_track=0, batch=256):

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ResNet18().to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=False)

    prob_hist = np.zeros((epochs, n_track), dtype=np.float32) if track_ds is not None else None
    el2n = np.zeros(n_track, dtype=np.float32) if track_ds is not None else None
    el2n_ep = min(9, epochs - 1)
    for ep in range(epochs):
        model.train()
        for xb, yb, ib in dl:
            xb, yb = xb.to(DEV, non_blocking=True), yb.to(DEV, non_blocking=True)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        if track_ds is not None:
            _record(model, track_ds, prob_hist, el2n, ep, el2n_ep)
    acc = evaluate(model, test_ds)
    return acc, prob_hist, el2n

def _record(model, track_ds, prob_hist, el2n, ep, el2n_ep):
    model.eval()
    dl = DataLoader(track_ds, batch_size=512, shuffle=False, num_workers=0)
    with torch.no_grad():
        for xb, yb, ib in dl:
            xb = xb.to(DEV, non_blocking=True)
            p = F.softmax(model(xb), dim=1).cpu()
            tp = p[torch.arange(len(yb)), yb].numpy()
            prob_hist[ep, ib.numpy()] = tp
            if ep == el2n_ep:
                oneh = F.one_hot(yb, p.shape[1]).float()
                el2n[ib.numpy()] = (p - oneh).norm(dim=1).numpy()

def evaluate(model, test_ds, batch=512):
    model.eval()
    dl = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=0)
    correct = tot = 0
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEV, non_blocking=True)
            pred = model(xb).argmax(1).cpu()
            correct += (pred == yb).sum().item()
            tot += len(yb)
    return correct / tot

class IdxDS(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, i

def dyn_unc(prob_hist, warm=5):

    w = min(warm, prob_hist.shape[0] - 1)
    return prob_hist[w:].std(axis=0)

def per_example_reliability(scores):

    S = scores.shape[0]
    m = scores.mean(0)
    sig2 = scores.var(0, ddof=1)
    var_total = m.var(ddof=1)
    var_signal = max(var_total - sig2.mean() / S, 1e-6)
    rel = var_signal / (var_signal + sig2 / S)
    return m, np.clip(rel, 0, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--retrain_epochs", type=int, default=40)
    ap.add_argument("--keeps", type=str, default="0.1,0.3,0.5")
    ap.add_argument("--out", type=str, default="data/prune_cifar.json")
    args = ap.parse_args()
    keeps = [float(x) for x in args.keeps.split(",")]

    base_tr, base_noaug, test = loaders()
    tr = IdxDS(base_tr)
    noaug = IdxDS(base_noaug)
    N = len(tr)
    print(f"CIFAR-100 train N={N}, seeds={args.seeds}, epochs={args.epochs}", flush=True)

    sdir = ROOT / "data" / ("prune_scores" if DATASET == "cifar100" else f"prune_scores_{DATASET}")
    sdir.mkdir(parents=True, exist_ok=True)
    du_all, el_all, full_acc = [], [], []
    for s in range(args.seeds):
        sp = sdir / f"seed{s}_e{args.epochs}.npz"
        if sp.exists():
            z = np.load(sp)
            du_all.append(z["du"]); el_all.append(z["el"]); full_acc.append(float(z["acc"]))
            print(f"seed {s}: loaded cached (acc={float(z['acc']):.4f})", flush=True)
            continue
        t0 = time.time()
        acc, ph, el = train_model(tr, test, args.epochs, seed=100 + s, track_ds=noaug, n_track=N)
        du_s = dyn_unc(ph)
        np.savez(sp, du=du_s, el=el, acc=acc)
        du_all.append(du_s); el_all.append(el); full_acc.append(acc)
        print(f"seed {s}: full-data test acc={acc:.4f}  ({time.time()-t0:.0f}s) [saved]", flush=True)

    du = np.stack(du_all)
    el = np.stack(el_all)
    duz = (du - du.mean(1, keepdims=True)) / (du.std(1, keepdims=True) + 1e-8)
    elz = (el - el.mean(1, keepdims=True)) / (el.std(1, keepdims=True) + 1e-8)

    def xcorr(z):
        cs = []
        for i in range(z.shape[0]):
            for j in range(i + 1, z.shape[0]):
                cs.append(np.corrcoef(z[i], z[j])[0, 1])
        return float(np.mean(cs))

    rel_du_pair = xcorr(duz)
    rel_el_pair = xcorr(elz)
    m_du, rel_du = per_example_reliability(duz)
    m_el, rel_el = per_example_reliability(elz)
    print(f"[cross-seed] mean pairwise cross-seed corr: Dyn-Unc={rel_du_pair:.3f}  EL2N={rel_el_pair:.3f}")
    print(f"[cross-seed] per-example reliability median: Dyn-Unc={np.median(rel_du):.3f}  EL2N={np.median(rel_el):.3f}")

    plain = m_du
    shrunk = rel_du * m_du

    rng = np.random.default_rng(0)
    outp = ROOT / args.out
    results = json.loads(outp.read_text()) if outp.exists() else {}
    results.update({"full_acc": full_acc, "N": N, "epochs": args.epochs, "seeds": args.seeds,
                    "cross_seed_corr": {"dyn_unc": rel_du_pair, "el2n": rel_el_pair},
                    "reliability_median": {"dyn_unc": float(np.median(rel_du)), "el2n": float(np.median(rel_el))}})
    results.setdefault("frontier", {})
    outp.write_text(json.dumps(results, indent=2))

    for keep in keeps:
        k = int(keep * N)
        sel = {
            "random": rng.permutation(N)[:k],
            "plain": np.argsort(-plain)[:k],
            "shrunk": np.argsort(-shrunk)[:k],
        }
        row = results["frontier"].get(f"{keep}", {})
        for name, idx in sel.items():
            if name in row:
                continue
            t0 = time.time()
            sub = Subset(tr, idx.tolist())
            acc, _, _ = train_model(sub, test, args.retrain_epochs, seed=7)
            row[name] = acc
            if "shrunk" in row and "plain" in row:
                row["shrunk_minus_plain"] = row["shrunk"] - row["plain"]
            results["frontier"][f"{keep}"] = row
            outp.write_text(json.dumps(results, indent=2))
            print(f"keep={keep:.2f} {name:7s} acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    print("=== frontier ===")
    for keep, row in results["frontier"].items():
        g = row.get
        print(f"keep {keep}: random {g('random', float('nan')):.4f}  plain {g('plain', float('nan')):.4f}  "
              f"shrunk {g('shrunk', float('nan')):.4f}  (shrunk-plain {g('shrunk_minus_plain', float('nan')):+.4f})")
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
