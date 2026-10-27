import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prune_cifar import DATASET, DEV, IdxDS, ResNet18, loaders

ROOT = Path(__file__).resolve().parents[1]
FDIR = ROOT / "data" / ("gauge_feats" if DATASET == "cifar100" else f"gauge_feats_{DATASET}")
FDIR.mkdir(parents=True, exist_ok=True)
EPOCHS = 20
SEEDS = 3

def features(model, ds, batch=512):
    model.eval()
    out = []
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEV)
            h = F.adaptive_avg_pool2d(model.layers(model.stem(xb)), 1).flatten(1)
            out.append(h.cpu().numpy())
    return np.concatenate(out).astype(np.float32)

def train(seed, epochs, train_ds):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = ResNet18().to(DEV)
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    dl = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    for _ in range(epochs):
        m.train()
        for xb, yb, ib in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            F.cross_entropy(m(xb), yb).backward()
            opt.step()
        sch.step()
    return m

def zc(Z):
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-8)

def per_dim_rel(Za, Zb):
    a, b = zc(Za), zc(Zb)
    num = (a * b).sum(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0)) + 1e-9
    return float(np.mean(num / den))

def procrustes(Zs, Z0):

    u, _, vt = np.linalg.svd(Zs.T @ Z0, full_matrices=False)
    return Zs @ (u @ vt)

def dist_rel(Za, Zb, rng, npair=30000):
    n = Za.shape[0]
    i, j = rng.integers(0, n, npair), rng.integers(0, n, npair)
    da = np.linalg.norm(Za[i] - Za[j], axis=1)
    db = np.linalg.norm(Zb[i] - Zb[j], axis=1)
    return float(np.corrcoef(da, db)[0, 1])

def cka(Za, Zb):
    a, b = Za - Za.mean(0), Zb - Zb.mean(0)
    hsic = np.linalg.norm(a.T @ b) ** 2
    return float(hsic / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b) + 1e-12))

def main():
    base_tr, _, test = loaders()
    tr = IdxDS(base_tr)
    Zs = []
    for s in range(SEEDS):
        fp = FDIR / f"feat_seed{s}_e{EPOCHS}.npy"
        if fp.exists():
            Zs.append(np.load(fp))
            print(f"seed {s}: loaded features", flush=True)
            continue
        t0 = time.time()
        m = train(100 + s, EPOCHS, tr)
        Z = features(m, test)
        np.save(fp, Z)
        Zs.append(Z)
        print(f"seed {s}: features {Z.shape} saved ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(0)
    raw, proc, dr, ck = [], [], [], []
    for a in range(SEEDS):
        for b in range(a + 1, SEEDS):
            Za, Zb = Zs[a], Zs[b]
            raw.append(per_dim_rel(Za, Zb))
            proc.append(per_dim_rel(procrustes(zc(Za), zc(Zb)), zc(Zb)))
            dr.append(dist_rel(Za, Zb, rng))
            ck.append(cka(Za, Zb))
    print("\n=== GAUGE PROBE (CIFAR-100 ResNet-18 penultimate, mean over seed pairs) ===")
    print(f"(1) raw per-dim reliability        : {np.mean(raw):.3f}")
    print(f"(2) Procrustes-aligned per-dim rel : {np.mean(proc):.3f}")
    print(f"(3a) rotation-invariant dist-rel   : {np.mean(dr):.3f}")
    print(f"(3b) linear CKA                    : {np.mean(ck):.3f}")
    verdict = "SIGNAL BASIS-HIDDEN -> rotation-invariant reliability recovers it"        if (np.mean(raw) < 0.5 and np.mean(dr) > 0.7) else        ("GENUINELY UNRELIABLE -> stay scalar-only" if np.mean(dr) < 0.5 else "MIXED -> inspect")
    print(f"VERDICT: {verdict}")

if __name__ == "__main__":
    main()
