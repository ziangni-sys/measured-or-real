import importlib.util
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pc", ROOT / "scripts" / "prune_cifar.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

START = int(os.environ.get("START", "4"))
END = int(os.environ.get("END", "8"))
EP = int(os.environ.get("EPOCHS", "30"))

base_tr, base_noaug, test = pc.loaders()
tr = pc.IdxDS(base_tr)
noaug = pc.IdxDS(base_noaug)
N = len(tr)
sdir = ROOT / "data" / ("prune_scores" if pc.DATASET == "cifar100" else f"prune_scores_{pc.DATASET}")
sdir.mkdir(parents=True, exist_ok=True)
print(f"{pc.DATASET}: training seeds {START}..{END-1}, N={N}, epochs={EP}", flush=True)
for s in range(START, END):
    sp = sdir / f"seed{s}_e{EP}.npz"
    if sp.exists():
        print(f"seed {s}: cached", flush=True)
        continue
    t0 = time.time()
    acc, ph, el = pc.train_model(tr, test, EP, seed=100 + s, track_ds=noaug, n_track=N)
    du_s = pc.dyn_unc(ph)
    np.savez(sp, du=du_s, el=el, acc=acc)
    print(f"seed {s}: acc={acc:.4f} ({time.time()-t0:.0f}s) [saved]", flush=True)
print(f"{pc.DATASET} SEEDS DONE", flush=True)
