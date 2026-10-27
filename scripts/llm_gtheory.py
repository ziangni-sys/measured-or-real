import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def gtheory(C):

    W, O, S, N = C.shape
    X = C.reshape(-1, N).T.astype(np.float64)
    F = X.shape[1]
    g = X.mean()
    mi = X.mean(1)
    mf = X.mean(0)
    ss_i = F * ((mi - g) ** 2).sum()
    ss_f = N * ((mf - g) ** 2).sum()
    resid = X - mi[:, None] - mf[None, :] + g
    ss_r = (resid ** 2).sum()
    ms_i = ss_i / (N - 1)
    ms_f = ss_f / (F - 1)
    ms_r = ss_r / ((N - 1) * (F - 1))
    v_i = max((ms_i - ms_r) / F, 0.0)
    v_f = max((ms_f - ms_r) / N, 0.0)
    v_r = ms_r

    g1 = v_i / (v_i + v_r) if v_i + v_r > 0 else float("nan")
    gF = v_i / (v_i + v_r / F) if v_i + v_r > 0 else float("nan")

    acc = C.mean(3)
    gm = acc.mean()
    aw = acc.mean((1, 2)) - gm
    ao = acc.mean((0, 2)) - gm
    as_ = acc.mean((0, 1)) - gm
    tot = ((acc - gm) ** 2).sum()
    share = dict(wording=(O * S * (aw ** 2).sum()) / tot,
                 order=(W * S * (ao ** 2).sum()) / tot,
                 shot=(W * O * (as_ ** 2).sum()) / tot)
    return dict(v_item=v_i, v_config=v_f, v_resid=v_r, g_single=g1, g_avg=gF, F=F, share=share)

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-0.5B-Instruct"
    bench = sys.argv[2] if len(sys.argv) > 2 else "arc"
    tag = model if bench == "arc" else f"{bench}_{model}"
    C = np.load(ROOT / "data" / f"llm_decomp_{tag}.npy")
    r = gtheory(C)
    print(f"=== G-theory: {model} on {bench} (item x config, F={r['F']}) ===")
    print(f"variance components: item={r['v_item']:.4f}  config={r['v_config']:.4f}  residual={r['v_resid']:.4f}")
    print(f"generalizability coeff (item-difficulty): single-prompt={r['g_single']:.3f}  {r['F']}-prompt-mean={r['g_avg']:.3f}")
    print(f"config-variance facet shares: wording={r['share']['wording']:.2f} "
          f"order={r['share']['order']:.2f} few-shot={r['share']['shot']:.2f}")

if __name__ == "__main__":
    main()
