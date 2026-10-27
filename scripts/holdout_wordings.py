import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NITEMS = 2365
REPOS = [
    "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "tiiuae/Falcon3-1B-Instruct", "tiiuae/Falcon3-3B-Instruct",
    "unsloth/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2-7B-Instruct", "tiiuae/Falcon3-7B-Instruct",
]
HEAVY_MAXMEM = {"Qwen/Qwen2.5-7B-Instruct": "10GiB", "Qwen/Qwen2-7B-Instruct": "10GiB",
                "tiiuae/Falcon3-7B-Instruct": "10GiB"}
NEW_WORDINGS = [
    "You will be given a question with four answer options. Decide which one is correct and reply with its letter alone.",
    "Below is a quiz item. State the letter of the right answer and nothing more.",
    "Consider the question and the options, then give exactly one letter (A, B, C, or D) as your answer.",
    "Identify the correct choice. Your entire response must be a single option letter.",
]
PERM = (0, 1, 2, 3)


def evaluate():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import llm_decomp as ld

    items = ld.load_bench(NITEMS)
    n = len(items)
    for repo in REPOS:
        name = repo.split("/")[-1]
        cache = ROOT / "data" / f"llm_holdout_{name}.npy"
        if cache.exists():
            C = np.load(cache)
            if C.shape == (4, 2, n) and (C >= 0).all():
                print(f"skip {name} (done)", flush=True)
                continue
        else:
            C = np.full((4, 2, n), -1, dtype=np.int8)
        tok = AutoTokenizer.from_pretrained(repo)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if repo in HEAVY_MAXMEM:
            model = AutoModelForCausalLM.from_pretrained(
                repo, dtype=torch.bfloat16, device_map="auto",
                max_memory={0: HEAVY_MAXMEM[repo], "cpu": "60GiB"}).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).cuda().eval()
        lid = [tok(L, add_special_tokens=False).input_ids[-1] for L in ld.LETTERS]
        for wi, instr in enumerate(NEW_WORDINGS):
            for si, shots in enumerate(ld.SHOTS):
                if (C[wi, si] >= 0).all():
                    continue
                t0 = time.time()
                C[wi, si] = ld.run_config(model, tok, items, PERM, instr, shots, lid)
                np.save(cache, C)
                print(f"{name} w{wi} s{shots}: acc={C[wi,si].mean():.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()


def analyze(pairwise_path, output_path):
    with pairwise_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["benchmark"] == "arc"]

    pairs = []
    for row in rows:
        verdict = row["verdict"]
        if verdict == "stable-a":
            pairs.append({
                "a": row["model_a"],
                "b": row["model_b"],
                "verdict": "certified",
                "sign": 1,
            })
        elif verdict == "stable-b":
            pairs.append({
                "a": row["model_a"],
                "b": row["model_b"],
                "verdict": "certified",
                "sign": -1,
            })
        elif verdict == "protocol-dependent":
            pairs.append({
                "a": row["model_a"],
                "b": row["model_b"],
                "verdict": verdict,
                "gaps": [
                    float(row["gap_protocol_0"]),
                    float(row["gap_protocol_1"]),
                ],
            })

    H = {}
    for repo in REPOS:
        name = repo.split("/")[-1]
        f = ROOT / "data" / f"llm_holdout_{name}.npy"
        if f.exists():
            C = np.load(f)
            if (C >= 0).all():
                H[name] = C.astype(np.float64)
    fails, checked, cfg_flips, cfg_total = [], 0, 0, 0
    pd_rep = []
    for p in pairs:
        a, b = p["a"], p["b"]
        if a not in H or b not in H:
            continue
        held = [(H[a][:, s, :] - H[b][:, s, :]).mean() for s in range(2)]
        if p["verdict"] == "certified":
            checked += 1
            if any(np.sign(h) != p["sign"] for h in held):
                fails.append((a, b, [float(h) * 100 for h in held]))
            per_cfg = (H[a].mean(2) - H[b].mean(2)).reshape(-1)
            cfg_flips += int((np.sign(per_cfg) == -p["sign"]).sum())
            cfg_total += per_cfg.size
        elif p["verdict"] == "protocol-dependent":
            pd_rep.append((a, b, [float(np.sign(h)) for h in held],
                           [float(np.sign(g)) for g in p["gaps"]]))
    print(f"\nout-of-pool wordings: {checked} certified pairs checked, "
          f"{len(fails)} direction failures ({100*len(fails)/max(checked,1):.2f}%), "
          f"per-config flips {cfg_flips}/{cfg_total} ({100*cfg_flips/max(cfg_total,1):.3f}%)")
    for a, b, h in fails:
        print(f"  FAIL {a.replace('-Instruct','')} vs {b.replace('-Instruct','')}: "
              f"held-out gaps {h[0]:+.1f}/{h[1]:+.1f}")
    pd_match = sum(1 for _, _, hs, gs in pd_rep if hs == gs)
    print(f"protocol-dependent sign patterns replicated on new wordings: "
          f"{pd_match}/{len(pd_rep)}")
    out = dict(checked=checked, failures=len(fails),
               fail_pairs=[(a, b) for a, b, _ in fails],
               cfg_flips=[cfg_flips, cfg_total], pd_replicated=[pd_match, len(pd_rep)],
               wordings=NEW_WORDINGS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--pairwise-path",
        type=Path,
        default=ROOT / "results" / "algo2_reported" / "pairwise.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "holdout_wordings.json",
        help="summary JSON to write (default: data/holdout_wordings.json)",
    )
    args = parser.parse_args()
    if not args.analyze_only:
        evaluate()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    analyze(args.pairwise_path, output)
