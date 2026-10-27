import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NITEMS = 2365
REPOS = [
    "unsloth/Llama-3.2-3B-Instruct", "unsloth/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct", "tiiuae/Falcon3-3B-Instruct",
]
SHOT_TURNS = [
    ("Question: Which gas do plants absorb from the air?\nA. Oxygen\nB. Carbon dioxide\nC. Nitrogen\nD. Helium\nAnswer:", "B"),
    ("Question: What is the boiling point of water at sea level?\nA. 50C\nB. 90C\nC. 100C\nD. 150C\nAnswer:", "C"),
]
PERM = (0, 1, 2, 3)
BATCH = 16


def run_chat_config(model, tok, items, instr, lid, ld, torch):
    prompts, ans = [], []
    for it in items:
        opts = [it["opts"][j] for j in PERM]
        ans.append(int(PERM.index(it["ans"])))
        body = "\n".join(f"{ld.LETTERS[i]}. {o}" for i, o in enumerate(opts))
        prompts.append(f"Question: {it['q']}\n{body}\nAnswer:")
    out = np.empty(len(items), dtype=np.int8)
    for b in range(0, len(prompts), BATCH):
        chunk = prompts[b:b + BATCH]
        texts = []
        for p in chunk:
            msgs = [{"role": "user", "content": f"{instr}\n\n{SHOT_TURNS[0][0]}"},
                    {"role": "assistant", "content": SHOT_TURNS[0][1]},
                    {"role": "user", "content": SHOT_TURNS[1][0]},
                    {"role": "assistant", "content": SHOT_TURNS[1][1]},
                    {"role": "user", "content": p}]
            texts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
        sub = logits[:, lid].argmax(1).tolist()
        for j in range(len(chunk)):
            out[b + j] = int(sub[j] == ans[b + j])
    return out


def evaluate():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import llm_decomp as ld

    items = ld.load_bench(NITEMS)
    n = len(items)
    for repo in REPOS:
        name = repo.split("/")[-1]
        cache = ROOT / "data" / f"llm_chatshot_{name}.npy"
        if cache.exists():
            C = np.load(cache)
            if C.shape == (4, n) and (C >= 0).all():
                print(f"skip {name} (done)", flush=True)
                continue
        else:
            C = np.full((4, n), -1, dtype=np.int8)
        tok = AutoTokenizer.from_pretrained(repo)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).cuda().eval()
        lid = [tok(L, add_special_tokens=False).input_ids[-1] for L in ld.LETTERS]
        for wi, instr in enumerate(ld.WORDINGS):
            if (C[wi] >= 0).all():
                continue
            t0 = time.time()
            C[wi] = run_chat_config(model, tok, items, instr, lid, ld, torch)
            np.save(cache, C)
            print(f"{name} w{wi} chat-2shot: acc={C[wi].mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()


def analyze():
    print("\n=== comparison (accuracy by wording) ===")
    for repo in REPOS:
        name = repo.split("/")[-1]
        f = ROOT / "data" / f"llm_chatshot_{name}.npy"
        if not f.exists():
            continue
        chat = np.load(f).astype(float)
        base = np.load(ROOT / "data" / f"llm_decomp_{name}.npy").astype(float)
        zero = base[:, 0, 0, :].mean(1)
        inline = base[:, 0, 1, :].mean(1)
        print(f"{name}:")
        print(f"  0-shot        {np.round(zero,3)}")
        print(f"  2-shot inline {np.round(inline,3)}")
        print(f"  2-shot chat   {np.round(chat.mean(1),3)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="analyze the included tensors without importing GPU dependencies",
    )
    args = parser.parse_args()
    if not args.analyze_only:
        evaluate()
    analyze()


if __name__ == "__main__":
    main()
