import os
import sys
import time
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

provenance_module = importlib.import_module("scripts.algo2_input_provenance")
SOURCES = provenance_module.SOURCES
canonical_bytes = provenance_module.canonical_bytes
item_record = provenance_module.item_record
sequence_digest = provenance_module.sequence_digest

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
NITEMS = int(os.environ.get("NITEMS", "200"))
BATCH = int(os.environ.get("BATCH", "16"))
LETTERS = "ABCD"
WORDINGS = [
    "Answer the following multiple-choice question. Respond with only the letter of the correct option.",
    "Choose the correct option. Reply with a single letter (A, B, C, or D).",
    "Read the question and pick the best answer. Output only the option letter.",
    "Which option is correct? Answer with the letter only.",
]
ORDERS = [(0, 1, 2, 3), (3, 2, 1, 0), (1, 2, 3, 0), (2, 3, 0, 1)]
SHOTS = [0, 2]
FEWSHOT = (
    "Question: Which gas do plants absorb from the air?\nA. Oxygen\nB. Carbon dioxide\nC. Nitrogen\nD. Helium\nAnswer: B\n\n"
    "Question: What is the boiling point of water at sea level?\nA. 50C\nB. 90C\nC. 100C\nD. 150C\nAnswer: C\n\n"
)

BENCH = os.environ.get("BENCH", "arc")
LAST_SOURCE_PATH = None


def load_bench(n):
    global LAST_SOURCE_PATH
    if BENCH == "mmlu":
        p = hf_hub_download("cais/mmlu", filename="all/test-00000-of-00001.parquet",
                            revision=SOURCES[BENCH]["revision"], repo_type="dataset", local_dir=ROOT / "data" / "mmlu")
        rows = pq.read_table(p).to_pylist()
        LAST_SOURCE_PATH = Path(p)
        items = []
        for r in rows:
            result = item_record(BENCH, r)
            if result is None:
                continue
            record, identity = result
            items.append({"q": r["question"], "opts": list(r["choices"]), "ans": int(r["answer"]), "source_identity": identity, "item_sha256": hashlib.sha256(canonical_bytes(record)).hexdigest()})
            if len(items) >= n:
                break
        return items
    if BENCH == "hellaswag":
        p = hf_hub_download("Rowan/hellaswag", filename="data/validation-00000-of-00001.parquet",
                            revision=SOURCES[BENCH]["revision"], repo_type="dataset", local_dir=ROOT / "data" / "hellaswag")
        rows = pq.read_table(p).to_pylist()
        LAST_SOURCE_PATH = Path(p)
        items = []
        for r in rows:
            result = item_record(BENCH, r)
            if result is None:
                continue
            record, identity = result
            items.append({"q": r["ctx"], "opts": list(r["endings"]), "ans": int(r["label"]), "source_identity": identity, "item_sha256": hashlib.sha256(canonical_bytes(record)).hexdigest()})
            if len(items) >= n:
                break
        return items
    if BENCH == "openbookqa":
        p = hf_hub_download("allenai/openbookqa", filename="main/train-00000-of-00001.parquet",
                            revision=SOURCES[BENCH]["revision"], repo_type="dataset", local_dir=ROOT / "data" / "openbookqa")
        rows = pq.read_table(p).to_pylist()
        LAST_SOURCE_PATH = Path(p)
        items = []
        for r in rows:
            result = item_record(BENCH, r)
            if result is None:
                continue
            record, identity = result
            ch = r["choices"]
            texts, labels = list(ch["text"]), list(ch["label"])
            items.append({"q": r["question_stem"], "opts": texts, "ans": labels.index(r["answerKey"]), "source_identity": identity, "item_sha256": hashlib.sha256(canonical_bytes(record)).hexdigest()})
            if len(items) >= n:
                break
        return items
    p = hf_hub_download("allenai/ai2_arc", filename="ARC-Easy/test-00000-of-00001.parquet",
                        revision=SOURCES[BENCH]["revision"], repo_type="dataset", local_dir=ROOT / "data" / "arc")
    rows = pq.read_table(p).to_pylist()
    LAST_SOURCE_PATH = Path(p)
    items = []
    for r in rows:
        result = item_record(BENCH, r)
        if result is None:
            continue
        record, identity = result
        ch = r["choices"]
        texts, labels = list(ch["text"]), list(ch["label"])
        items.append({"q": r["question"], "opts": texts, "ans": labels.index(r["answerKey"]), "source_identity": identity, "item_sha256": hashlib.sha256(canonical_bytes(record)).hexdigest()})
        if len(items) >= n:
            break
    return items


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(cache):
    return cache.with_suffix(".provenance.json")


def source_provenance(path):
    source = SOURCES[BENCH]
    relative = path.relative_to(ROOT / "data" / BENCH)
    metadata = ROOT / "data" / BENCH / ".cache" / "huggingface" / "download" / f"{relative}.metadata"
    lines = metadata.read_text(encoding="utf-8").splitlines()
    return {"repository": source["repository"], "filename": source["filename"], "revision": lines[0], "raw_sha256": sha256_file(path)}


def expected_provenance(cache, items, shape):
    item_rows = [{"item_sha256": item["item_sha256"]} for item in items]
    return {"schema_version": 1, "benchmark": BENCH, "model": MODEL, "source": source_provenance(LAST_SOURCE_PATH), "expected_item_sequence_sha256": sequence_digest(item_rows), "selected_items": len(items), "tensor": {"path": cache.relative_to(ROOT).as_posix(), "shape": list(shape), "dtype": "int8", "axis_order": ["wording", "option_order", "shots", "item"], "sha256": None, "complete": False}, "verification_status": "generator-sidecar-in-progress"}


def matching_expected(payload, expected):
    keys = ("benchmark", "model", "source", "expected_item_sequence_sha256", "selected_items")
    return all(payload.get(key) == expected[key] for key in keys) and payload.get("tensor", {}).get("shape") == expected["tensor"]["shape"]


def write_provenance(path, payload, cache):
    payload["tensor"]["sha256"] = sha256_file(cache)
    payload["tensor"]["complete"] = bool((np.load(cache) >= 0).all())
    payload["verification_status"] = "generator-sidecar-complete" if payload["tensor"]["complete"] else "generator-sidecar-in-progress"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_config(model, tok, items, perm, instr, shots, lid):
    prompts, ans = [], []
    for it in items:
        opts = [it["opts"][j] for j in perm]
        ans.append(int(perm.index(it["ans"])))
        body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
        shot = FEWSHOT if shots == 2 else ""
        prompts.append(f"{instr}\n\n{shot}Question: {it['q']}\n{body}\nAnswer:")
    out = np.empty(len(items), dtype=np.int8)
    for b in range(0, len(prompts), BATCH):
        chunk = prompts[b:b + BATCH]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         add_generation_prompt=True, tokenize=False) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
        sub = logits[:, lid].argmax(1).tolist()
        for j in range(len(chunk)):
            out[b + j] = int(sub[j] == ans[b + j])
    return out


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    maxmem = os.environ.get("MAXMEM")
    if maxmem:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="auto",
            max_memory={0: maxmem, "cpu": "60GiB"}).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    lid = [tok(L, add_special_tokens=False).input_ids[-1] for L in LETTERS]
    items = load_bench(NITEMS)
    wording_count, order_count, shot_count, item_count = (
        len(WORDINGS),
        len(ORDERS),
        len(SHOTS),
        len(items),
    )
    tag = MODEL.split("/")[-1] if BENCH == "arc" else f"{BENCH}_{MODEL.split('/')[-1]}"
    cache = ROOT / "data" / f"llm_decomp_{tag}.npy"
    sidecar = provenance_path(cache)
    expected = expected_provenance(
        cache, items, (wording_count, order_count, shot_count, item_count)
    )
    if cache.exists() and np.load(cache).shape == (
        wording_count,
        order_count,
        shot_count,
        item_count,
    ):
        C = np.load(cache)
        if sidecar.exists():
            stored = json.loads(sidecar.read_text(encoding="utf-8"))
            if not matching_expected(stored, expected):
                raise RuntimeError(f"cache provenance does not match current input: {cache}")
        elif not (C >= 0).all():
            raise RuntimeError(f"incomplete legacy cache has no provenance sidecar: {cache}")
    else:
        C = np.full(
            (wording_count, order_count, shot_count, item_count),
            -1,
            dtype=np.int8,
        )
        sidecar.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    for wi, instr in enumerate(WORDINGS):
        for oi, perm in enumerate(ORDERS):
            for si, shots in enumerate(SHOTS):
                if (C[wi, oi, si] >= 0).all():
                    continue
                t0 = time.time()
                C[wi, oi, si] = run_config(model, tok, items, perm, instr, shots, lid)
                np.save(cache, C)
                write_provenance(sidecar, expected, cache)
                print(f"w{wi} o{oi} s{SHOTS[si]}: acc={C[wi,oi,si].mean():.3f} ({time.time()-t0:.0f}s)", flush=True)

    acc = C.mean(3)
    gm = acc.mean()
    aw = acc.mean((1, 2)) - gm
    ao = acc.mean((0, 2)) - gm
    as_ = acc.mean((0, 1)) - gm
    ss_w = order_count * shot_count * (aw ** 2).sum()
    ss_o = wording_count * shot_count * (ao ** 2).sum()
    ss_s = wording_count * order_count * (as_ ** 2).sum()
    ss_tot = ((acc - gm) ** 2).sum()
    ss_res = ss_tot - ss_w - ss_o - ss_s
    print(
        f"\n=== {MODEL} on {BENCH}, {item_count} items, "
        f"{wording_count * order_count * shot_count} configs ==="
    )
    print(f"accuracy: mean={gm:.3f}  std={acc.std():.3f}  range=[{acc.min():.3f},{acc.max():.3f}]")
    print(f"variance share  wording={ss_w/ss_tot:.2f}  option-order={ss_o/ss_tot:.2f}  "
          f"few-shot={ss_s/ss_tot:.2f}  interaction/resid={ss_res/ss_tot:.2f}")


if __name__ == "__main__":
    main()
