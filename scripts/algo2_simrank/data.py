from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


MODEL_ORDER = [
    "Qwen2.5-0.5B-Instruct",
    "Qwen2-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2-7B-Instruct",
    "SmolLM2-135M-Instruct",
    "SmolLM2-360M-Instruct",
    "SmolLM2-1.7B-Instruct",
    "Falcon3-1B-Instruct",
    "Falcon3-3B-Instruct",
    "Falcon3-7B-Instruct",
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
]

SIDECAR_SUFFIX = ".provenance.json"
LEGACY_STATUS = "generator-enforced-not-artifact-verifiable"


def tensor_path(data_dir: Path, benchmark: str, model: str) -> Path:
    tag = model if benchmark == "arc" else f"{benchmark}_{model}"
    return data_dir / f"llm_decomp_{tag}.npy"


def tensor_sidecar_path(data_dir: Path, benchmark: str, model: str) -> Path:
    path = tensor_path(data_dir, benchmark, model)
    return path.with_suffix(SIDECAR_SUFFIX)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_item_provenance(
    data_dir: Path,
    benchmark: str,
    models: list[str],
    *,
    verify_tensor_hash: bool = True,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    expected_sequences: set[str] = set()
    all_verified = True
    for model in models:
        tensor = tensor_path(data_dir, benchmark, model)
        sidecar = tensor_sidecar_path(data_dir, benchmark, model)
        record: dict[str, object] = {
            "model": model,
            "tensor_path": str(tensor),
            "sidecar_path": str(sidecar),
        }
        if not sidecar.exists():
            record["verification_status"] = LEGACY_STATUS
            all_verified = False
            records.append(record)
            continue
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            record["verification_status"] = "invalid-sidecar"
            record["error"] = str(error)
            all_verified = False
            records.append(record)
            continue
        sequence = payload.get("expected_item_sequence_sha256")
        recorded_hash = payload.get("tensor", {}).get("sha256")
        complete = payload.get("tensor", {}).get("complete")
        hash_matches = (
            isinstance(recorded_hash, str)
            and recorded_hash == sha256_file(tensor)
            if verify_tensor_hash
            else isinstance(recorded_hash, str)
        )
        record.update(
            expected_item_sequence_sha256=sequence,
            tensor_hash_matches=hash_matches,
            sidecar_complete=complete,
        )
        if isinstance(sequence, str):
            expected_sequences.add(sequence)
        if not (isinstance(sequence, str) and complete is True and hash_matches):
            record["verification_status"] = "invalid-or-incomplete-sidecar"
            all_verified = False
        else:
            record["verification_status"] = "sidecar-verified"
        records.append(record)
    if all_verified and len(expected_sequences) == 1:
        status = "sidecar-verified"
    elif all(not tensor_sidecar_path(data_dir, benchmark, model).exists() for model in models):
        status = LEGACY_STATUS
    else:
        status = "mixed-or-inconsistent-sidecars-unverified"
    return {
        "benchmark": benchmark,
        "verification_status": status,
        "expected_item_sequence_sha256": (
            next(iter(expected_sequences))
            if status == "sidecar-verified"
            else None
        ),
        "tensors": records,
    }


def require_verified_item_provenance(
    data_dir: Path, benchmark: str, models: list[str]
) -> dict[str, object]:
    provenance = inspect_item_provenance(data_dir, benchmark, models)
    if provenance["verification_status"] != "sidecar-verified":
        raise ValueError(
            f"{benchmark}: item alignment is not artifact-verified; "
            f"status={provenance['verification_status']}"
        )
    return provenance


def load_benchmark(
    data_dir: Path,
    benchmark: str,
    *,
    min_items: int = 400,
    require_item_provenance: bool = False,
) -> tuple[list[str], np.ndarray, list[dict[str, str]]]:
    """Load complete model tensors with a common item count."""
    candidates: list[tuple[str, np.ndarray]] = []
    skipped: list[dict[str, str]] = []
    for model in MODEL_ORDER:
        path = tensor_path(data_dir, benchmark, model)
        if not path.exists():
            skipped.append({"model": model, "reason": "missing"})
            continue
        tensor = np.load(path)
        if tensor.ndim != 4 or tensor.shape[:3] != (4, 4, 2):
            skipped.append({"model": model, "reason": f"unexpected-shape-{tensor.shape}"})
            continue
        if tensor.shape[-1] < min_items:
            skipped.append({"model": model, "reason": f"only-{tensor.shape[-1]}-items"})
            continue
        if not np.isfinite(tensor).all() or (tensor < 0).any() or (tensor > 1).any():
            skipped.append({"model": model, "reason": "incomplete-or-nonbinary"})
            continue
        candidates.append((model, tensor.astype(np.float64, copy=False)))

    if len(candidates) < 2:
        raise ValueError(f"{benchmark}: fewer than two complete model tensors")
    item_counts = [tensor.shape[-1] for _, tensor in candidates]
    target_items = max(set(item_counts), key=item_counts.count)
    kept = [(model, tensor) for model, tensor in candidates if tensor.shape[-1] == target_items]
    for model, tensor in candidates:
        if tensor.shape[-1] != target_items:
            skipped.append({"model": model, "reason": f"item-count-{tensor.shape[-1]}-not-{target_items}"})
    if len(kept) < 2:
        raise ValueError(f"{benchmark}: fewer than two tensors share the modal item count")
    names = [model for model, _ in kept]
    if require_item_provenance:
        require_verified_item_provenance(data_dir, benchmark, names)
    return names, np.stack([tensor for _, tensor in kept]), skipped


def load_arc_eight_wordings(data_dir: Path) -> tuple[list[str], np.ndarray]:
    """Combine four original and four held-out ARC wordings at identity order.

    The held-out run did not cross the new wordings with all option orders, so
    the returned tensor has one order level and must not be described as an
    8 x 4 factorial design.
    """
    names, original, skipped = load_benchmark(data_dir, "arc")
    if skipped:
        missing = {row["model"] for row in skipped if row["reason"] != "missing"}
        if missing:
            raise ValueError(f"ARC contains invalid model tensors: {sorted(missing)}")
    combined = []
    for model_index, model in enumerate(names):
        path = data_dir / f"llm_holdout_{model}.npy"
        if not path.exists():
            raise ValueError(f"missing held-out wording tensor: {path}")
        heldout = np.load(path)
        if heldout.shape != (4, 2, original.shape[-1]) or (heldout < 0).any() or (heldout > 1).any():
            raise ValueError(f"invalid held-out wording tensor: {path}")
        original_identity_order = original[model_index, :, 0, :, :]
        eight_wordings = np.concatenate([original_identity_order, heldout], axis=0)
        combined.append(eight_wordings[:, None, :, :])
    return names, np.stack(combined).astype(np.float64, copy=False)
