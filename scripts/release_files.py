"""Which files belong to the release, and how their hashes are computed.

Shared by scripts/build_manifest.py and scripts/verify_release.py. Text files
are hashed with LF line endings, so a checkout that converts line endings
still verifies. Everything else is hashed byte for byte.
"""

from __future__ import annotations

import fnmatch
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.json"

# Folders that are never part of the release (environments, caches, build output).
EXCLUDED_DIRS = {".git", ".venv", "target", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}

# Generated or downloaded paths, matched against the POSIX path relative to the root.
# Keep in sync with .gitignore.
EXCLUDED_PATTERNS = [
    "MANIFEST.json",
    "figures/preview/*",
    "results/*_reproduced/*",
    "results/algo2_alpha*/*",
    "results/_smoke/*",
    "data/arc/*",
    "data/mmlu/*",
    "data/hellaswag/*",
    "data/openbookqa/*",
    "data/cifar_hf/*",
    "data/cifar100_hf/*",
    "data/cifar10_hf/*",
    "data/gauge_feats*/*",
    "data/cifar100_labels.npy",
    "data/prune_cifar.json",
    "figscripts/schematics/*.pdf",
    "*.pyc",
    "*.aux",
    "*.bbl",
    "*.blg",
    "*.fdb_latexmk",
    "*.fls",
    "*.log",
    "*.out",
    "*.synctex.gz",
    "*.toc",
]

TEXT_SUFFIXES = {
    ".md", ".txt", ".py", ".rs", ".toml", ".lock", ".json", ".csv", ".cff", ".tex",
}
TEXT_NAMES = {"LICENSE", "justfile", ".gitignore", ".gitattributes", ".python-version"}


def is_excluded(relative: str) -> bool:
    parts = relative.split("/")
    if any(part in EXCLUDED_DIRS for part in parts[:-1]):
        return True
    return any(fnmatch.fnmatchcase(relative, pattern) for pattern in EXCLUDED_PATTERNS)


def release_files() -> list[Path]:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if not is_excluded(relative):
            files.append(path)
    return files


def is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES


def content(path: Path) -> bytes:
    data = path.read_bytes()
    if is_text(path):
        data = data.replace(b"\r\n", b"\n")
    return data


def digest(path: Path) -> tuple[int, str]:
    data = content(path)
    return len(data), hashlib.sha256(data).hexdigest()
