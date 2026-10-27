"""Write MANIFEST.json: path, size and SHA-256 of every file in the release.

Run from the repository root:
    uv run python scripts/build_manifest.py
Text files are hashed with LF line endings (see scripts/release_files.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from release_files import MANIFEST, ROOT, digest, release_files  # noqa: E402


def main() -> None:
    files = []
    for path in release_files():
        size, sha = digest(path)
        files.append({"path": path.relative_to(ROOT).as_posix(), "bytes": size, "sha256": sha})
    payload = {
        "hashing": "sha256; text files are hashed with LF line endings",
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote MANIFEST.json with {len(files)} files")


if __name__ == "__main__":
    main()
