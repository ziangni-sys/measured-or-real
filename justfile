default:
    just --list

sync:
    uv sync

build:
    cargo build --release

test:
    cargo test
    uv run python -m unittest discover -s scripts/algo2_simrank/tests -v

ablate:
    cargo run --release -p erel-cli -- ablate

disatt:
    cargo run --release -p erel-cli -- disatt

verify:
    uv run python scripts/verify_release.py

manifest:
    uv run python scripts/build_manifest.py

smoke:
    uv run python scripts/smoke_test.py
