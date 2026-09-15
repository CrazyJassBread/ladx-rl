"""Verify named save-state hashes against a manifest."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="state_manifests/ladx_states.json")
    parser.add_argument("--state-dir", default="save_states")
    parser.add_argument("--skip-duplicates", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    failed = False
    for item in manifest["states"]:
        path = Path(item["path"])
        actual = _sha256(path)
        ok = actual == item["sha256"]
        print(f"{'OK' if ok else 'MISMATCH'} {item['id']}: {path}")
        failed |= not ok
    if not args.skip_duplicates:
        failed |= _report_duplicates(Path(args.state_dir))
    return int(failed)


def _report_duplicates(state_dir: Path) -> bool:
    by_hash: dict[str, list[Path]] = {}
    for path in sorted(state_dir.glob("*.state")):
        by_hash.setdefault(_sha256(path), []).append(path)
    duplicates = [paths for paths in by_hash.values() if len(paths) > 1]
    for paths in duplicates:
        print("DUPLICATE " + ", ".join(str(path) for path in paths))
    if not duplicates:
        print(f"OK no duplicate states in {state_dir}")
    return bool(duplicates)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
