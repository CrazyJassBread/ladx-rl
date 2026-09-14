"""Verify named save-state hashes against a manifest."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="state_manifests/ladx_states.json")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    failed = False
    for item in manifest["states"]:
        path = Path(item["path"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        ok = actual == item["sha256"]
        print(f"{'OK' if ok else 'MISMATCH'} {item['id']}: {path}")
        failed |= not ok
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
