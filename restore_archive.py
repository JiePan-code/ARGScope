"""Restore checksum-indexed original sources from an extracted source bundle.

New release utility; no experimental code is executed.
"""
import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path


def native_path(path):
    """Support archived Windows paths beyond the legacy MAX_PATH limit."""
    value = os.path.abspath(path)
    return "\\\\?\\" + value if os.name == "nt" else value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True, help="Directory containing archive/CODE_INDEX.csv")
    p.add_argument("--output", type=Path, required=True, help="New empty destination directory")
    args = p.parse_args()
    bundle = args.bundle.resolve()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; existing files are not overwritten")
    out = args.output.resolve()
    with (bundle / "archive/CODE_INDEX.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        src = (bundle / row["archive_member"]).resolve()
        dst = (out / row["root"] / row["original_relative_path"]).resolve()
        if not src.is_relative_to(bundle) or not dst.is_relative_to(out):
            raise ValueError("Unsafe path in archive index")
        if hashlib.sha256(src.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Archived source checksum mismatch")
        os.makedirs(native_path(dst.parent), exist_ok=True)
        shutil.copyfile(native_path(src), native_path(dst))
    print(f"Restored {len(rows)} original source paths under {out}")


if __name__ == "__main__":
    main()
