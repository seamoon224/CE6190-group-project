#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reorganize ImageNet val images according to dataset/queries.csv."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("dataset/queries.csv"),
        help="CSV containing path/source_id columns.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Root folder storing images grouped by `source` (spaces replaced by underscores).",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        required=True,
        help="Destination root that will receive <synset>/<filename> structure.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (default copy).",
    )
    parser.add_argument(
        "--missing-report",
        type=Path,
        default=Path("missing_images.txt"),
        help="Path to save list of missing images.",
    )
    return parser.parse_args()


def locate_file(source_dir: Path, source_id: int) -> Path | None:
    candidates = [
        source_dir / f"{source_id}.jpg",
        source_dir / f"{source_id}.jpeg",
        source_dir / f"{source_id:03d}.jpg",
        source_dir / f"{source_id:03d}.jpeg",
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    sid = str(source_id)
    for file in source_dir.glob("*"):
        stem = file.stem.lower()
        if stem == sid or stem == sid.zfill(3):
            return file
    return None


def main():
    args = parse_args()
    queries = pd.read_csv(args.queries, index_col=0)

    args.dest_root.mkdir(parents=True, exist_ok=True)
    missing = []
    copied = 0

    for idx, row in queries.iterrows():
        synset_path = Path(row["path"])
        filename = synset_path.name
        source_dir_name = str(row["source"]).replace(" ", "_").replace("-", "_").lower()
        source_dir = args.source_root / source_dir_name

        if not source_dir.exists():
            missing.append((idx, str(source_dir), filename, "source_dir_missing"))
            continue

        src_file = locate_file(source_dir, row["source_id"])
        if src_file is None:
            missing.append((idx, str(source_dir), f"{row['source_id']}.jpg", "file_not_found"))
            continue

        dest_file = args.dest_root / synset_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        if dest_file.exists():
            continue

        if args.move:
            shutil.move(str(src_file), dest_file)
        else:
            shutil.copy2(str(src_file), dest_file)
        copied += 1

    if missing:
        args.missing_report.parent.mkdir(parents=True, exist_ok=True)
        with args.missing_report.open("w") as f:
            for idx, src_dir, filename, reason in missing:
                f.write(f"{idx},{src_dir},{filename},{reason}\n")

    print(f"Processed {len(queries)} rows. Copied/moved {copied}. Missing {len(missing)}.")
    if missing:
        print(f"See {args.missing_report} for details.")


if __name__ == "__main__":
    main()
