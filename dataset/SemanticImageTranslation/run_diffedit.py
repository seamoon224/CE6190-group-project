#!/usr/bin/env python3
"""Lightweight runner that applies DiffEdit to the Semantic Image Translation benchmark."""
import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
import torch

# Ensure we can import both local benchmark helpers (utils/, src/)
# and the top-level diff_edit package.
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
for path in (THIS_DIR, THIS_DIR / "src", PROJECT_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from utils.io import load as load_yaml  # noqa: E402
from src.diffedit_editer import DiffEditEditer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run DiffEdit across the benchmark queries.")
    parser.add_argument("--output", type=Path, default=Path("generated/diffedit"),
                        help="Directory to store generated images.")
    parser.add_argument("--domain", choices=["test", "dev"], default="test",
                        help="Which split of queries to process.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional max number of samples for debugging.")
    parser.add_argument("--device", default="cuda", help="DiffEdit device (cuda/cpu/mps/best).")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="DiffEdit mask sampling iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for DiffEdit.")
    parser.add_argument("--return-blended-mask", action="store_true",
                        help="Save blended mask visualizations as well.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
        args.device = "cpu"

    cfg = load_yaml("global.yaml")
    imagenet_root = Path(cfg["IMAGENET_ROOT"])

    queries = pd.read_csv(THIS_DIR / "dataset" / "queries.csv", index_col=0)
    queries = queries[queries.is_test == (args.domain == "test")]
    if args.limit is not None:
        queries = queries.iloc[:args.limit]

    editer = DiffEditEditer(
        device=args.device,
        num_samples=args.num_samples,
        seed=args.seed,
        return_blended_mask=args.return_blended_mask,
    )

    torch.manual_seed(args.seed)

    for idx, row in queries.iterrows():
        img_path = imagenet_root / Path(row["path"]).name
        if not img_path.exists():
            print(f"[WARN] Skip {idx}: missing image {img_path}")
            continue
        with Image.open(img_path).convert("RGB") as pil_im:
            pil_im = pil_im.copy()

        outputs = editer(pil_im, row.source, row.target)
        if not isinstance(outputs, dict):
            outputs = {"images": outputs}

        for name, pil_image in outputs.items():
            out_dir = args.output / name
            out_dir.mkdir(parents=True, exist_ok=True)
            pil_image.save(out_dir / f"{idx}.png")

        print(f"[INFO] processed idx={idx} ({row.source}->{row.target})")


if __name__ == "__main__":
    main()
