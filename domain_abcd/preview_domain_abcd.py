#!/usr/bin/env python3
"""Preview Domain A/B/C/D transforms on a single point cloud sample."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
ULIP_DIR = (BASE_DIR / ".." / "ULIP").resolve()
if str(ULIP_DIR) not in sys.path:
    sys.path.insert(0, str(ULIP_DIR))

os.chdir(ULIP_DIR)

from utils.tokenizer import SimpleTokenizer  # noqa: E402
from utils.utils import get_dataset  # noqa: E402

from domain_abcd_config import DATASET_SPECS  # noqa: E402
from domain_abcd_utils import (  # noqa: E402
    apply_noisy_outlier,
    apply_plane_cut,
    apply_sparse,
    resample_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview domain A/B/C/D transforms")
    parser.add_argument(
        "--dataset",
        default="modelnet40",
        choices=sorted(DATASET_SPECS.keys()),
        help="dataset to preview",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val"],
        help="dataset split",
    )
    parser.add_argument("--index", type=int, default=-1, help="sample index (default: random)")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="rng seed base (omit for random sample each run)",
    )
    parser.add_argument("--npoints", type=int, default=1024, help="target points")
    parser.add_argument("--sparse-npoints", type=int, default=512, help="sparse points for domain B")
    parser.add_argument("--outlier-ratio", type=float, default=0.05, help="outlier ratio for domain C")
    parser.add_argument("--drop-ratio", type=float, default=0.05, help="drop ratio for domain C")
    parser.add_argument("--partial-keep-ratio", type=float, default=0.7, help="keep ratio for domain D")
    parser.add_argument("--prompt", default="modelnet40_64", help="prompt key for dataset loader")
    parser.add_argument("--use-height", action="store_true", help="use height in dataset loader")
    parser.add_argument("--model", default="ULIP_PointBERT", help="model name for dataset loader")
    parser.add_argument("--point-size", type=float, default=1.0, help="scatter point size")
    parser.add_argument("--font-size", type=int, default=14, help="base font size")
    parser.add_argument("--output", default="", help="output image path (png)")
    parser.add_argument("--show", action="store_true", help="show the figure interactively")
    return parser.parse_args()


def build_dataset_args(args: argparse.Namespace, train_name: str, test_name: str) -> argparse.Namespace:
    data_args = argparse.Namespace()
    data_args.pretrain_dataset_name = train_name
    data_args.validate_dataset_name = test_name
    data_args.pretrain_dataset_prompt = args.prompt
    data_args.validate_dataset_prompt = args.prompt
    data_args.model = args.model
    data_args.npoints = args.npoints
    data_args.use_height = args.use_height
    return data_args


def select_index(length: int, index: int, seed: int | None) -> int:
    if index >= 0:
        return min(index, length - 1)
    rng = np.random.default_rng(seed)
    return int(rng.integers(0, length))


def apply_domain(
    points: np.ndarray,
    domain: str,
    seed: int,
    target_npoints: int,
    sparse_npoints: int,
    outlier_ratio: float,
    drop_ratio: float,
    partial_keep_ratio: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if domain == "A":
        out = resample_points(points, target_npoints, rng)
    elif domain == "B":
        sparse = apply_sparse(points, sparse_npoints, rng)
        out = resample_points(sparse, target_npoints, rng)
    elif domain == "C":
        noisy = apply_noisy_outlier(points, outlier_ratio, drop_ratio, rng)
        out = resample_points(noisy, target_npoints, rng)
    elif domain == "D":
        cut = apply_plane_cut(points, partial_keep_ratio, rng)
        out = resample_points(cut, target_npoints, rng)
    else:
        out = resample_points(points, target_npoints, rng)
    return out


def collect_domains(args: argparse.Namespace, points: np.ndarray, base_idx: int) -> Dict[str, np.ndarray]:
    spec = DATASET_SPECS[args.dataset]
    domains = list(spec["domains"])
    seed_base = args.seed if args.seed is not None else 1000
    results: Dict[str, np.ndarray] = {}
    for domain in domains:
        results[domain] = apply_domain(
            points,
            domain,
            seed=seed_base + int(base_idx),
            target_npoints=args.npoints,
            sparse_npoints=args.sparse_npoints,
            outlier_ratio=args.outlier_ratio,
            drop_ratio=args.drop_ratio,
            partial_keep_ratio=args.partial_keep_ratio,
        )
    return results


def plot_domains(
    domain_points: Dict[str, np.ndarray],
    title: str,
    point_size: float,
    font_size: int,
    output: str,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    domains = list(domain_points.keys())
    if len(domains) == 4:
        rows, cols = 2, 2
    else:
        rows, cols = 1, len(domains)

    fig = plt.figure(figsize=(4 * cols, 4 * rows))

    all_points = np.concatenate(list(domain_points.values()), axis=0)
    mins = all_points[:, :3].min(axis=0)
    maxs = all_points[:, :3].max(axis=0)
    center = (mins + maxs) / 2.0
    span = (maxs - mins).max() / 2.0
    lims = [(center[i] - span, center[i] + span) for i in range(3)]

    for idx, domain in enumerate(domains, start=1):
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        pts = domain_points[domain][:, :3]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size)
        ax.set_title(f"Domain {domain}", fontsize=font_size)
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_zlim(*lims[2])
        ax.set_axis_off()

    fig.suptitle(title, y=0.98, fontsize=font_size + 2)
    fig.tight_layout()

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    data_args = build_dataset_args(args, spec["train"], spec["test"])
    tokenizer = SimpleTokenizer()
    dataset = get_dataset(None, tokenizer, data_args, args.split)

    idx = select_index(len(dataset), args.index, args.seed)
    points, label, label_name = dataset[idx]
    points = np.asarray(points)
    domain_points = collect_domains(args, points, idx)

    title = f"{args.dataset} {args.split} idx={idx} label={label_name}"
    output = args.output
    if not output and not args.show:
        output_dir = BASE_DIR / "preview_outputs"
        output = str(output_dir / f"domain_abcd_{args.dataset}_{args.split}_idx{idx}.png")

    plot_domains(domain_points, title, args.point_size, args.font_size, output, args.show)
    if output:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
