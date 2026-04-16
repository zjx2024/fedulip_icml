"""Utilities for domain A/B/C/D experiments."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset



def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0])
    return vec / norm


def resample_points(points: np.ndarray, target_npoints: int, rng: np.random.Generator) -> np.ndarray:
    npoints = points.shape[0]
    if npoints == target_npoints:
        return points
    if npoints > target_npoints:
        idx = rng.choice(npoints, size=target_npoints, replace=False)
        return points[idx]
    idx = rng.choice(npoints, size=target_npoints, replace=True)
    return points[idx]


def apply_sparse(points: np.ndarray, sparse_npoints: int, rng: np.random.Generator) -> np.ndarray:
    sparse_npoints = int(min(sparse_npoints, points.shape[0]))
    sparse = resample_points(points, sparse_npoints, rng)
    return sparse


def apply_noisy_outlier(
    points: np.ndarray,
    outlier_ratio: float,
    drop_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    npoints, dim = points.shape
    drop_n = max(1, int(round(npoints * drop_ratio)))
    outlier_n = max(1, int(round(npoints * outlier_ratio)))
    keep_n = max(1, npoints - drop_n)
    keep_idx = rng.choice(npoints, size=keep_n, replace=False)
    keep = points[keep_idx]

    mins = points[:, :3].min(axis=0)
    maxs = points[:, :3].max(axis=0)
    outliers = rng.uniform(mins, maxs, size=(outlier_n, 3))
    if dim > 3:
        extra = np.zeros((outlier_n, dim - 3), dtype=points.dtype)
        outliers = np.concatenate([outliers, extra], axis=1)
    outliers = outliers.astype(points.dtype)

    combined = np.concatenate([keep, outliers], axis=0)
    rng.shuffle(combined)
    return combined


def apply_plane_cut(
    points: np.ndarray,
    keep_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    npoints = points.shape[0]
    keep_n = max(1, int(round(npoints * keep_ratio)))
    center = points[:, :3].mean(axis=0)
    normal = random_unit_vector(rng)
    proj = (points[:, :3] - center) @ normal
    idx = np.argsort(proj)[-keep_n:]
    return points[idx]




def split_indices_evenly(
    indices: Sequence[int],
    domains: Sequence[str],
    seed: int,
) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    idx = np.array(list(indices))
    rng.shuffle(idx)
    counts = [len(idx) // len(domains)] * len(domains)
    for i in range(len(idx) % len(domains)):
        counts[i] += 1
    splits = {}
    start = 0
    for domain, count in zip(domains, counts):
        splits[domain] = idx[start:start + count].tolist()
        start += count
    return splits


def assign_clients_to_domains(num_clients: int, domains: Sequence[str]) -> List[str]:
    assignments = []
    for idx in range(num_clients):
        assignments.append(domains[idx % len(domains)])
    return assignments


def split_indices_randomly(
    indices: Sequence[int],
    num_clients: int,
    seed: int,
) -> List[List[int]]:
    if num_clients <= 1:
        return [list(indices)]
    rng = np.random.default_rng(seed)
    idx = np.array(list(indices))
    rng.shuffle(idx)
    counts = [len(idx) // num_clients] * num_clients
    for i in range(len(idx) % num_clients):
        counts[i] += 1
    splits = []
    start = 0
    for count in counts:
        splits.append(idx[start:start + count].tolist())
        start += count
    return splits


def resolve_labels(dataset) -> List[int]:
    if hasattr(dataset, "labels"):
        return [int(x) for x in dataset.labels]
    if hasattr(dataset, "list_of_labels"):
        labels = []
        for lbl in dataset.list_of_labels:
            if isinstance(lbl, (list, tuple, np.ndarray)):
                labels.append(int(np.array(lbl).reshape(-1)[0]))
            else:
                labels.append(int(lbl))
        return labels
    if hasattr(dataset, "file_list") and hasattr(dataset, "_label_from_path") and hasattr(dataset, "classes"):
        labels = []
        for path in dataset.file_list:
            label_name = dataset._label_from_path(Path(path))
            labels.append(int(dataset.classes[label_name]))
        return labels
    labels = []
    for idx in range(len(dataset)):
        _, label, _ = dataset[idx]
        labels.append(int(label))
    return labels


class DomainDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        indices: Sequence[int],
        domain: str,
        target_npoints: int,
        sparse_npoints: int,
        outlier_ratio: float,
        drop_ratio: float,
        partial_keep_ratio: float,
        seed: int,
        label_names: Sequence[str] | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.domain = domain
        self.target_npoints = target_npoints
        self.sparse_npoints = sparse_npoints
        self.outlier_ratio = outlier_ratio
        self.drop_ratio = drop_ratio
        self.partial_keep_ratio = partial_keep_ratio
        self.seed = int(seed)
        self.label_names = list(label_names) if label_names is not None else None
        self.class_to_indices = defaultdict(list)
        for local_idx, base_idx in enumerate(self.indices):
            label = self._label_for_index(base_idx)
            self.class_to_indices[int(label)].append(local_idx)

    def _label_for_index(self, base_idx: int) -> int:
        if hasattr(self.base_dataset, "labels"):
            return int(self.base_dataset.labels[base_idx])
        if hasattr(self.base_dataset, "list_of_labels"):
            label = self.base_dataset.list_of_labels[base_idx]
            if isinstance(label, (list, tuple, np.ndarray)):
                return int(np.array(label).reshape(-1)[0])
            return int(label)
        if hasattr(self.base_dataset, "file_list") and hasattr(self.base_dataset, "_label_from_path"):
            path = self.base_dataset.file_list[base_idx]
            name = self.base_dataset._label_from_path(Path(path))
            return int(self.base_dataset.classes[name])
        _, label, _ = self.base_dataset[base_idx]
        return int(label)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        points, label, label_name = self.base_dataset[base_idx]
        points = np.asarray(points)
        rng = np.random.default_rng(self.seed + int(base_idx))

        if self.domain == "A":
            points = resample_points(points, self.target_npoints, rng)
        elif self.domain == "B":
            sparse = apply_sparse(points, self.sparse_npoints, rng)
            points = resample_points(sparse, self.target_npoints, rng)
        elif self.domain == "C":
            points = apply_noisy_outlier(points, self.outlier_ratio, self.drop_ratio, rng)
            points = resample_points(points, self.target_npoints, rng)
        elif self.domain == "D":
            cut = apply_plane_cut(points, self.partial_keep_ratio, rng)
            points = resample_points(cut, self.target_npoints, rng)
        else:
            points = resample_points(points, self.target_npoints, rng)

        points = torch.from_numpy(points).float()
        if self.label_names is not None:
            label_name = self.label_names[int(label)]
        return points, int(label), label_name
