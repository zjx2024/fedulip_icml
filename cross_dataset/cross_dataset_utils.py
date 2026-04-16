"""Utilities for cross-dataset generalization experiments."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def apply_alias(dataset_name: str, class_name: str, alias_map: Dict[str, Dict[str, str]]) -> str:
    return alias_map.get(dataset_name, {}).get(class_name, class_name)


def canonicalize_class_name(class_name: str) -> str:
    name = class_name.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(name.split())


def to_global_name(dataset_name: str, class_name: str, alias_map: Dict[str, Dict[str, str]]) -> str:
    return canonicalize_class_name(apply_alias(dataset_name, class_name, alias_map))


def unique_in_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def dataset_class_names(dataset) -> List[str]:
    if hasattr(dataset, "shape_names") and dataset.shape_names:
        return list(dataset.shape_names)
    label_names = resolve_label_names(dataset)
    return unique_in_order(label_names)


def assign_classes_to_clients(
    classes: Sequence[str],
    num_clients: int,
    max_classes_per_client: int,
    seed: int,
) -> List[List[str]]:
    if num_clients * max_classes_per_client < len(classes):
        raise ValueError("Not enough client capacity for all training classes.")
    rng = np.random.default_rng(seed)
    classes = list(classes)
    rng.shuffle(classes)
    assignments = [[] for _ in range(num_clients)]
    for idx, name in enumerate(classes):
        assignments[idx % num_clients].append(name)
    return assignments


def _flatten_label(label) -> int:
    if isinstance(label, (list, tuple, np.ndarray)):
        return int(np.array(label).reshape(-1)[0])
    return int(label)


def resolve_label_names(dataset) -> List[str]:
    if hasattr(dataset, "labels") and hasattr(dataset, "shape_names"):
        return [dataset.shape_names[int(lbl)] for lbl in dataset.labels]
    if hasattr(dataset, "list_of_labels") and hasattr(dataset, "shape_names"):
        return [dataset.shape_names[_flatten_label(lbl)] for lbl in dataset.list_of_labels]
    if hasattr(dataset, "file_list") and hasattr(dataset, "_label_from_path"):
        if getattr(dataset, "label_source", "dir") == "dir":
            return [dataset._label_from_path(Path(path)) for path in dataset.file_list]
    names = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if isinstance(sample, (list, tuple)) and len(sample) >= 3:
            names.append(sample[2])
        elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
            names.append(str(sample[1]))
        else:
            names.append(str(idx))
    return names


class GlobalClassDataset(Dataset):
    def __init__(
        self,
        dataset,
        dataset_name: str,
        class_to_idx: Dict[str, int],
        alias_map: Dict[str, Dict[str, str]],
        allowed_classes: Optional[Iterable[str]] = None,
        max_samples: int = 0,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.class_to_idx = class_to_idx
        self.alias_map = alias_map
        allowed = set(allowed_classes) if allowed_classes is not None else None

        label_names = resolve_label_names(dataset)
        self.indices = []
        self.global_labels = []
        self.global_names = []
        for idx, name in enumerate(label_names):
            global_name = to_global_name(dataset_name, name, alias_map)
            if allowed is not None and global_name not in allowed:
                continue
            if global_name not in class_to_idx:
                raise KeyError(f"Global class not found: {global_name}")
            self.indices.append(idx)
            self.global_labels.append(class_to_idx[global_name])
            self.global_names.append(global_name)

        if max_samples and len(self.indices) > max_samples:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(len(self.indices), size=max_samples, replace=False)
            chosen = sorted(chosen.tolist())
            self.indices = [self.indices[i] for i in chosen]
            self.global_labels = [self.global_labels[i] for i in chosen]
            self.global_names = [self.global_names[i] for i in chosen]

        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.global_labels):
            self.class_to_indices[label].append(idx)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        sample = self.dataset[self.indices[idx]]
        points = sample[0] if isinstance(sample, (list, tuple)) else sample
        return points, self.global_labels[idx], self.global_names[idx]
