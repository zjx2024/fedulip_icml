"""Utilities for single-dataset non-domain-shift experiments."""

from pathlib import Path
from typing import List, Sequence

import numpy as np


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


def dirichlet_partition_by_class(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int,
) -> List[List[int]]:
    if num_clients <= 1:
        return [list(range(len(labels)))]
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    client_indices = [[] for _ in range(num_clients)]

    for cls in np.unique(labels):
        cls_mask = labels == cls
        cls_indices = indices[cls_mask]
        rng.shuffle(cls_indices)
        proportions = rng.dirichlet([alpha] * num_clients)
        counts = (proportions * len(cls_indices)).astype(int)
        while counts.sum() < len(cls_indices):
            counts[rng.integers(num_clients)] += 1
        while counts.sum() > len(cls_indices):
            counts[int(np.argmax(counts))] -= 1
        start = 0
        for client_id, count in enumerate(counts):
            if count <= 0:
                continue
            client_indices[client_id].extend(cls_indices[start:start + count].tolist())
            start += count

    all_indices = list(indices)
    for client_id, subset in enumerate(client_indices):
        if not subset:
            subset.append(int(rng.choice(all_indices)))
        rng.shuffle(subset)
    return client_indices
