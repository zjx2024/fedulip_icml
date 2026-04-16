"""Domain A/B/C/D experiment config."""

DATASETS = [
    "modelnet40",
    "shapenetcore_benchmark",
    "3dimage",
]

DATASET_SPECS = {
    "modelnet40": {
        "train": "modelnet40",
        "test": "modelnet40_test",
        "domains": ["A", "B", "C", "D"],
    },
    "shapenetcore_benchmark": {
        "train": "shapenetcore_benchmark",
        "test": "shapenetcore_benchmark_test",
        "domains": ["A", "B", "C", "D"],
    },
    "3dimage": {
        "train": "3dimage",
        "test": "3dimage_test",
        "domains": ["A", "B", "C"],
    },
}

DOMAIN_LABELS = {
    "A": "clean",
    "B": "sparse",
    "C": "noisy_outlier",
    "D": "partial",
}
