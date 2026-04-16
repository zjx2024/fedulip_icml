"""Cross-dataset generalization experiment config."""

TRAIN_DATASET = "modelnet40"

EVAL_DATASETS = [
    "scanobjectnn",
    "shapenetcore_benchmark",
    "mvtec3d",
    "mnist3d",
    "3dimage",
]

DATASET_SPECS = {
    "modelnet40": {
        "train": "modelnet40",
        "test": "modelnet40_test",
    },
    "scanobjectnn": {
        "test": "scanobjectnn_test",
    },
    "shapenetcore_benchmark": {
        "test": "shapenetcore_benchmark_test",
    },
    "mvtec3d": {
        "test": "mvtec3d_test",
    },
    "mnist3d": {
        "test": "mnist3d_test",
    },
    "3dimage": {
        "test": "3dimage_test",
    },
}

ALIAS_MAP = {
    "3dimage": {
        "cone": "3dimage cone",
    },
}
