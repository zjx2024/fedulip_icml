"""Config for single-dataset non-domain-shift experiments."""

DATASET_SPECS = {
    "modelnet40": {
        "train": "modelnet40",
        "test": "modelnet40_test",
        "prompt": "modelnet40_64",
    },
    "scanobjectnn": {
        "train": "scanobjectnn",
        "test": "scanobjectnn_test",
        "prompt": "scanobjectnn_64",
    },
    "3dimage": {
        "train": "3dimage",
        "test": "3dimage_test",
        "prompt": "modelnet40_64",
    },
    "mnist3d": {
        "train": "mnist3d",
        "test": "mnist3d_test",
        "prompt": "modelnet40_64",
    },
    "mvtec3d": {
        "train": "mvtec3d",
        "test": "mvtec3d_test",
        "prompt": "modelnet40_64",
    },
    "shapenetcore_benchmark": {
        "train": "shapenetcore_benchmark",
        "test": "shapenetcore_benchmark_test",
        "prompt": "modelnet40_64",
    },
}
