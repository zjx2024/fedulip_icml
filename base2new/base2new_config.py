"""Base-to-new splits and dataset specs for cross-dataset experiments."""

DATASET_ORDER = [
    "modelnet40",
    "scanobjectnn",
    "3dimage",
    "mnist3d",
    "mvtec3d",
    "shapenetcore_benchmark",
]

DATASET_SPECS = {
    "modelnet40": {
        "train": "modelnet40",
        "test": "modelnet40_test",
    },
    "scanobjectnn": {
        "train": "scanobjectnn",
        "test": "scanobjectnn_test",
    },
    "3dimage": {
        "train": "3dimage",
        "test": "3dimage_test",
    },
    "mnist3d": {
        "train": "mnist3d",
        "test": "mnist3d_test",
    },
    "mvtec3d": {
        "train": "mvtec3d",
        "test": "mvtec3d_test",
    },
    "shapenetcore_benchmark": {
        "train": "shapenetcore_benchmark",
        "test": "shapenetcore_benchmark_test",
    },
}

ALIAS_MAP = {
    "3dimage": {
        "cone": "3dimage cone",
    },
}

BASE_CLASSES = {
    "modelnet40": [
        "airplane",
        "bathtub",
        "bed",
        "bookshelf",
        "bottle",
        "bowl",
        "car",
        "chair",
        "cone",
        "curtain",
        "desk",
        "door",
        "glass box",
        "guitar",
        "lamp",
        "laptop",
        "night stand",
        "plant",
        "radio",
        "range hood",
        "sink",
        "sofa",
        "stool",
        "table",
        "tent",
        "toilet",
        "tv stand",
        "wardrobe",
    ],
    "scanobjectnn": [
        "bag",
        "bed",
        "bin",
        "box",
        "chair",
        "desk",
        "door",
        "sink",
        "sofa",
        "table",
        "toilet",
    ],
    "3dimage": [
        "cone",
        "cube",
        "sphere",
    ],
    "mnist3d": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
    ],
    "mvtec3d": [
        "bagel",
        "cable_gland",
        "carrot",
        "dowel",
        "peach",
        "potato",
        "rope",
    ],
    "shapenetcore_benchmark": [
        "Airplane",
        "Bag",
        "Car",
        "Chair",
        "Earphone",
        "Guitar",
        "Lamp",
        "Laptop",
        "Mug",
        "Pistol",
        "Table",
    ],
}

NEW_CLASSES = {
    "modelnet40": [
        "bench",
        "cup",
        "dresser",
        "flower pot",
        "keyboard",
        "mantel",
        "monitor",
        "person",
        "piano",
        "stairs",
        "vase",
        "xbox",
    ],
    "scanobjectnn": [
        "cabinet",
        "display",
        "pillow",
        "shelf",
    ],
    "3dimage": [
        "torus",
    ],
    "mnist3d": [
        "0",
        "8",
        "9",
    ],
    "mvtec3d": [
        "cookie",
        "foam",
        "tire",
    ],
    "shapenetcore_benchmark": [
        "Cap",
        "Knife",
        "Motorbike",
        "Rocket",
        "Skateboard",
    ],
}
