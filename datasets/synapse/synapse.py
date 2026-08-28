from functools import partial
from typing import Callable, Dict, List, Tuple

import h5py
import numpy as np
import os
from .transform import Transform
from registry import DATASET
from torch.utils.data import Dataset


class Synapse(Dataset):
    def __init__(self, data_dir: str, transform=None) -> None:
        self.image, self.label = self._load(data_dir=data_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image)

    def __getitem__(self, idx: int):
        image = self.image[idx]
        label = self.label[idx]

        if self.transform is not None:
            image, label = self.transform(image=image, label=label)

        return {"image": image, "label": label}

    def _load(self, data_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        filenames = os.listdir(path=data_dir)
        filenames.sort()

        image = []
        label = []

        for filename in filenames:
            file_path = os.path.join(data_dir, filename)

            if file_path.endswith(".npz"):
                data = np.load(file=file_path)
                image.append(data["image"])
                label.append(data["label"])
            elif file_path.endswith(".npy.h5"):
                data = h5py.File(file_path)
                image.append(data["image"][...])
                label.append(data["label"][...])

        return image, label


@DATASET.register(name="synapse")
def build_dataset(cfg) -> Dict[str, Callable[[], Dataset]]:
    train_transform = Transform(
        img_size=cfg.DATA.IMG_SIZE[1:],
        augment=cfg.DATA.TRANSFORM.AUGMENT,
        p=cfg.DATA.TRANSFORM.P,
        degree=cfg.DATA.TRANSFORM.DEGREE,
    )
    val_transform = Transform(
        img_size=cfg.DATA.IMG_SIZE[1:],
        augment=False,
    )
    test_transform = Transform(
        img_size=cfg.DATA.IMG_SIZE[1:],
        augment=False,
    )
    return {
        'train': partial(Synapse, data_dir=cfg.DATA.TRAIN_DIR, transform=train_transform),
        'train-no': partial(Synapse, data_dir=cfg.DATA.TRAIN_DIR, transform=None),
        'val': partial(Synapse, data_dir=cfg.DATA.VAL_DIR, transform=val_transform),
        'test': partial(Synapse, data_dir=cfg.DATA.TEST_DIR, transform=test_transform)
    }
