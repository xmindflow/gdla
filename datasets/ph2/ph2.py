from functools import partial
from typing import Callable, Dict, List, Tuple

import numpy as np
import os
import torch
from .transform import Transform
from registry import DATASET
from torch.utils.data import Dataset
from PIL import Image


class PH2(Dataset):
    def __init__(self, data_dir: str, transform=None) -> None:
        self.img, self.msk = self._load(data_dir=data_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img)

    def __getitem__(self, idx: int):
        img = self.img[idx]
        msk = self.msk[idx]

        if self.transform is not None:
            img, msk = self.transform(img=img, msk=msk)

        return {"image": img, "label": msk}

    def _load(self, data_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        sub_ids = os.listdir(path=data_dir)
        sub_ids.sort()

        imgs = []
        msks = []

        for sub_id in sub_ids:
            img_path = os.path.join(data_dir, sub_id, f"{sub_id}.bmp")
            msk_path = os.path.join(data_dir, sub_id, f"{sub_id}_lesion.bmp")

            img = Image.open(fp=img_path).convert(mode="RGB")
            img = np.array(object=img).astype(dtype=np.uint8)
            msk = Image.open(fp=msk_path).convert(mode='L')
            msk = np.array(object=msk).astype(dtype=np.uint8)

            imgs.append(img)
            msks.append(msk)

        return imgs, msks


@DATASET.register(name="ph2")
def build_dataset(cfg) -> Dict[str, Callable[[], Dataset]]:
    train_transform = Transform(
        dtype=torch.float32,
        img_size=cfg.DATA.IMG_SIZE[1:],
        augment=cfg.DATA.TRANSFORM.AUGMENT,
    )
    val_transform = Transform(
        dtype=torch.float32,
        img_size=cfg.DATA.IMG_SIZE[1:],
        augment=False,
    )
    test_transform = Transform(
        dtype=torch.float32,
        img_size=cfg.DATA.IMG_SIZE[1:],
        augment=False,
    )

    return {
        'train': partial(PH2, data_dir=cfg.DATA.TRAIN_DIR, transform=train_transform),
        'val': partial(PH2, data_dir=cfg.DATA.VAL_DIR, transform=val_transform),
        'test': partial(PH2, data_dir=cfg.DATA.TEST_DIR, transform=test_transform)
    }
