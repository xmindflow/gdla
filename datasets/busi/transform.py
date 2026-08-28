from torch import Tensor
from typing import Tuple, Union

import albumentations as A
import numpy as np
import torch
import torchvision.transforms.v2 as TF


class Transform:
    def __init__(
            self,
            dtype: torch.dtype,
            img_size: Union[int, Tuple[int, int]],
            augment: bool,
    ) -> None:
        self.img_size = img_size
        self.dtype = dtype
        self.augment = augment

        self.img_resize = TF.Resize(
            size=img_size,
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )

        self.msk_resize = TF.Resize(
            size=img_size,
            interpolation=TF.InterpolationMode.NEAREST,
        )

        self.albumentations = A.Compose([
            A.Rotate(limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        ])

    def __call__(self, img: np.ndarray, msk: np.ndarray) -> Tuple[Tensor, Tensor]:
        img = img.transpose((2, 0, 1))
        msk = msk[np.newaxis, ...]
        img = self.img_resize(torch.from_numpy(img))
        msk = self.msk_resize(torch.from_numpy(msk))

        if self.augment:
            img = img.permute(1, 2, 0).numpy()
            msk = msk.permute(1, 2, 0).numpy()
            augmented = self.albumentations(image=img, mask=msk)
            img = augmented["image"]
            msk = augmented["mask"]
            img = torch.from_numpy(img).permute(2, 0, 1)
            msk = torch.from_numpy(msk).permute(2, 0, 1)

        img = img.nan_to_num(nan=0.0).to(dtype=torch.float32)
        msk = msk.nan_to_num(nan=0.0).to(dtype=torch.float32)

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        msk = (msk - msk.min()) / (msk.max() - msk.min() + 1e-8)

        img = img.nan_to_num(nan=0, posinf=1e6, neginf=-1e6)
        msk = msk.nan_to_num(nan=0, posinf=1e6, neginf=-1e6)

        msk = msk.to(dtype=torch.int64).squeeze(0)

        return img, msk
