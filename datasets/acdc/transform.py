from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import random
import torch
from scipy import ndimage
from scipy.ndimage import zoom


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()

    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)

    return image, label


class Transform:
    def __init__(
            self,
            img_size: Union[List[int], Tuple[int]],
            augment: bool,
            p: Optional[float] = None,
            degree: Optional[Union[List[int], Tuple[int]]] = None,
    ) -> None:
        self.img_size = img_size
        self.augment = augment
        self.p = p
        self.degree = degree

    def __call__(self, image: np.ndarray, label: np.ndarray) -> Dict[str, Tensor]:
        if self.augment:
            if random.random() > self.p:
                image, label = random_rot_flip(image, label)
            elif random.random() > self.p:
                image, label = random_rotate(image, label)

            h, w = image.shape
            if h != self.img_size[0] or w != self.img_size[1]:
                image = zoom(
                    input=image,
                    zoom=(self.img_size[0] / h, self.img_size[1] / w),
                    order=3,
                )
                label = zoom(
                    input=label,
                    zoom=(self.img_size[0] / h, self.img_size[1] / w),
                    order=0,
                )
        else:
            _, h, w = image.shape
            if h != self.img_size[0] or w != self.img_size[1]:
                image = zoom(
                    input=image,
                    zoom=(1.0, self.img_size[0] / h, self.img_size[1] / w),
                    order=3,
                )

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32)).to(dtype=torch.int64)

        return image, label
