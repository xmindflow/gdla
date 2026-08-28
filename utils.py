from torch import Tensor

import torch


def one_hot(labels: Tensor, n_classes: int, dim: int) -> Tensor:
    assert labels.size(dim) == 1, "The specified dimension must have size 1."

    size = list(labels.size())
    size[dim] = n_classes

    o = torch.zeros(size=size, dtype=torch.float32, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels
