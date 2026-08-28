import numpy as np


class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()
        return

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum/self.count, self.sum)
        return
