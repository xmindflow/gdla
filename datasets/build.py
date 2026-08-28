from collections.abc import Mapping
from registry import DATASET
from torch.utils.data import Dataset
from typing import Callable, Dict, Iterator


class LazySplits(Mapping):
    """Split name -> `Dataset`, constructing each split on first access.

    Splits are built lazily so that a script only ever touches the directories it
    actually uses: `test_model.py` reads `DATA.TEST_DIR` alone, which means a
    released checkpoint can be evaluated without the training data on disk.
    """

    def __init__(self, builders: Dict[str, Callable[[], Dataset]]) -> None:
        self._builders = builders
        self._cache: Dict[str, Dataset] = {}

    def __getitem__(self, split: str) -> Dataset:
        if split not in self._cache:
            if split not in self._builders:
                raise KeyError(f"Unknown split `{split}`. "
                               f"Available: {list(self._builders.keys())}")
            self._cache[split] = self._builders[split]()
        return self._cache[split]

    def __iter__(self) -> Iterator[str]:
        return iter(self._builders)

    def __len__(self) -> int:
        return len(self._builders)


def build_dataset(cfg) -> LazySplits:
    name = cfg.DATA.NAME
    if name not in DATASET.registry:
        raise KeyError(f"Dataset `{name}` not found in registry. "
                       f"Available: {list(DATASET.registry.keys())}")
    return LazySplits(DATASET.registry[name](cfg=cfg))
