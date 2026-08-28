from typing import Any, Callable, Dict, Optional


class Registry:
    def __init__(self, name: str) -> None:
        self._name: str = name
        self._registry: Dict[str, Callable[..., Any]] = {}

    def __len__(self) -> int:
        return len(self._registry)

    @property
    def name(self) -> str:
        return self._name

    @property
    def registry(self) -> Dict[str, Callable[..., Any]]:
        return self._registry

    def register(
            self,
            name: Optional[str] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
            _name = name or obj.__name__
            if _name in self._registry:
                raise ValueError(f"'{_name}' is already registered.")
            self._registry[_name] = obj
            return obj
        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._registry:
            raise KeyError(f"'{name}' not found in registry.")
        return self._registry[name]


MODEL = Registry(name="model")
DATASET = Registry(name="dataset")
CRITERION = Registry(name="criterion")
