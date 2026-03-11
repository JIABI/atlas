
from __future__ import annotations
from typing import Any, Callable

class Registry(dict):
    """Simple string-to-callable registry."""
    def register(self, name: str, obj: Callable[..., Any]) -> None:
        self[name]=obj

    def build(self, name: str, **kwargs: Any) -> Any:
        if name not in self:
            raise KeyError(f"Unknown key: {name}")
        return self[name](**kwargs)

REGISTRY = Registry()
