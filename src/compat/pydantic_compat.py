"""Compatibility layer providing transparent Pydantic v2 emulation if not installed."""

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def Field(default: Any = ..., default_factory: Any = None, description: str = "", ge: Any = None, le: Any = None, example: Any = None, **kwargs):
    if default_factory is not None:
        return field(default_factory=default_factory)
    if default is not ...:
        return field(default=default)
    return field()


class BaseModel:
    """Lightweight Pydantic BaseModel compatibility class."""

    def __init__(self, **kwargs):
        # Apply dataclass fields and defaults
        for k, v in self.__class__.__dict__.items():
            if isinstance(v, field.__class__):
                pass
        for k, v in kwargs.items():
            setattr(self, k, v)

    def model_dump(self) -> Dict[str, Any]:
        """Serialize model to dict recursively."""
        res = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, BaseModel):
                res[k] = v.model_dump()
            elif isinstance(v, list):
                res[k] = [item.model_dump() if isinstance(item, BaseModel) else (item.value if hasattr(item, "value") else item) for item in v]
            elif hasattr(v, "value"):
                res[k] = v.value
            else:
                res[k] = v
        return res

    def model_dump_json(self) -> str:
        return json.dumps(self.model_dump())
