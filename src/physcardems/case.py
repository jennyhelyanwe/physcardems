"""case.py: load a case configuration. Relative paths resolve against the case folder."""
from __future__ import annotations
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    root: Path
    cfg: dict

    def get(self, *keys, default=None):
        value = self.cfg
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                return default
            value = value[k]
        return value

    def path(self, *keys) -> Path:
        value = self.get(*keys)
        if value is None:
            raise KeyError(f"{'.'.join(keys)} not set in case {self.name}")
        p = Path(value)
        return p if p.is_absolute() else self.root / p


def load_case(path) -> Case:
    path = Path(path).resolve()
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    return Case(name=cfg.get("name", path.parent.name), root=path.parent, cfg=cfg)