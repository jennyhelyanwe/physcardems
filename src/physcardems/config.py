"""config.py: load a simulation configuration (.toml or .json) and its case."""
from __future__ import annotations
import json
import subprocess
import tomllib
from pathlib import Path

from physcardems.case import load_case


def load_config(path):
    path = Path(path).resolve()
    if path.suffix == ".toml":
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    elif path.suffix == ".json":
        cfg = json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    case_path = Path(cfg["case"])
    if not case_path.is_absolute():
        case_path = path.parent / case_path
    return cfg, load_case(case_path)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def save_config(cfg, path, **extra):
    """Write the configuration actually used, with provenance, as JSON."""
    record = {"config": cfg, "git_commit": git_commit(), **extra}
    Path(path).write_text(json.dumps(record, indent=2))