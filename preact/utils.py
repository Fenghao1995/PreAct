from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = "auto") -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device=gpu, but CUDA is not available.")
        return torch.device("cuda:0")
    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install pyyaml to read YAML configs, or pass JSON.") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping config in {path}")
    return data


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(dict(out[key]), value)
        else:
            out[key] = value
    return out


def load_merged_config(default_config: str | Path, task_config: str | Path) -> dict[str, Any]:
    return deep_update(read_text_config(default_config), read_text_config(task_config))


def experiment_slug(config: Mapping[str, Any]) -> str:
    task_name = str(config.get("task", {}).get("name", "experiment"))
    kind = str(config.get("policy", {}).get("kind", "policy")).lower()
    family = "fm" if kind in {"flow", "fm", "flow_matching"} else "dp"
    return f"{task_name}_{family}"


def as_float_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().float()
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)
