#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preact.alignment import validate_paper_config
from preact.data import load_transition_dataset, make_synthetic_dataset
from preact.training import Stage1TrainConfig, make_world_model_from_task, train_stage1_lewm
from preact.utils import experiment_slug, load_merged_config, resolve_device, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage I: train PreAct LeWM.")
    parser.add_argument(
        "--task-config", default=str(PROJECT_ROOT / "configs/experiments/sorting_fm.yaml")
    )
    parser.add_argument("--base-config", default=str(PROJECT_ROOT / "configs/preact.yaml"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--data", default=None, help="Override paths.lewm_data.")
    parser.add_argument("--encoder-type", default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--smoke", action="store_true", help="Use synthetic data instead of real paths."
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_merged_config(args.base_config, args.task_config)
    validate_paper_config(cfg)
    task = cfg["task"]
    set_seed(int(cfg.get("seed", 7)))
    device = resolve_device(args.device or cfg.get("device", "auto"))

    stage1_cfg = _dataclass_from_config(Stage1TrainConfig, cfg.get("stage1", {}))
    if args.epochs is not None:
        stage1_cfg.epochs = args.epochs
    if args.batch_size is not None:
        stage1_cfg.batch_size = args.batch_size

    if args.smoke:
        dataset = make_synthetic_dataset(
            num_samples=args.max_samples or 256,
            obs_shape=task["obs_shape"],
            obs_history=int(task.get("obs_history", 1)),
            action_horizon=int(task["action_horizon"]),
            action_dim=int(task["action_dim"]),
            seed=int(cfg.get("seed", 7)),
        )
    else:
        data_path = (
            args.data if args.data is not None else cfg.get("paths", {}).get("lewm_data", "")
        )
        if not data_path:
            raise SystemExit("paths.lewm_data is empty. Fill the task config or pass --smoke.")
        dataset = load_transition_dataset(
            data_path,
            max_samples=args.max_samples or cfg.get("stage1", {}).get("max_samples"),
        )

    model = make_world_model_from_task(
        task,
        encoder_type=args.encoder_type,
        world_model_cfg=cfg.get("world_model", {}),
    )
    out_dir = (
        Path(args.out) if args.out else PROJECT_ROOT / "outputs" / experiment_slug(cfg) / "stage1"
    )
    _, metrics = train_stage1_lewm(
        dataset=dataset,
        model=model,
        cfg=stage1_cfg,
        device=device,
        out_dir=out_dir,
    )
    write_json(out_dir / "resolved_config.json", cfg)
    print(json.dumps({"checkpoint": str(out_dir / "lewm.pt"), "metrics": metrics}, indent=2))


def _dataclass_from_config(cls, values: dict):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in dict(values or {}).items() if key in allowed})


if __name__ == "__main__":
    main()
