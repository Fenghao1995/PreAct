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
from preact.training import Stage2TrainConfig, policy_kwargs_from_task, train_stage2_policy_and_rnd
from preact.utils import experiment_slug, load_merged_config, resolve_device, set_seed, write_json
from preact.world_model import load_world_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage II: train PreAct policy and DRS-O.")
    parser.add_argument(
        "--task-config", default=str(PROJECT_ROOT / "configs/experiments/sorting_fm.yaml")
    )
    parser.add_argument("--base-config", default=str(PROJECT_ROOT / "configs/preact.yaml"))
    parser.add_argument("--lewm-checkpoint", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--data", default=None, help="Override paths.expert_data.")
    parser.add_argument("--policy-kind", default=None, choices=["flow", "diffusion"])
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--smoke", action="store_true", help="Use synthetic data matching the LeWM checkpoint."
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_merged_config(args.base_config, args.task_config)
    if args.policy_kind is not None:
        cfg.setdefault("policy", {})["kind"] = args.policy_kind
        cfg.setdefault("stage2", {})["policy_kind"] = args.policy_kind
    validate_paper_config(cfg)
    task = cfg["task"]
    set_seed(int(cfg.get("seed", 7)))
    device = resolve_device(args.device or cfg.get("device", "auto"))

    world_model = load_world_model(args.lewm_checkpoint, map_location=device)
    policy_task = task
    if args.smoke:
        dataset = make_synthetic_dataset(
            num_samples=args.max_samples or 256,
            obs_shape=world_model.cfg.obs_shape,
            obs_history=int(world_model.cfg.obs_history),
            action_horizon=int(world_model.cfg.action_horizon),
            action_dim=int(world_model.cfg.action_dim),
            seed=int(cfg.get("seed", 7)) + 1,
        )
        policy_task = {
            "name": task.get("name", "smoke"),
            "obs_shape": list(world_model.cfg.obs_shape),
            "obs_history": int(world_model.cfg.obs_history),
            "latent_dim": int(world_model.cfg.latent_dim),
            "action_dim": int(world_model.cfg.action_dim),
            "action_horizon": int(world_model.cfg.action_horizon),
        }
    else:
        data_path = (
            args.data if args.data is not None else cfg.get("paths", {}).get("expert_data", "")
        )
        if not data_path:
            raise SystemExit("paths.expert_data is empty. Fill the task config or pass --smoke.")
        dataset = load_transition_dataset(
            data_path,
            max_samples=args.max_samples or cfg.get("stage2", {}).get("max_samples"),
        )

    stage2_cfg = _dataclass_from_config(Stage2TrainConfig, cfg.get("stage2", {}))
    if args.epochs is not None:
        stage2_cfg.epochs = args.epochs
    if args.batch_size is not None:
        stage2_cfg.batch_size = args.batch_size

    policy_cfg = dict(cfg.get("policy", {}))
    policy_kind = args.policy_kind or policy_cfg.get("kind") or stage2_cfg.policy_kind
    policy_kwargs = policy_kwargs_from_task(policy_task, policy_cfg)
    out_dir = (
        Path(args.out) if args.out else PROJECT_ROOT / "outputs" / experiment_slug(cfg) / "stage2"
    )

    _, _, metrics = train_stage2_policy_and_rnd(
        dataset=dataset,
        world_model=world_model,
        policy_kind=policy_kind,
        policy_kwargs=policy_kwargs,
        cfg=stage2_cfg,
        device=device,
        out_dir=out_dir,
    )
    write_json(out_dir / "resolved_config.json", cfg)
    print(
        json.dumps(
            {
                "policy_checkpoint": str(out_dir / "policy.pt"),
                "rnd_checkpoint": str(out_dir / "drs_o.pt"),
                "metrics": metrics,
            },
            indent=2,
        )
    )


def _dataclass_from_config(cls, values: dict):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in dict(values or {}).items() if key in allowed})


if __name__ == "__main__":
    main()
