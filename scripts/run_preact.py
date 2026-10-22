#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preact.alignment import validate_paper_config
from preact.cem import CEMConfig, CEMOptimizer
from preact.data import load_transition_dataset, make_synthetic_dataset
from preact.policies import load_policy
from preact.rnd import load_rnd
from preact.runtime import PreActRuntime, PreActRuntimeConfig
from preact.scoring import ActionEntropyConfig, ActionEntropyScorer, Thresholds
from preact.utils import load_merged_config, resolve_device, set_seed
from preact.world_model import load_world_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline replay/sanity check for PreAct. Use run_online_env.py for deployment."
    )
    parser.add_argument(
        "--task-config", default=str(PROJECT_ROOT / "configs/experiments/sorting_fm.yaml")
    )
    parser.add_argument("--base-config", default=str(PROJECT_ROOT / "configs/preact.yaml"))
    parser.add_argument("--lewm-checkpoint", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--rnd-checkpoint", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument(
        "--data", default=None, help="Defaults to paths.expert_data, then paths.calibration_data."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_merged_config(args.base_config, args.task_config)
    validate_paper_config(cfg)
    task = cfg["task"]
    set_seed(int(cfg.get("seed", 7)))
    device = resolve_device(args.device or cfg.get("device", "auto"))

    world_model = load_world_model(args.lewm_checkpoint, map_location=device)
    policy = load_policy(args.policy_checkpoint, map_location=device)
    rnd = load_rnd(args.rnd_checkpoint, map_location=device)

    if args.smoke:
        dataset = make_synthetic_dataset(
            num_samples=args.batch_size,
            obs_shape=world_model.cfg.obs_shape,
            obs_history=int(world_model.cfg.obs_history),
            action_horizon=int(world_model.cfg.action_horizon),
            action_dim=int(world_model.cfg.action_dim),
            seed=int(cfg.get("seed", 7)) + 1,
        )
    else:
        data_path = (
            args.data
            or cfg.get("paths", {}).get("expert_data")
            or cfg.get("paths", {}).get("calibration_data")
        )
        if not data_path:
            raise SystemExit(
                "No dataset path provided. Fill task config, pass --data, or use --smoke."
            )
        dataset = load_transition_dataset(data_path, max_samples=args.batch_size)

    runtime_cfg = _dataclass_from_config(PreActRuntimeConfig, cfg.get("runtime", {}))
    runtime_cfg.execution_horizon = int(
        task.get("execution_horizon", runtime_cfg.execution_horizon)
    )
    cem_cfg = _dataclass_from_config(CEMConfig, cfg.get("cem", {}))
    corrector = PreActRuntime(
        world_model=world_model,
        policy=policy,
        drs_o=rnd,
        action_scorer=ActionEntropyScorer(
            ActionEntropyConfig.from_dict(cfg.get("action_entropy", {}))
        ),
        thresholds=Thresholds.load(args.thresholds),
        config=runtime_cfg,
        cem=CEMOptimizer(cem_cfg),
    ).to(device)

    batch = next(
        iter(DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=False))
    )
    with torch.no_grad():
        out = corrector.step(batch["obs"].to(device), step_index=0)
    print(
        json.dumps(
            {
                "action_pred_shape": list(out.action_pred.shape),
                "executable_action_shape": list(out.executable_action.shape),
                "triggered": out.triggered.detach().cpu().int().tolist(),
                "obs_score_mean": float(out.obs_score.mean().item()),
                "action_score_mean": float(out.action_score.mean().item()),
            },
            indent=2,
        )
    )


def _dataclass_from_config(cls, values: dict):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in dict(values or {}).items() if key in allowed})


if __name__ == "__main__":
    main()
