#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preact.alignment import validate_paper_config
from preact.cem import CEMConfig, CEMOptimizer
from preact.online import OnlinePreActAgent, OnlinePreActAgentConfig, run_receding_horizon_episode
from preact.policies import load_policy
from preact.rnd import load_rnd
from preact.runtime import PreActRuntime, PreActRuntimeConfig
from preact.scoring import ActionEntropyConfig, ActionEntropyScorer, Thresholds
from preact.utils import load_merged_config, resolve_device, set_seed
from preact.world_model import load_world_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run online PreAct in a Gym-style environment.")
    parser.add_argument(
        "--env-factory", required=True, help="Import path like package.module:make_env."
    )
    parser.add_argument(
        "--env-kwargs-json", default="{}", help="JSON kwargs passed to the env factory."
    )
    parser.add_argument(
        "--task-config", default=str(PROJECT_ROOT / "configs/experiments/sorting_fm.yaml")
    )
    parser.add_argument("--base-config", default=str(PROJECT_ROOT / "configs/preact.yaml"))
    parser.add_argument("--lewm-checkpoint", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--rnd-checkpoint", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-policy-steps", type=int, default=None)
    parser.add_argument("--max-env-steps", type=int, default=None)
    parser.add_argument("--obs-key", default=None)
    parser.add_argument("--threshold-step-mode", choices=["policy", "env"], default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_merged_config(args.base_config, args.task_config)
    validate_paper_config(cfg)
    task = cfg["task"]
    set_seed(int(cfg.get("seed", 7)))
    device = resolve_device(args.device or cfg.get("device", "auto"))

    world_model = load_world_model(args.lewm_checkpoint, map_location=device)
    runtime_cfg = _dataclass_from_config(PreActRuntimeConfig, cfg.get("runtime", {}))
    runtime_cfg.execution_horizon = int(
        task.get("execution_horizon", runtime_cfg.execution_horizon)
    )
    runtime = PreActRuntime(
        world_model=world_model,
        policy=load_policy(args.policy_checkpoint, map_location=device),
        drs_o=load_rnd(args.rnd_checkpoint, map_location=device),
        action_scorer=ActionEntropyScorer(
            ActionEntropyConfig.from_dict(cfg.get("action_entropy", {}))
        ),
        thresholds=Thresholds.load(args.thresholds),
        config=runtime_cfg,
        cem=CEMOptimizer(_dataclass_from_config(CEMConfig, cfg.get("cem", {}))),
    ).to(device)

    factory = _load_factory(args.env_factory)
    env_kwargs = json.loads(args.env_kwargs_json)
    results = []
    for episode in range(args.episodes):
        env = factory(**env_kwargs)
        agent = OnlinePreActAgent(
            runtime,
            OnlinePreActAgentConfig(
                obs_history=int(world_model.cfg.obs_history),
                obs_key=args.obs_key,
                threshold_step_mode=args.threshold_step_mode
                or str(cfg.get("runtime", {}).get("threshold_step_mode", "policy")),
            ),
        )
        result = run_receding_horizon_episode(
            env,
            agent,
            max_policy_steps=args.max_policy_steps,
            max_env_steps=args.max_env_steps,
        )
        outputs = result.pop("outputs")
        result["episode"] = episode
        result["trigger_count"] = int(sum(int(out.triggered.any().item()) for out in outputs))
        results.append(result)
    print(json.dumps({"episodes": results}, indent=2, default=str))


def _load_factory(path: str):
    if ":" not in path:
        raise ValueError("--env-factory must have form package.module:callable")
    module_name, fn_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, fn_name)
    if not callable(factory):
        raise TypeError(f"{path} is not callable.")
    return factory


def _dataclass_from_config(cls, values: dict):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in dict(values or {}).items() if key in allowed})


if __name__ == "__main__":
    main()
