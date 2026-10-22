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
from preact.cem import CEMConfig, CEMOptimizer
from preact.online import OnlinePreActAgent, OnlinePreActAgentConfig, run_receding_horizon_episode
from preact.policies import load_policy
from preact.real_robot.config import load_real_task_config
from preact.real_robot.env import FrankaPreActEnv
from preact.real_robot.factory import make_observation_encoder
from preact.real_robot.franka import FrankaHardware
from preact.rnd import load_rnd
from preact.runtime import PreActRuntime, PreActRuntimeConfig
from preact.scoring import ActionEntropyConfig, ActionEntropyScorer, Thresholds
from preact.utils import load_merged_config, resolve_device, set_seed
from preact.world_model import load_world_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy PreAct on the paper Franka setup.")
    parser.add_argument(
        "--task-config", required=True, help="configs/experiments/<task>_<family>.yaml"
    )
    parser.add_argument("--real-config", required=True, help="configs/real_robot/<task>.yaml")
    parser.add_argument(
        "--hardware-config", required=True, help="Ignored-by-git local hardware overlay"
    )
    parser.add_argument("--base-config", default=str(PROJECT_ROOT / "configs/preact.yaml"))
    parser.add_argument("--lewm-checkpoint", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--rnd-checkpoint", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--encoder-factory", required=True, help="package.module:factory")
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-policy-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--auto-reset-robot", action="store_true")
    parser.add_argument("--prompt-success", action="store_true")
    parser.add_argument(
        "--confirm-hardware",
        required=True,
        help="Safety gate: must exactly equal the task name (for example pick_place).",
    )
    args = parser.parse_args()

    cfg = load_merged_config(args.base_config, args.task_config)
    validate_paper_config(cfg)
    real_cfg = load_real_task_config(
        args.real_config,
        hardware_path=args.hardware_config,
        require_runtime_values=True,
    )
    task = cfg["task"]
    if task["name"] != real_cfg.name:
        raise SystemExit("Core experiment and real-robot task names do not match.")
    if args.confirm_hardware != real_cfg.name:
        raise SystemExit(f"Refusing hardware access: --confirm-hardware must be {real_cfg.name!r}.")
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")

    set_seed(int(cfg.get("seed", 7)))
    device = resolve_device(args.device or cfg.get("device", "auto"))
    world_model = load_world_model(args.lewm_checkpoint, map_location=device)
    if int(world_model.cfg.latent_dim) != 64:
        raise SystemExit("Paper real-world deployment requires latent_dim=64.")
    runtime_cfg = _dataclass_from_config(PreActRuntimeConfig, cfg.get("runtime", {}))
    runtime_cfg.execution_horizon = int(task["execution_horizon"])
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
    encoder = make_observation_encoder(
        args.encoder_factory,
        checkpoint=args.encoder_checkpoint,
        device=device,
        task=real_cfg,
    )

    print(f"Task: {real_cfg.name}")
    print(f"Objective: {real_cfg.description}")
    print(f"Success criterion: {real_cfg.success}")
    print("Verify the physical emergency stop and clear the workspace before continuing.")

    env = FrankaPreActEnv(
        FrankaHardware(real_cfg.hardware),
        real_cfg,
        encoder,
        auto_connect=False,
        reset_robot_on_reset=args.auto_reset_robot,
    )
    results = []
    try:
        env.connect()
        for episode in range(args.episodes):
            if episode > 0 and not args.auto_reset_robot:
                input("Reset the scene and robot safely, then press Enter to continue...")
            agent = OnlinePreActAgent(
                runtime,
                OnlinePreActAgentConfig(
                    obs_history=int(task["obs_history"]),
                    threshold_step_mode=str(
                        cfg.get("runtime", {}).get("threshold_step_mode", "policy")
                    ),
                ),
            )
            result = run_receding_horizon_episode(
                env,
                agent,
                max_policy_steps=args.max_policy_steps,
                max_env_steps=real_cfg.maximum_episode_steps,
            )
            outputs = result.pop("outputs")
            result.pop("last_info", None)
            result["episode"] = episode
            result["trigger_count"] = sum(int(output.triggered.any().item()) for output in outputs)
            if args.prompt_success:
                answer = input("Was this rollout successful? [y/N] ").strip().lower()
                result["success"] = answer in {"y", "yes"}
            results.append(result)
    finally:
        env.close()
    print(json.dumps({"task": real_cfg.name, "episodes": results}, indent=2))


def _dataclass_from_config(cls, values: dict):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in dict(values or {}).items() if key in allowed})


if __name__ == "__main__":
    main()
