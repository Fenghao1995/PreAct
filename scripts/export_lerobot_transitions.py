#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preact.alignment import validate_paper_config
from preact.real_robot.config import load_real_task_config
from preact.real_robot.dataset import (
    episode_count,
    load_lerobot_transition_arrays,
    save_transition_npz,
)
from preact.real_robot.factory import make_observation_encoder
from preact.utils import load_merged_config, resolve_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export synchronized LeRobot frames to PreAct transition arrays."
    )
    parser.add_argument(
        "--task-config", required=True, help="configs/experiments/<task>_<family>.yaml"
    )
    parser.add_argument("--real-config", required=True, help="configs/real_robot/<task>.yaml")
    parser.add_argument("--base-config", default=str(PROJECT_ROOT / "configs/preact.yaml"))
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True, help="Local LeRobot dataset root; never copied.")
    parser.add_argument("--encoder-factory", required=True, help="package.module:factory")
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--split", required=True, choices=["expert", "random_play", "calibration"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--allow-episode-count-mismatch", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out).expanduser()
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite existing export: {out_path}")

    core_cfg = load_merged_config(args.base_config, args.task_config)
    validate_paper_config(core_cfg)
    real_cfg = load_real_task_config(args.real_config)
    if core_cfg["task"]["name"] != real_cfg.name:
        raise SystemExit("Core experiment and real-robot task names do not match.")

    device = resolve_device(args.device)
    encoder = make_observation_encoder(
        args.encoder_factory,
        checkpoint=args.encoder_checkpoint,
        device=device,
        task=real_cfg,
    )
    arrays = load_lerobot_transition_arrays(
        repo_id=args.repo_id,
        root=args.root,
        observation_encoder=encoder,
        obs_history=int(core_cfg["task"]["obs_history"]),
        action_horizon=int(core_cfg["task"]["action_horizon"]),
        max_frames=args.max_frames,
    )
    if arrays.obs.shape[-1] != int(core_cfg["task"]["latent_dim"]):
        raise SystemExit(
            f"Encoder emitted {arrays.obs.shape[-1]} features; "
            f"paper config requires {core_cfg['task']['latent_dim']}."
        )

    actual_episodes = episode_count(arrays.episode_id)
    expected_episodes = {
        "expert": real_cfg.collection.expert_rollouts,
        "random_play": real_cfg.collection.random_play_rollouts,
        "calibration": real_cfg.collection.calibration_rollouts,
    }[args.split]
    if (
        actual_episodes != expected_episodes
        and not args.allow_episode_count_mismatch
        and args.max_frames is None
    ):
        raise SystemExit(
            f"{args.split} contains {actual_episodes} episodes; "
            f"paper requires {expected_episodes}. "
            "Use --allow-episode-count-mismatch only for an intentional partial export."
        )

    out = save_transition_npz(out_path, arrays)
    metadata = {
        "task": real_cfg.name,
        "split": args.split,
        "source_repo_id": args.repo_id,
        "encoder_factory": args.encoder_factory,
        "episode_count": actual_episodes,
        "transition_count": len(arrays.obs),
        "obs_shape": list(arrays.obs.shape),
        "action_shape": list(arrays.action.shape),
        "physical_alignment": "next_obs follows action[:, 0]",
        "calibration_requirement": "successful training-side ID only"
        if args.split == "calibration"
        else None,
    }
    write_json(out.with_suffix(".metadata.json"), metadata)
    print(json.dumps({"dataset": str(out), **metadata}, indent=2))


if __name__ == "__main__":
    main()
