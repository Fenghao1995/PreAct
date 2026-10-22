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
from preact.real_robot.collection import collect_embedding_episode
from preact.real_robot.config import load_real_task_config
from preact.real_robot.dataset import save_transition_npz
from preact.real_robot.factory import load_object, make_observation_encoder
from preact.real_robot.franka import FrankaHardware
from preact.utils import load_merged_config, resolve_device, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect paper-aligned Franka rollout embeddings.")
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--real-config", required=True)
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--base-config", default=str(PROJECT_ROOT / "configs/preact.yaml"))
    parser.add_argument("--split", required=True, choices=["expert", "random_play", "calibration"])
    parser.add_argument("--action-source-factory", required=True, help="package.module:factory")
    parser.add_argument("--encoder-factory", required=True, help="package.module:factory")
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--out-root", required=True, help="Must be outside this source repository")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--auto-reset-robot", action="store_true")
    parser.add_argument("--confirm-hardware", required=True)
    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    if out_root == PROJECT_ROOT or out_root.is_relative_to(PROJECT_ROOT):
        raise SystemExit("Dataset output must be outside the source repository.")
    cfg = load_merged_config(args.base_config, args.task_config)
    validate_paper_config(cfg)
    real_cfg = load_real_task_config(
        args.real_config,
        hardware_path=args.hardware_config,
        require_runtime_values=True,
    )
    if cfg["task"]["name"] != real_cfg.name:
        raise SystemExit("Core experiment and real-robot task names do not match.")
    if args.confirm_hardware != real_cfg.name:
        raise SystemExit(f"--confirm-hardware must be {real_cfg.name!r}")

    expected = {
        "expert": real_cfg.collection.expert_rollouts,
        "random_play": real_cfg.collection.random_play_rollouts,
        "calibration": real_cfg.collection.calibration_rollouts,
    }[args.split]
    requested = expected if args.episodes is None else int(args.episodes)
    if requested <= 0:
        raise SystemExit("--episodes must be positive")
    device = resolve_device(args.device)
    encoder = make_observation_encoder(
        args.encoder_factory,
        checkpoint=args.encoder_checkpoint,
        device=device,
        task=real_cfg,
    )
    source_factory = load_object(args.action_source_factory)
    hardware = FrankaHardware(real_cfg.hardware)
    output_dir = out_root / real_cfg.name / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("episode_*.npz")):
        raise SystemExit(f"Refusing to overwrite existing rollout files in {output_dir}")
    saved: list[dict] = []
    try:
        hardware.connect()
        action_source = source_factory(task=real_cfg, hardware=hardware, device=device)
        if not callable(action_source):
            raise TypeError("action-source factory must return a callable")
        episode_id = 0
        while episode_id < requested:
            if args.auto_reset_robot:
                hardware.reset()
            else:
                input(
                    f"Prepare {real_cfg.name} episode {episode_id + 1}/{requested}; "
                    "press Enter..."
                )
            reset_source = getattr(action_source, "reset", None)
            if callable(reset_source):
                reset_source()
            collected = collect_embedding_episode(
                hardware,
                action_source,
                encoder,
                episode_id=episode_id,
                maximum_steps=real_cfg.maximum_episode_steps,
                fps=real_cfg.collection.fps,
                obs_history=int(cfg["task"]["obs_history"]),
                action_horizon=int(cfg["task"]["action_horizon"]),
            )
            successful = input("Successful rollout? [y/N] ").strip().lower() in {"y", "yes"}
            if args.split in {"expert", "calibration"} and not successful:
                print(f"Discarded: {args.split} accepts successful rollouts only.")
                continue
            path = output_dir / f"episode_{episode_id:04d}.npz"
            save_transition_npz(path, collected.arrays)
            entry = {
                "episode": episode_id,
                "path": path.name,
                "successful": successful,
                "frames": collected.frame_count,
                "duration_s": collected.duration_s,
            }
            saved.append(entry)
            manifest = {"task": real_cfg.name, "split": args.split, "episodes": saved}
            write_json(output_dir / "manifest.json", manifest)
            episode_id += 1
    finally:
        hardware.close()
    print(json.dumps({"output": str(output_dir), "episodes": saved}, indent=2))


if __name__ == "__main__":
    main()
