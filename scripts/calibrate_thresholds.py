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
from preact.data import load_transition_dataset, make_synthetic_dataset
from preact.policies import load_policy
from preact.rnd import load_rnd
from preact.scoring import ActionEntropyConfig, ActionEntropyScorer
from preact.training import calibrate_thresholds
from preact.utils import experiment_slug, load_merged_config, resolve_device, set_seed, write_json
from preact.world_model import load_world_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate PreAct DRS-O/DRS-A thresholds.")
    parser.add_argument(
        "--task-config", default=str(PROJECT_ROOT / "configs/experiments/sorting_fm.yaml")
    )
    parser.add_argument("--base-config", default=str(PROJECT_ROOT / "configs/preact.yaml"))
    parser.add_argument("--lewm-checkpoint", required=True)
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--rnd-checkpoint", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--data", default=None, help="Override paths.calibration_data.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quantile", type=float, default=None)
    parser.add_argument("--style", choices=["scalar", "time_varying"], default=None)
    parser.add_argument("--policy-samples", type=int, default=None)
    parser.add_argument("--future-action-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--smoke", action="store_true", help="Use synthetic data matching the LeWM checkpoint."
    )
    args = parser.parse_args()

    cfg = load_merged_config(args.base_config, args.task_config)
    validate_paper_config(cfg)
    set_seed(int(cfg.get("seed", 7)))
    device = resolve_device(args.device or cfg.get("device", "auto"))
    calib_cfg = cfg.get("calibration", {})

    world_model = load_world_model(args.lewm_checkpoint, map_location=device)
    if args.smoke:
        dataset = make_synthetic_dataset(
            num_samples=128,
            obs_shape=world_model.cfg.obs_shape,
            obs_history=int(world_model.cfg.obs_history),
            action_horizon=int(world_model.cfg.action_horizon),
            action_dim=int(world_model.cfg.action_dim),
            seed=int(cfg.get("seed", 7)) + 2,
        )
    else:
        data_path = (
            args.data if args.data is not None else cfg.get("paths", {}).get("calibration_data", "")
        )
        if not data_path:
            raise SystemExit(
                "paths.calibration_data is empty. Fill the task config or pass --smoke."
            )
        dataset = load_transition_dataset(data_path)

    thresholds = calibrate_thresholds(
        dataset=dataset,
        world_model=world_model,
        policy=load_policy(args.policy_checkpoint, map_location=device),
        rnd=load_rnd(args.rnd_checkpoint, map_location=device),
        action_scorer=ActionEntropyScorer(
            ActionEntropyConfig.from_dict(cfg.get("action_entropy", {}))
        ),
        batch_size=args.batch_size or int(calib_cfg.get("batch_size", 256)),
        device=device,
        quantile=float(
            args.quantile if args.quantile is not None else calib_cfg.get("quantile", 0.9)
        ),
        policy_samples=int(args.policy_samples or calib_cfg.get("policy_samples", 32)),
        future_action_samples=int(
            args.future_action_samples or calib_cfg.get("future_action_samples", 8)
        ),
        future_action_latent_index=calib_cfg.get("future_action_latent_index"),
        future_action_reduce=str(calib_cfg.get("future_action_reduce", "sum")),
        observation_reduce=str(calib_cfg.get("future_observation_reduce", "sum")),
        style=args.style or calib_cfg.get("threshold_style", "time_varying"),
    )
    out_path = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT / "outputs" / experiment_slug(cfg) / "thresholds.json"
    )
    thresholds.save(out_path)
    write_json(out_path.with_suffix(".config.json"), cfg)
    print(json.dumps({"thresholds": thresholds.to_dict(), "path": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
