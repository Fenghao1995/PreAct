#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preact.data import load_transition_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a prepared PreAct transition dataset.")
    parser.add_argument("path")
    args = parser.parse_args()
    arrays = load_transition_dataset(args.path).arrays
    action_flat = arrays.action.reshape(-1, arrays.action.shape[-1])
    report = {
        "transitions": int(arrays.obs.shape[0]),
        "obs_shape": list(arrays.obs.shape[1:]),
        "next_obs_shape": list(arrays.next_obs.shape[1:]),
        "action_chunk_shape": list(arrays.action.shape[1:]),
        "action_range": (action_flat.max(dim=0).values - action_flat.min(dim=0).values).tolist(),
        "has_episode_id": arrays.episode_id is not None,
        "has_step_id": arrays.step_id is not None,
        "episodes": (
            None if arrays.episode_id is None else int(arrays.episode_id.unique().numel())
        ),
        "supports_time_varying_calibration": (
            arrays.episode_id is not None and arrays.step_id is not None
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
