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
from preact.utils import load_merged_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a config against the submitted paper.")
    parser.add_argument("experiment_config")
    parser.add_argument("--base-config", default=str(PROJECT_ROOT / "configs/preact.yaml"))
    args = parser.parse_args()
    config = load_merged_config(args.base_config, args.experiment_config)
    validate_paper_config(config)
    print(
        json.dumps(
            {
                "status": "paper-aligned",
                "task": config["task"]["name"],
                "policy": config["policy"]["kind"],
                "action_horizon": config["task"]["action_horizon"],
                "execution_horizon": config["task"]["execution_horizon"],
                "stage2": config["stage2"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
