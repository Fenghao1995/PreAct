#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preact.real_robot.config import load_real_task_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a paper real-robot configuration.")
    parser.add_argument("task_config")
    parser.add_argument("--hardware-config", default=None)
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Also require resolved IP, serials, data root, and workspace bounds.",
    )
    args = parser.parse_args()
    cfg = load_real_task_config(
        args.task_config,
        hardware_path=args.hardware_config,
        require_runtime_values=args.runtime,
    )
    payload = asdict(cfg)
    if args.runtime:
        payload["hardware"]["host"] = "<redacted>"
        for camera in payload["hardware"]["cameras"]:
            camera["serial"] = "<redacted>"
        payload["collection"]["data_root"] = "<redacted>"
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
