# Real-robot reproduction

This directory integrates the three paper tasks with the cleaned PreAct core
without committing private datasets, model weights, robot addresses, camera
serial numbers, or generated media.

## What was retained from the development workspace

The audited source was the local LeRobot 0.3.4 checkout at commit
`da5d2f3e9187fa4690e6667fe8b294cae49016d6` and the nested Franka project at
commit `493690212f7acb9c857436bb3b721cf885295e53`, including its uncommitted
hardware-safety fixes. The release reimplements the required boundary as small,
reviewable modules:

- dual wrist/exterior RealSense capture with bounded retry and reconnect;
- Franka/Polymetis Zero-RPC client and NUC server;
- the recorded 7-D Cartesian-delta action order and 15-D state order;
- binary gripper commands, reset pose, EMA smoothing, norm limits, and explicit
  Cartesian workspace bounds;
- conversion from local LeRobot datasets to physically aligned PreAct tuples;
- online receding-horizon deployment through the same PreAct runtime used in
  simulation.

No `outputs/`, checkpoints, W&B logs, datasets, build products, IDE settings,
camera serials, or local absolute paths from the 120 GB workspace were copied.

## Paper task protocol

| Task | RGB input | Expert | Random play | Successful calibration | Max steps |
| --- | --- | ---: | ---: | ---: | ---: |
| PICK_PLACE | 2 × 240 × 424 | 50 | 10 | 20 | 49 |
| INSERT | 2 × 480 × 640 | 50 | 10 | 20 | 77 |
| PUSH_TENNIS | 2 × 240 × 424 | 50 | 10 | 20 | 19 |

All streams run at 15 FPS. Each action is
`[dx, dy, dz, dRx, dRy, dRz, gripper]`; proprioception is seven joint
positions, normalized gripper state, binary gripper command, and the six-value
end-effector pose. The complete order is exported as `ACTION_NAMES` and
`STATE_NAMES` in `preact.real_robot.config`.

PICK_PLACE uses a 12.5 cm orange 3 × 3 grid and a 5 cm blue cube. ID places the
cube in the center cell; OOD moves it roughly 1–2 cm outside that cell. INSERT
uses the same grid: the torch and holder occupy the third and first cells in ID;
OOD shifts the holder 1 cm right or down. PUSH_TENNIS uses a 7 cm ball and a
45 cm × 9.5 cm red track; the draft reports no separate OOD scene.

## Install and local hardware config

```bash
python -m pip install -e ".[real,data]"
cp configs/real_robot/hardware.local.example.yaml \
   configs/real_robot/hardware.local.yaml
```

Edit the ignored local file. For INSERT, change both cameras to 640 × 480; the
other tasks use 424 × 240. Measure workspace bounds in the Franka base frame.
Deployment refuses to start with missing bounds or unresolved placeholders.

Validate the public task configuration and then the merged machine config:

```bash
python scripts/validate_real_config.py configs/real_robot/pick_place.yaml
python scripts/validate_real_config.py configs/real_robot/pick_place.yaml \
  --hardware-config configs/real_robot/hardware.local.yaml --runtime
```

## Shared encoder contract

The paper uses one frozen policy-aligned encoder after Stage I. Dataset export
and deployment must call the same encoder factory, with signature:

```python
def make_encoder(*, checkpoint, device, task):
    # Load the Stage-I shared RGB/proprioceptive encoder.
    # The returned callable accepts one observation dict and emits shape (64,).
    return encoder
```

The input dictionary contains `observation.images.wrist_image`,
`observation.images.exterior_image`, and `observation.state`, matching the
existing LeRobot recordings. Encoder weights are intentionally absent from
this source release; pass their path with `--encoder-checkpoint`.

## Convert existing LeRobot data

```bash
python scripts/export_lerobot_transitions.py \
  --task-config configs/experiments/pick_place_dp.yaml \
  --real-config configs/real_robot/pick_place.yaml \
  --repo-id your_namespace/pick_place \
  --root /absolute/path/to/local/dataset \
  --encoder-factory your_package.encoder:make_encoder \
  --encoder-checkpoint /absolute/path/to/shared_encoder.pt \
  --split expert \
  --out /absolute/path/outside/repo/pick_place_expert.npz
```

Repeat for `random_play` and `calibration`. Calibration input must contain only
successful, training-side ID rollouts. The exporter edge-pads history/chunks,
but never corrupts labels: `next_obs` is the observation physically reached by
executing `action[:, 0]`.

For direct collection, `collect_real_rollouts.py` accepts an action-source
factory returning `action_source(observation) -> ndarray[7] | None`. This lets
the same safe recorder work with the existing SpaceMouse, Oculus, or a frozen
base-policy wrapper while keeping device-specific packages optional.

Installing the `real` extra also exposes a standard LeRobot third-party device
plugin named `preact_franka`. Its config is
`lerobot_robot_preact_franka.PreActFrankaConfig`; it accepts `task_config` and
`hardware_config` paths and preserves the existing `wrist_image`,
`exterior_image`, state, and Cartesian action fields. LeRobot 0.3.4 discovers
packages prefixed with `lerobot_robot_` automatically. Prefer the direct PreAct
collector when strict storage of post-limit/EMA executed actions is required;
the plugin also mutates the passed LeRobot action mapping to preserve that
alignment in the standard 0.3.4 recording loop.

## NUC bridge and deployment

The RPC channel has no authentication or encryption. Put it on an isolated
robot network and restrict port 4242 with a firewall. On the NUC:

```bash
python scripts/franka_rpc_server.py \
  --bind-host 192.0.2.10 --allow-network-rpc
```

On the policy workstation, after checking the physical emergency stop:

```bash
python scripts/run_real_robot.py \
  --task-config configs/experiments/pick_place_dp.yaml \
  --real-config configs/real_robot/pick_place.yaml \
  --hardware-config configs/real_robot/hardware.local.yaml \
  --lewm-checkpoint /weights/lewm.pt \
  --policy-checkpoint /weights/policy.pt \
  --rnd-checkpoint /weights/drs_o.pt \
  --thresholds /weights/thresholds.json \
  --encoder-factory your_package.encoder:make_encoder \
  --encoder-checkpoint /weights/shared_encoder.pt \
  --confirm-hardware pick_place --prompt-success
```

The command performs the paper's dual-threshold check, CEM correction, and
execution of only the first `h` actions before replanning. It does not infer
task success from pixels; reported evaluation labels remain human verified.
