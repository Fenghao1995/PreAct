# PreAct

**Thinking before Acting: Runtime Risk Correction for Robotic Manipulation**

[![Status](https://img.shields.io/badge/status-code%20available-brightgreen)](https://github.com/Fenghao1995/PreAct)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Hao Feng, Zhijian Wang, Sifan Lan, Yinglong Yan, Yonghao Wang, Weiying Xie,
and Leyuan Fang**

## Overview

PreAct wraps a frozen generative imitation-learning policy in a
Predict–Evaluate–Act loop. A latent world model first predicts the consequence
of every action in the proposed chunk. DRS-O and DRS-A then measure imagined
observation deviation and future action uncertainty. When both risks cross
their calibrated, time-dependent thresholds, CEM locally corrects the chunk;
otherwise the original policy output is preserved.

<img width="4433" height="2516" alt="PreAct framework overview" src="https://github.com/user-attachments/assets/9270ce86-e7ec-4c64-9645-267a92610f20" />

## Highlights

- Two-stage training from Algorithm 2: LeWM first, then policy and RND.
- ACT-style conditional flow matching and temporal U-Net DDPM policies.
- Transformer and residual-MLP one-step latent world-model predictors.
- Appendix A.2 action-chunk entropy (ACE), including per-timestep joint
  histograms and horizon aggregation.
- Successful-rollout, time-varying DRS-O/DRS-A threshold calibration.
- Dual-threshold triggering, policy anchor, CEM correction, and
  receding-horizon execution from Algorithm 1.
- Twelve paper-checked experiment configurations: six tasks and two policy
  families.
- Franka/LeRobot support for PICK_PLACE, INSERT, and PUSH_TENNIS, including
  synchronized dual RealSense observations, bounded Cartesian control, data
  conversion, collection, a NUC RPC bridge, and online deployment.

The stable data boundary is the **policy-aligned observation embedding**
(`dim(Oe)` in the paper): 128 dimensions for SORTING and STACKING, and 64 for
the other tasks. The base policy, LeWM, and DRS-O must all use embeddings from
the same frozen visual/proprioceptive encoder. Datasets, encoder weights, and
model checkpoints are intentionally excluded from this source repository.

See [Paper alignment](docs/PAPER_ALIGNMENT.md) for the equation-to-code map,
paper ambiguities, and release defaults that were not numerically reported in
the submitted draft.

## Results reported in the manuscript

The manuscript evaluates PreAct on SORTING, STACKING, PUSHT, PICK_PLACE,
INSERT, and PUSH_TENNIS under in-distribution and distribution-shift settings.
The following values are the average success rates over all six tasks on the
`ALL` split:

| Policy backbone | Base policy | With PreAct | Gain |
| --- | ---: | ---: | ---: |
| Flow matching | 44.2% | 59.5% | +15.3 pp |
| Diffusion policy | 40.0% | 62.2% | +22.2 pp |

These are manuscript results, not metrics regenerated from the lightweight
dry run below. Exact evaluation requires the datasets, checkpoints, physical
setup, and seeds described in the reproducibility scope.

## Requirements and installation

- Linux with Python 3.10 or newer
- PyTorch 2.0 or newer
- CUDA is optional and selected automatically when available

Create an isolated environment and install the core plus dataset support:

```bash
git clone https://github.com/Fenghao1995/PreAct.git
cd PreAct
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[data]"
```

For development checks, install `.[dev,data]`. On the Franka workstation and
NUC, install the isolated hardware dependencies with `.[real,data]`.

As an alternative, the repository retains a Conda entry point:

```bash
conda env create -f environment.yml
conda activate preact
```

## Validate the installation

The commands below do not need datasets, weights, cameras, or a robot:

```bash
ruff check .

for cfg in configs/experiments/*.yaml; do
  python scripts/validate_config.py "$cfg"
done

for cfg in configs/real_robot/{pick_place,insert,push_tennis}.yaml; do
  python scripts/validate_real_config.py "$cfg"
done
```

For a lightweight CPU dry run of the full Stage I → Stage II → calibration →
runtime path, use one experiment config and write artifacts outside the source
tree:

```bash
CFG=configs/experiments/sorting_fm.yaml
OUT=/tmp/preact_dry_run

python scripts/train_stage1_lewm.py \
  --task-config "$CFG" --smoke --epochs 1 --batch-size 32 \
  --max-samples 64 --device cpu --out "$OUT/stage1"

python scripts/train_stage2_policy_rnd.py \
  --task-config "$CFG" --lewm-checkpoint "$OUT/stage1/lewm.pt" \
  --smoke --epochs 1 --batch-size 32 --max-samples 64 --device cpu \
  --out "$OUT/stage2"

python scripts/calibrate_thresholds.py \
  --task-config "$CFG" --lewm-checkpoint "$OUT/stage1/lewm.pt" \
  --policy-checkpoint "$OUT/stage2/policy.pt" \
  --rnd-checkpoint "$OUT/stage2/drs_o.pt" --smoke \
  --policy-samples 2 --future-action-samples 2 --batch-size 32 \
  --device cpu --out "$OUT/thresholds.json"

python scripts/run_preact.py \
  --task-config "$CFG" --lewm-checkpoint "$OUT/stage1/lewm.pt" \
  --policy-checkpoint "$OUT/stage2/policy.pt" \
  --rnd-checkpoint "$OUT/stage2/drs_o.pt" \
  --thresholds "$OUT/thresholds.json" --smoke --batch-size 2 --device cpu
```

This dry run checks software integration only; it does not reproduce the
paper's reported success rates.

## Prepare data

Stage I uses `D_LeWM`: expert trajectories plus physically executed
perturbation or random-play trajectories. Stage II uses `D_Expert`.
Calibration uses only successful, training-side `D_ID+` rollouts. Never use
the final ID/OOD evaluation episodes for calibration.

Supported inputs are `.npz`, `.h5`/`.hdf5`, and rollout `.pkl` files. A
prepared transition dataset contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `obs` | `(N, Th, De)` | Current policy-aligned observation history |
| `next_obs` | `(N, Th, De)` | History physically reached after `action[:, 0]` |
| `action` | `(N, H, Da)` | Expert action chunk; the first action trains LeWM |
| `episode_id` | `(N,)` | Rollout ID, required for paper-style calibration |
| `step_id` | `(N,)` | Replanning timestep, required for time-varying thresholds |

`next_obs` must be aligned with the first actually executed action. Do not
construct LeWM samples by corrupting labels offline. Inspect a prepared file
before training:

```bash
python scripts/inspect_dataset.py /absolute/path/to/dataset.npz
```

Set `paths.lewm_data`, `paths.expert_data`, and `paths.calibration_data`
in a copy of an experiment config, or pass `--data` to the corresponding
command. Keep large datasets and generated artifacts outside the repository.

## Paper configurations

Files follow `configs/experiments/<task>_<policy>.yaml`, where `fm` is the
ACT-style flow policy and `dp` is DDPM with a temporal U-Net.

| Task | Config stems | Domain | Embedding |
| --- | --- | --- | ---: |
| SORTING | `sorting_fm`, `sorting_dp` | simulation | 128 |
| STACKING | `stacking_fm`, `stacking_dp` | simulation | 128 |
| PUSHT | `pusht_fm`, `pusht_dp` | simulation | 64 |
| PICK_PLACE | `pick_place_fm`, `pick_place_dp` | Franka | 64 |
| INSERT | `insert_fm`, `insert_dp` | Franka | 64 |
| PUSH_TENNIS | `push_tennis_fm`, `push_tennis_dp` | Franka | 64 |

Every training, calibration, and runtime CLI validates the merged config
against the paper table before doing work. `configs/preact.yaml` contains
shared defaults and explicitly labels values that the draft does not report.
Each training command also writes its resolved config beside the checkpoint.

## Train and calibrate

The example below uses SORTING with flow matching. Replace the three data paths
in the config or add `--data /absolute/path/to/file` to each stage.

```bash
CFG=configs/experiments/sorting_fm.yaml
RUN=outputs/sorting_fm

python scripts/train_stage1_lewm.py \
  --task-config "$CFG" \
  --out "$RUN/stage1"

python scripts/train_stage2_policy_rnd.py \
  --task-config "$CFG" \
  --lewm-checkpoint "$RUN/stage1/lewm.pt" \
  --out "$RUN/stage2"

python scripts/calibrate_thresholds.py \
  --task-config "$CFG" \
  --lewm-checkpoint "$RUN/stage1/lewm.pt" \
  --policy-checkpoint "$RUN/stage2/policy.pt" \
  --rnd-checkpoint "$RUN/stage2/drs_o.pt" \
  --out "$RUN/thresholds.json"
```

For paper-scale runs, do not override the policy epochs, batch size, horizon,
sampling count, or integration/diffusion steps encoded in the selected config.
Stage-I optimizer values, CEM settings, ACE cell size, threshold quantile, and
network widths are release defaults because the draft does not specify them;
record any changes with the experiment.

## Evaluate and deploy

Replay prepared observations without controlling an environment:

```bash
python scripts/run_preact.py \
  --task-config "$CFG" \
  --lewm-checkpoint "$RUN/stage1/lewm.pt" \
  --policy-checkpoint "$RUN/stage2/policy.pt" \
  --rnd-checkpoint "$RUN/stage2/drs_o.pt" \
  --thresholds "$RUN/thresholds.json" \
  --data /absolute/path/to/replay.npz
```

Run receding-horizon control in a Gym-style environment:

```bash
python scripts/run_online_env.py \
  --env-factory your_package.your_env:make_env \
  --task-config "$CFG" \
  --lewm-checkpoint "$RUN/stage1/lewm.pt" \
  --policy-checkpoint "$RUN/stage2/policy.pt" \
  --rnd-checkpoint "$RUN/stage2/drs_o.pt" \
  --thresholds "$RUN/thresholds.json"
```

The environment factory must return an object with Gym-style `reset()` and
`step(action)` methods. PreAct executes only the first `h` actions from
each chunk and then replans.

## Real-robot reproduction

Public task configs live in `configs/real_robot/`; machine addresses, camera
serials, workspace limits, datasets, and weights do not. Start by copying the
ignored hardware overlay and measuring conservative workspace bounds:

```bash
cp configs/real_robot/hardware.local.example.yaml \
  configs/real_robot/hardware.local.yaml

python scripts/validate_real_config.py configs/real_robot/pick_place.yaml \
  --hardware-config configs/real_robot/hardware.local.yaml --runtime
```

Deployment refuses unresolved placeholders or missing Cartesian workspace
bounds, and requires an exact `--confirm-hardware <task>` safety
confirmation. The RPC transport has no authentication or encryption and must
be restricted to an isolated robot network.

See [Real-robot reproduction](docs/REAL_ROBOT.md) for the exact task protocol,
LeRobot export, direct rollout collection, plugin configuration, NUC service,
and deployment commands.

## Repository layout

```text
preact/
  alignment.py    paper-configuration validation
  world_model.py  shared encoder and one-step action-conditioned LeWM
  policies.py     flow/ACT and DDPM/temporal-U-Net policies
  rnd.py          DRS-O target/predictor networks
  scoring.py      ACE, risk aggregation, and threshold calibration
  cem.py          local candidate optimization
  runtime.py      one Predict–Evaluate–Act decision
  online.py       receding-horizon environment loop
  training.py     Stage I, Stage II, and calibration
  real_robot/     Franka hardware, data, collection, and environment bridges
lerobot_robot_preact_franka/  LeRobot 0.3.x third-party robot plugin
configs/          shared defaults and paper-checked experiment/task configs
scripts/          training, validation, conversion, and deployment CLIs
docs/             reproducibility and hardware notes
```

## Reproducibility scope

This repository implements the submitted method and encodes all values that
can be recovered unambiguously from the draft. Exact benchmark reproduction
still requires the original demonstrations, physically executed random-play
or perturbation data, successful calibration rollouts, the shared encoder and
policy weights, environment versions, and evaluation seeds. See
[Paper alignment](docs/PAPER_ALIGNMENT.md) before reporting comparisons.

## License

This project is released under the [MIT License](LICENSE).

## Citation

The submitted draft does not yet provide final venue, year, or DOI metadata.
Use [CITATION.cff](CITATION.cff) as the placeholder and update it after
acceptance.
