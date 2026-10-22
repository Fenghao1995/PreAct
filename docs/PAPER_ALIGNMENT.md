# Paper alignment and reproducibility notes

This document records what can be reproduced exactly from `PreAct_Proof.pdf`
and separates it from values that are absent or internally inconsistent in the
submitted draft.

## Method-to-code map

| Paper item | Implementation |
| --- | --- |
| Eq. (4), Appendix A.1 | `LatentWorldModel.predict_next_latent(z_t, a_t)` predicts one transition from one action. `rollout_from_latent` applies it recursively for `H` steps. |
| Eq. (5), Eqs. (9)–(26) | `FlowMatchingPolicy` and `DiffusionPolicy` condition only on the current shared latent, never on the proposed action before generation. |
| Eq. (6), Eq. (14), Appendix A.3 | `RandomNetworkDistillation` freezes `g`; Stage II trains only `f_theta` on detached real and imagined futures with equal weighting. |
| Eqs. (1)–(6), Appendix A.2 | `ActionEntropyScorer` computes `R_d` offline, builds a joint D-dimensional histogram independently at every action timestep, and sums base-2 entropy across `H`. |
| Eq. (2), Appendix A.4 | `PreActRuntime` triggers only when both raw scores exceed their time-varying successful-rollout thresholds. |
| Eqs. (7)–(10) | Runtime cost uses positive-part standardized DRS-O/DRS-A plus the unnormalized mean-square anchor to the nominal chunk. |
| Algorithms 1–2 | `training.py`, `runtime.py`, `cem.py`, and `online.py` implement the two offline stages, calibration, CEM correction, and execution of the first `h` actions. |

## Paper-reported settings encoded verbatim

Every file under `configs/experiments/` is checked by
`preact.alignment.validate_paper_config` before a CLI starts.

- FM/ACT: `H=8`, `h=4`, `B=32`, 16 integration steps, 20 epochs, batch 256,
  learning rate `5e-4`.
- DDPM/U-Net simulation: `H=16`, `h=8`, `B=256`, 100 diffusion steps, 2000
  epochs, batch 256, learning rate `3e-5`.
- DDPM/U-Net real world: `H=16`, `h=8`, `B=30`, 100 diffusion steps, 600
  epochs, batch 64, learning rate `1e-4`.
- Task-specific observation histories, latent dimensions, action/state
  dimensions, calibration-rollout counts, and episode lengths come from
  Appendix Tables 1–3.
- SIGReg uses 1024 projections and weight 0.1.
- Thresholds are calibrated separately at each policy timestep.

## Values not reported in the draft

The following values are required by an executable implementation but are not
numerically specified in the draft. They are marked next to their defaults in
`configs/preact.yaml`; changing them does not trip the paper-table validator.

- Stage-I LeWM epochs, batch size, optimizer, learning rate, and weight decay.
- CEM sample count `N`, elite count `K`, iteration count `T`, covariance floor,
  and smoothing momentum.
- ACE cell-size factor `alpha`.
- Threshold quantile `1-delta`.
- RND output dimension `m`.
- Hidden size/depth/head count for LeWM, ACT, and the temporal U-Net.

For an archival release, record these resolved values with each checkpoint and
report them in the final paper or supplementary material.

## Inconsistencies in the submitted PDF

The main-paper dataset table and Appendix B.2 disagree for two simulation
tasks:

| Task | Main Table 1 perturbations | Appendix B.2 perturbations |
| --- | ---: | ---: |
| SORTING | 320 | 960 (six per 160 expert rollouts) |
| STACKING | 1000 | 6000 (six per 1000 expert rollouts) |

The checked configs use **main-paper Table 1** because it is the explicit
dataset-composition table. PUSHT (540) and all three real-world random-play
counts (10) agree. Resolve the two discrepancies before claiming an exact data
reproduction.

The method section sometimes writes correction over `K` proposals while
Algorithm 1 uses `N` CEM samples and `K` elites. The implementation follows the
algorithm: `N` candidates per iteration and `K` lowest-cost elites.

## Representation and real-robot boundary

The paper uses task-specific visual backbones and robot proprioception, but the
draft does not completely specify visual preprocessing. This repository defines
its stable model/data boundary at the shared policy-aligned representation
`Oe`. The same encoder factory is required for LeRobot export, direct physical
collection, and deployment, which prevents an accidental train/deploy feature
mismatch.

The release does include the surrounding Franka/Polymetis RPC, dual RealSense,
7-D Cartesian action, 15-D state, safety, data conversion, and online execution
code. See `docs/REAL_ROBOT.md`. It does not claim that synthetic dry-run data
reproduce the reported success rates. Exact benchmark numbers additionally
require the original demonstrations, physically rolled-out random-play data,
successful calibration rollouts, encoder/model weights, and evaluation seeds;
those artifacts are intentionally excluded.
