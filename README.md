# PreAct

**Thinking before Acting: Runtime Risk Correction for Robotic Manipulation**

[![Status](https://img.shields.io/badge/status-code%20release%20in%20preparation-yellow)](https://github.com/Fenghao1995/PreAct)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **PreAct** equips generative robot policies with a Predict-Evaluate-Act loop: before executing an action chunk, the robot imagines its latent consequence, assesses future risk, and selects a lower-risk alternative when correction is needed.

**Hao Feng, Zhijian Wang, Sifan Lan, Yinglong Yan, Yonghao Wang, Weiying Xie, and Leyuan Fang**

## Overview
Diffusion- and flow-matching policies can be brittle when deployment drifts from training data - for example, through object-pose changes, contact disturbances, or accumulated execution errors. Existing failure monitors can raise an alarm, but do not specify how to change the imminent action; recovery methods often depend on corrective demonstrations or human intervention.

PreAct treats runtime correction as **prospective counterfactual reasoning**. It uses a latent world model as an active feedback module to compare the predicted consequences of candidate action chunks before they are executed. The framework is model-agnostic and is designed to wrap both diffusion- and flow-matching-based manipulation policies.
<img width="4433" height="2516" alt="image2" src="https://github.com/user-attachments/assets/9270ce86-e7ec-4c64-9645-267a92610f20" />


## How PreAct Works

1. **Predict.** An action-conditioned latent world model (LeWM) predicts the short-horizon latent consequence of the policy's proposed action chunk.
2. **Evaluate.** Two complementary deviation scores assess the imagined future: **DRS-O** measures departure from successful latent behaviour, while **DRS-A** measures instability in the policy's future action distribution.
3. **Act.** Only when both calibrated risk signals exceed threshold, PreAct samples a compact, anchored set of proposals around the nominal action and executes the chunk with the lowest predicted future deviation. Otherwise, it preserves the original policy action chunk. This runs in a receding-horizon loop with all learned modules fixed at deployment.

## Results

The manuscript evaluates PreAct on three simulated tasks (**SORTING**, **STACKING**, and **PUSHT**) and three real-world tasks (**PICK_PLACE**, **INSERT**, and **PUSH_TENNIS**) under in-distribution and distribution-shift settings. The table reports the average success rate over all six tasks on the `ALL` split.

| Policy backbone | Base policy | With PreAct | Gain |
| --- | ---: | ---: | ---: |
| Flow matching | 44.2% | 59.5% | +15.3 pp |
| Diffusion policy | 40.0% | 62.2% | +22.2 pp |

PreAct's largest benefits appear under distribution shift, where predicting the consequences of a candidate action is more informative than judging the current observation alone. The complete framework attains a macro-average success rate of 0.63 versus 0.42 for the uncorrected base policies in the component ablation.


## Create the Conda environment

```bash
git clone https://github.com/Fenghao1995/PreAct.git
cd PreAct

conda env create -f environment.yml
conda activate preact
```
## Datasets preparation

Data for the three simulation tasks (SORTING, STACKING, PUSH-T) comes from https://alrhub.github.io/d3il-website/, while data for the three real-world tasks (PICK_PLACE, INSERT, PUSH_TENNIS) was collected via teleoperation. 


## Release Status

- [x] Method and experimental results described in the manuscript
- [x] Repository and license initialized
- [x] Configuration files and environment specification
- [ ] Datasets preparation
- [ ] Training and inference code
- [ ] Checkpoints and reproducible evaluation scripts

Code, model artifacts, and reproduction instructions are currently being prepared. Those instructions will be added together with the corresponding tested release artifacts.



## License

This project is released under the [MIT License](LICENSE).
