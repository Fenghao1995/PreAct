from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PaperTask:
    observation_shape: tuple[int, ...]
    latent_dim: int
    action_dim: int
    state_dim: int
    calibration_rollouts: int
    average_episode_length: int
    maximum_episode_length: int
    fm_obs_history: int


PAPER_TASKS: dict[str, PaperTask] = {
    "sorting": PaperTask((2, 3, 96, 96), 128, 2, 3, 50, 47, 75, 1),
    "stacking": PaperTask((2, 3, 96, 96), 128, 8, 8, 50, 79, 120, 5),
    "pusht": PaperTask((3, 512, 512), 64, 2, 2, 50, 31, 38, 1),
    "pick_place": PaperTask((2, 3, 240, 424), 64, 7, 15, 20, 33, 49, 5),
    "insert": PaperTask((2, 3, 480, 640), 64, 7, 15, 20, 59, 77, 5),
    "push_tennis": PaperTask((2, 3, 240, 424), 64, 7, 15, 20, 10, 19, 1),
}

PAPER_DATASETS: dict[str, dict[str, int]] = {
    "sorting": {
        "expert_rollouts": 160,
        "lewm_expert_rollouts": 160,
        "lewm_perturbation_rollouts": 320,
    },
    "stacking": {
        "expert_rollouts": 1000,
        "lewm_expert_rollouts": 1000,
        "lewm_perturbation_rollouts": 1000,
    },
    "pusht": {
        "expert_rollouts": 90,
        "lewm_expert_rollouts": 90,
        "lewm_perturbation_rollouts": 540,
    },
    "pick_place": {
        "expert_rollouts": 50,
        "lewm_expert_rollouts": 50,
        "lewm_random_play_rollouts": 10,
    },
    "insert": {
        "expert_rollouts": 50,
        "lewm_expert_rollouts": 50,
        "lewm_random_play_rollouts": 10,
    },
    "push_tennis": {
        "expert_rollouts": 50,
        "lewm_expert_rollouts": 50,
        "lewm_random_play_rollouts": 10,
    },
}


def validate_paper_config(config: dict[str, Any]) -> None:
    """Fail early when a runnable config contradicts Tables B.1--B.3."""
    task = config.get("task", {})
    name = str(task.get("name", "")).lower()
    if name not in PAPER_TASKS:
        return
    expected = PAPER_TASKS[name]
    policy = config.get("policy", {})
    kind = str(policy.get("kind", config.get("stage2", {}).get("policy_kind", ""))).lower()
    is_flow = kind in {"flow", "fm", "flow_matching"}
    is_diffusion = kind in {"diffusion", "ddpm", "dp"}
    if not (is_flow or is_diffusion):
        raise ValueError(f"Unknown policy family in config: {kind!r}")

    expected_values = {
        "task.paper_observation_shape": list(expected.observation_shape),
        "task.latent_dim": expected.latent_dim,
        "task.action_dim": expected.action_dim,
        "task.state_dim": expected.state_dim,
        "task.calibration_rollouts": expected.calibration_rollouts,
        "task.average_episode_length": expected.average_episode_length,
        "task.maximum_episode_length": expected.maximum_episode_length,
        "task.obs_history": expected.fm_obs_history if is_flow else 2,
        "task.action_horizon": 8 if is_flow else 16,
        "task.execution_horizon": 4 if is_flow else 8,
        "stage2.policy_kind": "flow" if is_flow else "diffusion",
        "policy.integration_steps" if is_flow else "policy.diffusion_steps": 16 if is_flow else 100,
        "stage2.epochs": 20
        if is_flow
        else (2000 if name in {"sorting", "stacking", "pusht"} else 600),
        "stage2.batch_size": 256 if is_flow or name in {"sorting", "stacking", "pusht"} else 64,
        "calibration.future_action_samples": 32
        if is_flow
        else (256 if name in {"sorting", "stacking", "pusht"} else 30),
        "runtime.future_action_samples": 32
        if is_flow
        else (256 if name in {"sorting", "stacking", "pusht"} else 30),
    }
    expected_lr = 5e-4 if is_flow else (3e-5 if name in {"sorting", "stacking", "pusht"} else 1e-4)
    expected_values["stage2.lr"] = expected_lr
    for key, value in PAPER_DATASETS[name].items():
        expected_values[f"dataset.{key}"] = value

    errors = []
    for dotted_key, expected_value in expected_values.items():
        actual = _get(config, dotted_key)
        if actual != expected_value:
            errors.append(f"{dotted_key}={actual!r}, paper={expected_value!r}")
    if str(config.get("calibration", {}).get("threshold_style")) != "time_varying":
        errors.append("calibration.threshold_style must be 'time_varying' (Appendix A.4)")
    if not bool(config.get("runtime", {}).get("require_both_thresholds", False)):
        errors.append("runtime.require_both_thresholds must be true (Eq. 2)")
    predictor = str(config.get("world_model", {}).get("predictor_kind", ""))
    expected_predictor = "transformer" if is_flow else "mlp"
    if predictor != expected_predictor:
        errors.append(f"world_model.predictor_kind={predictor!r}, paper={expected_predictor!r}")
    if errors:
        raise ValueError("Paper-alignment validation failed:\n- " + "\n- ".join(errors))


def _get(mapping: dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value
