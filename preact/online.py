from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .runtime import PreActOutput, PreActRuntime
from .utils import as_float_tensor


@dataclass
class OnlinePreActAgentConfig:
    obs_history: int = 1
    obs_key: str | None = None
    threshold_step_mode: str = "policy"


class OnlinePreActAgent:
    """Online receding-horizon wrapper around ``PreActRuntime``.

    ``PreActRuntime`` performs one Predict-Evaluate-Act decision. This wrapper
    maintains observation history and applies only the first ``h`` executable
    actions before asking ``PreActRuntime`` to re-plan from the new observation.
    """

    def __init__(
        self,
        runtime: PreActRuntime,
        cfg: OnlinePreActAgentConfig | None = None,
        obs_to_tensor: Callable[[Any], torch.Tensor] | None = None,
    ):
        self.runtime = runtime
        self.cfg = cfg or OnlinePreActAgentConfig()
        self.obs_to_tensor = obs_to_tensor or self._default_obs_to_tensor
        self.history: deque[torch.Tensor] = deque(maxlen=max(int(self.cfg.obs_history), 1))

    def reset(self, observation: Any) -> None:
        self.history.clear()
        obs = self.obs_to_tensor(observation)
        for _ in range(max(int(self.cfg.obs_history), 1)):
            self.history.append(obs.clone())

    def observe(self, observation: Any) -> None:
        self.history.append(self.obs_to_tensor(observation))

    @torch.no_grad()
    def select_action(self, *, step_index: int) -> PreActOutput:
        if not self.history:
            raise RuntimeError("Call reset(observation) before select_action().")
        obs = torch.stack(list(self.history), dim=0).unsqueeze(0)
        return self.runtime.step(obs, step_index=step_index)

    def _default_obs_to_tensor(self, observation: Any) -> torch.Tensor:
        if isinstance(observation, dict):
            if self.cfg.obs_key is not None:
                observation = observation[self.cfg.obs_key]
            else:
                for key in (
                    "obs_embedding",
                    "obs",
                    "observation",
                    "state",
                    "pixels",
                    "image",
                    "rgb",
                ):
                    if key in observation:
                        observation = observation[key]
                        break
                else:
                    raise KeyError(
                        "Observation is a dict. Set obs_key or include one of: "
                        "obs_embedding, obs, observation, state, pixels, image, rgb."
                    )
        tensor = as_float_tensor(observation)
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        return tensor


def run_receding_horizon_episode(
    env: Any,
    agent: OnlinePreActAgent,
    *,
    max_policy_steps: int | None = None,
    max_env_steps: int | None = None,
) -> dict[str, Any]:
    """Run online PreAct deployment in a Gym-style environment."""

    observation, reset_info = _reset_env(env)
    agent.reset(observation)
    policy_step = 0
    env_step = 0
    total_reward = 0.0
    done = False
    last_info: dict[str, Any] = dict(reset_info)
    outputs: list[PreActOutput] = []

    while not done:
        if max_policy_steps is not None and policy_step >= max_policy_steps:
            break
        step_index = _threshold_step_index(agent.cfg.threshold_step_mode, policy_step, env_step)
        output = agent.select_action(step_index=step_index)
        outputs.append(output)
        for action in output.executable_action[0]:
            observation, reward, done, info = _step_env(env, action.detach().cpu())
            total_reward += float(reward)
            env_step += 1
            last_info = info
            agent.observe(observation)
            if done or (max_env_steps is not None and env_step >= max_env_steps):
                done = True
                break
        policy_step += 1

    return {
        "reward": total_reward,
        "policy_steps": policy_step,
        "env_steps": env_step,
        "done": done,
        "success": _success_from_info(last_info),
        "last_info": last_info,
        "outputs": outputs,
    }


def _reset_env(env: Any) -> tuple[Any, dict[str, Any]]:
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
        return obs, dict(info or {})
    return result, {}


def _step_env(env: Any, action: torch.Tensor) -> tuple[Any, float, bool, dict[str, Any]]:
    action_np = action.numpy()
    result = env.step(action_np)
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated or truncated), dict(info or {})
    if isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward), bool(done), dict(info or {})
    raise TypeError("env.step(action) must return Gym-style 4- or 5-tuples.")


def _success_from_info(info: dict[str, Any]) -> bool | None:
    for key in ("success", "is_success", "successful"):
        if key in info:
            return bool(info[key])
    return None


def _threshold_step_index(mode: str, policy_step: int, env_step: int) -> int:
    if mode == "policy":
        return int(policy_step)
    if mode == "env":
        return int(env_step)
    raise ValueError("threshold_step_mode must be 'policy' or 'env'.")
