from __future__ import annotations

from dataclasses import dataclass

import torch

from .cem import CEMConfig, CEMOptimizer
from .rnd import RandomNetworkDistillation
from .scoring import ActionEntropyScorer, ObservationDeviationScorer, Thresholds, normalize_score
from .world_model import LatentWorldModel


@dataclass
class PreActRuntimeConfig:
    enabled: bool = True
    require_both_thresholds: bool = True
    execution_horizon: int = 1
    trigger_policy_samples: int = 32
    future_action_samples: int = 32
    future_action_latent_index: int | None = None
    future_action_reduce: str = "sum"
    future_observation_reduce: str = "sum"
    obs_weight: float = 1.0
    action_weight: float = 1.0
    anchor_weight: float = 1.0
    normalize_costs: bool = True


@dataclass
class PreActOutput:
    action_pred: torch.Tensor
    executable_action: torch.Tensor
    triggered: torch.Tensor
    obs_score: torch.Tensor
    action_score: torch.Tensor
    corrected_obs_score: torch.Tensor | None = None
    corrected_action_score: torch.Tensor | None = None
    total_cost: torch.Tensor | None = None


class PreActRuntime:
    """Single-decision Predict-Evaluate-Act correction kernel.

    This class implements Algorithm 1 for one re-planning instant: encode the
    current observation history, imagine the base action consequence, evaluate
    DRS-O/DRS-A, and optionally optimize a corrected action chunk. It deliberately
    does not own an environment loop; online receding-horizon execution is handled
    by ``preact.online``.
    """

    def __init__(
        self,
        *,
        world_model: LatentWorldModel,
        policy,
        drs_o: RandomNetworkDistillation,
        action_scorer: ActionEntropyScorer,
        thresholds: Thresholds,
        config: PreActRuntimeConfig | None = None,
        cem: CEMOptimizer | None = None,
    ):
        self.world_model = world_model
        self.policy = policy
        self.cfg = config or PreActRuntimeConfig()
        self.obs_scorer = ObservationDeviationScorer(
            drs_o, reduce=self.cfg.future_observation_reduce
        )
        self.action_scorer = action_scorer
        self.thresholds = thresholds
        self.action_scorer.configure_from_thresholds(thresholds)
        self.cem = cem or CEMOptimizer(CEMConfig())

    def to(self, device: torch.device | str) -> PreActRuntime:
        self.world_model.to(device)
        self.policy.to(device)
        self.obs_scorer.to(device)
        return self

    @torch.no_grad()
    def step(self, obs: torch.Tensor, *, step_index: int = 0) -> PreActOutput:
        device = next(self.world_model.parameters()).device
        obs = obs.to(device).float()
        latent = self.world_model.encode(obs)

        base_action = self.policy.sample_action_chunks(latent, num_samples=1)[:, 0]
        base_future = self.world_model.rollout_from_latent(latent, base_action[:, None])
        obs_score = self.obs_scorer.score(base_future)[:, 0]
        action_score = self.action_scorer.score_candidates_by_future_policy(
            policy=self.policy,
            future_latents=base_future,
            num_samples=max(int(self.cfg.future_action_samples), 1),
            latent_index=self.cfg.future_action_latent_index,
            reduce=self.cfg.future_action_reduce,
        )[:, 0]
        triggered = self._should_correct(obs_score, action_score, step_index)

        if not self.cfg.enabled or not bool(triggered.any()):
            return self._output(base_action, triggered, obs_score, action_score)

        policy_samples = self.policy.sample_action_chunks(
            latent,
            num_samples=max(int(self.cfg.trigger_policy_samples), 1),
        )

        tau_o = self.thresholds.obs_at(step_index)
        tau_a = self.thresholds.action_at(step_index)
        scale_o = self.thresholds.obs_scale_at(step_index)
        scale_a = self.thresholds.action_scale_at(step_index)

        def cost_fn(candidates: torch.Tensor):
            future = self.world_model.rollout_from_latent(latent, candidates)
            obs_raw = self.obs_scorer.score(future)
            action_raw = self.action_scorer.score_candidates_by_future_policy(
                policy=self.policy,
                future_latents=future,
                num_samples=max(int(self.cfg.future_action_samples), 1),
                latent_index=self.cfg.future_action_latent_index,
                reduce=self.cfg.future_action_reduce,
            )
            if self.cfg.normalize_costs:
                obs_cost = normalize_score(obs_raw, tau_o, scale_o)
                action_cost = normalize_score(action_raw, tau_a, scale_a)
            else:
                obs_cost = obs_raw
                action_cost = action_raw
            if self.cfg.anchor_weight > 0:
                action_cost = action_cost + float(self.cfg.anchor_weight) * _anchor_cost(
                    candidates,
                    base_action,
                )
            return obs_cost, action_cost, {"obs_raw": obs_raw, "action_raw": action_raw}

        cem_result = self.cem.optimize(
            base_action,
            cost_fn,
            initial_samples=policy_samples,
            obs_weight=float(self.cfg.obs_weight),
            action_weight=float(self.cfg.action_weight),
        )
        corrected = torch.where(triggered[:, None, None], cem_result.action_sequence, base_action)
        output = self._output(corrected, triggered, obs_score, action_score)
        output.corrected_obs_score = cem_result.obs_cost
        output.corrected_action_score = cem_result.action_cost
        output.total_cost = cem_result.total_cost
        return output

    def _should_correct(
        self,
        obs_score: torch.Tensor,
        action_score: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        obs_bad = obs_score > self.thresholds.obs_at(step_index)
        action_bad = action_score > self.thresholds.action_at(step_index)
        if self.cfg.require_both_thresholds:
            return obs_bad & action_bad
        return obs_bad | action_bad

    def _output(
        self,
        action_pred: torch.Tensor,
        triggered: torch.Tensor,
        obs_score: torch.Tensor,
        action_score: torch.Tensor,
    ) -> PreActOutput:
        h = min(max(int(self.cfg.execution_horizon), 1), action_pred.shape[1])
        return PreActOutput(
            action_pred=action_pred,
            executable_action=action_pred[:, :h],
            triggered=triggered,
            obs_score=obs_score,
            action_score=action_score,
        )


def _anchor_cost(candidates: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
    return (candidates - base_action[:, None]).pow(2).mean(dim=(-1, -2))
