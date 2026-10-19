"""AgilePF-specific reward-based command curriculum types."""

from __future__ import annotations

from typing import TypedDict, cast

import torch
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg


class RewardVelocityStage(TypedDict):
  """Stage configuration for reward-based velocity curriculum."""

  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class commands_vel_reward_based:
  """Increase velocity command ranges when episode rewards pass a threshold."""

  def __init__(self, cfg, env):
    self.cfg, self.env = cfg, env
    self.current_stage_idx = 0
    self.reward_weight = None
    self.episode_reward_buffer = []
    self.episode_buffer_size = 10
    self.last_episode_reward = 0.0
    self.last_update_step = -1000

  def reset(self, env_ids=None):
    del env_ids
    key = f"Episode_Reward/{self.cfg.params.get('reward_name', '')}"
    if key in self.env.extras.get("log", {}):
      reward = self.env.extras["log"][key]
      if isinstance(reward, torch.Tensor):
        reward = reward.item()
      self.last_episode_reward = reward
      self.episode_reward_buffer.append(reward)
      if len(self.episode_reward_buffer) > self.episode_buffer_size:
        self.episode_reward_buffer.pop(0)

  def _get_smoothed_reward(self) -> float:
    return (
      sum(self.episode_reward_buffer) / len(self.episode_reward_buffer)
      if self.episode_reward_buffer
      else self.last_episode_reward
    )

  def __call__(
    self,
    env,
    env_ids,
    command_name: str,
    reward_name: str,
    min_reward_ratio: float,
    velocity_stages: list[RewardVelocityStage],
    min_update_interval: int = 1000,
  ) -> dict[str, torch.Tensor]:
    del env_ids
    term = env.command_manager.get_term(command_name)
    assert term is not None
    command_cfg = cast(UniformVelocityCommandCfg, term.cfg)
    if self.reward_weight is None:
      try:
        self.reward_weight = env.reward_manager.get_term_cfg(reward_name).weight
      except ValueError:
        self.reward_weight = 1.0
    reward = self._get_smoothed_reward()
    step = env.common_step_counter
    if (
      step - self.last_update_step >= min_update_interval
      and self.current_stage_idx < len(velocity_stages) - 1
      and reward >= min_reward_ratio * self.reward_weight
    ):
      self.current_stage_idx += 1
      self.last_update_step = step
    stage = velocity_stages[self.current_stage_idx]
    for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
      if stage.get(name) is not None:
        setattr(command_cfg.ranges, name, stage[name])
    ranges = command_cfg.ranges
    return {
      "stage": torch.tensor(self.current_stage_idx),
      "reward": torch.tensor(reward),
      "lin_vel_x_min": torch.tensor(ranges.lin_vel_x[0]),
      "lin_vel_x_max": torch.tensor(ranges.lin_vel_x[1]),
      "lin_vel_y_min": torch.tensor(ranges.lin_vel_y[0]),
      "lin_vel_y_max": torch.tensor(ranges.lin_vel_y[1]),
      "ang_vel_z_min": torch.tensor(ranges.ang_vel_z[0]),
      "ang_vel_z_max": torch.tensor(ranges.ang_vel_z[1]),
    }
