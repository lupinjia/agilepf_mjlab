"""AgilePF gait clock command (local custom MDP term)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class GaitCommand(CommandTerm):
  """Generate gait clock signals and compute the periodic biped reward."""

  cfg: GaitCommandCfg

  def __init__(self, cfg: GaitCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.gait_time = torch.zeros(self.num_envs, 1, device=self.device)
    self.phi = torch.zeros(self.num_envs, 1, device=self.device)
    self.theta = torch.zeros(self.num_envs, 2, device=self.device)
    self.theta[:, 0], self.theta[:, 1] = cfg.theta_left, cfg.theta_right
    self.clock_input = torch.zeros(self.num_envs, 4, device=self.device)
    self.gait_period = cfg.gait_period
    self.gait_function_type = cfg.gait_function_type
    self.a_swing, self.b_swing, self.b_stance = (
      0.0,
      cfg.swing_ratio * 2 * torch.pi,
      2 * torch.pi,
    )
    self.kappa = cfg.kappa

  @property
  def command(self) -> torch.Tensor:
    return self.clock_input

  def _update_clock(self, dt: float | torch.Tensor) -> None:
    self.gait_time += dt
    over_limit = self.gait_time >= (self.gait_period - dt / 2)
    self.gait_time[over_limit] = 0.0
    self.phi = self.gait_time / self.gait_period
    for i in range(2):
      phase = 2 * torch.pi * (self.phi + self.theta[:, i].unsqueeze(1))
      self.clock_input[:, i] = torch.sin(phase).squeeze(-1)
      self.clock_input[:, i + 2] = torch.cos(phase).squeeze(-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.theta[env_ids, 0] = self.cfg.theta_left
    self.theta[env_ids, 1] = self.cfg.theta_right
    self.gait_time[env_ids] = (
      torch.rand(len(env_ids), 1, device=self.device) * self.gait_period
    )
    self.phi[env_ids] = self.gait_time[env_ids] / self.gait_period
    self.clock_input[env_ids, :] = 0.0

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    del env_ids
    if isinstance(dt, torch.Tensor):
      dt = dt.unsqueeze(-1)
    self._update_clock(dt)

  def get_periodic_gait_reward(
    self,
    left_foot_force,
    left_foot_vel,
    right_foot_force,
    right_foot_vel,
    stance_still_mask,
  ):
    left = self._uniped_periodic_gait(
      left_foot_force, left_foot_vel, 0, stance_still_mask
    )
    right = self._uniped_periodic_gait(
      right_foot_force, right_foot_vel, 1, stance_still_mask
    )
    return torch.exp((left + right).squeeze(-1))

  def _uniped_periodic_gait(self, foot_force, foot_vel, foot_idx, stance_still_mask):
    phi = ((self.phi + self.theta[:, foot_idx].unsqueeze(1)) % 1.0) * 2 * torch.pi
    if self.gait_function_type != "step":
      raise NotImplementedError(
        f"Gait function type '{self.gait_function_type}' not implemented; only 'step' is supported."
      )
    exp_C_frc = torch.zeros(self.num_envs, 1, device=self.device)
    exp_C_spd = torch.zeros(self.num_envs, 1, device=self.device)
    swing_mask = (phi >= self.a_swing) & (phi < self.b_swing)
    # Legacy upstream behavior intentionally flattens nonzero indices on [B, 1].
    swing_indices = swing_mask.nonzero(as_tuple=False).flatten()
    exp_C_frc[swing_indices, :] = -1.0
    stance_mask = (phi >= self.b_swing) & (phi < self.b_stance)
    stance_indices = stance_mask.nonzero(as_tuple=False).flatten()
    exp_C_spd[stance_indices, :] = -1.0
    stand_still_indices = stance_still_mask.nonzero(as_tuple=False).flatten()
    exp_C_spd[stand_still_indices, :] = -1.0
    foot_force, foot_vel = foot_force.view(-1, 1), foot_vel.view(-1, 1)
    return exp_C_spd * foot_vel + exp_C_frc * foot_force

  def is_in_swing_phase(self, foot_idx: int) -> torch.Tensor:
    phase = ((self.phi + self.theta[:, foot_idx].unsqueeze(1)) % 1.0) * 2 * torch.pi
    return (phase >= self.a_swing) & (phase < self.b_swing)

  def is_in_stance_phase(self, foot_idx: int) -> torch.Tensor:
    phase = ((self.phi + self.theta[:, foot_idx].unsqueeze(1)) % 1.0) * 2 * torch.pi
    return (phase >= self.b_swing) & (phase < self.b_stance)


@dataclass(kw_only=True)
class GaitCommandCfg(CommandTermCfg):
  gait_period: float = 0.5
  theta_left: float = 0.0
  theta_right: float = 0.5
  swing_ratio: float = 0.5
  gait_function_type: Literal["step", "smooth"] = "step"
  kappa: float = 10.0
  resampling_time_range: tuple[float, float] = (10.0, 20.0)

  def build(self, env: ManagerBasedRlEnv) -> GaitCommand:
    return GaitCommand(self, env)

  def __post_init__(self):
    if self.gait_function_type not in ("step", "smooth"):
      raise ValueError(
        f"gait_function_type must be 'step' or 'smooth', got '{self.gait_function_type}'"
      )
    if self.gait_function_type == "smooth":
      raise NotImplementedError(
        "Smooth gait function is reserved; use gait_function_type='step'."
      )
    if not 0 <= self.swing_ratio <= 1:
      raise ValueError(f"swing_ratio must be in [0, 1], got {self.swing_ratio}")
    if not 0 <= self.theta_left <= 1 or not 0 <= self.theta_right <= 1:
      raise ValueError("Gait phase offsets must be in [0, 1]")
