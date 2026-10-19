"""AgilePF-specific MDP reset event."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.events import (
  reset_joints_by_offset,
  reset_root_state_uniform,
  resolve_env_ids,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_from_sitting(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  sit_init_percent: float,
  sit_pos: tuple[float, float, float],
  sit_joint_angles: dict[str, float],
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]],
  joint_position_range: tuple[float, float],
  joint_velocity_range: tuple[float, float],
  sit_pitch_angle: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset environments to independently sampled sitting or standing poses."""
  if not 0.0 <= sit_init_percent <= 1.0:
    raise ValueError("sit_init_percent must be between 0 and 1")
  env_ids = resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return
  sit_mask = torch.rand(len(env_ids), device=env.device) < sit_init_percent
  standing_ids = env_ids[~sit_mask]
  if len(standing_ids) > 0:
    reset_root_state_uniform(
      env, standing_ids, pose_range, velocity_range, asset_cfg=asset_cfg
    )
    reset_joints_by_offset(
      env, standing_ids, joint_position_range, joint_velocity_range, asset_cfg=asset_cfg
    )
  env_ids = env_ids[sit_mask]
  if len(env_ids) == 0:
    return
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids, joint_names = asset.find_joints(tuple(sit_joint_angles))
  joint_pos = torch.tensor(
    [sit_joint_angles[name] for name in joint_names], device=env.device
  ).repeat(len(env_ids), 1)
  asset.write_joint_state_to_sim(
    joint_pos,
    torch.zeros_like(joint_pos),
    joint_ids=torch.tensor(joint_ids, device=env.device),
    env_ids=env_ids,
  )
  pos = torch.tensor(sit_pos, device=env.device).repeat(len(env_ids), 1)
  pos += env.scene.env_origins[env_ids]
  zeros = torch.zeros(len(env_ids), device=env.device)
  quat = quat_from_euler_xyz(zeros, zeros + sit_pitch_angle, zeros)
  asset.write_root_link_pose_to_sim(torch.cat((pos, quat), dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(
    torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids
  )
