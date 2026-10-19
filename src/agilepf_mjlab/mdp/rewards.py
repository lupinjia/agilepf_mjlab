"""AgilePF-specific velocity and gait reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from .gait_command import GaitCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _linear_velocity(env, asset: Entity, asset_cfg: SceneEntityCfg, use_root: bool):
  if use_root:
    return asset.data.root_link_lin_vel_b
  assert len(asset_cfg.body_ids) == 1, "Only support specify one body"
  body_id = asset_cfg.body_ids[0]
  velocity_w = asset.data.body_link_lin_vel_w[:, body_id, :]
  quat_w = asset.data.body_link_quat_w[:, body_id, :]
  return quat_apply_inverse(quat_w, velocity_w)


def track_linear_x_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  use_root: bool = True,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = _linear_velocity(env, asset, asset_cfg, use_root)
  return torch.exp(-torch.square(command[:, 0] - actual[:, 0]) / std**2)


def track_linear_y_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  use_root: bool = True,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = _linear_velocity(env, asset, asset_cfg, use_root)
  return torch.exp(-torch.square(command[:, 1] - actual[:, 1]) / std**2)


def track_angular_velocity_yaw(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  use_root: bool = True,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  if use_root:
    actual = asset.data.root_link_ang_vel_b
  else:
    assert len(asset_cfg.body_ids) == 1, "Only support specify one body"
    actual = asset.data.body_link_ang_vel_b[:, asset_cfg.body_ids, :].squeeze(1)
  return torch.exp(-torch.square(command[:, 2] - actual[:, 2]) / std**2)


def body_lin_vel_z_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  body_id = asset_cfg.body_ids
  velocity_w = asset.data.body_link_lin_vel_w[:, body_id, :]
  quat_w = asset.data.body_link_quat_w[:, body_id, :]
  velocity_b = quat_apply_inverse(quat_w, velocity_w)
  return torch.sum(torch.square(velocity_b[:, :, 2]), dim=1)


def feet_clearance_exp(
  env: ManagerBasedRlEnv,
  std: float,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_clearance_exp requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)
  cost = torch.sum(torch.square(foot_height - target_height) * vel_norm, dim=1)
  reward = torch.exp(-cost / std**2)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      active = (
        torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > command_threshold
      ).float()
      reward *= active
  return reward


def biped_periodic_gait(
  env: ManagerBasedRlEnv,
  gait_command_name: str,
  velocity_command_name: str,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_threshold: float = 0.1,
  stand_still_when_command_zero: bool = True,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  gait_command = env.command_manager.get_term(gait_command_name)
  assert isinstance(gait_command, GaitCommand)
  foot_vel_norm = torch.norm(
    asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :], dim=-1
  )
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  force_magnitude = torch.norm(sensor.data.force, dim=-1)
  if stand_still_when_command_zero:
    command = env.command_manager.get_command(velocity_command_name)
    assert command is not None
    still_mask = (
      torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) < command_threshold
    ).float()
  else:
    still_mask = torch.zeros_like(force_magnitude[:, 0])
  return gait_command.get_periodic_gait_reward(
    force_magnitude[:, 0],
    foot_vel_norm[:, 0],
    force_magnitude[:, 1],
    foot_vel_norm[:, 1],
    still_mask,
  )
