"""AgilePF-specific observations for dynamics randomization and gait."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.actuator import BuiltinPositionActuator, IdealPdActuator, XmlActuator
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .gait_command import GaitCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def geom_friction_obs(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids]
  return env.sim.model.geom_friction[:, geom_ids, 0]


def encoder_bias_obs(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.encoder_bias


def body_com_offset_obs(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  ids = asset.indexing.body_ids[asset_cfg.body_ids]
  current = env.sim.model.body_ipos[:, ids]
  default = env.sim.get_default_field("body_ipos")[ids]
  return (current - default).squeeze(1)


def actuator_delay_lags_obs(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  lags = torch.zeros(env.num_envs, len(asset.actuators), device=env.device)
  for i, actuator in enumerate(asset.actuators):
    if actuator.has_delay and actuator._delay_buffer is not None:
      lags[:, i] = (
        actuator._delay_buffer._current_lags.float()
        / actuator._delay_buffer._buffer.current_length
      )
  return lags


def pd_gains_scale_obs(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_gain = env.sim.get_default_field("actuator_gainprm")
  default_bias = env.sim.get_default_field("actuator_biasprm")
  kp_scales, kd_scales = [], []
  for actuator in asset.actuators:
    ids = actuator.global_ctrl_ids
    if isinstance(actuator, BuiltinPositionActuator) or (
      isinstance(actuator, XmlActuator) and actuator.command_field == "position"
    ):
      kp = env.sim.model.actuator_gainprm[:, ids, 0] / default_gain[ids, 0]
      kd = env.sim.model.actuator_biasprm[:, ids, 2] / default_bias[ids, 2]
    elif isinstance(actuator, IdealPdActuator):
      kp, kd = (
        actuator.stiffness / actuator.default_stiffness,
        actuator.damping / actuator.default_damping,
      )
    else:
      raise ValueError(f"Unknown actuator type: {type(actuator)}")
    kp_scales.append(kp)
    kd_scales.append(kd)
  return torch.cat(kp_scales + kd_scales, dim=-1)


def clock_input_obs(env: ManagerBasedRlEnv, gait_command_name: str) -> torch.Tensor:
  command = cast(GaitCommand, env.command_manager.get_term(gait_command_name))
  assert command is not None
  return command.clock_input
