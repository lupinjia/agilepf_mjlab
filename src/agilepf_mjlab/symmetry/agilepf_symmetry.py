"""Left/right symmetry augmentation for AgilePF observations and actions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# AgilePF has six actuated joints in MJCF order: left hip/thigh/calf,
# followed by right hip/thigh/calf.
_LEFT_INDICES = [0, 1, 2]
_RIGHT_INDICES = [3, 4, 5]
_FLIP_SIGN_INDICES = [0, 3]

_ACTOR_FRAME_DIMS = {
  "command": 3,
  "base_ang_vel": 3,
  "projected_gravity": 3,
  "joint_pos": 6,
  "joint_vel": 6,
  "actions": 6,
}
_ACTOR_FRAME_DIM = sum(_ACTOR_FRAME_DIMS.values())


def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
  """Swap left/right joints and flip hip-roll signs."""
  switched = joint_data.clone()
  switched[..., _LEFT_INDICES] = joint_data[..., _RIGHT_INDICES]
  switched[..., _RIGHT_INDICES] = joint_data[..., _LEFT_INDICES]
  switched[..., _FLIP_SIGN_INDICES] *= -1.0
  return switched


def _mirror_joint_history_block(
  block: torch.Tensor,
  history_len: int,
  frame_dim: int,
  switch_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
  """Mirror a flattened joint block independently for each history frame."""
  framed = block.reshape(*block.shape[:-1], history_len, frame_dim)
  return switch_fn(framed).reshape(block.shape)


def _transform_actor_obs_left_right(
  obs: torch.Tensor, history_len: int
) -> torch.Tensor:
  """Mirror the configured concatenated AgilePF actor observation terms."""
  obs = obs.clone()
  device = obs.device
  h = history_len

  start = 0
  end = start + _ACTOR_FRAME_DIMS["command"] * h
  obs[:, start:end] *= torch.tensor([1, -1, -1], device=device).repeat(h)

  start = end
  end = start + _ACTOR_FRAME_DIMS["base_ang_vel"] * h
  obs[:, start:end] *= torch.tensor([-1, 1, -1], device=device).repeat(h)

  start = end
  end = start + _ACTOR_FRAME_DIMS["projected_gravity"] * h
  obs[:, start:end] *= torch.tensor([1, -1, 1], device=device).repeat(h)

  for term in ("joint_pos", "joint_vel", "actions"):
    start = end
    end = start + _ACTOR_FRAME_DIMS[term] * h
    obs[:, start:end] = _mirror_joint_history_block(
      obs[:, start:end], h, _ACTOR_FRAME_DIMS[term], _switch_joints_left_right
    )

  return obs


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
  """Apply the AgilePF left/right transformation to six joint actions."""
  return _switch_joints_left_right(actions.clone())


@torch.no_grad()
def compute_symmetric_states_agilepf(
  env: ManagerBasedRlEnv,
  obs: TensorDict | None = None,
  actions: torch.Tensor | None = None,
):
  """Augment actor/critic observations and actions with left/right symmetry.

  The actor observation is assumed to concatenate command, base angular
  velocity, projected gravity, joint position, joint velocity, and actions in
  that order. Only the actor prefix of critic observations is transformed.
  """
  if obs is not None:
    batch_size = obs.batch_size[0]
    obs_aug = obs.repeat(2)  # type: ignore[reportArgumentType]

    obs_aug["actor"][:batch_size] = obs["actor"][:]
    obs_aug["critic"][:batch_size] = obs["critic"][:]
    actor_h = env.cfg.observations["actor"].history_length
    critic_h = env.cfg.observations["critic"].history_length
    assert actor_h is not None and critic_h is not None

    obs_aug["actor"][batch_size : 2 * batch_size] = _transform_actor_obs_left_right(
      obs["actor"][:], actor_h
    )
    actor_prefix = _ACTOR_FRAME_DIM * critic_h
    obs_aug["critic"][batch_size : 2 * batch_size, :actor_prefix] = (
      _transform_actor_obs_left_right(obs["critic"][:, :actor_prefix], critic_h)
    )
    obs_aug["critic"][batch_size : 2 * batch_size, actor_prefix:] = obs["critic"][
      :, actor_prefix:
    ]
  else:
    obs_aug = None

  if actions is not None:
    batch_size = actions.shape[0]
    actions_aug = torch.zeros(
      batch_size * 2, actions.shape[1], device=actions.device, dtype=actions.dtype
    )
    actions_aug[:batch_size] = actions
    actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
  else:
    actions_aug = None

  return obs_aug, actions_aug
