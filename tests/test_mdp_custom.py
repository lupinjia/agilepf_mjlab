"""Focused regression tests for AgilePF-specific MDP terms."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from agilepf_mjlab.mdp import events
from agilepf_mjlab.mdp.events import reset_from_sitting
from agilepf_mjlab.mdp.gait_command import GaitCommand


def test_sitting_reset_sets_pose_and_joint_state(monkeypatch):
  robot = Mock()
  robot.find_joints.return_value = ([0, 1], ["right", "left"])
  scene = type("Scene", (dict,), {})(robot=robot)
  scene.env_origins = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
  env = SimpleNamespace(scene=scene, num_envs=2, device="cpu")
  monkeypatch.setattr(torch, "rand", lambda n, **kwargs: torch.ones(n) * 0.1)
  monkeypatch.setattr(events, "reset_root_state_uniform", Mock())
  monkeypatch.setattr(events, "reset_joints_by_offset", Mock())
  reset_from_sitting(
    env,
    torch.tensor([1]),
    1.0,
    (0.0, 0.0, 0.18),
    {"left": 0.2, "right": -0.2},
    {},
    {},
    (-0.1, 0.1),
    (-0.5, 0.5),
  )
  torch.testing.assert_close(
    robot.write_joint_state_to_sim.call_args.args[0], torch.tensor([[-0.2, 0.2]])
  )
  torch.testing.assert_close(
    robot.write_root_link_pose_to_sim.call_args.args[0][:, :3],
    torch.tensor([[3.0, 4.0, 0.18]]),
  )


def test_gait_mask_preserves_legacy_flatten_behavior():
  command = SimpleNamespace(
    num_envs=3,
    device=torch.device("cpu"),
    phi=torch.tensor([[0.1], [0.6], [0.2]]),
    theta=torch.zeros((3, 2)),
    a_swing=0.0,
    b_swing=torch.pi,
    b_stance=2 * torch.pi,
    gait_function_type="step",
  )
  result = GaitCommand._uniped_periodic_gait(
    command, torch.ones(3), torch.ones(3), 0, torch.zeros(3)
  )
  # The legacy flatten() maps the selected [B, 1] row coordinate into the first
  # dimension; preserve this historical behavior rather than changing indexing.
  torch.testing.assert_close(result[:, 0], torch.tensor([-2.0, -1.0, -1.0]))
