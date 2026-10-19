"""AgilePF sitting reset regression tests."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from agilepf_mjlab.mdp import events
from agilepf_mjlab.mdp.events import reset_from_sitting
from agilepf_mjlab.tasks.env_cfgs import agilepf_flat_env_cfg


@pytest.mark.parametrize(
  "probability,draw,sitting",
  [(0.0, 0.0, False), (0.5, 0.2, True), (0.5, 0.8, False), (1.0, 0.9, True)],
)
def test_sitting_reset_subset(monkeypatch, probability, draw, sitting):
  cfg = agilepf_flat_env_cfg()
  params = dict(cfg.events["reset_from_sitting"].params)
  params["sit_init_percent"] = probability
  robot = Mock()
  # Resolve in a different order to catch accidental dictionary-order writes.
  names = list(reversed(params["sit_joint_angles"]))
  robot.find_joints.return_value = (list(range(6)), names)
  origins = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [6.0, 7.0, 0.0]])
  scene = type("Scene", (dict,), {})(robot=robot)
  scene.env_origins = origins
  env = SimpleNamespace(scene=scene, num_envs=3, device="cpu")
  monkeypatch.setattr(torch, "rand", lambda n, **kwargs: torch.full((n,), draw))
  root_reset = Mock()
  joint_reset = Mock()
  monkeypatch.setattr(events, "reset_root_state_uniform", root_reset)
  monkeypatch.setattr(events, "reset_joints_by_offset", joint_reset)
  ids = torch.tensor([1, 2])
  reset_from_sitting(env, ids, **params)
  assert robot.write_joint_state_to_sim.called == sitting
  assert robot.write_root_link_pose_to_sim.called == sitting
  assert root_reset.called == (not sitting)
  assert joint_reset.called == (not sitting)
  if sitting:
    joint_call = robot.write_joint_state_to_sim.call_args
    expected = torch.tensor([params["sit_joint_angles"][n] for n in names]).repeat(2, 1)
    torch.testing.assert_close(joint_call.args[0], expected)
    assert torch.count_nonzero(joint_call.args[1]) == 0
    torch.testing.assert_close(joint_call.kwargs["env_ids"], ids)
    pose = robot.write_root_link_pose_to_sim.call_args.args[0]
    torch.testing.assert_close(
      pose[:, :3], origins[ids] + torch.tensor([0.0, 0.0, 0.18])
    )
    torch.testing.assert_close(
      pose[:, 3:], torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    )
    assert (
      torch.count_nonzero(robot.write_root_link_velocity_to_sim.call_args.args[0]) == 0
    )


def test_sitting_reset_configuration():
  train = agilepf_flat_env_cfg()
  play = agilepf_flat_env_cfg(play=True)
  assert train.events["reset_from_sitting"].params["sit_init_percent"] == 0.5
  assert play.events["reset_from_sitting"].params["sit_init_percent"] == 1.0
  for cfg in (train, play):
    assert "reset_base" not in cfg.events
    assert "reset_robot_joints" not in cfg.events
  params = train.events["reset_from_sitting"].params
  assert params["joint_position_range"] == (-0.1, 0.1)
  assert params["joint_velocity_range"] == (-0.5, 0.5)


def test_mixed_reset_partitions_envs(monkeypatch):
  params = dict(agilepf_flat_env_cfg().events["reset_from_sitting"].params)
  robot = Mock()
  names = list(params["sit_joint_angles"])
  robot.find_joints.return_value = (list(range(6)), names)
  scene = type("Scene", (dict,), {})(robot=robot)
  scene.env_origins = torch.zeros(5, 3)
  env = SimpleNamespace(scene=scene, num_envs=5, device="cpu")
  monkeypatch.setattr(torch, "rand", lambda n, **kwargs: torch.tensor([0.2, 0.8, 0.1]))
  root_reset, joint_reset = Mock(), Mock()
  monkeypatch.setattr(events, "reset_root_state_uniform", root_reset)
  monkeypatch.setattr(events, "reset_joints_by_offset", joint_reset)
  reset_from_sitting(env, torch.tensor([4, 1, 3]), **params)
  torch.testing.assert_close(root_reset.call_args.args[1], torch.tensor([1]))
  torch.testing.assert_close(joint_reset.call_args.args[1], torch.tensor([1]))
  assert root_reset.call_args.args[2:4] == (
    params["pose_range"],
    params["velocity_range"],
  )
  assert joint_reset.call_args.args[2:4] == (
    params["joint_position_range"],
    params["joint_velocity_range"],
  )
  torch.testing.assert_close(
    robot.write_joint_state_to_sim.call_args.kwargs["env_ids"], torch.tensor([4, 3])
  )
