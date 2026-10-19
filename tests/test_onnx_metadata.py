from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import pytest
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from onnx import TensorProto, helper, numpy_helper

from agilepf_mjlab.onnx_metadata import (
  attach_agilepf_metadata,
  build_agilepf_metadata,
)
from agilepf_mjlab.tasks.env_cfgs import agilepf_flat_env_cfg
from agilepf_mjlab.tasks.rl_cfg import agilepf_ppo_runner_cfg
from agilepf_mjlab.tasks.runner import AgilePFVelocityOnPolicyRunner


@pytest.fixture(scope="module")
def agilepf_env():
  cfg = agilepf_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  yield env
  env.close()


def _write_test_policy(path: Path) -> None:
  observation_dim = 3 * (3 + 3 + 3 + 6 + 6 + 6)
  action_dim = 6
  graph = helper.make_graph(
    [
      helper.make_node("Gemm", ["obs", "weights", "bias"], ["actions"]),
    ],
    "agilepf_policy_test",
    [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, observation_dim])],
    [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, action_dim])],
    initializer=[
      numpy_helper.from_array(
        np.zeros((observation_dim, action_dim), dtype=np.float32), "weights"
      ),
      numpy_helper.from_array(np.zeros((action_dim,), dtype=np.float32), "bias"),
    ],
  )
  onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path)


def test_grouped_schema_matches_agilepf_deployment_format(
  agilepf_env, tmp_path: Path
) -> None:
  onnx_path = tmp_path / "policy.onnx"
  _write_test_policy(onnx_path)

  metadata = build_agilepf_metadata(
    agilepf_env,
    onnx_path,
    run_path="local",
    experiment_name="agilepf_velocity",
    run_name="flat",
  )

  assert metadata["onnx_input_names"] == ["obs"]
  assert metadata["onnx_input_shapes"] == [[1, 81]]
  assert metadata["onnx_input_groups"] == ["actor"]
  assert metadata["onnx_output_names"] == ["actions"]
  assert metadata["onnx_output_shapes"] == [[1, 6]]
  assert metadata["obs_group.obs.history_length"] == 3
  assert metadata["obs_group.obs.flatten_history_dim"] is True
  assert metadata["obs_group.obs.term_names"] == [
    "command",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
  ]
  assert metadata["obs_group.obs.term_history_lengths"] == [3] * 6
  assert metadata["obs_group.obs.term_clips"] == [[-float("inf"), float("inf")]] * 6
  assert metadata["obs_group.obs.term_scales"][:4] == [
    [1.0, 1.0, 0.25],
    0.25,
    1.0,
    1.0,
  ]
  assert metadata["obs_group.obs.term_scales"][4] == pytest.approx(0.05)
  assert metadata["obs_group.obs.term_scales"][5] == pytest.approx(0.25)
  assert metadata["joint_names"] == [
    "left_hip_joint",
    "left_thigh_joint",
    "left_calf_joint",
    "right_hip_joint",
    "right_thigh_joint",
    "right_calf_joint",
  ]
  assert metadata["experiment_name"] == "agilepf_velocity"
  assert metadata["run_name"] == "flat"
  for legacy_key in (
    "command_names",
    "observation_names",
    "observation_terms_scale",
    "observation_terms_flatten_history_dim",
    "observation_terms_history_length",
    "observation_terms_clip",
  ):
    assert legacy_key not in metadata


def test_metadata_serialization_is_exact_idempotent_and_graph_preserving(
  agilepf_env, tmp_path: Path
) -> None:
  onnx_path = tmp_path / "policy.onnx"
  _write_test_policy(onnx_path)
  model = onnx.load(onnx_path)
  unrelated = model.metadata_props.add()
  unrelated.key = "vendor.custom"
  unrelated.value = "keep-me"
  stale = model.metadata_props.add()
  stale.key = "observation_names"
  stale.value = "legacy"
  onnx.save(model, onnx_path)
  graph_before = onnx.load(onnx_path).graph.SerializeToString()

  metadata = build_agilepf_metadata(
    agilepf_env,
    onnx_path,
    run_path="local",
    experiment_name="agilepf_velocity",
    run_name="flat",
  )
  attach_agilepf_metadata(onnx_path, metadata)
  first = [(entry.key, entry.value) for entry in onnx.load(onnx_path).metadata_props]
  attach_agilepf_metadata(onnx_path, metadata)
  actual = onnx.load(onnx_path)
  second = [(entry.key, entry.value) for entry in actual.metadata_props]

  props = dict(second)
  assert props["onnx_input_shapes"] == "1.000;81.000"
  assert props["onnx_output_shapes"] == "1.000;6.000"
  assert props["run_path"] == "local"
  assert props["experiment_name"] == "agilepf_velocity"
  assert props["run_name"] == "flat"
  assert props["joint_stiffness"] == "30.021,30.021,30.021,30.021,30.021,30.021"
  assert props["joint_damping"] == "2.389,2.389,2.389,2.389,2.389,2.389"
  assert props["default_joint_pos"] == "0.000,0.500,-0.500,0.000,0.500,-0.500"
  assert props["action_scale"] == "0.749,0.749,0.749,0.749,0.749,0.749"
  assert props["obs_group.obs.history_length"] == "3"
  assert props["obs_group.obs.flatten_history_dim"] == "True"
  assert props["obs_group.obs.term_names"] == (
    "command,base_ang_vel,projected_gravity,joint_pos,joint_vel,actions"
  )
  assert props["obs_group.obs.term_clips"] == (
    "-inf;inf,-inf;inf,-inf;inf,-inf;inf,-inf;inf,-inf;inf"
  )
  assert props["obs_group.obs.term_scales"] == (
    "1.000;1.000;0.250,0.250,1.000,1.000,0.050,0.250"
  )
  assert props["vendor.custom"] == "keep-me"
  assert "observation_names" not in props
  assert len(props) == len(second)
  assert first == second
  assert actual.graph.SerializeToString() == graph_before
  onnx.checker.check_model(actual)


def test_runner_save_attaches_grouped_metadata_automatically(
  tmp_path: Path,
) -> None:
  cfg = agilepf_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  checkpoint_dir = tmp_path / "checkpoint"
  checkpoint_dir.mkdir()
  runner = None
  try:
    agent_cfg = asdict(agilepf_ppo_runner_cfg(max_iterations=1, run_name="save-test"))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg["clip_actions"])
    runner = AgilePFVelocityOnPolicyRunner(
      wrapped,
      agent_cfg,
      log_dir=str(tmp_path / "runner-logs"),
      device="cpu",
    )
    checkpoint = checkpoint_dir / "model_0.pt"
    runner.save(str(checkpoint))
    onnx_path = checkpoint_dir / f"{checkpoint_dir.name}.onnx"
    assert checkpoint.is_file()
    assert onnx_path.is_file()
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    assert metadata["onnx_input_names"] == "obs"
    assert metadata["onnx_input_shapes"] == "1.000;81.000"
    assert metadata["onnx_input_groups"] == "actor"
    assert metadata["obs_group.obs.term_names"] == (
      "command,base_ang_vel,projected_gravity,joint_pos,joint_vel,actions"
    )
    assert "observation_names" not in metadata
  finally:
    env.close()
