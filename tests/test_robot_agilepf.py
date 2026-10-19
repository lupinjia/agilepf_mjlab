import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import pytest
from mjlab.entity import Entity

from agilepf_mjlab.robots.agilepf.agilepf_constants import (
  ACTUATOR_E8112,
  AGILEPF_ACTION_SCALE,
  AGILEPF_ACTUATOR,
  AGILEPF_DEFAULT_ACTION_SCALE,
  AGILEPF_XML,
  DELAY_HOLD_PROB,
  DELAY_MAX_LAG,
  DELAY_MIN_LAG,
  get_agilepf_robot_cfg,
)

EXPECTED_JOINTS = [
  "left_hip_joint",
  "left_thigh_joint",
  "left_calf_joint",
  "right_hip_joint",
  "right_thigh_joint",
  "right_calf_joint",
]


def test_default_pd_gains_and_action_scale_match_robot_parameters() -> None:
  omega = 2.0 * 3.1415926535 * 4.0
  expected_kp = 0.04752756 * omega**2
  expected_kd = 2.0 * 0.04752756 * omega
  expected_scale = 0.25 * 90.0 / expected_kp

  assert ACTUATOR_E8112.reflected_inertia == pytest.approx(0.04752756)
  assert ACTUATOR_E8112.effort_limit == 90.0
  assert AGILEPF_ACTUATOR.stiffness == pytest.approx(expected_kp)
  assert AGILEPF_ACTUATOR.damping == pytest.approx(expected_kd)
  assert AGILEPF_DEFAULT_ACTION_SCALE == pytest.approx(expected_scale)
  assert AGILEPF_ACTION_SCALE == {
    r"(left|right)_(hip|thigh|calf)_joint": pytest.approx(expected_scale)
  }
  assert AGILEPF_ACTUATOR.delay_min_lag == DELAY_MIN_LAG == 0
  assert AGILEPF_ACTUATOR.delay_max_lag == DELAY_MAX_LAG == 4
  assert AGILEPF_ACTUATOR.delay_hold_prob == DELAY_HOLD_PROB == 0.5
  assert AGILEPF_ACTUATOR.target_names_expr == (r"(left|right)_(hip|thigh|calf)_joint",)


def test_robot_entity_builds_six_joint_actuators_in_expected_order() -> None:
  robot = Entity(get_agilepf_robot_cfg())
  model = robot.spec.compile()

  assert model.nu == 6
  actuator_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
  ]
  assert actuator_names == EXPECTED_JOINTS


def test_mjcf_and_all_referenced_mesh_resources_exist_and_compile() -> None:
  assert AGILEPF_XML.is_file()
  xml_dir = Path(AGILEPF_XML).parent
  mesh_files = [mesh.attrib["file"] for mesh in ET.parse(AGILEPF_XML).iter("mesh")]
  assert len(mesh_files) == 9
  assert all((xml_dir / "meshes" / filename).is_file() for filename in mesh_files)

  model = mujoco.MjSpec.from_file(str(AGILEPF_XML)).compile()
  assert model.nq == 13
  assert model.nbody == 10
