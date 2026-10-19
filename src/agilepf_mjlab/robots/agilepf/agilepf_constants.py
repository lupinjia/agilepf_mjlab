"""Robot-specific AgilePF constants and configuration."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from mjlab.utils.spec_config import CollisionCfg

AGILEPF_XML: Path = Path(__file__).parent / "xmls" / "agilepf.xml"
assert AGILEPF_XML.exists()


def get_spec() -> mujoco.MjSpec:
  """Load the AgilePF MJCF specification from this installed package."""
  return mujoco.MjSpec.from_file(str(AGILEPF_XML))


ACTUATOR_E8112 = ElectricActuator(
  reflected_inertia=0.04752756,
  velocity_limit=18.0,
  effort_limit=90.0,
)

DELAY_MIN_LAG = 0
DELAY_MAX_LAG = 4
DELAY_HOLD_PROB = 0.5

_JOINT_TARGETS = r"(left|right)_(hip|thigh|calf)_joint"
_NATURAL_FREQUENCY_HZ = 4.0
_DAMPING_RATIO = 1.0
_OMEGA = 2.0 * 3.1415926535 * _NATURAL_FREQUENCY_HZ
_POSITION_STIFFNESS = ACTUATOR_E8112.reflected_inertia * _OMEGA**2
_POSITION_DAMPING = 2.0 * ACTUATOR_E8112.reflected_inertia * _OMEGA

# mjlab 1.6 configures the built-in position actuator with direct PD gains.
# Match the original natural-frequency/damping-ratio gains using motor inertia.
AGILEPF_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(_JOINT_TARGETS,),
  stiffness=_POSITION_STIFFNESS,
  damping=_POSITION_DAMPING,
  effort_limit=ACTUATOR_E8112.effort_limit,
  armature=ACTUATOR_E8112.reflected_inertia,
  delay_min_lag=DELAY_MIN_LAG,
  delay_max_lag=DELAY_MAX_LAG,
  delay_hold_prob=DELAY_HOLD_PROB,
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.51),
  joint_pos={
    ".*_hip_joint": 0.0,
    ".*_thigh_joint": 0.5,
    ".*_calf_joint": -0.5,
  },
  joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision.*",),
  contype=1,
  conaffinity=1,
  solref=(0.002, 1),
  condim={
    r"^(left|right)_foot_collision$": 6,
    ".*_collision.*": 1,
  },
  priority={
    r"^(left|right)_foot_collision$": 1,
    ".*": 0,
  },
  friction={r"^(left|right)_foot_collision$": (1, 5e-3, 5e-4)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

AGILEPF_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(AGILEPF_ACTUATOR,),
  soft_joint_pos_limit_factor=0.9,
)


def get_agilepf_robot_cfg() -> EntityCfg:
  """Return a fresh AgilePF entity configuration."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=AGILEPF_ARTICULATION,
  )


AGILEPF_ACTION_SCALE: dict[str, float] = {}
for actuator in AGILEPF_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  effort = actuator.effort_limit
  stiffness = actuator.stiffness
  assert effort is not None
  assert stiffness is not None
  for name in actuator.target_names_expr:
    AGILEPF_ACTION_SCALE[name] = 0.25 * effort / stiffness

AGILEPF_DEFAULT_ACTION_SCALE = 0.25 * ACTUATOR_E8112.effort_limit / _POSITION_STIFFNESS

AGILEPF_OBS_SCALE: dict[str, float] = {
  "base_lin_vel": 1.0,
  "base_ang_vel": 0.25,
  "joint_pos": 1.0,
  "joint_vel": 0.05,
  "height_scan": 1 / 5.0,
  "contact_forces": 1 / 100.0,
  "actions": 1 / 4.0,
  "root_pos": 0.1,
}
