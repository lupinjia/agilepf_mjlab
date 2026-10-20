"""AgilePF velocity environment configurations."""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp as base_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import (
  flat,
  hf_pyramid_slope,
  hf_pyramid_slope_inv,
  pyramid_stairs,
  pyramid_stairs_inv,
  random_rough,
  wave_terrain,
)
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from agilepf_mjlab import mdp as agilepf_mdp
from agilepf_mjlab.robots.agilepf.agilepf_constants import (
  AGILEPF_ACTION_SCALE,
  AGILEPF_OBS_SCALE,
  get_agilepf_robot_cfg,
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  curriculum=True,
  border_width=10.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "flat": flat(proportion=0.2),
    "pyramid_stairs": pyramid_stairs(proportion=0.2, step_height_range=(0.0, 0.2)),
    "pyramid_stairs_inv": pyramid_stairs_inv(
      proportion=0.2, step_height_range=(0.0, 0.2)
    ),
    "hf_pyramid_slope": hf_pyramid_slope(proportion=0.1, slope_range=(0.0, 0.6)),
    "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
      proportion=0.1, slope_range=(0.0, 0.6)
    ),
    "random_rough": random_rough(proportion=0.1),
    "wave_terrain": wave_terrain(proportion=0.1),
  },
  add_lights=True,
)

ROUGH_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  curriculum=True,
  border_width=10.0,
  num_rows=1,
  num_cols=1,
  sub_terrains={
    "pyramid_stairs": pyramid_stairs(proportion=0.2, step_height_range=(0.15, 0.15)),
    "pyramid_stairs_inv": pyramid_stairs_inv(
      proportion=0.2, step_height_range=(0.15, 0.15)
    ),
    "hf_pyramid_slope": hf_pyramid_slope(proportion=0.1, slope_range=(0.4, 0.4)),
    "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
      proportion=0.1, slope_range=(0.4, 0.4)
    ),
    # "random_rough": random_rough(proportion=0.1),
    # "wave_terrain": wave_terrain(proportion=0.1),
  },
  add_lights=True,
)


def agilepf_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create AgilePF rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_agilepf_robot_cfg()}

  # terrain
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=replace(ROUGH_TERRAINS_CFG),
    max_init_terrain_level=2,
  )

  # Set raycast sensor frame to AgilePF base.
  terrain_scan = None
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      terrain_scan = sensor
      sensor.frame.name = "base"
      sensor.pattern = GridPatternCfg(size=(1.2, 0.8), resolution=0.1)

  site_names = ("left_foot_site", "right_foot_site")
  foot_geom_names = ("left_foot_collision", "right_foot_collision")

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = GridPatternCfg(size=(0.1, 0.1), resolution=0.05)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=foot_geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )
  body_ground_cfg = ContactSensorCfg(
    name="body_ground_touch",
    primary=ContactMatch(
      mode="body",
      entity="robot",
      pattern=(".*",),
      exclude=("left_foot", "right_foot"),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    body_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = AGILEPF_ACTION_SCALE

  cfg.viewer.body_name = "base"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  # Replace the base foot_friction with per-axis friction events for condim 6.
  del cfg.events["foot_friction"]
  cfg.events["foot_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
      "operation": "abs",
      "axes": [0],
      "ranges": (0.3, 1.5),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_spin"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [1],
      "ranges": (1e-4, 2e-2),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_roll"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],
      "ranges": (1e-5, 5e-3),
      "shared_random": True,
    },
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.01, 0.01),
    1: (-0.01, 0.01),
    2: (-0.01, 0.01),
  }
  cfg.events["encoder_bias"].params["bias_range"] = (-0.01, 0.01)
  cfg.events["pd_gains"] = EventTermCfg(
    func=envs_mdp.dr.pd_gains,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
      "kp_range": (0.9, 1.1),
      "kd_range": (0.9, 1.1),
      "operation": "scale",
    },
  )
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.5, 0.5)
  cfg.events["push_robot"].interval_range_s = (5.0, 8.0)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.4, 0.4),
    "roll": (-0.5, 0.5),
    "pitch": (-0.5, 0.5),
    "yaw": (-0.5, 0.5),
  }

  cfg.rewards = {
    "track_linear_x_velocity": RewardTermCfg(
      func=agilepf_mdp.track_linear_x_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_linear_y_velocity": RewardTermCfg(
      func=agilepf_mdp.track_linear_y_velocity,
      weight=0.5,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=agilepf_mdp.track_angular_velocity_yaw,
      weight=0.5,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_base_height": RewardTermCfg(
      func=agilepf_mdp.track_base_height,
      weight=0.5,
      params={
        "height_scan_sensor_name": "terrain_scan",
        "height_target": 0.42,
        "std": 0.1,
      },
    ),
    "base_orientation": RewardTermCfg(
      func=base_mdp.flat_orientation_l2,
      weight=-3.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base",))},
    ),
    "lin_vel_z": RewardTermCfg(
      func=agilepf_mdp.body_lin_vel_z_penalty,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base",))},
    ),
    "body_ang_vel": RewardTermCfg(
      func=base_mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base",))},
    ),
    "dof_power": RewardTermCfg(
      func=base_mdp.electrical_power_cost,
      weight=-2e-5,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "dof_torque": RewardTermCfg(func=base_mdp.joint_torques_l2, weight=-2e-5),
    "dof_acc": RewardTermCfg(func=base_mdp.joint_acc_l2, weight=-2.5e-7),
    "dof_pos_limits": RewardTermCfg(func=base_mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.01),
    "action_acc_l2": RewardTermCfg(func=base_mdp.action_acc_l2, weight=-0.01),
    "feet_clearance": RewardTermCfg(
      func=agilepf_mdp.feet_clearance_exp,
      weight=0.5,
      params={
        "std": 0.1,
        "target_height": 0.1 + 0.022,  # 0.022是脚的高度
        "height_sensor_name": "foot_height_scan",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),
    "biped_periodic_gait": RewardTermCfg(
      func=agilepf_mdp.biped_periodic_gait,
      weight=1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "gait_command_name": "periodic_gait",
        "velocity_command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
        "stand_still_when_command_zero": False,
      },
    ),
    "self_collisions": RewardTermCfg(
      func=base_mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": self_collision_cfg.name},
    ),
    "body_collision": RewardTermCfg(
      func=base_mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": body_ground_cfg.name},
    ),
  }

  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=agilepf_mdp.commands_vel_reward_based,
    params={
      "command_name": "twist",
      "reward_name": "track_linear_x_velocity",
      "min_reward_ratio": 0.8,
      "velocity_stages": [
        {
          "lin_vel_x": (-0.5, 0.5),
        },
        {
          "lin_vel_x": (-0.5, 0.8),
        },
      ],
    },
  )

  cfg.commands = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.1,
      rel_heading_envs=0.3,
      rel_forward_envs=0.2,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 0.5),
        lin_vel_y=(-0.4, 0.4),
        ang_vel_z=(-0.8, 0.8),
        heading=(-math.pi, math.pi),
      ),
    ),
    "periodic_gait": agilepf_mdp.GaitCommandCfg(
      gait_period=0.45,
      theta_left=0.0,
      theta_right=0.5,
      gait_function_type="step",
    ),
  }

  # observations
  actor_terms = {
    "command": ObservationTermCfg(
      func=base_mdp.generated_commands,
      params={"command_name": "twist"},
      scale=(
        AGILEPF_OBS_SCALE["base_lin_vel"],
        AGILEPF_OBS_SCALE["base_lin_vel"],
        AGILEPF_OBS_SCALE["base_ang_vel"],
      ),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=base_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      scale=AGILEPF_OBS_SCALE["base_ang_vel"],
    ),
    "projected_gravity": ObservationTermCfg(
      func=base_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=base_mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
      scale=AGILEPF_OBS_SCALE["joint_pos"],
    ),
    "joint_vel": ObservationTermCfg(
      func=base_mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      scale=AGILEPF_OBS_SCALE["joint_vel"],
    ),
    "actions": ObservationTermCfg(
      func=base_mdp.last_action, scale=AGILEPF_OBS_SCALE["actions"]
    ),
  }
  assert terrain_scan is not None
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=base_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
      scale=AGILEPF_OBS_SCALE["base_lin_vel"],
    ),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1 / terrain_scan.max_distance,
    ),
    "foot_height": ObservationTermCfg(
      func=base_mdp.foot_height,
      params={"sensor_name": "foot_height_scan"},
    ),
    "foot_air_time": ObservationTermCfg(
      func=base_mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=base_mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=base_mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
      scale=AGILEPF_OBS_SCALE["contact_forces"],
    ),
    "actuator_delay_lags": ObservationTermCfg(
      func=agilepf_mdp.actuator_delay_lags_obs,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "pd_gains_scale": ObservationTermCfg(
      func=agilepf_mdp.pd_gains_scale_obs,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "torso_com_offset": ObservationTermCfg(
      func=agilepf_mdp.body_com_offset_obs,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base",))},
    ),
    "encoder_bias": ObservationTermCfg(
      func=agilepf_mdp.encoder_bias_obs,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "foot_friction": ObservationTermCfg(
      func=agilepf_mdp.geom_friction_obs,
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names)},
    ),
    "clock_input": ObservationTermCfg(
      func=agilepf_mdp.clock_input_obs,
      params={"gait_command_name": "periodic_gait"},
    ),
  }

  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=3,
      nan_policy="sanitize",
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=3,
      nan_policy="sanitize",
    ),
  }

  cfg.terminations["out_of_terrain_bounds"].params = {
    "margin": 2.0,  # 加大边界，防止ray_cast超出边界
  }
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=base_mdp.bad_orientation,
    params={"limit_angle": math.radians(85.0)},
  )
  cfg.terminations["nan"] = TerminationTermCfg(
    func=base_mdp.nan_detection,  # 对于rough的task，机器人有可能卡在地形上或者出现其他异常情况导致nan值
    time_out=True,  # 不作为reward signal，仅重置出错的env
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      cfg.scene.terrain.terrain_generator = ROUGH_TERRAINS_PLAY_CFG

  return cfg


def agilepf_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create AgilePF flat terrain velocity configuration."""
  cfg = agilepf_rough_env_cfg(play=play)

  # A single event selects sitting or ordinary randomized resets per environment.
  base_reset = cfg.events.pop("reset_base")
  joint_reset = cfg.events.pop("reset_robot_joints")
  cfg.events["reset_from_sitting"] = EventTermCfg(
    func=agilepf_mdp.reset_from_sitting,
    mode="reset",
    params={
      "sit_init_percent": 1.0 if play else 0.5,
      "sit_pos": (0.0, 0.0, 0.18),
      "sit_joint_angles": {
        "left_hip_joint": 0.332,
        "left_thigh_joint": 1.22,
        "left_calf_joint": -1.98,
        "right_hip_joint": -0.316,
        "right_thigh_joint": 1.22,
        "right_calf_joint": -1.98,
      },
      "sit_pitch_angle": 0.0,
      "pose_range": base_reset.params["pose_range"],
      "velocity_range": base_reset.params["velocity_range"],
      "joint_position_range": joint_reset.params["position_range"],
      "joint_velocity_range": joint_reset.params["velocity_range"],
    },
  )

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensors and collision sensors not needed on flat.
  remove_sensors = {
    "terrain_scan",
  }
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
  )
  del cfg.observations["critic"].terms["height_scan"]
  cfg.rewards["track_base_height"].params["height_scan_sensor_name"] = None

  # On flat terrain fell_over is sufficient; thigh contact implies fallen.
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=base_mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 0.8)
    twist_cmd.ranges.ang_vel_z = (-0.8, 0.8)

  return cfg
