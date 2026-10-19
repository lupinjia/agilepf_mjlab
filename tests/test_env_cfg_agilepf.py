from agilepf_mjlab.tasks.env_cfgs import agilepf_flat_env_cfg


def test_flat_agilepf_configuration_contract() -> None:
  train = agilepf_flat_env_cfg()
  play = agilepf_flat_env_cfg(play=True)

  assert train.observations["actor"].history_length == 3
  assert train.observations["actor"].terms.keys() == {
    "command",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
  }
  assert train.observations["actor"].history_length * 27 == 81
  assert "clock_input" in train.observations["critic"].terms
  assert "clock_input" not in train.observations["actor"].terms
  assert train.events["reset_from_sitting"].params["sit_init_percent"] == 0.5
  assert play.events["reset_from_sitting"].params["sit_init_percent"] == 1.0
  assert train.scene.terrain is not None
  assert train.scene.terrain.terrain_type == "plane"
