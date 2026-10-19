"""RL configuration for AgilePF velocity tasks."""

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class AgilePfPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO settings with the symmetry extension supported by RSL-RL."""

  symmetry_cfg: dict[str, Any] | None = None


@dataclass
class AgilePfOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner config whose algorithm supports symmetry configuration."""

  algorithm: AgilePfPpoAlgorithmCfg = field(default_factory=AgilePfPpoAlgorithmCfg)


def agilepf_ppo_runner_cfg(
  max_iterations: int = 10001,
  run_name: str = "rough",
) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for AgilePF velocity task."""
  return AgilePfOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=AgilePfPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      symmetry_cfg={
        "data_augmentation_func": "agilepf_mjlab.symmetry.agilepf_symmetry:compute_symmetric_states_agilepf",
        "use_data_augmentation": True,
        "use_mirror_loss": True,
        "mirror_loss_coeff": 0.1,
      },
    ),
    experiment_name="agilepf_velocity",
    logger="tensorboard",
    run_name=run_name,
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=max_iterations,
  )
