"""AgilePF velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task

from agilepf_mjlab.tasks.env_cfgs import agilepf_flat_env_cfg, agilepf_rough_env_cfg
from agilepf_mjlab.tasks.rl_cfg import agilepf_ppo_runner_cfg
from agilepf_mjlab.tasks.runner import AgilePFVelocityOnPolicyRunner

# Keep rough config construction available for downstream experiments, but
# advertise only the validated flat task.
_ = agilepf_rough_env_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-AgilePF",
  env_cfg=agilepf_flat_env_cfg(),
  play_env_cfg=agilepf_flat_env_cfg(play=True),
  rl_cfg=agilepf_ppo_runner_cfg(max_iterations=6001, run_name="flat"),
  runner_cls=AgilePFVelocityOnPolicyRunner,
)
