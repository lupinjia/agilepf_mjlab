import torch
from tensordict import TensorDict

from agilepf_mjlab.symmetry.agilepf_symmetry import compute_symmetric_states_agilepf
from agilepf_mjlab.tasks.env_cfgs import agilepf_flat_env_cfg


class _Env:
  cfg = agilepf_flat_env_cfg()


def test_symmetry_callback_doubles_batch_and_preserves_originals() -> None:
  env = _Env()
  obs = TensorDict(
    {
      "actor": torch.arange(2 * 81, dtype=torch.float32).reshape(2, 81),
      "critic": torch.arange(2 * 165, dtype=torch.float32).reshape(2, 165),
    },
    batch_size=[2],
  )
  actions = torch.arange(12, dtype=torch.float32).reshape(2, 6)

  augmented_obs, augmented_actions = compute_symmetric_states_agilepf(env, obs, actions)

  assert augmented_obs.batch_size == torch.Size([4])
  assert augmented_actions.shape == (4, 6)
  assert torch.equal(augmented_obs["actor"][:2], obs["actor"])
  assert torch.equal(augmented_obs["critic"][:2], obs["critic"])
  assert torch.equal(augmented_actions[:2], actions)
  mirrored_obs, mirrored_actions = compute_symmetric_states_agilepf(
    env, augmented_obs[2:], augmented_actions[2:]
  )
  assert torch.equal(mirrored_obs["actor"][2:], obs["actor"])
  assert torch.equal(mirrored_actions[2:], actions)
