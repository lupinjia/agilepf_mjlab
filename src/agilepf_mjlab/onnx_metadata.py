"""AgilePF adapter for the grouped ONNX metadata schema used by deployment."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_INPUT_GROUP_ALIASES = {"obs": "actor", "current_obs": "actor"}
_LEGACY_FLAT_OBSERVATION_KEYS = {
  "command_names",
  "observation_names",
  "observation_terms_scale",
  "observation_terms_flatten_history_dim",
  "observation_terms_history_length",
  "observation_terms_clip",
}
_RECURRENT_STATE_INPUTS = {"h_in", "c_in"}


def _git_user_name() -> str:
  try:
    return subprocess.check_output(
      ["git", "config", "user.name"], text=True, stderr=subprocess.DEVNULL
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    return ""


def _onnx_io_names_shapes(
  onnx_path: str | Path,
) -> tuple[list[str], list[list[int | None]], list[str], list[list[int | None]]]:
  """Read actual graph inputs/outputs and preserve dynamic dimensions as None."""
  model = onnx.load(str(onnx_path))

  def get_io(values):
    names: list[str] = []
    shapes: list[list[int | None]] = []
    for value in values:
      dims = [
        dim.dim_value if dim.dim_value > 0 else None
        for dim in value.type.tensor_type.shape.dim
      ]
      names.append(value.name)
      shapes.append(dims)
    return names, shapes

  input_names, input_shapes = get_io(model.graph.input)
  output_names, output_shapes = get_io(model.graph.output)
  return input_names, input_shapes, output_names, output_shapes


def _term_metadata(env: ManagerBasedRlEnv, group_name: str) -> dict[str, list[Any]]:
  term_names = env.observation_manager.active_terms[group_name]
  history_lengths: list[Any] = []
  clips: list[Any] = []
  scales: list[Any] = []
  for term_name in term_names:
    cfg = env.observation_manager.get_term_cfg(group_name, term_name)
    history_lengths.append(cfg.history_length)
    clips.append(cfg.clip or [float("-inf"), float("inf")])
    scale = cfg.scale
    if isinstance(scale, torch.Tensor):
      scale = scale.cpu().tolist()
    elif isinstance(scale, (list, tuple)):
      scale = list(scale)
    scales.append(1.0 if scale is None else scale)
  return {
    "term_names": list(term_names),
    "term_history_lengths": history_lengths,
    "term_clips": clips,
    "term_scales": scales,
  }


def build_agilepf_metadata(
  env: ManagerBasedRlEnv,
  onnx_path: str | Path,
  *,
  run_path: str = "local",
  experiment_name: str = "",
  run_name: str = "",
) -> dict[str, Any]:
  """Build the grouped AgilePF metadata schema from an env and ONNX graph.

  Official mjlab 1.6 supplies the shared robot/control fields. This adapter
  removes that version's legacy flat observation fields and adds the grouped
  per-input schema expected by AgilePF consumers.
  """
  from mjlab.rl.exporter_utils import get_base_metadata

  metadata = get_base_metadata(env, run_path)
  for key in _LEGACY_FLAT_OBSERVATION_KEYS:
    metadata.pop(key, None)
  metadata.update(
    {
      "experiment_name": experiment_name,
      "run_name": run_name,
      "git_user_name": _git_user_name(),
    }
  )

  input_names, input_shapes, output_names, output_shapes = _onnx_io_names_shapes(
    onnx_path
  )
  observation_groups = env.cfg.observations
  input_groups = [
    name if name in observation_groups else _INPUT_GROUP_ALIASES.get(name, name)
    for name in input_names
  ]
  metadata.update(
    {
      "onnx_input_names": input_names,
      "onnx_input_shapes": input_shapes,
      "onnx_input_groups": input_groups,
      "onnx_output_names": output_names,
      "onnx_output_shapes": output_shapes,
    }
  )

  for input_name, group_name in zip(input_names, input_groups, strict=True):
    if input_name in _RECURRENT_STATE_INPUTS:
      continue
    if group_name not in observation_groups:
      raise ValueError(
        f"ONNX input {input_name!r} maps to missing observation group {group_name!r}"
      )
    group_cfg = observation_groups[group_name]
    terms = _term_metadata(env, group_name)
    metadata[f"obs_group.{input_name}.history_length"] = group_cfg.history_length or 0
    metadata[f"obs_group.{input_name}.flatten_history_dim"] = (
      group_cfg.flatten_history_dim
    )
    for key, value in terms.items():
      metadata[f"obs_group.{input_name}.{key}"] = value
  return metadata


def attach_agilepf_metadata(
  onnx_path: str | Path,
  metadata: Mapping[str, Any],
) -> None:
  """Replace AgilePF metadata without duplicating keys or touching the graph."""
  from mjlab.rl.exporter_utils import list_to_csv_str

  model = onnx.load(str(onnx_path))
  replacement_keys = set(metadata) | _LEGACY_FLAT_OBSERVATION_KEYS
  retained = [
    (prop.key, prop.value)
    for prop in model.metadata_props
    if prop.key not in replacement_keys
  ]
  del model.metadata_props[:]
  for key, value in retained:
    prop = model.metadata_props.add()
    prop.key = key
    prop.value = value
  for key, value in metadata.items():
    prop = model.metadata_props.add()
    prop.key = key
    prop.value = list_to_csv_str(value) if isinstance(value, list) else str(value)
  onnx.save(model, str(onnx_path))


def retrofit_agilepf_metadata(
  env: ManagerBasedRlEnv,
  onnx_path: str | Path,
  *,
  run_path: str = "local",
  experiment_name: str = "",
  run_name: str = "",
) -> dict[str, Any]:
  """Attach current AgilePF metadata to an existing ONNX without changing weights."""
  metadata = build_agilepf_metadata(
    env,
    onnx_path,
    run_path=run_path,
    experiment_name=experiment_name,
    run_name=run_name,
  )
  attach_agilepf_metadata(onnx_path, metadata)
  return metadata


def _create_env(device: str) -> ManagerBasedRlEnv:
  from mjlab.envs import ManagerBasedRlEnv

  from agilepf_mjlab.tasks.env_cfgs import agilepf_flat_env_cfg

  cfg = agilepf_flat_env_cfg(play=True)
  cfg.scene.num_envs = 1
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def _config_value(agent_yaml: Path, name: str, default: str) -> str:
  if not agent_yaml.is_file():
    return default
  match = re.search(
    rf"^{re.escape(name)}:\s*(.*?)\s*$", agent_yaml.read_text(), re.MULTILINE
  )
  if match is None:
    return default
  value = match.group(1).strip().strip("\"'")
  return value or default


def attach_metadata_main() -> None:
  """Retrofit grouped metadata onto an existing AgilePF ONNX file."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--onnx", required=True, type=Path)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--run-path", default="local")
  parser.add_argument("--experiment-name")
  parser.add_argument("--run-name")
  args = parser.parse_args()
  if not args.onnx.is_file():
    parser.error(f"ONNX file not found: {args.onnx}")
  params_dir = args.onnx.parent / "params"
  env = _create_env(args.device)
  try:
    retrofit_agilepf_metadata(
      env,
      args.onnx,
      run_path=args.run_path,
      experiment_name=args.experiment_name
      or _config_value(
        params_dir / "agent.yaml", "experiment_name", "agilepf_velocity"
      ),
      run_name=args.run_name
      or _config_value(params_dir / "agent.yaml", "run_name", args.onnx.parent.name),
    )
  finally:
    env.close()
  print(f"[INFO] AgilePF metadata attached to {args.onnx}")


def export_checkpoint_main() -> None:
  """Export a PPO checkpoint for AgilePF and attach grouped metadata."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    parser.error(f"Checkpoint not found: {args.checkpoint}")

  from dataclasses import asdict

  from mjlab.rl import RslRlVecEnvWrapper

  from agilepf_mjlab.tasks.rl_cfg import agilepf_ppo_runner_cfg
  from agilepf_mjlab.tasks.runner import AgilePFVelocityOnPolicyRunner

  params_dir = args.checkpoint.parent / "params"
  runner_cfg = agilepf_ppo_runner_cfg()
  runner_dict = asdict(runner_cfg)
  runner_dict["experiment_name"] = _config_value(
    params_dir / "agent.yaml", "experiment_name", runner_cfg.experiment_name
  )
  runner_dict["run_name"] = _config_value(
    params_dir / "agent.yaml", "run_name", args.checkpoint.parent.name
  )
  output_path = args.output or args.checkpoint.with_suffix(".onnx")
  output_path.parent.mkdir(parents=True, exist_ok=True)

  env = _create_env(args.device)
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
  runner = AgilePFVelocityOnPolicyRunner(
    wrapped_env, runner_dict, log_dir=None, device=args.device
  )
  try:
    runner.load(
      str(args.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
    runner.export_policy_to_onnx(str(output_path.parent), output_path.name)
    retrofit_agilepf_metadata(
      env,
      output_path,
      run_path=args.checkpoint.parent.name,
      experiment_name=runner_dict["experiment_name"],
      run_name=runner_dict["run_name"],
    )
  finally:
    env.close()
  print(f"[INFO] AgilePF ONNX exported to {output_path}")


def main() -> None:
  export_checkpoint_main()


if __name__ == "__main__":
  main()
