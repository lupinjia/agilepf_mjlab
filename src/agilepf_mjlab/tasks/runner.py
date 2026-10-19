"""AgilePF velocity runner that writes the grouped deployment metadata schema."""

from __future__ import annotations

import traceback

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from agilepf_mjlab.onnx_metadata import retrofit_agilepf_metadata


class AgilePFVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  """Preserve mjlab save behavior and replace flat export metadata afterward."""

  def save(self, path: str, infos=None) -> None:
    super().save(path, infos)
    _policy_dir, _filename, onnx_path = self._get_export_paths(path)
    if not onnx_path.is_file():
      return
    try:
      run_path = "local"
      import wandb

      if (
        self.logger.logger_type in ("wandb", "WandbLogWriter") and wandb.run is not None
      ):
        run_path = wandb.run.name
      retrofit_agilepf_metadata(
        self.env.unwrapped,
        onnx_path,
        run_path=run_path,
        experiment_name=self.cfg.get("experiment_name", ""),
        run_name=self.cfg.get("run_name", ""),
      )
      if self.logger.logger_type in ("wandb", "WandbLogWriter") and self.cfg.get(
        "upload_model", False
      ):
        wandb.save(str(onnx_path), base_path=str(_policy_dir))
    except Exception as exc:  # noqa: BLE001 - ONNX metadata must not stop PPO training.
      print(f"[WARN] AgilePF ONNX metadata update failed (training continues): {exc}")
      traceback.print_exc()
