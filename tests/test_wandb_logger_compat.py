from rsl_rl.utils.wandb_log_writer import WandbLogWriter


def test_rsl_wandb_writer_offline_init_log_and_close(tmp_path, monkeypatch) -> None:
  monkeypatch.setenv("WANDB_MODE", "offline")
  monkeypatch.setenv("WANDB_DIR", str(tmp_path))
  monkeypatch.setenv("WANDB_SILENT", "true")

  log_dir = tmp_path / "rsl-wandb-writer"
  writer = WandbLogWriter(log_dir=str(log_dir), project_name="agilepf-offline-test")
  writer.add_scalar("compatibility/smoke", 1.0, global_step=1)
  writer.stop()
  writer.close()
