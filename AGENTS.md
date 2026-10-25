# AgilePF project notes

- The registered task is `Mjlab-Velocity-Flat-AgilePF`; rough-terrain config
  helpers exist but are not registered or parity-validated. The RTX 4090
  16-env/2-iteration TensorBoard smoke is validated, but it establishes neither
  convergence nor long-run/deployment performance.
- The package uses `mjlab==1.6.0` and `rsl-rl-lib==5.4.2` was resolved from the
  official package index during standalone setup. Do not add local-path
  dependencies, vendor either framework, or use symlinks.
- Keep the direct `wandb>=0.28.2,<0.29` bound: RSL-RL 5.4.2 initializes
  `WandbLogWriter` with `wandb.Settings(start_method="thread")`, which is
  incompatible with W&B 0.29+. Only reconsider after the offline writer test
   and explicit W&B CLI smoke pass with the replacement version.
- Task registration is discovered through the `mjlab.tasks` entry-point group.
  Keep the AgilePF entry-point target aligned with `agilepf_mjlab.tasks`.
- Grouped ONNX metadata is owned by `agilepf_mjlab.onnx_metadata`. Reuse
  `mjlab.rl.exporter_utils.get_base_metadata` for robot/control metadata and its
  `list_to_csv_str` for exact CSV serialization; filter its legacy flat
  observation fields and derive I/O dimensions and group terms dynamically.
  `AgilePFVelocityOnPolicyRunner.save` must keep the base save/export behavior
  and idempotently replace metadata after export. Keep both the
  `agilepf-export-onnx` and `agilepf-attach-onnx-metadata` commands usable.
- Validate ONNX behavior with
  `uv run pytest tests/test_onnx_metadata.py --basetemp .tmp-pytest`; tests cover
  exact metadata formatting, graph preservation, checker validity, and an
  actual runner save. The README describes checkpoint export and metadata-only
  retrofit commands.
- Robot assets are package resources under
  `agilepf_mjlab/robots/agilepf/xmls/`. Keep MJCF mesh paths relative to the
  XML and resolve the entry XML relative to its Python module.
- The original optional linear-decay actuator mode is not supported in the
  current package. Preserve the default built-in-PD gains, motor inertia,
  effort limit, delay parameters, and six-joint target order.
- A legacy gait-mask indexing defect is known and intentionally unchanged;
  address it only in a separate scoped change.
- Full validation recipe: `uv run pytest`; `uv run ruff check .`;
  `uv run ruff format --check src/agilepf_mjlab tests`;
  `uv run list-envs`; `uv run train --help`; then
  execute the 16-env GPU
   TensorBoard smoke documented in README. TensorBoard is the default logger;
   W&B smoke runs require `WANDB_MODE=offline --agent.logger wandb` (set the
   environment variable before the command, and pass the flag to the CLI).
  The reset smoke must cover both standing and sitting branches and check finite
  observations/rewards. Training must report a finite `Loss/symmetry`; check
  `agent.yaml` for augmentation enabled and mirror coefficient 0.1.
- Rebuild the final wheel with `uv build --wheel`; verify the task entry point,
  MJCF, and all nine STL resources are present and no `mjlab`/`rsl_rl` package
  source has been bundled. Confirm framework imports resolve from external
  site-packages and there are no local-path dependencies or symlinks.
- The STL license/provenance is not established; do not describe the mesh files
  as any-licensed absent confirmation. See `NOTICE`.
- Project code is BSD-3-Clause (`LICENSE`, `pyproject.toml`
  `license`/`license-files`). mjlab-derived portions stay Apache-2.0: retain
  `LICENSES/Apache-2.0.txt`, the `NOTICE` attribution, and the `License-File`
  metadata that `uv build --wheel` emits. Keep `README.md` in English and keep
  its licensing section consistent with `NOTICE`.
