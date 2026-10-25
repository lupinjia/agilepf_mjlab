# AgilePF for mjlab

A standalone project for the AgilePF robot on mjlab. The registered and
supported task is the flat-terrain task `Mjlab-Velocity-Flat-AgilePF`, which
covers the six-joint model, standing/sitting resets, a custom gait command and
rewards, and left/right symmetry augmentation. Rough-terrain configs are kept as
helper functions only: they are not registered as a supported task and their
functional completeness is not guaranteed.

## Environment setup

Requires Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/). From the
repository root:

```bash
uv sync
```

The project pins `mjlab==1.6.0`. Standalone installation resolves
`mjlab==1.6.0` and `rsl-rl-lib==5.4.2` from the official PyPI index; the model,
the GPU environment steps, and the small PPO smoke test in this project were
validated on that combination.

The direct W&B dependency is bounded to `wandb>=0.28.2,<0.29`.
`WandbLogWriter` in `rsl-rl-lib==5.4.2` passes
`wandb.Settings(start_method="thread")` during initialization, which is
incompatible with W&B 0.29 and later. Do not loosen this upper bound unless the
offline writer regression test and the default W&B training smoke test both pass
on the newer version.

## What is included

- `agilepf_mjlab.robots.agilepf.agilepf_constants`: AgilePF robot parameters and
  the in-package MJCF path.
- `agilepf_mjlab.symmetry.agilepf_symmetry`: AgilePF left/right symmetry
  callback.
- `agilepf_mjlab.tasks.env_cfgs`: the flat-terrain task config plus unregistered
  rough config helpers.
- `agilepf_mjlab.tasks.rl_cfg`: PPO actor/critic and symmetry config.
- `agilepf_mjlab.mdp`: standing/sitting reset event, gait command,
  reward-based command curriculum, observation terms, and gait/velocity rewards.
- `agilepf_mjlab/robots/agilepf/xmls/`: the MJCF and the STL meshes it
  references.

## Entry points

Install runtime dependencies together with the development test tooling:

```bash
uv sync --group dev
```

Train the flat task:

```bash
uv run train Mjlab-Velocity-Flat-AgilePF
```

Play a trained checkpoint:

```bash
uv run play Mjlab-Velocity-Flat-AgilePF --checkpoint-file /path/to/model.pt
```

Training requires an NVIDIA CUDA environment supported by mjlab /
MuJoCo-Warp.

Task IDs and runtime arguments can be inspected with `uv run train --help` /
`uv run play --help`. `uv run list-envs` checks plugin discovery; this project
currently registers only the flat AgilePF task.

## ONNX and deployment metadata

Saving a training checkpoint automatically exports ONNX and writes grouped
metadata. An existing checkpoint can be re-exported on its own:

```bash
uv run agilepf-export-onnx --checkpoint /path/to/model_4000.pt --device cpu
```

You can also only backfill or update metadata on an existing AgilePF ONNX file;
this neither re-exports the network nor modifies the ONNX graph:

```bash
uv run agilepf-attach-onnx-metadata --onnx /path/to/policy.onnx --device cpu
```

## License

Original code in this repository is released under the BSD 3-Clause License; see
[LICENSE](LICENSE). Portions extracted from mjlab's AgilePF assets remain subject
to the upstream Apache License 2.0, whose terms and attribution are retained in
[NOTICE](NOTICE) and [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt). The
robot MJCF and the nine STL meshes are excluded from both grants: their original
authorship, provenance, and redistribution terms have not been independently
confirmed, and no license is asserted for them here.
