# AgilePF for mjlab

这是 AgilePF 机器人在 mjlab 上的独立项目。当前注册并支持平地任务 `Mjlab-Velocity-Flat-AgilePF`，包含六关节模型、站立/坐姿重置、自定义步态命令与奖励，以及左右对称增强。粗糙地形配置保留为辅助函数，但没有注册成受支持任务，功能完整性不保证。

## 环境准备

需要 Python 3.10–3.13 和 [uv](https://docs.astral.sh/uv/)。在仓库根目录执行：

```bash
uv sync
```

项目依赖锁定 `mjlab==1.6.0`。独立安装时从官方 PyPI 解析到 `mjlab==1.6.0` 和 `rsl-rl-lib==5.4.2`；本项目在此组合上完成了模型、GPU 环境步骤和小型 PPO 烟测。

W&B 直接依赖限定为 `wandb>=0.28.2,<0.29`。`rsl-rl-lib==5.4.2` 的 `WandbLogWriter` 初始化时传入 `wandb.Settings(start_method="thread")`；W&B 0.29 起此用法不兼容。不要放宽此上限，除非离线 writer 回归测试和默认 W&B 训练烟测在新版本上均通过。

已知限制：旧版 gait mask 的索引缺陷在本阶段保持原样；该问题留待单独变更处理。

`uv run pytest` 当前收集 17 个测试用例，包含坐姿/站立 reset 分支、混合 reset 子集、关节顺序和 ONNX 元数据回归测试。

## 当前包含内容

- `agilepf_mjlab.robots.agilepf.agilepf_constants`：AgilePF 机器人参数及包内 MJCF 路径。
- `agilepf_mjlab.symmetry.agilepf_symmetry`：AgilePF 左右对称回调。
- `agilepf_mjlab.tasks.env_cfgs`：平地任务配置，以及未注册的 rough 配置辅助函数。
- `agilepf_mjlab.tasks.rl_cfg`：PPO actor/critic 和 symmetry 配置。
- `agilepf_mjlab.mdp`：站立/坐姿 reset event、gait command、reward-based command curriculum、观测项和步态/速度奖励。
- `agilepf_mjlab/robots/agilepf/xmls/`：MJCF 和其引用的 STL 网格。

训练 runner 启用对称数据增强与镜像损失，`mirror_loss_coeff=0.1`。旧版 gait-mask 的 flatten/indexing 缺陷仍被保留，待单独修复。

## 运行入口

安装正式依赖及开发测试工具：

```bash
uv sync --group dev
```

训练平地任务：

```bash
uv run train Mjlab-Velocity-Flat-AgilePF
```

播放已训练检查点：

```bash
uv run play Mjlab-Velocity-Flat-AgilePF --checkpoint-file /path/to/model.pt
```

训练依赖 mjlab / MuJoCo-Warp 支持的 NVIDIA CUDA 环境。本次小型验证使用 RTX 4090 24GB、driver 580.126.20；小型 smoke 不是收敛、长期稳定性、吞吐量或 sim-to-real 验证。

已实际运行的无 W&B 2-iteration GPU smoke 命令（TensorBoard 本地日志）：

```bash
uv run train Mjlab-Velocity-Flat-AgilePF \
  --env.scene.num-envs 16 \
  --agent.max-iterations 2 \
  --agent.logger tensorboard \
  --agent.save-interval 1 \
  --agent.run-name gpu-integration-smoke \
  --log-root logs/integration
```

该次运行完成 2 次 PPO 更新，产生有限的 value/surrogate/entropy/symmetry loss；TensorBoard 记录了 `Loss/symmetry`。日志、配置和两个 checkpoint 保存在 `logs/integration/agilepf_velocity/2026-10-19_17-23-31_gpu-integration-smoke/`。另在同一 GPU 上对 16 个环境执行了 4 个零动作步；seed 42 的 reset 同时得到 11 个坐姿和 5 个站立环境，观察与奖励均为有限值。以上只证明这组小型 smoke 能运行，不代表学习效果。

任务 ID 与执行参数可由 `uv run train --help` / `uv run play --help` 查看。`uv run list-envs` 可检查插件发现；本项目当前只注册平地 AgilePF 任务。

## ONNX 与部署元数据

训练保存 checkpoint 时会自动导出 ONNX，并写入分组元数据。可从旧 checkpoint 单独重新导出：

```bash
uv run agilepf-export-onnx --checkpoint /path/to/model_4000.pt --device cpu
```

也可只给已有 AgilePF ONNX 补写/更新元数据；该操作不重新导出网络或改动 ONNX graph：

```bash
uv run agilepf-attach-onnx-metadata --onnx /path/to/policy.onnx --device cpu
```

ONNX 的实际输入/输出名称和 tensor 形状从 graph 读取；每个输入组的历史长度、flatten 标记、term 名称、历史长度、clip 和 scale 从运行环境配置中读取。当前 policy 输入为 `obs`，对应 `actor` observation group（81 维、3 帧历史），输出 `actions` 为 6 维。导出遵循 `mjlab.rl.exporter_utils.list_to_csv_str` 的逗号/分号分隔与三位小数格式；补写元数据会替换本 schema 字段、清除旧版扁平 `observation_*` 字段、保留不相关元数据，并且可重复执行。

默认 logger 是 W&B。已验证不覆盖 `--agent.logger` 的离线默认训练命令：

```bash
WANDB_MODE=offline uv run train Mjlab-Velocity-Flat-AgilePF \
  --env.scene.num-envs 16 \
  --agent.max-iterations 1 \
  --agent.run-name default-wandb-offline-smoke \
  --log-root logs/integration
```

该次默认 W&B logger 烟测在 RTX 4090 上完成 1 次 PPO 更新，日志位于 `logs/integration/agilepf_velocity/2026-10-19_19-50-10_default-wandb-offline-smoke/`，W&B 离线记录位于 `wandb/offline-run-20261019_195011-ushyi9ha/`。此结果仅验证 logger 和小批量训练链路，不表示模型收敛。

## 资源来源与许可

常量、对称回调、MJCF 和 9 个 STL 文件提取自 mjlab AgilePF 资源。代码来源项目包含 Apache-2.0 许可证；本项目 LICENSE 对项目代码的许可不应被理解为 STL 网格已获相同许可。网格的原始作者、来源和再分发许可尚未确认，见 [NOTICE](NOTICE)。如需公开分发，请先确认网格资产的权利和许可。
