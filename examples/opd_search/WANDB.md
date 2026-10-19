# W&B 实验记录

服务器源码：`/data/work/YingZhong/code/on_policy_distillation_verl`。

使用 verl 原生 `Tracking` 和 `ValidationGenerationsLogger`。默认 `trainer.logger=[console,wandb]`，训练主进程负责记录，不在每个 GPU worker 重复创建 run。opd_env 已安装 wandb 0.27.2，无需新增依赖。

## 记录内容

- 完整解析后的训练配置：teacher/student 路径、数据、学习率、batch、OPD 系数、采样与资源设置。
- `actor/distillation/loss`、`opd_ce`、`reference_kl`、`importance_mean`、`importance_clip_fraction`。
- `actor/grad_norm`、`actor/lr`、熵、生成长度、模型版本跨度和陈旧度。
- 验证 EM/格式分数、每步耗时、吞吐及训练框架提供的显存指标。
- 每次验证的 2 条问答样例，保存在 `val/generations` 表。可用 `trainer.log_val_generations=0` 关闭。
- 训练与验证的完整轨迹另存于实验输出目录下的 `rollouts/` 和 `validation/`。

指标沿用原训练框架提供的字段和真实 global step。未新增“实际 HTTP 检索次数”在线指标；此前 17 次检索来自保存轨迹与服务日志的事后审计。模型与优化器 checkpoint 保存在服务器，当前代码没有上传大体积 checkpoint artifact。

默认每 50 个训练 global step 执行验证并保存 checkpoint。配置为 `trainer.test_freq=50`、`trainer.save_freq=${trainer.test_freq}`；两个启动脚本均继承此设置。每个数据集的 EM 指标如 `val-core/<数据集>/acc/mean@1` 与训练 global step 对齐。`trainer.log_val_generations=2` 仅限制上传的问答示例数，不会把正式配置的验证集缩小为 2 条。

训练最终一步也会验证和保存，七卡正式入口跳过 step 0 的训练前验证，首次完整验证在第 50 步。小规模 smoke 保留 `data.val_max_samples=2`，并已移除此前每步保存/验证的覆盖；运行 2/3 步时会在对应最终步验证和保存。

## 在线运行

先在服务器登录。API key 由 SDK 登录保存或通过环境变量提供，不写入源码、Hydra 参数或实验身份文件。

```bash
source /data/work/YingZhong/miniconda3/etc/profile.d/conda.sh
conda activate opd_env
wandb login --relogin --verify
cd /data/work/YingZhong/code/on_policy_distillation_verl
export WANDB_MODE=online
export WANDB_PROJECT=opd_search
export WANDB_NAME=qwen25_7b_to_3b_opd
# 若使用团队空间，设置实际团队名；个人默认空间可不设：
# export WANDB_ENTITY=your_team
```

检索服务启动后，可运行正式配置：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export OPD_OUTPUT_DIR="$PWD/checkpoints/opd_search_7gpu"
bash examples/opd_search/run_formal_7gpu.sh
```

或者在全新目录复跑已验证的小规模训练：

```bash
export OPD_SMOKE_OUTPUT="$PWD/checkpoints/opd_wandb_$(date +%Y%m%d_%H%M%S)"
bash examples/opd_search/run_smoke.sh train
bash examples/opd_search/run_smoke.sh resume
```

`WANDB_PROJECT` / `WANDB_NAME` 提供默认项目和显示名称，末尾显式 Hydra overrides 优先。`WANDB_ENTITY` 由原生 SDK 读取。启动器在分配训练模型前检查在线身份，认证失败时明确提示重新登录或显式使用离线模式。

七卡正式入口会把所有资源、显存和动态批量参数作为解析后的配置记录到 W&B，具体参数见 [FORMAL_7GPU.md](FORMAL_7GPU.md)。每 50 步保存、验证和在线 run 身份管理均继承现有设置。

启动前的在线检查使用 SDK `wandb.login(verify=True)`，与 CLI 的服务端凭据验证一致；不依赖 Public API 的 viewer 查询。此版本 SDK 的 viewer 查询曾在有效凭据下返回 relogin required，不能将该查询错误直接解释为 key 失效。

## Run 身份与恢复

每个 `trainer.default_local_dir` 保存一个 `wandb_run.json`，只包含 run ID、项目、entity 和服务地址。默认本地记录目录为该输出目录下的 `wandb_logs/`，可用 `WANDB_DIR` 覆盖。

在线模式复用保存的 run ID，设置 `WANDB_RESUME=allow`，让原生训练器按 checkpoint 恢复的 global step 继续记录。对于此前未启用 W&B 的 checkpoint，会创建新 run，从接续训练的步数开始记录；不会自动补造历史指标。`resume_path` 模式读取源 checkpoint 父目录的身份文件；若源目录没有该文件，需显式提供 `WANDB_RUN_ID`。

不同实验使用不同输出目录。已有身份文件时，禁止 `resume_mode=disable` 覆盖，并拒绝意外切换项目、entity、服务地址或 run ID，避免把不同实验混在同一条曲线中。`WANDB_RUN_ID` 和 `WANDB_RESUME` 由启动器管理。

在线 W&B 历史若比所选 checkpoint 更新，SDK 可能丢弃回退步数的记录；这时应使用新输出目录和新 run，并注明恢复来源。当前代码未实现云端历史回滚。

## 离线运行与之后同步

```bash
export WANDB_MODE=offline
bash examples/opd_search/run_smoke.sh train
bash examples/opd_search/run_smoke.sh resume
```

W&B SDK 不支持离线 run 的原地恢复。每次离线启动会创建独立 run，用同一个 `WANDB_RUN_GROUP` 关联到实验，步数仍使用真实训练 global step。各段作为独立 runs 同步，不声称已经合并成一条连续在线历史。

登录恢复、网络可用后，对 SDK 输出的具体目录执行：

```bash
wandb sync /absolute/path/to/wandb_logs/wandb/offline-run-TIMESTAMP-RUNID
```

只使用控制台时，显式传入 `'trainer.logger=[console]'`；此时启动器不创建 W&B 身份或记录目录。

## 本轮验证（2026-10-17）

8 项身份管理回归测试通过；实际 W&B SDK 已记录并重新解析此前真实模型训练的 step 1、2、3，损失值与原始审计 JSON 逐项相同，配置、summary、验证问答表均保存。该 run 明确标注为已完成训练的数据回填，本轮未重新执行模型训练。

真实 Ray worker 收到的 W&B 模式、run ID、group 和解释器与启动器设置一致，实际写入了原始第 3 步指标。完整回归共 43 项（原有 35 项加新增 8 项）全部通过；Ruff 和 shell 语法检查通过。

原有凭据曾被服务端拒绝。用户更新凭据后，`wandb login --verify` 已通过，账号为 yingzhong。在线验证使用两个独立进程：第一个上传此前真实训练的 step 1/2，第二个沿用同一 run ID 上传 step 3，SDK 返回 `resumed=true`；随后通过云端 API 读回历史，确认 step 为 1、2、3，三步损失与原始训练记录一致，验证问答表上传成功。

在线实验：[qwen25_verified_e2e_backfill_20261017](https://wandb.ai/yingzhong-the-hong-kong-polytechnic-university/opd_search/runs/db2caf14c2b6)。该实验明确标注为此前真实训练的数据回填；本轮没有重新运行模型训练。模型/优化器 checkpoint 恢复的实际训练证据在 E2E_VALIDATION.md，W&B run 的在线续接验证在 online-verification.json。

日志、SDK 文件和结构化验证记录保存在 `/data/work/YingZhong/.cache/opd-wandb-20261017/`。此前完整模型训练与 checkpoint 恢复证据仍见 [E2E_VALIDATION.md](E2E_VALIDATION.md)。
