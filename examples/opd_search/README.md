# OPDSearch+：Qwen2.5-7B-Instruct → Qwen2.5-3B Base

代码位置：`/data/work/YingZhong/code/on_policy_distillation_verl`。

实现依据：[OPDSearch+ 论文 v1](https://arxiv.org/pdf/2608.24310)，第 3.3 节、公式 (4)–(6)、第 4.1 节。
上游版本：[verl main / 18cfb74852a053a27dcf301d3318a093be81b628](https://github.com/verl-project/verl/tree/18cfb74852a053a27dcf301d3318a093be81b628)。

## 已实现的内容

1. Student 是官方 Base 模型 `Qwen/Qwen2.5-3B`，teacher 是冻结的 `Qwen/Qwen2.5-7B-Instruct`。
2. Student 按当前策略采样，通过已有 Search-R1 `/retrieve` HTTP 服务实时检索 Wikipedia。
3. 轨迹保留 `<think>`、`<search>`、`<information>`、`<answer>` 格式。只追加生成的 token IDs，不把生成文本解码后重新编码。
4. Teacher 通过最新 verl 原生独立资源池，对 student 的完整轨迹打分；观察文本保留在 teacher/student 的条件前缀中。
5. 所有环境提供的 token（包括 information 分隔符）均为 `response_mask=0`；模型实际生成的 token 为 1。
6. 使用论文裁剪的重要性加权交叉熵，同时保留冻结初始 student 的 reference KL。没有将 OPD 替换成 top-k KL、k3 teacher KL、PPO 或离线 SFT。
7. 数据转换、验证 EM、训练配置、启动脚本、环境检查、CPU 合同测试和逐文件变更记录齐全。

这次实现的是 **OPD 阶段**。论文的完整 OPDSearch+ 还包含 OPD 后的 GRPO 阶段；本项目没有把只做 OPD 宣称为完成两阶段实验，也没有复现论文的最终 EM 数字。论文主结果采用 14B teacher，本次按用户要求采用 7B teacher，对应教师规模消融设置。

## 损失与梯度

令 `s_t = log pi_student(token_t | prefix_t)`，`t_t = log pi_teacher(token_t | prefix_t)`，`M` 为模型生成 token 的集合：

```text
w_t = stop_gradient(clip(exp(t_t - s_t), 1e-6, 10))
L_OPD = -sum(t in M, w_t * s_t) / |M|
L_total = alpha_opd * L_OPD + alpha_kl * L_reference
```

在 log 空间先裁剪再取 exp，避免先发生溢出。重要性比的分母使用本次前向计算的 student 概率；teacher 和权重均不反传梯度。裁剪意味着这是带可控偏差的 forward-KL 梯度估计器，不能把其数值称为精确全词表 KL。

`L_reference` 的前向数值使用非负 k3 估计器：`expm1(log_ref - log_student) - (log_ref - log_student)`。反向使用 `stopgrad(log_student - log_ref) * log_student` 的梯度，并用数值为零的梯度项保留 k3 的前向值。直接反传采样 k3 会遗漏采样分布的梯度，其期望对应另一方向的 KL；本实现的反向估计在固定前缀、student 当前策略采样下，与论文定义的 `KL(student || reference)` 的梯度一致，并有全词表枚举对照测试。

Reference 正则应用同一来源掩码，reference 完全冻结。原生纯蒸馏路径绕过 `ppo_loss`，因此 reference KL 在新的损失入口中显式加入一次，避免正则被跳过。

为保持 on-policy，配置显式使用 V1 sync 训练、temperature=1、top-p=1、top-k=-1、每个 rollout batch 只做一次完整 minibatch 更新、ppo_epochs=1。启动检查拒绝异步训练和按任务奖励筛选样本。每条轨迹保留所有生成调用的模型版本区间；跨版本轨迹会被明确拒绝。

OPD CE、概率比、上下界裁剪比例和 reference KL 指标按全局有效模型 token 数归一化，随后累加 microbatch 贡献并聚合 DP ranks，避免动态 microbatch 长度影响监控曲线。

## 论文设置和本实现的明确选择

| 参数 | 默认值 | 来源/说明 |
|---|---:|---|
| student | Qwen2.5-3B Base | 论文与用户要求 |
| teacher | Qwen2.5-7B-Instruct | 用户要求，论文包含相应消融 |
| 学习率 | 1e-6 | 论文 4.1 |
| batch | 512 个问题 | 论文 OPD 设置 |
| 每题 rollout | 8 | 论文 G=8 |
| 最大检索次数 | 4 | 论文 Tmax=4 |
| 每次检索文档数 | 3 | 论文/已有 Search-R1 检索器 |
| 概率比裁剪 | [1e-6, 10] | 论文公式 (5) |
| alpha_opd | 1.0 | 论文 |
| alpha_kl | 0.001 | 工程默认，沿用已有 Search-R1；所给 9 页 PDF 未给出完整附录超参数，不能声称这是作者确认值 |
| 训练上限 | 420 updates | 依据正文 420+ 的保守默认，可覆盖；不是复现已发表曲线 |
| prompt / 总 response 上限 | 2048 / 8192 | 工程显存与上下文预算，不是论文明确报告值 |
| 单段生成 / 单次观察内容 | 512 / 512 token | 工程默认，待按完整附录校准 |
| GPU 布局 | actor+rollout+ref 4 张，teacher 2 张，retriever 默认 GPU 7 | 针对本机 8×A100 的配置，区别于论文 4×H200 |

模型之间必须使用完全一致的 tokenizer token→ID 映射。启动前通过 `--check-models` 读取 tokenizer/config 元数据检查，不会下载完整模型权重。Teacher 对相同原始 token 前缀评分，不会再套用其 instruct chat template。

## 当前服务器验证状态

已安装独立 Conda 环境 `opd_env`，具体版本、兼容处理与 GPU 算子验证见 [OPD_ENVIRONMENT.md](OPD_ENVIRONMENT.md)。代码 review 记录见 [REVIEW.md](REVIEW.md)，后续真实模型端到端训练、参数更新和 checkpoint 恢复结果见 [E2E_VALIDATION.md](E2E_VALIDATION.md)。

- 数学、检索状态机、真实本地 HTTP 协议、数据转换和 Hydra 配置：28 项便携 CPU 测试通过，opd_env 与 searchr1 均可运行。
- 新增 7 项原生接口测试全部通过：直接使用 registered loss、jagged padding、vLLM prompt-score parser、AgentLoopOutput 和 V1 TransferQueue postprocessor；模型生成、检索和队列传输使用 stub。它覆盖真实接口契约，但不启动模型或 Ray 集群。
- 已转换完整数据到 `/data/work/YingZhong/data/nq_hotpotqa_opd/`：train 169615 行、test 51713 行。
- 输入 `/data/work/YingZhong/data/nq_hotpotqa_train/` 保留。
- `opd_env` 使用 Python 3.12.15 / torch 2.13.0+cu130 / vLLM 0.29.0 / transformers 5.12.1。
- 当前主机驱动 570.133.20；本环境的 CUDA forward-compatibility 库已通过 CUDA context 和 GPU 算子验证。环境检查以实际 CUDA API/context 检查为准。
- 已验证原生 verl 导入、注册损失的真实 jagged/padding 梯度，以及 FlashAttention 前向/反向和 vLLM GPU RMSNorm。
- 已从 ModelScope 对应仓库下载 teacher、student 和 E5 模型，40 个文件逐一通过仓库元数据 SHA-256 校验；本地 tokenizer 映射检查通过。
- 已完成真实模型训练 2 步，再从第 2 步恢复并训练第 3 步，共 24 条训练轨迹、17 次实际 Wikipedia 检索。3 步梯度均有限且非零；参数抽查确认每一步确实更新，恢复后的优化器步数为 3。
- 每步的轨迹模型版本跨度为 1，陈旧度为 0。模型、优化器、RNG、学习率调度器和数据加载状态的恢复链路通过。
- 这是缩小 batch 的运行验证，未执行论文规模 420 步或完整 benchmark。2 条 smoke 验证样本的 EM 均为 0，不用于判断学习效果。

便携测试保留隔离数学检查；原生测试补充真实运行时接口检查。两者都不能称作完成实际模型分布式训练验证。

## 使用步骤

使用服务器上已经安装好的 `opd_env`：

```bash
cd /data/work/YingZhong/code/on_policy_distillation_verl
source /data/work/YingZhong/miniconda3/etc/profile.d/conda.sh
conda activate opd_env
bash examples/opd_search/setup_env.sh
```

`setup_env.sh` 现在负责激活并验证现有 Conda 环境，执行依赖一致性检查和运行时检查。它不会重新安装依赖。实际安装清单与本机兼容处理记录在 `OPD_ENVIRONMENT.md` 和 `/data/work/YingZhong/.cache/opd-env-install/`。

已有数据已完成转换；需要重建时可用下面的命令。转换工具只需要 pyarrow，也能在原 `searchr1` 中执行：

```bash
/data/work/YingZhong/miniconda3/envs/searchr1/bin/python examples/opd_search/prepare_data.py
```

在另一终端启动已有 Search-R1 检索器：

```bash
cd /data/work/YingZhong/code/on_policy_distillation_verl
bash examples/opd_search/start_retriever.sh
```

默认沿用 `/data/work/YingZhong/data/search_data/e5_Flat.index` 和 `wiki-18.jsonl`，优先使用已下载的 `/data/work/YingZhong/models/e5-base-v2`。当前服务器已安装隔离 GPU 检索依赖，约 60 GiB 的完整 FP32 flat index 与 E5 一起放在 GPU 7；合批适配器复用原 Search-R1 检索类并保持 HTTP schema。重新部署时先执行 `bash examples/opd_search/install_retriever_gpu.sh`。显式选择 CPU 模式可设置 `RETRIEVER_FAISS_DEVICE=cpu`。

在兼容训练环境下启动 OPD（运行前会检查环境、数据、tokenizer 和检索服务）：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export OPD_OUTPUT_DIR="$PWD/checkpoints/opd_search_7gpu"
bash examples/opd_search/run_formal_7gpu.sh
```

复跑已验证的真实模型 smoke 流程，先启动检索服务，再在训练终端执行：

```bash
export OPD_SMOKE_OUTPUT="$PWD/checkpoints/opd_smoke_$(date +%Y%m%d_%H%M%S)"
bash examples/opd_search/run_smoke.sh train
bash examples/opd_search/run_smoke.sh resume
```

正式七卡入口使用 GPU 0–6，分配四卡 student 训练/生成和三卡 teacher 副本；GPU 7 留给 E5 和完整 FP32 FAISS 索引。启动器设置更大的动态 token 批量、推理并发和 KV cache，关闭 actor 参数/优化器 CPU 卸载。检索服务改为请求合批和单线程原生调用，保持原 HTTP schema，避免高并发原生崩溃。完整说明和实测状态见 [FORMAL_7GPU.md](FORMAL_7GPU.md)。Ray worker 使用同一个 `opd_env` Python，正式输出默认在 `checkpoints/opd_search_7gpu/`。历史三卡 smoke 入口仍使用 `0,1,2`，结果保存在 `checkpoints/opd_smoke_20261017/`。

七卡配置已通过 1 步真实训练校准：128 条训练轨迹、132 次实际检索、8 条验证问题；student 峰值 71.35–72.23 GiB，teacher 峰值 74.29–75.42 GiB。四卡 checkpoint 和在线 W&B 指标通过审计。该校准缩小了全局 batch。后续正式实验已运行至第 350 步，完成 51,713 条完整验证后按用户要求结束；汇总严格答案 EM 为 37.49%，分数据集结果见仓库根目录 README。七卡入口的默认目标仍为 420 步。

## Checkpoint 与验证频率

默认每 50 个训练 global step 保存一次 checkpoint，并在同一步执行验证：50、100、150、200……。`trainer.test_freq=50`，`trainer.save_freq` 默认引用同一配置；正式训练和 smoke 脚本均继承该间隔。验证成绩以 `val-core/<数据集>/acc/mean@1` 等指标记录到控制台和 W&B，验证生成保存在输出目录的 `validation/`。

沿用 verl 原生行为：训练最后一步也会保存并验证，即使不是 50 的倍数。七卡正式入口设置 trainer.val_before_train=false，跳过训练前的 step 0 验证。当前 420 步配置因此在 50、100……400，以及最终 420 步保存并验证。2/3 步 smoke 仅在各自最后一步触发；此前 E2E_VALIDATION.md 中每步保存/验证的结果属于当时使用频率 1 的历史验证配置。

## W&B 实验记录

默认同时记录到控制台和 W&B，包含 OPD 损失、reference KL、概率比、梯度、学习率、验证分数、性能指标、完整配置和每次验证的 2 条问答样例。opd_env 已有 wandb 0.27.2。

在线运行前执行 `wandb login --relogin --verify`，并按需设置 `WANDB_PROJECT`、`WANDB_NAME` 和 `WANDB_ENTITY`。运行身份保存在实验输出目录的 `wandb_run.json`，在线恢复复用同一个 run ID；`WANDB_MODE=offline` 会保存为同组的独立运行分段，之后可逐段 `wandb sync`。完整操作、恢复限制和验证状态见 [WANDB.md](WANDB.md)。

本轮已用原生 SDK 保存并核验此前 3 步真实训练的离线指标与验证样例。用户更新凭据后，在线认证、上传和独立进程续接同一 run 均通过；云端 API 读回 step 1/2/3，损失与原始训练记录一致。数据明确标注为此前真实训练的回填，没有重跑模型训练。

## 测试与检查

```bash
cd /data/work/YingZhong/code/on_policy_distillation_verl
source /data/work/YingZhong/miniconda3/etc/profile.d/conda.sh
conda activate opd_env
python="$CONDA_PREFIX/bin/python"
"$python" tests/experimental/opd_search/test_opd_search.py
"$python" tests/experimental/opd_search/test_native_integration.py
"$python" tests/experimental/opd_search/test_wandb_setup.py
"$python" tests/experimental/opd_search/test_batched_retriever.py
"$python" examples/opd_search/preflight.py --config-only
"$python" examples/opd_search/preflight.py --runtime-only
```

便携 CPU 测试仍可在已有 `searchr1` 环境运行。原生接口测试需要 `opd_env`。完整训练启动器还会检查数据、tokenizer 和检索服务。

逐文件说明见 [CHANGES.md](CHANGES.md)，上游基线和模型配置见 [BASELINE.json](BASELINE.json)。
