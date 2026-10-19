# 相对上游 18cfb74852a053a27dcf301d3318a093be81b628 的逐文件变更

以下路径均相对 `/data/work/YingZhong/code/on_policy_distillation_verl`。

| 文件 | 类型 | 改动及作用 |
|---|---|---|
| `verl/workers/config/distillation.py` | 修改 | 增加 importance_ratio_min/max；新模式要求纯 OPD、禁止额外 loss/logprob clamp 和 PPO/PG 模式，校验概率比边界。 |
| `verl/trainer/config/distillation/distillation.yaml` | 修改 | 将 [1e-6,10] 裁剪边界暴露给 Hydra，保持旧损失模式默认值不变。 |
| `verl/trainer/distillation/losses.py` | 修改 | 注册 opd_search_forward_kl；复用原生 sampled-token teacher 评分、padding 和全局 token 聚合；在纯 OPD 中显式加入 reference KL；处理全掩码 microbatch；指标同样按全局 token 聚合并统计上下界裁剪。 |
| `verl/trainer/distillation/opd_search_loss.py` | 新增 | 实现 log 空间裁剪、stop-gradient 重要性权重、来源掩码的 CE，以及具有正确 reverse-KL 期望梯度的冻结 reference 正则；reference 前向保留 k3 数值。 |
| `verl/experimental/agent_loop/search_r1_protocol.py` | 新增 | 实现实时检索 HTTP 客户端、严格结果格式检查、生成/检索循环、四次检索上限、原 token IDs 连续前缀、观察掩码、预算与异常处理；跨调用保留模型版本区间。 |
| `verl/experimental/agent_loop/opd_search_agent_loop.py` | 新增 | 将检索状态机接入最新 AgentLoopBase/AgentLoopOutput，拒绝跨版本 student 轨迹；由原生 worker 对完整轨迹调用 teacher 服务。 |
| `examples/opd_search/config/agent_loops.yaml` | 新增 | 注册多轮搜索 agent 的 Hydra target。 |
| `examples/opd_search/config/opd_search.yaml` | 新增 | 7B-Instruct→3B-Base 完整 OPD 配置，连接数据、teacher、reference、检索服务和资源池。 |
| `examples/opd_search/prepare_data.py` | 新增 | 分批转换 NQ+HotpotQA parquet，保留原 prompt/答案并补 agent_name；原始数据不变。 |
| `examples/opd_search/reward.py` | 新增 | 提供答案格式和 EM 验证指标，屏蔽检索文本中的答案标签；任务奖励不参与 OPD 优化。 |
| `examples/opd_search/preflight.py` | 新增 | 检查 Hydra 合成、on-policy 不变量、依赖和驱动、数据、tokenizer 对齐和检索服务。 |
| `examples/opd_search/run_opd.sh` | 新增 | 激活 opd_env 后启动 preflight、W&B 设置准备器和原生 main_ppo，设置镜像及同一 Conda Python 的 Ray worker 环境。 |
| `examples/opd_search/run_smoke.sh` | 新增 | 使用真实本地模型，以小 batch 执行 2 步训练和恢复后的第 3 步；保留轨迹及 checkpoint，检查已有/缺失 checkpoint 以防错误覆盖或静默从头训练。 |
| `examples/opd_search/run_formal_7gpu.sh` | 新增 | GPU 0–6 正式训练入口：四卡 student、三卡 TP=1 teacher 副本，配置驻留 actor 状态、更大的动态 token 批量、推理并发和 KV cache。 |
| `examples/opd_search/batched_retriever.py` | 新增 | 保留检索 HTTP schema，异步跨请求合批，通过单个线程调用原 E5/FAISS；支持完整 FP32 GPU 索引，提供健康计数。 |
| `examples/opd_search/install_retriever_gpu.sh` | 新增 | 将固定版本 FAISS GPU 与 CUDA 12 库安装到 opd_env/retriever_gpu 隔离目录，训练主环境依赖不被替换。 |
| `examples/opd_search/wandb_setup.py` | 新增 | 为原生 W&B 后端准备在线/离线模式，验证在线身份，持久化实验 run ID，处理恢复和离线分段；生成只含非敏感设置的 shell 环境文件。 |
| `examples/opd_search/setup_env.sh` | 新增 | 激活并验证已安装的 opd_env；执行 uv pip check 与实际 CUDA context 检查。 |
| `examples/opd_search/start_retriever.sh` | 新增 | 启动与 Search-R1 HTTP schema 兼容的合批适配器，使用已下载的索引/语料和 GPU 7；安装隔离依赖后默认完整 FP32 GPU 索引，支持显式 CPU 模式。 |
| `tests/experimental/opd_search/test_opd_search.py` | 新增 | 28 项便携 CPU 测试，验证数学、全词表 reference KL 梯度、掩码、上下文、搜索次数、真实 HTTP schema、数据/EM、模型版本及优化模式下的配置检查。 |
| `tests/experimental/opd_search/test_native_integration.py` | 新增 | 7 项原生接口测试，覆盖注册损失、真实 jagged/padding、teacher 分数位移、全局指标、全掩码 microbatch、AgentLoopOutput、V1 postprocessor 和跨版本拒绝。 |
| `tests/experimental/opd_search/test_wandb_setup.py` | 新增 | 8 项 W&B 身份与恢复测试，检查同一 run 续用、独立实验、错误身份拒绝、离线分段、resume_path 和凭据不落盘。 |
| `tests/experimental/opd_search/test_batched_retriever.py` | 新增 | 4 项合批、单原生线程、top-k 分组、异常传播/恢复和 HTTP schema 测试。 |
| `examples/opd_search/BASELINE.json` | 新增 | 记录上游提交、论文版本、模型及当前复现阶段。 |
| `examples/opd_search/README.md` | 新增 | 中文算法说明、论文/工程参数差异、opd_env/W&B 运行步骤、测试证据与真实模型验证状态。 |
| `examples/opd_search/OPD_ENVIRONMENT.md` | 新增 | 实际环境版本、兼容处理、依赖差异和安装验证记录。 |
| `examples/opd_search/REVIEW.md` | 新增 | 本次 code review 的缺陷、修复、验证范围和逐文件清单。 |
| `examples/opd_search/E2E_VALIDATION.md` | 新增 | 真实模型训练、checkpoint 恢复、参数/优化器检查、检索轨迹和运行结果记录。 |
| `examples/opd_search/WANDB.md` | 新增 | W&B 指标、在线登录、离线同步、run 身份与恢复、实测结果和限制。 |
| `examples/opd_search/FORMAL_7GPU.md` | 新增 | 七卡资源、显存参数、正式命令、算法设置、校准结果和限制。 |
| `examples/opd_search/CHANGES.md` | 新增 | 本逐文件变更清单。 |

数据输出另存 `/data/work/YingZhong/data/nq_hotpotqa_opd/`，不属于 Git 源文件；模型权重、系统驱动和已有 searchr1/retriever 环境未在此次实现中修改。

## 后续：按用户要求部署 opd_env

本次环境部署额外更新以下文件，算法数学内核没有改变：

| 文件 | 更新 |
|---|---|
| `examples/opd_search/run_opd.sh` | 改为激活 `opd_env`，Ray worker 也使用同一 Conda Python，避免自动创建或同步另一套 uv 环境。 |
| `examples/opd_search/preflight.py` | 当内核驱动低于 580 时，实际检查 CUDA 前向兼容 API 和 GPU context；兼容库能工作时允许通过。 |
| `examples/opd_search/README.md` | 将初次环境检查标为历史快照，并指向新的部署记录。 |
| `examples/opd_search/OPD_ENVIRONMENT.md` | 新增当前环境版本、兼容处理、依赖差异、验证结果和激活方法。 |
| `examples/opd_search/CHANGES.md` | 追加本次部署改动说明。 |

## 后续：代码 review 与修复（2026-10-17）

本轮修改 13 个文件，其中 2 个为新增文件。Teacher 加权 CE 数学内核不变；修正 reference KL 的梯度估计。

| 文件 | 本轮更新 |
|---|---|
| `verl/experimental/agent_loop/search_r1_protocol.py` | 修复遗漏模型版本号导致 V1 统计崩溃的问题；跨生成调用合并 min/max；检索 schema 对错误 JSON/document 类型给出明确错误。 |
| `verl/experimental/agent_loop/opd_search_agent_loop.py` | 当一条轨迹跨 student 权重版本时明确报错。 |
| `verl/trainer/distillation/losses.py` | CE、概率比、裁剪比例和 reference KL 指标改为全局 token 归一化的 Metric SUM；裁剪比例包含上下界。 |
| `verl/trainer/distillation/opd_search_loss.py` | 修正直接反传 sampled k3 导致的 KL 梯度方向偏差；保留 k3 前向值，用 score-function surrogate 得到正确 reverse-KL 期望梯度。 |
| `examples/opd_search/preflight.py` | 用 ValueError 替换 assert，保持 Python -O 下的校验；拒绝异步训练、奖励筛选和禁用蒸馏；检索检查复用实际 schema parser。 |
| `examples/opd_search/config/opd_search.yaml` | 显式指定 V1 sync，固定当前实现的 on-policy 调度。 |
| `examples/opd_search/setup_env.sh` | 统一为验证已有 opd_env，移除旧 .venv 安装流程和过时的驱动版本硬拒绝。 |
| `tests/experimental/opd_search/test_opd_search.py` | 新增版本保留/合并/缺失、异步配置拒绝、Python -O、全词表 reference KL 梯度回归；补充错误检索 JSON 类型检查。 |
| `tests/experimental/opd_search/test_native_integration.py` | 新增原生损失与 V1 数据传递回归测试。 |
| `examples/opd_search/README.md` | 更新当前启动方式、测试范围和验证状态。 |
| `examples/opd_search/OPD_ENVIRONMENT.md` | 同步 setup_env.sh 的当前用途及原生测试入口。 |
| `examples/opd_search/CHANGES.md` | 更新完整清单并记录本轮 13 个文件。 |
| `examples/opd_search/REVIEW.md` | 新增审查报告。 |

## 后续：真实模型端到端验证（2026-10-17）

已完成真实 Qwen2.5-7B-Instruct teacher、Qwen2.5-3B Base student 和 E5/Wikipedia 检索，先训练 2 步，再从第 2 步恢复训练第 3 步。详见 E2E_VALIDATION.md。此轮没有修改算法数学内核。

| 文件 | 本轮改动 |
|---|---|
| `examples/opd_search/preflight.py` | 新增干净 Python 进程的 PyArrow 导入检查，识别被 torch 导入顺序掩盖的 glibc/librt 混用问题。 |
| `examples/opd_search/start_retriever.sh` | 激活 opd_env 使用 CUDA compatibility hook；使用该环境的 FAISS 与 E5；优先选择本地已下载的 e5-base-v2，仍支持显式覆盖。 |
| `examples/opd_search/run_smoke.sh`（新增） | 固定本次验证的本地模型、两张 actor GPU/一张 teacher GPU、小 batch、轨迹导出、checkpoint 和 train/resume 入口。 |
| `examples/opd_search/README.md` | 更新为已完成真实模型验证的状态，增加本地模型启动和可复跑 smoke 步骤。 |
| `examples/opd_search/OPD_ENVIRONMENT.md` | 记录 librt ELF 修复、faiss-cpu 1.15.1、最终 261 个 Python 包及真实训练验证。 |
| `examples/opd_search/REVIEW.md` | 保留 review 阶段历史，并指向后续真实模型验证结果。 |
| `examples/opd_search/E2E_VALIDATION.md`（新增） | 本轮实际结果、限制、文件路径、参数变化和恢复证据。 |
| `examples/opd_search/CHANGES.md` | 更新完整清单并追加本轮 8 个文件。 |

环境内修改另计：仅为 opd_env/bin/python3.12 增加 DT_NEEDED librt.so.1（有修改前备份），并在 opd_env 安装 faiss-cpu 1.15.1。模型权重、原始数据和已有 searchr1/retriever 环境未改写。

下载校验、参数抽查和结果审计工具及日志保存在 `/data/work/YingZhong/.cache/opd-e2e-20261017/`，不属于上游 verl 源文件。

## 后续：W&B 实验记录（2026-10-17）

本轮共 9 个文件（6 个更新、3 个新增）。当前完整实现相对上游共 26 个文件。未改动 OPD 损失、teacher/student 更新逻辑或原始模型 checkpoint。

| 文件 | 本轮改动 |
|---|---|
| `examples/opd_search/config/opd_search.yaml` | 默认 logger 改为 console + wandb；项目/实验名支持环境变量；每次验证记录 2 条问答表；默认保留完整训练与验证轨迹。 |
| `examples/opd_search/run_opd.sh` | 调用 W&B 设置准备器，安全导入非敏感环境设置，再由原生 main_ppo 创建与记录 run。 |
| `examples/opd_search/run_smoke.sh` | 支持用 WANDB_NAME 设置 smoke 实验名称，末尾 Hydra override 仍优先。 |
| `examples/opd_search/wandb_setup.py`（新增） | 在线身份检查；输出目录保存 run ID；恢复沿用身份；拒绝串项目/身份和覆盖；离线分段用同一 group；支持 resume_path。 |
| `tests/experimental/opd_search/test_wandb_setup.py`（新增） | 8 项身份、恢复和凭据处理测试。 |
| `examples/opd_search/README.md` | 添加 W&B 使用入口、记录内容与当前验证状态。 |
| `examples/opd_search/OPD_ENVIRONMENT.md` | 记录已安装 SDK、无需修改依赖、登录状态及日志目录。 |
| `examples/opd_search/WANDB.md`（新增） | 完整在线/离线操作、指标、恢复语义、验证范围与限制。 |
| `examples/opd_search/CHANGES.md` | 更新完整文件清单并追加本轮记录。 |

离线和在线记录均使用此前已完成训练的真实指标回填，明确记录数据来源。原凭据无效；用户更新后在线认证、上传及独立进程续接同一 run 均已通过，云端读回三步历史与原始指标一致。README.md、WANDB.md、OPD_ENVIRONMENT.md 和本清单已同步最终状态，登录示例增加 --verify。未将回填冒充为新增模型训练。

## 后续：每 50 步保存与验证（2026-10-17）

本轮修改 5 个文件，没有新增源文件，完整实现仍为 26 个文件。

| 文件 | 本轮改动 |
|---|---|
| `examples/opd_search/config/opd_search.yaml` | 保持 test_freq=50，将 save_freq 默认引用 test_freq，使保存和验证间隔一致。 |
| `examples/opd_search/run_smoke.sh` | 删除 test_freq=1/save_freq=1 的覆盖，继承正式配置的 50 步间隔。 |
| `examples/opd_search/README.md` | 说明 50 步调度、最终步保存与验证、step 0 基线验证和历史 smoke 结果的区别。 |
| `examples/opd_search/WANDB.md` | 说明验证分数的 W&B 指标名、global step 对齐，以及问答示例数量与验证集大小的区别。 |
| `examples/opd_search/CHANGES.md` | 追加本轮逐文件修改说明。 |

复用 verl 原生 V1 的保存、验证与日志路径：50、100……触发周期事件，最后一步额外触发。未启动新的训练任务。

## 后续：七张 A100 80 GB 正式配置（2026-10-17）

本轮修改 11 个文件（5 个新增、6 个更新），完整实现相对上游共 31 个文件。

| 文件 | 本轮改动 |
|---|---|
| `examples/opd_search/run_formal_7gpu.sh`（新增） | 限制七张可见 GPU；四卡 student + 三卡 teacher；本地模型；actor 参数/优化器驻留 GPU；动态 token 24576/32768；推理并发、CUDA graph 和 0.75/0.90 显存比例；独立正式输出目录，检索超时 180 秒。 |
| `examples/opd_search/batched_retriever.py`（新增） | 对 HTTP 查询合批，单线程调用原 E5/FAISS；显式 FP32 GPU 索引和 512 MiB scratch，保持 schema，增加健康计数。 |
| `examples/opd_search/install_retriever_gpu.sh`（新增） | 安装三个固定版本依赖到 opd_env 内隔离目录，训练主环境不升级。 |
| `examples/opd_search/start_retriever.sh` | 使用合批适配器，支持 auto/cpu/gpu 模式；仅服务进程加载隔离 GPU 包，设置线程预算。 |
| `tests/experimental/opd_search/test_batched_retriever.py`（新增） | 4 项合批/线程/异常恢复和 HTTP 契约测试。 |
| `examples/opd_search/FORMAL_7GPU.md`（新增） | 资源划分、具体显存配置、正式启动指令、恢复方式和校准结果。 |
| `examples/opd_search/README.md` | 正式训练命令改为七卡入口，并链接资源说明。 |
| `examples/opd_search/WANDB.md` | 同步七卡正式命令，说明调优参数会记录在实验配置中。 |
| `examples/opd_search/OPD_ENVIRONMENT.md` | 记录隔离 FAISS GPU/CUDA 12 版本、与训练 CUDA 13 共存验证和服务并发实测。 |
| `examples/opd_search/wandb_setup.py` | 在线启动检查改用 SDK login(verify=True) 的凭据验证，避免 Public API viewer 查询误报 relogin required；子进程禁用交互 stdin，提示增加 --verify。 |
| `examples/opd_search/CHANGES.md` | 更新全量文件清单和本轮逐文件说明。 |

最终 0.75/0.90 配置的七卡真实校准通过：128 条轨迹、132 次实际检索、8 条验证问题、一次真实参数更新，四卡 checkpoint 与在线 W&B 已核验；47 项回归通过。详细峰值、测试限制、前期检索错误和最终修复见 FORMAL_7GPU.md。仅进行了限定步数校准，未启动正式 420 步实验。

## 正式 420 步启动（2026-10-17）

- run_formal_7gpu.sh：显式设置 total_training_steps=420、val_before_train=false；保留每 50 步和最终步验证与保存。
- README.md、WANDB.md、FORMAL_7GPU.md：同步跳过初始验证的正式运行说明；校准记录与正式任务状态分开记录。
- CHANGES.md：补充本次逐文件变更。

## GitHub 发布准备（2026-10-19）

- README.md：新增 SearchOPD 项目首页、文档入口、真实 step 350 结果及适用范围。
- README.verl.md：原样保留基线提交的上游 README。
- examples/opd_search/README.md：更新正式实验已经完成至 step 350 的状态。
- examples/opd_search/CHANGES.md：记录本次文档变更。

## 仓库独立提交历史（2026-10-19）

- README.md：说明 SearchOPD 使用上游基线源码快照，项目独立记录提交历史。
- examples/opd_search/CHANGES.md：记录本次历史整理。
- GitHub main：以 YingZHONGpolyu 为唯一提交作者建立初始快照；源码版权声明、LICENSE 与 README.verl.md 保留。旧历史已在服务器仓库外备份。
