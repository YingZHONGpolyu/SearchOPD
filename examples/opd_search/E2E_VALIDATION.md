# 真实模型端到端训练验证

日期：2026-10-17。服务器：16_Server_Ugreen。

仓库：`/data/work/YingZhong/code/on_policy_distillation_verl`。

上游基线：`18cfb74852a053a27dcf301d3318a093be81b628`。

## 结论与范围

真实模型端到端验证通过：student 采样、实时 E5/Wikipedia 检索、teacher 打分、reference 打分、OPD 反向与优化器更新、模型同步、验证生成、checkpoint 保存和恢复均实际执行。先运行 2 步，进程退出后从 global_step_2 恢复，再运行第 3 步；两次训练进程退出码均为 0。

本次使用真实完整权重，没有替换成小模型或模拟 teacher/检索。规模为 8 条训练问题、每步 batch=4、每题 rollout=2，3 次更新共 24 条训练轨迹；每步另对固定的 2 条测试问题生成验证答案。它验证运行链路，不代表完成 420 步训练、收敛或论文 benchmark 复现。2 条 smoke 验证问题的 EM 均为 0，不能据此判断学习效果。

## 模型、数据和资源

| 项目 | 实际设置 |
|---|---|
| Student | Qwen2.5-3B Base；`/data/work/YingZhong/models/Qwen2.5-3B` |
| Teacher | Qwen2.5-7B-Instruct；`/data/work/YingZhong/models/Qwen2.5-7B-Instruct` |
| Retriever | E5-base-v2；`/data/work/YingZhong/models/e5-base-v2` |
| 索引/语料 | 已下载的 e5_Flat.index 和 wiki-18.jsonl，实际加载完整索引 |
| 下载来源 | ModelScope 对应 Qwen/intfloat 仓库，40 个文件共 21,865,240,653 字节 |
| 完整性 | 逐文件 SHA-256 与仓库元数据一致，训练后再次核验全部 40 个原始文件未改变 |
| Tokenizer | 151665 个 token，teacher/student token→ID 映射完全一致，ID 均在模型词表范围内 |
| 训练数据 | nq_hotpotqa_opd/train.parquet，按 seed=42 选 8 条问题 |
| 验证数据 | nq_hotpotqa_opd/test.parquet，按 seed=42 选 2 条问题（TriviaQA、2Wiki） |
| GPU | actor/ref/student rollout 使用 0、1；teacher 使用 2；E5 编码使用 7 |
| 并行/运行时 | 真实两卡 FSDP、NCCL、Ray、vLLM；opd_env |
| 采样与更新 | temperature=1、top-p=1、top-k=-1；同步；每批只更新一次 |
| 上下文预算 | prompt 1024、总 response 2048、引擎 max_model_len 4096 |
| 检索 | top-3，最多 4 次；每次观察内容最多 512 token |
| 其他 | lr=1e-6，alpha_opd=1，alpha_kl=0.001；smoke 使用 eager 并关闭 torch.compile |

## 实际训练结果

| 更新步 | 蒸馏总损失 | 裁剪前梯度范数 | Reference KL | 训练轨迹内实际检索次数 |
|---:|---:|---:|---:|---:|
| 1 | 1.3406890035 | 14.8669576645 | 0 | 8 |
| 2 | 1.4270042777 | 23.3914413452 | 0.0008872890 | 5 |
| 3（恢复后） | 1.0057046711 | 17.1813430786 | 0.0010138117 | 4 |

3 步的损失、梯度和概率比均有限；轨迹版本跨度最大值为 1，陈旧度最大值为 0。第 1 步 reference KL 为 0，更新后为非零，与冻结初始 reference 的设计一致。

检索计数只统计完整 search 之后由工具插入、带 `Doc 1:` 标记的 observation，并与 HTTP 服务实际请求交叉确认。初始 Base 模型也会自行生成 `<information>` 文本，这些模型生成内容不能算作真实检索。训练代码按 token 来源掩码，而不是按标签字符串判断来源。

## 参数与恢复证据

抽查 rank 0 checkpoint 中 `model.layers.0.self_attn.q_proj.weight` 的 512 个元素：

| 比较 | 改变元素数 | 最大绝对变化 |
|---|---:|---:|
| 原始模型 → step 1 | 512 / 512 | 1.0002404451e-6 |
| step 1 → step 2 | 512 / 512 | 1.0002404451e-6 |
| step 2 → step 3 | 512 / 512 | 1.0021030903e-6 |

抽查值均有限。直接读取优化器 checkpoint，step 2 的所有已初始化状态步数为 2；恢复后的 step 3 为 3。没有用文件哈希差异代替参数数值比较。

恢复日志明确记录：从 global_step_2 设置 global step=2，载入两张卡的 model、optimizer、RNG 和 lr_scheduler。恢复进程仅产生 step 3 的训练指标，进度从 2/3 接续，未重跑 step 1/2。数据加载状态文件 data.pt 也由原生恢复路径载入。

## 本轮解决的运行问题

1. Hugging Face 及镜像不可达。下载对应 ModelScope 仓库的原始文件，校验 SHA-256，训练使用本地路径和 offline 模式。
2. PyArrow 先于 torch 导入时混用了系统 librt 与环境 glibc。仅为 opd_env 的 Python ELF 增加 DT_NEEDED librt.so.1，使环境内运行库在扩展导入前加载，并增加干净进程导入检查。
3. 旧 retriever 环境的 transformers 与 torch 版本不匹配。检索服务统一到 opd_env，增加 faiss-cpu 1.15.1，完成实际 E5 GPU 编码和 CPU flat index 查询。

本轮没有修改 OPD 数学内核。源码逐文件说明见 [CHANGES.md](CHANGES.md) 最后一节，共 8 个文件（6 个更新、2 个新增）。

## 复跑入口

先在一个终端启动检索服务：

```bash
cd /data/work/YingZhong/code/on_policy_distillation_verl
bash examples/opd_search/start_retriever.sh
```

再在训练终端创建一个新的输出目录，先训练，再恢复：

```bash
cd /data/work/YingZhong/code/on_policy_distillation_verl
export OPD_SMOKE_OUTPUT="$PWD/checkpoints/opd_smoke_$(date +%Y%m%d_%H%M%S)"
bash examples/opd_search/run_smoke.sh train
bash examples/opd_search/run_smoke.sh resume
```

脚本固定本次已经验证的模型和小规模配置，支持末尾追加 Hydra overrides。train 会拒绝覆盖已有 checkpoint 的目录；resume 会拒绝在没有 checkpoint 时静默从头训练。

## 输出、审计和验证后的状态

Checkpoint 根目录：

`/data/work/YingZhong/code/on_policy_distillation_verl/checkpoints/opd_smoke_20261017/`

保留 global_step_1、global_step_2、global_step_3 三个完整验证快照，共约 107 GiB；包括两个 rank 的模型/优化器/额外状态、HF 配置/tokenizer 和数据加载状态。rollouts/1.jsonl、2.jsonl、3.jsonl 保存 24 条训练轨迹；validation/ 保存每步 2 条验证生成。

日志和结构化证据：

`/data/work/YingZhong/.cache/opd-e2e-20261017/`

- train-attempt1.log / resume-attempt1.log：真实训练和恢复日志，exitcode 文件均为 0。
- e2e-verification.json：结果审计为 PASS，包含每步完整指标、实际检索次数和参数变化。
- step1/2/3-weight-probe.json：参数抽查数值与优化器步数。
- model-provenance.json：所有模型文件的仓库、revision、下载 URL、大小和 SHA-256。
- live-search-example.json：真实训练轨迹样例。
- nccl-check.json / ray-precheck.json：真实两卡通信和独立 Ray worker 验证。
- pip-installed-final.txt / conda-packages-explicit.txt：最终依赖清单。
- cleanup.json：仅停止本次启动的检索服务，GPU 显存占用恢复验证前状态。

最后复跑 28 项便携测试和 7 项原生接口测试，35 项全部通过。Ruff、shell 语法、261 个 Python 包依赖、PyArrow 干净进程导入与 CUDA context 检查通过。正式规模训练和完整评估仍是后续实验。
