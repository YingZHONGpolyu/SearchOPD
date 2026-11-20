# SearchOPD: On-Policy Distillation for Search-Augmented Language Models

On-policy distillation for search-augmented language models, built on verl with teacher-guided learning, multi-turn retrieval, and experiment tracking.

SearchOPD 在 verl 原生训练流程上实现带检索工具的 on-policy distillation：student 自主生成搜索与回答轨迹，teacher 对采样 token 评分，使用带重要性权重的蒸馏目标训练 student。检索观察不参与模型生成 token 的损失，任务答案 EM 用于记录与评测，不作为 OPD 的策略梯度奖励。

## 项目与上游

本仓库基于 [verl](https://github.com/verl-project/verl)，开发基线为 `18cfb74852a053a27dcf301d3318a093be81b628`，以该基线的源代码快照开始本项目的独立提交历史，保留上游[许可证](LICENSE)和[原始 README](README.verl.md)。主要新增实现集中在 `examples/opd_search/`、`tests/experimental/opd_search/`，以及 verl 的 distillation 和 agent-loop 接口。


## 已实现能力

- Qwen2.5-7B-Instruct teacher → Qwen2.5-3B Base student，使用相同 tokenizer 对齐约束。
- 多轮 search / information / think / answer 协议，来源 token 掩码、上下文预算和最多四次检索。
- Sampled-token 重要性加权蒸馏，权重 stop-gradient、区间裁剪，以及冻结 reference KL 正则。
- 对接 Search-R1 检索接口，异步合批与串行原生调用；可使用完整 FP32 GPU FAISS 索引。
- 原生 FSDP / vLLM 训练、checkpoint 恢复、W&B 在线与离线记录。
- 每 50 步完整验证和保存 checkpoint；七卡正式入口默认跳过训练前验证。

## 运行与文档

先阅读 [OPD 使用说明](examples/opd_search/README.md)，按数据、模型、环境与检索服务的顺序准备。当前脚本包含原实验服务器路径，需要按自己的机器修改或通过支持的配置项覆盖；它们不是在任意机器上自动安装全部依赖的一键脚本。

| 文档 | 内容 |
|---|---|
| [OPD 使用说明](examples/opd_search/README.md) | 算法、数据准备、配置与运行入口 |
| [环境说明](examples/opd_search/OPD_ENVIRONMENT.md) | 经过验证的软件版本与服务器兼容处理 |
| [七卡配置](examples/opd_search/FORMAL_7GPU.md) | student/teacher 资源分配、检索 GPU 和显存设置 |
| [W&B 记录](examples/opd_search/WANDB.md) | 登录、指标、实验身份与恢复 |
| [真实模型检查](examples/opd_search/E2E_VALIDATION.md) | 小规模训练、恢复与 checkpoint 验证记录 |

原实验使用 GPU 0–6 的七张 A100 80GB 训练（四卡 student、三卡 teacher），另用 GPU 7 运行 E5 与完整 FP32 FAISS 索引。检索服务需先就绪。正式训练入口为：

```bash
bash examples/opd_search/run_formal_7gpu.sh
```

该入口默认目标为 420 步。下述已完成实验按用户要求，在第 350 步 checkpoint 完整验证后结束；不要将 350 步结果理解为已经跑完 420 步。

## 如何运行

### 1. 环境与模型准备

已验证的正式环境为 Linux、8 张 A100 80GB 和独立 Conda 环境 `opd_env`。四张 GPU 用于 student，三张用于 teacher，一张用于检索。依赖版本及安装兼容处理见 [环境说明](examples/opd_search/OPD_ENVIRONMENT.md)。`setup_env.sh` 会激活并验证现有环境。

获取代码可以运行 `git clone https://github.com/YingZHONGpolyu/SearchOPD.git`。下面的命令使用原实验服务器路径，新机器请替换相应路径。

```bash
cd /data/work/YingZhong/code/on_policy_distillation_verl
export OPD_CONDA_ROOT=/data/work/YingZhong/miniconda3
export OPD_CONDA_ENV=opd_env
export OPD_MODEL_ROOT=/data/work/YingZhong/models
source "$OPD_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$OPD_CONDA_ENV"
bash examples/opd_search/setup_env.sh
```

提前将模型权重和 tokenizer 下载到 `$OPD_MODEL_ROOT/Qwen2.5-3B`、`$OPD_MODEL_ROOT/Qwen2.5-7B-Instruct`，将检索编码器放到 `$OPD_MODEL_ROOT/e5-base-v2`。正式入口以离线模式加载本地模型。

### 2. 准备问答数据

输入目录需要包含 `train.parquet` 和 `test.parquet`。原服务器已有转换后的数据，需要重建时执行以下命令。

```bash
python examples/opd_search/prepare_data.py \
  --input-dir /data/work/YingZhong/data/nq_hotpotqa_train \
  --output-dir /data/work/YingZhong/data/nq_hotpotqa_opd
```

### 3. 启动检索服务

先准备 Search-R1 代码目录，以及配套的 `e5_Flat.index` 和 `wiki-18.jsonl`。模型、语料和索引需要自行准备。在终端 A 中进入本仓库，设置上述环境变量后运行，并保持服务进程运行。

```bash
export SEARCH_R1_ROOT=/data/work/YingZhong/code/Search-R1
export SEARCH_DATA_DIR=/data/work/YingZhong/data/search_data
# 新环境首次使用 GPU FAISS 时执行
bash examples/opd_search/install_retriever_gpu.sh
RETRIEVER_GPU=7 RETRIEVER_FAISS_DEVICE=gpu \
  bash examples/opd_search/start_retriever.sh
```

默认服务地址为 `http://127.0.0.1:8000/retrieve`，每次返回 top-3 结果。完整 FP32 索引约占 60 GiB，GPU 7 同时运行 E5。可设置 `RETRIEVER_FAISS_DEVICE=cpu` 使用 CPU FAISS。

### 4. 启动 OPD 训练

在终端 B 中进入仓库，设置步骤 1 的环境变量并激活 `opd_env` 后执行。

```bash
python examples/opd_search/preflight.py --config-only
python examples/opd_search/preflight.py --runtime-only
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export OPD_OUTPUT_DIR="$PWD/checkpoints/opd_search_new"
export WANDB_MODE=offline
bash examples/opd_search/run_formal_7gpu.sh
```

训练启动器还会检查数据、模型 tokenizer 和检索服务。默认 batch=512，每题采样 8 条轨迹，目标 420 步，每 50 步验证和保存。在线记录可先运行 `wandb login`，再设置 `WANDB_MODE=online`，详见 [W&B 说明](examples/opd_search/WANDB.md)。

其他机器的数据路径和检索地址可通过 Hydra 参数覆盖。

```bash
bash examples/opd_search/run_formal_7gpu.sh \
  data.train_files=/path/to/opd/train.parquet \
  data.val_files=/path/to/opd/test.parquet \
  opd_search.url=http://127.0.0.1:8000/retrieve
```

### 5. 小规模检查与恢复

检索服务就绪后，可先运行独立的小规模训练和恢复流程。该入口的 GPU 配置见 [OPD 使用说明](examples/opd_search/README.md)。

```bash
export OPD_SMOKE_OUTPUT="$PWD/checkpoints/opd_smoke_new"
bash examples/opd_search/run_smoke.sh train
bash examples/opd_search/run_smoke.sh resume
```

正式入口在相同输出目录重复启动时，通过 `resume_mode=auto` 恢复 checkpoint。新实验请使用独立的 `OPD_OUTPUT_DIR`。训练指标显示在控制台和 W&B，验证成绩使用 `val-core/<dataset>/acc/mean@1`，验证生成记录位于输出目录的 `validation/`。

## 已完成实验：step 350

Teacher 为 Qwen2.5-7B-Instruct，student 为 Qwen2.5-3B Base；训练问题 batch=512，每题生成 8 条轨迹。第 350 步对 51,713 条验证样本执行单次生成评测，19,386 条严格答案匹配正确，按样本数加权 EM 为 **37.49%**。

| 数据集 | 答案 EM / acc mean@1 |
|---|---:|
| NQ | 34.90% |
| TriviaQA | 55.24% |
| PopQA | 35.18% |
| HotpotQA | 33.30% |
| 2WikiMultihopQA | 32.17% |
| MuSiQue | 12.58% |
| Bamboogle | 33.60% |

评测采用本项目的标准化答案严格匹配规则，不含 supporting-facts 评分；表格不宣称为各数据集官方完整评测或公平的外部模型对照。checkpoint 和原始生成数据不包含在代码仓库中。

## 测试

在已配置的 `opd_env` 中运行以下测试，无需加载模型权重或启动训练集群：

```bash
python -m unittest discover -s tests/experimental/opd_search -p 'test_*.py' -v
```

测试涵盖损失与梯度、来源掩码、原生 verl 接口、搜索状态机、检索合批和 W&B 身份/恢复。真实训练与完整验证属于单独的实验验证，不能由 CPU 测试代替。

## 许可证与实现记录

保留上游 Apache-2.0 许可证及源文件版权声明。实现过程使用了 AI 辅助；算法细节、实现范围和限制记录在上述文档中。
