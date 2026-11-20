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
