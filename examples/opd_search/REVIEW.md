# OPD 代码 review 与修复记录

日期：2026-10-17。服务器：`16_Server_Ugreen`。

本报告记录 code review 阶段的状态。随后已完成真实模型端到端训练和恢复验证，最新结果见 [E2E_VALIDATION.md](E2E_VALIDATION.md)。

仓库：`/data/work/YingZhong/code/on_policy_distillation_verl`。

上游基线：`18cfb74852a053a27dcf301d3318a093be81b628`。

审查范围：本项目增加的损失、检索状态机、AgentLoop、训练配置、环境入口和测试，以及它们调用的原生 verl V1 接口。模型为 `Qwen/Qwen2.5-7B-Instruct` teacher 和 `Qwen/Qwen2.5-3B` Base student。

## 发现并修复的问题

### P1：丢失模型版本信息，第一个训练 update 的统计阶段会崩溃

原来的搜索状态机重新创建 `extra_fields` 时只保留搜索轮数等自定义字段，丢掉 vLLM client 提供的 `min_global_steps` / `max_global_steps`。原生 V1 postprocessor 将缺失值写为 None，trainer 随后执行 `np.array(..., dtype=int)` 会报 TypeError。

修复前在服务器用带版本号 0 的生成输出成功复现：输出轨迹缺少版本号，后续整数转换失败。修复后跨每次生成调用计算最小/最大版本，保留版本 0；缺失版本号的后端输出会尽早报错。AgentLoop 还会拒绝跨 student 权重版本的轨迹。

旧的 22 项便携测试没有覆盖原生 V1 序列化和标签消费，这使问题漏检。本轮补充使用真实 AgentLoopOutput 和 V1 postprocessor 的回归测试。

### P1：reference 正则的期望梯度方向与论文定义不一致

论文将 reference 正则定义为 `KL(student || reference)`。原实现直接反传采样 k3 数值，只保留对 log-probability 的直接导数，遗漏采样分布的梯度；在 student 分布下取期望后，得到的是另一方向 KL 的梯度。

用二分类 logits `[0.7,-0.4]`、reference 概率 `[0.4,0.6]` 在服务器复现：原梯度为 `[0.350260,-0.350260]`，精确 reverse-KL 梯度为 `[0.282079,-0.282079]`，最大误差约 0.068181。

修复后前向仍返回非负 k3 数值，反向使用 `stopgrad(log_student - log_reference) * log_student` 的梯度。其在当前 student 策略采样、固定前缀下的期望与精确 reverse-KL 梯度一致。新增全词表枚举回归，分别检查数值与梯度；更新原生 registered loss 的 reference 梯度断言。Teacher 加权 CE 及其 clipping/stop-gradient 内核没有改变。

### P2：on-policy 配置约束存在绕过路径

原启动检查允许将 `trainer.v1.trainer_mode` 覆盖为异步模式，也允许开启按任务奖励筛选样本。前者引入陈旧 student 轨迹，后者改变训练样本分布，都不符合当前 OPD 实现的采样前提。此外，原先的 assert 检查可被 Python -O / PYTHONOPTIMIZE 关闭。

修复后显式配置 V1 sync，检查拒绝异步训练、奖励筛选和禁用蒸馏，所有配置合同均使用带具体原因的 ValueError。测试通过真正的 Python -O 子进程确认非法配置仍会失败。

### P2：动态 microbatch 的监控指标统计失真

原指标先计算每个 microbatch 均值，再由框架平均，导致长度不同的 microbatch 权重相同。例如一批 1 个有效 token、概率比 10，另一批 100 个有效 token、概率比均为 1，原逻辑会报告 5.5，而全局 token 均值应为 110/101≈1.0891。

修复后 CE、概率比、reference KL 和裁剪比例均按全局有效 token 数归一化，并使用原生 Metric SUM 聚合。新测试验证整体计算、拆成不同 microbatch 和模拟 DP rank 聚合的数值一致。裁剪比例也补计原先遗漏的下界。这个指标修复没有改变实际损失归一化；reference 梯度修复另见上一项。

### P2：安装入口和文档仍指向旧环境流程

尽管训练入口已使用 opd_env，旧 setup_env.sh 和 README 仍引导创建独立 .venv 并同步原始锁文件。这样会重复安装环境，且原始锁文件会重新引入部署时已记录的 NumPy/cusparselt 约束冲突。旧脚本也会直接拒绝本机已经验证有效的 CUDA forward-compatibility 方案。

修复后 setup_env.sh 激活并验证已有 opd_env，执行 uv pip check 和实际运行时检查；README 和环境记录采用一致入口。这个脚本现在是现有环境的检查入口，不是跨机器自动重装脚本。

### P2：检索启动检查没有检查真实 rollout 所需的 schema

原先仅检查 result 字段非空，错误的 document/contents 结构也可能被放行，直到 rollout 才失败。修复后 preflight 直接调用实际 format_passages parser，检查单查询 result batch、document 对象和 contents 字符串；合法空检索结果仍被允许。便携测试补充错误 JSON 和 document 类型。

## 验证范围

- 便携测试：28 项全部通过，在 opd_env 和 searchr1 分别运行。使用真实数学内核与检索状态机；部分框架适配在隔离环境执行。
- 原生接口测试：7 项全部通过，直接导入 verl 注册损失、真实 jagged/padding、vLLM prompt-score parser、AgentLoopOutput 和 V1 postprocessor。模型生成、检索和队列传输为 stub；没有启动实际模型或 Ray 集群。
- 全词表枚举检查确认 reference KL 的期望梯度与精确 reverse-KL 一致；原生梯度检查覆盖不同 prompt 长度、观察 token、reference KL 和全掩码 microbatch。prompt scoring 与训练的 token 位移检查通过。
- 原生指标检查覆盖 microbatch/DP 分拆不变性及上下界裁剪比例。
- shell 语法检查、Ruff lint/format、opd_env 依赖检查和 CUDA forward-compatibility/context 检查均通过。
- 联网 tokenizer/config 检查因 Hugging Face 镜像 TLS 握手超时，180 秒上限内未完成。本项不计为通过。错误日志在 models-preflight.log。

本轮不启动正式训练。真实 teacher/student 模型生成与打分、分布式优化器更新、checkpoint/resume 和 benchmark EM 仍待实际模型集成验证。便携测试和原生接口合同测试不能证明这些路径已经成功。

`alpha_kl=0.001`、2048/8192 上下文预算、512 token 单段生成/观察内容上限仍是已公开说明的工程默认值，需按作者完整超参数与资源情况校准。本次仍只实现 [论文](https://arxiv.org/pdf/2608.24310) 的 OPD 阶段。

## 本轮逐文件改动

以下均相对上述远程仓库根目录。共 13 个文件：11 个更新，2 个新增。

| 文件 | 改动 |
|---|---|
| verl/experimental/agent_loop/search_r1_protocol.py | 保留/合并模型版本区间，尽早拒绝缺失版本元数据；加强检索结果类型校验。 |
| verl/experimental/agent_loop/opd_search_agent_loop.py | 拒绝跨 student 权重版本轨迹。 |
| verl/trainer/distillation/losses.py | 改正动态 microbatch 指标聚合，补齐下界裁剪统计。 |
| verl/trainer/distillation/opd_search_loss.py | 修正 reference KL 的期望梯度方向，保留 k3 前向估计值。 |
| examples/opd_search/preflight.py | 显式配置合同、sync/on-policy 校验、实际检索 schema 校验。 |
| examples/opd_search/config/opd_search.yaml | 显式固定 V1 sync。 |
| examples/opd_search/setup_env.sh | 激活并检查现有 opd_env。 |
| tests/experimental/opd_search/test_opd_search.py | 补充版本信息、非法配置、Python -O 和 schema 回归。 |
| tests/experimental/opd_search/test_native_integration.py（新增） | 原生 loss、padding、AgentLoopOutput 与 V1 接口回归。 |
| examples/opd_search/README.md | 当前环境、启动方式和测试说明。 |
| examples/opd_search/OPD_ENVIRONMENT.md | 环境检查入口和原生测试命令。 |
| examples/opd_search/CHANGES.md | 当前完整清单与本轮变化。 |
| examples/opd_search/REVIEW.md（新增） | 本报告。 |

## 复查命令与备份

```bash
source /data/work/YingZhong/miniconda3/etc/profile.d/conda.sh
conda activate opd_env
cd /data/work/YingZhong/code/on_policy_distillation_verl
python tests/experimental/opd_search/test_opd_search.py
python tests/experimental/opd_search/test_native_integration.py
bash examples/opd_search/setup_env.sh
```

修改前源码快照与本轮日志保存在：

`/data/work/YingZhong/.cache/opd-code-review-20261017/`

其中 before-review.tar.gz 为修改前快照，portable-tests.log 和 native-tests.log 为测试结果。
