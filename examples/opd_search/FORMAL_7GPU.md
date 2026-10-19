# 7 × A100 80 GB 正式训练入口

服务器仓库：`/data/work/YingZhong/code/on_policy_distillation_verl`。

入口为 `examples/opd_search/run_formal_7gpu.sh`。使用 GPU 0–6，共七张训练 GPU；GPU 7 继续用于独立检索，运行 E5 编码器和完整的 FP32 FAISS 索引。磁盘上的原始索引和语料保持原样。

## 资源与显存设置

| 项目 | 正式设置 | 作用 |
|---|---:|---|
| Student/actor/ref/rollout 资源池 | 4 GPU | actor 用 FSDP，rollout 为 4 个 TP=1 副本，共用资源池并分阶段切换 |
| Teacher 资源池 | 3 GPU | 原生单 teacher 配置自动解析为 3 个 TP=1 副本，分担完整轨迹评分 |
| Actor 参数 / 优化器 CPU offload | false / false | 减少每步 CPU/GPU 数据搬运，用显存保留训练状态 |
| Actor 动态 token 批量上限 | 24576 / GPU | 增加单次前向/反向的有效 token 数，全球 batch 不变 |
| Student/ref 计算 log-prob 的 token 上限 | 32768 / GPU | 前向计算可采用更大动态批量 |
| Rollout vLLM 显存比例 | 0.75 | 根据首轮实测增加 KV cache，与训练状态共享显存，保留切换余量 |
| Rollout prefill token / 并发序列上限 | 16384 / 128 | 提高每个推理副本的批量与并发 |
| Rollout CUDA graph | 开启；捕获 1–64 的指定批量 | 提高生成阶段效率，限制 graph 常驻显存 |
| Teacher vLLM 显存比例 | 0.90 | 独立资源池可为权重与 KV cache 留更大空间 |
| Teacher prefill token / 并发序列上限 | 8192 / 64 | 完整轨迹评分以 prefill 为主，使用 chunked prefill |
| Teacher eager | true | 保留已验证的评分路径，避免为单 token 输出捕获大量 graph |
| Agent workers / reward workers / Ray CPU | 16 / 8 / 64 | 提供并行的轨迹、检索与打分处理 |
| 检索请求超时 | 180 秒 | 为大量并发请求保留排队时间；正常 GPU 查询不应达到该上限 |

Ray 在可见的 GPU 0–6 中安排两个资源池；以启动日志中的实际分配为准，不依赖某个物理编号必须承担固定角色。

Reference 沿用当前 verl FSDP 后端的 forward-only CPU offload；该上游路径会强制卸载 reference。这里关闭的是 actor 参数与优化器的卸载，没有把一个会被后端忽略的 reference 开关宣称为显存常驻。

梯度检查点、去 padding、动态 batching、vLLM sleep/free-cache 继续开启。同步 OPD 的生成、评分和参数更新分阶段进行，显存与计算利用率会随阶段变化。vLLM 预留的 KV cache 不等同于实际有效 token 占用；吞吐与显存实测应一起判断。

## 保留的实验设置

训练 batch=512、每题 rollout=8，即每步最多 4096 条轨迹；ppo_mini_batch_size=512、ppo_epochs=1，保持每批一次 on-policy 更新。teacher/student、学习率、OPD/KL 系数、温度、top-p、top-k、prompt/response 预算和检索规则均继承原配置。

总训练步数仍为 420；默认每 50 步保存 checkpoint 并验证，最后 420 步额外执行，按正式实验要求跳过训练前的 step 0 验证，第一次完整验证在第 50 步。W&B 记录完整配置、训练指标、验证成绩和问答样例。正式验证使用原验证集，上传 2 条样例并不限制验证集规模。

## 启动命令

检索 GPU 依赖已安装到 opd_env 内的隔离目录 `retriever_gpu/`。重新部署时先执行一次 `bash examples/opd_search/install_retriever_gpu.sh`，再启动检索；训练主环境仍使用原有 CUDA 13 依赖和 faiss-cpu，只有检索进程通过 PYTHONPATH 加载 CUDA 12 FAISS。

检索终端：

```bash
cd /data/work/YingZhong/code/on_policy_distillation_verl
RETRIEVER_GPU=7 RETRIEVER_FAISS_DEVICE=gpu OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  bash examples/opd_search/start_retriever.sh
```

训练终端：

```bash
cd /data/work/YingZhong/code/on_policy_distillation_verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export WANDB_MODE=online
export WANDB_PROJECT=opd_search
export WANDB_NAME=qwen25_7b_to_3b_7gpu
export OPD_OUTPUT_DIR=/data/work/YingZhong/code/on_policy_distillation_verl/checkpoints/opd_search_7gpu
bash examples/opd_search/run_formal_7gpu.sh
```

启动器自动激活 opd_env，并使用已下载的本地模型，无需访问 Hugging Face。重复启动同一实验时保留 OPD_OUTPUT_DIR，原生 resume_mode=auto 会按 checkpoint 恢复，W&B 复用输出目录保存的 run ID。全新实验应换输出目录和名称。不要将此前两卡 checkpoint 直接当作四卡 FSDP checkpoint 恢复。

末尾可追加显式 Hydra overrides。硬件 token/memory 设置属于吞吐调优；全局 batch 等算法设置需要与训练目标一起考虑。

## 校准证据

最终 0.75/0.90 配置已完成限定 1 步的真实训练校准：16 条问题、每题 8 条轨迹，共 128 条训练轨迹、132 次实际训练检索，另有 8 条验证问题；轨迹中止比例为 0。训练退出码为 0，loss=1.7505341545、裁剪前梯度范数=6.2841567993。参数抽查 512/512 元素改变，最大变化约 1e-6；优化器状态步数为 1，四个 rank 的模型、优化器、额外状态及 data.pt 均已保存。

逐卡峰值如下。数值来自每 2 秒采样的 nvidia-smi，包含服务器原有占用及 vLLM 预留的 KV cache，不表示整段时间都在执行有效计算。

| GPU | 本轮角色 | 峰值 MiB | 峰值 GiB |
|---:|---|---:|---:|
| 0 | student/actor/rollout | 73585 | 71.86 |
| 1 | student/actor/rollout | 73059 | 71.35 |
| 2 | student/actor/rollout | 73173 | 71.46 |
| 3 | student/actor/rollout | 73965 | 72.23 |
| 4 | teacher 副本 | 77231 | 75.42 |
| 5 | teacher 副本 | 76669 | 74.87 |
| 6 | teacher 副本 | 76077 | 74.29 |
| 7 | E5 + FP32 FAISS（独立检索） | 65981 | 64.43 |

包含冷启动总耗时约 433.5 秒；本步训练/保存/同步约 59.94 秒，验证约 63.72 秒。此处是 batch=16 的硬件校准，不代表已运行正式 batch=512 的 420 步或证明全量数据收敛，也不是不同配置间的严格吞吐对照实验。

本轮为真实在线训练记录：[qwen25_7gpu_memory_calibration_attempt4](https://wandb.ai/yingzhong-the-hong-kong-polytechnic-university/opd_search/runs/b28501387b17)。云端状态为 finished，读回的损失与本地一致，四卡/三卡资源配置、验证指标和问答表均存在。SDK 同步成功后退出回调出现一次被忽略的 BrokenPipeError；进程退出码为 0，云端数据已核对完整。

完整回归 47/47、主环境 261 个包依赖检查通过；新增 GPU 库只存在于检索专用目录。校准任务及其检索进程已退出，各卡显存恢复验证前基线。原始模型、索引和语料没有改写。以上记录为正式实验之前的校准结果；正式任务的实时状态以独立启动记录和训练日志为准。

校准输出：`/data/work/YingZhong/code/on_policy_distillation_verl/checkpoints/opd_7gpu_calibration_attempt4_20261017/`。日志、逐卡采样、权重/优化器审计、云端读回证据保存在 `/data/work/YingZhong/.cache/opd-7gpu-20261017/`。

首轮 0.70 rollout 配置完成更新但原 Search-R1 同步 HTTP 端点在高并发下发生原生段错误，不能把该轮视为完整通过。串行合批的 CPU 索引消除了并发进入原生库，但全量查询出现严重排队，第二轮在检索检查超时，未进入训练。

当前由 `batched_retriever.py` 保持 `/retrieve` 响应格式，异步合并最多 32 条查询，通过单个专用线程调用原 E5/FAISS 检索器；GPU 索引显式设定 `useFloat16=False`，scratch 512 MiB。GPU FAISS 的小规模 FP32 kNN 与 CPU 结果对照通过，并验证与当前 CUDA 13 Torch 在同一进程工作。真实完整索引的 64 路并发、128 次 HTTP 查询全部成功，总耗时约 1.39 秒，最大单次约 0.98 秒。

CPU 模式可通过 `RETRIEVER_FAISS_DEVICE=cpu` 显式选择；正式 4096 轨迹并发配置使用 GPU 模式。检索适配器复用原 Search-R1 的模型与检索类，未修改其仓库源文件。

调参依据：[verl 性能指南](https://github.com/verl-project/verl/blob/main/docs/perf/perf_tuning.rst)、[vLLM 调优指南](https://docs.vllm.ai/en/v0.20.0/configuration/optimization/)，具体行为以本仓库固定的 verl 提交与 vLLM 0.29.0 实测为准。
