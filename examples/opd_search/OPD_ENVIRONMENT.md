# opd_env 安装与验证记录

服务器：`16_Server_Ugreen`。环境位置：`/data/work/YingZhong/miniconda3/envs/opd_env`。

```bash
source /data/work/YingZhong/miniconda3/etc/profile.d/conda.sh
conda activate opd_env
cd /data/work/YingZhong/code/on_policy_distillation_verl
```

## 安装版本

| 组件 | 已安装版本 |
|---|---|
| Python | 3.12.15 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime | 13.0 |
| vLLM | 0.29.0 |
| FlashAttention | 2.8.3 |
| Transformers | 5.12.1 |
| verl | 0.10.0.dev0，可编辑安装，指向当前 OPD 仓库 |
| NumPy | 2.3.5 |
| nvidia-cusparselt-cu13 | 0.8.1 |
| FFmpeg | 7.1.0 |
| CUDA compatibility package | 13.0.2，用户态驱动库 580.95.05 |
| 环境内 glibc/sysroot | 2.34 |
| faiss-cpu | 1.15.1，用于现有 Wikipedia flat index 的 CPU 检索 |

依赖使用 uv 安装进用户明确要求的 Conda 环境。先从上游 uv.lock 导出 fsdp+vllm 依赖，再按本机平台选取锁文件记录的精确 wheel URL 和 SHA-256，避免全局多索引选错同版本的 CUDA 构建。除下述修正外，核心依赖版本沿用上游锁定设置。

## 为安装成功所做的修正

1. 上游 FlashAttention wheel 锁定地址缺失实际发布 tag 的 `-cu130` 后缀。使用同一 verl 官方仓库的 `flash-attention-v2.8.3-py3.12-torch2.13.0-cu130` 发布，资产 ID `646119999`，SHA-256 为 `ea61a750254b3f0b79171d58ee3fe4627d601411ee9801dd1705855c1ece6bf3`。
2. 上游固定的 `nvidia-cusparselt-cu13==0.8.0` 不满足所装 PyTorch 的 `==0.8.1` 要求，因此改为 0.8.1。
3. 上游 NumPy 2.4.6 与 mistral-common 的 `<2.4` 约束冲突，因此改为 2.3.5。修正后 uv pip check 通过。
4. 主机内核驱动仍为 570.133.20。A100 属于 NVIDIA 支持 CUDA 前向兼容的数据中心 GPU，因此在环境内安装 CUDA compatibility package。Conda 激活 hook 把本环境的 `cuda-compat` 目录加入 `LD_LIBRARY_PATH`，停用环境时恢复原值。
5. 主机系统 glibc 仍为 2.31，但官方 FlashAttention wheel 要求 GLIBC_2.32。环境内安装 sysroot/glibc 2.34，并仅修改 `opd_env/bin/python3.12` 的 ELF interpreter/RPATH，使这个 Python 使用环境内的 loader/glibc。修改前已断开 Conda hardlink，备份保存在 `/data/work/YingZhong/.cache/opd-env-install/python3.12.original`；没有修改系统 libc、系统 driver 或其他 Conda 环境。
6. 真实训练准备时发现 Arrow 在先于 torch 导入时会通过自身 RUNPATH 加载系统 librt，混用两套 glibc 并触发 GLIBC_PRIVATE 错误。在同一 Python ELF 增加 `DT_NEEDED librt.so.1`，使环境内 librt 在扩展导入前被加载。新备份在 `/data/work/YingZhong/.cache/opd-e2e-20261017/python3.12.before-librt-fix`。preflight 新增独立 Python 进程的 PyArrow 导入检查。
7. 使用 uv 将 `faiss-cpu==1.15.1` 安装到 opd_env，实际完成 E5 GPU 编码和已有 Wikipedia flat index 的 CPU 检索。最终 261 个 Python 包的依赖检查通过。

兼容方案依据 [NVIDIA CUDA Forward Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)。后续已通过真实 Ray worker、两卡 NCCL、FSDP/vLLM、3 次 OPD 更新和 checkpoint 恢复，具体证据见 [E2E_VALIDATION.md](E2E_VALIDATION.md)。验证范围为该报告中的小规模配置。

## 已完成验证

- uv pip check：所有已安装 Python 包的依赖兼容。
- `torch.cuda.is_available()` 为 True，GPU tensor 运算成功。
- FlashAttention、vLLM、verl 成功导入。
- FlashAttention 在 A100 上完成真实前向/反向传播，与 FP32 参考结果最大绝对误差约 0.000643。
- vLLM 原生 GPU RMSNorm 算子运行成功，与参考结果误差约 0.000732。
- 原生 verl `distillation_ppo_loss` 成功调用新的 OPD 模式，使用真实 jagged tensor 和原生 padding 逻辑；包含 reference KL，观察位置梯度为零。
- OPD AgentLoop 与原生 `verl.trainer.main_ppo` 成功导入。
- 环境初次安装时原有 22 项 OPD 测试通过；后续代码 review 扩充为 28 项便携测试与 7 项原生接口测试，均在 opd_env 通过，详见 [REVIEW.md](REVIEW.md)。

后续已下载并校验 teacher/student/E5 权重，本地 tokenizer 对齐、真实模型 rollout、teacher 打分、优化器更新、checkpoint/resume 和多 GPU 通信均已通过。完整论文规模训练与 benchmark 尚未执行。

## 启动

`run_opd.sh` 使用 `opd_env`，并将 Ray worker 的 Python 指向同一环境。代码 review 后，`setup_env.sh` 也统一为激活并验证现有 `opd_env`，执行 `uv pip check` 和运行时检查，不重新安装依赖。

```bash
conda activate opd_env
cd /data/work/YingZhong/code/on_policy_distillation_verl
python examples/opd_search/preflight.py
python tests/experimental/opd_search/test_opd_search.py
python tests/experimental/opd_search/test_native_integration.py
# 检索服务和模型准备好后再执行：
bash examples/opd_search/run_opd.sh
```

Conda 激活时还会设置 Hugging Face 镜像、禁用 Xet，以及设置此环境的 CUDA_HOME。以后若用 Conda 更新/替换 Python，可执行文件可能被覆盖，需要重新检查 ELF interpreter/RPATH 和 FlashAttention 导入；不要直接用原始 uv.lock 重新覆盖已修正的 NumPy/cusparselt 版本。

完整的实际安装清单、精确下载记录保存在 `/data/work/YingZhong/.cache/opd-env-install/`。

本轮补充的最终依赖清单、模型哈希、真实训练/恢复日志和清理记录保存在 `/data/work/YingZhong/.cache/opd-e2e-20261017/`。

## W&B 记录补充（2026-10-17）

opd_env 已有 `wandb==0.27.2`，本轮未安装或升级依赖。默认启用 console + wandb，由原生 verl Tracking 记录配置和指标。认证、离线保存和断点恢复说明见 [WANDB.md](WANDB.md)。认证信息通过 SDK 登录或环境变量传入，不存入配置和 run 身份文件。

W&B 本地 SDK 文件及验证日志保存在 `/data/work/YingZhong/.cache/opd-wandb-20261017/`。原有凭据曾被拒绝，用户更新后已通过 `wandb login --verify`；离线指标、问答表、在线上传、独立进程续接同一 run 与云端历史读回均通过。证据见 online-verification.json，run ID 为 db2caf14c2b6。

## 七卡正式配置与 GPU 检索补充（2026-10-17）

训练仍使用原有 opd_env 的 Torch 2.13.0+cu130、vLLM 0.29.0、FAISS CPU 1.15.1 及既有依赖。新增隔离目录 `$CONDA_PREFIX/retriever_gpu/`，只有检索启动器临时将它加到 PYTHONPATH：faiss-gpu-cu12 1.14.1.post1、nvidia-cuda-runtime-cu12 12.1.105、nvidia-cublas-cu12 12.1.3.1。安装入口是 install_retriever_gpu.sh，不卸载或覆盖训练主环境中的包。

已实际验证隔离 FAISS GPU 与 CUDA 13 Torch 同进程工作，FP32 kNN 与 CPU 结果一致。完整索引以 FP32 放到 GPU 7，E5 使用同一张卡。新的合批服务通过单个线程调用 E5/FAISS，避免旧同步 HTTP 端点的并发原生调用。64 路并发、128 次真实查询通过；具体校准记录见 FORMAL_7GPU.md 和 `/data/work/YingZhong/.cache/opd-7gpu-20261017/`。
