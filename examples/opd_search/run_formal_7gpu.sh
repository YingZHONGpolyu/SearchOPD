#!/usr/bin/env bash
# A100 80 GB: four colocated student GPUs plus three TP=1 teacher replicas.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if ((${#gpu_ids[@]} != 7)); then
    echo "This profile requires seven visible GPUs; set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6." >&2
    exit 2
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
model_root=${OPD_MODEL_ROOT:-/data/work/YingZhong/models}
output=${OPD_OUTPUT_DIR:-$repo_root/checkpoints/opd_search_7gpu}
cd "$repo_root"
exec bash examples/opd_search/run_opd.sh \
    "actor_rollout_ref.model.path=$model_root/Qwen2.5-3B" \
    "distillation.teacher_models.teacher_model.model_path=$model_root/Qwen2.5-7B-Instruct" \
    trainer.n_gpus_per_node=4 distillation.n_gpus_per_node=3 \
    distillation.teacher_models.teacher_model.num_replicas=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.enforce_eager=false \
    'actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32,64]' \
    actor_rollout_ref.rollout.disable_log_stats=false \
    actor_rollout_ref.rollout.agent.num_workers=16 \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.90 \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=8192 \
    distillation.teacher_models.teacher_model.inference.max_num_seqs=64 \
    distillation.teacher_models.teacher_model.inference.enforce_eager=true \
    distillation.teacher_models.teacher_model.inference.disable_log_stats=false \
    reward.num_workers=8 ray_kwargs.ray_init.num_cpus=64 \
    opd_search.timeout=180 \
    trainer.total_training_steps=420 trainer.val_before_train=false \
    "trainer.default_local_dir=$output" \
    "trainer.experiment_name=${WANDB_NAME:-qwen25_7b_to_3b_7gpu}" \
    "$@"
