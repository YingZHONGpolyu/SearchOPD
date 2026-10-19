#!/usr/bin/env bash
# Real Qwen 7B -> 3B, live E5/Wikipedia retrieval, optimizer and resume checks.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
phase=${1:-train}
if (($#)); then shift; fi
case "$phase" in
    train) steps=2; resume=disable ;;
    resume) steps=3; resume=auto ;;
    *) echo "Usage: $0 [train|resume] [Hydra overrides...]" >&2; exit 2 ;;
esac
model_root=${OPD_MODEL_ROOT:-/data/work/YingZhong/models}
output=${OPD_SMOKE_OUTPUT:-$repo_root/checkpoints/opd_smoke_20261017}
if [[ "$phase" == train && -f "$output/latest_checkpointed_iteration.txt" ]]; then
    echo "Checkpoint already exists in $output; select a new OPD_SMOKE_OUTPUT or use resume." >&2
    exit 1
fi
if [[ "$phase" == resume && ! -f "$output/latest_checkpointed_iteration.txt" ]]; then
    echo "No checkpoint found in $output; run train first." >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
cd "$repo_root"
exec bash examples/opd_search/run_opd.sh \
    "actor_rollout_ref.model.path=$model_root/Qwen2.5-3B" \
    "distillation.teacher_models.teacher_model.model_path=$model_root/Qwen2.5-7B-Instruct" \
    data.train_max_samples=8 data.val_max_samples=2 data.seed=42 \
    data.train_batch_size=4 data.val_batch_size=2 data.dataloader_num_workers=0 \
    data.max_prompt_length=1024 data.max_response_length=2048 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    distillation.n_gpus_per_node=1 \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.4 \
    distillation.teacher_models.teacher_model.inference.max_model_len=4096 \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=4096 \
    distillation.teacher_models.teacher_model.inference.max_num_seqs=8 \
    trainer.n_gpus_per_node=2 reward.num_workers=2 ray_kwargs.ray_init.num_cpus=24 \
    trainer.total_epochs=3 "trainer.total_training_steps=$steps" \
    trainer.val_before_train=false \
    trainer.max_actor_ckpt_to_keep=2 "trainer.resume_mode=$resume" \
    "trainer.default_local_dir=$output" \
    "trainer.rollout_data_dir=$output/rollouts" \
    "trainer.validation_data_dir=$output/validation" \
    "trainer.experiment_name=${WANDB_NAME:-qwen25_real_model_smoke}" \
    "$@"
