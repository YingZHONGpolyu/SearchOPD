"""Configuration audit and runtime readiness checks; no training is started."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    # Configuration contracts must still hold under python -O/PYTHONOPTIMIZE.
    if not condition:
        raise ValueError(f"Invalid OPD configuration: {message}")


def compose_config(overrides=()):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    os.environ["OPD_REPO_ROOT"] = str(ROOT)
    with initialize_config_dir(config_dir=str(ROOT / "examples/opd_search/config"), version_base=None):
        config = compose(config_name="opd_search", overrides=list(overrides))
    OmegaConf.resolve(config)
    loss = config.distillation.distillation_loss
    actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    require(config.distillation.enabled, "distillation.enabled must be true")
    require(config.trainer.use_v1 and config.trainer.v1.trainer_mode == "sync", "use the synchronous V1 trainer")
    require(not config.algorithm.filter_groups.enable, "reward-based group filtering biases on-policy OPD sampling")
    require(loss.loss_mode == "opd_search_forward_kl", "use the OPDSearch loss")
    require(not loss.use_task_rewards and not loss.use_policy_gradient, "disable task rewards and policy gradient")
    require(loss.loss_max_clamp is None and loss.log_prob_min_clamp is None, "disable extra loss/logprob clamps")
    require(actor.ppo_epochs == 1, "ppo_epochs must be 1")
    require(actor.ppo_mini_batch_size == config.data.train_batch_size, "update each rollout batch only once")
    require(
        rollout.temperature == 1.0 and rollout.top_p == 1.0 and rollout.top_k == -1,
        "sample the unwarped student distribution",
    )
    require(rollout.calculate_log_probs, "enable rollout log probabilities")
    require(
        actor.use_kl_loss and actor.kl_loss_type in ("k3", "low_var_kl"),
        "use the initial-student k3 reference regularizer",
    )
    require(actor.entropy_coeff == 0, "entropy_coeff must be zero")
    require(actor.loss_agg_mode == "token-mean", "use global token-mean loss aggregation")
    require(
        config.opd_search.topk == 3 and config.opd_search.max_search_turns == 4,
        "use top-3 retrieval and four search turns",
    )
    require(
        config.actor_rollout_ref.model.custom_chat_template == "{{ messages[0]['content'] }}",
        "use the plain Base-student prompt",
    )
    require(
        rollout.max_model_len >= config.data.max_prompt_length + config.data.max_response_length,
        "student context is shorter than the prompt/response budget",
    )
    teacher = config.distillation.teacher_models.teacher_model.inference
    require(
        teacher.max_model_len >= config.data.max_prompt_length + config.data.max_response_length + 1,
        "teacher context needs the full trajectory plus one scoring token",
    )
    return config


def runtime_errors():
    errors = []
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        errors.append(
            f"Python {sys.version.split()[0]} is outside latest verl's >=3.10,<3.13 range; use uv Python 3.12"
        )
    for package, prefix in [("torch", "2.13."), ("vllm", "0.29."), ("transformers", "5.12.")]:
        try:
            version = importlib.metadata.version(package)
            print(f"{package}: {version}")
            if not version.startswith(prefix):
                errors.append(f"{package} {version} differs from the checked-in uv.lock backend")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{package} is not installed")
    # Check in a fresh interpreter: importing torch first can hide a mixed
    # system/environment glibc failure in Arrow by preloading the correct librt.
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import pyarrow; print(pyarrow.__version__)"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if probe.returncode:
            errors.append(f"PyArrow fails in a clean Python process: {probe.stderr.strip()[-1500:]}")
        else:
            print("PyArrow clean import:", probe.stdout.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(f"Cannot verify PyArrow import: {exc}")
    try:
        drivers = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).splitlines()
        print("NVIDIA driver:", drivers[0])
        if int(drivers[0].split(".")[0]) < 580:
            # NVIDIA supports environment-local CUDA forward compatibility on
            # data-center GPUs. Verify the loaded driver API and CUDA context
            # instead of rejecting a working compatibility installation.
            try:
                import ctypes

                import torch

                library = ctypes.CDLL("libcuda.so.1")
                api_version = ctypes.c_int()
                status = library.cuDriverGetVersion(ctypes.byref(api_version))
                if status != 0 or api_version.value < 13000:
                    raise RuntimeError("loaded CUDA driver API is older than 13.0")
                torch.cuda.init()
                print("CUDA forward compatibility: PASS; driver API", api_version.value)
            except Exception as exc:
                errors.append(f"CUDA 13 needs a newer driver or working forward-compatibility libraries: {exc}")
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        errors.append("Cannot detect NVIDIA driver")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--check-models", action="store_true", help="Download tokenizer/config metadata, not weights")
    parser.add_argument("--check-retriever", action="store_true")
    args, overrides = parser.parse_known_args()
    if args.runtime_only and (args.check_models or args.check_retriever):
        parser.error("--runtime-only cannot be combined with model/retriever checks")
    config = None if args.runtime_only else compose_config(overrides)
    if config is not None:
        print("Hydra composition and OPD invariants: PASS")
    if args.config_only:
        return
    errors = runtime_errors()
    if config is not None:
        for p in (config.data.train_files, config.data.val_files):
            if not Path(p).is_file():
                errors.append(f"Prepared dataset missing: {p}")
    if errors:
        print("NOT READY:\n- " + "\n- ".join(errors))
        raise SystemExit(1)
    if args.check_models:
        from transformers import AutoConfig, AutoTokenizer

        student = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
        teacher_path = config.distillation.teacher_models.teacher_model.model_path
        teacher = AutoTokenizer.from_pretrained(teacher_path)
        if student.get_vocab() != teacher.get_vocab():
            errors.append("Student and teacher token IDs differ; sampled-token OPD needs shared tokenization")
        for path in (config.actor_rollout_ref.model.path, teacher_path):
            model_config = AutoConfig.from_pretrained(path)
            if max(student.get_vocab().values()) >= model_config.vocab_size:
                errors.append(f"Tokenizer ID outside model vocabulary: {path}")
    if args.check_retriever:
        try:
            from verl.experimental.agent_loop.search_r1_protocol import format_passages

            body = json.dumps(
                {"queries": ["What is the capital of France?"], "topk": 3, "return_scores": True}
            ).encode()
            req = urllib.request.Request(config.opd_search.url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=config.opd_search.timeout) as response:
                result = json.load(response)
            # Exercise the exact schema parser used during real rollouts.
            format_passages(result, config.opd_search.topk)
        except Exception as exc:
            errors.append(f"Retriever unavailable: {exc}")
    if errors:
        print("NOT READY:\n- " + "\n- ".join(errors))
        raise SystemExit(1)
    print("Runtime preflight: PASS")


if __name__ == "__main__":
    main()
