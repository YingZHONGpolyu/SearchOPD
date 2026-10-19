"""Prepare native verl W&B logging and persist experiment identity beside checkpoints."""

import argparse
import importlib.metadata
import json
import os
import re
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

from preflight import compose_config


def prepare_wandb(config, environ=None):
    env = os.environ if environ is None else environ
    backends = config.trainer.logger
    if "wandb" not in ([backends] if isinstance(backends, str) else backends):
        return {}
    importlib.metadata.version("wandb")
    mode = env.get("WANDB_MODE", "online")
    if mode not in ("online", "offline", "disabled"):
        raise ValueError("Use WANDB_MODE=online, offline or disabled")
    if mode == "disabled":
        return {"WANDB_MODE": mode}
    output = Path(config.trainer.default_local_dir).expanduser().resolve()
    state_path = output / "wandb_run.json"
    identity = {
        "project": str(config.trainer.project_name),
        "entity": env.get("WANDB_ENTITY") or None,
        "base_url": env.get("WANDB_BASE_URL", "https://api.wandb.ai").rstrip("/"),
    }
    requested_id = env.get("WANDB_RUN_ID") or None
    # Explicit resume paths must carry the source run identity as well.
    source_state = state_path
    if not source_state.exists() and config.trainer.resume_mode == "resume_path":
        checkpoint = Path(config.trainer.resume_from_path).expanduser().resolve()
        source_state = checkpoint.parent / "wandb_run.json"
        if not source_state.exists() and requested_id is None:
            raise ValueError("Source checkpoint has no wandb_run.json; supply its WANDB_RUN_ID explicitly")
    state = json.loads(source_state.read_text()) if source_state.exists() else None
    if state is not None:
        if config.trainer.resume_mode == "disable":
            raise ValueError("W&B experiment already exists; choose a new trainer.default_local_dir")
        if any(state.get(key) != value for key, value in identity.items()):
            raise ValueError("W&B project/entity/base URL differs from the saved experiment identity")
        if requested_id is not None and requested_id != state["run_id"]:
            raise ValueError("WANDB_RUN_ID differs from the saved experiment identity")
        run_id = state["run_id"]
    else:
        run_id = requested_id or uuid.uuid4().hex[:12]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("WANDB_RUN_ID must contain only letters, digits, underscores or hyphens")
    output.mkdir(parents=True, exist_ok=True)
    if not state_path.exists():
        # Exclusive creation prevents concurrent launchers from replacing identity.
        with state_path.open("x") as stream:
            json.dump({"run_id": run_id, **identity}, stream, indent=2)
            stream.write("\n")
    log_dir = Path(env.get("WANDB_DIR", str(output / "wandb_logs"))).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    settings = {"WANDB_MODE": mode, "WANDB_DIR": str(log_dir), "WANDB_RUN_ID": run_id}
    if mode == "online":
        # allow also supports attaching W&B to checkpoints created before logging
        # was enabled. A newly created run then starts at the resumed trainer step.
        settings["WANDB_RESUME"] = "allow"
    else:
        # The SDK cannot resume offline runs. Keep each segment separate and
        # associate it with the stable experiment through W&B grouping.
        settings["WANDB_RUN_ID"] = uuid.uuid4().hex[:12]
        settings["WANDB_RUN_GROUP"] = env.get("WANDB_RUN_GROUP") or run_id
        settings["WANDB_RESUME"] = "never"
    return settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    config = compose_config(overrides)
    if "wandb" in config.trainer.logger and os.environ.get("WANDB_MODE", "online") == "online":
        try:
            # Bound SDK retries as well as individual HTTP request timeouts.
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import wandb; raise SystemExit(0 if wandb.login(verify=True, timeout=1) else 1)",
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=20,
            )
            failed = probe.returncode != 0
        except (OSError, subprocess.TimeoutExpired):
            failed = True
        if failed:
            raise SystemExit(
                "W&B online login/network check failed. "
                "Run 'conda activate opd_env; wandb login --relogin --verify', "
                "or explicitly set WANDB_MODE=offline to record locally."
            )
    settings = prepare_wandb(config)
    args.env_file.write_text("".join(f"export {key}={shlex.quote(value)}\n" for key, value in settings.items()))
    if settings:
        print(f"W&B: mode={settings['WANDB_MODE']}, logs={settings.get('WANDB_DIR', 'disabled')}")


if __name__ == "__main__":
    main()
