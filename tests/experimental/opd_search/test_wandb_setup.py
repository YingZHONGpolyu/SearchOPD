"""Exercise run identity across launches without a W&B network connection."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

EXAMPLE = Path(__file__).resolve().parents[3] / "examples/opd_search"
sys.path.insert(0, str(EXAMPLE))
spec = importlib.util.spec_from_file_location("opd_wandb_setup", EXAMPLE / "wandb_setup.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class WandbSetupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "experiment"
        self.config = SimpleNamespace(
            trainer=SimpleNamespace(
                logger=["console", "wandb"],
                project_name="opd_search",
                default_local_dir=str(self.output),
                resume_mode="auto",
                resume_from_path=None,
            )
        )

    def test_online_resume_reuses_identity(self):
        first = setup.prepare_wandb(self.config, {})
        (self.output / "latest_checkpointed_iteration.txt").write_text("2")
        resumed = setup.prepare_wandb(self.config, {})
        self.assertEqual(first["WANDB_RUN_ID"], resumed["WANDB_RUN_ID"])
        self.assertEqual(resumed["WANDB_RESUME"], "allow")
        self.assertTrue(Path(first["WANDB_DIR"]).is_dir())

    def test_independent_output_gets_independent_run(self):
        first = setup.prepare_wandb(self.config, {})
        self.config.trainer.default_local_dir += "_second"
        second = setup.prepare_wandb(self.config, {})
        self.assertNotEqual(first["WANDB_RUN_ID"], second["WANDB_RUN_ID"])

    def test_changed_project_and_run_id_are_rejected(self):
        setup.prepare_wandb(self.config, {})
        with self.assertRaisesRegex(ValueError, "WANDB_RUN_ID differs"):
            setup.prepare_wandb(self.config, {"WANDB_RUN_ID": "another"})
        self.config.trainer.project_name = "another_project"
        with self.assertRaisesRegex(ValueError, "project/entity/base URL differs"):
            setup.prepare_wandb(self.config, {})

    def test_fresh_training_cannot_overwrite_identity(self):
        setup.prepare_wandb(self.config, {})
        self.config.trainer.resume_mode = "disable"
        with self.assertRaisesRegex(ValueError, "already exists"):
            setup.prepare_wandb(self.config, {})

    def test_offline_segments_have_distinct_ids_and_shared_group(self):
        first = setup.prepare_wandb(self.config, {"WANDB_MODE": "offline"})
        second = setup.prepare_wandb(self.config, {"WANDB_MODE": "offline"})
        self.assertNotEqual(first["WANDB_RUN_ID"], second["WANDB_RUN_ID"])
        self.assertEqual(first["WANDB_RUN_GROUP"], second["WANDB_RUN_GROUP"])
        self.assertEqual(second["WANDB_RESUME"], "never")

    def test_disabled_or_console_logging_creates_no_run_state(self):
        self.assertEqual(setup.prepare_wandb(self.config, {"WANDB_MODE": "disabled"}), {"WANDB_MODE": "disabled"})
        self.config.trainer.logger = ["console"]
        self.assertEqual(setup.prepare_wandb(self.config, {}), {})
        self.assertFalse(self.output.exists())

    def test_resume_path_imports_source_identity(self):
        original = setup.prepare_wandb(self.config, {})
        self.config.trainer.resume_mode = "resume_path"
        self.config.trainer.resume_from_path = str(self.output / "global_step_2")
        self.config.trainer.default_local_dir += "_resumed"
        resumed = setup.prepare_wandb(self.config, {})
        self.assertEqual(original["WANDB_RUN_ID"], resumed["WANDB_RUN_ID"])
        state = json.loads((Path(self.config.trainer.default_local_dir) / "wandb_run.json").read_text())
        self.assertEqual(state["run_id"], original["WANDB_RUN_ID"])

    def test_credentials_are_not_written_to_experiment_state(self):
        setup.prepare_wandb(self.config, {"WANDB_API_KEY": "test_secret_never_save"})
        self.assertNotIn("test_secret_never_save", (self.output / "wandb_run.json").read_text())


if __name__ == "__main__":
    unittest.main()
