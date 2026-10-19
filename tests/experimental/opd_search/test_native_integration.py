"""Native verl contract tests, run in opd_env (no model weights or Ray cluster).

Unlike the portable CPU suite, these import the registered loss, real jagged
padding, AgentLoopOutput, vLLM prompt-score parser and V1 TransferQueue adapter.
Only model generation, retrieval and the queue transport are stubbed.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import torch
from tensordict import TensorDict

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.opd_search_agent_loop import OPDSearchAgentLoop
from verl.trainer.distillation.losses import distillation_ppo_loss
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ
from verl.utils import tensordict_utils as tu
from verl.utils.metric import Metric
from verl.utils.model import compute_position_id_with_mask
from verl.workers.config import DistillationLossConfig
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs


def jagged(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def loss_call(rows, dp_size=1, global_tokens=4):
    """Rows contain full-sequence student/teacher scores, response-only ref/mask."""
    student_values = torch.cat([row[0] for row in rows]).detach().requires_grad_(True)
    lengths = torch.tensor([len(row[0]) for row in rows])
    offsets = torch.cat([torch.tensor([0]), lengths.cumsum(0)])
    student = torch.nested.nested_tensor_from_jagged(student_values, offsets)
    prompts = [torch.arange(len(row[0]) - len(row[3])) for row in rows]
    responses = [torch.arange(len(row[3])) for row in rows]
    data = TensorDict(
        {
            "prompts": jagged(prompts),
            "responses": jagged(responses),
            "response_mask": jagged([row[3] for row in rows]),
            "teacher_logprobs": jagged([row[1].unsqueeze(-1) for row in rows]),
            "ref_log_prob": jagged([row[2] for row in rows]),
        },
        batch_size=[len(rows)],
    )
    tu.assign_non_tensor(data, dp_size=dp_size, batch_num_tokens=global_tokens, global_batch_size=2)
    actor = SimpleNamespace(
        use_kl_loss=True,
        kl_loss_type="low_var_kl",
        kl_loss_coef=0.1,
        loss_agg_mode="token-mean",
        loss_scale_factor=None,
        global_batch_info={},
    )
    distillation = SimpleNamespace(
        distillation_loss=DistillationLossConfig(
            loss_mode="opd_search_forward_kl",
            use_task_rewards=False,
            use_policy_gradient=False,
            loss_max_clamp=None,
            log_prob_min_clamp=None,
            distillation_loss_coef=2.0,
        )
    )
    loss, metrics = distillation_ppo_loss(actor, distillation, {"log_probs": student}, data)
    return loss, metrics, student_values


class NativeLossTests(unittest.TestCase):
    def setUp(self):
        # Prompt lengths differ; the first response also contains an observation.
        self.rows = [
            (
                torch.tensor([-2.0, -3.0, -4.0, -5.0]),
                torch.tensor([-8.0, 0.0, -7.0, 0.0]),
                torch.tensor([-2.8, 10.0]),
                torch.tensor([1, 0]),
            ),
            (
                torch.tensor([-6.0, -7.0, -8.0, -9.0, -10.0, -11.0]),
                torch.tensor([-6.0, -7.0, -100.0, -9.2, -9.6, 0.0]),
                torch.tensor([-7.7, -9.1, -9.6]),
                torch.tensor([1, 1, 1]),
            ),
        ]

    def test_registered_loss_alignment_reference_and_exact_gradient(self):
        loss, _, values = loss_call(self.rows)
        student = torch.tensor([-3.0, -8.0, -9.0, -10.0])
        teacher = torch.tensor([0.0, -100.0, -9.2, -9.6])
        reference = torch.tensor([-2.8, -7.7, -9.1, -9.6])
        weights = (teacher - student).exp().clamp(1e-6, 10)
        delta = reference - student
        expected_loss = (-2 * weights * student + 0.1 * (delta.expm1() - delta)).sum() / 4
        torch.testing.assert_close(loss, expected_loss)
        loss.backward()
        expected_grad = torch.zeros_like(values)
        expected_grad[[1, 6, 7, 8]] = (-2 * weights - 0.1 * delta) / 4
        torch.testing.assert_close(values.grad, expected_grad)

    def test_metrics_invariant_to_microbatch_and_dp_partition(self):
        whole_loss, whole_metrics, _ = loss_call(self.rows)
        micro_parts = [loss_call([row]) for row in self.rows]
        torch.testing.assert_close(micro_parts[0][0] + micro_parts[1][0], whole_loss)
        parts = [loss_call([row], dp_size=2) for row in self.rows]
        torch.testing.assert_close((parts[0][0] + parts[1][0]) / 2, whole_loss)
        keys = ("opd_ce", "importance_mean", "importance_clip_fraction", "reference_kl")
        for key in keys:
            key = "distillation/" + key
            with self.subTest(metric=key):
                accumulated = whole_metrics[key].init_list()
                for part in micro_parts:
                    accumulated.extend(part[1][key])
                self.assertAlmostEqual(accumulated.aggregate(), whole_metrics[key].aggregate(), places=5)
                self.assertAlmostEqual(
                    Metric.aggregate_dp([part[1][key] for part in parts]), whole_metrics[key].aggregate(), places=5
                )
        # One upper-clipped token and one lower-clipped token, out of four.
        self.assertAlmostEqual(whole_metrics["distillation/importance_clip_fraction"].aggregate(), 0.5)

    def test_native_vllm_prompt_scores_shift_exactly_once(self):
        output = SimpleNamespace(
            prompt_logprobs=[
                None,
                {12: SimpleNamespace(logprob=-0.1)},
                {13: SimpleNamespace(logprob=-0.2)},
                {14: SimpleNamespace(logprob=-0.3)},
            ]
        )
        result = {}
        extract_prompt_logprobs(output, 0, result)
        self.assertEqual(result["prompt_ids"], [[12], [13], [14], [0]])
        torch.testing.assert_close(
            torch.tensor(result["prompt_logprobs"]), torch.tensor([[-0.1], [-0.2], [-0.3], [0.0]])
        )

    def test_all_masked_microbatch_has_zero_loss_metrics_and_gradient(self):
        row = (
            torch.full((4,), float("nan")),
            torch.full((4,), float("nan")),
            torch.full((2,), float("nan")),
            torch.zeros(2, dtype=torch.long),
        )
        loss, metrics, values = loss_call([row], global_tokens=4)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.zeros_like(values))
        for key in ("opd_ce", "importance_mean", "importance_clip_fraction", "reference_kl"):
            self.assertEqual(metrics["distillation/" + key].aggregate(), 0)


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class NativeAgentTests(unittest.IsolatedAsyncioTestCase):
    async def run_agent(self, versions):
        # Skip base-class token-builder initialization; exercise the actual run
        # method and all native output validation/serialization below.
        agent = OPDSearchAgentLoop.__new__(OPDSearchAgentLoop)
        agent.config = SimpleNamespace(opd_search={})
        agent.rollout_config = SimpleNamespace(prompt_length=512, response_length=1024)
        agent.tokenizer = CharacterTokenizer()
        texts = iter(["<search>q</search>", "<answer>Paris</answer>"])
        versions = iter(versions)

        async def generate(**kwargs):
            ids = agent.tokenizer.encode(next(texts))
            version = next(versions)
            return TokenOutput(
                token_ids=ids,
                log_probs=[-1.0] * len(ids),
                extra_fields={"min_global_steps": version, "max_global_steps": version},
            )

        agent.server_manager = SimpleNamespace(generate=AsyncMock(side_effect=generate))
        with patch("verl.experimental.agent_loop.opd_search_agent_loop.retrieve", AsyncMock(return_value="EVIDENCE")):
            output = await agent.run({"logprobs": True}, raw_prompt=[{"role": "user", "content": "Question?"}])
        return output, agent.server_manager

    async def test_native_output_keeps_version_zero_and_live_observation(self):
        output, manager = await self.run_agent([0, 0])
        self.assertIsInstance(output, AgentLoopOutput)
        self.assertIn("EVIDENCE", CharacterTokenizer().decode(manager.generate.call_args_list[1].kwargs["prompt_ids"]))
        data = output.as_dict()
        self.assertEqual(data["extra_fields"]["min_global_steps"], 0)
        self.assertEqual(data["extra_fields"]["max_global_steps"], 0)
        self.assertTrue((data["response_mask"] == 0).any())

    async def test_mixed_student_versions_fail_before_teacher_scoring(self):
        with self.assertRaisesRegex(RuntimeError, "one frozen student version"):
            await self.run_agent([4, 5])

    async def test_actual_v1_postprocessor_writes_teacher_fields_and_numeric_tags(self):
        output, _ = await self.run_agent([0, 0])
        output.reward_score = 1.0

        async def score_teacher(result, **kwargs):
            count = len(kwargs["prompt_ids"]) + len(kwargs["response_ids"])
            result.extra_fields.update(
                teacher_ids=torch.zeros(count, 1, dtype=torch.int32), teacher_logprobs=torch.full((count, 1), -1.0)
            )

        worker = SimpleNamespace(
            _compute_score=AsyncMock(),
            _compute_teacher_logprobs=AsyncMock(side_effect=score_teacher),
            _compute_multi_modal_inputs=lambda *args: None,
            _compute_position_ids=lambda ids, mask, mm: compute_position_id_with_mask(mask),
        )
        native_class = AgentLoopWorkerTQ.__ray_metadata__.modified_class
        with patch("verl.trainer.ppo.v1.agent_loop_tq.tq.async_kv_batch_put", new_callable=AsyncMock) as put:
            await native_class._agent_loop_postprocess(
                worker,
                output,
                validate=False,
                uid="review",
                session_id=0,
                global_steps=1,
            )
        call = put.call_args.kwargs
        self.assertEqual(call["partition_id"], "train")
        self.assertEqual(np.array([tag["min_global_steps"] for tag in call["tags"]], dtype=int).tolist(), [0])
        self.assertEqual(np.array([tag["max_global_steps"] for tag in call["tags"]], dtype=int).tolist(), [0])
        self.assertIn("teacher_logprobs", call["fields"])
        self.assertIn("loss_mask", call["fields"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
