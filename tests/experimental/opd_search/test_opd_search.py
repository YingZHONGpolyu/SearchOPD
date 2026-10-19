"""CPU unit tests; intentionally do not import the distributed verl runtime.

Load the actual torch kernel/protocol by file, and the registered integration
function from its AST. This permits contract testing in the existing Python 3.9
Search-R1 environment, but is NOT a substitute for a full latest-verl GPU run.
"""

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


kernel = load("opd_search_test_kernel", "verl/trainer/distillation/opd_search_loss.py")
protocol = load("opd_search_test_protocol", "verl/experimental/agent_loop/search_r1_protocol.py")
reward = load("opd_search_test_reward", "examples/opd_search/reward.py")


class LossTests(unittest.TestCase):
    def test_detached_weights_and_exact_gradient(self):
        s = torch.tensor([[-2.0, -1.0, -3.0]], requires_grad=True)
        t = torch.tensor([[-1.0, -2.0, -1.0]], requires_grad=True)
        mask = torch.tensor([[True, True, False]])
        losses, weights = kernel.clipped_teacher_cross_entropy(s, t, mask)
        expected = torch.tensor([[torch.e, 1 / torch.e, 0.0]])
        torch.testing.assert_close(weights, expected)
        (losses.sum() / mask.sum()).backward()
        torch.testing.assert_close(s.grad, -expected / 2)
        self.assertFalse(weights.requires_grad)
        self.assertIsNone(t.grad)

    def test_extreme_ratios_clip_before_exp(self):
        s = torch.tensor([[-10000.0, 0.0]], requires_grad=True)
        t = torch.tensor([[0.0, -10000.0]])
        losses, weights = kernel.clipped_teacher_cross_entropy(s, t, torch.ones_like(s, dtype=torch.bool))
        torch.testing.assert_close(weights, torch.tensor([[10.0, 1e-6]]))
        losses.sum().backward()
        self.assertTrue(torch.isfinite(s.grad).all())

    def test_observation_and_padding_nan_have_zero_gradient(self):
        s = torch.tensor([[-1.0, float("nan"), float("inf")]], requires_grad=True)
        t = torch.tensor([[-2.0, float("inf"), float("nan")]])
        losses, _ = kernel.clipped_teacher_cross_entropy(s, t, torch.tensor([[1, 0, 0]]))
        losses.sum().backward()
        torch.testing.assert_close(s.grad[0, 1:], torch.zeros(2))
        self.assertTrue(torch.isfinite(losses).all())

    def test_all_masked_microbatch_is_differentiable_zero(self):
        s = torch.tensor([[float("nan")]], requires_grad=True)
        loss, _ = kernel.clipped_teacher_cross_entropy(s, s.detach(), torch.zeros_like(s))
        loss.sum().backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(s.grad.item(), 0)

    def test_invalid_bounds_and_shapes(self):
        x = torch.zeros(2)
        for bounds in [(0, 10), (2, 1), (1e-6, float("inf"))]:
            with self.assertRaises(ValueError):
                kernel.clipped_teacher_cross_entropy(x, x, x, *bounds)
        with self.assertRaises(ValueError):
            kernel.clipped_teacher_cross_entropy(x, x[:1], x)

    def test_nonfinite_model_token_rejected(self):
        x = torch.tensor([float("nan")])
        with self.assertRaises(ValueError):
            kernel.clipped_teacher_cross_entropy(x, x, torch.ones(1))

    def test_reference_kl_is_zero_at_initial_student(self):
        s = torch.tensor([[-2.0, -1.0]], requires_grad=True)
        loss = kernel.reference_reverse_kl(s, s.detach(), torch.ones_like(s))
        loss.sum().backward()
        torch.testing.assert_close(loss, torch.zeros_like(loss))
        torch.testing.assert_close(s.grad, torch.zeros_like(s))

    def test_reference_gradient_and_detachment(self):
        s = torch.tensor([[-2.0, -1.0]], requires_grad=True)
        r = torch.tensor([[-1.0, 3.0]], requires_grad=True)
        loss = kernel.reference_reverse_kl(s, r, torch.tensor([[1, 0]]))
        loss.sum().backward()
        torch.testing.assert_close(s.grad, torch.tensor([[-1.0, 0]]))
        self.assertIsNone(r.grad)

    def test_reference_expected_gradient_matches_exact_full_vocabulary_reverse_kl(self):
        logits = torch.tensor([0.7, -0.4], requires_grad=True)
        logp = logits.log_softmax(0)
        logr = torch.tensor([0.4, 0.6]).log()
        # Enumerate every possible sampled action, weighted by its detached
        # sampling probability, exactly as an expectation over an OPD rollout.
        sampled = kernel.reference_reverse_kl(logp, logr, torch.ones(2))
        estimated = (logp.detach().exp() * sampled).sum()
        expected = (logp.exp() * (logp - logr)).sum()
        actual_gradient = torch.autograd.grad(estimated, logits, retain_graph=True)[0]
        exact_gradient = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(estimated, expected)
        torch.testing.assert_close(actual_gradient, exact_gradient)

    def test_registered_loss_retains_reference_kl_without_task_rewards(self):
        # Execute the actual registry entry in isolation, mocking only verl's
        # full-sequence padding adapter (not the mathematical implementation).
        source = (ROOT / "verl/trainer/distillation/losses.py").read_text()
        # The upstream file uses Python >=3.10 match/case. Parse only this
        # production function so these CPU contracts remain runnable on 3.9.
        source = source[source.index("\ndef compute_opd_search_forward_kl(") + 1 :]
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_opd_search_forward_kl")
        fn.decorator_list = []
        ns = {
            "clipped_teacher_cross_entropy": kernel.clipped_teacher_cross_entropy,
            "reference_reverse_kl": kernel.reference_reverse_kl,
            "no_padding_2_padding": lambda value, data: value,
            "Metric": lambda aggregation, value: value,
            "AggregationType": SimpleNamespace(SUM="sum"),
            "agg_loss": lambda values, mask, mode, **kwargs: (values * mask).sum() / mask.sum().clamp_min(1),
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "registry_entry", "exec"), ns)
        cfg = SimpleNamespace(use_kl_loss=True, kl_loss_type="low_var_kl", kl_loss_coef=0.1)
        loss_cfg = SimpleNamespace(importance_ratio_min=1e-6, importance_ratio_max=10.0, distillation_loss_coef=2.0)
        s = torch.tensor([[-2.0, -1.0]], requires_grad=True)
        t = torch.tensor([[[-1.0], [-2.0]]])
        r = torch.tensor([[-1.5, -1.2]])
        mask = torch.tensor([[1, 0]])
        result, metrics = ns[fn.name](
            cfg,
            SimpleNamespace(distillation_loss=loss_cfg),
            {"log_probs": s},
            {"teacher_logprobs": t, "ref_log_prob": r, "response_mask": mask},
        )
        ce, _ = kernel.clipped_teacher_cross_entropy(s, t.squeeze(-1), mask)
        expected = 2 * ce + 0.1 * kernel.reference_reverse_kl(s, r, mask)
        torch.testing.assert_close(result, expected)
        self.assertIn("distillation/reference_kl", metrics)
        result.sum().backward()
        self.assertEqual(s.grad[0, 1].item(), 0)


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def run_scripted(self, texts, settings=None, budget=4096, passages="REAL_EVIDENCE", versions=None):
        tokenizer = CharacterTokenizer()
        prompts, queries = [], []
        queue = list(texts)

        async def generate(ids, params):
            prompts.append(list(ids))
            self.assertTrue(params["include_stop_str_in_output"])
            text = queue.pop(0) if queue else "\0"
            tokens = tokenizer.encode(text)[: params["max_tokens"]]
            version = versions[len(prompts) - 1] if versions is not None else (0, 0)
            return SimpleNamespace(
                token_ids=tokens,
                log_probs=[-0.5] * len(tokens),
                extra_fields={"min_global_steps": version[0], "max_global_steps": version[1]},
            )

        async def search(query, settings):
            queries.append(query)
            return passages

        output = await protocol.run_search_rollout(
            tokenizer,
            tokenizer.encode("QUESTION:"),
            budget,
            {"temperature": 1.0},
            generate,
            search,
            settings or protocol.SearchSettings(),
        )
        return output, prompts, queries

    async def test_live_feedback_in_next_prefix_and_masked(self):
        first = "<think>Need evidence</think><search>capital France</search>"
        final = "<think>Use evidence</think><answer>Paris</answer>"
        result, prompts, queries = await self.run_scripted([first, final])
        text = CharacterTokenizer().decode(result["response_ids"])
        self.assertEqual(queries, ["capital France"])
        self.assertIn("REAL_EVIDENCE", CharacterTokenizer().decode(prompts[1]))
        a = text.index("\n<information>")
        b = text.index("\n</information>\n") + len("\n</information>\n")
        self.assertTrue(all(v == 0 for v in result["response_mask"][a:b]))
        self.assertEqual(result["response_ids"][: len(first)], CharacterTokenizer().encode(first))
        self.assertEqual(sum(result["response_mask"]), len(first) + len(final))
        self.assertEqual(len(result["response_ids"]), len(result["response_logprobs"]))

    async def test_four_search_turn_limit(self):
        result, _, queries = await self.run_scripted(["<search>q</search>"] * 5)
        self.assertEqual(len(queries), 4)
        self.assertEqual(result["extra_fields"]["search_turns"], 4)

    async def test_direct_answer_no_retrieval(self):
        result, _, queries = await self.run_scripted(["<answer>Paris</answer>"])
        self.assertEqual(queries, [])
        self.assertTrue(all(result["response_mask"]))

    async def test_preserves_version_zero_for_native_v1_metrics(self):
        result, _, _ = await self.run_scripted(["<answer>a</answer>"])
        self.assertEqual(result["extra_fields"]["min_global_steps"], 0)
        self.assertEqual(result["extra_fields"]["max_global_steps"], 0)

    async def test_merges_model_version_intervals_across_calls(self):
        result, _, _ = await self.run_scripted(["<search>q</search>", "<answer>a</answer>"], versions=[(2, 3), (1, 4)])
        self.assertEqual(result["extra_fields"]["min_global_steps"], 1)
        self.assertEqual(result["extra_fields"]["max_global_steps"], 4)

    async def test_missing_model_version_is_rejected(self):
        async def generate(ids, params):
            return SimpleNamespace(token_ids=[65], log_probs=[-1.0], extra_fields={})

        with self.assertRaisesRegex(ValueError, "min_global_steps"):
            await protocol.run_search_rollout(
                CharacterTokenizer(), [1], 512, {}, generate, None, protocol.SearchSettings()
            )

    async def test_partial_search_tag_across_generation_calls(self):
        _, _, queries = await self.run_scripted(["<think>x</think><sea", "rch>q</search>", "<answer>a</answer>"])
        self.assertEqual(queries, ["q"])

    async def test_observation_truncation_retains_closing_delimiter(self):
        result, _, _ = await self.run_scripted(
            ["<search>q</search>", "<answer>a</answer>"],
            settings=protocol.SearchSettings(max_observation_tokens=5),
            passages="X" * 100,
        )
        text = CharacterTokenizer().decode(result["response_ids"])
        self.assertIn("\n<information>\nXXXXX\n</information>\n", text)
        self.assertNotIn("XXXXXX", text)

    async def test_total_response_budget_includes_environment(self):
        result, _, _ = await self.run_scripted(["<search>q</search>", "A" * 100], budget=70, passages="X" * 500)
        self.assertLessEqual(len(result["response_ids"]), 70)
        self.assertEqual(len(result["response_mask"]), len(result["response_ids"]))

    async def test_retrieval_failure_is_not_fabricated_feedback(self):
        async def generate(ids, params):
            tokens = CharacterTokenizer().encode("<search>q</search>")
            return SimpleNamespace(
                token_ids=tokens,
                log_probs=[-1] * len(tokens),
                extra_fields={"min_global_steps": 0, "max_global_steps": 0},
            )

        async def fail(query, settings):
            raise ConnectionError("unavailable")

        with self.assertRaises(ConnectionError):
            await protocol.run_search_rollout(
                CharacterTokenizer(), [1], 512, {}, generate, fail, protocol.SearchSettings()
            )

    async def test_real_http_retriever_contract(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps({"result": [[{"document": {"contents": "Paris evidence"}, "score": 0.8}]]}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            settings = protocol.SearchSettings(url=f"http://127.0.0.1:{server.server_port}/retrieve")
            result = await protocol.retrieve("France", settings)
            self.assertIn("Paris evidence", result)
            self.assertEqual(received[0], {"queries": ["France"], "topk": 3, "return_scores": True})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class DataAndRewardTests(unittest.TestCase):
    def test_retriever_schema_validation(self):
        self.assertIn("none", protocol.format_passages({"result": [[{"contents": "none"}]]}, 3))
        with self.assertRaises(ValueError):
            protocol.format_passages({"result": [[], []]}, 3)
        with self.assertRaises(ValueError):
            protocol.format_passages({"result": [[{"text": "wrong field"}]]}, 3)
        for payload in ([], {"result": [[None]]}, {"result": [["wrong type"]]}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                protocol.format_passages(payload, 3)

    def test_em_normalization_and_missing_answer(self):
        self.assertEqual(reward.compute_score("nq", "<answer>The Paris.</answer>", {"target": ["Paris"]})["acc"], 1)
        self.assertEqual(reward.compute_score("nq", "Paris", {"target": ["Paris"]})["acc"], 0)

    def test_retrieved_answer_tag_cannot_earn_reward(self):
        text = "<information><answer>Paris</answer></information><answer>London</answer>"
        self.assertEqual(reward.compute_score("nq", text, {"target": ["Paris"]})["acc"], 0)

    def test_streaming_data_conversion_preserves_prompt_and_targets(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        module = load("opd_search_test_prepare", "examples/opd_search/prepare_data.py")
        rows = [
            {
                "data_source": "nq",
                "prompt": [{"role": "user", "content": "Question?"}],
                "reward_model": {"ground_truth": {"target": ["Paris", "City of Paris"]}},
            }
        ]
        with tempfile.TemporaryDirectory() as folder:
            source, dest = Path(folder) / "in.parquet", Path(folder) / "out.parquet"
            pq.write_table(pa.Table.from_pylist(rows), source)
            self.assertEqual(module.convert(source, dest, "train"), 1)
            result = pq.read_table(dest).to_pylist()[0]
            self.assertEqual(result["prompt"], rows[0]["prompt"])
            self.assertEqual(result["agent_name"], "opd_search_agent")
            self.assertEqual(result["reward_model"]["ground_truth"]["target"], ["Paris", "City of Paris"])

    def test_hydra_config_and_on_policy_invariants(self):
        module = load("opd_search_test_preflight", "examples/opd_search/preflight.py")
        config = module.compose_config()
        self.assertEqual(config.actor_rollout_ref.model.path, "Qwen/Qwen2.5-3B")
        self.assertEqual(config.distillation.teacher_models.teacher_model.model_path, "Qwen/Qwen2.5-7B-Instruct")
        with self.assertRaisesRegex(ValueError, "ppo_epochs"):
            module.compose_config(["actor_rollout_ref.actor.ppo_epochs=2"])

    def test_off_policy_overrides_rejected(self):
        module = load("opd_search_test_preflight_modes", "examples/opd_search/preflight.py")
        for override in (
            "trainer.v1.trainer_mode=colocate_async",
            "trainer.v1.trainer_mode=separate_async",
            "algorithm.filter_groups.enable=true",
            "distillation.enabled=false",
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                module.compose_config([override])

    def test_preflight_contracts_survive_python_optimization(self):
        process = subprocess.run(
            [
                sys.executable,
                "-O",
                str(ROOT / "examples/opd_search/preflight.py"),
                "--config-only",
                "actor_rollout_ref.actor.ppo_epochs=2",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("ppo_epochs must be 1", process.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
