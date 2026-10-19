"""Latest verl AgentLoop adapter for the OPDSearch+ live-search protocol."""

from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.search_r1_protocol import SearchSettings, retrieve, run_search_rollout


@register("opd_search_agent")
class OPDSearchAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, priority=0, **kwargs):
        settings = SearchSettings(**dict(self.config.opd_search))
        # Base student: plain Search-R1 prompt, with no instruct chat wrapper.
        # The dataset uses the identical custom_chat_template for length checks.
        messages = list(kwargs["raw_prompt"])
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError("OPDSearch expects one plain user prompt per question")
        prompt_ids = self.tokenizer.encode(messages[0]["content"], add_special_tokens=False)
        if len(prompt_ids) > self.rollout_config.prompt_length:
            raise ValueError("Prompt exceeds configured prompt_length")

        async def generate(ids, params):
            return await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=ids,
                sampling_params=params,
                priority=int(priority),
            )

        output = await run_search_rollout(
            self.tokenizer,
            prompt_ids,
            self.rollout_config.response_length,
            sampling_params,
            generate,
            retrieve,
            settings,
        )
        versions = output["extra_fields"]
        if versions["min_global_steps"] != versions["max_global_steps"]:
            raise RuntimeError("OPDSearch requires one frozen student version per trajectory; use synchronous rollout")
        # Teacher scoring is automatically performed by AgentLoopWorker on the
        # complete token sequence after this returns. Do not reapply a teacher
        # chat template or remove observations from its conditioning prefix.
        return AgentLoopOutput(**output)
