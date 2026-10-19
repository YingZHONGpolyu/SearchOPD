"""Search-R1 text protocol and rollout state machine, independent of Ray/vLLM."""

import asyncio
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class SearchSettings:
    url: str = "http://127.0.0.1:8000/retrieve"
    topk: int = 3
    max_search_turns: int = 4
    max_generation_tokens: int = 512
    max_observation_tokens: int = 512
    timeout: float = 60.0
    retries: int = 2

    def __post_init__(self):
        if min(self.topk, self.max_generation_tokens, self.max_observation_tokens) < 1:
            raise ValueError("Search and token limits must be positive")
        if self.max_search_turns < 0 or self.retries < 0 or self.timeout <= 0:
            raise ValueError("Invalid search turns, retry count, or timeout")


def format_passages(payload, topk):
    """Accept the existing Search-R1 /retrieve endpoint's result schema."""
    batches = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(batches, list) or len(batches) != 1 or not isinstance(batches[0], list):
        raise ValueError("Retriever must return {'result': [[document, ...]]} for one query")
    passages = []
    for i, item in enumerate(batches[0][:topk], 1):
        if not isinstance(item, dict):
            raise ValueError("Each retrieved result must be a document object")
        document = item.get("document", item)
        if not isinstance(document, dict) or not isinstance(document.get("contents"), str):
            raise ValueError("Retrieved document must contain string 'contents'")
        passages.append(f"Doc {i}: {document['contents']}")
    return "\n".join(passages) or "No relevant passages found."


async def retrieve(query, settings):
    """Real retrieval only; connection failures fail the rollout after bounded retries."""

    def request():
        body = json.dumps({"queries": [query], "topk": settings.topk, "return_scores": True}).encode()
        req = urllib.request.Request(settings.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=settings.timeout) as response:
            return format_passages(json.load(response), settings.topk)

    for attempt in range(settings.retries + 1):
        try:
            return await asyncio.to_thread(request)
        except (urllib.error.URLError, TimeoutError):
            if attempt == settings.retries:
                raise
            await asyncio.sleep(2**attempt)


async def run_search_rollout(tokenizer, prompt_ids, response_length, sampling_params, generate, search, settings):
    """Append generated token IDs without retokenizing any model-produced span.

    Environment tokens stay in every subsequent student/teacher prefix and get
    mask=0. Even the <information> delimiters have mask=0. Only passage content
    is truncated; closing delimiters are always retained. The total response
    budget includes both generated tokens and observations.
    """
    response, mask, logprobs, pending = [], [], [], []
    min_global_steps = max_global_steps = None
    search_turns = generation_turns = 0
    generation_seconds = retrieval_seconds = 0.0
    while len(response) < response_length:
        remaining = response_length - len(response)
        params = dict(sampling_params)
        params.update(
            max_tokens=min(settings.max_generation_tokens, remaining),
            stop=["</search>", "</answer>"],
            include_stop_str_in_output=True,
        )
        start = time.monotonic()
        output = await generate(prompt_ids + response, params)
        generation_seconds += time.monotonic() - start
        if getattr(output, "stop_reason", None) == "aborted":
            raise RuntimeError("Rollout generation aborted; refusing partial training sample")
        ids = list(output.token_ids)
        scores = output.log_probs
        if not ids:
            break
        if len(ids) > remaining or scores is None or len(scores) != len(ids):
            raise ValueError("Generation must respect the token budget and return per-token log probabilities")
        # V1 consumes these fields when writing TransferQueue tags and computing
        # trajectory staleness. Preserve the complete version interval across
        # every generation call, including version 0 and partial continuations.
        metadata = getattr(output, "extra_fields", {})
        lower, upper = metadata.get("min_global_steps"), metadata.get("max_global_steps")
        if lower is None or upper is None or lower > upper:
            raise ValueError("Generation must report valid min_global_steps/max_global_steps metadata")
        min_global_steps = int(lower) if min_global_steps is None else min(min_global_steps, int(lower))
        max_global_steps = int(upper) if max_global_steps is None else max(max_global_steps, int(upper))
        generation_turns += 1
        response.extend(ids)
        pending.extend(ids)
        mask.extend([1] * len(ids))
        logprobs.extend(scores)
        text = tokenizer.decode(pending, skip_special_tokens=False)
        if "</answer>" in text:
            break
        queries = re.findall(r"<search>(.*?)</search>", text, flags=re.DOTALL)
        if queries:
            query = queries[-1].strip()
            if not query or search_turns >= settings.max_search_turns:
                break
            opening = tokenizer.encode("\n<information>\n", add_special_tokens=False)
            closing = tokenizer.encode("\n</information>\n", add_special_tokens=False)
            available = response_length - len(response) - len(opening) - len(closing) - 1
            if available < 1:
                break
            start = time.monotonic()
            passages = await search(query, settings)
            retrieval_seconds += time.monotonic() - start
            content = tokenizer.encode(passages, add_special_tokens=False)
            observation = opening + content[: min(available, settings.max_observation_tokens)] + closing
            response.extend(observation)
            mask.extend([0] * len(observation))
            logprobs.extend([0.0] * len(observation))
            pending = []
            search_turns += 1
        elif ids[-1] == tokenizer.eos_token_id:
            break
        # A length-limited reasoning segment can continue on the next call.
    if not any(mask):
        raise ValueError("Empty model trajectory")
    return {
        "prompt_ids": list(prompt_ids),
        "response_ids": response,
        "response_mask": mask,
        "response_logprobs": logprobs,
        "num_turns": 1 + generation_turns + search_turns,
        "metrics": {"generate_sequences": generation_seconds, "tool_calls": retrieval_seconds},
        "extra_fields": {
            "min_global_steps": min_global_steps,
            "max_global_steps": max_global_steps,
            "search_turns": search_turns,
            "turn_scores": [],
            "tool_rewards": [],
        },
    }
