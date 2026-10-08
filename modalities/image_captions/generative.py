"""Generative token pooling (Wang et al., 2026) with vLLM. The model describes the caption and its output is pooled.

The caption is wrapped in "Imagine what it would look like to see: {}", formatted with the chat template (thinking
enabled) and decoded greedily for at most 128 tokens. The embedding is the mean of the last-layer hidden states that
produce the generated tokens. vLLM computes them in each step for the logits, and a hook on its sampler sums them per
request, so no second pass is needed.

Needs the ``vllm`` pixi environment, because vLLM pins its own torch. vLLM batches the prompts of one call, and the
numbers depend on the batch, so texts are embedded in calls of 64 prompts as for the stored files. Prefix caching is
off. With it on, a prompt whose start was cached earlier reuses the cached keys and values, so the result depends on
what the engine embedded before. The stored files were computed with it on and so cannot be reproduced bit for bit
(see ``tests/embeddings/test_image_captions.py``).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

GENERATION_PROMPT = "Imagine what it would look like to see: {}"
BATCH_SIZE = 64  # prompts per vLLM call

# Running sum and count of the captured hidden states per request of the current call.
_hidden_sums: dict[str, Tensor] = {}
_hidden_counts: dict[str, int] = {}


def _accumulate(
    request_ids: list[str],
    hidden_states: Tensor,
) -> None:
    """Add one hidden state per request of a step. A step whose rows do not match its requests is skipped. This
    only happens with speculative decoding, which we do not use."""
    rows = hidden_states.detach().float()
    if rows.shape[0] != len(request_ids):
        return
    for row, runner_id in zip(rows, request_ids):
        request_id = runner_id.rsplit("-", 1)[0]  # the runner appends a suffix to the request id
        if request_id in _hidden_sums:
            _hidden_sums[request_id] += row
            _hidden_counts[request_id] += 1
        else:
            _hidden_sums[request_id] = row.clone()
            _hidden_counts[request_id] = 1


def _install_hook() -> None:
    """Record the hidden states that vLLM's model runner (V2, the default for Qwen3) turns into logits."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_captures_hidden_states", False):
        return
    sample = GPUModelRunner.sample

    def sample_and_capture(
        self: GPUModelRunner,
        hidden_states: Tensor,
        input_batch: InputBatch,
        *args: object,
        **kwargs: object,
    ) -> tuple[SamplerOutput, Tensor, Tensor]:
        _accumulate(input_batch.req_ids, hidden_states[input_batch.logits_indices])
        return sample(self, hidden_states, input_batch, *args, **kwargs)

    GPUModelRunner.sample = sample_and_capture
    GPUModelRunner._captures_hidden_states = True


class GenerativePooling:
    """Qwen3 with generative token pooling. ``__call__`` takes prompts that already contain the caption."""

    def __init__(
        self,
        name: str,
        gpu_memory_utilization: float,
    ) -> None:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # the engine must run in this process
        _install_hook()
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        with torch.autocast("cuda", enabled=False):  # vLLM sets its own precision
            self.engine = LLM(
                model=name,
                runner="generate",
                max_model_len=32768,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=True,
                dtype="auto",
                seed=42,
                enable_prefix_caching=False,
            )
        self.sampling = SamplingParams(max_tokens=128, temperature=0.0, top_p=1.0, seed=42)  # greedy

    def __call__(
        self,
        prompts: list[str],
    ) -> Tensor:
        from vllm.inputs import TokensPrompt

        chats = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            for prompt in prompts
        ]
        token_ids = self.tokenizer(chats)["input_ids"]
        _hidden_sums.clear()
        _hidden_counts.clear()
        with torch.autocast("cuda", enabled=False):
            outputs = self.engine.generate(
                [TokensPrompt(prompt_token_ids=ids) for ids in token_ids], self.sampling, use_tqdm=False
            )
        if not _hidden_sums:
            raise RuntimeError("no hidden states were captured: the vLLM version does not use the hooked runner")
        means = [_hidden_sums[output.request_id] / _hidden_counts[output.request_id] for output in outputs]
        return torch.stack(means).cpu()


GENERATIVE_ENCODERS = {  # model name -> (checkpoint, prompt repeats, GPU memory share for vLLM)
    "qwen3-8b-gen_69445574b3021b8a2733e87d8603d769": ("Qwen/Qwen3-8B", 3, 0.92),
    "qwen3-1.7b-gen_1aa1557c74992c52c60f81e5a1bd6621": ("Qwen/Qwen3-1.7B", 1, 0.55),
}
