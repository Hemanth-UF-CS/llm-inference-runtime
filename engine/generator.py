"""
Token-by-token LLM generator with KV-cache reuse.

Wraps a HuggingFace AutoModelForCausalLM and uses the model's built-in
past_key_values mechanism so prefill costs are paid once per prompt.
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .kv_cache import KVCacheManager
from .scheduler import GenerationRequest


class TokenGenerator:
    """
    Single-model inference engine.

    Supports:
    - One-shot generation (generate_sync)
    - Step-by-step decode with external KV-cache management (prefill / decode)
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
        kv_cache_capacity: int = 64,
    ) -> None:
        self.device = device
        self.dtype = dtype

        print(f"[generator] loading {model_name} on {device} ({dtype})")
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        )
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.kv_cache = KVCacheManager(capacity=kv_cache_capacity)

    # ------------------------------------------------------------------
    # Public: prefill + decode steps (used by the batcher)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def prefill(self, request: GenerationRequest) -> None:
        """
        Run the prompt through the model; store KV cache for reuse.
        Returns the first generated token id.
        """
        t0 = time.perf_counter()
        inputs = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=False,
        ).to(self.device)

        outputs = self.model(
            **inputs,
            use_cache=True,
        )

        next_token_id = self._sample(outputs.logits[:, -1, :], request)
        request.generated_ids.append(next_token_id)

        self.kv_cache.put(
            request_id=request.request_id,
            past_key_values=list(outputs.past_key_values),
            seq_len=inputs["input_ids"].shape[1],
        )

        prefill_ms = (time.perf_counter() - t0) * 1000
        request.__dict__.setdefault("_metrics", {})["prefill_ms"] = prefill_ms

    @torch.inference_mode()
    def decode_step(self, request: GenerationRequest) -> int:
        """
        Generate one next token reusing the cached KV state.
        Returns the new token id.
        """
        entry = self.kv_cache.get(request.request_id)
        if entry is None:
            raise RuntimeError(f"No KV cache for request {request.request_id}; run prefill first")

        last_token = torch.tensor(
            [[request.generated_ids[-1]]],
            dtype=torch.long,
            device=self.device,
        )

        outputs = self.model(
            input_ids=last_token,
            past_key_values=tuple(
                (k.to(self.device), v.to(self.device))
                for k, v in entry.past_key_values
            ),
            use_cache=True,
        )

        next_token_id = self._sample(outputs.logits[:, -1, :], request)
        request.generated_ids.append(next_token_id)

        self.kv_cache.put(
            request_id=request.request_id,
            past_key_values=list(outputs.past_key_values),
            seq_len=entry.seq_len + len(request.generated_ids),
        )

        return next_token_id

    def decode_text(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # One-shot convenience wrapper
    # ------------------------------------------------------------------

    def generate_sync(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> Tuple[str, dict]:
        """Blocking single-prompt generation. Returns (text, metrics)."""
        req = GenerationRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        t_start = time.perf_counter()
        self.prefill(req)

        eos = self.eos_token_id()
        decode_start = time.perf_counter()

        for _ in range(max_new_tokens - 1):
            tok = self.decode_step(req)
            if tok == eos:
                req.finish_reason = "eos"
                break
        else:
            req.finish_reason = "max_tokens"

        total_s = time.perf_counter() - t_start
        decode_s = time.perf_counter() - decode_start
        n_tokens = len(req.generated_ids)

        metrics = {
            "total_ms": total_s * 1000,
            "prefill_ms": req.__dict__.get("_metrics", {}).get("prefill_ms", 0),
            "decode_ms": decode_s * 1000,
            "tokens_generated": n_tokens,
            "tokens_per_sec": n_tokens / decode_s if decode_s > 0 else 0,
            "finish_reason": req.finish_reason,
        }

        self.kv_cache.delete(req.request_id)
        return self.decode_text(req.generated_ids), metrics

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample(self, logits: torch.Tensor, req: GenerationRequest) -> int:
        """Temperature + top-p (nucleus) sampling."""
        if req.temperature <= 0:
            return int(logits.argmax(dim=-1).item())

        logits = logits / req.temperature
        probs = torch.softmax(logits, dim=-1)

        # nucleus filter
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > req.top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()

        sampled = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_idx[0, sampled].item())
