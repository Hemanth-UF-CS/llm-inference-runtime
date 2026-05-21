"""
Benchmark script: prefill/decode latency, tokens/sec, p50/p95, GPU memory.

Usage:
    python benchmarks/bench.py --model gpt2 --batch-sizes 1 4 8 --prompt-lengths 32 128 512
"""

from __future__ import annotations
import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import GenerationRequest, TokenGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_prompt(length: int) -> str:
    """Synthetic prompt of approximately `length` tokens."""
    words = (
        "the quick brown fox jumps over the lazy dog "
        "machine learning inference latency benchmark test "
    )
    chunk = (words * ((length // 10) + 2)).split()
    return " ".join(chunk[:length])


def gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def percentile(data: list[float], p: float) -> float:
    data_sorted = sorted(data)
    idx = (len(data_sorted) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(data_sorted) - 1)
    frac = idx - lo
    return data_sorted[lo] * (1 - frac) + data_sorted[hi] * frac


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------
def run_single(
    gen: TokenGenerator,
    prompt_length: int,
    max_new_tokens: int = 128,
    n_warmup: int = 2,
    n_trials: int = 10,
) -> dict:
    prompt = make_prompt(prompt_length)

    prefill_latencies: list[float] = []
    decode_latencies: list[float] = []
    tokens_per_sec: list[float] = []

    for i in range(n_warmup + n_trials):
        req = GenerationRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,   # greedy for reproducibility
        )

        # Prefill
        t0 = time.perf_counter()
        gen.prefill(req)
        prefill_ms = (time.perf_counter() - t0) * 1000

        # Decode
        eos = gen.eos_token_id()
        t_decode = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            tok = gen.decode_step(req)
            if tok == eos:
                break
        decode_ms = (time.perf_counter() - t_decode) * 1000

        gen.kv_cache.delete(req.request_id)

        if i >= n_warmup:
            n_gen = len(req.generated_ids)
            prefill_latencies.append(prefill_ms)
            decode_latencies.append(decode_ms)
            tokens_per_sec.append(n_gen / (decode_ms / 1000) if decode_ms > 0 else 0)

    return {
        "prompt_length": prompt_length,
        "max_new_tokens": max_new_tokens,
        "prefill_ms_mean": statistics.mean(prefill_latencies),
        "prefill_ms_p50": percentile(prefill_latencies, 50),
        "prefill_ms_p95": percentile(prefill_latencies, 95),
        "decode_ms_mean": statistics.mean(decode_latencies),
        "decode_ms_p50": percentile(decode_latencies, 50),
        "decode_ms_p95": percentile(decode_latencies, 95),
        "tokens_per_sec_mean": statistics.mean(tokens_per_sec),
        "gpu_mem_mb": gpu_mem_mb(),
    }


# ---------------------------------------------------------------------------
# Batch benchmark
# ---------------------------------------------------------------------------
def run_batch(
    gen: TokenGenerator,
    batch_size: int,
    prompt_length: int,
    max_new_tokens: int = 64,
    n_trials: int = 5,
) -> dict:
    """
    Naive batch: run `batch_size` requests sequentially and measure aggregate
    throughput.  (A proper batched forward pass would require padding; this
    measures the scheduler overhead baseline.)
    """
    prompt = make_prompt(prompt_length)
    total_tokens = []
    total_times = []

    for _ in range(n_trials):
        requests = [
            GenerationRequest(prompt=prompt, max_new_tokens=max_new_tokens, temperature=0.0)
            for _ in range(batch_size)
        ]
        t0 = time.perf_counter()
        for req in requests:
            gen.prefill(req)
            eos = gen.eos_token_id()
            for _ in range(max_new_tokens - 1):
                tok = gen.decode_step(req)
                if tok == eos:
                    break
            gen.kv_cache.delete(req.request_id)
        elapsed = time.perf_counter() - t0
        n_toks = sum(len(r.generated_ids) for r in requests)
        total_tokens.append(n_toks)
        total_times.append(elapsed)

    return {
        "batch_size": batch_size,
        "prompt_length": prompt_length,
        "total_tokens_mean": statistics.mean(total_tokens),
        "wall_sec_mean": statistics.mean(total_times),
        "aggregate_tokens_per_sec": statistics.mean(total_tokens) / statistics.mean(total_times),
        "gpu_mem_mb": gpu_mem_mb(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LLM inference benchmark")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--out", default="benchmark_results.json")
    args = parser.parse_args()

    print(f"Loading {args.model} on {args.device}...")
    gen = TokenGenerator(model_name=args.model, device=args.device)

    results = {"model": args.model, "device": args.device, "single": [], "batch": []}

    print("\n=== Single-request benchmarks (latency / tokens-per-sec) ===")
    for pl in args.prompt_lengths:
        r = run_single(gen, pl, args.max_new_tokens, args.n_warmup, args.n_trials)
        results["single"].append(r)
        print(
            f"  prompt_len={pl:4d}  prefill p50={r['prefill_ms_p50']:6.1f}ms  "
            f"prefill p95={r['prefill_ms_p95']:6.1f}ms  "
            f"decode p50={r['decode_ms_p50']:7.1f}ms  "
            f"tps={r['tokens_per_sec_mean']:6.1f}  gpu={r['gpu_mem_mb']:.0f}MB"
        )

    print("\n=== Batch benchmarks (aggregate throughput) ===")
    for bs in args.batch_sizes:
        for pl in args.prompt_lengths[:2]:    # keep runtime reasonable
            r = run_batch(gen, bs, pl, min(64, args.max_new_tokens))
            results["batch"].append(r)
            print(
                f"  batch={bs}  prompt_len={pl:4d}  "
                f"agg_tps={r['aggregate_tokens_per_sec']:7.1f}  "
                f"wall={r['wall_sec_mean']:.2f}s  gpu={r['gpu_mem_mb']:.0f}MB"
            )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
