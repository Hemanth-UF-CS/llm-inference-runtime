# Mini LLM Inference Runtime with KV Cache

A from-scratch LLM inference runtime built around HuggingFace causal language models,
implementing the core abstractions behind systems like vLLM: request scheduling,
KV-cache reuse, continuous batching, and FastAPI serving.

Built to understand vLLM-style inference internals — not to replace production systems.

## Architecture

```
┌───────────────────────────────────────────────────┐
│                    FastAPI server                  │
│   POST /generate      POST /generate/async        │
└──────────────┬────────────────────┬───────────────┘
               │                    │
        sync path            async path
               │                    ▼
               │         ┌─────────────────────┐
               │         │  ContinuousBatcher  │
               │         │  (asyncio task)     │
               │         └────────┬────────────┘
               │                  │
               ▼                  ▼
        ┌─────────────────────────────────┐
        │         TokenGenerator          │
        │  prefill() / decode_step()      │
        │  temperature + top-p sampling   │
        └──────────────┬──────────────────┘
                       │
               ┌───────▼────────┐
               │ KVCacheManager │  LRU eviction, GPU memory tracking
               └────────────────┘
```

### Components

| Module | Description |
|--------|-------------|
| `engine/kv_cache.py` | LRU KV-cache pool; stores `past_key_values` per request |
| `engine/scheduler.py` | FIFO admission scheduler; tracks active / waiting / finished |
| `engine/generator.py` | HuggingFace model wrapper; prefill + decode with cache reuse |
| `engine/batcher.py` | Continuous batching via asyncio; new requests join between steps |
| `server.py` | FastAPI server with `/generate`, `/generate/async`, `/health`, `/metrics` |
| `benchmarks/bench.py` | Latency / throughput / GPU-memory benchmarking across batch sizes |

## Quick start

```bash
pip install -r requirements.txt

# Start the server (default: gpt2, auto-detects CUDA)
python server.py

# Blocking endpoint
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_new_tokens": 64}' | python -m json.tool

# Run benchmarks
python benchmarks/bench.py --model gpt2 --prompt-lengths 32 128 512 --batch-sizes 1 4 8
```

Set `MODEL_NAME`, `DEVICE`, `MAX_BATCH_SIZE`, `KV_CACHE_CAPACITY` via environment variables.

## Benchmark results

Measured on an **RTX 4090** with **GPT-2** (124M parameters, fp16), torch 2.5.1+cu121,
transformers 4.46.3, git `ee64ac5`. Raw numbers: `results/gpt2_rtx4090.json`,
written by `python benchmarks/bench.py --model gpt2 --out results/gpt2_rtx4090.json`.
Ten trials after two warmups for single requests; five trials per batch row.

### Single request (128 new tokens)

| Prompt tokens | Prefill p50 | Prefill p95 | Decode p50 | Decode p95 | Tokens/sec |
|---:|---:|---:|---:|---:|---:|
| 32 | 3.1 ms | 3.2 ms | 375 ms | 381 ms | 340 |
| 128 | 3.3 ms | 3.7 ms | 377 ms | 381 ms | 339 |
| 512 | 4.1 ms | 4.5 ms | 385 ms | 387 ms | 333 |

Prefill grows only slightly with prompt length (3.1 to 4.1 ms from 32 to 512 tokens) because one forward
pass over a few hundred tokens barely loads the GPU. Decode is about 2.9 ms per token regardless
of prompt length. At this model size each step is bound by Python and kernel-launch overhead in
the Hugging Face forward pass, not by attention over the cache.

### Several requests (64 new tokens each)

| Requests | Prompt tokens | Tokens generated | Wall time | Aggregate tokens/sec |
|---:|---:|---:|---:|---:|
| 1 | 32 | 64 | 0.19 s | 337 |
| 4 | 32 | 256 | 0.76 s | 338 |
| 8 | 32 | 512 | 1.52 s | 338 |
| 1 | 128 | 64 | 0.19 s | 339 |
| 4 | 128 | 256 | 0.76 s | 335 |
| 8 | 128 | 512 | 1.52 s | 338 |

Aggregate throughput stays flat at about 337 tokens/sec, and wall time scales linearly with the
number of requests. The benchmark runs requests one after another and the generator has no
padded batched forward pass, so more concurrent requests do not use the GPU any better. That
is the main gap to vLLM:

1. **No batched decode**: vLLM runs every active sequence's decode step in one forward pass.
2. **No paged attention**: the KV cache is stored as dense Hugging Face tensors per request,
   so it cannot share or pack memory in blocks.
3. **Python overhead**: scheduling, tokenization and sampling all run in Python.

GPU memory allocated after the runs: 262 MB, mostly the fp16 weights. The KV cache costs
12 layers x 2 (K, V) x 768 x 2 bytes = 36 KiB per token, about 4.5 MiB for a 128-token sequence.

## Key design decisions

### KV-cache reuse
HuggingFace models return `past_key_values` — a tuple of `(key, value)` tensors per
transformer layer. After prefill, these are stored in `KVCacheManager`. Each decode step
feeds only the last generated token and the cached state, reducing compute from
O(seq_len) to O(1) per step.

### Continuous batching
The `ContinuousBatcher` runs a background asyncio loop. New requests are prefilled as
soon as they arrive and join the decode batch without waiting for existing requests to
finish. This keeps GPU utilization higher than static batching at the cost of more
complex state management.

### LRU cache eviction
The KV cache has a configurable capacity (default: 64 entries). When full, the
least-recently-used entry is evicted. For long-running servers, this bounds GPU memory
growth while allowing cache reuse for concurrent requests.

## What I studied building this

- **vLLM**: paged attention, block-based KV memory, preemption
- **SGLang**: RadixAttention for KV-cache sharing across requests with common prefixes
- **FlashAttention**: IO-aware exact attention that fuses Q/K/V matmuls and softmax
- **Paged attention**: treating KV cache like virtual memory pages to eliminate fragmentation

## References

- [vLLM — Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180)
- [SGLang — Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [HuggingFace Transformers `past_key_values` docs](https://huggingface.co/docs/transformers/main/en/model_doc/auto)
