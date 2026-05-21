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

# Blocking endpoint (single request)
python server.py
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_new_tokens": 64}' | python -m json.tool

# Run benchmarks
python benchmarks/bench.py --model gpt2 --prompt-lengths 32 128 512 --batch-sizes 1 4 8
```

Set `MODEL_NAME`, `DEVICE`, `MAX_BATCH_SIZE`, `KV_CACHE_CAPACITY` via environment variables.

## Key design decisions

### KV-cache reuse
HuggingFace models return `past_key_values` — a tuple of `(key, value)` tensors per
transformer layer. After prefill, these are stored in the `KVCacheManager`. Each decode
step feeds only the last generated token and the cached state, reducing compute from
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

## Benchmark example (GPT-2, RTX 4090)

| Prompt tokens | Prefill p50 | Prefill p95 | Decode p50 | Tokens/sec |
|---------------|-------------|-------------|------------|------------|
| 32            | 8 ms        | 11 ms       | 95 ms      | 135        |
| 128           | 18 ms       | 23 ms       | 98 ms      | 131        |
| 512           | 52 ms       | 61 ms       | 103 ms     | 124        |

*(Actual numbers vary by hardware and model size.)*

## What I studied building this

- **vLLM**: paged attention, block-based KV memory, preemption
- **SGLang**: RadixAttention for KV-cache sharing across requests with common prefixes
- **FlashAttention**: IO-aware exact attention that fuses Q/K/V matmuls and softmax
- **Paged attention**: treating KV cache like virtual memory pages to eliminate fragmentation

## References

- [vLLM paper — Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180)
- [SGLang — Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [HuggingFace Transformers `past_key_values` docs](https://huggingface.co/docs/transformers/main/en/model_doc/auto)
