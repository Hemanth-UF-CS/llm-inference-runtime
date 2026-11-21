# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Fixed
- Benchmark tables in the README were not reproducible: `results/gpt2_rtx3090_20260521.json`
  had fields and rows that `bench.py` does not produce. Removed it, reran `bench.py` on an
  RTX 4090, and rebuilt the tables from `results/gpt2_rtx4090.json`.
- `bench.py` now records GPU, torch and transformers versions and the git SHA in its output.

## [1.1.0] — 2026-05-21

### Added
- Unit tests for `KVCacheManager` (LRU eviction, memory tracking, update semantics)
  and `Scheduler` (admission, step, max-tokens/EOS finish conditions) in `tests/`
- `benchmark_results.json` — measured latency / throughput / GPU memory on RTX 3090
  with GPT-2 (32, 128, 512 prompt tokens; batch sizes 1, 4, 8)
- README: replaced speculative benchmark table with measured numbers; added
  explanation of why sequential scheduling shows sub-linear batch scaling

### Fixed
- Nothing in this release

## [1.0.0] — 2026-05-21

### Added
- `engine/kv_cache.py` — LRU KV-cache pool (`KVCacheManager`) with configurable
  capacity and GPU memory tracking
- `engine/scheduler.py` — FIFO admission scheduler tracking waiting / active /
  finished request state
- `engine/generator.py` — HuggingFace causal LM wrapper with `prefill()` /
  `decode_step()` and temperature + top-p sampling
- `engine/batcher.py` — Async continuous batcher (asyncio background task);
  new requests join the active decode batch between steps
- `server.py` — FastAPI server: `POST /generate` (blocking),
  `POST /generate/async`, `GET /health`, `GET /metrics`
- `benchmarks/bench.py` — p50/p95 prefill/decode latency, tokens/sec, GPU memory
  across prompt lengths and batch sizes
- `.gitignore` — excludes pycache, venv, HuggingFace model cache, profiling output
