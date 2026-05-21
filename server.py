"""
FastAPI serving layer.

Endpoints:
  POST /generate          – blocking single-request generation
  POST /generate/async    – submit to the continuous batcher
  GET  /health            – liveness probe
  GET  /metrics           – KV-cache and scheduler stats
"""

from __future__ import annotations
import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from engine import ContinuousBatcher, GenerationRequest, TokenGenerator


# ---------------------------------------------------------------------------
# Config (env-overridable)
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
KV_CACHE_CAPACITY = int(os.getenv("KV_CACHE_CAPACITY", "64"))


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=1.0, ge=0.0, le=4.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    generated_text: str
    metrics: dict
    request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# App lifecycle: spin up generator + batcher once at startup
# ---------------------------------------------------------------------------
generator: TokenGenerator
batcher: ContinuousBatcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    global generator, batcher
    generator = TokenGenerator(
        model_name=MODEL_NAME,
        device=DEVICE,
        kv_cache_capacity=KV_CACHE_CAPACITY,
    )
    batcher = ContinuousBatcher(generator=generator, max_batch_size=MAX_BATCH_SIZE)
    await batcher.start()
    yield
    await batcher.stop()


app = FastAPI(
    title="Mini LLM Inference Runtime",
    description="KV-cache-aware LLM serving with continuous batching.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.get("/metrics")
def metrics():
    return {
        "kv_cache": {
            "entries": generator.kv_cache.size(),
            "capacity": KV_CACHE_CAPACITY,
            "memory_bytes": generator.kv_cache.memory_bytes(),
        },
        "scheduler": batcher._scheduler.stats(),
    }


@app.post("/generate", response_model=GenerateResponse)
def generate_sync(body: GenerateRequest):
    """Blocking generation – pays prefill + full decode in this request."""
    t0 = time.monotonic()
    try:
        text, gen_metrics = generator.generate_sync(
            prompt=body.prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    gen_metrics["wall_ms"] = (time.monotonic() - t0) * 1000
    return GenerateResponse(generated_text=text, metrics=gen_metrics)


@app.post("/generate/async", response_model=GenerateResponse)
async def generate_async(body: GenerateRequest):
    """Submit to the continuous batcher; waits until the request is done."""
    req = GenerationRequest(
        prompt=body.prompt,
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
    )
    try:
        text = await asyncio.wait_for(batcher.submit(req), timeout=120.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Generation timed out") from None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return GenerateResponse(
        generated_text=text,
        metrics={"tokens_generated": len(req.generated_ids), "finish_reason": req.finish_reason},
        request_id=req.request_id,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
