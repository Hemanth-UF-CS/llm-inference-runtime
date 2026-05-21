"""
Continuous batching engine.

Groups all active decode steps into a single per-request loop iteration so
multiple prompts share the forward pass overhead.  New requests can join
between decode steps (continuous batching) rather than waiting for the
current batch to finish.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Dict, List

from .generator import TokenGenerator
from .scheduler import GenerationRequest, Scheduler

logger = logging.getLogger(__name__)


class ContinuousBatcher:
    """
    Async engine that accepts requests and generates tokens continuously.

    Architecture:
        - A background asyncio task runs the generate_loop.
        - Each submitted request gets a Future; it resolves when generation
          finishes.
        - Prefill is done once per request on admission; subsequent steps
          reuse the KV cache via the TokenGenerator.
    """

    def __init__(
        self,
        generator: TokenGenerator,
        max_batch_size: int = 8,
    ) -> None:
        self._gen = generator
        self._scheduler = Scheduler(max_batch_size=max_batch_size)
        self._futures: Dict[str, asyncio.Future] = {}
        self._prefilled: set[str] = set()
        self._running = False
        self._loop_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._loop_task = asyncio.create_task(self._generate_loop())
        logger.info("ContinuousBatcher started")

    async def stop(self) -> None:
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("ContinuousBatcher stopped")

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    async def submit(self, request: GenerationRequest) -> str:
        """
        Submit a request and await its completion.
        Returns the generated text.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        self._futures[request.request_id] = fut
        self._scheduler.add(request)
        return await fut

    # ------------------------------------------------------------------
    # Internal generate loop
    # ------------------------------------------------------------------

    async def _generate_loop(self) -> None:
        eos_id = self._gen.eos_token_id()

        while self._running:
            active = self._scheduler.get_active_batch()

            if not active:
                await asyncio.sleep(0.001)
                continue

            eos_this_step: set[str] = set()

            for req in active:
                try:
                    # Prefill on first encounter
                    if req.request_id not in self._prefilled:
                        self._gen.prefill(req)
                        self._prefilled.add(req.request_id)
                    else:
                        tok = self._gen.decode_step(req)
                        if tok == eos_id:
                            eos_this_step.add(req.request_id)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Error processing request %s", req.request_id)
                    fut = self._futures.pop(req.request_id, None)
                    if fut and not fut.done():
                        fut.set_exception(exc)
                    eos_this_step.add(req.request_id)

            self._scheduler.step(eos_ids=eos_this_step)

            for done_req in self._scheduler.pop_finished():
                self._prefilled.discard(done_req.request_id)
                self._gen.kv_cache.delete(done_req.request_id)

                fut = self._futures.pop(done_req.request_id, None)
                if fut and not fut.done():
                    text = self._gen.decode_text(done_req.generated_ids)
                    fut.set_result(text)

            # yield to event loop between steps
            await asyncio.sleep(0)
