"""
Request scheduler.

Maintains a queue of pending GenerationRequest objects and decides which
requests join the active batch on each scheduling step.

Policy: FIFO admission up to max_batch_size; requests that have exceeded
max_new_tokens are marked done and removed from the active set.
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, List, Optional


@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9

    # filled in by the scheduler
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    arrived_at: float = field(default_factory=time.monotonic)

    # generation state
    generated_ids: List[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: Optional[str] = None   # "max_tokens" | "eos"


class Scheduler:
    """
    Manages the waiting queue and active batch.

    Usage:
        sched = Scheduler(max_batch_size=8)
        sched.add(req)
        active = sched.get_active_batch()
        # ... generate one token for each request in active ...
        sched.step(eos_ids={req.request_id})   # marks finished reqs, admits new ones
        done = sched.pop_finished()
    """

    def __init__(self, max_batch_size: int = 8) -> None:
        self._max_batch = max_batch_size
        self._waiting: Deque[GenerationRequest] = deque()
        self._active: Dict[str, GenerationRequest] = {}
        self._finished: List[GenerationRequest] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, request: GenerationRequest) -> None:
        self._waiting.append(request)
        self._admit_from_queue()

    def get_active_batch(self) -> List[GenerationRequest]:
        return list(self._active.values())

    def step(self, eos_ids: set[str] | None = None) -> None:
        """
        Called after one decode step.

        eos_ids: set of request_ids for which the model emitted an EOS token.
        """
        eos_ids = eos_ids or set()
        to_remove = []

        for rid, req in self._active.items():
            if rid in eos_ids:
                req.finished = True
                req.finish_reason = "eos"
                to_remove.append(rid)
            elif len(req.generated_ids) >= req.max_new_tokens:
                req.finished = True
                req.finish_reason = "max_tokens"
                to_remove.append(rid)

        for rid in to_remove:
            self._finished.append(self._active.pop(rid))

        self._admit_from_queue()

    def pop_finished(self) -> List[GenerationRequest]:
        done = list(self._finished)
        self._finished.clear()
        return done

    def is_idle(self) -> bool:
        return len(self._active) == 0 and len(self._waiting) == 0

    def stats(self) -> dict:
        return {
            "waiting": len(self._waiting),
            "active": len(self._active),
            "finished_pending_pop": len(self._finished),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _admit_from_queue(self) -> None:
        while self._waiting and len(self._active) < self._max_batch:
            req = self._waiting.popleft()
            self._active[req.request_id] = req
