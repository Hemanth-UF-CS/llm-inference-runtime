"""
Unit tests for Scheduler.

Pure Python — no GPU, no model required.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.scheduler import GenerationRequest, Scheduler


def _req(**kwargs) -> GenerationRequest:
    return GenerationRequest(prompt="test prompt", **kwargs)


class TestAdmission:
    def test_request_admitted_immediately_when_batch_not_full(self):
        sched = Scheduler(max_batch_size=4)
        req = _req()
        sched.add(req)

        active = sched.get_active_batch()
        assert len(active) == 1
        assert active[0].request_id == req.request_id

    def test_request_queued_when_batch_full(self):
        sched = Scheduler(max_batch_size=2)
        r1, r2, r3 = _req(), _req(), _req()
        sched.add(r1)
        sched.add(r2)
        sched.add(r3)

        active_ids = {r.request_id for r in sched.get_active_batch()}
        assert r3.request_id not in active_ids
        assert len(sched.get_active_batch()) == 2

    def test_queued_request_admitted_after_slot_opens(self):
        sched = Scheduler(max_batch_size=1)
        r1, r2 = _req(max_new_tokens=1), _req()
        sched.add(r1)
        sched.add(r2)

        # r1 is active; r2 is waiting
        assert len(sched.get_active_batch()) == 1

        # Simulate r1 hitting max_new_tokens (step marks it done)
        sched.get_active_batch()[0].generated_ids.append(99)
        sched.step()

        # r2 should now be admitted
        active = sched.get_active_batch()
        assert len(active) == 1
        assert active[0].request_id == r2.request_id


class TestStep:
    def test_eos_marks_request_finished(self):
        sched = Scheduler(max_batch_size=4)
        req = _req()
        sched.add(req)

        sched.step(eos_ids={req.request_id})
        assert len(sched.get_active_batch()) == 0

        done = sched.pop_finished()
        assert len(done) == 1
        assert done[0].request_id == req.request_id
        assert done[0].finish_reason == "eos"

    def test_max_tokens_marks_request_finished(self):
        sched = Scheduler(max_batch_size=4)
        req = _req(max_new_tokens=3)
        sched.add(req)

        # Simulate generating exactly max_new_tokens tokens
        for _ in range(3):
            sched.get_active_batch()[0].generated_ids.append(42)
        sched.step()

        done = sched.pop_finished()
        assert len(done) == 1
        assert done[0].finish_reason == "max_tokens"

    def test_pop_finished_clears_list(self):
        sched = Scheduler(max_batch_size=4)
        req = _req()
        sched.add(req)
        sched.step(eos_ids={req.request_id})

        assert len(sched.pop_finished()) == 1
        assert len(sched.pop_finished()) == 0  # already popped

    def test_is_idle_when_empty(self):
        sched = Scheduler(max_batch_size=4)
        assert sched.is_idle()

    def test_not_idle_with_active_request(self):
        sched = Scheduler(max_batch_size=4)
        sched.add(_req())
        assert not sched.is_idle()

    def test_stats_fields_present(self):
        sched = Scheduler(max_batch_size=4)
        stats = sched.stats()
        assert "waiting" in stats
        assert "active" in stats
        assert "finished_pending_pop" in stats
