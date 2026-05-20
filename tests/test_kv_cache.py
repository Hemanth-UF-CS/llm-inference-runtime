"""
Unit tests for KVCacheManager.

These run on CPU — no GPU or HuggingFace model required.
KV tensors are tiny CPU floats; the logic under test is pure Python.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.kv_cache import KVCacheManager


def _fake_kv(n_layers: int = 2, seq_len: int = 8) -> list:
    """Return a list of (key, value) tuples with CPU float32 tensors."""
    return [
        (torch.randn(1, 1, seq_len, 64), torch.randn(1, 1, seq_len, 64))
        for _ in range(n_layers)
    ]


# ---------------------------------------------------------------------------
# Basic put / get
# ---------------------------------------------------------------------------
class TestPutGet:
    def test_put_and_hit(self):
        cache = KVCacheManager(capacity=4)
        kv = _fake_kv()
        cache.put("req-1", kv, seq_len=8)

        assert cache.has("req-1")
        entry = cache.get("req-1")
        assert entry is not None
        assert entry.request_id == "req-1"
        assert entry.seq_len == 8

    def test_miss_returns_none(self):
        cache = KVCacheManager(capacity=4)
        assert not cache.has("nonexistent")
        assert cache.get("nonexistent") is None

    def test_update_existing_entry(self):
        cache = KVCacheManager(capacity=4)
        kv_v1 = _fake_kv(seq_len=8)
        kv_v2 = _fake_kv(seq_len=16)

        cache.put("req-1", kv_v1, seq_len=8)
        cache.put("req-1", kv_v2, seq_len=16)

        entry = cache.get("req-1")
        assert entry.seq_len == 16
        assert cache.size() == 1  # still one entry, not two

    def test_delete(self):
        cache = KVCacheManager(capacity=4)
        cache.put("req-1", _fake_kv(), seq_len=8)
        assert cache.has("req-1")
        cache.delete("req-1")
        assert not cache.has("req-1")
        assert cache.size() == 0

    def test_delete_nonexistent_is_noop(self):
        cache = KVCacheManager(capacity=4)
        cache.delete("does-not-exist")  # should not raise


# ---------------------------------------------------------------------------
# Capacity and LRU eviction
# ---------------------------------------------------------------------------
class TestLRUEviction:
    def test_evicts_when_full(self):
        cache = KVCacheManager(capacity=3)
        for i in range(3):
            cache.put(f"req-{i}", _fake_kv(), seq_len=8)
        assert cache.size() == 3

        # Adding a 4th entry must evict the LRU (req-0, inserted first, never touched)
        cache.put("req-3", _fake_kv(), seq_len=8)
        assert cache.size() == 3
        assert not cache.has("req-0"), "req-0 should have been evicted (LRU)"
        assert cache.has("req-1")
        assert cache.has("req-2")
        assert cache.has("req-3")

    def test_get_updates_lru_order(self):
        cache = KVCacheManager(capacity=3)
        for i in range(3):
            cache.put(f"req-{i}", _fake_kv(), seq_len=8)

        # Touch req-0 so it becomes the most recently used
        cache.get("req-0")

        # Now req-1 is the LRU; adding a new entry should evict it
        cache.put("req-3", _fake_kv(), seq_len=8)
        assert cache.size() == 3
        assert not cache.has("req-1"), "req-1 should have been evicted"
        assert cache.has("req-0")
        assert cache.has("req-2")
        assert cache.has("req-3")

    def test_capacity_one(self):
        cache = KVCacheManager(capacity=1)
        cache.put("a", _fake_kv(), seq_len=4)
        cache.put("b", _fake_kv(), seq_len=4)
        assert cache.size() == 1
        assert cache.has("b")
        assert not cache.has("a")


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------
class TestMemoryTracking:
    def test_memory_bytes_increases_with_entries(self):
        cache = KVCacheManager(capacity=8)
        assert cache.memory_bytes() == 0

        cache.put("req-1", _fake_kv(n_layers=2, seq_len=16), seq_len=16)
        mem_1 = cache.memory_bytes()
        assert mem_1 > 0

        cache.put("req-2", _fake_kv(n_layers=2, seq_len=16), seq_len=16)
        assert cache.memory_bytes() == 2 * mem_1

    def test_memory_bytes_decreases_after_delete(self):
        cache = KVCacheManager(capacity=8)
        cache.put("req-1", _fake_kv(), seq_len=8)
        before = cache.memory_bytes()
        cache.delete("req-1")
        assert cache.memory_bytes() == 0
        assert before > 0
