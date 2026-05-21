"""
KV-cache manager.

Stores past_key_values tensors for active sequences so prefill is paid only
once per prompt.  Each slot is identified by a request_id and holds the
(key, value) pairs for every transformer layer.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch


KVLayer = Tuple[torch.Tensor, torch.Tensor]   # (key, value) for one layer


@dataclass
class CacheEntry:
    request_id: str
    past_key_values: List[KVLayer]             # one element per layer
    seq_len: int                               # tokens already processed


class KVCacheManager:
    """
    Fixed-capacity KV-cache pool.

    Eviction policy: LRU by access order (evict the least-recently-used
    entry when the pool is full).
    """

    def __init__(self, capacity: int = 64) -> None:
        self._capacity = capacity
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []    # front = oldest

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has(self, request_id: str) -> bool:
        return request_id in self._cache

    def get(self, request_id: str) -> Optional[CacheEntry]:
        if request_id not in self._cache:
            return None
        self._touch(request_id)
        return self._cache[request_id]

    def put(self, request_id: str, past_key_values: List[KVLayer], seq_len: int) -> None:
        if request_id in self._cache:
            self._cache[request_id].past_key_values = past_key_values
            self._cache[request_id].seq_len = seq_len
            self._touch(request_id)
            return

        if len(self._cache) >= self._capacity:
            self._evict_lru()

        self._cache[request_id] = CacheEntry(
            request_id=request_id,
            past_key_values=past_key_values,
            seq_len=seq_len,
        )
        self._access_order.append(request_id)

    def delete(self, request_id: str) -> None:
        if request_id in self._cache:
            del self._cache[request_id]
            self._access_order.remove(request_id)

    def size(self) -> int:
        return len(self._cache)

    def memory_bytes(self) -> int:
        """Rough estimate of GPU bytes consumed by all cached KV tensors."""
        total = 0
        for entry in self._cache.values():
            for k, v in entry.past_key_values:
                total += k.nelement() * k.element_size()
                total += v.nelement() * v.element_size()
        return total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _touch(self, request_id: str) -> None:
        self._access_order.remove(request_id)
        self._access_order.append(request_id)

    def _evict_lru(self) -> None:
        oldest = self._access_order[0]
        self.delete(oldest)
