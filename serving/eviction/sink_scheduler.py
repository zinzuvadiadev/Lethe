from __future__ import annotations

import os

from vllm.v1.core.sched.scheduler import Scheduler


def sink_len_from_env() -> int | None:
    value = os.environ.get("LETHE_SINK_LEN")
    return int(value) if value else None


def clamp_num_prompt_tokens(sink_len: int, num_prompt_tokens: int | None) -> int | None:
    """The core new logic: reinterpret RSWA's "protect the whole prompt"
    boundary as "protect only the first sink_len tokens". vLLM's
    RSWAManager.remove_skipped_blocks treats whatever value flows in as
    num_prompt_tokens as the protected-front-region boundary with no
    semantic dependency on it actually being the prompt length (confirmed
    by reading vllm/v1/core/single_type_kv_cache_manager.py directly) — so
    clamping it here is sufficient, no changes needed inside vLLM itself.
    """
    if num_prompt_tokens is None:
        return None
    return min(sink_len, num_prompt_tokens)


class SinkScheduler(Scheduler):
    """Wraps the stock vLLM Scheduler to apply clamp_num_prompt_tokens()
    to every call that reaches KVCacheCoordinator.remove_skipped_blocks.

    Patches exactly one bound method on the already-constructed
    KVCacheCoordinator instance (vllm/v1/core/kv_cache_coordinator.py) —
    this single seam covers both call paths that lead to gap eviction (the
    main per-step allocation path in KVCacheManager.allocate_slots, and the
    separate P/D-connector-cleanup path), since both ultimately call this
    same coordinator instance's remove_skipped_blocks. See design doc §6
    for why "subclass KVCacheManager instead" was ruled out (eviction
    logic is spread across kv_cache_manager.py, kv_cache_coordinator.py,
    and single_type_kv_cache_manager.py with no single clean override
    point below the Scheduler level).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        sink_len = sink_len_from_env()
        if sink_len is None:
            return
        coordinator = self.kv_cache_manager.coordinator
        original_remove_skipped_blocks = coordinator.remove_skipped_blocks

        def patched_remove_skipped_blocks(
            request_id: str,
            processed_computed_tokens: int,
            num_prompt_tokens: int | None = None,
        ) -> None:
            original_remove_skipped_blocks(
                request_id,
                processed_computed_tokens,
                clamp_num_prompt_tokens(sink_len, num_prompt_tokens),
            )

        coordinator.remove_skipped_blocks = patched_remove_skipped_blocks
