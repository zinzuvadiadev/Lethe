from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class AggressivenessSetting:
    name: str
    sink_len: int | None
    recent_window: int | None


# See docs/superpowers/specs/2026-08-09-kv-cache-eviction-benchmark-design.md
# §5: aggressiveness = recent-window size. sink_len fixed at 64 (the value
# already validated live in milestone 4) across all eviction points; only
# the window shrinks as aggressiveness increases.
DEFAULT_SETTINGS: tuple[AggressivenessSetting, ...] = (
    AggressivenessSetting("baseline", sink_len=None, recent_window=None),
    AggressivenessSetting("window_2048", sink_len=64, recent_window=2048),
    AggressivenessSetting("window_1024", sink_len=64, recent_window=1024),
    AggressivenessSetting("window_512", sink_len=64, recent_window=512),
    AggressivenessSetting("window_256", sink_len=64, recent_window=256),
)
