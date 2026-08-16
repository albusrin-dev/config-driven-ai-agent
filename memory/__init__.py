"""Memory adapters implementing the core MemoryPort contract."""

from __future__ import annotations

from config.models import MemoryStrategy
from core.memory import MemoryPort

from .null import NullMemory
from .window import WindowMemory

__all__ = ["NullMemory", "WindowMemory", "adapter_for"]


def adapter_for(strategy: MemoryStrategy) -> MemoryPort:
    """Select the adapter for an agent's configured strategy. The loop only
    ever sees the port; this factory is for composition roots (the CLI)."""
    if strategy is MemoryStrategy.NONE:
        return NullMemory()
    if strategy is MemoryStrategy.WINDOW:
        return WindowMemory()
    raise ValueError(f"unknown memory strategy: {strategy!r}")
