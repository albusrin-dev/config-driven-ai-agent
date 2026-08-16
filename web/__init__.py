"""Web layer: search client, bounded fetcher, HTML→text.

Imports inward only (``core``), never from ``tools``. The URL safety guard
itself lives in ``core.netguard`` because the policy gate needs it — check
time and use time share one implementation.
"""
