"""Infrastructure-failure classification shared by the agent and sandbox."""

from __future__ import annotations

from collections import deque


INFRASTRUCTURE_MARKERS = (
    "connectionerror",
    "connection refused",
    "connection reset",
    "could not connect",
    "failed to establish a new connection",
    "name or service not known",
    "name resolution",
    "name resolution failed",
    "nameresolutionerror",
    "failed to resolve",
    "temporary failure in name resolution",
    "getaddrinfo failed",
    "gaierror",
    "dns",
    "connecttimeout",
    "readtimeout",
    "connect timeout",
    "read timed out",
    "maxretryerror",
    "network is unreachable",
    "no route to host",
    "ollama at http",
    "ollama returned http 5",
    "openml.exceptions",
    "temporarily unavailable",
)


def is_infrastructure_error(error: BaseException | str) -> bool:
    """Recognize transient network/service failures without masking candidate bugs."""
    pending = deque([error] if isinstance(error, BaseException) else [])
    seen: set[int] = set()
    while pending:
        current = pending.popleft()
        if id(current) in seen:
            continue
        seen.add(id(current))
        error_text = f"{type(current).__module__}.{type(current).__name__}: {current}".lower()
        if any(marker in error_text for marker in INFRASTRUCTURE_MARKERS):
            return True
        for cause in (current.__cause__, current.__context__):
            if cause is not None:
                pending.append(cause)

    text = str(error).lower()
    return any(marker in text for marker in INFRASTRUCTURE_MARKERS)
