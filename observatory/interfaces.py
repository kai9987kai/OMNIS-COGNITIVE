"""Formal typed interface protocols for the OMNIS OBSERVATORY FastAPI & UI subsystem."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from fastapi import FastAPI


@runtime_checkable
class ObservatoryFactoryProtocol(Protocol):
    """Callable contract for constructing an Observatory FastAPI application bound to an OmnisCore instance."""

    def __call__(self, core: Optional[object] = None) -> FastAPI: ...
