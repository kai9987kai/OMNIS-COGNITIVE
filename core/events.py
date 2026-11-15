"""Event-driven pub/sub communication backbone for OMNIS subsystems."""

from __future__ import annotations

from collections import defaultdict, deque
import threading
from typing import Callable, Deque, Dict, List, Optional

from core.schemas import OmnisEvent

EventCallback = Callable[[OmnisEvent], None]


class EventBus:
    """Thread-safe synchronous event bus with bounded history and telemetry counts."""

    def __init__(self, max_history: int = 2000) -> None:
        self._max_history = max(10, int(max_history))
        self._subscribers: Dict[str, List[EventCallback]] = defaultdict(list)
        self._history: Deque[OmnisEvent] = deque(maxlen=self._max_history)
        self._counts: Dict[str, int] = defaultdict(int)
        self._errors: Deque[Dict[str, str]] = deque(maxlen=100)
        self._lock = threading.RLock()

    def subscribe(self, event_type: str, callback: EventCallback) -> None:
        """Subscribe a callback to a specific event_type or '*' for all events."""
        with self._lock:
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: EventCallback) -> None:
        """Remove a callback subscription if present."""
        with self._lock:
            listeners = self._subscribers.get(event_type)
            if listeners and callback in listeners:
                listeners.remove(callback)

    def publish(self, event: OmnisEvent) -> OmnisEvent:
        """Publish an OmnisEvent to specific and wildcard subscribers and record history."""
        with self._lock:
            self._history.append(event)
            self._counts[event.event_type] += 1
            callbacks = list(self._subscribers.get(event.event_type, ()))
            if event.event_type != "*":
                callbacks.extend(self._subscribers.get("*", ()))

        for cb in callbacks:
            try:
                cb(event)
            except Exception as exc:  # pragma: no cover - defensive recording
                with self._lock:
                    self._errors.append(
                        {
                            "event_type": event.event_type,
                            "event_id": event.event_id,
                            "error": str(exc),
                        }
                    )
        return event

    def emit(
        self,
        event_type: str,
        source: str,
        payload: Optional[Dict] = None,
        tick: int = 0,
    ) -> OmnisEvent:
        """Convenience helper to construct and publish an OmnisEvent."""
        event = OmnisEvent(
            event_type=event_type,
            source=source,
            payload=payload or {},
            tick=tick,
        )
        return self.publish(event)

    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[OmnisEvent]:
        """Return most recent events, optionally filtered by event_type."""
        with self._lock:
            items = list(self._history)
        if event_type is not None:
            items = [e for e in items if e.event_type == event_type]
        if limit > 0:
            items = items[-limit:]
        return items

    def get_counts(self) -> Dict[str, int]:
        """Return cumulative published event counts keyed by event_type."""
        with self._lock:
            return dict(self._counts)

    def get_errors(self) -> List[Dict[str, str]]:
        """Return recent listener callback errors, if any."""
        with self._lock:
            return list(self._errors)

    def clear_history(self) -> None:
        """Clear stored event history while keeping subscriptions intact."""
        with self._lock:
            self._history.clear()
            self._counts.clear()
            self._errors.clear()
