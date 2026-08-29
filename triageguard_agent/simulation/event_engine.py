"""
event_engine.py
---------------
Event-driven simulation backbone for tracking all dynamic hospital operational occurrences.

Events record state transitions, triage decisions, bed allocations, LOS discharges,
capacity alerts, and scenario changes with simulated timestamps and structured metadata.
"""

from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Supported simulation event categories."""
    PATIENT_ARRIVAL = "PATIENT_ARRIVAL"
    PATIENT_TRIAGED = "PATIENT_TRIAGED"
    PATIENT_ADMITTED = "PATIENT_ADMITTED"
    PATIENT_IN_TREATMENT = "PATIENT_IN_TREATMENT"
    PATIENT_DISCHARGED = "PATIENT_DISCHARGED"
    PATIENT_TRANSFERRED = "PATIENT_TRANSFERRED"
    BED_OPENED = "BED_OPENED"
    BED_CLOSED = "BED_CLOSED"
    CAPACITY_WARNING = "CAPACITY_WARNING"
    SURGE_START = "SURGE_START"
    SURGE_END = "SURGE_END"
    STAFF_OVERRIDE = "STAFF_OVERRIDE"
    SCENARIO_CHANGED = "SCENARIO_CHANGED"
    TIME_ADVANCED = "TIME_ADVANCED"


@dataclass
class SimEvent:
    """Represents a single structured occurrence in the hospital simulation."""
    event_id: str
    event_type: EventType
    sim_time_minutes: int
    formatted_time: str
    message: str
    department: Optional[str] = None
    patient_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else str(self.event_type),
            "sim_time_minutes": self.sim_time_minutes,
            "formatted_time": self.formatted_time,
            "department": self.department,
            "patient_id": self.patient_id,
            "message": self.message,
            "data": self.data,
            "created_at": self.created_at,
        }

    def format_log_line(self) -> str:
        """Format as a clean feed line, e.g.: '10:15  Patient #105 admitted to ICU'."""
        return f"{self.formatted_time:<6} {self.message}"


class EventEngine:
    """
    Manages simulation time, event publishing, listener dispatch, and history audit trails.
    """

    def __init__(self, start_hour: int = 10, start_minute: int = 0) -> None:
        self._start_hour = start_hour
        self._start_minute = start_minute
        self._sim_time_minutes: int = 0
        self._events: List[SimEvent] = []
        self._listeners: List[Callable[[SimEvent], None]] = []

    # ------------------------------------------------------------------
    # Time Management
    # ------------------------------------------------------------------

    @property
    def sim_time_minutes(self) -> int:
        return self._sim_time_minutes

    @property
    def formatted_time(self) -> str:
        """Calculate HH:MM from base start time + elapsed simulated minutes."""
        total_minutes = self._start_hour * 60 + self._start_minute + self._sim_time_minutes
        hours = (total_minutes // 60) % 24
        minutes = total_minutes % 60
        return f"{hours:02d}:{minutes:02d}"

    def advance_time(self, delta_minutes: int) -> str:
        """Advance the simulated clock by delta_minutes."""
        if delta_minutes < 0:
            raise ValueError("delta_minutes cannot be negative.")
        self._sim_time_minutes += delta_minutes
        return self.formatted_time

    def set_time(self, minutes: int) -> None:
        """Set absolute simulation elapsed minutes."""
        if minutes < 0:
            raise ValueError("minutes cannot be negative.")
        self._sim_time_minutes = minutes

    # ------------------------------------------------------------------
    # Event Publishing & Subscription
    # ------------------------------------------------------------------

    def emit(
        self,
        event_type: EventType | str,
        message: str,
        department: Optional[str] = None,
        patient_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> SimEvent:
        """Publish a new event, log it, and notify all registered listeners."""
        if isinstance(event_type, str):
            try:
                event_type = EventType(event_type)
            except ValueError:
                # Fallback if custom string provided
                pass

        event = SimEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            event_type=event_type,
            sim_time_minutes=self._sim_time_minutes,
            formatted_time=self.formatted_time,
            message=message,
            department=department,
            patient_id=str(patient_id) if patient_id is not None else None,
            data=data or {},
        )

        self._events.append(event)
        logger.debug("SimEvent emitted: %s", event.format_log_line())

        # Notify listeners
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception as exc:
                logger.error("Error in event listener %s: %s", listener, exc)

        return event

    def add_listener(self, listener: Callable[[SimEvent], None]) -> None:
        """Register a callback for all emitted events."""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[SimEvent], None]) -> None:
        """Unregister an event listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    # ------------------------------------------------------------------
    # History & Queries
    # ------------------------------------------------------------------

    def get_history(
        self,
        limit: Optional[int] = None,
        event_type: Optional[EventType | str] = None,
        department: Optional[str] = None,
        patient_id: Optional[str] = None,
    ) -> List[SimEvent]:
        """Query past events with optional filtering."""
        filtered = self._events
        if event_type:
            target = event_type.value if isinstance(event_type, EventType) else str(event_type)
            filtered = [
                e for e in filtered
                if (e.event_type.value if isinstance(e.event_type, EventType) else str(e.event_type)) == target
            ]
        if department:
            filtered = [e for e in filtered if e.department == department]
        if patient_id:
            filtered = [e for e in filtered if e.patient_id == str(patient_id)]

        if limit is not None and limit > 0:
            return filtered[-limit:]
        return list(filtered)

    def get_recent_feed(self, limit: int = 10) -> List[str]:
        """Return formatted string lines for recent events (most recent last)."""
        recent = self.get_history(limit=limit)
        return [e.format_log_line() for e in recent]

    def clear(self) -> None:
        """Reset the event log and clock."""
        self._events.clear()
        self._sim_time_minutes = 0
