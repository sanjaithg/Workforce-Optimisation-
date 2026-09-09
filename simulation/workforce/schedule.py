"""Work schedules as first-class workforce state.

Previously shift membership was an attribute on the worker and availability was
recomputed ad hoc. A schedule makes the roster explicit: who is planned on, for
which window, with what overtime authorisation, and who did not turn up. That is
the object real workforce management actually manipulates, and it is what the
staffing half of the agent's action space is really editing.

The schedule drives:
    availability      - is this worker on duty at time t?
    shift boundaries  - when do STAFFING decision epochs occur?
    overtime          - extending a planned window, at a cost and a fatigue penalty
    staffing actions  - activating reserves adds schedule entries

All synthetic and configurable; no real rostering data is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from simulation.core.config import DomainConfig
from simulation.entities.core import Worker

MINUTES_PER_DAY = 1440.0


@dataclass
class ScheduleEntry:
    """One worker's planned duty window on one day."""

    worker_id: str
    day_index: int
    calendar_date: date
    shift_id: str
    planned_start: float           # simulation minutes from episode start
    planned_end: float
    department_id: str
    location_id: str
    absent: bool = False
    overtime_minutes_authorised: float = 0.0
    actual_busy_minutes: float = 0.0

    def covers(self, t: float) -> bool:
        return self.planned_start <= t < (self.planned_end + self.overtime_minutes_authorised)

    def as_dict(self) -> dict:
        return {
            "worker_id": self.worker_id, "date": self.calendar_date.isoformat(),
            "day_index": self.day_index, "shift_id": self.shift_id,
            "department_id": self.department_id, "location_id": self.location_id,
            "planned_start_min": round(self.planned_start, 1),
            "planned_end_min": round(self.planned_end, 1),
            "absent": self.absent,
            "overtime_minutes_authorised": round(self.overtime_minutes_authorised, 1),
            "actual_busy_minutes": round(self.actual_busy_minutes, 1),
        }


class Roster:
    """The set of schedule entries for the episode, indexed for fast lookup."""

    def __init__(self, domain: DomainConfig, start_date: date):
        self.domain = domain
        self.start_date = start_date
        self.entries: list[ScheduleEntry] = []
        self._by_worker_day: dict[tuple[str, int], ScheduleEntry] = {}

    def build_day(self, workers: list[Worker], day_index: int) -> list[ScheduleEntry]:
        """Create the planned roster for one day, before absences are known."""
        day = self.start_date + timedelta(days=day_index)
        created = []
        for w in workers:
            shift = self.domain.shifts[w.shift_id]
            start = day_index * MINUTES_PER_DAY + shift.start_hour * 60.0
            entry = ScheduleEntry(
                worker_id=w.id, day_index=day_index, calendar_date=day, shift_id=w.shift_id,
                planned_start=start, planned_end=start + shift.duration_hours * 60.0,
                department_id=w.department_id, location_id=w.home_location_id,
            )
            self.entries.append(entry)
            self._by_worker_day[(w.id, day_index)] = entry
            created.append(entry)
        return created

    def entry_for(self, worker_id: str, t: float) -> ScheduleEntry | None:
        return self._by_worker_day.get((worker_id, int(t // MINUTES_PER_DAY)))

    def mark_absent(self, worker_id: str, day_index: int, absent: bool) -> None:
        entry = self._by_worker_day.get((worker_id, day_index))
        if entry is not None:
            entry.absent = absent

    def authorise_overtime(self, worker_id: str, day_index: int, minutes: float) -> None:
        entry = self._by_worker_day.get((worker_id, day_index))
        if entry is not None:
            entry.overtime_minutes_authorised = minutes

    def is_on_duty(self, worker: Worker, t: float) -> bool:
        """Availability, resolved from the schedule rather than from shift arithmetic.

        Night shifts wrap past midnight, so a worker's duty window can belong to
        the previous day's entry; both are checked.
        """
        day_index = int(t // MINUTES_PER_DAY)
        for idx in (day_index, day_index - 1):
            entry = self._by_worker_day.get((worker.id, idx))
            if entry is not None and not entry.absent and entry.covers(t):
                return True
        return False

    def shift_boundaries(self, day_index: int) -> list[float]:
        return sorted(
            day_index * MINUTES_PER_DAY + s.start_hour * 60.0
            for s in self.domain.shifts.values()
        )

    def to_records(self) -> list[dict]:
        return [e.as_dict() for e in self.entries]
