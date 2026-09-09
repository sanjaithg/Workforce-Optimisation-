"""Event log and KPI accumulation.

The log is written in a PM4Py-compatible shape: `case_id`, `activity`,
`timestamp` are the three mandatory XES columns (renamed to
`case:concept:name`, `concept:name`, `time:timestamp` on export), with
resource/role/team/location and timing attributes alongside. That means any run
of this simulator is directly minable, without a bespoke export step.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np


@dataclass
class LogRow:
    case_id: str
    activity: str
    timestamp: float          # simulation minutes from episode start
    lifecycle: str            # start | complete | defer | breach
    process_id: str
    location_id: str
    department_id: str
    resource_id: str | None
    role_id: str | None
    team_id: str | None
    queue_time: float
    processing_time: float
    status: str
    priority: int
    outcome: str
    spawned_by_event: str | None = None


class EventLog:
    def __init__(self, episode_start: datetime):
        self.episode_start = episode_start
        self.rows: list[LogRow] = []
        self.events: list[dict] = []           # narrative event feed (all sim events)

    def log_activity(self, row: LogRow) -> None:
        self.rows.append(row)

    def log_event(self, time: float, kind: str, message: str, location_id: str | None = None,
                  detail: dict | None = None, caused_by: str | None = None,
                  event_id: str | None = None) -> None:
        self.events.append({
            "time": float(time),
            "kind": kind,
            "message": message,
            "location_id": location_id,
            "detail": detail or {},
            "caused_by": caused_by,
            "id": event_id,
        })

    def to_dataframe(self):
        import pandas as pd

        if not self.rows:
            return pd.DataFrame()
        df = pd.DataFrame([r.__dict__ for r in self.rows])
        df["timestamp"] = [self.episode_start + timedelta(minutes=float(m)) for m in df["timestamp"]]
        return df

    def to_xes_dataframe(self):
        """Rename to the XES conventions PM4Py expects by default."""
        df = self.to_dataframe()
        if df.empty:
            return df
        return df.rename(columns={
            "case_id": "case:concept:name",
            "activity": "concept:name",
            "timestamp": "time:timestamp",
            "resource_id": "org:resource",
            "role_id": "org:role",
        })


@dataclass
class KPIAccumulator:
    """Running operational KPIs. These are the quantities the reward is built from
    and the quantities we validate against plausibility checks."""

    tasks_created: int = 0
    tasks_completed: int = 0
    tasks_breached: int = 0
    tasks_deferred: int = 0
    rework_events: int = 0
    disruptions: int = 0
    cascade_disruptions: int = 0
    queue_time_total: float = 0.0
    processing_time_total: float = 0.0
    labour_cost: float = 0.0
    overtime_cost: float = 0.0
    overtime_minutes: float = 0.0
    absent_workers: int = 0
    passengers_served: int = 0
    waitlisted_passengers: int = 0
    queue_length_samples: list[float] = field(default_factory=list)
    utilisation_samples: list[float] = field(default_factory=list)
    backlog_samples: list[float] = field(default_factory=list)
    per_department_completed: dict[str, int] = field(default_factory=dict)
    per_department_breached: dict[str, int] = field(default_factory=dict)
    abandon_reasons: "Counter[str]" = field(default_factory=Counter)

    @property
    def mean_queue_time(self) -> float:
        return self.queue_time_total / self.tasks_completed if self.tasks_completed else 0.0

    @property
    def sla_breach_rate(self) -> float:
        total = self.tasks_completed + self.tasks_breached
        return self.tasks_breached / total if total else 0.0

    @property
    def completion_rate(self) -> float:
        return self.tasks_completed / self.tasks_created if self.tasks_created else 0.0

    @property
    def mean_queue_length(self) -> float:
        return float(np.mean(self.queue_length_samples)) if self.queue_length_samples else 0.0

    @property
    def mean_utilisation(self) -> float:
        return float(np.mean(self.utilisation_samples)) if self.utilisation_samples else 0.0

    @property
    def mean_backlog(self) -> float:
        return float(np.mean(self.backlog_samples)) if self.backlog_samples else 0.0

    def summary(self) -> dict:
        return {
            "tasks_created": self.tasks_created,
            "tasks_completed": self.tasks_completed,
            "tasks_breached": self.tasks_breached,
            "tasks_deferred": self.tasks_deferred,
            "rework_events": self.rework_events,
            "completion_rate": round(self.completion_rate, 4),
            "sla_breach_rate": round(self.sla_breach_rate, 4),
            "mean_queue_time_min": round(self.mean_queue_time, 2),
            "mean_queue_length": round(self.mean_queue_length, 2),
            "mean_backlog": round(self.mean_backlog, 2),
            "mean_utilisation": round(self.mean_utilisation, 4),
            "labour_cost": round(self.labour_cost, 2),
            "overtime_cost": round(self.overtime_cost, 2),
            "overtime_minutes": round(self.overtime_minutes, 1),
            "disruptions": self.disruptions,
            "cascade_disruptions": self.cascade_disruptions,
            "absent_workers": self.absent_workers,
            "passengers_served": self.passengers_served,
            "waitlisted_passengers": self.waitlisted_passengers,
            "abandon_reasons": dict(self.abandon_reasons),
        }
