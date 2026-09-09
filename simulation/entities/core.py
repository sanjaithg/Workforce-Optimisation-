"""Generic domain entities.

Nothing here mentions railways. A domain (railway, hospital, factory) is supplied
as configuration that populates these types; see config/domain/_schema.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    BREACHED = "breached"
    CANCELLED = "cancelled"


class EpochType(str, Enum):
    DISPATCH = "dispatch"
    STAFFING = "staffing"


@dataclass
class Skill:
    id: str
    name: str
    department_id: str


@dataclass
class Role:
    id: str
    name: str
    department_id: str
    skills: dict[str, float]           # skill_id -> typical proficiency for the role
    qualifications: set[str] = field(default_factory=set)
    hourly_cost: float = 1.0


@dataclass
class Department:
    id: str
    name: str


@dataclass
class Location:
    """A place where work happens. `tier` scales demand and staffing weight."""

    id: str
    name: str
    tier: int = 3
    platforms: int = 2
    neighbours: dict[str, float] = field(default_factory=dict)   # location_id -> travel minutes


@dataclass
class Shift:
    id: str
    name: str
    start_hour: float
    duration_hours: float

    def covers(self, hour_of_day: float) -> bool:
        end = self.start_hour + self.duration_hours
        if end <= 24.0:
            return self.start_hour <= hour_of_day < end
        return hour_of_day >= self.start_hour or hour_of_day < (end - 24.0)


@dataclass
class Worker:
    """A heterogeneous resource. Workers are deliberately not interchangeable."""

    id: str
    role_id: str
    department_id: str
    team_id: str
    home_location_id: str
    shift_id: str
    skills: dict[str, float]                       # skill_id -> proficiency in [0, 1]
    qualifications: set[str]
    hourly_cost: float
    experience_years: float
    absence_propensity: float                      # drawn per worker (Beta), not shared
    capacity_units: float = 1.0

    # Mutable simulation state
    fatigue: float = 0.0
    absent_today: bool = False
    on_overtime: bool = False
    current_task_id: str | None = None
    location_id: str | None = None
    busy_until: float = 0.0
    completed_tasks: int = 0
    busy_minutes: float = 0.0
    overtime_minutes: float = 0.0

    def __post_init__(self) -> None:
        if self.location_id is None:
            self.location_id = self.home_location_id

    @property
    def is_available(self) -> bool:
        return not self.absent_today and self.current_task_id is None

    def proficiency(self, skill_id: str) -> float:
        return self.skills.get(skill_id, 0.0)

    def speed_multiplier(self) -> float:
        """Fatigue slows work; experience speeds it. Bounded to stay sensible."""
        fatigue_penalty = 1.0 + 0.6 * self.fatigue
        experience_gain = 1.0 / (1.0 + 0.02 * min(self.experience_years, 20.0))
        return float(max(0.5, min(2.5, fatigue_penalty * experience_gain)))


@dataclass
class Activity:
    """One node of a process graph."""

    id: str
    name: str
    department_id: str
    required_skill: str
    min_proficiency: float
    required_qualifications: set[str]
    duration_median_min: float
    duration_sigma: float
    successors: list[str] = field(default_factory=list)
    rework_prob: float = 0.0
    rework_target: str | None = None
    parallel_group: str | None = None


@dataclass
class ProcessTemplate:
    """A directed graph of activities: the unit of work, not an isolated task."""

    id: str
    name: str
    start_activity: str
    activities: dict[str, Activity]
    default_priority: int = 3
    sla_minutes: float = 120.0

    def activity(self, activity_id: str) -> Activity:
        return self.activities[activity_id]


@dataclass
class Task:
    """A single activity instance belonging to a process case."""

    id: str
    case_id: str
    process_id: str
    activity_id: str
    location_id: str
    department_id: str
    required_skill: str
    min_proficiency: float
    required_qualifications: set[str]
    priority: int
    created_at: float
    deadline: float
    duration_median_min: float
    duration_sigma: float
    status: TaskStatus = TaskStatus.PENDING
    assigned_worker_id: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    deferrals: int = 0
    rework_count: int = 0
    spawned_by_event: str | None = None       # provenance, so cascades are traceable

    @property
    def queue_time(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, self.started_at - self.created_at)

    @property
    def processing_time(self) -> float:
        if self.started_at is None or self.completed_at is None:
            return 0.0
        return max(0.0, self.completed_at - self.started_at)


@dataclass
class Asset:
    """A piece of equipment that can degrade and fail (latent health)."""

    id: str
    location_id: str
    asset_type: str
    health: float = 1.0            # latent, never directly observed by the agent
    failed: bool = False
    last_serviced_at: float = 0.0


@dataclass
class SimEvent:
    """Anything that happened. The event log is built from these."""

    time: float
    kind: str
    location_id: str | None = None
    detail: dict = field(default_factory=dict)
    caused_by: str | None = None
    id: str = ""
