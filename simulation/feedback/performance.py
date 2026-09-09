"""Employee performance metrics as an outcome signal.

Performance is *derived*, never drawn: it emerges from what a worker actually did
under the conditions they faced. A worker who looks slow may simply have been
handed hard tasks while fatigued, and the difficulty adjustment is what separates
those cases — which is the whole point of measuring performance in a workforce
study rather than assuming it.

Like the survey instruments, these are synthetic and configurable, and they are
*periodic*: records are cut at shift boundaries, so the agent sees performance
history, never a live per-task quality readout it could not have in reality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from simulation.entities.core import Task, Worker


@dataclass
class TaskOutcome:
    """One completed task, recorded for performance attribution."""

    worker_id: str
    department_id: str
    task_id: str
    activity_id: str
    difficulty: float           # required proficiency x duration, normalised
    expected_minutes: float
    actual_minutes: float
    on_time: bool
    reworked: bool
    fatigue_at_start: float
    overtime: bool
    at: float


@dataclass
class PerformanceRecord:
    """A worker's performance over one review period (a shift)."""

    worker_id: str
    department_id: str
    role_id: str
    period_start: float
    period_end: float
    tasks_completed: int
    on_time_rate: float
    mean_efficiency: float          # expected / actual; >1 is faster than expected
    difficulty_handled: float
    rework_rate: float
    mean_fatigue: float
    overtime_minutes: float
    performance_index: float        # difficulty-adjusted composite

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class PerformanceTracker:
    """Accumulates task outcomes and cuts periodic performance records."""

    def __init__(self):
        self.outcomes: list[TaskOutcome] = []
        self.records: list[PerformanceRecord] = []
        self._period_start = 0.0

    def record_task(self, worker: Worker, task: Task, expected_minutes: float,
                    actual_minutes: float, on_time: bool, fatigue_at_start: float,
                    now: float) -> None:
        difficulty = float(np.clip(
            task.min_proficiency * (task.duration_median_min / 45.0), 0.05, 3.0))
        self.outcomes.append(TaskOutcome(
            worker_id=worker.id, department_id=worker.department_id, task_id=task.id,
            activity_id=task.activity_id, difficulty=difficulty,
            expected_minutes=float(expected_minutes), actual_minutes=float(actual_minutes),
            on_time=bool(on_time), reworked=task.rework_count > 0,
            fatigue_at_start=float(fatigue_at_start), overtime=bool(worker.on_overtime),
            at=float(now),
        ))

    def cut_period(self, workers: list[Worker], now: float) -> list[PerformanceRecord]:
        """Close the review period and emit one record per worker who did anything."""
        start, end = self._period_start, float(now)
        by_worker: dict[str, list[TaskOutcome]] = {}
        for o in self.outcomes:
            if start <= o.at <= end:
                by_worker.setdefault(o.worker_id, []).append(o)

        index = {w.id: w for w in workers}
        new_records = []
        for wid, outs in by_worker.items():
            worker = index.get(wid)
            if worker is None or not outs:
                continue
            efficiencies = [o.expected_minutes / max(o.actual_minutes, 1e-6) for o in outs]
            mean_eff = float(np.mean(efficiencies))
            on_time_rate = float(np.mean([o.on_time for o in outs]))
            rework_rate = float(np.mean([o.reworked for o in outs]))
            difficulty = float(np.mean([o.difficulty for o in outs]))
            mean_fatigue = float(np.mean([o.fatigue_at_start for o in outs]))

            # Difficulty-adjusted composite: credit for speed and reliability,
            # scaled up for harder work, penalised for rework. Bounded so a single
            # freak-fast task cannot dominate a worker's rating.
            raw = (0.5 * min(mean_eff, 2.0) + 0.5 * on_time_rate) * (0.75 + 0.35 * difficulty)
            performance_index = float(np.clip(raw - 0.4 * rework_rate, 0.0, 2.0))

            new_records.append(PerformanceRecord(
                worker_id=wid, department_id=worker.department_id, role_id=worker.role_id,
                period_start=start, period_end=end, tasks_completed=len(outs),
                on_time_rate=on_time_rate, mean_efficiency=mean_eff,
                difficulty_handled=difficulty, rework_rate=rework_rate,
                mean_fatigue=mean_fatigue,
                overtime_minutes=float(sum(o.actual_minutes for o in outs if o.overtime)),
                performance_index=performance_index,
            ))

        self.records.extend(new_records)
        self._period_start = end
        return new_records

    def department_summary(self, department_id: str | None = None) -> dict:
        """Aggregate of *closed* periods only — nothing from the period in progress."""
        recs = [r for r in self.records
                if department_id is None or r.department_id == department_id]
        if not recs:
            return {"n_records": 0, "mean_performance": 0.0, "mean_on_time": 0.0}
        return {
            "n_records": len(recs),
            "mean_performance": float(np.mean([r.performance_index for r in recs])),
            "mean_on_time": float(np.mean([r.on_time_rate for r in recs])),
            "mean_efficiency": float(np.mean([r.mean_efficiency for r in recs])),
        }
