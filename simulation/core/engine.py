"""Environment A: the model-driven discrete-event simulation.

The engine advances a SimPy clock through meaningful events. Work is not drawn
from a fixed list — tasks exist because something happened:

    train departure  -> turnaround case (cleaning -> checking -> dispatch)
    footfall surge   -> ticketing cases, and crowd-response cases past a threshold
    asset failure    -> emergency repair case, and the asset stays down until fixed
    delay            -> more passengers accumulate -> more crowd/commercial work

Control is handed to a policy at *decision epochs*. Between epochs the clock runs
freely; at an epoch the simulation blocks until an action is supplied. This is the
standard SMDP-style DES/RL coupling: the agent acts at event-triggered times, not
on a fixed tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import simpy

from simulation.core.config import DomainConfig, ScenarioConfig
from simulation.core.rng import RNGRegistry
from simulation.demand.model import DemandModel, TrainDeparture
from simulation.disruptions.generators import Disruption, DisruptionEngine
from simulation.distributions.library import lognormal_duration
from simulation.entities.core import Asset, EpochType, Task, TaskStatus, Worker
from simulation.feedback.experience import ExperienceFactors, ExperienceModel
from simulation.feedback.performance import PerformanceTracker
from simulation.feedback.surveys import FeedbackRegistry, SurveyConfig
from simulation.metrics.costs import CostLedger, CostRates
from simulation.metrics.event_log import EventLog, KPIAccumulator, LogRow
from simulation.processes.templates import CUSTOMER_FACING_PROCESSES, all_templates
from simulation.state.true_state import TrueState
from simulation.workforce.schedule import Roster
from simulation.workforce.population import (
    accrue_fatigue,
    build_reserve_pool,
    build_workforce,
    decay_fatigue,
    is_eligible,
    roll_absences,
)

MINUTES_PER_DAY = 1440.0
CANDIDATE_SLOTS = 10               # bounded action space: top-K eligible workers
ABANDON_GRACE_MINUTES = 180.0      # how long overdue work lingers before abandonment

# Workload-generation constants. All [SYNTH]: chosen so that the reference
# scenario sits at a realistic-but-not-saturated load, since a world with no
# scarcity poses no allocation problem and a permanently saturated one poses no
# solvable problem either.
PASSENGERS_PER_COUNTER_CASE = 65.0
SUBURBAN_TURNAROUND_FRACTION = 0.07
DAILY_INSPECTION_RATE = 0.30       # per asset per day


@dataclass
class DecisionEpoch:
    """A point at which a policy must act."""

    kind: EpochType
    time: float
    task: Task | None = None
    candidates: list[Worker] = field(default_factory=list)
    department_id: str | None = None


class SimulationEngine:
    """Owns the world. Drives itself forward until a decision is needed."""

    def __init__(self, domain: DomainConfig, scenario: ScenarioConfig, seed: int | None = None,
                 collect_trace: bool = False):
        self.domain = domain
        self.scenario = scenario
        self.seed = scenario.seed if seed is None else int(seed)
        self.rng = RNGRegistry(self.seed)
        self.collect_trace = collect_trace

        self.env = simpy.Environment()
        self.templates = all_templates()
        self.demand = DemandModel(domain, scenario, self.rng)
        self.disruptions = DisruptionEngine(domain, scenario, self.rng)

        self.workers: list[Worker] = build_workforce(domain, scenario, self.rng)
        self.reserve_pool: list[Worker] = build_reserve_pool(domain, self.workers, self.rng)
        self.activated_reserves: list[Worker] = []

        self.assets: list[Asset] = self._build_assets()
        self.assets_by_location: dict[str, list[Asset]] = {}
        for a in self.assets:
            self.assets_by_location.setdefault(a.location_id, []).append(a)

        self.episode_start = datetime.combine(scenario.start_date, datetime.min.time())
        self.log = EventLog(self.episode_start)
        self.kpis = KPIAccumulator()

        # Latent state vs observed instruments. TrueState holds what the world
        # knows; feedback/performance are the lagged, sampled views of it.
        fb_cfg = scenario.raw.get("feedback", {})
        self.true_state = TrueState()
        self.experience = ExperienceModel(
            ExperienceFactors.from_config(fb_cfg.get("experience")), self.rng.observation_noise)
        self.feedback = FeedbackRegistry(
            SurveyConfig.from_config(fb_cfg.get("surveys")), self.rng.observation_noise)
        self.performance = PerformanceTracker()
        self.costs = CostLedger(CostRates.from_config(fb_cfg.get("costs")))
        self.roster = Roster(domain, scenario.start_date)

        self.pending_tasks: list[Task] = []
        self.all_tasks: dict[str, Task] = {}
        self.case_counter = 0
        self.task_counter = 0
        self.horizon = scenario.duration_days * MINUTES_PER_DAY

        # Set by the runner/env loop while a decision is outstanding.
        self.current_epoch: DecisionEpoch | None = None
        self._pending_action: int | None = None
        self.last_reward_components: dict = {}
        self.done = False

        # Trace for the visualiser
        self.trace_snapshots: list[dict] = []
        self.trace_decisions: list[dict] = []
        self._last_snapshot_time = -1e9

        self.overtime_authorised: dict[str, bool] = {d: False for d in domain.departments}
        self.departure_schedule: list[TrainDeparture] = []
        self.station_queues: dict[str, int] = {loc: 0 for loc in domain.locations}

    # ------------------------------------------------------------------ setup

    def _build_assets(self) -> list[Asset]:
        assets = []
        spec = self.domain.assets["per_location_by_tier"]
        counter = 0
        for loc in self.domain.locations.values():
            for asset_type, count in spec[loc.tier].items():
                for _ in range(int(count)):
                    counter += 1
                    assets.append(Asset(id=f"A{counter:05d}", location_id=loc.id, asset_type=asset_type))
        return assets

    # ------------------------------------------------------------- task creation

    def _new_case_id(self, prefix: str) -> str:
        self.case_counter += 1
        return f"{prefix}-{self.case_counter:06d}"

    def _spawn_task(self, template_id: str, activity_id: str, case_id: str, location_id: str,
                    priority: int | None = None, caused_by: str | None = None,
                    duration_scale: float = 1.0) -> Task:
        template = self.templates[template_id]
        activity = template.activity(activity_id)
        self.task_counter += 1
        now = float(self.env.now)
        task = Task(
            id=f"T{self.task_counter:06d}",
            case_id=case_id,
            process_id=template_id,
            activity_id=activity_id,
            location_id=location_id,
            department_id=activity.department_id,
            required_skill=activity.required_skill,
            min_proficiency=activity.min_proficiency,
            required_qualifications=set(activity.required_qualifications),
            priority=priority if priority is not None else template.default_priority,
            created_at=now,
            deadline=now + template.sla_minutes,
            duration_median_min=activity.duration_median_min * duration_scale,
            duration_sigma=activity.duration_sigma,
            status=TaskStatus.PENDING,
            spawned_by_event=caused_by,
        )
        self.pending_tasks.append(task)
        self.all_tasks[task.id] = task
        self.kpis.tasks_created += 1
        self.station_queues[location_id] = self.station_queues.get(location_id, 0) + 1
        return task

    def _spawn_case(self, template_id: str, location_id: str, priority: int | None = None,
                    caused_by: str | None = None, duration_scale: float = 1.0) -> Task:
        template = self.templates[template_id]
        case_id = self._new_case_id(template_id[:3].upper())
        return self._spawn_task(template_id, template.start_activity, case_id, location_id,
                                priority, caused_by, duration_scale)

    def _advance_case(self, task: Task) -> None:
        """On completion, spawn successor activities — this is what makes a case a
        process rather than a one-shot job, and what creates department handoffs."""
        template = self.templates[task.process_id]
        activity = template.activity(task.activity_id)

        # Rework: verification can send the case back rather than closing it.
        if activity.rework_prob > 0 and self.rng.process_rework.random() < activity.rework_prob:
            self.kpis.rework_events += 1
            target = activity.rework_target or activity.id
            new = self._spawn_task(task.process_id, target, task.case_id, task.location_id,
                                   task.priority, caused_by=task.spawned_by_event)
            new.rework_count = task.rework_count + 1
            self.log.log_event(self.env.now, "rework",
                               f"Rework on {template.name} at {task.location_id}: back to {target}",
                               task.location_id, {"case_id": task.case_id})
            return

        for successor in activity.successors:
            self._spawn_task(task.process_id, successor, task.case_id, task.location_id,
                             task.priority, caused_by=task.spawned_by_event)

    # ----------------------------------------------------------- world processes

    def _day_index(self, t: float) -> int:
        return int(t // MINUTES_PER_DAY)

    def daily_process(self):
        """Start-of-day: roster absences, schedule departures, reset fatigue partly."""
        while True:
            day_idx = self._day_index(self.env.now)
            day = self.scenario.start_date + timedelta(days=day_idx)

            decay_fatigue(self.workers)
            # Publish the planned roster first, then discover who did not turn up:
            # the schedule is the plan, absence is the deviation from it.
            self.roster.build_day(self._roster(), day_idx)
            absent = roll_absences(self.workers + self.activated_reserves, self.scenario, self.rng)
            for w in self._roster():
                self.roster.mark_absent(w.id, day_idx, w.absent_today)
            self.kpis.absent_workers = absent
            self.log.log_event(self.env.now, "roster",
                               f"Day {day_idx + 1} ({day.isoformat()}): {absent} of "
                               f"{len(self.workers)} staff absent")

            self.departure_schedule = self.demand.departures_for_day(day)
            for dep in self.departure_schedule:
                self.env.process(self.train_departure_process(dep, day_idx))

            yield self.env.timeout(MINUTES_PER_DAY)

    def train_departure_process(self, departure: TrainDeparture, day_idx: int):
        """A train departure: passengers, turnaround work, and WL-driven counter load."""
        target = day_idx * MINUTES_PER_DAY + departure.departure_hour * 60.0
        delay = max(0.0, target - self.env.now)
        yield self.env.timeout(delay)

        onboard = departure.total_onboard
        waitlisted = departure.total_waitlist
        self.kpis.passengers_served += onboard
        self.kpis.waitlisted_passengers += waitlisted

        self.log.log_event(
            self.env.now, "train_departure",
            f"{departure.name} ({departure.train_id}) departs {departure.origin}: "
            f"{onboard} onboard, {waitlisted} waitlisted",
            departure.origin,
            {"train_id": departure.train_id, "onboard": onboard, "waitlist": waitlisted,
             "wl_pressure": round(departure.mean_wl_pressure, 3)},
        )

        # Every departure generates a turnaround case. Load scales with occupancy,
        # so a fuller train is genuinely more work to clean and check.
        load_scale = 1.0 + 0.6 * min(2.0, onboard / 400.0)
        self._spawn_case("turnaround", departure.origin, duration_scale=load_scale)

        # Waitlist pressure is unmet demand showing up at the counter: people
        # enquiring, rebooking, seeking alternatives. This is the causal link from
        # the booking model to commercial workload.
        wl_pressure = departure.mean_wl_pressure
        if wl_pressure > 0.15:
            extra = int(min(4, np.ceil(wl_pressure * 3)))
            for _ in range(extra):
                self._spawn_case("ticketing", departure.origin, priority=3)
            self.log.log_event(
                self.env.now, "wl_pressure",
                f"Waitlist pressure {wl_pressure:.2f} on {departure.train_id} generates "
                f"{extra} extra counter cases at {departure.origin}",
                departure.origin, {"wl_pressure": round(wl_pressure, 3)},
            )

    def footfall_process(self):
        """Hourly station pressure -> ticketing workload, and crowd cases past a threshold."""
        while True:
            hour = (self.env.now / 60.0) % 24.0
            for loc_id in self.domain.locations:
                counts = self.demand.footfall(loc_id, hour)
                pressure = counts["pressure"]
                # One counter case represents a batch of PASSENGERS_PER_COUNTER_CASE
                # transactions, so counter workload scales with realised demand
                # rather than being a fixed rate. [SYNTH divisor]
                cases = int((counts["unreserved_tickets"] + counts["platform_tickets"])
                            / PASSENGERS_PER_COUNTER_CASE)
                for _ in range(min(cases, 14)):
                    self._spawn_case("ticketing", loc_id)
                if pressure > 1.15:
                    self._spawn_case("crowd_response", loc_id, priority=2)
                    self.log.log_event(
                        self.env.now, "crowd_surge",
                        f"Crowding at {self.domain.locations[loc_id].name} "
                        f"(pressure {pressure:.2f}, footfall {counts['footfall']})",
                        loc_id, {"pressure": round(pressure, 3), "footfall": counts["footfall"]},
                    )
            yield self.env.timeout(60.0)

    def suburban_process(self):
        """Suburban services generate turnaround work without individual bookings.

        Modelled in aggregate: only a fraction of arrivals need attention, but at a
        major terminus that fraction is still substantial, and it is what keeps
        cleaning and platform staff genuinely occupied between express departures.
        """
        rates = self.domain.suburban["arrivals_per_hour_by_tier"]
        while True:
            hour = (self.env.now / 60.0) % 24.0
            profile = self.demand.hourly_profile[int(hour) % 24]
            for loc_id, loc in self.domain.locations.items():
                arrivals = rates[loc.tier] * profile
                needing_work = self.rng.demand_unreserved.binomial(
                    max(0, int(arrivals)), SUBURBAN_TURNAROUND_FRACTION)
                for _ in range(int(needing_work)):
                    self._spawn_case("turnaround", loc_id, priority=3, duration_scale=0.55)
            yield self.env.timeout(60.0)

    def maintenance_programme_process(self):
        """Scheduled preventive maintenance.

        Without this, maintenance staff would be idle until something breaks, which
        is both unrealistic and removes the interesting trade-off: routine work has
        slack and can be deferred, emergency work cannot.
        """
        while True:
            for loc_id, loc in self.domain.locations.items():
                n_assets = len(self.assets_by_location.get(loc_id, []))
                inspections = self.rng.process_duration.binomial(n_assets, DAILY_INSPECTION_RATE)
                for _ in range(int(inspections)):
                    self._spawn_case("maintenance", loc_id, priority=3)
            yield self.env.timeout(MINUTES_PER_DAY)

    def disruption_process(self):
        """Generate disruptions in hourly windows and turn them into real work."""
        while True:
            t0 = float(self.env.now)
            t1 = t0 + 60.0
            self.disruptions.step_regime(60.0)
            new_regime_demand = self.disruptions.demand_multiplier()
            self.demand.regime_multiplier = new_regime_demand

            for d in self.disruptions.sample_window(t0, t1):
                yield self.env.timeout(max(0.0, d.time - self.env.now))
                self._apply_disruption(d)
            remaining = t1 - self.env.now
            if remaining > 0:
                yield self.env.timeout(remaining)

    def _apply_disruption(self, d: Disruption) -> None:
        self.kpis.disruptions += 1
        self.costs.add_disruption()
        if d.caused_by:
            self.kpis.cascade_disruptions += 1
        self.disruptions.excite_neighbours(d)
        loc_name = self.domain.locations[d.location_id].name
        cascade_note = f" (cascade from {d.caused_by})" if d.caused_by else ""

        if d.kind == "asset_failure":
            candidates = [a for a in self.assets_by_location.get(d.location_id, []) if not a.failed]
            if not candidates:
                return
            asset = candidates[int(self.rng.disruption_base.integers(len(candidates)))]
            asset.failed = True
            asset.health = 0.0
            skill = self.domain.assets["asset_skill_map"][asset.asset_type]
            self.log.log_event(
                self.env.now, "asset_failure",
                f"{asset.asset_type} failure at {loc_name}{cascade_note} "
                f"— needs {skill}, {d.severity_minutes:.0f} min of work",
                d.location_id, {"asset_id": asset.id, "asset_type": asset.asset_type,
                                "severity": round(d.severity_minutes, 1)},
                caused_by=d.caused_by, event_id=d.id,
            )
            scale = d.severity_minutes / 55.0
            task = self._spawn_case("emergency_repair", d.location_id, priority=1,
                                    caused_by=d.id, duration_scale=max(0.4, scale))
            task.deadline = self.env.now + 90.0

        elif d.kind == "train_delay":
            self.log.log_event(
                self.env.now, "train_delay",
                f"Delay at {loc_name}{cascade_note}: {d.severity_minutes:.0f} min "
                f"— passengers accumulate on platform",
                d.location_id, {"delay_minutes": round(d.severity_minutes, 1)},
                caused_by=d.caused_by, event_id=d.id,
            )
            # Delay -> accumulation -> crowd work and extra cleaning. The cascade
            # the brief asks for, produced by mechanics rather than a script.
            self._spawn_case("crowd_response", d.location_id, priority=2, caused_by=d.id,
                             duration_scale=1.0 + d.severity_minutes / 90.0)
            if d.severity_minutes > 40:
                self._spawn_case("turnaround", d.location_id, priority=2, caused_by=d.id)

        elif d.kind == "demand_spike":
            self.log.log_event(
                self.env.now, "demand_spike",
                f"Demand spike at {loc_name}{cascade_note}",
                d.location_id, {"severity": round(d.severity_minutes, 1)},
                caused_by=d.caused_by, event_id=d.id,
            )
            for _ in range(2):
                self._spawn_case("ticketing", d.location_id, priority=2, caused_by=d.id)
            self._spawn_case("crowd_response", d.location_id, priority=2, caused_by=d.id)

    def monitor_process(self):
        """Sample KPIs and (optionally) trace snapshots on a fixed cadence."""
        while True:
            active = self.on_duty()
            busy = [w for w in active if w.current_task_id is not None]
            utilisation = len(busy) / len(active) if active else 0.0
            self.kpis.utilisation_samples.append(utilisation)
            self.costs.add_idle(len(active) - len(busy), 15.0)
            self.kpis.queue_length_samples.append(len(self.pending_tasks))
            self.kpis.backlog_samples.append(
                sum(1 for t in self.pending_tasks if t.deadline < self.env.now)
            )
            if self.collect_trace:
                self._snapshot(utilisation)
            yield self.env.timeout(15.0)

    def sla_monitor_process(self):
        """Tasks abandoned long after their deadline are recorded as breached.

        The grace period matters: cull too eagerly and the backlog is invisible to
        both the metrics and the agent, because overdue work is removed before it
        can be observed accumulating.
        """
        while True:
            now = float(self.env.now)
            for task in list(self.pending_tasks):
                if task.deadline < now - ABANDON_GRACE_MINUTES:
                    self.pending_tasks.remove(task)
                    task.status = TaskStatus.BREACHED
                    self.kpis.tasks_breached += 1
                    # Record *why* it was never served: without this, a breach with
                    # idle staff on the roster is indistinguishable from overload.
                    self.kpis.abandon_reasons[self._blocking_reason(task)] += 1
                    self.kpis.per_department_breached[task.department_id] = (
                        self.kpis.per_department_breached.get(task.department_id, 0) + 1
                    )
                    self.station_queues[task.location_id] = max(
                        0, self.station_queues.get(task.location_id, 0) - 1)
                    self.log.log_activity(LogRow(
                        case_id=task.case_id, activity=task.activity_id, timestamp=now,
                        lifecycle="breach", process_id=task.process_id,
                        location_id=task.location_id, department_id=task.department_id,
                        resource_id=None, role_id=None, team_id=None,
                        queue_time=now - task.created_at, processing_time=0.0,
                        status=task.status.value, priority=task.priority,
                        outcome="sla_breach", spawned_by_event=task.spawned_by_event,
                    ))
            yield self.env.timeout(15.0)

    # ------------------------------------------------------------- work execution

    @property
    def now(self) -> float:
        """Current simulation time in minutes (the shared engine-protocol accessor)."""
        return float(self.env.now)

    def observed_feedback(self) -> dict:
        """Feedback instruments as the agent may see them at the current time.

        Surveys are filtered on delivery time and performance on closed review
        periods, so nothing here reveals the present or the future.
        """
        summary = self.feedback.observed_summary(float(self.env.now))
        summary.update(self.performance.department_summary())
        summary["cost_total"] = self.costs.total
        return summary

    def outcome_summary(self) -> dict:
        """Full outcome bundle for reporting and validation (NOT for the agent)."""
        out = {"costs": self.costs.summary(),
               "feedback_observed": self.feedback.observed_summary(float(self.env.now)),
               "feedback_true": self.feedback.true_summary(),
               "performance": self.performance.department_summary(),
               "latent": self.true_state.summary()}
        return out

    def observed_pressure(self, hour: float) -> float:
        """Observable demand pressure at the major stations.

        Part of o_t, not s_t: the agent sees an aggregate crowding signal, never
        the latent per-train demand intensity that produced it.
        """
        majors = [lid for lid, loc in self.domain.locations.items() if loc.tier == 1][:4]
        if not majors:
            majors = list(self.domain.locations)[:4]
        return float(np.mean([self.demand.station_pressure(lid, hour) for lid in majors]))

    def _roster(self) -> list[Worker]:
        return self.workers + self.activated_reserves

    def on_duty(self) -> list[Worker]:
        """Staff actually available to work right now: present, and on shift.

        Utilisation must be measured against this, not the whole establishment —
        two-thirds of the roster is off-shift at any moment, and dividing by all of
        them makes a saturated system look idle.
        """
        hour = (float(self.env.now) / 60.0) % 24.0
        return [
            w for w in self._roster()
            if not w.absent_today and (w.on_overtime or self.domain.shifts[w.shift_id].covers(hour))
        ]

    def _blocking_reason(self, task: Task) -> str:
        """The most common reason no worker could take this task."""
        now_hour = (float(self.env.now) / 60.0) % 24.0
        reasons: dict[str, int] = {}
        for w in self._roster():
            res = is_eligible(w, task, now_hour, self.domain)
            if res.eligible:
                return "eligible_but_unserved"
            key = res.reason.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
        return max(reasons, key=reasons.get) if reasons else "unknown"

    def rank_candidates(self, task: Task) -> list[Worker]:
        """Eligible workers, canonically ordered, truncated to the action slots.

        Ordering must be deterministic given state, or the meaning of "action 3"
        would drift between steps and the policy could not learn anything stable.
        """
        now_hour = (self.env.now / 60.0) % 24.0
        eligible = [
            w for w in self._roster()
            if is_eligible(w, task, now_hour, self.domain).eligible
        ]

        def sort_key(w: Worker):
            travel = self.domain.travel_minutes(w.location_id or w.home_location_id, task.location_id)
            # Proximity dominates, then skill. Ranking skill first would send a
            # specialist across the network for a short local job and blow the SLA
            # on travel alone — the composite score keeps "action 0" sensible.
            score = travel - 25.0 * w.proficiency(task.required_skill) + 8.0 * w.fatigue
            return (score, w.hourly_cost, w.id)

        eligible.sort(key=sort_key)
        return eligible[:CANDIDATE_SLOTS]

    def execute_task(self, task: Task, worker: Worker):
        """Assign and run a task to completion, then advance its case."""
        now = float(self.env.now)
        travel = self.domain.travel_minutes(worker.location_id or worker.home_location_id,
                                            task.location_id)
        base = lognormal_duration(self.rng.process_duration, task.duration_median_min,
                                  task.duration_sigma)
        # Proficiency above the floor speeds the job up; fatigue slows it down.
        skill_factor = 1.0 / max(0.35, worker.proficiency(task.required_skill))
        duration = base * skill_factor * 0.6 * worker.speed_multiplier()
        # Nominal pace: a competent (0.8-proficiency), unfatigued worker.
        expected_minutes = task.duration_median_min * 0.6 / 0.8
        fatigue_at_start = worker.fatigue

        task.started_at = now + travel
        worker.current_task_id = task.id      # already claimed in apply_dispatch; idempotent

        self.log.log_activity(LogRow(
            case_id=task.case_id, activity=task.activity_id, timestamp=now,
            lifecycle="start", process_id=task.process_id, location_id=task.location_id,
            department_id=task.department_id, resource_id=worker.id, role_id=worker.role_id,
            team_id=worker.team_id, queue_time=now - task.created_at, processing_time=0.0,
            status=task.status.value, priority=task.priority, outcome="assigned",
            spawned_by_event=task.spawned_by_event,
        ))

        yield self.env.timeout(travel + duration)

        end = float(self.env.now)
        task.completed_at = end
        breached = end > task.deadline
        task.status = TaskStatus.BREACHED if breached else TaskStatus.COMPLETED

        worker.current_task_id = None
        worker.location_id = task.location_id
        worker.completed_tasks += 1
        worker.busy_minutes += duration + travel
        accrue_fatigue(worker, duration + travel)
        entry = self.roster.entry_for(worker.id, end)
        if entry is not None:
            entry.actual_busy_minutes += duration + travel

        cost_hours = (duration + travel) / 60.0
        if worker.on_overtime:
            mult = float(self.domain.workforce["overtime_cost_multiplier"])
            overtime_amount = worker.hourly_cost * cost_hours * mult
            self.kpis.overtime_cost += overtime_amount
            self.kpis.overtime_minutes += duration + travel
            worker.overtime_minutes += duration + travel
            self.costs.add_labour(overtime_amount, overtime=True)
        else:
            self.costs.add_labour(worker.hourly_cost * cost_hours)
        self.kpis.labour_cost += worker.hourly_cost * cost_hours

        # Outcome signals: performance is attributed here, where the conditions the
        # worker actually faced (difficulty, fatigue, overtime) are still known.
        # Travel is excluded: it reflects where the dispatcher sent the worker,
        # not how well the worker performed.
        self.performance.record_task(worker, task, expected_minutes, duration,
                                     on_time=not breached, fatigue_at_start=fatigue_at_start,
                                     now=end)
        if task.rework_count > 0:
            self.costs.add_rework()
        if breached:
            self.costs.add_breach()

        # Customer feedback: only passenger-facing work produces an experience, and
        # only a sampled, delayed fraction of it ever comes back as a survey.
        if task.process_id in CUSTOMER_FACING_PROCESSES:
            self._emit_customer_experience(task, worker, breached, end)

        if breached:
            self.kpis.tasks_breached += 1
            self.kpis.per_department_breached[task.department_id] = (
                self.kpis.per_department_breached.get(task.department_id, 0) + 1)
        else:
            self.kpis.tasks_completed += 1
            self.kpis.per_department_completed[task.department_id] = (
                self.kpis.per_department_completed.get(task.department_id, 0) + 1)
        self.kpis.queue_time_total += task.queue_time
        self.kpis.processing_time_total += task.processing_time

        self.log.log_activity(LogRow(
            case_id=task.case_id, activity=task.activity_id, timestamp=end,
            lifecycle="complete", process_id=task.process_id, location_id=task.location_id,
            department_id=task.department_id, resource_id=worker.id, role_id=worker.role_id,
            team_id=worker.team_id, queue_time=task.queue_time,
            processing_time=task.processing_time, status=task.status.value,
            priority=task.priority, outcome="breached" if breached else "completed",
            spawned_by_event=task.spawned_by_event,
        ))

        self._advance_case(task)

    # ---------------------------------------------------------- feedback layer

    def _emit_customer_experience(self, task: Task, worker: Worker, breached: bool,
                                  now: float) -> None:
        """Compute latent experience and (rarely) sample a survey response.

        The latent value is stored in true_state and never exposed to the agent;
        only a delayed, sampled survey can surface any of it.
        """
        hour = (now / 60.0) % 24.0
        crowding = self.demand.station_pressure(task.location_id, hour)
        active_delay = self._active_delay_minutes(task.location_id, now)
        sla = self.templates[task.process_id].sla_minutes

        experience = self.experience.latent_experience(
            task=task, sla_minutes=sla, crowding=crowding,
            active_delay_minutes=active_delay,
            server_proficiency=worker.proficiency(task.required_skill),
            server_fatigue=worker.fatigue, breached=breached,
        )
        self.true_state.record_experience(task.location_id, now, experience)
        self.feedback.maybe_survey(task.case_id, task.location_id, task.process_id,
                                   now, experience)

    def _active_delay_minutes(self, location_id: str, now: float) -> float:
        """Delay currently affecting this station, from disruptions in the last hour."""
        total = 0.0
        for d in self.disruptions.history[-40:]:
            if d.kind == "train_delay" and d.location_id == location_id and 0 <= now - d.time < 60:
                total += d.severity_minutes
        return total

    def shift_review_process(self):
        """Cut performance records at shift boundaries.

        Performance is periodic on purpose: a live per-task quality readout is not
        something a real operator has, and exposing one would let the agent learn
        against information it could never obtain.
        """
        boundaries = sorted(s.start_hour for s in self.domain.shifts.values())
        while True:
            hour = (float(self.env.now) / 60.0) % 24.0
            next_boundary = next((b for b in boundaries if b > hour), boundaries[0] + 24.0)
            yield self.env.timeout(max(1.0, (next_boundary - hour) * 60.0))
            records = self.performance.cut_period(self._roster(), float(self.env.now))
            if records:
                self.log.log_event(
                    self.env.now, "performance_review",
                    f"Shift review: {len(records)} performance records cut "
                    f"(mean index {np.mean([r.performance_index for r in records]):.2f})",
                )

    # ------------------------------------------------------------ trace/snapshot

    def _snapshot(self, utilisation: float) -> None:
        now = float(self.env.now)
        roster = self._roster()
        on_duty = self.on_duty()
        by_dept: dict[str, dict] = {}
        for dept in self.domain.departments:
            active = [w for w in on_duty if w.department_id == dept]
            busy = [w for w in active if w.current_task_id is not None]
            queued = [t for t in self.pending_tasks if t.department_id == dept]
            by_dept[dept] = {
                "available": len(active) - len(busy),
                "busy": len(busy),
                "absent": sum(1 for w in roster if w.department_id == dept and w.absent_today),
                "queue": len(queued),
                "overdue": sum(1 for t in queued if t.deadline < now),
            }
        by_location = {
            loc: {
                "queue": sum(1 for t in self.pending_tasks if t.location_id == loc),
                "busy": sum(1 for w in roster if w.location_id == loc and w.current_task_id),
                "staff": sum(1 for w in roster if w.location_id == loc and not w.absent_today),
            }
            for loc in self.domain.locations
        }
        fb = self.observed_feedback()
        self.trace_snapshots.append({
            "t": round(now, 1),
            "csat": round(fb["csat_mean"], 2) if fb["has_data"] else None,
            "nps": round(fb["nps_score"], 1) if fb["has_data"] else None,
            "surveys": fb["n_responses"],
            "cost_total": round(self.costs.total, 1),
            "performance": round(fb.get("mean_performance", 0.0), 3),
            "utilisation": round(utilisation, 3),
            "queue": len(self.pending_tasks),
            "overdue": sum(1 for t in self.pending_tasks if t.deadline < now),
            "completed": self.kpis.tasks_completed,
            "breached": self.kpis.tasks_breached,
            "labour_cost": round(self.kpis.labour_cost, 1),
            "overtime_minutes": round(self.kpis.overtime_minutes, 1),
            "regime": self.disruptions.regime,
            "departments": by_dept,
            "locations": by_location,
        })

    # ---------------------------------------------------------------- lifecycle

    def start_world_processes(self) -> None:
        self.env.process(self.daily_process())
        self.env.process(self.footfall_process())
        self.env.process(self.suburban_process())
        self.env.process(self.maintenance_programme_process())
        self.env.process(self.disruption_process())
        self.env.process(self.monitor_process())
        self.env.process(self.sla_monitor_process())
        self.env.process(self.shift_review_process())

    def assignable_tasks(self) -> list[Task]:
        """Pending tasks that have at least one eligible worker right now."""
        return [t for t in sorted(self.pending_tasks, key=lambda t: (t.priority, t.deadline))
                if self.rank_candidates(t)]

    def next_epoch(self) -> DecisionEpoch | None:
        """Advance the clock until a decision is genuinely required, or the episode ends.

        Only *actionable* situations become epochs: a pending task with at least one
        eligible worker, or a shift boundary. Everything else the world handles by
        itself, which keeps the agent's step count proportional to real decisions.
        """
        while self.env.now < self.horizon:
            # Shift boundary -> staffing epoch (the slower level of the hierarchy).
            now = float(self.env.now)
            hour = (now / 60.0) % 24.0
            for shift in self.domain.shifts.values():
                boundary = shift.start_hour
                if abs(hour - boundary) < 0.13 and now - getattr(self, "_last_staffing_epoch", -1e9) > 120:
                    self._last_staffing_epoch = now
                    return DecisionEpoch(kind=EpochType.STAFFING, time=now)

            ready = self.assignable_tasks()
            if ready:
                task = ready[0]
                return DecisionEpoch(kind=EpochType.DISPATCH, time=now, task=task,
                                     candidates=self.rank_candidates(task))

            # Nothing to decide: let the world run to the next event.
            try:
                self.env.step()
            except simpy.core.EmptySchedule:
                break
        self.done = True
        return None

    def apply_dispatch(self, task: Task, worker: Worker | None) -> dict:
        """Apply a dispatch action. `None` means defer this task for now."""
        if worker is None:
            task.deferrals += 1
            task.status = TaskStatus.QUEUED
            self.kpis.tasks_deferred += 1
            self.log.log_activity(LogRow(
                case_id=task.case_id, activity=task.activity_id, timestamp=float(self.env.now),
                lifecycle="defer", process_id=task.process_id, location_id=task.location_id,
                department_id=task.department_id, resource_id=None, role_id=None, team_id=None,
                queue_time=float(self.env.now) - task.created_at, processing_time=0.0,
                status=task.status.value, priority=task.priority, outcome="deferred",
                spawned_by_event=task.spawned_by_event,
            ))
            # Let the clock move so a deferral cannot be repeated at the same instant.
            try:
                self.env.step()
            except simpy.core.EmptySchedule:
                self.done = True
            return {"deferred": True, "task_id": task.id}

        self.pending_tasks.remove(task)
        self.station_queues[task.location_id] = max(0, self.station_queues.get(task.location_id, 0) - 1)
        # Claim the worker synchronously. `env.process` only *schedules* the
        # generator; it does not run until the scheduler advances, so without
        # claiming here the next decision epoch would still see this worker as
        # free and could assign them a second task at the same instant.
        worker.current_task_id = task.id
        task.assigned_worker_id = worker.id
        task.status = TaskStatus.IN_PROGRESS
        self.env.process(self.execute_task(task, worker))
        return {"deferred": False, "task_id": task.id, "worker_id": worker.id}

    def apply_staffing(self, department_id: str, action: str) -> dict:
        """Apply a shift-level staffing action: the slower level of the hierarchy."""
        result = {"department": department_id, "action": action, "changed": 0}
        if action == "activate_reserve" and self.scenario.workforce.get("reserve_available", True):
            available = [w for w in self.reserve_pool if w.department_id == department_id]
            if available:
                w = available[0]
                self.reserve_pool.remove(w)
                self.activated_reserves.append(w)
                self.costs.add_reserve_activation()
                self.roster.build_day([w], self._day_index(self.env.now))
                result["changed"] = 1
                self.log.log_event(self.env.now, "staffing",
                                   f"Reserve worker {w.id} activated for {department_id}",
                                   w.home_location_id, {"worker_id": w.id})
        elif action == "authorise_overtime" and self.scenario.workforce.get("overtime_allowed", True):
            self.overtime_authorised[department_id] = True
            changed = 0
            for w in self._roster():
                if w.department_id == department_id and not w.absent_today and not w.on_overtime:
                    cap = float(self.domain.workforce["max_overtime_minutes_per_shift"])
                    if w.overtime_minutes < cap:
                        w.on_overtime = True
                        self.roster.authorise_overtime(w.id, self._day_index(self.env.now), cap)
                        changed += 1
            result["changed"] = changed
            self.log.log_event(self.env.now, "staffing",
                               f"Overtime authorised for {department_id} ({changed} staff)")
        elif action == "release_overtime":
            self.overtime_authorised[department_id] = False
            changed = 0
            for w in self._roster():
                if w.department_id == department_id and w.on_overtime:
                    w.on_overtime = False
                    changed += 1
            result["changed"] = changed
            self.log.log_event(self.env.now, "staffing", f"Overtime released for {department_id}")
        # "hold" is a real, legal choice: doing nothing has a cost and is sometimes right.

        try:
            self.env.step()
        except simpy.core.EmptySchedule:
            self.done = True
        return result

    def run_to_completion(self) -> None:
        """Run the world with no policy involvement (used by validation/dataset builds)."""
        self.start_world_processes()
        while self.env.now < self.horizon:
            epoch = self.next_epoch()
            if epoch is None:
                break
            if epoch.kind == EpochType.DISPATCH and epoch.candidates:
                self.apply_dispatch(epoch.task, epoch.candidates[0])
            else:
                self.apply_staffing(list(self.domain.departments)[0], "hold")
