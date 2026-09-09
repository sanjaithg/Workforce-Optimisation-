"""Workforce generation and eligibility.

Workers are heterogeneous by construction: proficiency, experience, absence
propensity and cross-training are all drawn per individual. Two workers in the
same role are not substitutes, which is what makes allocation a real decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.core.config import DomainConfig, ScenarioConfig
from simulation.distributions.library import beta_binomial_rate
from simulation.entities.core import Task, Worker


def shift_weights(domain: DomainConfig, scenario: ScenarioConfig) -> dict[str, float]:
    """Allocate staff to shifts in proportion to the demand each shift covers.

    A uniform split would put a third of the establishment on nights, when demand
    is a fraction of peak — leaving night staff idle while the morning peak
    breaches its SLAs. Real rosters are demand-weighted, and without this the
    allocation problem is dominated by an artefact of the roster rather than by
    the decisions we want to study.
    """
    profile = scenario.demand["hourly_profile"]
    weights: dict[str, float] = {}
    for shift_id, shift in domain.shifts.items():
        hours = [int((shift.start_hour + h) % 24) for h in range(int(shift.duration_hours))]
        weights[shift_id] = float(np.mean([profile[h] for h in hours]))
    total = sum(weights.values()) or 1.0
    # Floor each shift at 12% of the establishment: even a quiet night needs cover.
    floored = {k: max(0.12, v / total) for k, v in weights.items()}
    norm = sum(floored.values())
    return {k: v / norm for k, v in floored.items()}


def build_workforce(domain: DomainConfig, scenario: ScenarioConfig, rng_registry) -> list[Worker]:
    rng = rng_registry.workforce_population
    wf_cfg = domain.workforce
    headcount_cfg = wf_cfg["headcount_per_location_by_tier"]
    multiplier = float(scenario.workforce["multiplier"])
    exp_lo, exp_hi = wf_cfg["experience_years_range"]
    cross = domain.cross_training
    all_skills = list(domain.skills)

    roles_by_dept: dict[str, list] = {}
    for role in domain.roles.values():
        roles_by_dept.setdefault(role.department_id, []).append(role)

    # Role proportions per department, so specialist supply matches the work mix.
    role_mix = domain.raw.get("role_mix", {})
    role_probs: dict[str, np.ndarray] = {}
    for dept_id, roles in roles_by_dept.items():
        mix = role_mix.get(dept_id, {})
        weights = np.array([float(mix.get(r.id, 1.0)) for r in roles], dtype=float)
        role_probs[dept_id] = weights / weights.sum()

    shift_ids = list(domain.shifts)
    weights = shift_weights(domain, scenario)
    shift_p = np.array([weights[s] for s in shift_ids], dtype=float)

    workers: list[Worker] = []
    counter = 0
    for loc in domain.locations.values():
        per_dept = headcount_cfg[loc.tier]
        for dept_id, headcount in per_dept.items():
            roles = roles_by_dept.get(dept_id, [])
            if not roles:
                continue
            n = int(round(headcount * multiplier))
            for _ in range(n):
                role = roles[int(rng.choice(len(roles), p=role_probs[dept_id]))]
                counter += 1
                skills = {
                    skill_id: float(np.clip(rng.normal(level, 0.08), 0.05, 1.0))
                    for skill_id, level in role.skills.items()
                }
                # Cross-training gives some workers a secondary skill outside
                # their role, creating genuine multi-skill reassignment options.
                if rng.random() < float(cross["probability"]):
                    lo, hi = cross["secondary_proficiency_range"]
                    extra = all_skills[int(rng.integers(len(all_skills)))]
                    skills.setdefault(extra, float(rng.uniform(lo, hi)))

                shift_id = shift_ids[int(rng.choice(len(shift_ids), p=shift_p))]
                workers.append(
                    Worker(
                        id=f"W{counter:05d}",
                        role_id=role.id,
                        department_id=dept_id,
                        team_id=f"{loc.id}-{dept_id}",
                        home_location_id=loc.id,
                        shift_id=shift_id,
                        skills=skills,
                        qualifications=set(role.qualifications),
                        hourly_cost=float(role.hourly_cost * rng.uniform(0.92, 1.12)),
                        experience_years=float(rng.uniform(exp_lo, exp_hi)),
                        absence_propensity=beta_binomial_rate(
                            rng,
                            float(wf_cfg["absence_base_rate"]),
                            float(wf_cfg["absence_concentration"]),
                        ),
                    )
                )
    return workers


def build_reserve_pool(domain: DomainConfig, workers: list[Worker], rng_registry) -> list[Worker]:
    """Off-roster staff the agent can activate at a cost. Drawn from the same
    generator so reserves are ordinary workers, not superhuman fill-ins."""
    rng = rng_registry.workforce_population
    fraction = float(domain.workforce["reserve_pool_fraction"])
    n = int(round(len(workers) * fraction))
    reserve = []
    for i in range(n):
        template = workers[int(rng.integers(len(workers)))]
        clone = Worker(
            id=f"R{i + 1:04d}",
            role_id=template.role_id,
            department_id=template.department_id,
            team_id=f"RESERVE-{template.department_id}",
            home_location_id=template.home_location_id,
            shift_id=template.shift_id,
            skills=dict(template.skills),
            qualifications=set(template.qualifications),
            hourly_cost=template.hourly_cost * 1.15,   # reserves cost more to call in
            experience_years=float(rng.uniform(0.5, 12.0)),
            absence_propensity=template.absence_propensity,
        )
        reserve.append(clone)
    return reserve


@dataclass
class EligibilityResult:
    eligible: bool
    reason: str = ""


def is_eligible(worker: Worker, task: Task, now_hour: float, domain: DomainConfig,
                shift_enforced: bool = True, max_travel_minutes: float | None = None) -> EligibilityResult:
    """Hard feasibility check. Everything here is a constraint, not a preference.

    These become the action mask: an ineligible worker is not merely a bad option
    that the agent should learn to avoid via reward, it is an illegal one that the
    agent is never allowed to pick.
    """
    if worker.absent_today:
        return EligibilityResult(False, "absent")
    if worker.current_task_id is not None:
        return EligibilityResult(False, "busy")
    if not task.required_qualifications.issubset(worker.qualifications):
        missing = task.required_qualifications - worker.qualifications
        return EligibilityResult(False, f"missing_qualification:{','.join(sorted(missing))}")
    if worker.proficiency(task.required_skill) < task.min_proficiency:
        return EligibilityResult(False, "insufficient_skill")
    if shift_enforced and not worker.on_overtime:
        shift = domain.shifts[worker.shift_id]
        if not shift.covers(now_hour % 24.0):
            return EligibilityResult(False, "off_shift")
    # Location constraint: nobody is sent across the network for a short local job.
    limit = max_travel_minutes if max_travel_minutes is not None else float(
        domain.workforce.get("max_travel_minutes", 45.0))
    if domain.travel_minutes(worker.location_id or worker.home_location_id, task.location_id) > limit:
        return EligibilityResult(False, "too_far")
    return EligibilityResult(True)


def eligible_workers(workers: list[Worker], task: Task, now_hour: float,
                     domain: DomainConfig) -> list[Worker]:
    return [w for w in workers if is_eligible(w, task, now_hour, domain).eligible]


def roll_absences(workers: list[Worker], scenario: ScenarioConfig, rng_registry) -> int:
    """Daily absence draw. Each worker has their own propensity (Beta-drawn), so
    the aggregate absence count is overdispersed relative to a shared-p Binomial."""
    rng = rng_registry.workforce_absence
    mult = float(scenario.workforce["absence_rate_multiplier"])
    absent = 0
    for w in workers:
        # Fatigue raises absence risk: overworked staff call in sick more.
        p = min(0.95, w.absence_propensity * mult * (1.0 + 0.8 * w.fatigue))
        w.absent_today = bool(rng.random() < p)
        w.on_overtime = False
        absent += int(w.absent_today)
    return absent


def decay_fatigue(workers: list[Worker], rest_factor: float = 0.45) -> None:
    for w in workers:
        w.fatigue = max(0.0, w.fatigue * (1.0 - rest_factor))


def accrue_fatigue(worker: Worker, minutes_worked: float, shift_minutes: float = 480.0) -> None:
    """Fatigue accumulates with time on task and is worse under overtime."""
    load = minutes_worked / max(shift_minutes, 1.0)
    penalty = 1.6 if worker.on_overtime else 1.0
    worker.fatigue = float(min(1.0, worker.fatigue + 0.55 * load * penalty))
