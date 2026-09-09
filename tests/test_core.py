"""Tests for the generative core and the RL environment contracts."""

from __future__ import annotations

import numpy as np
import pytest

from simulation.core.config import available_scenarios, load_domain, load_scenario
from simulation.core.rng import RNGRegistry
from simulation.demand.model import DemandModel
from simulation.distributions.library import (
    AR1,
    HawkesProcess,
    beta_binomial_rate,
    heavy_tailed_severity,
    lognormal_duration,
    negative_binomial,
)
from simulation.workforce.population import build_workforce, is_eligible


# --------------------------------------------------------------- reproducibility

def test_named_streams_are_independent():
    """Consuming one stream must not shift another, or paired-seed comparison breaks."""
    a = RNGRegistry(42)
    b = RNGRegistry(42)
    for _ in range(100):
        a.demand_latent.random()
    assert a.workforce_population.random() == b.workforce_population.random()


def test_same_seed_same_world():
    domain = load_domain()
    scenario = load_scenario("normal_weekday")
    w1 = build_workforce(domain, scenario, RNGRegistry(7))
    w2 = build_workforce(domain, scenario, RNGRegistry(7))
    assert [w.id for w in w1] == [w.id for w in w2]
    assert [round(w.hourly_cost, 6) for w in w1] == [round(w.hourly_cost, 6) for w in w2]


def test_different_seeds_differ():
    domain = load_domain()
    scenario = load_scenario("normal_weekday")
    w1 = build_workforce(domain, scenario, RNGRegistry(1))
    w2 = build_workforce(domain, scenario, RNGRegistry(2))
    assert [w.hourly_cost for w in w1] != [w.hourly_cost for w in w2]


# ------------------------------------------------------------------ distributions

def test_negative_binomial_is_overdispersed():
    rng = np.random.default_rng(0)
    draws = [negative_binomial(rng, 20.0, 5.0) for _ in range(6000)]
    assert np.var(draws) > np.mean(draws)


def test_negative_binomial_approaches_poisson_for_large_k():
    rng = np.random.default_rng(0)
    draws = [negative_binomial(rng, 20.0, 1e6) for _ in range(4000)]
    ratio = np.var(draws) / np.mean(draws)
    assert 0.85 < ratio < 1.15


def test_lognormal_durations_positive_and_skewed():
    rng = np.random.default_rng(0)
    draws = [lognormal_duration(rng, 30.0, 0.5) for _ in range(4000)]
    assert min(draws) > 0
    assert np.mean(draws) > np.median(draws)


def test_heavy_tail_produces_outliers():
    rng = np.random.default_rng(0)
    draws = [heavy_tailed_severity(rng, 35.0, 0.6, 0.05, 1.6) for _ in range(8000)]
    assert max(draws) > 10 * np.median(draws)


def test_beta_binomial_rate_in_range():
    rng = np.random.default_rng(0)
    rates = [beta_binomial_rate(rng, 0.06, 12.0) for _ in range(1000)]
    assert all(0.0 < r < 1.0 for r in rates)
    assert 0.03 < np.mean(rates) < 0.10


def test_ar1_is_temporally_correlated():
    rng = np.random.default_rng(0)
    ar = AR1(phi=0.85, sigma=0.1)
    series = [ar.step(rng) for _ in range(3000)]
    lag1 = np.corrcoef(series[:-1], series[1:])[0, 1]
    assert lag1 > 0.5


def test_hawkes_rejects_unstable_branching_ratio():
    """A branching ratio >= 1 is non-stationary; it must fail loudly, not silently."""
    with pytest.raises(ValueError):
        HawkesProcess(mu=0.01, branching_ratio=1.2, beta=0.02)


def test_hawkes_self_excites_then_decays():
    proc = HawkesProcess(mu=0.01, branching_ratio=0.4, beta=0.05)
    base = proc.intensity(100.0)
    proc.record(100.0)
    assert proc.intensity(100.1) > base
    assert proc.intensity(100.1) > proc.intensity(400.0)


def test_hawkes_weighted_events_contribute_less():
    full = HawkesProcess(mu=0.01, branching_ratio=0.4, beta=0.05)
    damped = HawkesProcess(mu=0.01, branching_ratio=0.4, beta=0.05)
    full.record(10.0, weight=1.0)
    damped.record(10.0, weight=0.25)
    assert damped.intensity(11.0) < full.intensity(11.0)


# ----------------------------------------------------------------- booking model

def test_booking_respects_capacity_and_conserves_demand():
    domain = load_domain()
    scenario = load_scenario("normal_weekday")
    model = DemandModel(domain, scenario, RNGRegistry(3))
    for train in domain.trains:
        for b in model.bookings_for(train, scenario.start_date):
            if b.class_id == "UR":
                continue
            assert b.confirmed <= b.capacity
            assert b.rac <= b.rac_capacity
            assert min(b.confirmed, b.rac, b.waitlist, b.cancellations) >= 0
            assert b.confirmed + b.rac + b.waitlist + b.cancellations == b.gross_demand


def test_waitlist_only_forms_beyond_capacity():
    """WL must be a consequence of the capacity wall, not an independent draw."""
    domain = load_domain()
    scenario = load_scenario("normal_weekday")
    model = DemandModel(domain, scenario, RNGRegistry(5))
    for train in domain.trains:
        for b in model.bookings_for(train, scenario.start_date):
            if b.class_id == "UR" or b.waitlist <= 0:
                continue
            assert b.gross_demand > b.capacity + b.rac_capacity


def test_higher_demand_multiplier_increases_waitlist():
    domain = load_domain()
    low = load_scenario("normal_weekday")
    high = low.with_overrides(**{"demand.multiplier": 2.5})
    wl_low = sum(b.waitlist for t in domain.trains
                 for b in DemandModel(domain, low, RNGRegistry(11)).bookings_for(t, low.start_date))
    wl_high = sum(b.waitlist for t in domain.trains
                  for b in DemandModel(domain, high, RNGRegistry(11)).bookings_for(t, high.start_date))
    assert wl_high > wl_low


# -------------------------------------------------------------------- workforce

def test_workers_are_heterogeneous():
    domain = load_domain()
    workers = build_workforce(domain, load_scenario("normal_weekday"), RNGRegistry(0))
    profs = [w.skills.get(next(iter(w.skills)), 0) for w in workers[:80]]
    assert len(set(round(p, 3) for p in profs)) > 20


def test_eligibility_enforces_qualifications():
    domain = load_domain()
    scenario = load_scenario("normal_weekday")
    workers = build_workforce(domain, scenario, RNGRegistry(0))
    from simulation.entities.core import Task, TaskStatus

    task = Task(
        id="T1", case_id="C1", process_id="maintenance", activity_id="repair",
        location_id=workers[0].home_location_id, department_id="MNT",
        required_skill="signal_telecom", min_proficiency=0.5,
        required_qualifications={"signal_authority"}, priority=1,
        created_at=0.0, deadline=100.0, duration_median_min=30.0, duration_sigma=0.4,
    )
    for w in workers:
        w.absent_today = False
        w.current_task_id = None
        result = is_eligible(w, task, 10.0, domain, shift_enforced=False)
        if result.eligible:
            assert "signal_authority" in w.qualifications
            assert w.proficiency("signal_telecom") >= 0.5


def test_shift_coverage_wraps_midnight():
    domain = load_domain()
    night = domain.shifts["night"]
    assert night.covers(23.0)
    assert night.covers(2.0)
    assert not night.covers(12.0)


# ------------------------------------------------------------------- scenarios

def test_all_scenarios_load():
    for name in available_scenarios():
        cfg = load_scenario(name)
        assert cfg.id == name
        assert 0.0 <= cfg.disruptions["hawkes_branching_ratio"] < 1.0, \
            f"{name} has an unstable branching ratio"


def test_scenario_overrides_do_not_mutate_original():
    base = load_scenario("normal_weekday")
    original = base.demand["multiplier"]
    modified = base.with_overrides(**{"demand.multiplier": 9.9})
    assert modified.demand["multiplier"] == 9.9
    assert base.demand["multiplier"] == original
