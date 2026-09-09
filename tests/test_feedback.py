"""Tests for the synthetic feedback / outcome layer.

The critical property under test is NO LEAKAGE: the agent must never receive
latent service experience, an undelivered survey, or an open performance period.
If any of these leaked, a learned policy would depend on information no real
operator has, and the whole experiment would be invalid.
"""

from __future__ import annotations

import numpy as np
import pytest

from rl.environment.env_a import WorkforceEnvA
from rl.policies.baselines import build_policy
from simulation.core.config import load_scenario
from simulation.feedback.experience import ExperienceFactors, ExperienceModel
from simulation.feedback.performance import PerformanceTracker
from simulation.feedback.surveys import FeedbackRegistry, SurveyConfig
from simulation.metrics.costs import CostLedger, CostRates
from simulation.workforce.schedule import Roster

SHORT = {"scenario.duration_days": 1}


def run_short(scenario: str = "normal_weekday", seed: int = 0, policy: str = "wfm_heuristic"):
    env = WorkforceEnvA(scenario=load_scenario(scenario).with_overrides(**SHORT), seed=seed)
    pol = build_policy(policy)
    obs, _ = env.reset(seed=seed)
    while True:
        obs, r, term, trunc, _ = env.step(pol.act(obs, env.action_masks(), env))
        if term or trunc:
            break
    return env


# ------------------------------------------------------------------ no leakage

def test_agent_never_sees_undelivered_surveys():
    """`observed()` must filter on delivery time, not interaction time."""
    rng = np.random.default_rng(0)
    reg = FeedbackRegistry(SurveyConfig(response_rate=1.0, detractor_response_boost=0.0,
                                        delay_median_minutes=200.0, delay_sigma=0.01), rng)
    for i in range(50):
        reg.maybe_survey(f"C{i}", "MAS", "ticketing", now=0.0, latent_experience=0.5)

    assert reg.observed_summary(now=10.0)["n_responses"] == 0, "survey visible before delivery"
    # Visible once delivered, while still inside the rolling observation window.
    later = reg.observed_summary(now=300.0)
    assert later["n_responses"] > 0, "survey never became visible"
    for r in reg.observed(now=300.0):
        assert r.delivered_at <= 300.0
    # And it ages out of the rolling window rather than accumulating forever.
    assert reg.observed_summary(now=5000.0)["n_responses"] == 0


def test_observation_contains_no_latent_experience():
    """The observation vector must not encode the latent experience value."""
    env = run_short()
    e = env.engine
    obs = env._build_observation()
    latent = e.true_state.true_experience_mean()
    assert latent > 0, "no latent experience was recorded, test is vacuous"
    # No observation feature should equal the latent mean (which is not exposed).
    assert not np.any(np.isclose(obs, latent, atol=1e-6)), "latent experience leaked into obs"


def test_observed_feedback_lags_and_biases_the_truth():
    """Observed CSAT must be a biased, lagged sample, not the population mean."""
    env = run_short()
    e = env.engine
    true = e.feedback.true_summary()
    assert true["n"] > 10, "too few responses to test"
    # Non-response bias: responders are less satisfied than the population.
    assert true["latent_mean"] < e.true_state.true_experience_mean(), \
        "responders should skew dissatisfied"


def test_surveys_are_sparse():
    env = run_short()
    e = env.engine
    assert e.feedback.interactions > 0
    rate = len(e.feedback.responses) / e.feedback.interactions
    assert rate < 0.35, f"survey response rate {rate:.2f} is not sparse"


def test_performance_records_are_periodic_not_live():
    """Records are cut at shift boundaries; the open period is not visible."""
    env = run_short()
    e = env.engine
    assert len(e.performance.records) > 0
    assert len(e.performance.outcomes) > len(e.performance.records), \
        "records should aggregate many task outcomes, not mirror them 1:1"
    summary = e.observed_feedback()
    assert "mean_performance" in summary


def test_performance_summary_excludes_open_period():
    tracker = PerformanceTracker()
    assert tracker.department_summary()["n_records"] == 0
    # Outcomes exist but no period has been cut, so nothing is observable yet.
    from simulation.entities.core import Task, Worker

    worker = Worker(id="W1", role_id="r", department_id="COM", team_id="t",
                    home_location_id="MAS", shift_id="morning", skills={"ticketing": 0.8},
                    qualifications=set(), hourly_cost=2.0, experience_years=3.0,
                    absence_propensity=0.05)
    task = Task(id="T1", case_id="C1", process_id="ticketing", activity_id="serve_counter",
                location_id="MAS", department_id="COM", required_skill="ticketing",
                min_proficiency=0.3, required_qualifications=set(), priority=3,
                created_at=0.0, deadline=60.0, duration_median_min=18.0, duration_sigma=0.4)
    tracker.record_task(worker, task, 15.0, 14.0, True, 0.1, 20.0)
    assert tracker.department_summary()["n_records"] == 0, "open period leaked into summary"
    tracker.cut_period([worker], 480.0)
    assert tracker.department_summary()["n_records"] == 1


# --------------------------------------------------------------- signal quality

def test_experience_responds_to_operational_conditions():
    """Worse operations must produce worse latent experience, monotonically."""
    from simulation.entities.core import Task

    model = ExperienceModel(ExperienceFactors(noise_sigma=0.0), np.random.default_rng(0))

    def make_task(queue_time: float) -> Task:
        t = Task(id="T", case_id="C", process_id="ticketing", activity_id="serve_counter",
                 location_id="MAS", department_id="COM", required_skill="ticketing",
                 min_proficiency=0.3, required_qualifications=set(), priority=3,
                 created_at=0.0, deadline=60.0, duration_median_min=18.0, duration_sigma=0.4)
        t.started_at = queue_time
        return t

    good = model.latent_experience(make_task(2.0), 60.0, 0.5, 0.0, 0.9, 0.0, False)
    slow = model.latent_experience(make_task(55.0), 60.0, 0.5, 0.0, 0.9, 0.0, False)
    breached = model.latent_experience(make_task(55.0), 60.0, 0.5, 0.0, 0.9, 0.0, True)
    crowded = model.latent_experience(make_task(55.0), 60.0, 1.8, 40.0, 0.4, 0.8, True)

    assert good > slow > breached > crowded


def test_nps_is_harsher_than_csat():
    """The two instruments must not be redundant restatements of each other."""
    rng = np.random.default_rng(0)
    reg = FeedbackRegistry(SurveyConfig(csat_noise=0.0, nps_noise=0.0), rng)
    e = 0.7
    csat_frac = (reg._to_csat(e) - 1) / 4.0
    nps_frac = reg._to_nps(e) / 10.0
    assert nps_frac < csat_frac


def test_costs_decompose_and_sum():
    ledger = CostLedger(CostRates())
    ledger.add_labour(100.0)
    ledger.add_labour(50.0, overtime=True)
    ledger.add_reserve_activation(2)
    ledger.add_breach(3)
    ledger.add_disruption()
    ledger.add_rework()
    ledger.add_idle(10, 60.0)
    s = ledger.summary()
    assert s["cost_overtime"] == 50.0
    assert s["cost_reserve_activation"] == 90.0
    assert s["cost_sla_penalty"] == 36.0
    components = (s["cost_labour"] + s["cost_reserve_activation"] + s["cost_sla_penalty"]
                  + s["cost_disruption_response"] + s["cost_idle_capacity"] + s["cost_rework"])
    assert abs(components - s["cost_total"]) < 1e-6


def test_performance_penalises_rework_and_credits_difficulty():
    from simulation.entities.core import Task, Worker

    def build(rework: int, difficulty_minutes: float) -> float:
        tracker = PerformanceTracker()
        worker = Worker(id="W1", role_id="r", department_id="MNT", team_id="t",
                        home_location_id="MAS", shift_id="morning",
                        skills={"signal_telecom": 0.8}, qualifications=set(),
                        hourly_cost=3.0, experience_years=5.0, absence_propensity=0.05)
        for i in range(6):
            task = Task(id=f"T{i}", case_id=f"C{i}", process_id="maintenance",
                        activity_id="repair", location_id="MAS", department_id="MNT",
                        required_skill="signal_telecom", min_proficiency=0.55,
                        required_qualifications=set(), priority=2, created_at=0.0,
                        deadline=240.0, duration_median_min=difficulty_minutes,
                        duration_sigma=0.5)
            task.rework_count = rework
            tracker.record_task(worker, task, 30.0, 30.0, True, 0.1, 100.0 + i)
        return tracker.cut_period([worker], 480.0)[0].performance_index

    assert build(0, 45.0) > build(1, 45.0), "rework should reduce the performance index"
    assert build(0, 90.0) > build(0, 20.0), "harder work should earn more credit"


# -------------------------------------------------------------------- schedules

def test_schedule_drives_availability():
    env = run_short()
    e = env.engine
    assert len(e.roster.entries) > 0
    entry = e.roster.entries[0]
    assert entry.planned_end > entry.planned_start
    worker = next(w for w in e._roster() if w.id == entry.worker_id)
    mid_shift = (entry.planned_start + entry.planned_end) / 2
    if not entry.absent:
        assert e.roster.is_on_duty(worker, mid_shift)
    assert not e.roster.is_on_duty(worker, entry.planned_start - 60.0)


def test_absence_marked_on_schedule():
    env = run_short(scenario="high_absence")
    e = env.engine
    absent_entries = [x for x in e.roster.entries if x.absent]
    assert len(absent_entries) > 0, "high absence scenario recorded no absences on the roster"


def test_schedule_records_export_shape():
    env = run_short()
    records = env.engine.roster.to_records()
    assert records and {"worker_id", "shift_id", "planned_start_min", "absent"} <= set(records[0])


# ----------------------------------------------------------------- integration

def test_outcome_summary_shape():
    env = run_short()
    a = env.engine.outcome_summary()
    assert set(a) == {"costs", "feedback_observed", "feedback_true", "performance", "latent"}
    assert a["costs"]["cost_total"] > 0


def test_surveys_can_be_disabled():
    cfg = load_scenario("normal_weekday").with_overrides(**SHORT)
    cfg.raw["feedback"]["surveys"]["enabled"] = False
    env = WorkforceEnvA(scenario=cfg, seed=0)
    pol = build_policy("greedy")
    obs, _ = env.reset(seed=0)
    for _ in range(200):
        obs, r, term, trunc, _ = env.step(pol.act(obs, env.action_masks(), env))
        if term or trunc:
            break
    assert len(env.engine.feedback.responses) == 0
