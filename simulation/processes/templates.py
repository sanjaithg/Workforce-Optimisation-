"""Process templates: work is a graph of activities, not a flat task list.

Each template is a directed graph. Completing an activity spawns its successors,
so a single trigger (a train arrival, an equipment failure) unfolds into a
multi-stage case with handoffs between departments — which is what produces a
process-mining-usable event log rather than a list of independent jobs.

All durations, probabilities and SLAs here are [SYNTH]: invented but chosen to be
operationally plausible and internally consistent.
"""

from __future__ import annotations

from simulation.entities.core import Activity, ProcessTemplate


def _a(id_, name, dept, skill, prof, quals, median, sigma, successors=(), rework=0.0, rework_target=None):
    return Activity(
        id=id_, name=name, department_id=dept, required_skill=skill,
        min_proficiency=prof, required_qualifications=set(quals),
        duration_median_min=median, duration_sigma=sigma,
        successors=list(successors), rework_prob=rework, rework_target=rework_target,
    )


def maintenance_process() -> ProcessTemplate:
    """Inspection -> defect -> approval -> repair -> verification -> closure.

    The canonical multi-stage maintenance lifecycle, with a rework loop: failed
    verification sends the case back to repair rather than silently closing.
    """
    acts = [
        _a("inspect", "Inspection", "MNT", "track_civil", 0.4, [], 22, 0.45, ["assess"]),
        _a("assess", "Defect assessment", "MNT", "signal_telecom", 0.5, ["safety_cert"], 18, 0.5, ["approve"]),
        _a("approve", "Repair approval", "OPS", "station_ops", 0.5, ["safety_cert"], 10, 0.6, ["repair"]),
        _a("repair", "Repair", "MNT", "signal_telecom", 0.55, ["safety_cert"], 55, 0.7, ["verify"]),
        _a("verify", "Verification", "MNT", "track_civil", 0.5, ["safety_cert"], 15, 0.4,
           ["close"], rework=0.18, rework_target="repair"),
        _a("close", "Closure", "OPS", "station_ops", 0.3, [], 8, 0.3, []),
    ]
    return ProcessTemplate(
        id="maintenance", name="Maintenance lifecycle", start_activity="inspect",
        activities={a.id: a for a in acts}, default_priority=2, sla_minutes=240.0,
    )


def emergency_repair_process() -> ProcessTemplate:
    """Fast-path maintenance spawned by an actual failure: no inspection stage,
    tighter SLA, higher skill floor. Escalation is a different process, not a flag."""
    acts = [
        _a("triage", "Failure triage", "MNT", "signal_telecom", 0.55, ["safety_cert"], 12, 0.4, ["emergency_repair"]),
        _a("emergency_repair", "Emergency repair", "MNT", "signal_telecom", 0.65,
           ["safety_cert", "signal_authority"], 70, 0.8, ["verify_emergency"]),
        _a("verify_emergency", "Post-repair verification", "MNT", "track_civil", 0.5, ["safety_cert"], 18, 0.45,
           ["close_emergency"], rework=0.22, rework_target="emergency_repair"),
        _a("close_emergency", "Closure", "OPS", "traffic_control", 0.4, ["safety_cert"], 9, 0.3, []),
    ]
    return ProcessTemplate(
        id="emergency_repair", name="Emergency repair", start_activity="triage",
        activities={a.id: a for a in acts}, default_priority=1, sla_minutes=120.0,
    )


def crowd_response_process() -> ProcessTemplate:
    """Commercial/passenger response to a crowding surge.

    Note the department handoff: assessment sits with Operations, the counter and
    checking work with Commercial, crowd control with Support. A surge therefore
    consumes capacity across three departments at once, which is exactly the
    cross-department contention the agent has to manage.
    """
    acts = [
        _a("assess_crowd", "Crowd assessment", "OPS", "station_ops", 0.4, [], 8, 0.35,
           ["open_counter", "crowd_control"]),
        _a("open_counter", "Additional ticket counter", "COM", "ticketing", 0.45, [], 40, 0.5, ["passenger_assist"]),
        _a("crowd_control", "Crowd control", "SUP", "security", 0.4, [], 35, 0.55, []),
        _a("passenger_assist", "Passenger assistance", "COM", "passenger_svc", 0.4, [], 25, 0.5, []),
    ]
    return ProcessTemplate(
        id="crowd_response", name="Crowd response", start_activity="assess_crowd",
        activities={a.id: a for a in acts}, default_priority=2, sla_minutes=90.0,
    )


def turnaround_process() -> ProcessTemplate:
    """Routine train turnaround at a station: cleaning, checking, dispatch."""
    acts = [
        _a("coach_clean", "Coach cleaning", "SUP", "cleaning", 0.35, [], 30, 0.45, ["ticket_check"]),
        _a("ticket_check", "Ticket checking", "COM", "ticket_checking", 0.4, [], 20, 0.4, ["dispatch"]),
        _a("dispatch", "Platform dispatch", "OPS", "station_ops", 0.45, [], 12, 0.35, []),
    ]
    return ProcessTemplate(
        id="turnaround", name="Train turnaround", start_activity="coach_clean",
        activities={a.id: a for a in acts}, default_priority=3, sla_minutes=75.0,
    )


def ticketing_process() -> ProcessTemplate:
    """Counter workload driven by unreserved/platform-ticket demand."""
    acts = [
        _a("serve_counter", "Counter service", "COM", "ticketing", 0.30, [], 18, 0.4, []),
    ]
    return ProcessTemplate(
        id="ticketing", name="Ticketing service", start_activity="serve_counter",
        activities={a.id: a for a in acts}, default_priority=3, sla_minutes=60.0,
    )


# Processes a passenger actually experiences. Only these generate customer
# feedback: nobody surveys a passenger about a track inspection they never saw.
CUSTOMER_FACING_PROCESSES = frozenset({"ticketing", "crowd_response", "turnaround"})


def all_templates() -> dict[str, ProcessTemplate]:
    templates = [
        maintenance_process(),
        emergency_repair_process(),
        crowd_response_process(),
        turnaround_process(),
        ticketing_process(),
    ]
    return {t.id: t for t in templates}
