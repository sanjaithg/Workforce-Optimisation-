"""Operational cost model.

Costs are an *outcome* signal, not a feedback instrument: they are computed
exactly and known to the operator, unlike NPS/CSAT which arrive late and sampled.
Keeping the two in separate modules keeps that distinction honest.

Real Indian Railways cost structures are not published at the granularity this
simulation works at, so every rate here is [SYNTH] and configurable. The units
are deliberately abstract ("cost units per hour"), because a fabricated rupee
figure would imply a precision we do not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostRates:
    """[SYNTH] cost rates, all in abstract cost units."""

    reserve_activation: float = 45.0       # one-off cost of calling in a reserve
    sla_breach: float = 12.0               # penalty per breached task
    disruption_response: float = 8.0       # per disruption handled
    idle_capacity_per_hour: float = 0.25   # cost of staffed-but-unused capacity
    rework: float = 6.0                    # cost of redoing work

    @classmethod
    def from_config(cls, cfg: dict | None) -> "CostRates":
        if not cfg:
            return cls()
        return cls(**{k: float(v) for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class CostLedger:
    """Running operational cost, decomposed so trade-offs stay visible.

    A single blended number would hide the decision that matters: whether the
    agent bought service quality with overtime, with reserves, or by leaving
    capacity idle.
    """

    rates: CostRates = field(default_factory=CostRates)
    labour: float = 0.0
    overtime: float = 0.0
    reserve_activation: float = 0.0
    sla_penalty: float = 0.0
    disruption_response: float = 0.0
    idle_capacity: float = 0.0
    rework: float = 0.0

    def add_labour(self, amount: float, overtime: bool = False) -> None:
        self.labour += float(amount)
        if overtime:
            self.overtime += float(amount)

    def add_reserve_activation(self, n: int = 1) -> None:
        self.reserve_activation += self.rates.reserve_activation * n

    def add_breach(self, n: int = 1) -> None:
        self.sla_penalty += self.rates.sla_breach * n

    def add_disruption(self, n: int = 1) -> None:
        self.disruption_response += self.rates.disruption_response * n

    def add_rework(self, n: int = 1) -> None:
        self.rework += self.rates.rework * n

    def add_idle(self, idle_workers: float, minutes: float) -> None:
        self.idle_capacity += self.rates.idle_capacity_per_hour * idle_workers * (minutes / 60.0)

    @property
    def total(self) -> float:
        return (self.labour + self.reserve_activation + self.sla_penalty
                + self.disruption_response + self.idle_capacity + self.rework)

    def summary(self) -> dict:
        return {
            "cost_total": round(self.total, 2),
            "cost_labour": round(self.labour, 2),
            "cost_overtime": round(self.overtime, 2),
            "cost_reserve_activation": round(self.reserve_activation, 2),
            "cost_sla_penalty": round(self.sla_penalty, 2),
            "cost_disruption_response": round(self.disruption_response, 2),
            "cost_idle_capacity": round(self.idle_capacity, 2),
            "cost_rework": round(self.rework, 2),
        }
