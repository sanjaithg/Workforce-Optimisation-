"""Demand model: latent intensity -> bookings -> station pressure -> workload.

The chain is deliberately causal. Nothing here draws an "amount of work" directly;
work exists because passengers exist, and passengers exist because a latent demand
intensity was realised under capacity constraints.

Design note on the reservation split (RESEARCH_REPORT.md section 6): confirmed,
RAC and waitlist are NOT independent random variables that get summed. There is
one demand pool per (train, class, day); capacity decides how that pool is
partitioned. WL is what demand looks like after it hits a capacity wall, which is
exactly why it is a useful pressure signal for a workforce agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from simulation.core.config import DomainConfig, ScenarioConfig
from simulation.distributions.library import AR1, negative_binomial


@dataclass
class ClassBooking:
    """Outcome of the allocation process for one (train, class, departure date)."""

    train_id: str
    class_id: str
    departure_date: date
    gross_demand: int          # would-be passengers (latent, never fully observed)
    capacity: int
    rac_capacity: int
    confirmed: int
    rac: int
    waitlist: int
    cancellations: int

    @property
    def occupancy(self) -> float:
        return self.confirmed / self.capacity if self.capacity else 0.0

    @property
    def wl_pressure(self) -> float:
        """Waitlist as a fraction of capacity: unmet demand, scaled."""
        return self.waitlist / self.capacity if self.capacity else 0.0


@dataclass
class TrainDeparture:
    """A train event at a location, carrying the passenger load it generates."""

    train_id: str
    name: str
    origin: str
    destination: str
    departure_hour: float
    departure_date: date
    bookings: list[ClassBooking]
    delay_minutes: float = 0.0

    @property
    def total_onboard(self) -> int:
        return sum(b.confirmed + b.rac for b in self.bookings)

    @property
    def total_waitlist(self) -> int:
        return sum(b.waitlist for b in self.bookings)

    @property
    def mean_wl_pressure(self) -> float:
        reserved = [b for b in self.bookings if b.class_id != "UR"]
        return float(np.mean([b.wl_pressure for b in reserved])) if reserved else 0.0


class DemandModel:
    """Generates the latent demand process and realises it as bookings/footfall.

    Latent intensity per (train, class):
        lambda = base * weekday * hourly * festival * scenario_mult * regime * AR1

    Gross demand ~ NegBinomial(lambda, k)  — overdispersed, because unobserved
    heterogeneity (route popularity, local events) makes variance exceed the mean.
    """

    def __init__(self, domain: DomainConfig, scenario: ScenarioConfig, rng_registry):
        self.domain = domain
        self.scenario = scenario
        self.rng = rng_registry
        cfg = scenario.demand

        self.multiplier = float(cfg["multiplier"])
        self.festival_mode = bool(cfg.get("festival_mode", False))
        self.dispersion_k = float(cfg["dispersion_k"])
        self.weekday_profile = list(cfg["weekday_profile"])
        self.hourly_profile = list(cfg["hourly_profile"])
        self.unreserved_ratio = float(cfg["unreserved_ratio"])
        self.platform_ticket_ratio = float(cfg["platform_ticket_ratio"])

        # One AR(1) per train so demand is temporally correlated per service, not
        # independently redrawn each day.
        self._ar1: dict[str, AR1] = {
            t["id"]: AR1(phi=float(cfg["ar1_phi"]), sigma=float(cfg["ar1_sigma"]))
            for t in domain.trains
        }
        # Latent station-pressure factor shared by unreserved demand, platform
        # tickets and footfall, so those three are correlated without being equal.
        self._station_ar1: dict[str, AR1] = {
            loc_id: AR1(phi=0.6, sigma=0.1) for loc_id in domain.locations
        }
        self.regime_multiplier = 1.0

    # ---------- latent layer ----------

    def latent_intensity(self, train: dict, class_id: str, day: date) -> float:
        """True (unobservable) demand intensity for a train-class on a date."""
        cls = self.domain.classes[class_id]
        weekday = self.weekday_profile[day.weekday()]
        base = float(train["base_demand"])
        # Class share of a train's demand, proportional to capacity but skewed
        # towards cheaper classes (more people want sleeper than AC2). [SYNTH]
        share = cls["capacity_per_train"] / max(1.0, cls["fare_index"] ** 0.6)
        total_share = sum(
            self.domain.classes[c]["capacity_per_train"]
            / max(1.0, self.domain.classes[c]["fare_index"] ** 0.6)
            for c in train["classes"]
        )
        festival = 1.0 + (0.35 if self.festival_mode else 0.0)
        ar1 = self._ar1[train["id"]].step(self.rng.demand_latent)
        return base * (share / total_share) * weekday * festival * self.multiplier * self.regime_multiplier * ar1

    # ---------- booking / allocation layer ----------

    def _allocate(self, gross: int, capacity: int, rac_capacity: int) -> tuple[int, int, int]:
        """Capacity-constrained FCFS allocation into confirmed / RAC / waitlist.

        Bookings arrive in order; the first `capacity` get berths, the next
        `rac_capacity` get RAC, everyone after that is waitlisted. WL is therefore
        emergent from the capacity wall rather than an independent draw.
        """
        confirmed = min(gross, capacity)
        rac = min(max(gross - capacity, 0), rac_capacity)
        waitlist = max(gross - capacity - rac_capacity, 0)
        return confirmed, rac, waitlist

    def bookings_for(self, train: dict, day: date) -> list[ClassBooking]:
        out: list[ClassBooking] = []
        rac_fraction = float(self.domain.booking["rac_fraction"])
        cancel_rate = float(self.domain.booking["cancellation_rate"])

        for class_id in train["classes"]:
            cls = self.domain.classes[class_id]
            capacity = int(cls["capacity_per_train"])
            lam = self.latent_intensity(train, class_id, day)
            gross = negative_binomial(self.rng.demand_booking, lam, self.dispersion_k)

            if class_id == "UR":
                # Unreserved has no allocation process: everyone travels, crowding
                # is absorbed by the coach. Excess over capacity becomes crowding
                # pressure rather than waitlist.
                out.append(
                    ClassBooking(
                        train_id=train["id"], class_id=class_id, departure_date=day,
                        gross_demand=gross, capacity=capacity, rac_capacity=0,
                        confirmed=gross, rac=0, waitlist=0, cancellations=0,
                    )
                )
                continue

            rac_capacity = int(round(capacity * rac_fraction))
            confirmed, rac, waitlist = self._allocate(gross, capacity, rac_capacity)

            # Aggregate cancellations free seats, promoting WL -> RAC -> confirmed.
            # (Lean model: an aggregate rate, not a per-passenger hazard. The
            # promotion *chain* is what matters for downstream workload.)
            booked = confirmed + rac
            cancellations = int(self.rng.demand_booking.binomial(booked, cancel_rate)) if booked else 0
            # Freed berths promote RAC -> confirmed, which in turn frees RAC slots
            # for the head of the waitlist. Two-stage promotion, order preserved.
            rac_promoted = min(rac, cancellations)
            rac_after = rac - rac_promoted
            confirmed_after = confirmed - cancellations + rac_promoted
            wl_promoted = min(waitlist, max(0, rac_capacity - rac_after))
            waitlist_after = waitlist - wl_promoted
            rac_after += wl_promoted

            out.append(
                ClassBooking(
                    train_id=train["id"], class_id=class_id, departure_date=day,
                    gross_demand=gross, capacity=capacity, rac_capacity=rac_capacity,
                    confirmed=max(0, confirmed_after), rac=max(0, rac_after),
                    waitlist=max(0, waitlist_after), cancellations=cancellations,
                )
            )
        return out

    def departures_for_day(self, day: date) -> list[TrainDeparture]:
        departures = []
        for train in self.domain.trains:
            departures.append(
                TrainDeparture(
                    train_id=train["id"],
                    name=train["name"],
                    origin=train["origin"],
                    destination=train["destination"],
                    departure_hour=float(train["dep_hour"]),
                    departure_date=day,
                    bookings=self.bookings_for(train, day),
                )
            )
        return sorted(departures, key=lambda d: d.departure_hour)

    # ---------- station pressure layer ----------

    def station_pressure(self, location_id: str, hour_of_day: float) -> float:
        """Latent crowding factor at a station, shared by footfall/UR/platform tickets."""
        loc = self.domain.locations[location_id]
        tier_weight = {1: 1.0, 2: 0.55, 3: 0.25}[loc.tier]
        hourly = self.hourly_profile[int(hour_of_day) % 24]
        ar1 = self._station_ar1[location_id].step(self.rng.demand_unreserved)
        return float(tier_weight * hourly * self.multiplier * self.regime_multiplier * ar1)

    def footfall(self, location_id: str, hour_of_day: float) -> dict[str, int]:
        """Realised, observable station counts for one hour.

        Unreserved ticket sales, platform tickets and general footfall all load on
        the same latent pressure factor with their own base rates and their own
        noise, so they are correlated but not redundant.
        """
        pressure = self.station_pressure(location_id, hour_of_day)
        loc = self.domain.locations[location_id]
        base = {1: 900.0, 2: 320.0, 3: 110.0}[loc.tier]
        rng = self.rng.demand_unreserved

        general = negative_binomial(rng, base * pressure, self.dispersion_k)
        unreserved = negative_binomial(rng, base * pressure * self.unreserved_ratio, self.dispersion_k)
        platform = negative_binomial(rng, base * pressure * self.platform_ticket_ratio, self.dispersion_k)
        return {
            "footfall": general,
            "unreserved_tickets": unreserved,
            "platform_tickets": platform,
            "pressure": pressure,
        }
