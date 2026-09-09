"""Loading of domain + scenario configuration into typed entity objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from simulation.entities.core import Department, Location, Role, Shift, Skill

REPO_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_DIR = REPO_ROOT / "config" / "domain"
SCENARIO_DIR = REPO_ROOT / "config" / "scenarios"


@dataclass
class DomainConfig:
    """The domain skin: what kind of world this is."""

    raw: dict[str, Any]
    departments: dict[str, Department]
    skills: dict[str, Skill]
    roles: dict[str, Role]
    locations: dict[str, Location]
    shifts: dict[str, Shift]

    @property
    def name(self) -> str:
        return self.raw["domain"]["name"]

    @property
    def trains(self) -> list[dict]:
        return self.raw["trains"]

    @property
    def classes(self) -> dict[str, dict]:
        return {c["id"]: c for c in self.raw["classes"]}

    @property
    def booking(self) -> dict:
        return self.raw["booking"]

    @property
    def workforce(self) -> dict:
        return self.raw["workforce"]

    @property
    def assets(self) -> dict:
        return self.raw["assets"]

    @property
    def suburban(self) -> dict:
        return self.raw["suburban"]

    @property
    def cross_training(self) -> dict:
        return self.raw.get("cross_training", {"probability": 0.0, "secondary_proficiency_range": [0.3, 0.5]})

    def travel_minutes(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        table = self.raw.get("travel_minutes", {})
        if a in table and b in table[a]:
            return float(table[a][b])
        if b in table and a in table[b]:
            return float(table[b][a])
        return 90.0     # unconnected pair: expensive but not impossible


@dataclass
class ScenarioConfig:
    """The experiment knobs: what is happening in this world, and how hard."""

    raw: dict[str, Any]

    @property
    def id(self) -> str:
        return self.raw["scenario"]["id"]

    @property
    def name(self) -> str:
        return self.raw["scenario"]["name"]

    @property
    def domain_name(self) -> str:
        return self.raw["scenario"]["domain"]

    @property
    def seed(self) -> int:
        return int(self.raw["scenario"].get("seed", 0))

    @property
    def duration_days(self) -> int:
        return int(self.raw["scenario"].get("duration_days", 1))

    @property
    def start_date(self) -> date:
        return date.fromisoformat(str(self.raw["scenario"]["start_date"]))

    @property
    def demand(self) -> dict:
        return self.raw["demand"]

    @property
    def workforce(self) -> dict:
        return self.raw["workforce"]

    @property
    def disruptions(self) -> dict:
        return self.raw["disruptions"]

    @property
    def drift(self) -> dict:
        return self.raw.get("drift", {"enabled": False})

    @property
    def reward(self) -> dict:
        return self.raw["reward"]

    def with_overrides(self, **overrides: Any) -> "ScenarioConfig":
        """Return a copy with dotted-path overrides applied, e.g.
        `with_overrides(**{"demand.multiplier": 2.0, "scenario.seed": 7})`."""
        import copy

        raw = copy.deepcopy(self.raw)
        for path, value in overrides.items():
            node = raw
            parts = path.split(".")
            for key in parts[:-1]:
                node = node.setdefault(key, {})
            node[parts[-1]] = value
        return ScenarioConfig(raw)


def load_domain(name: str = "southern_railway") -> DomainConfig:
    path = DOMAIN_DIR / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text())

    departments = {d["id"]: Department(id=d["id"], name=d["name"]) for d in raw["departments"]}
    skills = {
        s["id"]: Skill(id=s["id"], name=s["name"], department_id=s["department_id"])
        for s in raw["skills"]
    }
    roles = {
        r["id"]: Role(
            id=r["id"],
            name=r["name"],
            department_id=r["department_id"],
            skills=dict(r["skills"]),
            qualifications=set(r.get("qualifications", [])),
            hourly_cost=float(r["hourly_cost"]),
        )
        for r in raw["roles"]
    }
    locations = {
        loc["id"]: Location(
            id=loc["id"],
            name=loc["name"],
            tier=int(loc["tier"]),
            platforms=int(loc["platforms"]),
            neighbours=dict(raw.get("travel_minutes", {}).get(loc["id"], {})),
        )
        for loc in raw["locations"]
    }
    shifts = {
        s["id"]: Shift(
            id=s["id"],
            name=s["name"],
            start_hour=float(s["start_hour"]),
            duration_hours=float(s["duration_hours"]),
        )
        for s in raw["shifts"]
    }
    return DomainConfig(raw, departments, skills, roles, locations, shifts)


def load_scenario(name: str = "normal_weekday") -> ScenarioConfig:
    path = SCENARIO_DIR / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml")))
        raise FileNotFoundError(f"No scenario '{name}'. Available: {available}")
    return ScenarioConfig(yaml.safe_load(path.read_text()))


def available_scenarios() -> list[str]:
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))
