# Workforce Simulation Environment
### A synthetic, stochastic world for studying dynamic workforce allocation
#### Initial domain configuration: Southern Railway (Chennai division subset)

> **This is not a digital twin.** It is a *plausible synthetic world* built to study
> workforce allocation under changing demand, uncertainty and disruption. Station
> and train names are real; **every operational number in it is invented**
> (`[SYNTH]`). No IRCTC or Indian Railways operational data is used at runtime,
> and none is required. Results describe the behaviour of *this model*, not of
> Southern Railway.

---

## What this is

A discrete-event simulation in which work **exists because something happened**,
rather than being drawn from a fixed list:

```
latent demand intensity  (trend x weekday x hourly x festival x regime x AR(1))
        │
        ▼
negative-binomial demand  ->  capacity-constrained booking  ->  CNF / RAC / WL
        │                                                             │
        ▼                                                             ▼
station pressure, footfall, unreserved & platform tickets     waitlist pressure
        │                                                             │
        └──────────────────────────┬──────────────────────────────────┘
                                   ▼
                          cases (multi-stage processes)
                                   │
        disruptions (Hawkes, self-exciting) ──┤ spawn work through the SAME engine
                                   ▼
                    queues -> eligible workers -> assignment
                                   ▼
              task execution -> outcomes -> KPIs, costs, event log
                                   ▼
               latent service experience -> sparse delayed CSAT/NPS
```

The chain is causal in one direction. Nothing is "made up at the KPI level": a
station is busy because passengers are there, and passengers are there because a
demand process put them there under capacity constraints.

The engine (`simulation/core/engine.py`) is a SimPy discrete-event simulation:
demand, disruptions and their cascades are drawn live as the episode runs, so
decisions influence what happens next rather than replaying a fixed script.

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Run one day of the world and watch it

```bash
.venv/bin/python -m experiments.runners.run_sim \
    --scenario normal_weekday --policy wfm_heuristic --seed 0 --trace out/run.json
```

Prints an interleaved event timeline (cascades marked `<-- cascade`) and the KPI
summary. Then open the visualiser and load `out/run.json`:

```bash
xdg-open viz/index.html      # or: open viz/index.html
```

The visualiser is a dependency-free local page. It gives you a play/scrub clock,
live per-department load, per-station queues, the event feed with cascades
highlighted, the decision log, and the synthetic feedback panel. Load a second
trace via **Compare** to put two runs side by side on the same seed.

### Check that the world behaves sensibly

```bash
.venv/bin/python -m analysis.validation.validate --scenario normal_weekday --seed 0
.venv/bin/python -m analysis.validation.diagnose --scenario normal_weekday --policy greedy
```

`validate` separates **invariants** (a failure is a bug: capacity exceeded,
double-booked workers, qualification violations, feedback leakage) from
**plausibility** (a failure is miscalibration: seasonality, overdispersion,
duration skew, sub-critical disruption branching). `diagnose` explains *why* a run
looks the way it does — where lateness concentrates, what blocks stuck work, where
capacity goes.

### Compare scenarios

```bash
.venv/bin/python -m experiments.runners.run_baselines \
    --scenarios normal_weekday,high_demand,high_absence,combined_stress \
    --policies greedy --seeds 3
```

### Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

---

## Scenarios

All eight live in `config/scenarios/` and are expressed as deltas from
`normal_weekday.yaml`:

| Scenario | What changes |
|---|---|
| `normal_weekday` | Reference world |
| `weekend` | Leisure-shifted demand, higher unreserved share |
| `high_demand` | Demand well above establishment planning |
| `festival` | Extreme peak, heavier tails, more unreserved crowding |
| `high_absence` | Normal demand, badly depleted roster |
| `equipment_disruption` | Elevated failure rate, strong cascade excitation |
| `high_demand_absence` | Squeezed from both sides |
| `combined_stress` | High demand + cascading failures + absence + regime switching |

Observed behaviour under a greedy policy (3 seeds) — service degrades
monotonically with stress, which is the basic sanity property we want:

| Scenario | SLA breach | Mean queue (min) | Utilisation |
|---|---|---|---|
| `normal_weekday` | 0.15 | 36 | 0.24 |
| `high_absence` | 0.19 | 44 | 0.28 |
| `high_demand` | 0.29 | 59 | 0.31 |
| `combined_stress` | 0.36 | 81 | 0.38 |

---

## Modelling choices worth knowing

**Waitlist is emergent, not drawn.** There is one demand pool per
(train, class, day); capacity partitions it into confirmed / RAC / waitlist by
arrival order, and cancellations promote WL → RAC → CNF. WL is therefore *demand
that hit a capacity wall*, which is why it works as a pressure signal. Demand is
**never** the sum of confirmed + RAC + WL + unreserved + platform tickets — those
are different processes with a shared cause.

**Counts are overdispersed, durations are skewed.** Demand uses Negative Binomial
(variance > mean, from unobserved heterogeneity), task durations Lognormal,
absence a Beta-Binomial hierarchy, disruption severity a lognormal body with a
Pareto tail for rare severe events. Nothing is `random.normal()` by default.

**Disruptions self-excite and are sub-critical.** Failures and delays raise the
short-term rate of further trouble via a Hawkes kernel parameterised by
**branching ratio** (`< 1` enforced at construction — an earlier version with an
implicit ratio above 1 produced 2,770 disruptions in one day). Neighbouring
stations receive damped, *weighted* excitation, so network spread cannot push the
aggregate branching ratio over the stability threshold.

**Disruptions create ordinary work.** There is no separate "disruption handler":
a failure spawns cases through the same process engine as routine work, so
`delay → crowding → extra commercial and cleaning work → knock-on delay` emerges
from the mechanics instead of being scripted.

**Constraints are hard, not penalties.** Qualification, skill floor, shift and
travel-distance constraints are enforced when candidates are enumerated. An
unqualified assignment is impossible, not merely expensive.

**Workers are not interchangeable.** Proficiency, experience, fatigue, cost,
absence propensity and cross-training are drawn per individual. Role mix within a
department reflects the work mix, and shift allocation is demand-weighted.

---

## Synthetic feedback and outcome layer

NPS, CSAT, operational costs, employee performance and schedules are sparse or
not consistently available in public railway data, so this simulator **generates**
them. They are synthetic instruments on a synthetic world, configurable under
`feedback:` in each scenario file. **They are not sourced from IRCTC or Indian
Railways and must not be presented as such.**

| Signal | Treated as | Key property |
|---|---|---|
| **CSAT / NPS** | Sparse, delayed customer feedback | ~6% base response rate, responses arrive hours later, dissatisfied customers respond more |
| **Operational cost** | Outcome / KPI | Decomposed: labour, overtime, reserve activation, SLA penalty, disruption response, idle capacity, rework |
| **Employee performance** | Outcome / KPI | Difficulty-adjusted, cut at shift boundaries — never a live per-task readout |
| **Schedules** | Workforce state | Drives availability, shift boundaries, overtime authorisation, staffing actions |

### Latent state vs observation

The separation is **structural**, not a convention:

```
latent service experience  e ∈ [0,1]      <- simulation/state/true_state.py, NEVER observable
        │  (wait, breach, crowding, delay exposure, server quality, rework)
        ▼
sparse sampling + delivery delay
        ▼
CSAT 1-5 / NPS 0-10 responses             <- observable ONLY after delivered_at <= now
```

`FeedbackRegistry.observed()` filters on **delivery time**, not interaction time,
so a response cannot be seen before it arrives. Performance is aggregated over
**closed** review periods only. A run of the reference scenario shows the
instrument behaving as a real one would:

```
latent experience (hidden, population)  0.666
responders' latent experience           0.595   <- non-response bias
observed CSAT                           3.46    <- pessimistic, lagged, small sample
survey response rate                    10%     <- sparse
```

An agent reading observed CSAT is therefore reading a biased, lagged, small
sample — which is the point. `latent_experience` is exported to CSV **for
validation only** and must never be used as a model input.

---

## Event log

Every activity start/end emits a PM4Py-compatible row (`case_id`, `activity`,
`timestamp`, plus resource/role/team/location/queue_time/processing_time/status/
outcome) via `simulation/metrics/event_log.py`. `EventLog.to_xes_dataframe()`
renames to PM4Py's default column names (`case:concept:name`, `concept:name`,
`time:timestamp`, `org:resource`, `org:role`), so a run's log imports into PM4Py
without a conversion step.

---

## Reproducibility

One scenario seed derives **named, independent RNG streams** per subsystem
(`simulation/core/rng.py`). Adding a draw in the workforce generator does not
shift the demand draws, so paired-seed comparison is meaningful. Same seed ⇒ same
world, verified by test.

---

## Repository layout

```
config/domain/       Southern Railway skin (swap this to change domain entirely)
config/scenarios/    Eight scenarios as deltas from the reference
simulation/
  core/              rng, config, engine (SimPy DES + decision epochs)
  entities/          generic Worker/Task/Process/Location/Shift/Asset
  distributions/     NB, lognormal, beta-binomial, heavy tail, AR(1), Hawkes
  demand/            latent intensity, booking allocation, footfall
  workforce/         population, eligibility, absence, fatigue, schedule
  processes/         activity-graph templates
  disruptions/       Hawkes cascade generation
  feedback/          latent experience, surveys (CSAT/NPS), performance
  metrics/           KPIs, cost ledger, PM4Py-compatible event log
  state/             true (latent) state, kept apart from observations
rl/                  control interface used to drive the sim (see note below)
experiments/runners/ run_sim, run_baselines
analysis/validation/ validate, diagnose
viz/                 zero-dependency interactive run viewer
```

### Note on the `rl/` directory

The Gymnasium wrapper and the three simple policies (`random`, `greedy`,
`wfm_heuristic`) exist because the simulation needs *something* to make dispatch
and staffing decisions in order to run at all. **No learning is implemented and
none is intended at this stage** — the priority is getting the simulated world
right first. The policies are deterministic rules used as drivers and as
reference points, nothing more.

---

## Extending to another domain

The engine mentions no railway concepts. To model a hospital or a factory, write
a new `config/domain/<name>.yaml` (departments, skills, roles, qualifications,
shifts, locations, assets, headcount) and new process templates in
`simulation/processes/templates.py`. Locations become wards or cells, trains
become admissions or production orders, and the demand → workload → task chain is
unchanged.

---

## Known limitations

- Every rate, cost and probability not traceable to a public aggregate is
  invented. This world is about *mechanism*, not Southern Railway's real numbers.
- The booking model is deliberately lean: aggregate cancellation and promotion, no
  per-passenger cancellation hazard, no chart preparation, no GNWL/TQWL sub-quotas.
- Overall utilisation is moderate while specific skills bottleneck — realistic
  (specialists are scarce, generalists idle), but it means the aggregate
  utilisation number understates local pressure.
- Observability is currently near-complete apart from the feedback layer; the
  latent/observed seam exists so a fuller POMDP variant can be added without
  restructuring the generative model.
