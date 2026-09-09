# Workforce Optimisation under Dynamic Demand — Research Report
### Southern Railway configuration of a generic workforce-simulation engine

> **Framing correction (applied throughout).** This project builds a *plausible
> synthetic world*, **not a digital twin**. No claim of fidelity to real Southern
> Railway operations is made anywhere in this report or the codebase. Separately,
> a learned policy is **not** considered successful merely because it trains — see
> §31.
>
> **Status: RL work is paused** at the user's direction until the simulated world
> is judged correct. The Gymnasium wrapper and the rule-based policies exist only
> because the simulation needs something to make dispatch decisions in order to
> run.

Status labels used throughout: **[FACT]** verifiable from a cited source · **[INFERENCE]** reasoned from facts but not directly stated · **[ASSUMPTION]** a modelling choice we are making · **[SYNTH]** a synthetic parameter we will invent and must calibrate/justify later · **[REC]** a recommendation for this project.

---

## 1. Current understanding of the project

You want a research platform for studying **dynamic workforce allocation under uncertainty** — not a railway-ops product. Indian Railways / Southern Railway is the *skin*: a believable, checkable domain that gives the abstractions (Worker, Task, Queue, Disruption, KPI) real semantics and lets us sanity-check the simulator against public statistics. The underlying engine must stay domain-agnostic so the same code could later wear a hospital or a factory skin. Two environments are wanted: (A) a live generative discrete-event simulation, and (B) a pre-generated coherent synthetic dataset replayed through a lighter reactive engine. Both are RL-environment-shaped (Gymnasium), both must produce PM4Py-compatible event logs, and both must support non-stationary/drifting scenarios with seeded reproducibility.

## 2. Critique of the current approach (as stated in the brief)

- The "Demand = confirmed + RAC + WL + unreserved + platform tickets" formula in the brief is explicitly flagged as wrong, correctly — these are **outputs of a shared capacity-constrained booking process**, not independent draws that sum. **[ASSUMPTION→REC]** We model one latent demand intensity per (train, class, date) and *allocate* it into confirmed/RAC/WL via a capacity mechanism (§6).
- "Do not use `random.normal()` everywhere" is right: counts should be Poisson/Negative-Binomial family, durations Lognormal/Gamma, rare severe events heavy-tailed, not Gaussian.
- The 25-part brief is, honestly, a multi-month research programme (full POMDP formalism, Hawkes-process disruptions, calibration against public statistics, 8 scenarios × multiple seeds, PPO baseline, validation suite). **[REC]** Build one coherent, correctly-structured MVP first with the right seams (config-driven domain, shared stochastic modules between A and B, generic entities, event-log output), rather than shipping 20 shallow scripts. Depth in the causal chain matters more than breadth of features — this matches your own Part 24 principle.
- Southern Railway "reality" is used only for *plausible parameterisation* (station counts, class mix, rough punctuality), never as a live data dependency — no scraping, no IRCTC API calls. Everything at runtime is synthetic and seeded.

## 3. Recommended simulation paradigm

**[REC]** Formally, task execution is a **Semi-Markov Decision Process (SMDP)**: the agent (and the environment) act at *decision epochs* — irregular, event-triggered times (task created, worker freed, disruption detected) — and sojourn times between epochs are generally non-exponential (Lognormal/Gamma durations), which is exactly what a semi-Markov formalism is for. We treat the RL interface as **"MDP at decision epochs"**: `env.step()` is called once per decision epoch, and the simulation clock is advanced internally (skipping non-decision events) between epochs — this is the standard DES+RL pattern used in the flexible-job-shop and production-scheduling RL literature (see §14). We architect the observation as **state, not full history**, but keep a clean `s_t` (latent) vs `o_t` (observed) split from day one (§6) so a POMDP filter/belief-state can be dropped in later without restructuring the engine.

## 4. Recommended mathematical modelling techniques (variable → distribution)

| Variable | Model | Why |
|---|---|---|
| Daily/hourly booking counts per (train, class, OD) | Poisson if intensity known, **Negative Binomial** for the marginal (overdispersion from unobserved heterogeneity — festivals, route popularity) | NB is the standard overdispersed-count model for demand with unobserved heterogeneity; arXiv:2208.06372 and the Hawkes/NB literature both flag transportation demand as overdispersed relative to Poisson. **[FACT-adjacent, cited]** |
| Intraday booking arrival process (early-bird vs last-minute rush, Tatkal opening) | **Self-exciting (Hawkes) point process**, or NB derived as a marked/self-exciting process | A booking opening or price/quota event triggers clustered follow-on bookings — the self-exciting property is exactly this. arXiv:2112.14942 shows the self-exciting NB distribution *is* a marked Hawkes process, which is a clean bridge between the two model families. **[FACT, cited]** |
| Latent daily demand intensity (trend/season/holiday) | **Log-Gaussian state-space / AR(1)-in-log-intensity** with weekly + annual seasonal components, holiday/festival regressors | Standard structural-time-series decomposition (trend + seasonality + regressors + AR residual), keeps intensity positive via log-link. **[ASSUMPTION, standard practice]** |
| WL → RAC → Confirmed transition | **Capacity-constrained sequential allocation + cancellation-driven promotion (order-statistics thinning)**, not an independent RV | Matches the real PRS mechanism: fixed berth capacity, RAC as first overflow buffer, WL beyond that, and chart-preparation-time promotion as cancellations arrive — see §6. **[FACT for mechanism, cited; ASSUMPTION for our parametrisation]** |
| Unreserved ticket demand, platform-ticket / footfall | **Poisson/NB driven by a shared latent "station pressure" factor**, correlated with but not equal to reserved demand | Footfall correlates with reserved arrivals/departures but has its own local dynamics (local commuters, well-wishers). Shared-latent-factor (copula-style) coupling avoids the "just sum it" trap. **[ASSUMPTION]** |
| Task durations (repair, inspection, ticket-checking pass, etc.) | **Lognormal or Gamma** | Durations are positive, right-skewed, standard in service-ops and reliability modelling. **[ASSUMPTION, standard practice]** |
| Equipment failures / infrastructure disruptions | **Poisson process for baseline rate**, with **self-exciting (Hawkes) escalation** for cascades (a failure raises the short-term rate of related failures/overload), severity drawn from a **heavy-tailed** distribution (Lognormal or Pareto-tailed) for rare severe events | Standard reliability baseline + cascade realism; heavy tail captures the "rare but severe" requirement explicitly asked for. **[ASSUMPTION]** |
| Worker absence | **Bernoulli per worker-shift**, rate itself drawn from a **Beta** (hierarchical: role/team-level heterogeneity) → **Beta-Binomial** at the aggregate level | Captures both individual randomness and role-level overdispersion (some teams chronically short-staffed) without needing per-worker history. **[ASSUMPTION]** |
| Regime/season/structural shifts | **Regime-switching (hidden Markov) on the latent-intensity parameters**, plus explicit scenario multipliers for controlled experiments | Cleanly supports "Year 2 has different baseline" (structural break) vs "festival week" (temporary regime) without conflating the two. **[ASSUMPTION]** |
| Observation noise on counts we "measure" (e.g. reported footfall vs true footfall) | Small multiplicative lognormal or NB overdispersion on top of the true count | Keeps the true/observed split meaningful even in the MVP's mostly-observed setting. **[ASSUMPTION]** |

## 5. Demand modelling approach (detail)

Latent state per (date, train, class): `λ_base(train,class) × season_week(dow) × season_annual(doy) × holiday_mult(date) × regime(t) × AR1_noise(t)`, log-scale AR(1) residual for temporal correlation. Daily gross demand for that (train,class) ~ **NegativeBinomial(mean=λ_t, dispersion=φ)**. This gross demand is the pool of *would-be* bookers over the multi-day booking window before departure; we generate their booking-lead-time via a **booking-curve** (fraction of gross demand arriving each day-before-departure, front-loaded with a Tatkal/last-minute spike) — this reproduces "days before journey" from Part 4 without inventing an extra independent variable.

## 6. WL / RAC / reservation modelling approach (the part the brief specifically flags)

This is a **capacity allocation process**, not a sum:

1. Each (train, class, date) has a fixed berth **capacity C** and an **RAC buffer size R** (RAC berths, e.g. shared side-lower berths — **[FACT]** RAC holders share a berth, typically two per side-lower, per the reservation-chart mechanics described by IndiaMike/RailRestro sources below).
2. Bookings arrive over the lead-time window (via the booking curve above) and are allocated **in arrival order** (FCFS, matching real PRS sequential allocation — **[FACT, cited below]**):
   - positions `1..C` → **Confirmed**
   - positions `C+1..C+R` → **RAC**
   - positions `>C+R` → **Waitlist**, with sub-quota classification (GNWL/TQWL/RLWL) as an **[ASSUMPTION]** simplification we may add later — MVP uses a single WL pool.
**Scope correction (as built):** the implemented model is the *lean* version —
capacity-constrained FCFS allocation with an aggregate cancellation rate driving
WL -> RAC -> CNF promotion. Waitlist remains emergent from the capacity wall,
which is the load-bearing property. Steps 3-4 below describe the fuller model,
deliberately deferred as scope we chose not to buy.

3. **Cancellations** arrive as their own thinning process on already-booked passengers (each booked passenger has a small hazard of cancelling before departure, higher closer to booking, lower near departure — a decreasing hazard, modelled as a Weibull/Gamma survival curve). Each cancellation **frees one confirmed/RAC slot**, which triggers **promotion**: the earliest-positioned WL passenger is promoted to RAC/Confirmed, and the earliest RAC passenger is promoted to Confirmed. This reproduces the real "WL → RAC → Confirmed" promotion chain **[FACT, cited]**.
4. **Chart preparation** (a fixed offset before departure — real system ~4 hours, but "up to 8 hours" is cited loosely in press coverage; we treat the exact offset as a **[SYNTH]** configurable parameter, not a fact we assert precisely) freezes the list; any online e-ticket still on WL is auto-cancelled **[FACT, cited]**, generating "no-show avoided" and a final confirmed/RAC/WL split that feeds station workload (a WL-heavy train correlates with more day-of-departure enquiry/re-booking work at the counter — this is the operational-workload link Part 4 wants).

This makes **WL emergent from the booking+capacity process** exactly as asked, and gives a mathematically defensible (if simplified) generative mechanism instead of a formula.

Sources: [Indian Railway Reservation System Algorithm (Medium)](https://medium.com/@hariv.krish47/indian-railway-reservation-system-algorithm-part-1-625961a470dc) · [Reservation against Cancellation (Wikipedia)](https://en.wikipedia.org/wiki/Reservation_against_Cancellation) · [WL to RAC to Confirmed — Republic World](https://www.republicworld.com/india/indian-railways-reservation-chart-process-ticket-confirmation-timings) · [RAC and Waitlists explained — IndiaMike](https://www.indiamike.com/india-articles/indian-railways-rac-and-indian-railways-waitlists/) · [RAC/GNWL/PQWL/RLWL/TQWL — RailRestro](https://www.railrestro.com/blog/rac-gnwl-pqwl-rlwl-tqwl)

## 7. Unreserved / platform-ticket modelling approach

Unreserved-class demand and platform-ticket sales are modelled as **NB counts driven by a shared "station pressure" latent factor** that also loads (with a smaller, delayed coefficient) onto reserved-class footfall — e.g. `station_pressure(t) = f(scheduled arrivals/departures in window, day-type, local event flag)`, and each of {unreserved demand, platform tickets, general footfall} is `NB(mean = base_rate × g(station_pressure), dispersion)` with its own base rate and a small idiosyncratic AR(1) noise term so they're correlated but not identical. This is a lightweight Gaussian-copula-style coupling via a shared factor rather than a full copula library — sufficient for MVP, documented as an upgrade path.

## 8. Workforce modelling approach

Generic `Worker` entity: `id, role_id, department_id, team_id, home_location_id, skills: {skill_id: proficiency ∈[0,1]}, qualifications: set, shift_pattern, capacity_units_per_shift, hourly_cost, overtime_cost_mult, fatigue: float∈[0,1], absence_today: bool, current_task_id, workload_units`. Eligibility for a task = qualification subset match + skill proficiency ≥ task's minimum + location within task's allowed radius + on-shift + not absent. Absence: Beta-Binomial hierarchical (§4). Fatigue accumulates with consecutive workload above a threshold and decays on rest; above a fatigue threshold, absence probability and task-duration multiplier increase (**[ASSUMPTION]**, small realistic touch, not overbuilt). Overtime = allowing capacity beyond nominal shift at a cost multiplier and a fatigue penalty — this is the "activate reserve workforce / overtime" action lever from Part 12/13.

**[REC]** Borrow from mature WFM (NICE/Genesys, §16) conceptually: forecast → required-staffing translation → shrinkage (absence + non-productive time) → multi-skill assignment. We deliberately **do not** reimplement their scheduling optimisers (Erlang-C-style curve fitting, ARIMA ensembles) — that's their product; our research value is in the *RL agent replacing their scheduling/reforecasting loop under disruption*, not in re-deriving classical WFM forecasting.

## 9. Process modelling approach

A `Process` is a directed graph of `Activity` nodes with precedence edges (supports parallel branches), each activity specifying: required skill(s) + minimum proficiency, duration distribution (Lognormal/Gamma, parameterised per activity), a rework/failure probability (loops back to an earlier activity or escalates), and priority/deadline inherited or overridden from the originating `Task`. Two canonical process templates ship in MVP config: an operations/maintenance process (Inspection → Defect → Approval → Repair → Verification → Closure) and a passenger/commercial process (e.g. Crowd surge → Additional counter/checking staffing → Resolution). Every activity start/end emits an event; the event log is the process graph's execution trace — this is what makes PM4Py-compatible mining possible "for free" from the engine rather than as a bolt-on.

## 10. Disruption modelling approach

Independent baseline Poisson processes per disruption type (absence spike, equipment failure, infra failure, train delay, demand spike) calibrated per asset/location, **plus** a Hawkes-style excitation term: an active disruption temporarily raises the rate of related disruption types at the same or downstream location (equipment failure → raised probability of a second related failure and of a maintenance backlog; a train delay → raised station-pressure rate downstream). Severity ~ heavy-tailed (Lognormal with occasional Pareto-tail draws for the "rare severe" requirement). Every disruption is implemented as **a first-class event that spawns Task(s)** through the same Process/Task machinery as normal work — there is no separate "disruption handler," which is what makes cascades (Part 14's example) fall out naturally instead of being scripted.

## 11. Drift / non-stationarity approach

Three orthogonal knobs, all seeded/config-driven for reproducibility: (1) **deterministic trend** (year-over-year demand_multiplier growth curve in config), (2) **regime-switching** (hidden-Markov jumps between named regimes — baseline/festival/disrupted — applied to intensity and disruption-rate parameters), (3) **scenario overrides** (Part 19's 8 scenarios = named config presets combining multipliers). Gradual drift = slow trend; sudden shift = regime switch; both are logged so downstream analysis can locate change points.

## 12. State estimation approach

MVP: the RL observation is a set of **sufficient statistics of true state** (queue lengths, counts, worker availability) — effectively fully observed for now, which is an explicit, documented simplification (**[ASSUMPTION]**), not an oversight. The engine still maintains a genuine `s_t` (latent demand intensity, true equipment health, true absence cause) separate from `o_t` (what gets logged/observed), so a later POMDP variant only has to change *what's exposed to the agent*, not the underlying generative model. Formal target for a future iteration: `s_{t+1} = F(s_t, a_t, exogenous_t, ε_t)`, `o_t = H(s_t) + η_t`, i.e. a nonlinear non-Gaussian state-space model — a particle filter or learned encoder would sit between `H` and the agent.

## 13. Existing simulation frameworks

- **SimPy** — process-based, generator/`yield`-driven DES in pure Python; minimal core, you build stats/plots yourself; large community, long track record of being paired with RL (NREL's DSS-SimPy-RL, multiple production-scheduling Gym wrappers). **[FACT, cited]**
- **salabim** — also pure-Python DES, doesn't require `yield`-based generators, ships built-in animation/statistics/monitors; smaller but very actively maintained by a single author. **[FACT, cited]**
- **AnyLogic, Arena, FlexSim, JaamSim** — GUI-first / commercial or Java-based DES tools, strong for enterprise process simulation and 3D animation, weak for tight programmatic RL-loop integration and not seed-reproducible in the way a scripted Python stack is. **[INFERENCE]** — not evaluated hands-on for this project; ruled out primarily on "programmatic RL integration + free + scriptable" grounds.

**[REC]** Use **SimPy** as the engine for Environment A: generator-based process modelling maps directly onto Worker/Task/Queue semantics, it's free, scriptable, has direct precedent for Gym-style step/reset wrapping, and is the more citable/standard choice for a research write-up than salabim. We do **not** reimplement DES primitives from scratch — that would be scope creep with no research payoff (Part 24's own principle). Environment B does not need a full DES engine at all (see §24 comparison) — it needs the same stochastic generator modules plus a much lighter reactive queue/consequence layer, which we build directly (no framework needed there).

Sources: [SimPy/DES+RL search summary — NREL DSS-SimPy-RL](https://github.com/NREL/DSS-SimPy-RL) · [salabim GitHub](https://github.com/salabim/salabim) · [SimPy vs Salabim comparison — School of Simulation](https://www.schoolofsimulation.com/blog_posts/simpy-vs-salabim-simulation-comparison)

## 14. Existing railway simulators

No open, general-purpose "Indian Railways workforce RL simulator" was found in the research pass — this is a genuine gap, which is consistent with this being a legitimate research contribution rather than a reimplementation. What *does* exist and is directly relevant as prior art for the DES+RL pattern we're adopting: production-scheduling Gym/Gymnasium environments built on discrete-event simulation, e.g. a ScienceDirect paper "Modeling Production Scheduling Problems as Reinforcement Learning Environments based on Discrete-Event Simulation and OpenAI Gym," and open repos like `spaceVStab/Discrete-Event-Simulator` (job scheduling for RL) and `fareskhlifi/Intelligent-Scheduling-using-Reinforcement-learning-and-Deep-Q-Networks`. **[FACT, cited, not railway-specific]**

Sources: [Modeling Production Scheduling Problems as RL Environments — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2405896321008399) · [Discrete-Event-Simulator (GitHub)](https://github.com/spaceVStab/Discrete-Event-Simulator) · [Intelligent-Scheduling-using-RL-and-DQN (GitHub)](https://github.com/fareskhlifi/Intelligent-Scheduling-using-Reinforcement-learning-and-Deep-Q-Networks)

## 15. Existing WFM software

**NICE WFM**: forecasting via curve fitting + 3 exponential-smoothing variants + ARIMA, auto-selects best fit; configurable shrinkage, multi-skill/multi-site, "what-if" alternate forecasts. **Genesys WFM**: ~25 ML-trained forecast models, forecasts up to two years out, explicit shrinkage management, multi-skill/multi-queue optimisation for agents handling multiple interaction types. **[FACT, cited]**

**What we borrow [REC]:** the forecast→required-staffing→shrinkage→multi-skill-assignment *pipeline shape*, and the idea of a "what-if"/scenario mechanism (→ our scenario configs, Part 19). **What we simplify [REC]:** we do not build an ARIMA/ML forecast ensemble — our RL agent's job is exactly the decision these products hand to a human scheduler, so over-forecasting would blunt the research question. **What's research-worthy [REC]:** using RL to do *intraday reforecasting + reallocation under disruption*, which commercial WFM handles with rule-based reforecast triggers, not learned policies — that gap is our contribution.

Sources: [NICE WFM review](https://getvoip.com/blog/nice-workforce-management/) · [Genesys Workforce Scheduling/Forecasting](https://www.genesys.com/capabilities/workforce-scheduling-forecasting) · [Genesys Multi-Forecasting Primer](https://all.docs.genesys.com/PEC-WFM/Current/Administrator/MltiForecst)

## 16. Existing process-mining tools

**PM4Py**: the standard Python process-mining library; minimal required event-log schema is `case:concept:name` (case id), `concept:name` (activity), `time:timestamp`, with optional resource/role/cost attributes for filtering and annotation. XES is the underlying XML standard log format. **[FACT, cited]** We generate a CSV event log with these three mandatory columns plus `role, team, location, resource_id, start, end, queue_time, processing_time, status, outcome` so it imports into PM4Py (and, if desired, ProM/Celonis) with a trivial column-rename step. Celonis/BPI Challenge datasets are referenced as the field's standard benchmarks but are not needed at runtime.

Sources: [PM4Py primer — Medium](https://malinian.medium.com/unmasking-the-truth-process-mining-with-pm4py-6664c21e296f) · [PM4Py: Bridging Process- and Data Science (arXiv:1905.06169)](https://arxiv.org/pdf/1905.06169)

## 17. Relevant GitHub repositories

- [NREL/DSS-SimPy-RL](https://github.com/NREL/DSS-SimPy-RL) — SimPy-based cyber-physical RL environment; direct precedent for "SimPy + Gym" pattern.
- [salabim/salabim](https://github.com/salabim/salabim) — alternative DES engine, considered and set aside (§13).
- [spaceVStab/Discrete-Event-Simulator](https://github.com/spaceVStab/Discrete-Event-Simulator) — job-scheduling DES for RL, precedent for reset/step/possibleAction shape.
- [fareskhlifi/Intelligent-Scheduling-using-Reinforcement-learning-and-Deep-Q-Networks](https://github.com/fareskhlifi/Intelligent-Scheduling-using-Reinforcement-learning-and-Deep-Q-Networks) — Gymnasium scheduling env precedent.
- [fit-alessandro-berti/pm-manuals-for-llms](https://github.com/fit-alessandro-berti/pm-manuals-for-llms) — PM4Py reference manual, useful for building the event-log exporter correctly.

Licenses were not individually re-verified line-by-line in this pass; **[REC]** confirm each repo's LICENSE file before copying any code (we are not vendoring any of them — they're used as design precedent only).

## 18. Relevant academic papers

- Excess demand in public transportation — Pittsburgh Port Authority, [arXiv:2208.06372](https://arxiv.org/pdf/2208.06372) — overdispersion in transit demand, motivates NB.
- "From the multi-terms urn model to the self-exciting negative binomial distribution and Hawkes processes," [arXiv:2112.14942](https://arxiv.org/pdf/2112.14942) — formal bridge between NB and Hawkes, underlies §4/§6/§10.
- "Hawkes Models and Their Applications" (Annual Reviews) and survey [arXiv:2405.10527](https://arxiv.org/html/2405.10527v1) — general Hawkes background.
- "Steady-State Analysis and Online Learning for Queues with Hawkes Arrivals," [arXiv:2311.02577](https://arxiv.org/pdf/2311.02577) — queueing theory under self-exciting arrivals, relevant to station-queue dynamics under a demand surge.
- "Modeling Production Scheduling Problems as Reinforcement Learning Environments based on Discrete-Event Simulation and OpenAI Gym" ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2405896321008399)) — direct DES+Gym architectural precedent.
- Action-masking / constrained RL for scheduling: [arXiv:2601.09293](https://arxiv.org/abs/2601.09293) (masked PPO for job-shop scheduling under uncertainty), [arXiv:2305.13824](https://arxiv.org/pdf/2305.13824) (constrained RL for dynamic material handling), [arXiv:2504.02662](https://arxiv.org/pdf/2504.02662) (action masking for OR problems) — underlies Part 13's "hard constraints, not weak penalties" requirement (§19 below).
- "Process Mining for Python (PM4Py)," [arXiv:1905.06169](https://arxiv.org/pdf/1905.06169).

## 19. Relevant datasets

- **data.gov.in** "Summary of Railway Statistics" and the "Statistics of Indian Railways" collection (via dataful.in mirror) — aggregate national figures: route length, passengers carried, staff employed, punctuality index. **[FACT, cited]** Useful for *plausibility calibration only* (order-of-magnitude checks), not per-station ingestion.
- **Southern Railway structure** — 6 divisions (Chennai, Tiruchirappalli, Madurai, Thiruvananthapuram, Palakkad, Salem), ~727 stations, ~5,081 route-km, HQ Chennai. **[FACT, cited, Wikipedia — cross-check against Southern Railway's official site before hard-coding into config]**
- No IRCTC live data is used; all reservation dynamics are our own generative model (§6), *inspired by* publicly documented PRS mechanics, not scraped from IRCTC.

Sources: [Summary of Railway Statistics — data.gov.in](https://www.data.gov.in/catalog/summary-railway-statistics) · [Statistics of Indian Railways — dataful.in](https://dataful.in/collections/554/) · [Southern Railway zone — Wikipedia](https://en.wikipedia.org/wiki/Southern_Railway_zone)

## 20. Recommended technology stack

Python 3.11+, **SimPy** (Environment A engine), **Gymnasium** (RL interface for both environments), NumPy/SciPy (distributions: Poisson, NegBinom via `scipy.stats.nbinom`, Gamma, Lognormal), **pandas** (event logs, CSV I/O), **PyYAML** (scenario configs), **pytest** (validation/unit tests), matplotlib (validation plots). Optional, later: `pm4py` for process-mining analysis of generated logs, `stable-baselines3` for the PPO/DQN baseline once the heuristic baseline is working (Part 23 explicitly says start with random/heuristic first).

## 21. Architecture

```
REAL-WORLD PUBLIC STATS  ──►  config/scenarios/*.yaml  (calibration targets, not runtime deps)
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                      ▼
        simulation/  (shared generative core)      data/generators (batch driver)
  demand/  workforce/  processes/  disruptions/            │
  resources/  events/  state/  metrics/                    ▼
                     │                          data/*.csv  (Environment B dataset)
                     ▼                                      │
        core/engine.py (SimPy DES loop)                     ▼
                     │                          rl/environment/env_b.py (reactive replay + consequence engine)
                     ▼
        rl/environment/env_a.py (Gymnasium wrapper, pauses at decision epochs)
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  rl/observation/ rl/action/   rl/reward/  (shared by A and B)
                     │
        experiments/runners  →  analysis/validation, analysis/plots
```

Both environments import the *same* `simulation.demand`, `simulation.workforce`, `simulation.processes`, `simulation.disruptions` modules — that shared core is what makes the A/B comparison meaningful instead of two unrelated codebases.

## 22–23. Environment A and B design

**Environment A (model-driven DES).** A SimPy `Environment` drives the clock. Demand/disruption modules are SimPy processes that generate events (booking arrival, failure, absence) on their own probabilistic schedules; each event may spawn Task processes through the shared Process-graph engine; Tasks request Worker resources via SimPy `Resource`/`Store` with skill/qualification filtering. Whenever a decision epoch condition is met (new unassigned task, worker freed, escalation) the SimPy process yields control back to the Gymnasium wrapper, which exposes `obs`, waits for `action`, applies it (assign/reassign/defer/overtime), and resumes the clock. Because everything downstream of an action is generated live, **action-dependent cascades are possible** (e.g., deferring a task can itself raise the probability of a secondary failure) — this is A's core strength.

**Environment B (synthetic-dataset-driven).** Run the *same* generative modules once, offline, under a fixed baseline heuristic policy, with full logging, to materialise the CSVs (`workers.csv`, `stations.csv`, `trains.csv`, `demand.csv`, `reservations.csv`(confirmed/RAC/WL), `unreserved.csv`, `platform_tickets.csv`, `tasks.csv`, `process_events.csv`, `failures.csv`, `delays.csv`, `absences.csv`, `outcomes.csv`). The RL environment for B replays the exogenous stream (arrivals, failures, absences — these are **fixed**, not re-simulated) in timestamp order and runs a lightweight **reactive consequence engine** (queueing math + task-completion bookkeeping, no SimPy) that recomputes queue lengths, wait times and completion given the *agent's* assignment choices. This is what stops B from being "just CSVs" — the agent's actions still causally change downstream KPIs — but exogenous shocks can't be altered by the agent's choices the way they can in A. B trades that closed-loop realism for reproducibility and much higher rollout throughput (no per-step Python-generator DES overhead), which matters for offline RL / large sweep experiments.

## 24. Comparison of A vs B

| Dimension | A (model-driven DES) | B (synthetic-dataset-driven) |
|---|---|---|
| Realism of cascades | High — actions can change future exogenous draws | Partial — exogenous stream fixed at generation time |
| Reproducibility | Reproducible given seed, but a changed action can change the entire future trajectory (butterfly effect), complicating apples-to-apples policy comparison | Very high — identical exogenous stream for every policy, clean counterfactual comparison |
| Speed / scalability | Slower (SimPy per-event overhead) | Fast — vectorisable, good for large sweeps / offline RL |
| Ability to create unseen scenarios | Native — change config, rerun | Requires regenerating the dataset |
| Drift modelling | Native (regime changes mid-episode) | Must be baked into the pre-generated file, less flexible mid-episode |
| Event-log / process-mining output | Yes | Yes (arguably easier, since it's already tabular) |
| Research use-case fit | Studying agent-influenced cascades, closed-loop policy effects | Benchmarking/comparing many policies fairly on the identical world, faster iteration, easier offline-RL dataset use |

**[REC] Environment A should be the main research environment** — it is the only one of the two that honours Part 14's cascading-consequence requirement (actions changing the future world), which is the actual scientific question ("dynamic workforce optimisation under changing demand, uncertainty... and disruptions"). Environment B remains valuable as a **fast, fair benchmarking harness** (fixed exogenous stream ⇒ clean policy comparison) and as the natural offline-RL / process-mining dataset source — keep both, use B for large-scale policy comparison sweeps and A for the headline closed-loop results.

## 25. Validation methodology

Compare simulated aggregate behaviour against the public statistics in §19 at the *order-of-magnitude / distributional-shape* level only (mean, variance/dispersion, weekly seasonality, autocorrelation, tail behaviour of delays/backlogs) — never claim point-level accuracy, since the dataset is synthetic by design. Sanity checks: conservation (every task has a valid lifecycle end-state), capacity never exceeded, WL never negative, cancellation-promotion chain never promotes past capacity, queue lengths stable (not diverging) under baseline scenario. These become `analysis/validation` scripts run against each generated scenario.

## 26. Sim-to-real / reality-gap methodology

Because we deliberately never train on real operational data, "sim-to-real" here means: (1) documenting every **[SYNTH]** parameter explicitly so a future researcher can swap in calibrated values, (2) keeping the generative *mechanisms* (capacity-constrained booking, cascading disruptions) structurally faithful even where exact rates are invented, (3) reporting results as "plausible under this class of world," not "true of Southern Railway" — the reality gap is managed by honesty about assumption boundaries, not by chasing unavailable ground truth.

## 27. MVP scope

One division-scale world (e.g. Chennai division subset: ~10–15 stations, ~8–12 train services, 4 classes), 6 departments (Operations, Commercial, Maintenance, Support, IT, plus a generic "Other"), ~150–300 workers, 2 process templates (maintenance lifecycle, commercial/crowd-response), 5 disruption types, one clear workforce-assignment action space (assign/reassign/overtime/defer over a bounded worker pool), Gymnasium envs for A and B, event-log export, 3–4 of the 8 scenarios (normal weekday, weekend, high-demand, disruption) with multi-seed runs, random + greedy-heuristic baselines, one validation script, one plotting script.

## 28. Research-grade scope (post-MVP)

Full 6-division Southern Railway config, all 8 scenarios × many seeds with confidence intervals, POMDP belief-state observation variant, PPO/maskable-PPO baseline compared against heuristics, Hawkes-process disruption cascades fully parameterised, drift/regime experiments across simulated "years," PM4Py-based process-mining analysis of generated logs, formal validation report against public statistics.

## 29. Risks and limitations

- Every rate/probability not sourced from §19's aggregate stats is invented (**[SYNTH]**) — results are about *mechanism*, not Southern Railway's true operations.
- SimPy's per-event Python overhead limits Environment A's scalability for very large worlds or huge scenario sweeps — mitigated by using Environment B for sweeps.
- Action-dependent cascades in A make naive seed-controlled A/B policy comparison harder (different actions ⇒ different futures) — mitigate with paired-seed exogenous-process replay where feasible, or accept variance and report confidence intervals.
- Scope discipline: the 25-part brief is large enough that "finish everything" is not a realistic single-session goal — the MVP (§27) is the actually-shippable, defensible slice.

## 30. Final recommendation

Build the shared generative core first (demand/workforce/process/disruption modules + generic entities + config schema), then Environment A (SimPy + Gymnasium) as the primary research environment, then Environment B as a faster benchmarking/offline-data harness reusing the same core. Ship the MVP scope (§27) as real, runnable code with tests and one validation pass, rather than attempting full coverage of all 25 parts shallowly.


---

## 32. Synthetic feedback / outcome layer (added after initial implementation)

NPS, CSAT, operational costs, employee performance metrics and work schedules are
**sparse or not consistently available in public railway data**. Rather than
pretend otherwise, the simulator generates them as *synthetic instruments*, fully
configurable under `feedback:` in each scenario file. **[REC]** They must never be
presented as IRCTC- or Indian-Railways-sourced.

**Customer feedback (NPS/CSAT) — sparse, delayed observations, not dense rewards.**
Modelled as an observation process on a latent service-experience variable
`e in [0,1]`, itself a function of wait time relative to SLA, SLA breach, station
crowding, exposure to active delay, server proficiency and fatigue, and rework.
Sampling is sparse (~6% base response rate) and **biased** (dissatisfied customers
respond more, via a configurable `detractor_response_boost`), and every response
carries a lognormal delivery delay. `FeedbackRegistry.observed()` filters on
delivery time, so nothing can be seen before it arrives. **[ASSUMPTION]** CSAT
maps linearly from `e`; NPS maps through a convex exponent so it is harsher at the
same experience level, keeping the two instruments non-redundant.

Measured behaviour on the reference scenario confirms the instrument is doing its
job: population latent experience 0.666, responders' latent experience 0.595
(non-response bias), observed CSAT 3.42 from 97 delivered responses out of 1,418
interactions. An agent reading observed CSAT reads a biased, lagged, small sample —
which is the realistic case, and the reason NPS/CSAT should never be wired up as a
per-timestep reward.

**Operational cost — an outcome signal.** Unlike feedback, cost is known exactly
and immediately. Decomposed into labour, overtime, reserve activation, SLA
penalty, disruption response, idle capacity and rework, so the trade-off the agent
is actually making (quality bought with overtime vs reserves vs idle capacity)
stays visible. **[SYNTH]** Units are abstract cost units; a fabricated rupee figure
would imply precision we do not have.

**Employee performance — an outcome signal, periodic not live.** Derived from what
a worker did under the conditions they faced: efficiency against expected
duration, on-time rate, rework rate, all difficulty-adjusted (a worker handed hard
tasks while fatigued is not penalised for the difficulty). Records are cut at
shift boundaries; the open period is never observable, because a live per-task
quality readout is not something a real operator has.

**Schedules — workforce state.** A first-class `Roster` of `ScheduleEntry` objects
(planned window, absence, overtime authorisation) that drives availability, shift
boundaries and the staffing action space, rather than availability being
recomputed ad hoc from shift arithmetic.

**Latent/observed separation is structural.** `simulation/state/true_state.py`
holds what the world knows (latent experience, true asset health, true demand
intensity); the observation builder reads only from delivered surveys, closed
performance periods and realised counts. Seven validation invariants enforce this,
including "no undelivered surveys are observable" and "performance is periodic,
not per-task".

---

## 33. Corrections made during implementation (what the report got wrong)

Recorded because they are the substantive findings of the build phase:

1. **Hawkes process was supercritical.** The first implementation parameterised
   excitation by `alpha` directly and let neighbour spread add unbounded history
   entries. Result: 2,770 disruptions in one simulated day, 2,742 of them
   cascades, saturating the workforce. Fixed by parameterising on **branching
   ratio** (validated `< 1` at construction) and giving neighbour excitation
   *weighted* history entries so network spread cannot push the aggregate ratio
   over the stability threshold. Post-fix: 19 disruptions/day, 3 cascades.

2. **Workers could be double-booked.** `simpy.Environment.process()` only
   *schedules* a generator; the worker was not marked busy until the scheduler ran
   it, so the next decision epoch at the same instant could assign the same worker
   again. This inflated worker-minutes to 2.4x the sampled utilisation. Fixed by
   claiming the worker synchronously in `apply_dispatch`.

3. **Utilisation was measured against the whole roster**, including off-shift
   staff, making a busy system look idle (0.11 vs a true ~0.4).

4. **Shift allocation was uniform**, putting a third of staff on nights when
   demand is ~15% of peak. Now demand-weighted, with a floor for night cover.

5. **Role mix was uniform within departments**, starving specialist-dependent work
   (every turnaround needs a ticket examiner). Now configured to match the work mix.

6. **Greedy ranked candidates by skill before proximity**, dispatching specialists
   90 minutes across the network for 20-minute local jobs. Ranking is now a
   proximity-dominant composite, and a hard travel-distance constraint was added.

7. **Waitlist almost never formed** (2 train-classes) because base demand sat below
   capacity, making the WL pressure signal meaningless. Demand recalibrated so
   reserved classes run at/above capacity, as popular services do.

None of these were visible from the code alone; all were found by instrumenting
the world and asking why the numbers looked wrong. That is an argument for keeping
`analysis/validation/diagnose.py` in the loop as the world grows.
