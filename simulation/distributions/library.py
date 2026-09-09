"""Distribution helpers used across the generative core.

Each helper documents *why* that family fits the variable it models. See
RESEARCH_REPORT.md section 4 for the full justification table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def negative_binomial(rng: np.random.Generator, mean: float, dispersion: float) -> int:
    """Overdispersed count draw, parameterised by mean and dispersion k.

    Var = mean + mean^2 / k, so k -> inf recovers Poisson. Used for demand counts,
    where unobserved heterogeneity (route popularity, local events) makes the
    variance exceed the mean.
    """
    mean = max(float(mean), 1e-9)
    if dispersion <= 0 or not np.isfinite(dispersion):
        return int(rng.poisson(mean))
    p = dispersion / (dispersion + mean)
    return int(rng.negative_binomial(dispersion, p))


def lognormal_duration(rng: np.random.Generator, median: float, sigma: float) -> float:
    """Positive, right-skewed service time. Median is the geometric mean."""
    return float(rng.lognormal(mean=np.log(max(median, 1e-6)), sigma=max(sigma, 1e-6)))


def gamma_duration(rng: np.random.Generator, mean: float, shape: float) -> float:
    """Alternative positive duration with a lighter tail than lognormal."""
    shape = max(shape, 1e-6)
    return float(rng.gamma(shape=shape, scale=max(mean, 1e-9) / shape))


def beta_binomial_rate(rng: np.random.Generator, mean_rate: float, concentration: float) -> float:
    """Per-worker absence propensity drawn from Beta(a, b).

    Individual heterogeneity means the aggregate absence count is overdispersed
    relative to Binomial, which matches real rostering data better than a single
    shared probability.
    """
    mean_rate = float(np.clip(mean_rate, 1e-4, 1 - 1e-4))
    a = mean_rate * concentration
    b = (1.0 - mean_rate) * concentration
    return float(rng.beta(a, b))


def heavy_tailed_severity(
    rng: np.random.Generator, median: float, sigma: float, tail_prob: float, tail_alpha: float
) -> float:
    """Lognormal body with an occasional Pareto tail draw.

    Rare severe disruptions should not be truncated by a thin-tailed model; the
    mixture keeps typical severity realistic while admitting genuine outliers.
    """
    if rng.random() < tail_prob:
        return float(median * (1.0 - rng.random()) ** (-1.0 / max(tail_alpha, 1e-6)))
    return lognormal_duration(rng, median, sigma)


@dataclass
class AR1:
    """Log-scale AR(1) process giving temporal correlation to latent intensity.

    x_t = phi * x_{t-1} + eps,  eps ~ N(0, sigma).  Multiplier is exp(x_t).
    """

    phi: float = 0.7
    sigma: float = 0.15
    value: float = 0.0

    def step(self, rng: np.random.Generator) -> float:
        self.value = self.phi * self.value + float(rng.normal(0.0, self.sigma))
        return float(np.exp(self.value))


class HawkesProcess:
    """Self-exciting point process with an exponential kernel.

    Intensity: lambda(t) = mu + sum_i w_i * alpha * exp(-beta * (t - t_i))

    Parameterised by the **branching ratio** n rather than alpha directly, because
    n is the parameter with the stability meaning: alpha = n * beta makes the
    expected number of direct offspring per event exactly n. The process is
    stationary only for n < 1; at n >= 1 it explodes and produces an absurd number
    of events (a mistake worth guarding against explicitly, hence the assertion).

    Weights w_i let a neighbouring location's event contribute a damped share of
    excitation without adding a full unit of branching mass, so network spread
    cannot quietly push the total branching ratio over 1.
    """

    def __init__(self, mu: float, branching_ratio: float, beta: float):
        if not 0.0 <= branching_ratio < 1.0:
            raise ValueError(
                f"branching_ratio must be in [0, 1) for a stable process, got {branching_ratio}"
            )
        self.mu = float(mu)
        self.branching_ratio = float(branching_ratio)
        self.beta = float(beta)
        self.alpha = float(branching_ratio) * float(beta)
        self.history: list[tuple[float, float]] = []      # (time, weight)

    def intensity(self, t: float) -> float:
        if not self.history:
            return self.mu
        times = np.fromiter((h[0] for h in self.history), dtype=float, count=len(self.history))
        weights = np.fromiter((h[1] for h in self.history), dtype=float, count=len(self.history))
        ages = t - times
        valid = ages >= 0
        if not valid.any():
            return self.mu
        return float(self.mu + self.alpha * (weights[valid] * np.exp(-self.beta * ages[valid])).sum())

    def excitation(self, t: float) -> float:
        """The self-excited component alone, useful as an observable feature."""
        return self.intensity(t) - self.mu

    def record(self, t: float, weight: float = 1.0) -> None:
        self.history.append((float(t), float(weight)))

    def prune(self, t: float, lookback: float) -> None:
        """Drop events too old to contribute, keeping intensity evaluation cheap."""
        cutoff = t - lookback
        if self.history and self.history[0][0] < cutoff:
            self.history = [h for h in self.history if h[0] >= cutoff]
