"""Named, independent random streams derived from one scenario seed.

Each subsystem draws from its own stream so that changing the number of draws in
one subsystem (e.g. adding a worker attribute) does not shift the draws of any
other subsystem. Without this, paired-seed policy comparison is meaningless.
"""

from __future__ import annotations

import hashlib

import numpy as np

STREAM_NAMES = (
    "demand_latent",
    "demand_booking",
    "demand_unreserved",
    "workforce_population",
    "workforce_absence",
    "workforce_fatigue",
    "process_duration",
    "process_rework",
    "disruption_base",
    "disruption_hawkes",
    "disruption_severity",
    "observation_noise",
    "regime",
    "policy",
)


def _substream_seed(master_seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class RNGRegistry:
    """Container of named `numpy.random.Generator` streams."""

    def __init__(self, master_seed: int):
        self.master_seed = int(master_seed)
        self._streams: dict[str, np.random.Generator] = {}

    def stream(self, name: str) -> np.random.Generator:
        if name not in self._streams:
            self._streams[name] = np.random.default_rng(_substream_seed(self.master_seed, name))
        return self._streams[name]

    def __getattr__(self, name: str) -> np.random.Generator:
        if name.startswith("_"):
            raise AttributeError(name)
        return self.stream(name)

    def spawn(self, name: str) -> "RNGRegistry":
        """A child registry, for nesting (e.g. per-episode streams)."""
        return RNGRegistry(_substream_seed(self.master_seed, name) % (2**63))
