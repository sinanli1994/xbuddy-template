"""Paired comparison of two retrieval configurations over the same queries.

With ~30 queries a 0.03 MRR gap is one query changing rank, so a difference in the
means is not evidence on its own. Each query is scored under both configurations and
the per-query differences are resampled; if the 95% interval spans zero, the eval has
not distinguished the two. Seeded, so the same inputs always give the same interval.
"""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedResult:
    mean: float
    low: float
    high: float
    wins: int
    losses: int
    ties: int

    @property
    def significant(self) -> bool:
        return self.low > 0 or self.high < 0


def paired_bootstrap(
    a: list[float], b: list[float], *, resamples: int = 10_000, seed: int = 7
) -> PairedResult:
    """Mean of (a - b) with a percentile bootstrap 95% interval over queries."""
    if len(a) != len(b) or not a:
        raise ValueError("paired scores must be non-empty and the same length")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(resamples))
    return PairedResult(
        mean=sum(diffs) / n,
        low=means[int(0.025 * resamples)],
        high=means[int(0.975 * resamples) - 1],
        wins=sum(d > 1e-9 for d in diffs),
        losses=sum(d < -1e-9 for d in diffs),
        ties=sum(abs(d) <= 1e-9 for d in diffs),
    )
