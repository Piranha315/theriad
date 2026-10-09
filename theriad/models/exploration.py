"""
Epsilon-greedy exploration for within-tier candidate selection.

The problem
-----------
The resolver only ever observes the outcome of the candidate it picked. If the
registry's prior says candidate B is worse than A, B is never selected, so no
evidence about B is ever collected, so the prior is never corrected. The router
converges on whatever it believed on day one — including any authoring mistake
in the registry, and including models that have since improved.

The mechanism
-------------
With probability ``epsilon``, ignore the score ranking and pick uniformly at
random from the other candidates in the tier ("explore"). Otherwise take the
top-scored candidate as usual ("exploit"). Epsilon is the fraction of traffic
knowingly spent buying information rather than the current best guess.

Cost of exploration is bounded and priceable: it is at most ``epsilon`` of
requests routed within the same tier, so it inherits the tier's cost band. It
does not cross tiers — exploration answers "which candidate in this band", not
"which band", because the band is a cost decision the routing layers already
made on evidence.

Defaults and safety
-------------------
- ``epsilon=0.0`` (off) is the default everywhere. Exploration must be switched
  on deliberately; a router that silently makes 5% of decisions at random is a
  debugging nightmare nobody asked for.
- Exploration is skipped when a tier has only one candidate.
- ``PolicyResolution.exploration_applied`` records whether a given decision was
  an explore step. **Outcome logs must carry this flag.** Analysing exploit rows
  as if they were a random sample is the classic way to conclude that your
  current policy is optimal — the data was selected by that very policy.
- ``rng`` is injectable so tests are deterministic. Production callers can leave
  it None to get module-level randomness.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ExplorationConfig:
    """Configuration for epsilon-greedy candidate exploration.

    Args:
        epsilon: Probability in [0, 1] of picking a non-top candidate. 0 disables
                 exploration entirely (the default).
        rng:     Optional random source. Inject a seeded ``random.Random`` for
                 reproducible tests; None uses the module-level generator.
    """
    epsilon: float = 0.0
    rng: random.Random | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")

    @property
    def enabled(self) -> bool:
        return self.epsilon > 0.0

    def should_explore(self) -> bool:
        """Return True when this decision should be an exploration step."""
        if not self.enabled:
            return False
        source = self.rng or random
        return source.random() < self.epsilon

    def choose_index(self, n_candidates: int, exploit_index: int) -> int:
        """Pick a candidate index to explore, excluding ``exploit_index``.

        Returns ``exploit_index`` unchanged when there is nothing else to try.
        """
        if n_candidates <= 1:
            return exploit_index
        source = self.rng or random
        alternatives = [i for i in range(n_candidates) if i != exploit_index]
        return source.choice(alternatives)


NO_EXPLORATION = ExplorationConfig()
"""Shared default: exploration disabled."""
