"""
Budget guard for benchmark runs.

Tracks estimated and actual spend, and blocks calls that would exceed the
configured ceiling. All costs are in USD.
"""
from __future__ import annotations


class BudgetExceeded(Exception):
    """Raised when a call would push spend past the configured ceiling."""
    def __init__(self, spent: float, limit: float, estimate: float | None = None):
        self.spent = spent
        self.limit = limit
        self.estimate = estimate
        msg = (
            f"Budget ceiling ${limit:.2f} reached "
            f"(spent ${spent:.4f}"
            + (f", estimated this call ${estimate:.4f}" if estimate is not None else "")
            + ")."
        )
        super().__init__(msg)


class BudgetGuard:
    """
    Thread-safe-ish budget tracker for sequential benchmark runs.

    Usage:
        guard = BudgetGuard(limit_usd=10.0)
        guard.check_before_call(estimated_cost=0.002)  # raises BudgetExceeded if over
        ...make the call...
        guard.record(actual_cost=0.0018)
    """

    def __init__(self, limit_usd: float = 10.0):
        if limit_usd <= 0:
            raise ValueError("Budget limit must be positive.")
        self._limit = limit_usd
        self._spent = 0.0
        self._calls = 0

    @property
    def limit(self) -> float:
        return self._limit

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return max(0.0, self._limit - self._spent)

    @property
    def calls_recorded(self) -> int:
        return self._calls

    @property
    def is_exhausted(self) -> bool:
        return self._spent >= self._limit

    def check_before_call(self, estimated_cost: float | None) -> None:
        """
        Raise BudgetExceeded if the estimated cost would push total past ceiling.

        A None estimated_cost (unknown pricing) is allowed through — the actual
        cost will be recorded after the call.
        """
        if self.is_exhausted:
            raise BudgetExceeded(self._spent, self._limit)
        if estimated_cost is not None and (self._spent + estimated_cost) > self._limit:
            raise BudgetExceeded(self._spent, self._limit, estimated_cost)

    def record(self, actual_cost: float | None) -> None:
        """Record the actual cost from a completed call."""
        if actual_cost is not None and actual_cost > 0:
            self._spent += actual_cost
        self._calls += 1

    def summary_line(self) -> str:
        return (
            f"Budget: ${self._spent:.4f} spent / ${self._limit:.2f} limit "
            f"({self.remaining:.4f} remaining, {self._calls} calls recorded)"
        )
