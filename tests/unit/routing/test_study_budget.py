from pathlib import Path

import pytest

from theriad.routing.study_budget import StudyBudgetExceeded, StudyBudgetLedger


def test_non_test_calls_cannot_consume_test_reserve(tmp_path: Path):
    ledger = StudyBudgetLedger(tmp_path / "budget.json")
    ledger.reserve(phase="train", model="a", maximum_cost_usd=15.0)
    with pytest.raises(StudyBudgetExceeded, match="test reserve"):
        ledger.reserve(phase="validation", model="b", maximum_cost_usd=0.01)
    ledger.reserve(phase="test", model="b", maximum_cost_usd=10.0)
    assert ledger.status()["remaining_usd"] == 0.0


def test_settlement_releases_unused_authorization(tmp_path: Path):
    ledger = StudyBudgetLedger(tmp_path / "budget.json")
    reservation = ledger.reserve(phase="train", model="a", maximum_cost_usd=5.0)
    ledger.settle(reservation, 1.25)
    assert ledger.status()["committed_usd"] == 1.25


def test_actual_cost_cannot_exceed_reserved_maximum(tmp_path: Path):
    ledger = StudyBudgetLedger(tmp_path / "budget.json")
    reservation = ledger.reserve(phase="test", model="a", maximum_cost_usd=1.0)
    with pytest.raises(StudyBudgetExceeded, match="pre-authorized"):
        ledger.settle(reservation, 1.01)


def test_corrupt_ledger_fails_closed(tmp_path: Path):
    path = tmp_path / "budget.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(StudyBudgetExceeded, match="unreadable"):
        StudyBudgetLedger(path).reserve(phase="train", model="a", maximum_cost_usd=0.1)
