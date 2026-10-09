"""Tests for logging subsystem."""
import logging

from theriad.core.logging.logger import get_logger


def test_get_logger_returns_child():
    logger = get_logger("test")
    assert logger.name == "theriad.test"
    assert isinstance(logger, logging.Logger)
