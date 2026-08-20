"""Exceptions raised while executing nested sweeps."""

from __future__ import annotations


class CriticalMeasurementError(Exception):
    """Abort the complete measurement sweep after safely flushing data."""


class StorageCommitError(CriticalMeasurementError):
    """Abort because measurement data could not be committed safely."""


__all__ = ["CriticalMeasurementError", "StorageCommitError"]
