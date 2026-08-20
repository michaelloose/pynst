"""Compatibility coverage for the package split introduced in PyNST 0.4."""

from __future__ import annotations

import importlib
import sys


def test_legacy_sweep_manager_is_the_canonical_module() -> None:
    canonical = importlib.import_module("pynst.execution.manager")
    legacy = importlib.import_module("pynst.sweep_manager")

    assert legacy is canonical
    assert sys.modules["pynst.sweep_manager"] is canonical
    assert legacy.SweepManager is canonical.SweepManager
    assert legacy.OnDiskChunkManager is canonical.OnDiskChunkManager
    assert legacy.CriticalMeasurementError is canonical.CriticalMeasurementError


def test_legacy_manager_monkeypatch_reaches_canonical_globals(
    monkeypatch,
) -> None:
    canonical = importlib.import_module("pynst.execution.manager")
    legacy = importlib.import_module("pynst.sweep_manager")
    sentinel = object()

    monkeypatch.setattr(legacy, "tqdm", sentinel)

    assert canonical.tqdm is sentinel


def test_legacy_data_modules_reexport_identical_objects() -> None:
    legacy_model = importlib.import_module("pynst.data_model")
    model = importlib.import_module("pynst.data.model")
    legacy_dataset = importlib.import_module("pynst.dataset")
    dataset = importlib.import_module("pynst.data.dataset")
    legacy_utilities = importlib.import_module("pynst.utilities")
    frame = importlib.import_module("pynst.data.frame")

    assert legacy_model.BlockMetadata is model.BlockMetadata
    assert legacy_model.SweepMetadata is model.SweepMetadata
    assert legacy_model.VariableMetadata is model.VariableMetadata
    assert legacy_dataset.BaseSweepDataset is dataset.BaseSweepDataset
    assert legacy_dataset.GenericSweepDataset is dataset.GenericSweepDataset
    assert (
        legacy_utilities.insert_row_into_df
        is frame.insert_row_into_df
    )
