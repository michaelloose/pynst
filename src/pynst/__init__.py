"""PyNST: plan, execute, persist, and inspect nested parameter sweeps."""

from __future__ import annotations

import sys as _sys

from ._version import __version__


# ``pynst.sweep_manager`` used to be the implementation module and is patched
# directly by some downstream tests and integrations.  Install a true module
# alias so those patches still affect the canonical module globals.
from .execution import manager as sweep_manager  # noqa: E402

_sys.modules[f"{__name__}.sweep_manager"] = sweep_manager

from .data import (  # noqa: E402
    BaseSweepDataset,
    BlockMetadata,
    GenericSweepDataset,
    SweepMetadata,
    VariableMetadata,
    insert_row_into_df,
)
from .execution.manager import (  # noqa: E402
    CriticalMeasurementError,
    OnDiskChunkManager,
    StorageCommitError,
    SweepManager,
)

__all__ = [
    "BaseSweepDataset",
    "BlockMetadata",
    "CriticalMeasurementError",
    "GenericSweepDataset",
    "OnDiskChunkManager",
    "StorageCommitError",
    "SweepManager",
    "SweepMetadata",
    "VariableMetadata",
    "insert_row_into_df",
]
