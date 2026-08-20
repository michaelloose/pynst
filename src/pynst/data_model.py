"""Compatibility imports for the pre-0.4 metadata module path."""

from .data.model import (
    BlockMetadata,
    SweepMetadata,
    VariableMetadata,
    dumps_metadata,
    json_default,
    loads_metadata,
)

__all__ = [
    "BlockMetadata",
    "SweepMetadata",
    "VariableMetadata",
    "dumps_metadata",
    "json_default",
    "loads_metadata",
]
