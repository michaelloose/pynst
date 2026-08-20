"""Shared metadata model for generic nested-sweep measurement files.

The model deliberately contains no domain-specific assumptions. A PA, pulsed,
bias, temperature or other measurement library may build a specialised
Dataset class on top of these structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def json_default(value: Any) -> Any:
    """Convert common scientific Python objects to JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable."
    )


@dataclass
class VariableMetadata:
    """Optional semantic metadata for one independent or dependent variable.

    None of these fields is required. Structural interpretation remains
    possible from ``BlockMetadata.ivars`` and ``BlockMetadata.dvars`` alone.
    """

    dtype: str | None = None
    unit: str | None = None
    role: str | None = None
    description: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> "VariableMetadata":
        values = dict(values or {})
        known = {
            name: values.pop(name, None)
            for name in ("dtype", "unit", "role", "description")
        }
        return cls(**known, extra=values)

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.extra)
        for name in ("dtype", "unit", "role", "description"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out


@dataclass
class BlockMetadata:
    """Structural and optional semantic metadata for one HDF data block."""

    hdf_key: str
    ivars: list[str]
    sweep_ivars: list[str]
    local_ivars: list[str]
    dvars: list[str]
    variables: dict[str, VariableMetadata] = field(default_factory=dict)
    required: bool = True
    default_load: bool = True
    description: str | None = None
    schema_version: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate the structural invariants used by the storage format."""
        if not self.hdf_key:
            raise ValueError("Block hdf_key must not be empty.")

        for field_name in ("ivars", "sweep_ivars", "local_ivars", "dvars"):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(
                    f"{self.hdf_key!r}: duplicate names in {field_name}: "
                    f"{values!r}"
                )

        expected_ivars = self.sweep_ivars + self.local_ivars
        if self.ivars != expected_ivars:
            raise ValueError(
                f"{self.hdf_key!r}: ivars must equal sweep_ivars + "
                f"local_ivars. Expected {expected_ivars!r}, got "
                f"{self.ivars!r}."
            )

        overlap = set(self.ivars) & set(self.dvars)
        if overlap:
            raise ValueError(
                f"{self.hdf_key!r}: names cannot be both ivars and dvars: "
                f"{sorted(overlap)!r}"
            )

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
        *,
        default_key: str | None = None,
    ) -> "BlockMetadata":
        values = dict(values)
        hdf_key = str(values.pop("hdf_key", default_key or "")).strip("/")
        variables_raw = dict(values.pop("variables", {}) or {})
        variables = {
            str(name): VariableMetadata.from_dict(meta)
            for name, meta in variables_raw.items()
        }

        known_names = {
            "ivars",
            "sweep_ivars",
            "local_ivars",
            "dvars",
            "required",
            "default_load",
            "description",
            "schema_version",
        }
        known = {
            name: values.pop(name)
            for name in list(values)
            if name in known_names
        }

        block = cls(
            hdf_key=hdf_key,
            ivars=list(known.pop("ivars", [])),
            sweep_ivars=list(known.pop("sweep_ivars", [])),
            local_ivars=list(known.pop("local_ivars", [])),
            dvars=list(known.pop("dvars", [])),
            variables=variables,
            required=bool(known.pop("required", True)),
            default_load=bool(known.pop("default_load", True)),
            description=known.pop("description", None),
            schema_version=int(known.pop("schema_version", 1)),
            extra=values,
        )
        block.validate()
        return block

    def __getitem__(self, key: str) -> Any:
        """Provide convenient read-only dictionary-style access."""
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        out = dict(self.extra)
        out.update(
            {
                "schema_version": self.schema_version,
                "hdf_key": self.hdf_key,
                "ivars": list(self.ivars),
                "sweep_ivars": list(self.sweep_ivars),
                "local_ivars": list(self.local_ivars),
                "dvars": list(self.dvars),
                "required": self.required,
                "default_load": self.default_load,
            }
        )
        if self.description is not None:
            out["description"] = self.description
        if self.variables:
            out["variables"] = {
                name: metadata.to_dict()
                for name, metadata in self.variables.items()
            }
        return out


@dataclass
class SweepMetadata:
    """Global metadata for one merged nested-sweep measurement file."""

    measurement_name: str
    nested_sweep_levels: list[str]
    blocks: dict[str, BlockMetadata] = field(default_factory=dict)
    created_at: str | None = None
    merged_at: str | None = None
    resume_enabled: bool | None = None
    schema_version: int = 2
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.nested_sweep_levels) != len(
            set(self.nested_sweep_levels)
        ):
            raise ValueError(
                "nested_sweep_levels contains duplicate names."
            )

        nested = set(self.nested_sweep_levels)
        for key, block in self.blocks.items():
            block.validate()
            if key != block.hdf_key:
                raise ValueError(
                    f"Block dictionary key {key!r} does not match "
                    f"hdf_key {block.hdf_key!r}."
                )
            unknown = set(block.sweep_ivars) - nested
            if unknown:
                raise ValueError(
                    f"{key!r}: sweep_ivars are not present in global "
                    f"nested_sweep_levels: {sorted(unknown)!r}"
                )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SweepMetadata":
        values = dict(values)
        blocks_raw = dict(values.pop("blocks", {}) or {})
        blocks = {
            str(key).strip("/"): BlockMetadata.from_dict(
                block_values,
                default_key=str(key).strip("/"),
            )
            for key, block_values in blocks_raw.items()
        }

        known_names = {
            "measurement_name",
            "nested_sweep_levels",
            "created_at",
            "merged_at",
            "resume_enabled",
            "schema_version",
        }
        known = {
            name: values.pop(name)
            for name in list(values)
            if name in known_names
        }

        metadata = cls(
            measurement_name=str(
                known.pop("measurement_name", "")
            ),
            nested_sweep_levels=list(
                known.pop("nested_sweep_levels", [])
            ),
            blocks=blocks,
            created_at=known.pop("created_at", None),
            merged_at=known.pop("merged_at", None),
            resume_enabled=known.pop("resume_enabled", None),
            schema_version=int(known.pop("schema_version", 2)),
            extra=values,
        )
        metadata.validate()
        return metadata

    def __getitem__(self, key: str) -> Any:
        """Provide convenient read-only dictionary-style access."""
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        out = dict(self.extra)
        out.update(
            {
                "schema_version": self.schema_version,
                "measurement_name": self.measurement_name,
                "nested_sweep_levels": list(
                    self.nested_sweep_levels
                ),
                "blocks": {
                    key: block.to_dict()
                    for key, block in self.blocks.items()
                },
            }
        )
        if self.created_at is not None:
            out["created_at"] = self.created_at
        if self.merged_at is not None:
            out["merged_at"] = self.merged_at
        if self.resume_enabled is not None:
            out["resume_enabled"] = self.resume_enabled
        return out


def dumps_metadata(metadata: SweepMetadata | Mapping[str, Any]) -> str:
    """Serialize metadata with stable formatting."""
    values = (
        metadata.to_dict()
        if isinstance(metadata, SweepMetadata)
        else dict(metadata)
    )
    return json.dumps(
        values,
        default=json_default,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def loads_metadata(text: str) -> SweepMetadata:
    """Deserialize and validate metadata JSON."""
    return SweepMetadata.from_dict(json.loads(text))
