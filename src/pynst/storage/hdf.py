"""Shared HDF naming and schema helpers for PyNST storage."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..data.model import json_default


_MISSING_INDEX_VALUE = object()
_PANDAS_STRING_DTYPE_NAMES = frozenset(
    {"str", "string", "string[python]", "string[pyarrow]"}
)


def _canonical_hdf_dtype_name(dtype: Any) -> str:
    """Return the stable dtype name used by PyNST's HDF schema.

    Pandas 3 represents ordinary text with ``StringDtype`` by default and
    also reconstructs legacy HDF object strings that way.  PyNST continues
    to store those values as the Pandas-2-compatible NumPy ``object`` dtype.
    """
    if isinstance(dtype, pd.StringDtype):
        return "object"
    name = str(dtype)
    if name.lower() in _PANDAS_STRING_DTYPE_NAMES:
        return "object"
    return name


def _normalise_hdf_string_index(index: pd.Index) -> pd.Index:
    """Return an index whose Pandas string levels use NumPy object dtype."""
    if isinstance(index, pd.MultiIndex):
        levels = list(index.levels)
        changed = False
        for position, level in enumerate(levels):
            if not isinstance(level.dtype, pd.StringDtype):
                continue
            levels[position] = pd.Index(
                level.to_numpy(dtype=object),
                dtype=object,
                name=level.name,
            )
            changed = True
        if changed:
            # Replacing levels, rather than rebuilding from row values,
            # preserves codes, unused levels, ordering and missing entries.
            return index.set_levels(levels, verify_integrity=True)
        return index

    if isinstance(index.dtype, pd.StringDtype):
        return pd.Index(
            index.to_numpy(dtype=object),
            dtype=object,
            name=index.name,
        )
    return index


def _normalise_hdf_string_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an HDF-compatible frame without mutating the input frame."""
    normalised_index = _normalise_hdf_string_index(frame.index)
    string_columns = [
        position
        for position, dtype in enumerate(frame.dtypes)
        if isinstance(dtype, pd.StringDtype)
    ]
    if normalised_index is frame.index and not string_columns:
        return frame

    normalised = frame.copy(deep=False)
    if normalised_index is not frame.index:
        normalised.index = normalised_index
    for position in string_columns:
        # Replacing the complete backing array keeps a shallow copy from
        # mutating a DataFrame returned by the user's measurement callback.
        normalised.isetitem(
            position,
            frame.iloc[:, position].astype(object),
        )
    return normalised


def _normalise_hdf_string_blocks(
    blocks: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalise Pandas string arrays at an HDF storage boundary."""
    return {
        name: (
            _normalise_hdf_string_frame(value)
            if isinstance(value, pd.DataFrame)
            else value
        )
        for name, value in blocks.items()
    }


def _normalise_block_name(name: str) -> str:
    """Validate and normalise a user-defined HDF block name."""
    if not isinstance(name, str):
        raise TypeError("Measurement block names must be strings.")

    key = name.strip("/")
    if not key:
        raise ValueError("Measurement block names must not be empty.")
    if "/" in key:
        raise ValueError(
            f"Measurement block name {name!r} must not contain '/'."
        )
    if key.startswith("__metadata__"):
        raise ValueError(
            f"Measurement block name {name!r} uses the reserved "
            "'__metadata__' prefix."
        )
    return key


def _is_legacy_block_name(name: str) -> bool:
    key = str(name).strip("/")
    return (
        key.startswith("block_")
        and key.removeprefix("block_").isdigit()
    )


def _block_sort_key(name: str) -> tuple[int, int | str]:
    key = str(name).strip("/")
    if _is_legacy_block_name(key):
        return (0, int(key.removeprefix("block_")))
    return (1, key)


def _is_data_block_key(key: str) -> bool:
    """Return whether an HDF key is one top-level user data block.

    Pandas stores categorical table-index metadata below paths such as
    ``/measurement/meta/category/meta``. Those nodes are implementation
    details of the top-level ``/measurement`` block, not additional result
    blocks.
    """
    normalised = str(key).strip("/")
    return (
        bool(normalised)
        and "/" not in normalised
        and not normalised.startswith("__metadata__")
    )


def _canonical_index_value(value: Any) -> Any:
    """Make scalar missing values compare equal across validation batches."""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return _MISSING_INDEX_VALUE
    return value


def _data_block_keys(store: pd.HDFStore) -> list[str]:
    keys = list(store.keys())
    data_keys = [key for key in keys if _is_data_block_key(key)]
    top_level = {key.strip("/") for key in data_keys}
    for key in keys:
        normalised = key.strip("/")
        if not normalised or normalised.startswith("__metadata__"):
            continue
        if "/" not in normalised:
            continue
        parts = normalised.split("/")
        is_pandas_categorical_metadata = (
            len(parts) >= 4
            and parts[0] in top_level
            and parts[1] == "meta"
            and parts[-1] == "meta"
        )
        if not is_pandas_categorical_metadata:
            raise ValueError(
                "HDF block names must not contain nested '/' paths; "
                f"{key!r} is not pandas categorical metadata."
            )
    return data_keys


def _dtype_spec(dtype: Any) -> dict[str, Any]:
    """Return a JSON-stable dtype identity, including category semantics."""
    spec: dict[str, Any] = {"dtype": _canonical_hdf_dtype_name(dtype)}
    if isinstance(dtype, pd.CategoricalDtype):
        spec["categories"] = [
            json.loads(
                json.dumps(
                    value,
                    default=json_default,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            for value in dtype.categories.tolist()
        ]
        spec["ordered"] = bool(dtype.ordered)
    return spec


def _hdf_object_dtype_issue(
    values: Any,
    label: str,
    *,
    allow_empty: bool = False,
) -> str | None:
    """Describe object/category values pandas HDF table cannot serialize."""
    dtype = getattr(values, "dtype", None)
    if isinstance(dtype, pd.CategoricalDtype):
        values = dtype.categories
        dtype = values.dtype
    if not isinstance(dtype, np.dtype) or dtype != np.dtype("object"):
        return None
    inferred = pd.api.types.infer_dtype(values, skipna=True)
    if inferred in {"string", "unicode"} or (
        allow_empty and inferred == "empty"
    ):
        return None
    return f"{label}=object[{inferred}]"
