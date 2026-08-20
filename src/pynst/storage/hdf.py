"""Shared HDF naming and schema helpers for PyNST storage."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..data.model import json_default


_MISSING_INDEX_VALUE = object()


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
    spec: dict[str, Any] = {"dtype": str(dtype)}
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
