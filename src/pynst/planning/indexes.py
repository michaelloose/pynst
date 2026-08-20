"""Construction and composition helpers for sweep ``MultiIndex`` objects."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import chain, product
from typing import Any, Literal

import numpy as np
import pandas as pd


def _round_numeric_values(
    values: np.ndarray,
    *,
    max_error: float,
) -> np.ndarray:
    """Round as far as possible without merging values or exceeding error."""
    if values.size == 0 or values.dtype.kind not in "iufc":
        return values.copy()
    if not np.isfinite(values).all():
        return values.copy()
    original = values.copy()
    unique_count = len(pd.unique(original.ravel()))
    scale = float(np.max(np.abs(original)))
    tolerance = max_error * scale
    best = original
    # Floating-point inputs carry at most roughly 15 useful decimal places.
    for decimals in range(0, 16):
        if np.iscomplexobj(original):
            candidate = np.round(original.real, decimals) + 1j * np.round(
                original.imag, decimals
            )
        else:
            candidate = np.round(original, decimals)
        if len(pd.unique(candidate.ravel())) != unique_count:
            continue
        if float(np.max(np.abs(original - candidate))) > tolerance:
            continue
        best = candidate
        break
    return best


def _axis_values(
    values: Iterable[Any],
    *,
    round_values: bool,
    max_error: float,
) -> list[Any]:
    if isinstance(values, (str, bytes)):
        raise TypeError("Each MultiIndex axis must be an iterable of values.")
    try:
        array = np.asarray(list(values))
    except TypeError as exc:
        raise TypeError("Each MultiIndex axis must be iterable.") from exc
    if array.ndim != 1:
        raise ValueError("Each MultiIndex axis must be one-dimensional.")
    if round_values:
        array = _round_numeric_values(array, max_error=max_error)
    return array.tolist()


def _validate_names(names: Sequence[Any], *, expected: int) -> list[Any]:
    result = list(names)
    if len(result) != expected:
        raise ValueError("The number of arrays must match the number of names.")
    non_null = [name for name in result if name is not None]
    if len(set(non_null)) != len(non_null):
        raise ValueError("MultiIndex level names must be unique when provided.")
    return result


def create_multiindex(
    arrays: Sequence[Iterable[Any]],
    names: Sequence[Any],
    *,
    round_values: bool = True,
    max_error: float = 1e-4,
) -> pd.MultiIndex:
    """Create a Cartesian-product ``MultiIndex`` from independent axes.

    Numeric axes are smart-rounded by default.  Rounding is accepted only if
    it neither merges distinct values nor exceeds ``max_error`` relative to the
    largest value.  Disable it for exact or non-numeric sweep contracts with
    ``round_values=False``.
    """
    axes = list(arrays)
    resolved_names = _validate_names(names, expected=len(axes))
    if not axes:
        raise ValueError("At least one array is required.")
    if not np.isfinite(max_error) or max_error < 0:
        raise ValueError("'max_error' must be finite and non-negative.")
    resolved_axes = [
        _axis_values(axis, round_values=round_values, max_error=float(max_error))
        for axis in axes
    ]
    return pd.MultiIndex.from_product(resolved_axes, names=resolved_names)


def _observed_unique(index: pd.MultiIndex, level: int) -> list[Any]:
    # pd.unique preserves first-observed order, which is normally the intended
    # sweep order and may differ from sorted MultiIndex level metadata.
    return pd.unique(index.get_level_values(level)).tolist()


def combine_multiindexes(
    *indexes: pd.MultiIndex,
    mode: Literal["existing", "levels"] = "existing",
    include_unused_levels: bool = False,
) -> pd.MultiIndex:
    """Cross-combine one or more ``MultiIndex`` objects.

    Args:
        indexes: Indexes to combine, in output level order.
        mode: ``"existing"`` preserves the tuples already present within each
            input index.  ``"levels"`` breaks those correlations and expands
            the Cartesian product of every level value.
        include_unused_levels: In ``"levels"`` mode, use the complete pandas
            level metadata instead of only values observed in rows.  The
            default avoids silently reintroducing values removed by filtering.

    Returns:
        A new index.  Existing row order and first-observed level order are
        retained.
    """
    if not indexes:
        raise ValueError("At least one MultiIndex must be provided.")
    if mode not in {"existing", "levels"}:
        raise ValueError("'mode' must be either 'existing' or 'levels'.")
    if not isinstance(include_unused_levels, bool):
        raise TypeError("'include_unused_levels' must be a boolean.")
    for position, index in enumerate(indexes):
        if not isinstance(index, pd.MultiIndex):
            raise TypeError(f"indexes[{position}] must be a pandas MultiIndex.")

    names = list(chain.from_iterable(index.names for index in indexes))
    non_null_names = [name for name in names if name is not None]
    if len(set(non_null_names)) != len(non_null_names):
        raise ValueError("Combined MultiIndex level names must be unique.")

    if mode == "existing":
        tuple_axes = [index.tolist() for index in indexes]
        combined = [
            tuple(chain.from_iterable(grouped_tuples))
            for grouped_tuples in product(*tuple_axes)
        ]
        return pd.MultiIndex.from_tuples(combined, names=names)

    levels: list[list[Any]] = []
    for index in indexes:
        for level in range(index.nlevels):
            if include_unused_levels:
                levels.append(index.levels[level].tolist())
            else:
                levels.append(_observed_unique(index, level))
    return pd.MultiIndex.from_product(levels, names=names)


__all__ = ["combine_multiindexes", "create_multiindex"]
