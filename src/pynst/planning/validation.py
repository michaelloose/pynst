"""Validation helpers for Cartesian sweep grids."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Literal

import numpy as np
import pandas as pd


_NA_KEY = object()


def _value_key(value: Any) -> Any:
    """Return a stable, hashable comparison key, including for missing data."""
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return _NA_KEY
    except (TypeError, ValueError):
        pass
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"Grid values must be hashable; got {value!r}.") from exc
    return value


def _tuple_key(values: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(_value_key(value) for value in values)


def _unique(values: Iterable[Any]) -> tuple[Any, ...]:
    result: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        key = _value_key(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return tuple(result)


def _index_rows(index: pd.Index) -> tuple[list[tuple[Any, ...]], tuple[Any, ...]]:
    if isinstance(index, pd.MultiIndex):
        return [tuple(row) for row in index.tolist()], tuple(index.names)
    return [(value,) for value in index.tolist()], (index.name,)


def _coerce_grid(
    grid: Any,
    *,
    axis: Literal["index", "columns"],
    names: Sequence[Any] | None,
) -> tuple[list[tuple[Any, ...]], tuple[Any, ...]]:
    if axis not in {"index", "columns"}:
        raise ValueError("'axis' must be either 'index' or 'columns'.")
    if isinstance(grid, pd.Series):
        if axis == "columns":
            raise ValueError("A pandas Series has no columns axis.")
        index = grid.index
        rows, inferred_names = _index_rows(index)
    elif isinstance(grid, pd.DataFrame):
        index = grid.index if axis == "index" else grid.columns
        rows, inferred_names = _index_rows(index)
    elif isinstance(grid, pd.Index):
        rows, inferred_names = _index_rows(grid)
    else:
        if isinstance(grid, (str, bytes)):
            raise TypeError("'grid' must be an index or iterable of points.")
        try:
            raw_rows = list(grid)
        except TypeError as exc:
            raise TypeError("'grid' must be an index or iterable of points.") from exc
        if not raw_rows:
            if names is None:
                raise ValueError("'names' is required for an empty iterable grid.")
            inferred_names = tuple(names)
            rows = []
        else:
            first = raw_rows[0]
            if isinstance(first, tuple):
                rows = [tuple(row) for row in raw_rows]
            elif isinstance(first, (list, np.ndarray)):
                rows = [tuple(row) for row in raw_rows]
            else:
                rows = [(row,) for row in raw_rows]
            widths = {len(row) for row in rows}
            if len(widths) != 1:
                raise ValueError("Every grid point must have the same dimension.")
            width = widths.pop()
            inferred_names = tuple([None] * width)

    resolved_names = tuple(names) if names is not None else inferred_names
    width = len(resolved_names)
    if width == 0:
        raise ValueError("A grid must have at least one dimension.")
    if any(len(row) != width for row in rows):
        raise ValueError("The number of names must match each grid point.")
    non_null_names = [name for name in resolved_names if name is not None]
    if len(set(non_null_names)) != len(non_null_names):
        raise ValueError("Grid level names must be unique when provided.")
    return rows, resolved_names


def _resolve_expected_levels(
    expected_levels: Mapping[Any, Iterable[Any]] | Sequence[Iterable[Any]] | None,
    *,
    names: tuple[Any, ...],
    observed: tuple[tuple[Any, ...], ...],
) -> tuple[tuple[Any, ...], ...]:
    if expected_levels is None:
        return observed
    if isinstance(expected_levels, Mapping):
        if any(name is None for name in names):
            raise ValueError(
                "Named 'expected_levels' require names for every grid dimension."
            )
        missing = [name for name in names if name not in expected_levels]
        extra = [name for name in expected_levels if name not in names]
        if missing or extra:
            raise ValueError(
                "'expected_levels' keys must exactly match grid names; "
                f"missing={missing!r}, extra={extra!r}."
            )
        raw_levels = [expected_levels[name] for name in names]
    else:
        raw_levels = list(expected_levels)
        if len(raw_levels) != len(names):
            raise ValueError(
                "The number of expected levels must match the grid dimension."
            )
    result: list[tuple[Any, ...]] = []
    for position, level in enumerate(raw_levels):
        if isinstance(level, (str, bytes)):
            raise TypeError(
                f"expected_levels[{position}] must be an iterable of values."
            )
        try:
            values = _unique(level)
        except TypeError as exc:
            raise TypeError(
                f"expected_levels[{position}] must be iterable."
            ) from exc
        result.append(values)
    return tuple(result)


@dataclass(frozen=True)
class GridValidationReport:
    """Compact result of validating a Cartesian sweep grid."""

    level_names: tuple[Any, ...]
    observed_level_values: tuple[tuple[Any, ...], ...]
    expected_level_values: tuple[tuple[Any, ...], ...]
    observed_count: int
    unique_count: int
    expected_count: int
    duplicate_count: int
    missing_count: int
    unexpected_count: int
    missing_combinations: tuple[tuple[Any, ...], ...]
    unexpected_combinations: tuple[tuple[Any, ...], ...]
    missing_combinations_truncated: bool = False
    unexpected_combinations_truncated: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        """Expected Cartesian shape, one length per level."""
        return tuple(len(values) for values in self.expected_level_values)

    @property
    def is_complete(self) -> bool:
        """Whether every expected combination is present at least once."""
        return self.expected_count > 0 and self.missing_count == 0

    @property
    def is_regular(self) -> bool:
        """Whether the grid is a duplicate-free complete Cartesian product."""
        return (
            self.is_complete
            and self.duplicate_count == 0
            and self.unexpected_count == 0
        )

    @property
    def unique_values(self) -> dict[Any, list[Any]]:
        """Observed values keyed by level name, for legacy inspection code."""
        return {
            name: list(values)
            for name, values in zip(
                self.level_names, self.observed_level_values, strict=True
            )
        }

    def summary(self) -> str:
        """Return a one-line human-readable validation summary."""
        state = "regular" if self.is_regular else "irregular"
        return (
            f"{state} grid: observed={self.observed_count}, "
            f"expected={self.expected_count}, missing={self.missing_count}, "
            f"duplicates={self.duplicate_count}, "
            f"unexpected={self.unexpected_count}"
        )


def validate_grid(
    grid: Any,
    *,
    axis: Literal["index", "columns"] = "index",
    names: Sequence[Any] | None = None,
    expected_levels: Mapping[Any, Iterable[Any]]
    | Sequence[Iterable[Any]]
    | None = None,
    max_reported: int | None = 100,
) -> GridValidationReport:
    """Inspect whether *grid* is a complete Cartesian product.

    ``grid`` may be an Index, MultiIndex, DataFrame/Series, or an iterable of
    scalar/tuple points.  For pandas objects, ``axis`` selects the inspected
    axis.  By default, expected values are inferred from observed values.  Pass
    ``expected_levels`` to also detect an entirely absent level value.

    ``max_reported`` bounds the stored examples without compromising the exact
    missing/unexpected counts.  Set it to ``None`` to retain every combination.
    """
    if max_reported is not None:
        if (
            isinstance(max_reported, (bool, np.bool_))
            or not isinstance(max_reported, (int, np.integer))
            or int(max_reported) < 0
        ):
            raise ValueError("'max_reported' must be a non-negative integer or None.")
        max_reported = int(max_reported)

    rows, level_names = _coerce_grid(grid, axis=axis, names=names)
    dimension = len(level_names)
    observed_values = tuple(
        _unique(row[level] for row in rows) for level in range(dimension)
    )
    expected_values = _resolve_expected_levels(
        expected_levels,
        names=level_names,
        observed=observed_values,
    )

    row_keys = [_tuple_key(row) for row in rows]
    unique_key_set = set(row_keys)
    duplicate_count = len(row_keys) - len(unique_key_set)
    expected_axis_keys = [
        {_value_key(value) for value in values} for values in expected_values
    ]
    valid_observed_keys = {
        key
        for key in unique_key_set
        if all(
            key[level] in expected_axis_keys[level]
            for level in range(dimension)
        )
    }
    unexpected_rows: list[tuple[Any, ...]] = []
    unexpected_seen: set[tuple[Any, ...]] = set()
    for row, key in zip(rows, row_keys, strict=True):
        if key in valid_observed_keys or key in unexpected_seen:
            continue
        unexpected_seen.add(key)
        if max_reported is None or len(unexpected_rows) < max_reported:
            unexpected_rows.append(row)

    expected_count = math.prod(len(values) for values in expected_values)
    missing_count = expected_count - len(valid_observed_keys)
    missing_rows: list[tuple[Any, ...]] = []
    if missing_count:
        for combination in product(*expected_values):
            if _tuple_key(tuple(combination)) in valid_observed_keys:
                continue
            if max_reported is None or len(missing_rows) < max_reported:
                missing_rows.append(tuple(combination))
            else:
                break

    return GridValidationReport(
        level_names=level_names,
        observed_level_values=observed_values,
        expected_level_values=expected_values,
        observed_count=len(rows),
        unique_count=len(unique_key_set),
        expected_count=expected_count,
        duplicate_count=duplicate_count,
        missing_count=missing_count,
        unexpected_count=len(unique_key_set) - len(valid_observed_keys),
        missing_combinations=tuple(missing_rows),
        unexpected_combinations=tuple(unexpected_rows),
        missing_combinations_truncated=missing_count > len(missing_rows),
        unexpected_combinations_truncated=(
            len(unique_key_set) - len(valid_observed_keys) > len(unexpected_rows)
        ),
    )


def is_regular_grid(grid: Any, **kwargs: Any) -> bool:
    """Return only the regularity flag from :func:`validate_grid`."""
    return validate_grid(grid, **kwargs).is_regular


def find_missing_combinations(
    grid: Any,
    **kwargs: Any,
) -> tuple[tuple[Any, ...], ...]:
    """Return missing Cartesian combinations.

    The function defaults to reporting every missing combination.  Supply
    ``max_reported`` explicitly when inspecting a potentially huge grid.
    """
    kwargs.setdefault("max_reported", None)
    return validate_grid(grid, **kwargs).missing_combinations


def inspect_multiindex(
    index: pd.MultiIndex | pd.DataFrame,
    *,
    axis: Literal["index", "columns"] = "columns",
    verbose: bool = False,
) -> tuple[bool, dict[Any, list[Any]]]:
    """Legacy-compatible MultiIndex inspection helper.

    DataFrames default to their columns, matching the notebook helper that
    originally provided this function.  New code should prefer
    :func:`validate_grid` and its structured report.
    """
    report = validate_grid(index, axis=axis)
    if verbose:
        print(report.summary())
        for name, values in report.unique_values.items():
            print(f"Level {name!r}: {values}")
    return report.is_regular, report.unique_values


# Descriptive alias for code that historically referred to a regular matrix.
is_regular_matrix = is_regular_grid
missing_grid_combinations = find_missing_combinations


__all__ = [
    "GridValidationReport",
    "find_missing_combinations",
    "inspect_multiindex",
    "is_regular_grid",
    "is_regular_matrix",
    "missing_grid_combinations",
    "validate_grid",
]
