"""Small Matplotlib helpers for pandas objects with MultiIndex columns."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


_VISUALIZATION_INSTALL_HINT = (
    "Install PyNST with its visualization extra: "
    "pip install 'pynst[visualization]'"
)


def _format_value(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{value:.4g}"
    return str(value)


def _label_positions(
    names: Sequence[Any], label_levels: Sequence[str | int] | str | int | None
) -> list[int]:
    if label_levels is None:
        return list(range(len(names)))
    if isinstance(label_levels, (str, int)):
        label_levels = [label_levels]
    positions = []
    for level in label_levels:
        if isinstance(level, int):
            position = level if level >= 0 else len(names) + level
            if position < 0 or position >= len(names):
                raise ValueError(f"label level position {level} is out of range")
        else:
            matches = [position for position, name in enumerate(names) if name == level]
            if not matches:
                available = list(names)
                raise ValueError(
                    f"Unknown label level {level!r}; "
                    f"available names are {available!r}"
                )
            if len(matches) > 1:
                raise ValueError(f"Label level name {level!r} is ambiguous")
            position = matches[0]
        if position not in positions:
            positions.append(position)
    return positions


def _make_label(
    column: Any,
    names: Sequence[Any],
    label_levels: Sequence[str | int] | str | int | None,
    separator: str,
) -> str:
    values = column if isinstance(column, tuple) else (column,)
    normalised_names = list(names) if names else [None] * len(values)
    if len(normalised_names) < len(values):
        normalised_names.extend([None] * (len(values) - len(normalised_names)))
    positions = _label_positions(normalised_names, label_levels)
    parts = []
    for position in positions:
        value = values[position]
        name = normalised_names[position]
        rendered = _format_value(value)
        parts.append(rendered if name is None else f"{name} = {rendered}")
    return separator.join(parts)


def _column_metadata(frame: pd.DataFrame, column: Any) -> tuple[Any, list[Any]]:
    return column, list(frame.columns.names)


def _series_metadata(series: pd.Series) -> tuple[Any, list[Any]] | None:
    if series.name is None:
        return None
    if isinstance(series.name, tuple):
        return series.name, [None] * len(series.name)
    return series.name, [None]


def _require_aligned_index(left: Any, right: Any) -> None:
    if isinstance(left, (pd.Series, pd.DataFrame)) and isinstance(
        right, (pd.Series, pd.DataFrame)
    ):
        if not left.index.equals(right.index):
            raise ValueError("pandas x and y objects must share the same index")


def _curves_for_y(y: Any) -> list[tuple[Any, Any, tuple[Any, list[Any]] | None]]:
    if isinstance(y, pd.DataFrame):
        return [
            (y.index, y[column], _column_metadata(y, column))
            for column in y.columns
        ]
    if isinstance(y, pd.Series):
        return [(y.index, y, _series_metadata(y))]
    return [(None, y, None)]


def _curves_for_xy(
    x: Any, y: Any
) -> list[tuple[Any, Any, tuple[Any, list[Any]] | None]]:
    _require_aligned_index(x, y)

    if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        if x.shape[1] == 1 and y.shape[1] == 1:
            y_column = y.columns[0]
            return [
                (
                    x.iloc[:, 0],
                    y.iloc[:, 0],
                    _column_metadata(y, y_column),
                )
            ]
        if x.shape[1] == 1 and y.shape[1] > 1:
            shared_x = x.iloc[:, 0]
            return [
                (shared_x, y[column], _column_metadata(y, column))
                for column in y.columns
            ]
        if y.shape[1] == 1 and x.shape[1] > 1:
            shared_y = y.iloc[:, 0]
            return [
                (x[column], shared_y, _column_metadata(x, column))
                for column in x.columns
            ]
        if not x.columns.equals(y.columns):
            raise ValueError(
                "x and y DataFrames must have matching columns, unless one "
                "of them has exactly one shared column"
            )
        return [
            (x[column], y[column], _column_metadata(y, column))
            for column in y.columns
        ]

    if isinstance(x, pd.DataFrame):
        return [
            (x[column], y, _column_metadata(x, column)) for column in x.columns
        ]
    if isinstance(y, pd.DataFrame):
        return [
            (x, y[column], _column_metadata(y, column)) for column in y.columns
        ]

    metadata = _series_metadata(y) if isinstance(y, pd.Series) else None
    if metadata is None and isinstance(x, pd.Series):
        metadata = _series_metadata(x)
    return [(x, y, metadata)]


def plot_mi(
    *args: Any,
    ax: Any = None,
    label_levels: Sequence[str | int] | str | int | None = None,
    label_separator: str = "; ",
    **kwargs: Any,
) -> list[Any]:
    """Plot pandas data with one labelled line per DataFrame column.

    This is a standalone counterpart of Matplotlib's ``Axes.plot`` and works
    with ordinary Matplotlib axes; it does not register a projection or patch
    Matplotlib classes.

    Supported forms are ``plot_mi(y)`` and ``plot_mi(x, y)``.  A one-column
    DataFrame is treated as a shared x or y vector when paired with a
    multi-column DataFrame.  Otherwise two DataFrames must have matching
    indexes and columns.

    Parameters
    ----------
    ax:
        Existing Matplotlib Axes.  If omitted, a new figure and axes are
        created lazily.
    label_levels:
        MultiIndex level names or positions to include in generated labels.
    label_separator:
        Separator between generated label components.
    **kwargs:
        Forwarded to ``Axes.plot``.  If ``label`` is supplied for multiple
        curves, the generated MultiIndex label is appended to it.

    Returns
    -------
    list
        The created Matplotlib ``Line2D`` objects.
    """

    if len(args) not in (1, 2):
        raise ValueError("plot_mi supports exactly y or x, y positional arguments")

    if ax is None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "plot_mi requires Matplotlib. " + _VISUALIZATION_INSTALL_HINT
            ) from exc
        _, ax = plt.subplots()
    elif not hasattr(ax, "plot"):
        raise TypeError("ax must be a Matplotlib-like Axes with a plot method")

    curves = _curves_for_y(args[0]) if len(args) == 1 else _curves_for_xy(*args)
    explicit_label = kwargs.pop("label", None)
    lines: list[Any] = []
    multiple = len(curves) > 1

    for x, y, metadata in curves:
        line_kwargs = dict(kwargs)
        generated_label = None
        if metadata is not None:
            column, names = metadata
            generated_label = _make_label(
                column, names, label_levels, label_separator
            )

        if explicit_label is not None:
            if multiple and generated_label:
                line_kwargs["label"] = f"{explicit_label} | {generated_label}"
            else:
                line_kwargs["label"] = explicit_label
        elif generated_label:
            line_kwargs["label"] = generated_label

        plotted = ax.plot(y, **line_kwargs) if x is None else ax.plot(
            x, y, **line_kwargs
        )
        lines.extend(plotted)
    return lines


__all__ = ["plot_mi"]
