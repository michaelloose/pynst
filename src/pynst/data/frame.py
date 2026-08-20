"""Small DataFrame utilities used by measurement functions."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def _to_scalar(value: Any, *, column: str) -> Any:
    """Normalise common scalar-like values and reject true arrays."""
    if isinstance(value, (list, tuple, np.ndarray)):
        value = np.squeeze(value)

    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(
                f"Value for column {column!r} is not scalar after squeeze: "
                f"shape={value.shape}."
            )
        value = value.item()

    if isinstance(value, np.generic):
        value = value.item()

    return value


def _empty_series_for_value(value: Any) -> pd.Series:
    """Create an empty Series with a suitable initial dtype."""
    # bool must be checked before int because bool is an int subclass.
    if isinstance(value, bool):
        return pd.Series(dtype=bool)
    if isinstance(value, complex):
        return pd.Series(dtype=np.complex128)
    if isinstance(value, float):
        return pd.Series(dtype=float)
    if isinstance(value, int):
        return pd.Series(dtype=np.int64)
    if isinstance(value, str) or value is None:
        return pd.Series(dtype=object)

    # Pandas scalar types such as Timestamp are safely represented as object
    # initially and may be promoted by pandas after assignment.
    if pd.api.types.is_scalar(value):
        return pd.Series(dtype=object)

    raise TypeError(
        f"Unsupported scalar value type: {type(value).__name__}."
    )


def insert_row_into_df(
    df: pd.DataFrame,
    row_index: Any,
    data_dict: Mapping[str, Any],
) -> pd.DataFrame:
    """Insert one dictionary of scalar values at ``row_index``.

    The function preserves the historical in-place behaviour and also returns
    ``df`` for convenient chaining.
    """
    sanitised: dict[str, Any] = {}

    for column, raw_value in data_dict.items():
        value = _to_scalar(raw_value, column=str(column))
        sanitised[column] = value

        if column not in df.columns:
            df[column] = _empty_series_for_value(value)

    df.loc[row_index, sanitised.keys()] = list(sanitised.values())
    return df
