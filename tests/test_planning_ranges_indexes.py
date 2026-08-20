"""Focused tests for range and MultiIndex planning helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pynst.planning import (
    combine_multiindexes,
    create_multiindex,
    create_range,
    define_area,
    define_range,
)


def test_define_range_resolves_center_span_and_singletons_consistently() -> None:
    definition = define_range(center=5, span=10, num=3)

    assert definition == {
        "center": 5.0,
        "span": 10.0,
        "min": 0.0,
        "max": 10.0,
        "step": 5.0,
        "num": 3,
    }

    singleton = define_range(min=6.8, max=7.125, num=1)
    assert singleton == {
        "center": 6.8,
        "span": 0.0,
        "min": 6.8,
        "max": 6.8,
        "step": 0.0,
        "num": 1,
    }
    np.testing.assert_array_equal(create_range(min=6.8, max=7.125, num=1), [6.8])


def test_step_is_adjusted_to_reach_authoritative_bounds() -> None:
    with pytest.warns(UserWarning, match="Adjusting 'step'"):
        definition = define_range(min=0, max=10, step=3)
    assert definition["num"] == 4
    assert definition["step"] == pytest.approx(10 / 3, abs=1e-8)

    with pytest.warns(UserWarning):
        values = create_range(min=0, max=10, step=3)
    np.testing.assert_allclose(values, [0, 10 / 3, 20 / 3, 10], atol=1e-8)


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"min": 0, "max": 1, "num": 2, "step": 0.5}, "exactly one"),
        ({"center": 0, "span": -1, "num": 2}, "non-negative"),
        ({"min": 0, "max": 1, "num": 2.5}, "integer"),
        ({"min": 0, "max": np.inf, "num": 2}, "finite"),
        ({"min": 0, "max": 0, "num": 2}, "zero-span"),
    ],
)
def test_define_range_rejects_ambiguous_or_degenerate_inputs(
    arguments: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        define_range(**arguments)


def test_define_area_splits_general_values_and_applies_axis_overrides() -> None:
    area = define_area(center=0j, center_x=2, span=(2, 4), num=(3, 5))

    assert area["center_x"] == 2
    assert area["min_x"] == 1
    assert area["max_x"] == 3
    assert area["num_x"] == 3
    assert area["center_y"] == 0
    assert area["min_y"] == -2
    assert area["max_y"] == 2
    assert area["num_y"] == 5


def test_define_area_rejects_duplicate_axis_aliases() -> None:
    with pytest.raises(ValueError, match="Conflicting aliases"):
        define_area(
            center_x=0,
            center_re=1,
            span=2,
            num=3,
        )


def test_create_multiindex_builds_product_and_smart_rounds_numeric_axes() -> None:
    index = create_multiindex(
        [[0.0, 0.30000000000000004], ["cold", "hot"]],
        names=["bias", "state"],
    )

    assert index.names == ["bias", "state"]
    assert index.tolist() == [
        (0.0, "cold"),
        (0.0, "hot"),
        (0.3, "cold"),
        (0.3, "hot"),
    ]


def test_combine_existing_preserves_correlated_tuples_but_levels_expand() -> None:
    correlated = pd.MultiIndex.from_tuples(
        [("a", 1), ("b", 2)], names=["letter", "number"]
    )
    temperature = pd.MultiIndex.from_tuples(
        [(20,), (30,)], names=["temperature"]
    )

    existing = combine_multiindexes(correlated, temperature, mode="existing")
    expanded = combine_multiindexes(correlated, temperature, mode="levels")

    assert existing.tolist() == [
        ("a", 1, 20),
        ("a", 1, 30),
        ("b", 2, 20),
        ("b", 2, 30),
    ]
    assert len(expanded) == 8
    assert ("a", 2, 20) in expanded
    assert expanded.is_unique


def test_level_combination_ignores_unused_metadata_unless_requested() -> None:
    filtered = pd.MultiIndex(
        levels=[["a", "b"], [1, 2]],
        codes=[[0], [0]],
        names=["letter", "number"],
    )

    observed = combine_multiindexes(filtered, mode="levels")
    declared = combine_multiindexes(
        filtered, mode="levels", include_unused_levels=True
    )

    assert observed.tolist() == [("a", 1)]
    assert len(declared) == 4

