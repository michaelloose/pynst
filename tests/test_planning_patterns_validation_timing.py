"""Focused tests for patterns, grid validation, and timing estimates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from pynst.planning import (
    PointPattern,
    calc_max_precision,
    estimate_remaining_time,
    estimate_sweep_time,
    future_time,
    get_next_weekday,
    inspect_multiindex,
    nudge_points_apart,
    round_complex_smart,
    round_smart,
    validate_grid,
)


def test_circular_pattern_retains_established_ninety_five_point_grid() -> None:
    pattern = PointPattern()
    generated = pattern.create_circ(0, 0.7, 0.14)

    assert len(generated) == 95
    assert len(pattern) == 95
    assert generated[0] == 0j
    assert np.max(np.abs(generated)) == pytest.approx(0.7)
    assert pattern.get_mi("ma", names=("magnitude", "angle")).names == [
        "magnitude",
        "angle",
    ]


def test_truncated_rectangular_pattern_and_wrapped_sector_cut() -> None:
    rectangle = PointPattern()
    generated = rectangle.create_rect(
        center=0,
        span=2,
        num=3,
        staggered="truncate",
        stagger_even=True,
    )
    assert len(generated) == 7
    assert np.max(np.abs(generated.real)) <= 1

    sector = PointPattern(
        np.exp(1j * np.deg2rad([350.0, 10.0, 180.0]))
    )
    retained = sector.cut_circ(min_ang=300, max_ang=30)
    assert len(retained) == 2
    np.testing.assert_allclose(
        np.sort(np.mod(np.rad2deg(np.angle(retained)), 360)),
        [10, 350],
    )


def test_random_patterns_accept_reproducible_rng_without_shared_state() -> None:
    first = PointPattern()
    second = PointPattern()

    points_a = first.create_rand(5, rng=123, center=0, span=2)
    points_b = second.create_rand(5, rng=123, center=0, span=2)

    np.testing.assert_array_equal(points_a, points_b)
    assert first.points is not second.points


def test_smart_rounding_preserves_shape_uniqueness_and_relative_error() -> None:
    values = np.asarray([[0.30000000000000004, 0.6000000000000001]])
    rounded = round_smart(values)

    assert rounded.shape == values.shape
    np.testing.assert_array_equal(rounded, [[0.3, 0.6]])
    assert len(np.unique(rounded)) == len(np.unique(values))
    assert calc_max_precision(values) >= 1

    complex_values = np.asarray([0.10000000000001 + 0.20000000000001j])
    np.testing.assert_array_equal(
        round_complex_smart(complex_values),
        [0.1 + 0.2j],
    )


def test_nudge_points_apart_is_deterministic_and_preserves_centroid() -> None:
    points = np.asarray([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])

    adjusted = nudge_points_apart(points, 0.5)
    repeated = nudge_points_apart(points, 0.5)

    np.testing.assert_array_equal(adjusted, repeated)
    np.testing.assert_allclose(adjusted.mean(axis=0), points.mean(axis=0))
    pairwise = [
        np.linalg.norm(adjusted[left] - adjusted[right])
        for left in range(len(adjusted))
        for right in range(left + 1, len(adjusted))
    ]
    assert min(pairwise) >= 0.5 - 1e-12


def test_scaled_rectangle_handles_nonzero_centers() -> None:
    pattern = PointPattern()
    points = pattern.create_rect(
        center=(2, -1),
        span=(4, 2),
        num=(5, 5),
        scale_to_circle=True,
    )
    normalized_radius = np.sqrt(
        ((points.real - 2) / 2) ** 2 + ((points.imag + 1) / 1) ** 2
    )
    assert np.max(normalized_radius) <= 1 + 1e-12


def test_grid_report_identifies_missing_combination_and_duplicates() -> None:
    grid = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 2), ("b", 1), ("a", 1)],
        names=["letter", "number"],
    )

    report = validate_grid(grid)

    assert not report.is_regular
    assert report.shape == (2, 2)
    assert report.expected_count == 4
    assert report.duplicate_count == 1
    assert report.missing_count == 1
    assert report.missing_combinations == (("b", 2),)
    assert "missing=1" in report.summary()


def test_expected_levels_reveal_an_entirely_absent_axis_value() -> None:
    grid = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 2)], names=["letter", "number"]
    )

    report = validate_grid(
        grid,
        expected_levels={"letter": ["a", "b"], "number": [1, 2]},
    )

    assert report.missing_count == 2
    assert report.missing_combinations == (("b", 1), ("b", 2))


def test_legacy_inspection_uses_dataframe_columns() -> None:
    columns = pd.MultiIndex.from_product(
        [["a", "b"], [1, 2]], names=["letter", "number"]
    )
    frame = pd.DataFrame([[1, 2, 3, 4]], columns=columns)

    regular, values = inspect_multiindex(frame)

    assert regular
    assert values == {"letter": ["a", "b"], "number": [1, 2]}


def test_running_sweep_eta_uses_observed_average_after_warmup() -> None:
    start = datetime(2026, 1, 5, 12, tzinfo=timezone.utc)
    now = start + timedelta(minutes=8)

    estimate = estimate_sweep_time(
        10,
        completed_points=4,
        start_time=start,
        now=now,
    )

    assert estimate.seconds_per_point == 120
    assert estimate.total_duration == timedelta(minutes=20)
    assert estimate.remaining_duration == timedelta(minutes=12)
    assert estimate.estimated_finish == start + timedelta(minutes=20)
    assert estimate.ready


def test_eta_is_withheld_during_first_points_unless_rate_is_given() -> None:
    assert (
        estimate_remaining_time(
            10,
            2,
            elapsed=timedelta(minutes=4),
        )
        is None
    )

    estimate = estimate_sweep_time(
        10,
        30,
        start_time=datetime(2026, 1, 5, 12, tzinfo=timezone.utc),
        now=datetime(2026, 1, 5, 11, tzinfo=timezone.utc),
    )
    assert estimate.total_duration == timedelta(minutes=5)
    assert estimate.estimated_finish == datetime(
        2026, 1, 5, 12, 5, tzinfo=timezone.utc
    )


def test_timing_helpers_preserve_timezones_and_reject_mixed_awareness() -> None:
    aware = datetime(2026, 1, 5, 10, tzinfo=timezone.utc)
    assert future_time(90, aware) == aware + timedelta(seconds=90)

    with pytest.raises(ValueError, match="must not be mixed"):
        estimate_sweep_time(
            10,
            1,
            start_time=aware,
            now=datetime(2026, 1, 5, 10),
        )

    same_day = get_next_weekday(
        0,
        hour=11,
        from_time=aware,
        include_today=True,
    )
    assert same_day == datetime(2026, 1, 5, 11, tzinfo=timezone.utc)
