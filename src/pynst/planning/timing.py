"""Sweep-duration and completion-time estimates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from numbers import Integral, Real
from typing import Any

import numpy as np


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name!r} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name!r} must be >= {minimum}.")
    return result


def _seconds(value: Any, *, name: str) -> float:
    if isinstance(value, timedelta):
        result = value.total_seconds()
    elif isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        result = float(value)
    else:
        raise ValueError(f"{name!r} must be seconds or a timedelta.")
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name!r} must be finite and non-negative.")
    return result


def _aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def _validate_datetime(value: Any, *, name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise TypeError(f"{name!r} must be a datetime or None.")
    return value


def _compatible_datetimes(first: datetime, second: datetime) -> None:
    if _aware(first) != _aware(second):
        raise ValueError(
            "Timezone-naive and timezone-aware datetimes must not be mixed."
        )


def _current_time(reference: datetime | None) -> datetime:
    if reference is not None and _aware(reference):
        return datetime.now(reference.tzinfo)
    return datetime.now()


@dataclass(frozen=True)
class SweepTimeEstimate:
    """Structured estimate for a planned or running sweep."""

    total_points: int
    completed_points: int
    seconds_per_point: float | None
    elapsed: timedelta | None
    total_duration: timedelta | None
    remaining_duration: timedelta | None
    start_time: datetime | None
    reference_time: datetime
    estimated_finish: datetime | None
    minimum_completed: int

    @property
    def remaining_points(self) -> int:
        return self.total_points - self.completed_points

    @property
    def progress(self) -> float:
        """Completed fraction in the closed interval zero to one."""
        if self.total_points == 0:
            return 1.0
        return self.completed_points / self.total_points

    @property
    def ready(self) -> bool:
        """Whether a remaining duration and finish time are available."""
        return self.remaining_duration is not None and self.estimated_finish is not None


def estimate_sweep_duration(
    point_count: int,
    seconds_per_point: float | timedelta,
) -> timedelta:
    """Estimate total duration from a point count and per-point duration."""
    count = _integer(point_count, name="point_count")
    seconds = _seconds(seconds_per_point, name="seconds_per_point")
    return timedelta(seconds=count * seconds)


def estimate_sweep_time(
    total_points: int,
    seconds_per_point: float | timedelta | None = None,
    *,
    completed_points: int = 0,
    elapsed: float | timedelta | None = None,
    start_time: datetime | None = None,
    now: datetime | None = None,
    minimum_completed: int = 3,
) -> SweepTimeEstimate:
    """Estimate total duration, remaining duration, and finish time.

    Provide ``seconds_per_point`` for a plan-time estimate.  For a running
    sweep, omit it and provide either ``elapsed`` or ``start_time``; the mean
    duration is inferred once at least ``minimum_completed`` points exist.
    Before then the timing fields remain ``None`` instead of presenting a
    misleading ETA.

    Datetimes retain their timezone.  Mixing timezone-aware and naive values
    raises :class:`ValueError` rather than producing a silently shifted ETA.
    """
    total = _integer(total_points, name="total_points")
    completed = _integer(completed_points, name="completed_points")
    if completed > total:
        raise ValueError("'completed_points' must not exceed 'total_points'.")
    minimum = _integer(minimum_completed, name="minimum_completed", minimum=1)
    start = _validate_datetime(start_time, name="start_time")
    reference = _validate_datetime(now, name="now")
    if start is not None and reference is not None:
        _compatible_datetimes(start, reference)
    if elapsed is not None and start is not None:
        raise ValueError("Provide either 'elapsed' or 'start_time', not both.")
    if reference is None:
        reference = _current_time(start)

    if elapsed is not None:
        elapsed_seconds = _seconds(elapsed, name="elapsed")
        elapsed_duration = timedelta(seconds=elapsed_seconds)
        resolved_start = reference - elapsed_duration
    elif start is not None:
        _compatible_datetimes(start, reference)
        if start > reference:
            if completed:
                raise ValueError(
                    "'start_time' cannot be in the future after points completed."
                )
            elapsed_duration = timedelta(0)
        else:
            elapsed_duration = reference - start
        elapsed_seconds = elapsed_duration.total_seconds()
        resolved_start = start
    else:
        elapsed_seconds = None
        elapsed_duration = None
        resolved_start = None

    if seconds_per_point is not None:
        average_seconds = _seconds(
            seconds_per_point, name="seconds_per_point"
        )
    elif (
        completed >= minimum
        and completed > 0
        and elapsed_seconds is not None
    ):
        average_seconds = elapsed_seconds / completed
    else:
        average_seconds = None

    if total == 0:
        total_duration = timedelta(0)
        remaining_duration = timedelta(0)
    elif average_seconds is None:
        total_duration = None
        remaining_duration = timedelta(0) if completed == total else None
    else:
        total_duration = timedelta(seconds=total * average_seconds)
        remaining_duration = timedelta(
            seconds=(total - completed) * average_seconds
        )

    if remaining_duration is None:
        finish = None
    elif start is not None and start > reference and completed == 0:
        finish = start + remaining_duration
    else:
        finish = reference + remaining_duration

    return SweepTimeEstimate(
        total_points=total,
        completed_points=completed,
        seconds_per_point=average_seconds,
        elapsed=elapsed_duration,
        total_duration=total_duration,
        remaining_duration=remaining_duration,
        start_time=resolved_start,
        reference_time=reference,
        estimated_finish=finish,
        minimum_completed=minimum,
    )


def estimate_remaining_time(
    total_points: int,
    completed_points: int,
    *,
    seconds_per_point: float | timedelta | None = None,
    elapsed: float | timedelta | None = None,
    start_time: datetime | None = None,
    now: datetime | None = None,
    minimum_completed: int = 3,
) -> timedelta | None:
    """Return the remaining duration, or ``None`` while the ETA is unstable."""
    return estimate_sweep_time(
        total_points,
        seconds_per_point,
        completed_points=completed_points,
        elapsed=elapsed,
        start_time=start_time,
        now=now,
        minimum_completed=minimum_completed,
    ).remaining_duration


def estimate_finish_time(
    total_points: int,
    completed_points: int = 0,
    *,
    seconds_per_point: float | timedelta | None = None,
    elapsed: float | timedelta | None = None,
    start_time: datetime | None = None,
    now: datetime | None = None,
    minimum_completed: int = 3,
) -> datetime | None:
    """Return the estimated finish datetime, or ``None`` if not yet stable."""
    return estimate_sweep_time(
        total_points,
        seconds_per_point,
        completed_points=completed_points,
        elapsed=elapsed,
        start_time=start_time,
        now=now,
        minimum_completed=minimum_completed,
    ).estimated_finish


def future_time(
    seconds: float | timedelta,
    start_time: datetime | None = None,
) -> datetime:
    """Return ``start_time + seconds`` (legacy notebook helper)."""
    start = _validate_datetime(start_time, name="start_time")
    if start is None:
        start = datetime.now()
    return start + timedelta(seconds=_seconds(seconds, name="seconds"))


def get_next_weekday(
    target_weekday: int,
    hour: int = 0,
    minute: int = 0,
    *,
    from_time: datetime | None = None,
    include_today: bool = False,
) -> datetime:
    """Return the next selected weekday while preserving timezone information.

    Weekdays follow :meth:`datetime.weekday` (Monday is zero).  By default,
    selecting the current weekday means the following week, matching the
    historical notebook helper.  Set ``include_today=True`` to allow today when
    the requested wall-clock time has not passed.
    """
    weekday = _integer(target_weekday, name="target_weekday")
    if weekday > 6:
        raise ValueError("'target_weekday' must be between 0 and 6.")
    resolved_hour = _integer(hour, name="hour")
    resolved_minute = _integer(minute, name="minute")
    if resolved_hour > 23:
        raise ValueError("'hour' must be between 0 and 23.")
    if resolved_minute > 59:
        raise ValueError("'minute' must be between 0 and 59.")
    if not isinstance(include_today, (bool, np.bool_)):
        raise TypeError("'include_today' must be a boolean.")
    origin = _validate_datetime(from_time, name="from_time")
    if origin is None:
        origin = datetime.now()

    days_ahead = (weekday - origin.weekday()) % 7
    candidate = (origin + timedelta(days=days_ahead)).replace(
        hour=resolved_hour,
        minute=resolved_minute,
        second=0,
        microsecond=0,
    )
    if days_ahead == 0 and (not include_today or candidate < origin):
        candidate += timedelta(days=7)
    return candidate


__all__ = [
    "SweepTimeEstimate",
    "estimate_finish_time",
    "estimate_remaining_time",
    "estimate_sweep_duration",
    "estimate_sweep_time",
    "future_time",
    "get_next_weekday",
]
