"""Reusable complex-plane point-pattern generation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from .indexes import _round_numeric_values
from .ranges import create_range, define_area


def calc_max_precision(values: Any) -> int:
    """Return the maximum count of significant decimal digits in *values*.

    Real and imaginary components are inspected independently.  The helper is
    retained for code that used ``MultiADSweep.simpoints.utilities``; smart
    rounding itself does not rely on decimal strings anymore.
    """
    array = np.asarray(values)
    if array.dtype.kind not in "iufc":
        raise TypeError("'values' must contain numeric data.")
    if array.size == 0:
        return 0

    def component_precision(component: float) -> int:
        if not math.isfinite(component):
            return 0
        text = np.format_float_positional(
            float(component), unique=True, trim="-"
        ).lstrip("-+")
        return sum(character.isdigit() for character in text)

    maximum = 0
    for value in array.ravel():
        maximum = max(maximum, component_precision(float(np.real(value))))
        if np.iscomplexobj(array):
            maximum = max(maximum, component_precision(float(np.imag(value))))
    return maximum


def round_smart(values: Any, max_error: float = 1e-4) -> np.ndarray:
    """Round numeric values without merging distinct values.

    The lowest decimal precision satisfying the relative ``max_error`` bound
    is selected.  Shape is preserved and existing duplicate values are
    tolerated; no *additional* duplicates may be introduced.
    """
    array = np.asarray(values)
    if array.dtype.kind not in "iufc":
        raise TypeError("'values' must contain numeric data.")
    if not math.isfinite(max_error) or max_error < 0:
        raise ValueError("'max_error' must be finite and non-negative.")
    return _round_numeric_values(array, max_error=float(max_error))


def round_complex_smart(values: Any, max_error: float = 1e-4) -> np.ndarray:
    """Complex-compatible legacy name for :func:`round_smart`."""
    return round_smart(np.asarray(values, dtype=complex), max_error=max_error)


def nudge_points_apart(
    points: Any,
    min_distance: float,
    *,
    max_iterations: int = 500,
) -> np.ndarray:
    """Deterministically separate coordinates closer than ``min_distance``.

    ``points`` may be an ``(n, dimensions)`` real array or a one-dimensional
    complex array.  The algorithm iteratively shares each overlap equally
    between both points and preserves the overall centroid.  Identical points
    receive deterministic directions, avoiding the hidden global random state
    used by the historical helper.
    """
    distance = _finite_real(
        min_distance, name="min_distance", minimum=0.0
    )
    iterations = _count(max_iterations, name="max_iterations")
    if iterations < 1:
        raise ValueError("'max_iterations' must be >= 1.")

    source = np.asarray(points)
    complex_input = np.iscomplexobj(source)
    if complex_input:
        if source.ndim != 1:
            raise ValueError("Complex points must form a one-dimensional array.")
        coordinates = np.column_stack((source.real, source.imag)).astype(float)
    else:
        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim != 2:
            raise ValueError("Real coordinates must have shape (n, dimensions).")
    if coordinates.shape[1] < 1:
        raise ValueError("Points must have at least one coordinate dimension.")
    if not np.isfinite(coordinates).all():
        raise ValueError("'points' must contain only finite coordinates.")
    adjusted = coordinates.copy()
    if len(adjusted) < 2 or distance == 0:
        return (
            adjusted[:, 0] + 1j * adjusted[:, 1]
            if complex_input
            else adjusted
        )

    tolerance = max(distance * 1e-12, np.finfo(float).eps)
    dimensions = adjusted.shape[1]
    for _ in range(iterations):
        changed = False
        for left in range(len(adjusted) - 1):
            for right in range(left + 1, len(adjusted)):
                delta = adjusted[right] - adjusted[left]
                current = float(np.linalg.norm(delta))
                if current + tolerance >= distance:
                    continue
                if current <= tolerance:
                    direction = np.zeros(dimensions, dtype=float)
                    if dimensions == 1:
                        direction[0] = 1.0 if (left + right) % 2 else -1.0
                    else:
                        angle = (left * len(adjusted) + right) * (
                            math.pi * (3 - math.sqrt(5))
                        )
                        direction[0] = math.cos(angle)
                        direction[1] = math.sin(angle)
                else:
                    direction = delta / current
                correction = direction * ((distance - current) / 2)
                adjusted[left] -= correction
                adjusted[right] += correction
                changed = True
        if not changed:
            break
    else:
        raise RuntimeError(
            "Could not achieve the requested minimum distance within "
            f"{iterations} iterations."
        )

    if complex_input:
        return adjusted[:, 0] + 1j * adjusted[:, 1]
    return adjusted


def _finite_real(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name!r} must be a real scalar.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name!r} must be finite.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name!r} must be >= {minimum}.")
    return result


def _finite_complex(value: Any, *, name: str) -> complex:
    try:
        result = complex(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name!r} must be a complex scalar.") from exc
    if not (math.isfinite(result.real) and math.isfinite(result.imag)):
        raise ValueError(f"{name!r} must be finite.")
    return result


def _count(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name!r} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name!r} must be non-negative.")
    return result


def _generator(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None or (
        isinstance(rng, Integral) and not isinstance(rng, (bool, np.bool_))
    ):
        return np.random.default_rng(None if rng is None else int(rng))
    raise TypeError("'rng' must be a numpy Generator, integer seed, or None.")


class PointPattern:
    """Mutable collection for constructing complex-plane sampling patterns.

    Generator calls append to :attr:`points`, allowing several regions or
    resolutions to be composed.  Their return value contains only the points
    added by that call; existing code that ignores the return value remains
    compatible with the historical ``MultiADSweep`` implementation.
    """

    def __init__(self, points: Iterable[complex] | None = None) -> None:
        self.points: list[complex] = []
        if points is not None:
            self.points.extend(
                _finite_complex(point, name="point") for point in points
            )

    def __len__(self) -> int:
        return len(self.points)

    def clear(self) -> None:
        """Remove every stored point."""
        self.points.clear()

    def get_complex(
        self,
        round_values: bool = True,
        *,
        max_error: float = 1e-4,
    ) -> np.ndarray:
        """Return the stored points as a complex NumPy array.

        Smart rounding never merges distinct points and bounds the absolute
        rounding error relative to the largest point magnitude.
        """
        values = np.asarray(self.points, dtype=complex)
        if not round_values:
            return values.copy()
        if not math.isfinite(max_error) or max_error < 0:
            raise ValueError("'max_error' must be finite and non-negative.")
        return _round_numeric_values(values, max_error=float(max_error))

    def get_mi(
        self,
        format: str = "ri",
        names: tuple[Any, Any] = ("re", "im"),
        *,
        round_values: bool = True,
        max_error: float = 1e-4,
    ) -> pd.MultiIndex:
        """Return the points as a two-level :class:`pandas.MultiIndex`.

        Supported formats are real/imaginary (``"ri"`` and ``"ir"``), or
        magnitude/angle (``"ma"`` and ``"am"``).  Append ``"_rad"`` to
        either polar format for radians; otherwise angles are in degrees.
        """
        if len(names) != 2:
            raise ValueError("'names' must contain exactly two level names.")
        values = self.get_complex(round_values, max_error=max_error)
        if format == "ri":
            tuples = list(zip(values.real, values.imag))
        elif format == "ir":
            tuples = list(zip(values.imag, values.real))
        elif format in {"ma", "am", "ma_rad", "am_rad"}:
            magnitude = np.abs(values)
            angle = np.angle(values)
            if format in {"ma", "am"}:
                angle = np.rad2deg(angle)
            if format.startswith("ma"):
                tuples = list(zip(magnitude, angle))
            else:
                tuples = list(zip(angle, magnitude))
        else:
            raise ValueError(
                "'format' must be one of 'ri', 'ir', 'ma', 'am', "
                "'ma_rad', or 'am_rad'."
            )
        return pd.MultiIndex.from_tuples(tuples, names=list(names))

    def create_circ(
        self,
        center: complex = 0j,
        radius: float = 1.0,
        step: float | None = None,
        include_center: bool = True,
        fill: bool = True,
    ) -> np.ndarray:
        """Append an approximately equally spaced circular pattern.

        Filled circles use radial spacing ``step`` and an even number of points
        per ring with approximately the same arc spacing.  ``fill=False`` adds
        only the outer boundary.  The historical point counts and ordering are
        retained for ordinary positive radii.
        """
        resolved_center = _finite_complex(center, name="center")
        resolved_radius = _finite_real(radius, name="radius", minimum=0.0)
        if step is None:
            raise ValueError("'step' must be specified.")
        resolved_step = _finite_real(step, name="step", minimum=0.0)
        if resolved_step == 0:
            raise ValueError("'step' must be positive.")
        if not isinstance(include_center, (bool, np.bool_)):
            raise TypeError("'include_center' must be a boolean.")
        if not isinstance(fill, (bool, np.bool_)):
            raise TypeError("'fill' must be a boolean.")

        if resolved_radius == 0:
            generated = (
                np.asarray([resolved_center], dtype=complex)
                if include_center or not fill
                else np.asarray([], dtype=complex)
            )
            self.points.extend(generated.tolist())
            return generated

        radii = create_range(
            min=0,
            max=resolved_radius,
            step=resolved_step,
            endpoint=True,
        )
        radii = radii[~np.isclose(radii, 0.0, rtol=0.0, atol=1e-15)]
        if not fill:
            radii = np.asarray([resolved_radius])

        chunks: list[np.ndarray] = []
        if fill and include_center:
            chunks.append(np.asarray([resolved_center], dtype=complex))
        for ring_radius in radii:
            point_count = int(2 * np.pi * float(ring_radius) / resolved_step)
            point_count = max(2, point_count)
            if point_count % 2:
                point_count += 1
            angles = np.linspace(0.0, 2 * np.pi, point_count, endpoint=False)
            chunks.append(
                resolved_center + float(ring_radius) * np.exp(1j * angles)
            )
        generated = (
            np.concatenate(chunks)
            if chunks
            else np.asarray([], dtype=complex)
        )
        self.points.extend(generated.tolist())
        return generated

    def create_rect(
        self,
        staggered: bool | str = False,
        stagger_even: bool = True,
        scale_to_circle: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Append a rectangular, staggered, or ellipse-scaled grid.

        Range arguments are forwarded to :func:`pynst.planning.define_area`.
        ``staggered`` may be ``"shift"``, ``"truncate"`` or ``"scale"``.
        With ``scale_to_circle=True``, rows are mapped into the ellipse bounded
        by the requested x/y area (a circle when both spans are equal).
        """
        if staggered is True or staggered not in {
            False,
            None,
            "shift",
            "truncate",
            "scale",
        }:
            raise ValueError(
                "'staggered' must be False, 'shift', 'truncate', or 'scale'."
            )
        if not isinstance(stagger_even, (bool, np.bool_)):
            raise TypeError("'stagger_even' must be a boolean.")
        if not isinstance(scale_to_circle, (bool, np.bool_)):
            raise TypeError("'scale_to_circle' must be a boolean.")

        area = define_area(**kwargs)
        num_x, num_y = int(area["num_x"]), int(area["num_y"])
        x_values = np.linspace(area["min_x"], area["max_x"], num_x)
        if scale_to_circle:
            if area["span_y"] <= 0:
                raise ValueError(
                    "'scale_to_circle' requires a positive y-axis span."
                )
            y_values = np.linspace(
                area["min_y"], area["max_y"], num_y + 2
            )[1:-1]
        else:
            y_values = np.linspace(area["min_y"], area["max_y"], num_y)
        x_grid, y_grid = np.meshgrid(x_values, y_values)

        row_start = 0 if stagger_even else 1
        if staggered:
            if num_x < 2:
                raise ValueError("Staggered grids require at least two x points.")
            half_step = (x_values[1] - x_values[0]) / 2
            if staggered == "shift":
                x_grid[row_start::2] += half_step
            elif staggered == "truncate":
                x_grid[row_start::2, :-1] += half_step
                x_grid[row_start::2, -1] = np.nan
            elif staggered == "scale":
                scaled = np.linspace(
                    area["min_x"] + half_step,
                    area["max_x"] - half_step,
                    num_x,
                )
                x_grid[row_start::2] = scaled

        if scale_to_circle:
            center_x = float(area["center_x"])
            center_y = float(area["center_y"])
            radius_y = float(area["span_y"]) / 2
            relative_y = (y_grid[:, 0] - center_y) / radius_y
            row_scale = np.sqrt(np.clip(1.0 - relative_y**2, 0.0, 1.0))
            x_grid = center_x + (x_grid - center_x) * row_scale[:, None]

        valid = ~np.isnan(x_grid)
        generated = x_grid[valid] + 1j * y_grid[valid]
        self.points.extend(generated.tolist())
        return generated

    @staticmethod
    def _random_area(kwargs: Mapping[str, Any]) -> dict[str, float | int]:
        # Random patterns need bounds, not a sampling resolution.  A two-point
        # definition resolves those bounds without conflicting with user input.
        area_args = dict(kwargs)
        for key in tuple(area_args):
            if key == "step" or key.startswith("step_"):
                area_args.pop(key)
            if key == "num" or key.startswith("num_"):
                area_args.pop(key)
        area_args["num"] = 2
        return define_area(**area_args)

    def create_rand(
        self,
        num_points: int = 100,
        *,
        rng: np.random.Generator | int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Append uniformly distributed random points inside an area."""
        count = _count(num_points, name="num_points")
        area = self._random_area(kwargs)
        generator = _generator(rng)
        real = generator.uniform(area["min_x"], area["max_x"], count)
        imag = generator.uniform(area["min_y"], area["max_y"], count)
        generated = real + 1j * imag
        self.points.extend(generated.tolist())
        return generated

    def create_randn(
        self,
        num_points: int = 100,
        *,
        rng: np.random.Generator | int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Append normally distributed points centered on an area.

        One standard deviation is one fifth of the corresponding area span.
        As with any normal distribution, points are not clipped to the bounds.
        """
        count = _count(num_points, name="num_points")
        area = self._random_area(kwargs)
        generator = _generator(rng)
        real = generator.normal(area["center_x"], area["span_x"] / 5, count)
        imag = generator.normal(area["center_y"], area["span_y"] / 5, count)
        generated = real + 1j * imag
        self.points.extend(generated.tolist())
        return generated

    def cut_circ(
        self,
        center: complex = 0j,
        min_r: float = 0.0,
        max_r: float = np.inf,
        min_ang: float = 0.0,
        max_ang: float = 360.0,
        keep_within: bool = True,
    ) -> np.ndarray:
        """Keep or remove points in a radial/angular sector.

        Angles are in degrees.  Wrapped sectors such as 300° through 30° are
        supported, and spans of at least 360° cover the full circle.
        """
        resolved_center = _finite_complex(center, name="center")
        lower_radius = _finite_real(min_r, name="min_r", minimum=0.0)
        if max_r == np.inf:
            upper_radius = float("inf")
        else:
            upper_radius = _finite_real(max_r, name="max_r", minimum=0.0)
        if upper_radius < lower_radius:
            raise ValueError("'max_r' must be greater than or equal to 'min_r'.")
        lower_angle = _finite_real(min_ang, name="min_ang")
        upper_angle = _finite_real(max_ang, name="max_ang")
        if not isinstance(keep_within, (bool, np.bool_)):
            raise TypeError("'keep_within' must be a boolean.")

        values = np.asarray(self.points, dtype=complex)
        relative = values - resolved_center
        radii = np.abs(relative)
        angles = np.mod(np.rad2deg(np.angle(relative)), 360.0)
        if abs(upper_angle - lower_angle) >= 360:
            in_sector = np.ones(len(values), dtype=bool)
        else:
            start = lower_angle % 360
            stop = upper_angle % 360
            if start <= stop:
                in_sector = (angles >= start) & (angles <= stop)
            else:
                in_sector = (angles >= start) | (angles <= stop)
        in_radius = (radii >= lower_radius) & (radii <= upper_radius)
        selected = in_sector & in_radius
        retained = values[selected if keep_within else ~selected]
        self.points = retained.tolist()
        return retained.copy()

    def cut_rect(self, keep_within: bool = True, **kwargs: Any) -> np.ndarray:
        """Keep or remove points inside a rectangular area."""
        if not isinstance(keep_within, (bool, np.bool_)):
            raise TypeError("'keep_within' must be a boolean.")
        area = self._random_area(kwargs)
        values = np.asarray(self.points, dtype=complex)
        in_rectangle = (
            (values.real >= area["min_x"])
            & (values.real <= area["max_x"])
            & (values.imag >= area["min_y"])
            & (values.imag <= area["max_y"])
        )
        retained = values[in_rectangle if keep_within else ~in_rectangle]
        self.points = retained.tolist()
        return retained.copy()

    def plot(
        self,
        smith: bool = False,
        ax: Any | None = None,
        **scatter_kwargs: Any,
    ) -> tuple[Any, Any]:
        """Scatter the pattern on a Matplotlib axis.

        Matplotlib and the optional scikit-rf Smith background are imported
        lazily, so point generation itself has no visualization dependency.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "PointPattern.plot requires matplotlib; install the PyNST "
                "visualization extra."
            ) from exc
        if ax is None:
            figure, ax = plt.subplots()
        else:
            figure = ax.figure
        if smith:
            try:
                import skrf.plotting
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("smith=True requires scikit-rf.") from exc
            skrf.plotting.smith(
                ax=ax,
                smithR=1,
                chart_type="z",
                draw_labels=False,
                border=False,
                ref_imm=1.0,
                draw_vswr=None,
            )
        values = np.asarray(self.points, dtype=complex)
        scatter_kwargs.setdefault("color", "tab:blue")
        ax.scatter(values.real, values.imag, **scatter_kwargs)
        ax.text(
            0.95,
            0.05,
            f"n_points: {len(values)}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )
        ax.set(aspect="equal", xlabel="Real", ylabel="Imaginary")
        return figure, ax


__all__ = [
    "PointPattern",
    "calc_max_precision",
    "nudge_points_apart",
    "round_complex_smart",
    "round_smart",
]
