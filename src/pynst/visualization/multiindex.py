"""Selection and interactive exploration of MultiIndex-labelled data.

The selector in this module contains no notebook or plotting dependencies.  It
is also the data-selection engine used by :class:`InteractiveMultiIndexPlotter`,
which keeps the constructor used by MultiADSweep notebooks while avoiding
imports and display calls at module import time.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


FrameKey = Hashable
FrameMapping = Mapping[FrameKey, pd.DataFrame]
FrameInput = pd.DataFrame | FrameMapping

_VISUALIZATION_INSTALL_HINT = (
    "Install PyNST with its visualization extra: "
    "pip install 'pynst[visualization]'"
)


def _axis_name(axis: str | int) -> str:
    if axis in ("columns", 1):
        return "columns"
    if axis in ("index", 0):
        return "index"
    raise ValueError("axis must be 'columns', 'index', 1, or 0")


def _format_value(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{value:.4g}"
    return str(value)


def _normalised_level_names(index: pd.Index) -> list[str]:
    """Return stable, unique string names without modifying *index*."""

    raw_names = list(index.names) if isinstance(index, pd.MultiIndex) else [index.name]
    result: list[str] = []
    used: set[str] = set()
    for position, raw_name in enumerate(raw_names):
        base = f"level_{position}" if raw_name is None else str(raw_name)
        candidate = base
        suffix = position
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        result.append(candidate)
        used.add(candidate)
    return result


def _as_named_multiindex(index: pd.Index) -> pd.MultiIndex:
    names = _normalised_level_names(index)
    if isinstance(index, pd.MultiIndex):
        return index.set_names(names)
    return pd.MultiIndex.from_arrays([index], names=names)


def _replace_axis(
    frame: pd.DataFrame, axis: str, index: pd.Index
) -> pd.DataFrame:
    result = frame.copy(deep=False)
    if axis == "columns":
        result.columns = index
    else:
        result.index = index
    return result


def _value_mask(index: pd.Index, value: Any) -> np.ndarray:
    """Match one level value, including missing values, without sorting it."""

    try:
        return np.asarray(index.isin([value]), dtype=bool)
    except (TypeError, ValueError):
        # ``Index.isin`` covers normal scalar values.  The fallback is useful
        # for unusual extension dtypes with values that reject hashing.
        if pd.isna(value):
            return np.asarray(index.isna(), dtype=bool)
        return np.asarray(index == value, dtype=bool)


def _values_equal(left: Any, right: Any) -> bool:
    try:
        if pd.isna(left) and pd.isna(right):
            return True
    except (TypeError, ValueError):
        pass
    try:
        result = left == right
        return bool(result) if np.ndim(result) == 0 else bool(np.all(result))
    except (TypeError, ValueError):
        return False


class MultiIndexSelector:
    """Select aligned pandas objects along a named MultiIndex axis.

    Parameters
    ----------
    dataframes:
        A non-empty mapping of labels to aligned DataFrames.  For backwards
        compatibility, a DataFrame with MultiIndex columns can be supplied
        directly; by default its outer column level is interpreted as the
        variable label and split into a mapping.
    axis:
        Axis containing the selector levels.  Both ``"columns"`` (the
        historical default) and ``"index"`` are supported.
    active_levels:
        Independently selectable levels.  The default is every level not part
        of ``combined_levels``.
    combined_levels:
        Groups whose *observed* tuples should be selected together.  This is
        useful for correlated coordinates such as magnitude and angle.
    drop_selected_levels:
        Drop selected levels from result axes.  Hidden singleton levels are
        dropped as well when ``hide_empty_levels`` is true.
    hide_empty_levels:
        Hide levels or combined groups with only one observed option.
    split_outer_level:
        Control splitting for a directly supplied DataFrame.  ``None`` keeps
        the historical behaviour for MultiIndex columns and otherwise stores
        the DataFrame under the key ``"data"``.

    Notes
    -----
    Input DataFrames are never modified.  Internally, shallow DataFrame copies
    are used so that unnamed or duplicate level names can be normalised safely.
    """

    def __init__(
        self,
        dataframes: FrameInput,
        *,
        axis: str | int = "columns",
        active_levels: Sequence[str] | None = None,
        combined_levels: Sequence[Sequence[str]] | None = None,
        drop_selected_levels: bool = False,
        hide_empty_levels: bool = True,
        split_outer_level: bool | None = None,
    ) -> None:
        self.axis = _axis_name(axis)
        self.drop_selected_levels = bool(drop_selected_levels)
        self.hide_empty_levels = bool(hide_empty_levels)

        raw_frames = self._coerce_frames(dataframes, split_outer_level)
        self.dataframes = self._normalise_frames(raw_frames)
        first_frame = next(iter(self.dataframes.values()))
        self.selector_index = getattr(first_frame, self.axis)
        if len(self.selector_index) == 0:
            raise ValueError(f"The selector {self.axis} must not be empty")
        self.level_names = list(self.selector_index.names)

        self._validate_alignment()
        self._values_by_level = {
            level: tuple(
                self.selector_index.get_level_values(level).unique().tolist()
            )
            for level in self.level_names
        }

        requested_combinations = [] if combined_levels is None else combined_levels
        self.combined_levels = self._validate_combined_levels(
            requested_combinations
        )

        if active_levels is None:
            active = list(self.level_names)
        else:
            active = self._validate_levels(active_levels, "active_levels")

        combined_members = {
            level for combination in self.combined_levels for level in combination
        }
        # Historical behaviour silently removed combined levels from the
        # independent controls.  Keep that useful behaviour for migration.
        active = [level for level in active if level not in combined_members]

        if self.hide_empty_levels:
            active = [
                level for level in active if len(self._values_by_level[level]) > 1
            ]
        self.active_levels = active

        all_combined_options = {
            combination: self._observed_combinations(combination)
            for combination in self.combined_levels
        }
        self.filtered_combined_levels = [
            combination
            for combination in self.combined_levels
            if not self.hide_empty_levels
            or len(all_combined_options[combination]) > 1
        ]
        self.observed_combined_options = {
            combination: all_combined_options[combination]
            for combination in self.filtered_combined_levels
        }

    @property
    def options(self) -> dict[str, tuple[Any, ...]]:
        """Observed options for the independently selectable levels."""

        return {
            level: self._values_by_level[level] for level in self.active_levels
        }

    def values_for(self, level: str) -> tuple[Any, ...]:
        """Return observed values for one selector level in source order."""

        self._validate_levels([level], "level")
        return self._values_by_level[level]

    @staticmethod
    def combined_key(levels: Sequence[str]) -> str:
        """Return the legacy keyword used for a combined widget control."""

        return str(tuple(levels))

    def default_selection(self) -> dict[str, Any]:
        """Return the first observed value for every visible control."""

        defaults = {
            level: self._values_by_level[level][0] for level in self.active_levels
        }
        defaults.update(
            {
                self.combined_key(combination): options[0]
                for combination, options in self.observed_combined_options.items()
            }
        )
        return defaults

    def select(
        self,
        selections: Mapping[str | tuple[str, ...], Any] | None = None,
        /,
        **level_values: Any,
    ) -> dict[FrameKey, pd.DataFrame]:
        """Return data matching level selections without invoking a plotter.

        Individual levels may be passed as keyword arguments.  A mapping also
        accepts a combined-level tuple as key, or the legacy ``str(tuple)`` key
        used by :class:`InteractiveMultiIndexPlotter` widgets.
        """

        raw_selections: dict[str | tuple[str, ...], Any] = {}
        if selections is not None:
            if not isinstance(selections, Mapping):
                raise TypeError("selections must be a mapping")
            raw_selections.update(selections)
        for key, value in level_values.items():
            if key in raw_selections and not _values_equal(raw_selections[key], value):
                raise ValueError(f"Conflicting selections for {key!r}")
            raw_selections[key] = value

        expanded = self._expand_selections(raw_selections)
        mask = np.ones(len(self.selector_index), dtype=bool)
        for level, value in expanded.items():
            if isinstance(value, slice) and value == slice(None):
                continue
            level_index = self.selector_index.get_level_values(level)
            mask &= _value_mask(level_index, value)

        if not bool(mask.any()):
            rendered = ", ".join(
                f"{name}={value!r}" for name, value in expanded.items()
            )
            raise KeyError(f"No data match the requested selection ({rendered})")

        result: dict[FrameKey, pd.DataFrame] = {}
        for label, frame in self.dataframes.items():
            if self.axis == "columns":
                subset = frame.loc[:, mask].copy(deep=False)
            else:
                subset = frame.loc[mask, :].copy(deep=False)
            result[label] = self._drop_levels(subset, label, expanded)
        return result

    def _coerce_frames(
        self, dataframes: FrameInput, split_outer_level: bool | None
    ) -> dict[FrameKey, pd.DataFrame]:
        if isinstance(dataframes, pd.DataFrame):
            selector_axis = getattr(dataframes, self.axis)
            should_split = split_outer_level
            if should_split is None:
                should_split = (
                    self.axis == "columns"
                    and isinstance(selector_axis, pd.MultiIndex)
                    and selector_axis.nlevels > 1
                )
            if not should_split:
                return {"data": dataframes}
            if not isinstance(selector_axis, pd.MultiIndex):
                raise ValueError(
                    "split_outer_level requires a MultiIndex on the selector axis"
                )
            if selector_axis.nlevels < 2:
                raise ValueError(
                    "split_outer_level requires at least two MultiIndex levels"
                )

            outer_values = selector_axis.get_level_values(0).unique()
            result: dict[FrameKey, pd.DataFrame] = {}
            for outer_value in outer_values:
                mask = _value_mask(selector_axis.get_level_values(0), outer_value)
                if self.axis == "columns":
                    frame = dataframes.loc[:, mask].copy(deep=False)
                    frame.columns = frame.columns.droplevel(0)
                else:
                    frame = dataframes.loc[mask, :].copy(deep=False)
                    frame.index = frame.index.droplevel(0)
                result[outer_value] = frame
            return result

        if not isinstance(dataframes, Mapping):
            raise TypeError("dataframes must be a DataFrame or a mapping of DataFrames")
        if not dataframes:
            raise ValueError("dataframes must not be empty")
        result = {}
        for label, frame in dataframes.items():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"dataframes[{label!r}] must be a pandas DataFrame")
            result[label] = frame
        return result

    def _normalise_frames(
        self, frames: Mapping[FrameKey, pd.DataFrame]
    ) -> dict[FrameKey, pd.DataFrame]:
        result = {}
        for label, frame in frames.items():
            selector_axis = getattr(frame, self.axis)
            result[label] = _replace_axis(
                frame, self.axis, _as_named_multiindex(selector_axis)
            )
        return result

    def _validate_alignment(self) -> None:
        for label, frame in self.dataframes.items():
            candidate = getattr(frame, self.axis)
            if not candidate.equals(self.selector_index):
                raise ValueError(
                    f"Selector-axis mismatch for {label!r}: all DataFrames "
                    f"must have identical {self.axis}"
                )

    def _validate_levels(
        self, levels: Sequence[str], argument_name: str
    ) -> list[str]:
        values = list(levels)
        if len(values) != len(set(values)):
            raise ValueError(f"{argument_name} must not contain duplicate levels")
        unknown = [level for level in values if level not in self.level_names]
        if unknown:
            raise ValueError(
                f"Unknown level(s) in {argument_name}: {unknown!r}; "
                f"available levels are {self.level_names!r}"
            )
        return values

    def _validate_combined_levels(
        self, combinations: Sequence[Sequence[str]]
    ) -> list[tuple[str, ...]]:
        result: list[tuple[str, ...]] = []
        occupied: set[str] = set()
        for raw_combination in combinations:
            combination = tuple(raw_combination)
            if not combination:
                raise ValueError("combined_levels must not contain an empty group")
            self._validate_levels(combination, "combined_levels")
            overlap = occupied.intersection(combination)
            if overlap:
                raise ValueError(
                    "A level may occur in only one combined group; duplicate "
                    f"level(s): {sorted(overlap)!r}"
                )
            occupied.update(combination)
            result.append(combination)
        return result

    def _observed_combinations(
        self, combination: tuple[str, ...]
    ) -> tuple[tuple[Any, ...], ...]:
        positions = [self.level_names.index(level) for level in combination]
        frame = self.selector_index.to_frame(index=False).iloc[:, positions]
        unique = frame.drop_duplicates()
        return tuple(unique.itertuples(index=False, name=None))

    def _expand_selections(
        self, selections: Mapping[str | tuple[str, ...], Any]
    ) -> dict[str, Any]:
        expanded: dict[str, Any] = {}
        combined_by_key = {
            self.combined_key(combination): combination
            for combination in self.combined_levels
        }
        combined_by_label = {
            " / ".join(combination): combination
            for combination in self.combined_levels
        }

        for key, value in selections.items():
            if key in self.level_names:
                pairs = [(str(key), value)]
            else:
                combination: tuple[str, ...] | None
                if isinstance(key, tuple):
                    combination = key if key in self.combined_levels else None
                else:
                    combination = combined_by_key.get(str(key))
                    if combination is None:
                        combination = combined_by_label.get(str(key))
                if combination is None:
                    raise KeyError(
                        f"Unknown selector {key!r}; available levels are "
                        f"{self.level_names!r}"
                    )
                try:
                    values = tuple(value)
                except TypeError as exc:
                    raise ValueError(
                        f"Selection for combined levels {combination!r} must "
                        "be a tuple-like value"
                    ) from exc
                if len(values) != len(combination):
                    raise ValueError(
                        f"Selection for {combination!r} needs "
                        f"{len(combination)} values, got {len(values)}"
                    )
                pairs = list(zip(combination, values))

            for level, selected_value in pairs:
                if level in expanded and not _values_equal(
                    expanded[level], selected_value
                ):
                    raise ValueError(f"Conflicting selections for level {level!r}")
                expanded[level] = selected_value
        return expanded

    def _drop_levels(
        self,
        frame: pd.DataFrame,
        label: FrameKey,
        selections: Mapping[str, Any],
    ) -> pd.DataFrame:
        if not self.drop_selected_levels:
            return frame

        levels_to_drop = set(selections)
        if self.hide_empty_levels:
            levels_to_drop.update(
                level
                for level, values in self._values_by_level.items()
                if len(values) == 1
            )
        positions = sorted(
            self.level_names.index(level)
            for level in levels_to_drop
            if level in self.level_names
        )
        if not positions:
            return frame

        selected_axis = getattr(frame, self.axis)
        if selected_axis.nlevels > len(positions):
            new_axis: pd.Index = selected_axis.droplevel(positions)
        elif len(selected_axis) == 1 and self.axis == "columns":
            # Preserve the useful historical label for a fully selected
            # variable while avoiding the old one-column assignment bug.
            new_axis = pd.Index([label])
        else:
            new_axis = pd.RangeIndex(len(selected_axis))
        return _replace_axis(frame, self.axis, new_axis)


class InteractiveMultiIndexPlotter(MultiIndexSelector):
    """Interactive adapter around :class:`MultiIndexSelector`.

    The constructor and commonly inspected attributes are compatible with the
    former MultiADSweep implementation.  Unlike ``ipywidgets.interact``,
    :meth:`widget` merely returns a widget tree; it does not display it.
    Use :meth:`show` for an explicit display side effect.
    """

    def __init__(
        self,
        dataframes: FrameInput,
        plot_func: Callable[[Any], Any] | None = None,
        active_levels: Sequence[str] | None = None,
        combined_levels: Sequence[Sequence[str]] | None = None,
        drop_selected_levels: bool = False,
        hide_empty_levels: bool = True,
        use_df: bool = False,
        *,
        axis: str | int = "columns",
        split_outer_level: bool | None = None,
    ) -> None:
        super().__init__(
            dataframes,
            axis=axis,
            active_levels=active_levels,
            combined_levels=combined_levels,
            drop_selected_levels=drop_selected_levels,
            hide_empty_levels=hide_empty_levels,
            split_outer_level=split_outer_level,
        )
        self.use_df = bool(use_df)
        self.plot_func = plot_func if plot_func is not None else self.default_plot_func
        self.last_data: dict[FrameKey, pd.DataFrame] | None = None
        self.last_plot: Any = None
        self._widget: Any = None

        # Compatibility attributes used by existing notebooks.
        if self.axis == "columns":
            self.columns = self.selector_index
        else:
            self.index = self.selector_index
        self.combined_options = {
            combination: [
                (str(position), option)
                for position, option in enumerate(options)
            ]
            for combination, options in self.observed_combined_options.items()
        }

    def plot(self, **selections: Any) -> Any:
        """Select data, invoke the callback, and return its result."""

        return self._interactive_plot(**selections)

    def _interactive_plot(self, **kwargs: Any) -> Any:
        """Backward-compatible callback used by existing notebooks."""

        self.last_data = self.select(kwargs)
        callback_data: Any = self.last_data_df if self.use_df else self.last_data
        self.last_plot = self.plot_func(callback_data)
        return self.last_plot

    @property
    def last_data_df(self) -> pd.DataFrame:
        """Concatenate the most recently selected variables by columns."""

        if self.last_data is None:
            raise RuntimeError("No selection has been plotted yet")
        result = pd.concat(self.last_data, axis=1)
        if isinstance(result.columns, pd.MultiIndex):
            duplicate_levels: list[int] = []
            for left in range(result.columns.nlevels):
                for right in range(left + 1, result.columns.nlevels):
                    if result.columns.get_level_values(left).equals(
                        result.columns.get_level_values(right)
                    ):
                        duplicate_levels.append(right)
            if duplicate_levels and result.columns.nlevels > len(
                set(duplicate_levels)
            ):
                result.columns = result.columns.droplevel(
                    sorted(set(duplicate_levels))
                )
        return result

    def widget(self) -> Any:
        """Create and return controls without displaying them.

        Raises
        ------
        ImportError
            If ipywidgets is not installed.
        """

        try:
            import ipywidgets as widgets
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Interactive plotting requires ipywidgets. "
                + _VISUALIZATION_INSTALL_HINT
            ) from exc

        controls: dict[str, Any] = {}
        for level in self.active_levels:
            controls[level] = widgets.SelectionSlider(
                options=[
                    (self.format_slider_label(value), value)
                    for value in self.values_for(level)
                ],
                description=level,
                continuous_update=False,
                layout=widgets.Layout(width="500px"),
            )

        for combination, options in self.observed_combined_options.items():
            key = self.combined_key(combination)
            controls[key] = widgets.SelectionSlider(
                options=[
                    (
                        " / ".join(
                            f"{level}={self.format_slider_label(value)}"
                            for level, value in zip(combination, option)
                        ),
                        option,
                    )
                    for option in options
                ],
                description=" / ".join(combination),
                continuous_update=False,
                layout=widgets.Layout(width="500px"),
            )

        output = widgets.interactive_output(self._interactive_plot, controls)
        self._widget = widgets.VBox([*controls.values(), output])
        # Useful for programmatic notebook/tests access, without depending on
        # non-public ipywidgets internals.
        self._widget.pynst_controls = controls
        self._widget.pynst_output = output
        return self._widget

    def create_interactive_widget(self) -> Any:
        """Backward-compatible alias for :meth:`widget`."""

        return self.widget()

    def show(self) -> Any:
        """Explicitly display and return this plotter's widget tree."""

        widget = self._widget if self._widget is not None else self.widget()
        try:
            from IPython.display import display
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Displaying widgets requires IPython. "
                + _VISUALIZATION_INSTALL_HINT
            ) from exc
        display(widget)
        return widget

    @staticmethod
    def format_slider_label(value: Any) -> str:
        """Format numeric slider labels with four significant digits."""

        return _format_value(value)

    @staticmethod
    def default_plot_func(selected_data: FrameInput) -> Any:
        """Draw every selected column on one ordinary Matplotlib Axes."""

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "The default plot callback requires Matplotlib. "
                + _VISUALIZATION_INSTALL_HINT
            ) from exc

        frames: FrameMapping
        if isinstance(selected_data, pd.DataFrame):
            frames = {"data": selected_data}
        else:
            frames = selected_data

        figure, axes = plt.subplots(figsize=(6, 4))
        for label, frame in frames.items():
            for column in frame.columns:
                axes.plot(
                    frame.index,
                    frame[column],
                    label=f"{label} | {column}",
                )
        axes.set_xlabel("Index")
        axes.set_ylabel("Value")
        axes.legend()
        axes.grid(True)
        figure.tight_layout()
        plt.show()
        return figure, axes


__all__ = ["InteractiveMultiIndexPlotter", "MultiIndexSelector"]
