from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pynst.visualization import (
    InteractiveMultiIndexPlotter,
    MultiIndexSelector,
    plot_mi,
)


def _column_frames() -> dict[str, pd.DataFrame]:
    columns = pd.MultiIndex.from_tuples(
        [
            (1.0, 0.25, 0),
            (2.0, 0.50, 90),
        ],
        names=["frequency", "magnitude", "angle"],
    )
    return {
        "gain": pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=columns),
        "phase": pd.DataFrame([[5.0, 6.0], [7.0, 8.0]], columns=columns),
    }


def test_selector_uses_observed_values_and_combined_tuples() -> None:
    columns = pd.MultiIndex(
        levels=[[1.0, 2.0, 3.0], [0.25, 0.50], [0, 90, 180]],
        codes=[[0, 1], [0, 1], [0, 2]],
        names=["frequency", "magnitude", "angle"],
    )
    frame = pd.DataFrame([[10.0, 20.0]], columns=columns)

    selector = MultiIndexSelector(
        {"value": frame},
        combined_levels=[("magnitude", "angle")],
    )

    assert selector.active_levels == ["frequency"]
    assert selector.values_for("frequency") == (1.0, 2.0)
    assert selector.observed_combined_options[("magnitude", "angle")] == (
        (0.25, 0),
        (0.50, 180),
    )

    selected = selector.select(
        {("magnitude", "angle"): (0.50, 180)}, frequency=2.0
    )
    assert selected["value"].iloc[0, 0] == 20.0
    assert selected["value"].columns.tolist() == [(2.0, 0.50, 180)]


def test_selector_rejects_an_unobserved_cross_combination() -> None:
    selector = MultiIndexSelector(
        _column_frames(), combined_levels=[("magnitude", "angle")]
    )

    with pytest.raises(KeyError, match="No data match"):
        selector.select(
            {selector.combined_key(("magnitude", "angle")): (0.50, 90)},
            frequency=1.0,
        )


def test_hide_empty_levels_uses_observed_not_declared_values() -> None:
    columns = pd.MultiIndex(
        levels=[[1.0, 2.0]],
        codes=[[0, 0]],
        names=["bias"],
    )
    frame = pd.DataFrame([[1.0, 2.0]], columns=columns)

    selector = MultiIndexSelector({"value": frame})

    assert selector.values_for("bias") == (1.0,)
    assert selector.active_levels == []


def test_selector_does_not_mutate_plain_or_multiindex_inputs() -> None:
    simple = pd.DataFrame([[1.0, 2.0]], columns=pd.Index([10, 20]))
    original_simple_columns = simple.columns.copy()
    multi = _column_frames()["gain"]
    original_multi_columns = multi.columns.copy()

    simple_selector = MultiIndexSelector(
        {"simple": simple}, hide_empty_levels=False
    )
    MultiIndexSelector({"multi": multi})

    assert isinstance(simple.columns, pd.Index)
    assert not isinstance(simple.columns, pd.MultiIndex)
    assert simple.columns.equals(original_simple_columns)
    assert multi.columns.equals(original_multi_columns)
    assert isinstance(simple_selector.dataframes["simple"].columns, pd.MultiIndex)


def test_selector_supports_index_axis_and_drops_only_result_levels() -> None:
    index = pd.MultiIndex.from_product(
        [[1.0, 2.0], [0, 1]], names=["frequency", "point"]
    )
    frame = pd.DataFrame({"value": [10.0, 11.0, 20.0, 21.0]}, index=index)
    original_index = frame.index.copy()
    selector = MultiIndexSelector(
        {"data": frame},
        axis="index",
        active_levels=["frequency"],
        drop_selected_levels=True,
    )

    selected = selector.select(frequency=2.0)["data"]

    assert selected.index.name == "point"
    assert selected.index.tolist() == [0, 1]
    assert selected["value"].tolist() == [20.0, 21.0]
    assert frame.index.equals(original_index)


def test_selector_validates_mapping_alignment() -> None:
    frames = _column_frames()
    frames["phase"] = frames["phase"].iloc[:, ::-1]

    with pytest.raises(ValueError, match="Selector-axis mismatch"):
        MultiIndexSelector(frames)


def test_direct_dataframe_keeps_legacy_outer_variable_split() -> None:
    columns = pd.MultiIndex.from_product(
        [["gain", "phase"], [1.0, 2.0]], names=["variable", "frequency"]
    )
    frame = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], columns=columns)

    selector = MultiIndexSelector(frame)

    assert list(selector.dataframes) == ["gain", "phase"]
    assert selector.level_names == ["frequency"]
    assert selector.dataframes["gain"].iloc[0].tolist() == [1.0, 2.0]


def test_plotter_preserves_callback_api_and_exposes_public_plot() -> None:
    calls: list[pd.DataFrame] = []
    plotter = InteractiveMultiIndexPlotter(
        _column_frames(),
        plot_func=calls.append,
        active_levels=["frequency"],
        drop_selected_levels=True,
        use_df=True,
    )

    result = plotter.plot(frequency=2.0)

    assert result is None
    assert len(calls) == 1
    assert isinstance(calls[0], pd.DataFrame)
    assert plotter.last_data is not None
    assert list(plotter.last_data) == ["gain", "phase"]
    assert plotter.last_data["gain"].iloc[:, 0].tolist() == [2.0, 4.0]


def test_widget_creation_does_not_display_it(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("ipywidgets")
    display = pytest.importorskip("IPython.display")
    displayed: list[object] = []
    monkeypatch.setattr(display, "display", displayed.append)
    plotter = InteractiveMultiIndexPlotter(
        _column_frames(),
        plot_func=lambda selected: selected,
        active_levels=["frequency"],
    )

    widget = plotter.widget()

    assert widget is not None
    assert displayed == []
    assert list(widget.pynst_controls) == ["frequency"]


@pytest.fixture
def pyplot():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def test_plot_mi_labels_multiindex_columns_on_normal_axes(pyplot) -> None:
    columns = pd.MultiIndex.from_tuples(
        [(1.0, "cold"), (2.0, "hot")], names=["frequency", "state"]
    )
    frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=columns)
    _, axes = pyplot.subplots()

    lines = plot_mi(frame, ax=axes)

    assert len(lines) == 2
    assert lines[0].axes is axes
    assert [line.get_label() for line in lines] == [
        "frequency = 1; state = cold",
        "frequency = 2; state = hot",
    ]


def test_plot_mi_supports_shared_x_and_selected_label_levels(pyplot) -> None:
    index = pd.Index([0, 1, 2], name="sample")
    x = pd.DataFrame({"x": [10.0, 20.0, 30.0]}, index=index)
    columns = pd.MultiIndex.from_tuples(
        [(1.0, "cold"), (2.0, "hot")], names=["frequency", "state"]
    )
    y = pd.DataFrame(np.arange(6).reshape(3, 2), index=index, columns=columns)
    _, axes = pyplot.subplots()

    lines = plot_mi(
        x,
        y,
        ax=axes,
        label_levels=["frequency"],
        label="measurement",
    )

    assert len(lines) == 2
    assert lines[0].get_xdata().tolist() == [10.0, 20.0, 30.0]
    assert [line.get_label() for line in lines] == [
        "measurement | frequency = 1",
        "measurement | frequency = 2",
    ]


def test_plot_mi_accepts_differently_named_single_column_frames(pyplot) -> None:
    index = pd.Index([0, 1])
    x = pd.DataFrame({"output_power": [1.0, 2.0]}, index=index)
    y = pd.DataFrame({"efficiency": [0.1, 0.2]}, index=index)
    _, axes = pyplot.subplots()

    lines = plot_mi(x, y, ax=axes)

    assert len(lines) == 1
    assert lines[0].get_xdata().tolist() == [1.0, 2.0]
    assert lines[0].get_ydata().tolist() == [0.1, 0.2]
    assert lines[0].get_label() == "efficiency"


def test_plot_mi_rejects_misaligned_pandas_inputs(pyplot) -> None:
    x = pd.Series([1.0, 2.0], index=[0, 1])
    y = pd.Series([3.0, 4.0], index=[1, 2])
    _, axes = pyplot.subplots()

    with pytest.raises(ValueError, match="share the same index"):
        plot_mi(x, y, ax=axes)
