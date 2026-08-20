from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pynst import CriticalMeasurementError, GenericSweepDataset, SweepManager


def _numeric_grid() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([(1,)], names=["point"])


def _string_frame(
    location: str,
    *,
    include_missing: bool = True,
) -> pd.DataFrame:
    if location == "local_ivar":
        return pd.DataFrame(
            {"value": [1.0, 2.0]},
            index=pd.Index(
                pd.array(["µ", "tab\tvalue"], dtype="string"),
                name="sample",
            ),
        )
    return pd.DataFrame(
        {
            "text": pd.Series(
                pd.array(
                    ["µ", pd.NA if include_missing else "plain"],
                    dtype="string",
                ),
                index=[0, 1],
            )
        },
        index=pd.Index([0, 1], name="sample"),
    )


@pytest.mark.parametrize("strategy", ["fixed", "streaming"])
def test_string_extension_sweep_grid_is_canonical_and_resumable(
    tmp_path: Path,
    strategy: str,
) -> None:
    source = pd.MultiIndex(
        levels=[
            pd.Index(
                pd.array(["plain", "retry", "unused"], dtype="string")
            )
        ],
        codes=[[0, 1]],
        names=["point"],
    )
    assert isinstance(source.levels[0].dtype, pd.StringDtype)

    measurement_name = f"string_extension_grid_{strategy}"
    target = tmp_path / f"{measurement_name}.h5"
    with SweepManager(
        measurement_func=lambda _params: {
            "measurement": pd.DataFrame(
                {"value": [1.0]},
                index=pd.Index([0], name="sample"),
            )
        },
        ivars=source,
        meas_name=measurement_name,
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    ) as manager:
        assert source.levels[0].dtype != np.dtype("object")
        assert manager.multi_index.levels[0].dtype == np.dtype("object")
        assert manager.multi_index.levels[0].tolist() == [
            "plain",
            "retry",
            "unused",
        ]
        assert manager.run() is True
        contract = json.loads(
            manager.contract_file.read_text(encoding="utf-8")
        )
        manager.merge(target, strategy=strategy)  # type: ignore[arg-type]

    assert contract["grid"]["level_dtypes"] == ["object"]
    assert contract["grid"]["level_dtype_specs"] == [{"dtype": "object"}]

    resumed_calls: list[dict[str, Any]] = []
    with SweepManager(
        measurement_func=lambda params: resumed_calls.append(dict(params))
        or {},
        ivars=source,
        meas_name=measurement_name,
        output_root=tmp_path,
        chunk_size=1,
        resume=True,
    ) as resumed:
        assert resumed.run() is True

    assert resumed_calls == []
    assert isinstance(source.levels[0].dtype, pd.StringDtype)

    dataset = GenericSweepDataset(target)
    stored = dataset.get_block("measurement")
    assert stored.index.get_level_values("point").dtype == np.dtype("object")
    dataset.validate_storage(deep=True)
    merged_log = SweepManager.read_merged_log(target)
    assert all(
        not isinstance(dtype, pd.StringDtype)
        for dtype in merged_log.dtypes
    )
    with pd.HDFStore(target, mode="r") as store:
        assert bool(store.get_storer("measurement").is_table) is (
            strategy == "streaming"
        )


@pytest.mark.parametrize("location", ["dvar", "local_ivar"])
@pytest.mark.parametrize("strategy", ["fixed", "streaming"])
@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
def test_string_extension_measurement_data_round_trips_as_object(
    tmp_path: Path,
    location: str,
    strategy: str,
) -> None:
    callback_frames: list[pd.DataFrame] = []

    def measurement(_params: dict[str, Any]) -> dict[str, pd.DataFrame]:
        frame = _string_frame(
            location,
            include_missing=True,
        )
        callback_frames.append(frame)
        return {"measurement": frame}

    name = f"string_extension_{location}_{strategy}"
    target = tmp_path / f"{name}.h5"
    with SweepManager(
        measurement_func=measurement,
        ivars=_numeric_grid(),
        meas_name=name,
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
    ) as manager:
        assert manager.run() is True
        result = manager.get_results()
        assert isinstance(result, dict)
        stored = result["measurement"]
        contract = json.loads(
            manager.contract_file.read_text(encoding="utf-8")
        )
        manager.merge(target, strategy=strategy)  # type: ignore[arg-type]

    callback_frame = callback_frames[0]
    if location == "dvar":
        assert isinstance(callback_frame["text"].dtype, pd.StringDtype)
        assert stored["text"].dtype == np.dtype("object")
        assert stored["text"].iloc[0] == "µ"
        assert pd.isna(stored["text"].iloc[1])
        assert contract["result_schema"]["measurement"]["dtypes"] == [
            "object"
        ]
    else:
        assert isinstance(callback_frame.index.dtype, pd.StringDtype)
        sample = stored.index.get_level_values("sample")
        assert sample.dtype == np.dtype("object")
        assert sample.tolist() == ["µ", "tab\tvalue"]
        assert contract["result_schema"]["measurement"][
            "index_dtypes"
        ] == ["object"]

    resumed_calls: list[dict[str, Any]] = []
    with SweepManager(
        measurement_func=lambda params: resumed_calls.append(dict(params))
        or {"measurement": _string_frame(location)},
        ivars=_numeric_grid(),
        meas_name=name,
        output_root=tmp_path,
        chunk_size=1,
        resume=True,
    ) as resumed:
        assert resumed.run() is True
    assert resumed_calls == []

    dataset = GenericSweepDataset(target)
    dataset.validate_storage(deep=True)
    merged = dataset.get_block("measurement")
    if location == "dvar":
        assert merged["text"].dtype == np.dtype("object")
        assert merged["text"].iloc[0] == "µ"
        assert pd.isna(merged["text"].iloc[1])
    else:
        merged_sample = merged.index.get_level_values("sample")
        assert merged_sample.dtype == np.dtype("object")
        assert merged_sample.tolist() == ["µ", "tab\tvalue"]
    with pd.HDFStore(target, mode="r") as store:
        assert bool(store.get_storer("measurement").is_table) is (
            strategy == "streaming"
        )


@pytest.mark.parametrize(
    "declared_dtype",
    ["str", "string", "string[python]", "string[pyarrow]"],
)
def test_declared_string_schema_and_metadata_use_storage_dtype(
    tmp_path: Path,
    declared_dtype: str,
) -> None:
    expected_schema = {
        "measurement": {
            "index_names": ["sample"],
            "index_dtypes": ["int64"],
            "index_dtype_specs": [{"dtype": "int64"}],
            "columns": ["text"],
            "dtypes": [declared_dtype],
            "dtype_specs": [{"dtype": declared_dtype}],
        }
    }
    metadata = {
        "blocks": {
            "measurement": {
                "variables": {"text": {"dtype": declared_dtype}},
            }
        }
    }

    safe_dtype = declared_dtype.replace("[", "_").replace("]", "")
    measurement_name = f"declared_string_schema_{safe_dtype}"
    with SweepManager(
        measurement_func=lambda _params: {
            "measurement": pd.DataFrame(
                {
                    "text": pd.Series(
                        pd.array(["declared"], dtype="string")
                    )
                },
                index=pd.Index([0], name="sample"),
            )
        },
        ivars=_numeric_grid(),
        meas_name=measurement_name,
        output_root=tmp_path,
        resume=False,
        expected_result_schema=expected_schema,
        metadata=metadata,
    ) as manager:
        assert manager.run() is True
        contract = json.loads(
            manager.contract_file.read_text(encoding="utf-8")
        )
        observed = manager.block_metadata["measurement"].variables["text"]
        original_contract = manager.contract_file.read_bytes()

    result_schema = contract["result_schema"]["measurement"]
    assert result_schema["dtypes"] == ["object"]
    assert result_schema["dtype_specs"] == [{"dtype": "object"}]
    assert observed.dtype == "object"
    assert expected_schema["measurement"]["dtypes"] == [declared_dtype]
    assert metadata["blocks"]["measurement"]["variables"]["text"][
        "dtype"
    ] == declared_dtype

    resumed_calls: list[dict[str, Any]] = []
    with SweepManager(
        measurement_func=lambda params: resumed_calls.append(dict(params))
        or {},
        ivars=_numeric_grid(),
        meas_name=measurement_name,
        output_root=tmp_path,
        resume=True,
        expected_result_schema=expected_schema,
        metadata=metadata,
    ) as resumed:
        assert resumed.run() is True
        assert resumed.contract_file.read_bytes() == original_contract
    assert resumed_calls == []


def test_persisted_string_result_is_object_when_passed_to_resumed_callback(
    tmp_path: Path,
) -> None:
    grid = pd.MultiIndex.from_tuples([(1,), (2,)], names=["point"])

    def interrupted_measurement(
        params: dict[str, Any],
        _previous_result: Any,
    ) -> dict[str, pd.DataFrame]:
        if params["point"] == 2:
            raise CriticalMeasurementError("resume from the first point")
        return {"measurement": _string_frame("dvar", include_missing=False)}

    first = SweepManager(
        measurement_func=interrupted_measurement,
        ivars=grid,
        meas_name="string_previous_result",
        output_root=tmp_path,
        chunk_size=1,
        resume=False,
        provide_previous_result=True,
    )
    assert first.run() is False

    previous_results: list[Any] = []

    def resumed_measurement(
        _params: dict[str, Any],
        previous_result: Any,
    ) -> dict[str, pd.DataFrame]:
        previous_results.append(previous_result)
        return {"measurement": _string_frame("dvar", include_missing=False)}

    resumed = SweepManager(
        measurement_func=resumed_measurement,
        ivars=grid,
        meas_name="string_previous_result",
        output_root=tmp_path,
        chunk_size=1,
        resume=True,
        provide_previous_result=True,
    )
    assert resumed.run() is True

    assert len(previous_results) == 1
    previous = previous_results[0]
    assert isinstance(previous, dict)
    assert previous["measurement"]["text"].dtype == np.dtype("object")
    assert previous["measurement"]["text"].tolist() == ["µ", "plain"]


def test_non_string_extension_sweep_grid_remains_rejected(
    tmp_path: Path,
) -> None:
    grid = pd.MultiIndex.from_arrays(
        [pd.array([1, 2], dtype="Int64")],
        names=["point"],
    )

    with pytest.raises(TypeError, match="extension"):
        SweepManager(
            measurement_func=lambda _params: {},
            ivars=grid,
            meas_name="nullable_integer_grid",
            output_root=tmp_path,
            resume=False,
        )
